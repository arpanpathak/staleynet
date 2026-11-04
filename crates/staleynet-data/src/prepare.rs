//! Catalogue → feature cache: decode, resample, level-normalise, GPU log-mel,
//! and training-split statistics.

use crate::catalog::{Split, Track};
use crate::features::GpuMelExtractor;
use crate::store::{BandStatistics, CacheLayout, DatasetManifest, PreparedTrack, read_f16, write_f16};
use anyhow::Result;
use rayon::prelude::*;
use staleynet_audio::resample::resample;
use staleynet_audio::{MelAnalyzer, MelConfig, decode_mono};
use staleynet_cuda::Gpu;

/// RMS level every recording is scaled to before analysis (−20 dBFS), so the
/// model sees consistent loudness across sources.
/// See <https://en.wikipedia.org/wiki/Audio_normalization>
const TARGET_RMS: f32 = 0.1;

/// Recordings quieter than this are left unscaled (digital silence).
const SILENCE_RMS: f32 = 1e-6;

/// Recordings decoded concurrently on the CPU while the GPU analyses the previous group.
const DECODE_GROUP_SIZE: usize = 8;

/// What happened to one catalogue entry.
pub enum PrepareOutcome {
    /// Features written.
    Prepared(PreparedTrack),

    /// Skipped, with the reason.
    Skipped {
        /// Catalogue id.
        id: String,

        /// Why it could not be used.
        reason: String,
    },
}

/// Settings for [`prepare`].
pub struct PrepareOptions {
    /// Analysis parameters.
    pub mel: MelConfig,

    /// Recordings are cut to this many seconds (None keeps everything).
    pub max_seconds: Option<f32>,
}

/// Scales `samples` to [`TARGET_RMS`] and clips to `[-1, 1]`.
pub fn normalize_level(samples: &mut [f32]) {
    let mean_square = samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len().max(1) as f32;
    let rms = mean_square.sqrt();
    if rms < SILENCE_RMS {
        return;
    }

    let gain = TARGET_RMS / rms;
    for sample in samples.iter_mut() {
        *sample = (*sample * gain).clamp(-1.0, 1.0);
    }
}

/// Decodes, resamples, trims and level-normalises one recording.
pub fn load_signal(track: &Track, options: &PrepareOptions) -> Result<Vec<f32>> {
    let audio = decode_mono(&track.audio_path)?;
    let mut signal = resample(&audio.samples, audio.sample_rate, options.mel.sample_rate);

    if let Some(seconds) = options.max_seconds {
        signal.truncate((seconds * options.mel.sample_rate as f32) as usize);
    }
    anyhow::ensure!(signal.len() >= options.mel.fft_size, "shorter than one analysis frame");

    normalize_level(&mut signal);
    Ok(signal)
}

/// Builds the feature cache for `tracks` and writes `manifest.json`.
/// `report` is called once per track, in order.
pub fn prepare(
    gpu: &Gpu,
    tracks: Vec<Track>,
    layout: &CacheLayout,
    options: &PrepareOptions,
    mut report: impl FnMut(&PrepareOutcome),
) -> Result<DatasetManifest> {
    layout.create()?;

    let analyzer = MelAnalyzer::new(options.mel);
    let extractor = GpuMelExtractor::new(gpu, &analyzer)?;
    let mut prepared = Vec::with_capacity(tracks.len());

    for group in tracks.chunks(DECODE_GROUP_SIZE) {
        let signals: Vec<Result<Vec<f32>>> = group.par_iter().map(|track| load_signal(track, options)).collect();

        for (track, signal) in group.iter().zip(signals) {
            let outcome = match signal.and_then(|signal| analyse(gpu, &extractor, &analyzer, track, &signal, layout)) {
                Ok(entry) => PrepareOutcome::Prepared(entry),
                Err(error) => PrepareOutcome::Skipped { id: track.id.clone(), reason: format!("{error:#}") },
            };

            report(&outcome);
            if let PrepareOutcome::Prepared(entry) = outcome {
                prepared.push(entry);
            }
        }
    }

    let mut statistics = BandStatistics::new(options.mel.mel_bands);
    for entry in prepared.iter().filter(|entry| entry.track.split == Split::Train) {
        statistics.add(&read_f16(&layout.mel(&entry.track.id))?);
    }

    let manifest = DatasetManifest { mel: options.mel.into(), normalization: statistics.finish(), tracks: prepared };
    manifest.save(layout)?;
    Ok(manifest)
}

/// GPU log-mel of one signal, written to the cache.
fn analyse(
    gpu: &Gpu,
    extractor: &GpuMelExtractor,
    analyzer: &MelAnalyzer,
    track: &Track,
    signal: &[f32],
    layout: &CacheLayout,
) -> Result<PreparedTrack> {
    let log_mel = extractor.log_mel(gpu, analyzer, signal)?;
    write_f16(&layout.mel(&track.id), &log_mel)?;

    Ok(PreparedTrack {
        track: track.clone(),
        seconds: signal.len() as f32 / analyzer.config.sample_rate as f32,
        frames: log_mel.len() / analyzer.config.mel_bands,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt()
    }

    #[test]
    fn digital_silence_is_left_alone() {
        let mut samples = vec![0.0f32; 16];

        normalize_level(&mut samples);

        assert!(samples.iter().all(|sample| *sample == 0.0));
    }

    #[test]
    fn constant_signal_reaches_the_target_rms() {
        let mut samples = vec![0.01f32; 100];

        normalize_level(&mut samples);

        assert!((rms(&samples) - TARGET_RMS).abs() < 1e-5);
    }

    #[test]
    fn peaks_are_clipped_to_full_scale() {
        let mut samples = vec![0.0f32; 201];
        samples[0] = 10.0;

        normalize_level(&mut samples);

        assert_eq!(samples[0], 1.0);
        assert!(samples.iter().all(|sample| (-1.0..=1.0).contains(sample)));
    }

    #[test]
    fn empty_input_is_a_noop() {
        let mut samples: Vec<f32> = Vec::new();

        normalize_level(&mut samples);

        assert!(samples.is_empty());
    }
}
