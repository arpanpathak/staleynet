//! What the vocoder learns from: pairs of a spectrogram decoded from tokens
//! and the stretch of recording it came from.
//!
//! The vocoder only ever hears spectrograms that the language model's tokens
//! decode to, so that is what it trains on. Each track's codes are decoded with
//! the tokenizer, a window of frames is cut out, and the matching samples are
//! read from the waveform cache. The vocoder thereby learns phase and also
//! part of the detail the tokens lost.
//!
//! Frame `t` of a log-mel spectrogram is centered on sample `t · hop`, so the
//! samples behind frames `t₀ … t₀ + T − 1`, with half a frame on each side,
//! start at sample `t₀ · hop − fft_size / 2` and number
//! `(T − 1) · hop + fft_size`: the padded signal the vocoder synthesizes.

use crate::config::VocoderConfig;
use anyhow::{Context, Result, ensure};
use half::f16;
use rayon::prelude::*;
use staleynet_audio::MelConfig;
use staleynet_data::catalog::Split;
use staleynet_data::prepare::{PrepareOptions, load_signal};
use staleynet_data::store::{CacheLayout, DatasetManifest, PreparedTrack, read_codes, write_f16};
use staleynet_data::tokenizer::AudioTokenizer;
use staleynet_model::random::Random;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, sync_channel};

/// Bytes of one stored half-precision sample.
const F16_BYTES: usize = 2;

/// Bytes of one stored code.
const CODE_BYTES: usize = 2;

/// Recordings decoded at once while building the waveform cache.
const DECODE_GROUP_SIZE: usize = 32;

/// What happened to one track while building the waveform cache.
pub enum WaveformOutcome {
    /// Samples written.
    Written {
        /// Catalog id.
        id: String,

        /// Seconds of audio.
        seconds: f32,
    },

    /// Already in the cache from an earlier run.
    Present(String),

    /// Could not be decoded.
    Failed {
        /// Catalog id.
        id: String,

        /// Why.
        reason: String,
    },
}

/// Writes the samples of every track in the cache's manifest, decoded,
/// resampled and level-normalized exactly as when its log-mel was computed.
/// Tracks already present are kept, so an interrupted run can resume.
pub fn prepare_waveforms(layout: &CacheLayout, mut report: impl FnMut(&WaveformOutcome)) -> Result<()> {
    let manifest = DatasetManifest::load(layout)?;
    std::fs::create_dir_all(layout.root.join("waveforms"))?;

    let options = PrepareOptions { mel: MelConfig::from(manifest.mel), max_seconds: None };
    let sample_rate = options.mel.sample_rate as f32;

    for group in manifest.tracks.chunks(DECODE_GROUP_SIZE) {
        let outcomes: Vec<WaveformOutcome> = group
            .par_iter()
            .map(|entry| {
                let id = entry.track.id.clone();
                let path = layout.waveform(&id);
                if path.is_file() {
                    return WaveformOutcome::Present(id);
                }

                match load_signal(&entry.track, &options).and_then(|signal| write_f16(&path, &signal).map(|()| signal.len())) {
                    Ok(samples) => WaveformOutcome::Written { id, seconds: samples as f32 / sample_rate },
                    Err(error) => WaveformOutcome::Failed { id, reason: format!("{error:#}") },
                }
            })
            .collect();

        outcomes.iter().for_each(&mut report);
    }

    Ok(())
}

/// A batch ready for [`crate::vocoder::TrainingPass::run`].
pub struct Batch {
    /// Standardized log-mel frames, `[sequences · frames, mel_bands]`.
    pub mel: Vec<f32>,

    /// Target samples, `[sequences, padded samples]`.
    pub target: Vec<f32>,
}

/// One track that is long enough to cut segments from.
struct Clip {
    /// Catalog id.
    id: String,

    /// Last frame a segment may start at; the first is [`SegmentSampler::first_start`].
    last_start: usize,
}

/// Cuts training segments out of a tokenized cache that also holds waveforms.
pub struct SegmentSampler {
    /// The cache.
    layout: CacheLayout,

    /// Decodes codes into log-mel frames, and standardizes them.
    tokenizer: AudioTokenizer,

    /// Residual levels used when decoding.
    levels: usize,

    /// Frames per segment.
    frames: usize,

    /// Samples between frames.
    hop_size: usize,

    /// Samples per frame.
    fft_size: usize,

    /// Training tracks.
    train: Vec<Clip>,

    /// Held-out tracks.
    validation: Vec<Clip>,
}

impl SegmentSampler {
    /// A sampler of `frames`-frame segments from `layout`, decoding with the
    /// first `levels` residual levels (all of them when `None`).
    pub fn new(layout: &CacheLayout, config: &VocoderConfig, frames: usize, levels: Option<usize>) -> Result<Self> {
        let manifest = DatasetManifest::load(layout)?;
        let tokenizer = AudioTokenizer::load(&layout.tokenizer())?;

        ensure!(
            manifest.mel.sample_rate == config.sample_rate,
            "cache sample rate {} ≠ vocoder {}",
            manifest.mel.sample_rate,
            config.sample_rate
        );
        ensure!(
            manifest.mel.hop_size == config.hop_size && manifest.mel.fft_size == config.fft_size,
            "cache framing differs from the vocoder's"
        );
        ensure!(
            manifest.mel.mel_bands == config.mel_bands,
            "cache has {} mel bands, the vocoder {}",
            manifest.mel.mel_bands,
            config.mel_bands
        );

        let mut sampler = Self {
            layout: layout.clone(),
            levels: levels.unwrap_or(tokenizer.config.levels()),
            tokenizer,
            frames,
            hop_size: config.hop_size,
            fft_size: config.fft_size,
            train: Vec::new(),
            validation: Vec::new(),
        };

        for entry in &manifest.tracks {
            let Some(clip) = sampler.clip(entry) else { continue };
            match entry.track.split {
                Split::Train => sampler.train.push(clip),
                Split::Validation => sampler.validation.push(clip),
            }
        }

        ensure!(!sampler.train.is_empty(), "no training track in {} has codes, a waveform and {frames} frames", layout.root.display());
        Ok(sampler)
    }

    /// Training and held-out tracks usable for segments.
    pub fn track_counts(&self) -> (usize, usize) {
        (self.train.len(), self.validation.len())
    }

    /// The tokenizer's standardization, which the vocoder must share.
    pub fn normalization(&self) -> (&[f32], &[f32]) {
        (&self.tokenizer.normalization.mean, &self.tokenizer.normalization.std)
    }

    /// First frame a segment may start at: its first sample, half a frame
    /// before the frame's center, must not fall before the recording.
    fn first_start(&self) -> usize {
        (self.fft_size / 2).div_ceil(self.hop_size)
    }

    /// Samples behind a segment, padding included.
    fn segment_samples(&self) -> usize {
        (self.frames - 1) * self.hop_size + self.fft_size
    }

    /// The track as a [`Clip`], when its codes and waveform exist and hold a whole segment.
    fn clip(&self, entry: &PreparedTrack) -> Option<Clip> {
        let id = &entry.track.id;
        let code_bytes = std::fs::metadata(self.layout.codes(id)).ok()?.len() as usize;
        let sample_bytes = std::fs::metadata(self.layout.waveform(id)).ok()?.len() as usize;

        let config = &self.tokenizer.config;
        let mel_frames = code_bytes / CODE_BYTES / config.codebooks * config.frames_per_step;
        let samples = sample_bytes / F16_BYTES;

        let half = self.fft_size / 2;
        let last_by_frames = mel_frames.checked_sub(self.frames)?;
        let last_by_samples = (samples + half).checked_sub(self.segment_samples())? / self.hop_size;
        let last_start = last_by_frames.min(last_by_samples);

        (last_start >= self.first_start()).then(|| Clip { id: id.clone(), last_start })
    }

    /// Decoded log-mel frames and target samples of the segment of `clip` starting at frame `start`.
    fn segment(&self, clip: &Clip, start: usize, mel: &mut [f32], target: &mut [f32]) -> Result<()> {
        let codes = read_codes(&self.layout.codes(&clip.id))?;
        let decoded = self.tokenizer.decode(&codes, self.levels);
        let bands = self.tokenizer.mel_bands;
        let frames = &decoded[start * bands..(start + self.frames) * bands];

        let normalization = &self.tokenizer.normalization;
        for ((value, frame_value), band) in mel.iter_mut().zip(frames).zip((0..bands).cycle()) {
            *value = (frame_value - normalization.mean[band]) / normalization.std[band];
        }

        let first_sample = start * self.hop_size - self.fft_size / 2;
        let path = self.layout.waveform(&clip.id);
        let mut file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let mut bytes = vec![0u8; target.len() * F16_BYTES];
        file.seek(SeekFrom::Start((first_sample * F16_BYTES) as u64))?;
        file.read_exact(&mut bytes).with_context(|| format!("read {}", path.display()))?;

        for (sample, pair) in target.iter_mut().zip(bytes.chunks_exact(F16_BYTES)) {
            *sample = f16::from_le_bytes([pair[0], pair[1]]).to_f32();
        }
        Ok(())
    }

    /// `sequences` segments from random places in `clips`.
    fn batch(&self, clips: &[Clip], sequences: usize, random: &mut Random) -> Result<Batch> {
        let bands = self.tokenizer.mel_bands;
        let length = self.segment_samples();
        let picks: Vec<(usize, usize)> = (0..sequences)
            .map(|_| {
                let clip = random.below(clips.len());
                let first = self.first_start();
                (clip, first + random.below(clips[clip].last_start - first + 1))
            })
            .collect();

        let mut batch = Batch { mel: vec![0.0; sequences * self.frames * bands], target: vec![0.0; sequences * length] };
        batch
            .mel
            .par_chunks_mut(self.frames * bands)
            .zip(batch.target.par_chunks_mut(length))
            .zip(&picks)
            .try_for_each(|((mel, target), &(clip, start))| self.segment(&clips[clip], start, mel, target))?;

        Ok(batch)
    }

    /// A batch of training segments, the same for the same `seed` and `step`.
    pub fn training_batch(&self, sequences: usize, seed: u64, step: usize) -> Result<Batch> {
        self.batch(&self.train, sequences, &mut Random::derived(seed, step as u64))
    }

    /// `count` fixed batches of held-out segments (training segments when
    /// the cache holds no held-out track).
    pub fn validation_batches(&self, sequences: usize, count: usize, seed: u64) -> Result<Vec<Batch>> {
        let clips = if self.validation.is_empty() { &self.train } else { &self.validation };
        let mut random = Random::new(seed);

        (0..count).map(|_| self.batch(clips, sequences, &mut random)).collect()
    }

    /// Ids of the first `count` held-out tracks, for listening tests.
    pub fn listening_tracks(&self, count: usize) -> Vec<String> {
        let clips = if self.validation.is_empty() { &self.train } else { &self.validation };

        clips.iter().take(count).map(|clip| clip.id.clone()).collect()
    }

    /// The whole log-mel spectrogram of track `id` decoded from its codes, and
    /// its recorded samples, each cut to at most `frames` frames.
    pub fn whole_track(&self, id: &str, frames: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let bands = self.tokenizer.mel_bands;
        let mut decoded = self.tokenizer.decode(&read_codes(&self.layout.codes(id))?, self.levels);
        decoded.truncate(frames.min(decoded.len() / bands) * bands);

        let mut samples = staleynet_data::store::read_f16(&self.layout.waveform(id))?;
        samples.truncate((decoded.len() / bands).saturating_sub(1) * self.hop_size);
        Ok((decoded, samples))
    }

    /// Training batches for steps `first_step, first_step + 1, …`, prepared on
    /// a background thread `depth` batches ahead of the GPU.
    pub fn prefetch(self: Arc<Self>, sequences: usize, seed: u64, first_step: usize, depth: usize) -> Receiver<Result<Batch>> {
        let (sender, receiver) = sync_channel(depth);

        std::thread::spawn(move || {
            for step in first_step.. {
                if sender.send(self.training_batch(sequences, seed, step)).is_err() {
                    return;
                }
            }
        });
        receiver
    }
}
