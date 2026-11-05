//! `staleynet reconstruct`: how much of a recording survives the audio
//! pipeline when no language model is involved.
//!
//! Generated audio passes through two lossy stages after the model: the
//! tokenizer's codebooks (codes → log-mel) and the Griffin-Lim phase
//! reconstruction (log-mel → waveform). Their losses bound the quality any
//! model trained on these tokens can reach. For excerpts of held-out vocal
//! tracks, this command renders
//!
//! 1. the original audio, resampled and level-normalised as in training;
//! 2. its exact log-mel spectrogram through Griffin-Lim (vocoder loss only);
//! 3. its tokens decoded with the first residual level only, through Griffin-Lim;
//! 4. its tokens decoded with all codebooks, through Griffin-Lim (the best a
//!    perfect language model could sound);
//!
//! and, when a trained neural vocoder run is given with `--vocoder`,
//!
//! 5. its exact log-mel spectrogram through the neural vocoder;
//! 6. its tokens decoded with all codebooks, through the neural vocoder (the
//!    best a perfect language model could sound in the studio);
//!
//! and measures, in mean absolute log-mel difference, the error the
//! tokenizer adds and how consistently each vocoder reproduces the
//! spectrogram it was given.

use super::open_gpu;
use anyhow::{Result, ensure};
use clap::Args;
use serde::Serialize;
use staleynet_audio::griffin_lim::{GriffinLimConfig, griffin_lim};
use staleynet_audio::resample::resample;
use staleynet_audio::wav::write_mono_pcm16;
use staleynet_audio::{MelAnalyzer, MelConfig, decode_mono};
use staleynet_cuda::Gpu;
use staleynet_data::catalog::Split;
use staleynet_data::corpus::TrackFilter;
use staleynet_data::prepare::normalize_level;
use staleynet_data::store::{CacheLayout, DatasetManifest, PreparedTrack, read_codes, read_f16};
use staleynet_data::tokenizer::AudioTokenizer;
use staleynet_model::vocoder::{self, Vocoder};
use staleynet_train::vocoder::preferred_checkpoint;
use std::path::{Path, PathBuf};

/// Arguments of `staleynet reconstruct`.
#[derive(Args)]
pub struct Arguments {
    /// Tokenised dataset cache.
    #[arg(long, default_value = "data/cache")]
    cache: PathBuf,

    /// Where the WAV files and `report.json` are written.
    #[arg(long, default_value = "samples/reconstruction")]
    out: PathBuf,

    /// Keep tracks whose captions contain one of these words.
    #[arg(long, value_delimiter = ',', default_value = "vocal,vocals,voice,singer,singing,sung,sings")]
    require: Vec<String>,

    /// Number of tracks to render.
    #[arg(long, default_value_t = 4)]
    tracks: usize,

    /// Start of each excerpt, in seconds (clamped to the track).
    #[arg(long, default_value_t = 30.0)]
    offset: f32,

    /// Length of each excerpt, in seconds.
    #[arg(long, default_value_t = 12.0)]
    seconds: f32,

    /// Griffin-Lim iterations, as used by `sing`.
    #[arg(long, default_value_t = 64)]
    griffin_lim_iterations: usize,

    /// A trained neural vocoder run (`staleynet vocoder train`) to render
    /// through as well as Griffin-Lim.
    #[arg(long)]
    vocoder: Option<PathBuf>,
}

/// Measurements for one excerpt. All errors are mean absolute differences
/// of natural-log mel energies (1 unit is a factor of e in energy, about 4.3 dB).
#[derive(Serialize)]
struct ExcerptReport {
    /// Track id.
    id: String,

    /// One of its captions.
    caption: String,

    /// Excerpt start, seconds.
    offset: f32,

    /// Tokenizer error after decoding with the first 1, 2, … codebooks.
    tokenizer_error_by_levels: Vec<f32>,

    /// Log-mel of the Griffin-Lim waveform against the exact log-mel it was given.
    vocoder_error_from_exact_mel: f32,

    /// Log-mel of the Griffin-Lim waveform made from tokens against the
    /// track's exact log-mel: everything lost after a perfect language model.
    end_to_end_error_from_tokens: f32,

    /// Log-mel of the neural vocoder's waveform against the exact log-mel it
    /// was given, when a vocoder run was given.
    #[serde(skip_serializing_if = "Option::is_none")]
    neural_vocoder_error_from_exact_mel: Option<f32>,

    /// Log-mel of the neural vocoder's waveform made from tokens against the
    /// track's exact log-mel, when a vocoder run was given.
    #[serde(skip_serializing_if = "Option::is_none")]
    neural_end_to_end_error_from_tokens: Option<f32>,
}

/// Mean absolute difference over the shorter of two log-mels.
fn mean_absolute_error(reference: &[f32], estimate: &[f32]) -> f32 {
    let compared = reference.len().min(estimate.len());
    reference[..compared]
        .iter()
        .zip(&estimate[..compared])
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / compared.max(1) as f32
}

/// Captions that mark a track as having no singing, whatever else it says.
const NO_VOICE_WORDS: [&str; 2] = ["instrumental", "no vocals"];

/// Renders and measures an excerpt of each selected held-out track.
pub fn run(arguments: Arguments) -> Result<()> {
    let layout = CacheLayout::new(&arguments.cache);
    let manifest = DatasetManifest::load(&layout)?;
    std::fs::create_dir_all(&arguments.out)?;

    let pipeline = Pipeline {
        tokenizer: AudioTokenizer::load(&layout.tokenizer())?,
        analyzer: MelAnalyzer::new(MelConfig::from(manifest.mel)),
        griffin: GriffinLimConfig { iterations: arguments.griffin_lim_iterations, seed: 0 },
        neural: arguments
            .vocoder
            .as_deref()
            .map(NeuralVocoder::load)
            .transpose()?,
        layout,
    };

    let selected = held_out_vocal_tracks(&manifest, &arguments);
    ensure!(!selected.is_empty(), "no validation track matches {:?}", arguments.require);

    let mut reports = Vec::with_capacity(selected.len());
    for entry in selected {
        let report = pipeline.reconstruct(entry, &arguments)?;
        eprintln!(
            "{}: tokenizer {:.3?}, Griffin-Lim on exact mel {:.3}, tokens end to end {:.3}",
            report.id, report.tokenizer_error_by_levels, report.vocoder_error_from_exact_mel, report.end_to_end_error_from_tokens
        );

        if let (Some(exact), Some(tokens)) = (report.neural_vocoder_error_from_exact_mel, report.neural_end_to_end_error_from_tokens) {
            eprintln!("{}: neural vocoder on exact mel {exact:.3}, tokens end to end {tokens:.3}", report.id);
        }
        reports.push(report);
    }

    std::fs::write(arguments.out.join("report.json"), serde_json::to_string_pretty(&reports)?)?;
    Ok(())
}

/// The first `arguments.tracks` validation tracks whose captions mention a
/// voice and do not describe them as instrumental.
fn held_out_vocal_tracks<'a>(manifest: &'a DatasetManifest, arguments: &Arguments) -> Vec<&'a PreparedTrack> {
    let filter = TrackFilter { require_any: arguments.require.clone(), exclude_any: NO_VOICE_WORDS.map(String::from).to_vec() };
    let captions = |entry: &PreparedTrack| {
        entry
            .track
            .captions
            .iter()
            .map(|caption| caption.text.clone())
            .collect::<Vec<_>>()
    };

    manifest
        .tracks
        .iter()
        .filter(|entry| entry.track.split == Split::Validation && filter.accepts(&captions(entry)))
        .take(arguments.tracks)
        .collect()
}

/// A stretch of a track, in spectrogram frames, aligned to whole tokenizer steps.
struct Excerpt {
    /// First frame.
    start_frame: usize,

    /// Number of frames.
    frames: usize,
}

impl Excerpt {
    /// The excerpt of `arguments.seconds` starting near `arguments.offset`,
    /// shortened or moved earlier where the track is too short.
    fn within(track_frames: usize, frames_per_second: f32, frames_per_step: usize, arguments: &Arguments) -> Self {
        let to_whole_steps = |frames: usize| frames / frames_per_step * frames_per_step;

        let frames = to_whole_steps((arguments.seconds * frames_per_second) as usize).min(track_frames);
        let latest_start = track_frames - frames;
        let start_frame = to_whole_steps(((arguments.offset * frames_per_second) as usize).min(latest_start));

        Self { start_frame, frames }
    }

    /// The excerpt's rows of a `[frame, band]` log-mel spectrogram.
    fn of_log_mel<'a>(&self, log_mel: &'a [f32], bands: usize) -> &'a [f32] {
        &log_mel[self.start_frame * bands..(self.start_frame + self.frames) * bands]
    }

    /// The excerpt's codes, `[step, codebook]`.
    fn of_codes<'a>(&self, codes: &'a [u32], frames_per_step: usize, codebooks: usize) -> &'a [u32] {
        let first_step = self.start_frame / frames_per_step;
        let steps = self.frames / frames_per_step;
        &codes[first_step * codebooks..(first_step + steps) * codebooks]
    }
}

/// An excerpt rendered through the neural vocoder.
struct NeuralRenders {
    /// From its exact log-mel spectrogram.
    from_mel: Vec<f32>,

    /// From its tokens decoded with all codebooks.
    from_tokens: Vec<f32>,
}

/// The lossy stages that follow the language model, and where their inputs live.
struct Pipeline {
    /// The cache's tokenised spectrograms.
    layout: CacheLayout,

    /// Codes to log-mel.
    tokenizer: AudioTokenizer,

    /// Log-mel analysis with the cache's settings.
    analyzer: MelAnalyzer,

    /// Log-mel to waveform.
    griffin: GriffinLimConfig,

    /// Log-mel to waveform with a trained network, when one was given.
    neural: Option<NeuralVocoder>,
}

/// A trained neural vocoder and the device it runs on.
struct NeuralVocoder {
    /// The CUDA device holding its weights.
    gpu: Gpu,

    /// The network.
    vocoder: Vocoder,
}

impl NeuralVocoder {
    /// Loads the preferred checkpoint (`best`, else `latest`) of a vocoder run.
    fn load(run: &Path) -> Result<Self> {
        let gpu = open_gpu()?;
        let checkpoint = preferred_checkpoint(run);
        let restored = vocoder::checkpoint::load(&gpu, &checkpoint)?;
        eprintln!("neural vocoder: {} (step {})", checkpoint.display(), restored.step);

        Ok(Self { gpu, vocoder: restored.vocoder })
    }

    /// Renders an excerpt's exact log-mel and its decoded tokens.
    fn render(&self, exact: &[f32], from_tokens: &[f32]) -> Result<NeuralRenders> {
        let from_mel = self.vocoder.synthesize(&self.gpu, exact)?;
        let from_tokens = self.vocoder.synthesize(&self.gpu, from_tokens)?;

        Ok(NeuralRenders { from_mel, from_tokens })
    }
}

impl Pipeline {
    /// Renders the four versions of one excerpt, six with a neural vocoder,
    /// and measures their errors.
    fn reconstruct(&self, entry: &PreparedTrack, arguments: &Arguments) -> Result<ExcerptReport> {
        let id = &entry.track.id;
        let bands = self.analyzer.config.mel_bands;
        let frames_per_step = self.tokenizer.config.frames_per_step;

        let log_mel = read_f16(&self.layout.mel(id))?;
        let excerpt = Excerpt::within(log_mel.len() / bands, self.analyzer.config.frames_per_second(), frames_per_step, arguments);
        let exact = excerpt.of_log_mel(&log_mel, bands);

        let codes = read_codes(&self.layout.codes(id))?;
        let excerpt_codes = excerpt.of_codes(&codes, frames_per_step, self.tokenizer.config.codebooks);
        let decoded: Vec<Vec<f32>> = (1..=self.tokenizer.config.levels())
            .map(|levels| self.tokenizer.decode(excerpt_codes, levels))
            .collect();

        let from_mel = griffin_lim(&self.analyzer, exact, self.griffin);
        let from_coarse_tokens = griffin_lim(&self.analyzer, &decoded[0], self.griffin);
        let from_tokens = griffin_lim(&self.analyzer, &decoded[decoded.len() - 1], self.griffin);

        let all_codebooks = &decoded[decoded.len() - 1];
        let neural = self
            .neural
            .as_ref()
            .map(|network| network.render(exact, all_codebooks))
            .transpose()?;

        let original = self.original_excerpt(entry, &excerpt)?;
        let mut renders = vec![
            ("1-original", original.as_slice()),
            ("2-exact-mel-griffin-lim", &from_mel),
            ("3-tokens-1-codebook", &from_coarse_tokens),
            ("4-tokens-all-codebooks", &from_tokens),
        ];
        if let Some(neural) = &neural {
            renders.push(("5-exact-mel-vocoder", &neural.from_mel));
            renders.push(("6-tokens-all-codebooks-vocoder", &neural.from_tokens));
        }
        self.write_renders(&arguments.out, id, &renders)?;

        let error_against_exact = |waveform: &[f32]| mean_absolute_error(exact, &self.analyzer.log_mel(waveform));

        Ok(ExcerptReport {
            id: id.clone(),
            caption: entry
                .track
                .captions
                .first()
                .map(|caption| caption.text.clone())
                .unwrap_or_default(),
            offset: excerpt.start_frame as f32 / self.analyzer.config.frames_per_second(),
            tokenizer_error_by_levels: decoded
                .iter()
                .map(|estimate| mean_absolute_error(exact, estimate))
                .collect(),
            vocoder_error_from_exact_mel: error_against_exact(&from_mel),
            end_to_end_error_from_tokens: error_against_exact(&from_tokens),
            neural_vocoder_error_from_exact_mel: neural
                .as_ref()
                .map(|renders| error_against_exact(&renders.from_mel)),
            neural_end_to_end_error_from_tokens: neural
                .as_ref()
                .map(|renders| error_against_exact(&renders.from_tokens)),
        })
    }

    /// The excerpt of the original recording, resampled and level-normalised
    /// as during preparation.
    fn original_excerpt(&self, entry: &PreparedTrack, excerpt: &Excerpt) -> Result<Vec<f32>> {
        let audio = decode_mono(&entry.track.audio_path)?;
        let mut original = resample(&audio.samples, audio.sample_rate, self.analyzer.config.sample_rate);
        normalize_level(&mut original);

        let hop = self.analyzer.config.hop_size;
        let first_sample = (excerpt.start_frame * hop).min(original.len());
        let last_sample = (first_sample + excerpt.frames * hop).min(original.len());
        Ok(original[first_sample..last_sample].to_vec())
    }

    /// Writes each `(name, samples)` render as `<id>-<name>.wav` in `out`.
    fn write_renders(&self, out: &Path, id: &str, renders: &[(&str, &[f32])]) -> Result<()> {
        let rate = self.analyzer.config.sample_rate;

        for (name, samples) in renders {
            write_mono_pcm16(&out.join(format!("{id}-{name}.wav")), samples, rate)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_signals_have_zero_error() {
        assert_eq!(mean_absolute_error(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn error_is_the_mean_absolute_difference() {
        assert_eq!(mean_absolute_error(&[1.0, 2.0], &[2.0, 4.0]), 1.5);
    }

    #[test]
    fn comparison_stops_at_the_shorter_signal() {
        assert_eq!(mean_absolute_error(&[1.0, 2.0, 100.0], &[2.0, 2.0]), 0.5);
    }

    #[test]
    fn empty_signals_have_zero_error() {
        assert_eq!(mean_absolute_error(&[], &[]), 0.0);
    }
}
