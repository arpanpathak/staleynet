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
//! and measures, in mean absolute log-mel difference, the error the
//! tokenizer adds and how consistently Griffin-Lim reproduces the spectrogram
//! it was given.

use anyhow::{Result, ensure};
use clap::Args;
use serde::Serialize;
use staleynet_audio::griffin_lim::{GriffinLimConfig, griffin_lim};
use staleynet_audio::resample::resample;
use staleynet_audio::wav::write_mono_pcm16;
use staleynet_audio::{MelAnalyzer, MelConfig, decode_mono};
use staleynet_data::catalog::Split;
use staleynet_data::corpus::TrackFilter;
use staleynet_data::prepare::normalize_level;
use staleynet_data::store::{CacheLayout, DatasetManifest, read_codes, read_f16};
use staleynet_data::tokenizer::AudioTokenizer;
use std::path::PathBuf;

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
}

/// Measurements for one excerpt. All errors are mean absolute differences
/// of natural-log mel energies (1 unit ≈ 8.7 dB).
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
}

/// Mean absolute difference over the shorter of two log-mels.
fn mean_absolute_error(reference: &[f32], estimate: &[f32]) -> f32 {
    let compared = reference.len().min(estimate.len());
    reference[..compared].iter().zip(&estimate[..compared]).map(|(a, b)| (a - b).abs()).sum::<f32>() / compared.max(1) as f32
}

pub fn run(arguments: Arguments) -> Result<()> {
    let layout = CacheLayout::new(&arguments.cache);
    let manifest = DatasetManifest::load(&layout)?;
    let tokenizer = AudioTokenizer::load(&layout.tokenizer())?;
    let analyzer = MelAnalyzer::new(MelConfig::from(manifest.mel));
    let bands = manifest.mel.mel_bands;
    let frames_per_step = tokenizer.config.frames_per_step;
    let codebooks = tokenizer.config.codebooks;
    let griffin = GriffinLimConfig { iterations: arguments.griffin_lim_iterations, seed: 0 };
    let filter = TrackFilter { require_any: arguments.require.clone(), exclude_any: vec!["instrumental".into(), "no vocals".into()] };
    std::fs::create_dir_all(&arguments.out)?;

    let selected: Vec<_> = manifest
        .tracks
        .iter()
        .filter(|entry| entry.track.split == Split::Validation)
        .filter(|entry| filter.accepts(&entry.track.captions.iter().map(|caption| caption.text.clone()).collect::<Vec<_>>()))
        .take(arguments.tracks)
        .collect();
    ensure!(!selected.is_empty(), "no validation track matches {:?}", arguments.require);

    let mut reports = Vec::new();
    for entry in selected {
        let id = &entry.track.id;
        let log_mel = read_f16(&layout.mel(id))?;
        let frames = log_mel.len() / bands;
        let excerpt_frames =
            ((arguments.seconds * analyzer.config.frames_per_second()) as usize / frames_per_step * frames_per_step).min(frames);
        let latest_start = frames - excerpt_frames;
        let start_frame =
            ((arguments.offset * analyzer.config.frames_per_second()) as usize).min(latest_start) / frames_per_step * frames_per_step;
        let exact = &log_mel[start_frame * bands..(start_frame + excerpt_frames) * bands];

        let codes = read_codes(&layout.codes(id))?;
        let first_step = start_frame / frames_per_step;
        let steps = excerpt_frames / frames_per_step;
        let excerpt_codes = &codes[first_step * codebooks..(first_step + steps) * codebooks];
        let decoded: Vec<Vec<f32>> = (1..=tokenizer.config.levels()).map(|levels| tokenizer.decode(excerpt_codes, levels)).collect();

        let audio = decode_mono(&entry.track.audio_path)?;
        let mut original = resample(&audio.samples, audio.sample_rate, analyzer.config.sample_rate);
        normalize_level(&mut original);
        let first_sample = (start_frame * analyzer.config.hop_size).min(original.len());
        let last_sample = (first_sample + excerpt_frames * analyzer.config.hop_size).min(original.len());

        let from_mel = griffin_lim(&analyzer, exact, griffin);
        let from_coarse_tokens = griffin_lim(&analyzer, &decoded[0], griffin);
        let from_tokens = griffin_lim(&analyzer, &decoded[decoded.len() - 1], griffin);

        let rate = analyzer.config.sample_rate;
        write_mono_pcm16(&arguments.out.join(format!("{id}-1-original.wav")), &original[first_sample..last_sample], rate)?;
        write_mono_pcm16(&arguments.out.join(format!("{id}-2-exact-mel-griffin-lim.wav")), &from_mel, rate)?;
        write_mono_pcm16(&arguments.out.join(format!("{id}-3-tokens-1-codebook.wav")), &from_coarse_tokens, rate)?;
        write_mono_pcm16(&arguments.out.join(format!("{id}-4-tokens-all-codebooks.wav")), &from_tokens, rate)?;

        let report = ExcerptReport {
            id: id.clone(),
            caption: entry.track.captions.first().map(|caption| caption.text.clone()).unwrap_or_default(),
            offset: start_frame as f32 / analyzer.config.frames_per_second(),
            tokenizer_error_by_levels: decoded.iter().map(|estimate| mean_absolute_error(exact, estimate)).collect(),
            vocoder_error_from_exact_mel: mean_absolute_error(exact, &analyzer.log_mel(&from_mel)),
            end_to_end_error_from_tokens: mean_absolute_error(exact, &analyzer.log_mel(&from_tokens)),
        };
        eprintln!(
            "{id}: tokenizer {:.3?}, Griffin-Lim on exact mel {:.3}, tokens end to end {:.3}",
            report.tokenizer_error_by_levels, report.vocoder_error_from_exact_mel, report.end_to_end_error_from_tokens
        );
        reports.push(report);
    }

    std::fs::write(arguments.out.join("report.json"), serde_json::to_string_pretty(&reports)?)?;
    Ok(())
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
