//! `staleynet tokenize`: train the residual quantiser, encode every track and
//! report how much each level reduces reconstruction error.

use super::open_gpu;
use anyhow::Result;
use clap::Args;
use serde::Serialize;
use staleynet_audio::griffin_lim::{GriffinLimConfig, griffin_lim};
use staleynet_audio::wav::write_mono_pcm16;
use staleynet_audio::{MelAnalyzer, MelConfig};
use staleynet_cuda::Gpu;
use staleynet_data::catalog::Split;
use staleynet_data::store::{CacheLayout, DatasetManifest, read_f16, write_codes};
use staleynet_data::tokenizer::{AudioTokenizer, TokenizerConfig};
use std::path::PathBuf;

/// Griffin-Lim iterations for listening examples.
const LISTENING_ITERATIONS: usize = 64;

/// Seconds of each listening example.
const LISTENING_SECONDS: f32 = 20.0;

/// Validation tracks rendered as listening examples.
const LISTENING_TRACKS: usize = 3;

/// Arguments of `staleynet tokenize`.
#[derive(Args)]
pub struct Arguments {
    /// Feature cache written by `staleynet prepare`.
    #[arg(long, default_value = "data/cache")]
    cache: PathBuf,

    /// Residual levels within each band group.
    #[arg(long, default_value_t = TokenizerConfig::default().codebooks)]
    levels: usize,

    /// Equal groups of mel bands, each quantised separately (product quantisation).
    #[arg(long, default_value_t = TokenizerConfig::default().band_groups)]
    band_groups: usize,

    /// Centroids per level.
    #[arg(long, default_value_t = TokenizerConfig::default().codebook_size)]
    codebook_size: usize,

    /// Log-mel frames joined into one step; fewer frames per step keep more
    /// temporal detail at the cost of longer sequences.
    #[arg(long, default_value_t = TokenizerConfig::default().frames_per_step)]
    frames_per_step: usize,

    /// Lloyd iterations per level.
    #[arg(long, default_value_t = TokenizerConfig::default().iterations)]
    iterations: usize,

    /// Encode with this existing tokenizer (e.g. data/cache/tokenizer.json)
    /// instead of training one, so a new dataset shares the vocabulary of a
    /// model already trained. A copy is written into the cache.
    #[arg(long)]
    reuse: Option<PathBuf>,

    /// Also write original-vs-tokenized Griffin-Lim renderings here.
    #[arg(long)]
    listen: Option<PathBuf>,
}

/// Reconstruction quality of one track.
#[derive(Serialize)]
struct TrackReport {
    /// Track id.
    id: String,

    /// Split.
    split: Split,

    /// Mean |log-mel error| after decoding with 1, 2, … levels.
    log_mel_error_by_levels: Vec<f32>,
}

/// Mean absolute difference over the shorter of two log-mels.
fn mean_absolute_error(reference: &[f32], estimate: &[f32]) -> f32 {
    let compared = reference.len().min(estimate.len());
    reference[..compared].iter().zip(&estimate[..compared]).map(|(a, b)| (a - b).abs()).sum::<f32>() / compared.max(1) as f32
}

/// Fits a new tokenizer to the training split of the cache.
fn train_tokenizer(gpu: &Gpu, arguments: &Arguments, layout: &CacheLayout, manifest: &DatasetManifest) -> Result<AudioTokenizer> {
    let mel_bands = manifest.mel.mel_bands;
    let config = TokenizerConfig {
        codebooks: arguments.levels * arguments.band_groups,
        band_groups: arguments.band_groups,
        codebook_size: arguments.codebook_size,
        frames_per_step: arguments.frames_per_step,
        iterations: arguments.iterations,
        ..TokenizerConfig::default()
    };

    let mut vectors = Vec::new();
    for entry in manifest.tracks.iter().filter(|entry| entry.track.split == Split::Train) {
        let log_mel = read_f16(&layout.mel(&entry.track.id))?;
        vectors.extend(AudioTokenizer::step_vectors(&log_mel, mel_bands, config.frames_per_step, &manifest.normalization));
    }
    eprintln!("training tokenizer on {} steps", vectors.len() / (config.frames_per_step * mel_bands));

    AudioTokenizer::train(gpu, config, mel_bands, manifest.normalization.clone(), vectors, |progress| {
        eprintln!(
            "  level {} iteration {:2}: mse {:.4}, reseeded {}",
            progress.level, progress.iteration, progress.mean_squared_error, progress.reseeded
        );
    })
    .map_err(Into::into)
}

pub fn run(arguments: Arguments) -> Result<()> {
    let gpu = open_gpu()?;
    let layout = CacheLayout::new(&arguments.cache);
    let manifest = DatasetManifest::load(&layout)?;
    let mel_bands = manifest.mel.mel_bands;
    let tokenizer = match &arguments.reuse {
        Some(path) => AudioTokenizer::load(path)?,
        None => train_tokenizer(&gpu, &arguments, &layout, &manifest)?,
    };
    tokenizer.save(&layout.tokenizer())?;

    let analyzer = MelAnalyzer::new(MelConfig::from(manifest.mel));
    let mut reports = Vec::with_capacity(manifest.tracks.len());
    let mut rendered = 0;

    for entry in &manifest.tracks {
        let log_mel = read_f16(&layout.mel(&entry.track.id))?;
        let codes = tokenizer.encode(&gpu, &log_mel)?;
        write_codes(&layout.codes(&entry.track.id), &codes)?;

        let errors =
            (1..=tokenizer.config.levels()).map(|levels| mean_absolute_error(&log_mel, &tokenizer.decode(&codes, levels))).collect();
        reports.push(TrackReport { id: entry.track.id.clone(), split: entry.track.split, log_mel_error_by_levels: errors });

        let Some(listen) = &arguments.listen else { continue };
        if entry.track.split != Split::Validation || rendered == LISTENING_TRACKS {
            continue;
        }

        std::fs::create_dir_all(listen)?;
        let frames = (LISTENING_SECONDS * analyzer.config.frames_per_second()) as usize * mel_bands;
        let griffin = GriffinLimConfig { iterations: LISTENING_ITERATIONS, seed: 0 };
        let original = griffin_lim(&analyzer, &log_mel[..frames.min(log_mel.len())], griffin);
        let tokenized_mel = tokenizer.decode(&codes, tokenizer.config.codebooks);
        let tokenized = griffin_lim(&analyzer, &tokenized_mel[..frames.min(tokenized_mel.len())], griffin);

        write_mono_pcm16(&listen.join(format!("{}-original-mel.wav", entry.track.id)), &original, analyzer.config.sample_rate)?;
        write_mono_pcm16(&listen.join(format!("{}-tokenized.wav", entry.track.id)), &tokenized, analyzer.config.sample_rate)?;
        rendered += 1;
    }

    for split in [Split::Train, Split::Validation] {
        let selected: Vec<&TrackReport> = reports.iter().filter(|report| report.split == split).collect();
        if selected.is_empty() {
            continue;
        }
        let mean_by_level: Vec<f32> = (0..tokenizer.config.levels())
            .map(|level| selected.iter().map(|report| report.log_mel_error_by_levels[level]).sum::<f32>() / selected.len() as f32)
            .collect();
        eprintln!("{split:?}: mean |log-mel error| by levels used {mean_by_level:.3?}");
    }

    std::fs::write(arguments.cache.join("tokenizer_report.json"), serde_json::to_string_pretty(&reports)?)?;
    Ok(())
}
