//! `staleynet evaluate`: does the model read the lyrics?
//!
//! The same validation crops are scored three times: with the lyrics sung in
//! them, with no lyrics, and with lyrics from another track. A model that
//! ignores its lyrics field scores all three alike. One that reads it scores
//! the aligned lyrics best, and its gap to the mismatched lyrics is the
//! information, in nats per code, that the words carry about the sound.

use super::open_gpu;
use anyhow::{Result, ensure};
use clap::Args;
use serde::Serialize;
use staleynet_data::catalog::Split;
use staleynet_data::corpus::{BatchSampler, LyricsCondition, SamplerOptions, TokenCorpus, TrackFilter};
use staleynet_data::store::{CacheLayout, DatasetManifest};
use staleynet_model::{Random, SequenceShape, TrainingBatch, TrainingWorkspace, checkpoint};
use staleynet_train::preferred_checkpoint;
use std::path::PathBuf;

/// How many standard errors the paired gap must exceed to count as evidence
/// (about 97.5 % one-sided confidence under a normal approximation).
const PAIRED_Z: f32 = 2.0;

/// Arguments of `staleynet evaluate`.
#[derive(Args)]
pub struct Arguments {
    /// Training run to evaluate (its best checkpoint, else its latest).
    #[arg(long)]
    run: PathBuf,

    /// Evaluate this checkpoint directory instead (e.g. runs/x/latest);
    /// results are still written into the run directory.
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// Tokenised dataset whose tracks have lyrics.
    #[arg(long, default_value = "data/cache-jamendolyrics")]
    cache: PathBuf,

    /// Which split to score.
    #[arg(long, value_enum, default_value_t = SplitArgument::Validation)]
    split: SplitArgument,

    /// Sequences per batch.
    #[arg(long, default_value_t = 8)]
    batch_size: usize,

    /// Caption and audio positions per sequence (the lyrics field comes on top).
    #[arg(long, default_value_t = 512)]
    sequence_length: usize,

    /// Batches per condition.
    #[arg(long, default_value_t = 32)]
    batches: usize,

    /// Seed of the crops; the same crops are used for every condition.
    #[arg(long, default_value_t = 7)]
    seed: u64,
}

/// Command-line spelling of [`Split`].
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum SplitArgument {
    /// Training tracks.
    Train,

    /// Held-out tracks.
    Validation,
}

/// Mean loss under one lyrics condition.
#[derive(Serialize)]
struct ConditionScore {
    /// The condition.
    condition: String,

    /// Mean cross-entropy over all codebooks, nats.
    loss: f32,

    /// Per-codebook cross-entropy.
    loss_per_codebook: Vec<f32>,
}

pub fn run(arguments: Arguments) -> Result<()> {
    let gpu = open_gpu()?;
    let checkpoint_directory = arguments.checkpoint.clone().unwrap_or_else(|| preferred_checkpoint(&arguments.run));
    let (model, metadata) = checkpoint::load(&gpu, &checkpoint_directory, None)?;
    ensure!(model.config.lyrics_bytes > 0, "{} has no lyrics field", checkpoint_directory.display());

    let layout = CacheLayout::new(&arguments.cache);
    let manifest = DatasetManifest::load(&layout)?;
    let split = match arguments.split {
        SplitArgument::Train => Split::Train,
        SplitArgument::Validation => Split::Validation,
    };
    let corpus = TokenCorpus::load(&layout, &manifest, split, &TrackFilter::default(), model.config.codebooks)?.retain_tracks_with_lyrics();
    ensure!(corpus.tracks.len() >= 2, "need at least two tracks with lyrics in the {split:?} split");

    let shape = SequenceShape { batch: arguments.batch_size, time: arguments.sequence_length + model.config.lyrics_bytes };
    let mut workspace = TrainingWorkspace::new(&gpu, &model.config, shape)?;
    let mut batch = TrainingBatch::empty(shape, model.config.codebooks);
    eprintln!("{} (step {}), {} {split:?} tracks with lyrics", checkpoint_directory.display(), metadata.step, corpus.tracks.len());

    let mut scores = Vec::new();
    let mut batch_losses: Vec<Vec<f32>> = Vec::new();
    for condition in [LyricsCondition::Aligned, LyricsCondition::Removed, LyricsCondition::Mismatched] {
        let options = SamplerOptions { lyrics: condition, ..SamplerOptions::VALIDATION };
        let sampler = BatchSampler::new(model.vocabulary(), options);
        let mut random = Random::new(arguments.seed);
        let mut total = 0.0;
        let mut per_codebook = vec![0.0; model.config.codebooks];
        let mut losses = Vec::with_capacity(arguments.batches);

        for _ in 0..arguments.batches {
            sampler.fill(&corpus, &mut batch, &mut random);
            let loss = model.forward_loss(&gpu, &mut workspace, &batch)?;
            losses.push(loss.mean);
            total += loss.mean;
            for (sum, value) in per_codebook.iter_mut().zip(&loss.per_codebook) {
                *sum += value;
            }
        }

        let batches = arguments.batches as f32;
        let score = ConditionScore {
            condition: format!("{condition:?}"),
            loss: total / batches,
            loss_per_codebook: per_codebook.iter().map(|sum| sum / batches).collect(),
        };
        eprintln!("{:>10}: {:.4} {:.3?}", score.condition, score.loss, score.loss_per_codebook);
        scores.push(score);
        batch_losses.push(losses);
    }

    // The conditions share their crops, so the per-batch differences are
    // paired; their standard error measures how much of the gap is noise.
    let differences: Vec<f32> = batch_losses[2].iter().zip(&batch_losses[0]).map(|(mismatched, aligned)| mismatched - aligned).collect();
    let count = differences.len() as f32;
    let gap = differences.iter().sum::<f32>() / count;
    let variance = differences.iter().map(|difference| (difference - gap).powi(2)).sum::<f32>() / (count - 1.0).max(1.0);
    let standard_error = (variance / count).sqrt();
    let verdict =
        if gap > PAIRED_Z * standard_error { "the model uses the lyrics" } else { "no reliable evidence that the lyrics are used" };
    eprintln!("mismatched minus aligned: {gap:+.4} ± {standard_error:.4} nats per code ({verdict})");
    let report = serde_json::json!({
        "checkpoint": checkpoint_directory, "step": metadata.step, "split": format!("{split:?}"), "scores": scores,
        "mismatched_minus_aligned": gap, "standard_error": standard_error, "verdict": verdict,
    });
    let name = format!("lyrics-evaluation-{split:?}-step-{}.json", metadata.step).to_lowercase();
    std::fs::write(arguments.run.join(name), serde_json::to_string_pretty(&report)?)?;
    Ok(())
}
