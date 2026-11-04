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
use staleynet_cuda::Gpu;
use staleynet_data::catalog::Split;
use staleynet_data::corpus::{BatchSampler, LyricsCondition, SamplerOptions, TokenCorpus, TrackFilter};
use staleynet_data::store::{CacheLayout, DatasetManifest};
use staleynet_model::{Gpt, Random, SequenceShape, TrainingBatch, TrainingWorkspace, checkpoint};
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

/// The three lyrics conditions, in the order they are scored and reported.
const CONDITIONS: [LyricsCondition; 3] = [LyricsCondition::Aligned, LyricsCondition::Removed, LyricsCondition::Mismatched];

/// Scores the same crops under each lyrics condition and reports whether the
/// model reads its lyrics.
pub fn run(arguments: Arguments) -> Result<()> {
    let gpu = open_gpu()?;

    let checkpoint_directory = arguments.checkpoint.clone().unwrap_or_else(|| preferred_checkpoint(&arguments.run));
    let (model, metadata) = checkpoint::load(&gpu, &checkpoint_directory, None)?;
    ensure!(model.config.lyrics_bytes > 0, "{} has no lyrics field", checkpoint_directory.display());

    let split = match arguments.split {
        SplitArgument::Train => Split::Train,
        SplitArgument::Validation => Split::Validation,
    };

    let layout = CacheLayout::new(&arguments.cache);
    let manifest = DatasetManifest::load(&layout)?;
    let corpus = TokenCorpus::load(&layout, &manifest, split, &TrackFilter::default(), model.config.codebooks)?.retain_tracks_with_lyrics();
    ensure!(corpus.tracks.len() >= 2, "need at least two tracks with lyrics in the {split:?} split");

    eprintln!("{} (step {}), {} {split:?} tracks with lyrics", checkpoint_directory.display(), metadata.step, corpus.tracks.len());

    let mut scorer = Scorer::new(&gpu, &model, &corpus, &arguments)?;
    let mut scores = Vec::with_capacity(CONDITIONS.len());
    let mut batch_losses = Vec::with_capacity(CONDITIONS.len());

    for condition in CONDITIONS {
        let (score, losses) = scorer.score(condition)?;
        eprintln!("{:>10}: {:.4} {:.3?}", score.condition, score.loss, score.loss_per_codebook);
        scores.push(score);
        batch_losses.push(losses);
    }

    let paired = PairedDifference::of(&batch_losses[2], &batch_losses[0]);
    let verdict = if paired.is_evidence() { "the model uses the lyrics" } else { "no reliable evidence that the lyrics are used" };
    eprintln!("mismatched minus aligned: {:+.4} ± {:.4} nats per code ({verdict})", paired.mean, paired.standard_error);

    let report = serde_json::json!({
        "checkpoint": checkpoint_directory, "step": metadata.step, "split": format!("{split:?}"), "scores": scores,
        "mismatched_minus_aligned": paired.mean, "standard_error": paired.standard_error, "verdict": verdict,
    });

    let name = format!("lyrics-evaluation-{split:?}-step-{}.json", metadata.step).to_lowercase();
    std::fs::write(arguments.run.join(name), serde_json::to_string_pretty(&report)?)?;
    Ok(())
}

/// Scores a model on fixed crops of a corpus, reusing one workspace and batch.
struct Scorer<'a> {
    /// The device.
    gpu: &'a Gpu,

    /// The model under test.
    model: &'a Gpt,

    /// Tracks with lyrics.
    corpus: &'a TokenCorpus,

    /// Reused device buffers.
    workspace: TrainingWorkspace,

    /// Reused host batch.
    batch: TrainingBatch,

    /// Batches per condition.
    batches: usize,

    /// Seed of the crops; the same for every condition, so the scores are paired.
    seed: u64,
}

impl<'a> Scorer<'a> {
    /// Allocates the buffers for batches of `arguments.batch_size` sequences.
    fn new(gpu: &'a Gpu, model: &'a Gpt, corpus: &'a TokenCorpus, arguments: &Arguments) -> Result<Self> {
        let shape = SequenceShape { batch: arguments.batch_size, time: arguments.sequence_length + model.config.lyrics_bytes };

        Ok(Self {
            gpu,
            model,
            corpus,
            workspace: TrainingWorkspace::new(gpu, &model.config, shape)?,
            batch: TrainingBatch::empty(shape, model.config.codebooks),
            batches: arguments.batches,
            seed: arguments.seed,
        })
    }

    /// The mean loss under `condition`, and the loss of each batch.
    fn score(&mut self, condition: LyricsCondition) -> Result<(ConditionScore, Vec<f32>)> {
        let sampler = BatchSampler::new(self.model.vocabulary(), SamplerOptions { lyrics: condition, ..SamplerOptions::VALIDATION });
        let mut random = Random::new(self.seed);

        let mut losses = Vec::with_capacity(self.batches);
        let mut per_codebook = vec![0.0; self.model.config.codebooks];
        for _ in 0..self.batches {
            sampler.fill(self.corpus, &mut self.batch, &mut random);
            let loss = self.model.forward_loss(self.gpu, &mut self.workspace, &self.batch)?;

            losses.push(loss.mean);
            per_codebook.iter_mut().zip(&loss.per_codebook).for_each(|(sum, value)| *sum += value);
        }

        let batches = self.batches as f32;
        let score = ConditionScore {
            condition: format!("{condition:?}"),
            loss: losses.iter().sum::<f32>() / batches,
            loss_per_codebook: per_codebook.iter().map(|sum| sum / batches).collect(),
        };
        Ok((score, losses))
    }
}

/// The mean of paired differences and its standard error.
#[derive(Debug, PartialEq)]
struct PairedDifference {
    /// Mean of `first[i] − second[i]`, nats per code.
    mean: f32,

    /// Standard error of that mean: the sample standard deviation of the
    /// differences divided by the square root of their number.
    standard_error: f32,
}

impl PairedDifference {
    /// Pairs `first` and `second` element by element. The conditions share
    /// their crops, so most of the variation between crops cancels.
    fn of(first: &[f32], second: &[f32]) -> Self {
        let differences: Vec<f32> = first.iter().zip(second).map(|(a, b)| a - b).collect();
        let count = differences.len() as f32;

        let mean = differences.iter().sum::<f32>() / count;
        let variance = differences.iter().map(|difference| (difference - mean).powi(2)).sum::<f32>() / (count - 1.0).max(1.0);
        Self { mean, standard_error: (variance / count).sqrt() }
    }

    /// True when the mean exceeds [`PAIRED_Z`] standard errors.
    fn is_evidence(&self) -> bool {
        self.mean > PAIRED_Z * self.standard_error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_difference_of_a_constant_gap_has_no_error() {
        let paired = PairedDifference::of(&[3.0, 4.0, 5.0], &[2.0, 3.0, 4.0]);

        assert_eq!(paired, PairedDifference { mean: 1.0, standard_error: 0.0 });
        assert!(paired.is_evidence());
    }

    #[test]
    fn paired_difference_uses_the_sample_variance() {
        // Differences 1 and 3: mean 2, sample variance 2, standard error 1.
        let paired = PairedDifference::of(&[2.0, 5.0], &[1.0, 2.0]);

        assert_eq!(paired, PairedDifference { mean: 2.0, standard_error: 1.0 });
        assert!(!paired.is_evidence());
    }
}
