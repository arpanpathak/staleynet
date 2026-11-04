//! The training loop.
//!
//! A run directory holds everything needed to inspect or resume a run:
//!
//! ```text
//! <run>/
//!   config.json      the TrainingConfig used
//!   tokenizer.json   the tokenizer whose codes the model predicts
//!   metrics.jsonl    one JSON object per logged step / validation
//!   latest/          newest checkpoint, with optimiser state (resumable)
//!   best/            lowest-validation-loss checkpoint, weights only
//! ```
//!
//! Batches are drawn from a random stream derived from `(seed, step)`, so a
//! resumed run sees exactly the batches an uninterrupted run would have.
//! `latest/` carries the AdamW moments and step, so `--resume` continues the
//! same optimisation rather than restarting it; it is rewritten every
//! `checkpoint_every` steps, which bounds the work an interruption can lose.
//!
//! The neural vocoder's training loop lives in [`vocoder`].

pub mod vocoder;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use staleynet_cuda::Gpu;
use staleynet_data::catalog::Split;
use staleynet_data::corpus::{BatchSampler, SamplerOptions, TokenCorpus, TrackFilter};
use staleynet_data::store::{CacheLayout, DatasetManifest};
use staleynet_data::tokenizer::AudioTokenizer;
use staleynet_model::optimizer::{AdamW, OptimizerConfig, StepReport};
use staleynet_model::{Gpt, LossReport, ModelConfig, Parameters, Random, SequenceShape, TrainingBatch, TrainingWorkspace, checkpoint};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// File inside a run directory holding the tokenizer the run was trained with.
pub const RUN_TOKENIZER: &str = "tokenizer.json";

/// Seed offset of the validation batch stream, keeping it disjoint from training streams.
const VALIDATION_STREAM: u64 = u64::MAX;

/// A further tokenised dataset mixed into training, e.g. songs with
/// time-aligned lyrics or your own recordings. It must have been tokenised
/// with the primary cache's tokenizer (`staleynet tokenize --reuse`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataSource {
    /// Its cache directory.
    pub cache: PathBuf,

    /// Which of its tracks to use.
    #[serde(default)]
    pub track_filter: TrackFilter,

    /// How likely each of its tracks is to be drawn, relative to a track of
    /// the primary cache. Small but valuable sets (your takes) deserve more.
    #[serde(default = "DataSource::default_weight")]
    pub weight: f64,
}

impl DataSource {
    /// Same weight as the primary cache's tracks.
    fn default_weight() -> f64 {
        1.0
    }
}

/// Everything that defines a training run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Architecture (codebook fields are overwritten from the tokenizer).
    pub model: ModelConfig,

    /// Optimiser and schedule.
    pub optimizer: OptimizerConfig,

    /// Sequences per step.
    pub batch_size: usize,

    /// Positions per sequence excluding the lyrics field (caption + audio);
    /// at most the context length. The lyrics field is added on top.
    pub sequence_length: usize,

    /// Optimiser steps to run in total.
    pub steps: usize,

    /// Steps between progress reports.
    pub log_every: usize,

    /// Steps between validations.
    pub validate_every: usize,

    /// Batches averaged per validation.
    pub validation_batches: usize,

    /// Steps between `latest/` checkpoints.
    pub checkpoint_every: usize,

    /// Seed of initialisation and batch sampling.
    pub seed: u64,

    /// Which tracks of the primary cache to train and validate on (all by default).
    #[serde(default)]
    pub track_filter: TrackFilter,

    /// Further datasets mixed in.
    #[serde(default)]
    pub extra_sources: Vec<DataSource>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            optimizer: OptimizerConfig::default(),
            batch_size: 8,
            sequence_length: 768,
            steps: 50_000,
            log_every: 25,
            validate_every: 500,
            validation_batches: 16,
            checkpoint_every: 250,
            seed: 1234,
            track_filter: TrackFilter::default(),
            extra_sources: Vec::new(),
        }
    }
}

/// Where the weights of a run come from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StartPoint {
    /// Freshly initialised weights.
    Fresh,

    /// Continue from `<run>/latest`, including optimiser state and step, if it exists.
    ResumeLatest,

    /// Start from the weights of another checkpoint with a fresh optimiser and
    /// schedule (fine-tuning). The architecture may be larger than the
    /// checkpoint's where `checkpoint::load_weights` allows it, e.g. a model
    /// with a new lyrics field warm-started from one without.
    FineTune(PathBuf),
}

/// Something worth reporting while training.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum TrainingEvent {
    /// Run started.
    Started {
        /// Trainable scalars.
        parameters: usize,
        /// Training tracks.
        train_tracks: usize,
        /// Training code steps.
        train_steps: usize,
        /// Validation tracks.
        validation_tracks: usize,
        /// First step of this invocation (non-zero when resumed).
        first_step: usize,
        /// Training tracks that come with lyrics.
        tracks_with_lyrics: usize,
        /// Best validation loss carried over from before a resume.
        best_validation: Option<f32>,
    },

    /// Periodic training progress.
    Progress {
        /// Steps completed.
        step: usize,
        /// Training loss of the latest batch, nats.
        loss: f32,
        /// Per-codebook training loss.
        loss_per_codebook: Vec<f32>,
        /// Gradient norm before clipping.
        gradient_norm: f32,
        /// Learning rate used.
        learning_rate: f32,
        /// Positions processed per second since the previous report.
        positions_per_second: f64,
        /// Milliseconds per step since the previous report.
        milliseconds_per_step: f64,
    },

    /// Validation result.
    Validation {
        /// Steps completed.
        step: usize,
        /// Mean validation loss, nats.
        loss: f32,
        /// Per-codebook validation loss.
        loss_per_codebook: Vec<f32>,
        /// Whether this is the best so far (saved to `best/`).
        improved: bool,
    },
}

/// Paths inside a run directory.
struct RunLayout {
    /// Run root.
    root: PathBuf,
}

impl RunLayout {
    /// `latest/`.
    fn latest(&self) -> PathBuf {
        self.root.join("latest")
    }

    /// `best/`.
    fn best(&self) -> PathBuf {
        self.root.join("best")
    }
}

/// Lowest validation loss in a run's metrics log, so a resumed run only
/// replaces `best/` with something better.
fn best_logged_validation(metrics: &Path) -> Option<f32> {
    let text = std::fs::read_to_string(metrics).ok()?;
    text.lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|event| event["event"] == "validation")
        .filter_map(|event| event["loss"].as_f64())
        .map(|loss| loss as f32)
        .reduce(f32::min)
}

/// Loads the training and validation tracks of the primary cache and every
/// extra source into two corpora.
fn load_corpora(cache: &CacheLayout, config: &TrainingConfig) -> Result<(TokenCorpus, TokenCorpus)> {
    let codebooks = config.model.codebooks;
    let manifest = DatasetManifest::load(cache)?;

    let mut train = TokenCorpus::load(cache, &manifest, Split::Train, &config.track_filter, codebooks)?;
    let mut validation = TokenCorpus::load(cache, &manifest, Split::Validation, &config.track_filter, codebooks)?;

    for source in &config.extra_sources {
        let layout = CacheLayout::new(&source.cache);
        let manifest = DatasetManifest::load(&layout)?;
        let tokenizer =
            AudioTokenizer::load(&layout.tokenizer()).with_context(|| format!("{} is not tokenised", source.cache.display()))?;
        ensure!(tokenizer.config.codebooks == codebooks, "{} was tokenised with a different tokenizer", source.cache.display());

        train.absorb(TokenCorpus::load(&layout, &manifest, Split::Train, &source.track_filter, codebooks)?, source.weight);
        validation.absorb(TokenCorpus::load(&layout, &manifest, Split::Validation, &source.track_filter, codebooks)?, source.weight);
    }

    Ok((train, validation))
}

/// The best checkpoint of a run if one exists, else the latest.
pub fn preferred_checkpoint(run: &Path) -> PathBuf {
    let layout = RunLayout { root: run.to_path_buf() };
    if layout.best().join("model.json").exists() { layout.best() } else { layout.latest() }
}

/// State that lives for the whole run.
struct Session<'a> {
    /// Device.
    gpu: &'a Gpu,

    /// Run configuration.
    config: TrainingConfig,

    /// Run directory layout.
    run: RunLayout,

    /// Network being trained.
    model: Gpt,

    /// Optimiser state.
    optimizer: AdamW,

    /// Reused device buffers for one step.
    workspace: TrainingWorkspace,

    /// Reused host batch.
    batch: TrainingBatch,

    /// Training tracks.
    train: TokenCorpus,

    /// Held-out tracks.
    validation: TokenCorpus,

    /// Lowest validation loss so far.
    best_validation: f32,

    /// Append-only metrics log.
    metrics: File,
}

/// Trains a model on the tokenised dataset in `cache`, writing into `run`.
///
/// The run directory receives the configuration, a copy of the tokenizer, the
/// metrics log, and the `best/` and `latest/` checkpoints. `start` decides
/// whether the weights begin fresh, resume from `latest/`, or are copied from
/// another checkpoint.
pub fn train(
    gpu: &Gpu,
    cache: &CacheLayout,
    run: &Path,
    mut config: TrainingConfig,
    start: StartPoint,
    mut report: impl FnMut(&TrainingEvent),
) -> Result<()> {
    adopt_tokenizer(cache, &mut config)?;

    let (train, validation) = load_corpora(cache, &config)?;
    ensure!(!train.tracks.is_empty(), "no training tracks pass the track filter");

    let run = prepare_run_directory(run, cache, &config)?;
    let state = starting_state(gpu, &config, &run, &start)?;

    let shape = SequenceShape { batch: config.batch_size, time: config.sequence_length + config.model.lyrics_bytes };
    let session = Session {
        gpu,
        workspace: TrainingWorkspace::new(gpu, &config.model, shape)?,
        batch: TrainingBatch::empty(shape, config.model.codebooks),
        metrics: OpenOptions::new()
            .create(true)
            .append(true)
            .open(run.root.join("metrics.jsonl"))?,
        best_validation: state.best_validation,
        model: state.model,
        optimizer: state.optimizer,
        config,
        run,
        train,
        validation,
    };

    session.run(&mut report)
}

/// Sizes the model's output heads to the cache's tokenizer and checks that a
/// training sequence fits in the model's context.
fn adopt_tokenizer(cache: &CacheLayout, config: &mut TrainingConfig) -> Result<()> {
    let tokenizer = AudioTokenizer::load(&cache.tokenizer()).context("load tokenizer (run `staleynet tokenize` first)")?;
    config.model.codebooks = tokenizer.config.codebooks;
    config.model.codebook_size = tokenizer.config.codebook_size;

    ensure!(config.sequence_length <= config.model.context_length, "sequence_length exceeds the model's context_length");
    Ok(())
}

/// Creates the run directory and records the configuration and the tokenizer
/// in it, so that the run can be reproduced and decoded on its own.
fn prepare_run_directory(run: &Path, cache: &CacheLayout, config: &TrainingConfig) -> Result<RunLayout> {
    std::fs::create_dir_all(run)?;
    std::fs::write(run.join("config.json"), serde_json::to_string_pretty(config)?)?;
    std::fs::copy(cache.tokenizer(), run.join(RUN_TOKENIZER)).context("copy the tokenizer into the run directory")?;

    Ok(RunLayout { root: run.to_path_buf() })
}

/// The model, its optimizer, and the lowest validation loss seen so far.
struct StartingState {
    /// Model weights, fresh or restored.
    model: Gpt,

    /// AdamW state, restored only when resuming.
    optimizer: AdamW,

    /// Lowest logged validation loss, or infinity when none is known.
    best_validation: f32,
}

/// Builds a fresh model and optimizer, then applies `start` to them.
fn starting_state(gpu: &Gpu, config: &TrainingConfig, run: &RunLayout, start: &StartPoint) -> Result<StartingState> {
    let mut model = Gpt::new(gpu, config.model.clone(), config.seed)?;
    let optimizer = AdamW::new(gpu, config.optimizer.clone(), &mut model)?;
    let mut state = StartingState { model, optimizer, best_validation: f32::INFINITY };

    match start {
        StartPoint::Fresh => {}
        StartPoint::ResumeLatest => resume_latest(gpu, config, run, &mut state)?,
        StartPoint::FineTune(source) => warm_start(gpu, source, &mut state.model)?,
    }

    Ok(state)
}

/// Restores weights, optimizer moments and the best validation loss from
/// `latest/`. A run without a `latest/` checkpoint starts fresh.
fn resume_latest(gpu: &Gpu, config: &TrainingConfig, run: &RunLayout, state: &mut StartingState) -> Result<()> {
    if !run.latest().join("model.json").exists() {
        return Ok(());
    }

    let (model, metadata) = checkpoint::load(gpu, &run.latest(), Some(&mut state.optimizer))?;
    ensure!(metadata.model == config.model, "the checkpoint's architecture differs from the configuration");

    state.model = model;
    state.best_validation = best_logged_validation(&run.root.join("metrics.jsonl")).unwrap_or(f32::INFINITY);
    Ok(())
}

/// Copies every compatible tensor of the checkpoint at `source` into `model`
/// and reports which tensors were copied, grown, or left fresh.
fn warm_start(gpu: &Gpu, source: &Path, model: &mut Gpt) -> Result<()> {
    let transfer = checkpoint::load_weights(gpu, source, model)?;

    eprintln!(
        "warm start from {}: {} tensors copied, grown {:?}, fresh {:?}",
        source.display(),
        transfer.copied.len(),
        transfer.grown,
        transfer.fresh
    );
    Ok(())
}

/// Throughput bookkeeping between two progress reports.
struct ProgressWindow {
    /// When the window began.
    started: Instant,

    /// Optimizer steps taken inside the window.
    steps: usize,
}

impl ProgressWindow {
    /// An empty window beginning now.
    fn new() -> Self {
        Self { started: Instant::now(), steps: 0 }
    }
}

impl Session<'_> {
    /// The main loop: one optimizer step per iteration, followed by whatever
    /// logging, validation and checkpointing falls due at that step.
    fn run(mut self, report: &mut impl FnMut(&TrainingEvent)) -> Result<()> {
        let started = self.started_event();
        self.emit(report, started)?;

        let sampler = BatchSampler::new(self.model.vocabulary(), SamplerOptions::TRAINING);
        let mut window = ProgressWindow::new();

        for step in self.optimizer.steps_taken..self.config.steps {
            let (loss, update) = self.train_step(step, &sampler)?;
            window.steps += 1;

            self.after_step(step + 1, &loss, &update, &mut window, report)?;
        }

        Ok(())
    }

    /// The event that opens a run's log.
    fn started_event(&mut self) -> TrainingEvent {
        let codebooks = self.config.model.codebooks;

        TrainingEvent::Started {
            parameters: self.model.parameter_count(),
            train_tracks: self.train.tracks.len(),
            train_steps: self.train.total_steps(codebooks),
            validation_tracks: self.validation.tracks.len(),
            first_step: self.optimizer.steps_taken,
            tracks_with_lyrics: self.train.tracks_with_lyrics(),
            best_validation: Some(self.best_validation).filter(|loss| loss.is_finite()),
        }
    }

    /// Samples a batch for `step`, accumulates gradients, and updates the weights.
    /// The batch depends only on the seed and the step, so a resumed run sees
    /// the same data it would have seen without interruption.
    fn train_step(&mut self, step: usize, sampler: &BatchSampler) -> Result<(LossReport, StepReport)> {
        let mut random = Random::derived(self.config.seed, step as u64);
        sampler.fill(&self.train, &mut self.batch, &mut random);

        let loss = self
            .model
            .accumulate_gradients(self.gpu, &mut self.workspace, &self.batch)?;
        ensure!(loss.mean.is_finite(), "loss became {} at step {step}", loss.mean);

        let update = self.optimizer.step(self.gpu, &mut self.model)?;
        Ok((loss, update))
    }

    /// Logs progress, validates and saves `latest/` when each falls due after
    /// `completed` steps.
    fn after_step(
        &mut self,
        completed: usize,
        loss: &LossReport,
        update: &StepReport,
        window: &mut ProgressWindow,
        report: &mut impl FnMut(&TrainingEvent),
    ) -> Result<()> {
        if self.is_due(completed, self.config.log_every) {
            self.gpu.synchronize()?;
            let progress = self.progress(completed, loss, update, window.started.elapsed().as_secs_f64(), window.steps);
            self.emit(report, progress)?;
            *window = ProgressWindow::new();
        }

        if self.is_due(completed, self.config.validate_every) {
            self.validate(completed, report)?;
        }

        if self.is_due(completed, self.config.checkpoint_every) {
            self.save_latest(completed)?;
        }

        Ok(())
    }

    /// True every `interval` steps and at the final step.
    fn is_due(&self, completed: usize, interval: usize) -> bool {
        completed.is_multiple_of(interval) || completed == self.config.steps
    }

    /// Saves weights and optimizer moments to `latest/`, the resumable checkpoint.
    fn save_latest(&mut self, completed: usize) -> Result<()> {
        let config = serde_json::to_value(&self.config)?;

        checkpoint::save(self.gpu, &self.run.latest(), &mut self.model, Some(&self.optimizer), completed, config)
    }

    /// A progress event for the window of `window_steps` steps that took `seconds`.
    fn progress(&self, step: usize, loss: &LossReport, update: &StepReport, seconds: f64, window_steps: usize) -> TrainingEvent {
        let positions = (window_steps * self.batch.shape.rows()) as f64;

        TrainingEvent::Progress {
            step,
            loss: loss.mean,
            loss_per_codebook: loss.per_codebook.clone(),
            gradient_norm: update.gradient_norm,
            learning_rate: update.learning_rate,
            positions_per_second: positions / seconds,
            milliseconds_per_step: seconds * 1e3 / window_steps as f64,
        }
    }

    /// Measures the held-out loss, saves `best/` when it improves, and logs the result.
    fn validate(&mut self, step: usize, report: &mut impl FnMut(&TrainingEvent)) -> Result<()> {
        let (loss, loss_per_codebook) = self.mean_validation_loss()?;

        let improved = loss < self.best_validation;
        if improved {
            self.best_validation = loss;
            checkpoint::save(self.gpu, &self.run.best(), &mut self.model, None, step, serde_json::to_value(&self.config)?)?;
        }

        self.emit(report, TrainingEvent::Validation { step, loss, loss_per_codebook, improved })
    }

    /// Mean loss, overall and per codebook, over a fixed set of validation
    /// batches. The batches are drawn from their own random stream, so every
    /// validation sees the same data. Without a validation split, the training
    /// corpus is used instead.
    fn mean_validation_loss(&mut self) -> Result<(f32, Vec<f32>)> {
        let corpus = if self.validation.tracks.is_empty() { &self.train } else { &self.validation };
        let sampler = BatchSampler::new(self.model.vocabulary(), SamplerOptions::VALIDATION);
        let mut random = Random::derived(self.config.seed, VALIDATION_STREAM);

        let mut total = 0.0;
        let mut per_codebook = vec![0.0; self.config.model.codebooks];
        for _ in 0..self.config.validation_batches {
            sampler.fill(corpus, &mut self.batch, &mut random);
            let loss = self
                .model
                .forward_loss(self.gpu, &mut self.workspace, &self.batch)?;

            total += loss.mean;
            per_codebook
                .iter_mut()
                .zip(&loss.per_codebook)
                .for_each(|(sum, value)| *sum += value);
        }

        let batches = self.config.validation_batches as f32;
        Ok((total / batches, per_codebook.into_iter().map(|sum| sum / batches).collect()))
    }

    /// Logs `event` to `metrics.jsonl` and forwards it to the caller.
    fn emit(&mut self, report: &mut impl FnMut(&TrainingEvent), event: TrainingEvent) -> Result<()> {
        writeln!(self.metrics, "{}", serde_json::to_string(&event)?)?;
        report(&event);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn temp_root(tag: &str) -> PathBuf {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);

        std::env::temp_dir().join(format!("staleynet-train-{tag}-{}-{id}", std::process::id()))
    }

    #[test]
    fn defaults_match_the_recorded_runs() {
        let config = TrainingConfig::default();

        assert_eq!(config.batch_size, 8);
        assert_eq!(config.sequence_length, 768);
        assert_eq!(config.steps, 50_000);
        assert_eq!(config.log_every, 25);
        assert_eq!(config.validate_every, 500);
        assert_eq!(config.validation_batches, 16);
        assert_eq!(config.checkpoint_every, 250);
        assert_eq!(config.seed, 1_234);
        assert!(config.extra_sources.is_empty());
    }

    #[test]
    fn best_logged_validation_ignores_non_validation_events() {
        let root = temp_root("metrics");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("metrics.jsonl");

        std::fs::write(
            &path,
            concat!(
                "{\"event\":\"started\"}\n",
                "not json\n",
                "{\"event\":\"validation\",\"loss\":3.0}\n",
                "{\"event\":\"progress\",\"loss\":0.1}\n",
                "{\"event\":\"validation\",\"loss\":2.5}\n",
                "{\"event\":\"validation\",\"loss\":2.7}\n"
            ),
        )
        .unwrap();

        assert_eq!(best_logged_validation(&path), Some(2.5));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn best_logged_validation_handles_missing_and_empty_logs() {
        let root = temp_root("empty");
        std::fs::create_dir_all(&root).unwrap();

        assert_eq!(best_logged_validation(&root.join("missing.jsonl")), None);

        let empty = root.join("empty.jsonl");
        std::fs::write(&empty, "").unwrap();

        assert_eq!(best_logged_validation(&empty), None);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn preferred_checkpoint_prefers_best_then_latest() {
        let root = temp_root("checkpoint");
        let best = root.join("best");
        let latest = root.join("latest");

        std::fs::create_dir_all(&best).unwrap();
        std::fs::create_dir_all(&latest).unwrap();

        assert_eq!(preferred_checkpoint(&root), latest);

        std::fs::write(best.join("model.json"), "{}").unwrap();

        assert_eq!(preferred_checkpoint(&root), best);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn run_layout_names_best_and_latest() {
        let layout = RunLayout { root: PathBuf::from("runs/base") };

        assert_eq!(layout.best(), PathBuf::from("runs/base/best"));
        assert_eq!(layout.latest(), PathBuf::from("runs/base/latest"));
    }
}
