//! Training the vocoder on a tokenized cache with waveforms.
//!
//! ```text
//! <run>/
//!   config.json        the training configuration
//!   metrics.jsonl      one JSON object per progress, validation or sample event
//!   latest/            the last checkpoint, with the optimizer's moments
//!   best/              the checkpoint with the lowest validation loss
//!   samples/reference/ recordings and their Griffin-Lim rendering, for comparison
//!   samples/step-N/    the vocoder's rendering of the same tracks at step N
//! ```
//!
//! Every step draws its segments from a generator seeded by the step number,
//! so a resumed run sees the batches it would have seen without stopping.

use crate::config::VocoderConfig;
use crate::data::{Batch, SegmentSampler};
use crate::loss::TRAINING_RESOLUTIONS;
use crate::vocoder::{TrainingPass, Vocoder};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::json;
use staleynet_audio::griffin_lim::{GriffinLimConfig, griffin_lim};
use staleynet_audio::wav::write_mono_pcm16;
use staleynet_audio::{MelAnalyzer, MelConfig};
use staleynet_cuda::Gpu;
use staleynet_data::store::CacheLayout;
use staleynet_model::optimizer::{AdamW, OptimizerConfig};
use staleynet_model::parameter::Parameters;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Name of the file that holds the optimizer's moments inside a checkpoint.
const OPTIMIZER_FILE: &str = "optimizer.bin";

/// Bytes per stored value.
const BYTES_PER_VALUE: usize = std::mem::size_of::<f32>();

/// Batches the loader keeps ready ahead of the GPU.
const PREFETCH_DEPTH: usize = 4;

/// Griffin-Lim iterations for the reference renderings.
const GRIFFIN_LIM_ITERATIONS: usize = 64;

/// Everything a vocoder training run is set by.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VocoderTrainingConfig {
    /// The network.
    pub vocoder: VocoderConfig,

    /// AdamW and its schedule.
    pub optimizer: OptimizerConfig,

    /// Segments per batch.
    pub sequences: usize,

    /// Frames per segment.
    pub frames: usize,

    /// Optimizer steps in the whole run.
    pub steps: usize,

    /// Residual levels the input spectrograms are decoded with (all when absent).
    pub levels: Option<usize>,

    /// Steps between progress lines.
    pub log_every: usize,

    /// Steps between validations.
    pub validate_every: usize,

    /// Held-out batches per validation.
    pub validation_batches: usize,

    /// Steps between checkpoints.
    pub checkpoint_every: usize,

    /// Steps between renderings of the listening tracks.
    pub render_every: usize,

    /// Held-out tracks rendered for listening.
    pub render_tracks: usize,

    /// Seconds rendered from each listening track.
    pub render_seconds: f32,

    /// Seed of the initialization and of the batches.
    pub seed: u64,
}

impl Default for VocoderTrainingConfig {
    /// Sizes from Vocos (Siuzdak, 2023): 16 segments of about 0.7 s, AdamW
    /// with β = (0.8, 0.9) and a peak rate of 5·10⁻⁴, decayed by a cosine.
    fn default() -> Self {
        let steps = 200_000;

        Self {
            vocoder: VocoderConfig::default(),
            optimizer: OptimizerConfig {
                peak_learning_rate: 5e-4,
                final_learning_rate: 5e-5,
                warmup_steps: 1_000,
                decay_steps: steps,
                beta1: 0.8,
                beta2: 0.9,
                epsilon: 1e-8,
                weight_decay: 0.01,
                max_gradient_norm: 1.0,
            },
            sequences: 16,
            frames: 64,
            steps,
            levels: None,
            log_every: 50,
            validate_every: 1_000,
            validation_batches: 8,
            checkpoint_every: 1_000,
            render_every: 5_000,
            render_tracks: 3,
            render_seconds: 10.0,
            seed: 0,
        }
    }
}

/// The checkpoint to synthesize with: `best` when there is one, else `latest`.
pub fn preferred_checkpoint(run: &Path) -> PathBuf {
    let best = run.join("best");
    if best.is_dir() { best } else { run.join("latest") }
}

/// A training run in progress.
struct Session<'a> {
    /// The device.
    gpu: &'a Gpu,

    /// The run directory.
    run: &'a Path,

    /// Its configuration.
    config: &'a VocoderTrainingConfig,

    /// Segments and listening tracks.
    sampler: Arc<SegmentSampler>,

    /// The network being trained.
    vocoder: Vocoder,

    /// AdamW state.
    optimizer: AdamW,

    /// Buffers of one step.
    pass: TrainingPass,

    /// Fixed held-out batches.
    validation: Vec<Batch>,

    /// Lowest validation loss so far.
    best_validation: f64,

    /// `metrics.jsonl`, opened for appending.
    metrics: fs::File,
}

/// Trains a vocoder on `cache` into `run`, resuming from `run/latest` when it exists.
pub fn train(gpu: &Gpu, cache: &Path, run: &Path, config: VocoderTrainingConfig) -> Result<()> {
    config.vocoder.validate();
    fs::create_dir_all(run)?;
    fs::write(run.join("config.json"), serde_json::to_string_pretty(&config)?)?;

    let layout = CacheLayout::new(cache);
    let sampler = Arc::new(SegmentSampler::new(&layout, &config.vocoder, config.frames, config.levels)?);
    let (train_tracks, held_out_tracks) = sampler.track_counts();
    eprintln!("vocoder: {train_tracks} training and {held_out_tracks} held-out tracks");

    let (mean, std) = sampler.normalization();
    let latest = run.join("latest");
    let (mut vocoder, step) = if latest.join("vocoder.json").is_file() {
        Vocoder::load(gpu, &latest)?
    } else {
        (Vocoder::new(gpu, config.vocoder.clone(), mean.to_vec(), std.to_vec(), config.seed)?, 0)
    };
    ensure!(vocoder.config == config.vocoder, "{} was trained with a different network", latest.display());

    let mut optimizer = AdamW::new(gpu, config.optimizer.clone(), &mut vocoder.network)?;
    if step > 0 {
        load_moments(gpu, &latest, &mut optimizer)?;
        eprintln!("vocoder: resuming at step {step}");
    }
    optimizer.steps_taken = step;

    let session = Session {
        gpu,
        run,
        pass: TrainingPass::new(gpu, &vocoder, config.sequences, config.frames, &TRAINING_RESOLUTIONS)?,
        validation: sampler.validation_batches(config.sequences, config.validation_batches, config.seed ^ 1)?,
        sampler,
        vocoder,
        optimizer,
        best_validation: best_validation_so_far(run),
        metrics: fs::OpenOptions::new().create(true).append(true).open(run.join("metrics.jsonl"))?,
        config: &config,
    };
    session.run(step)
}

impl Session<'_> {
    /// Steps from `first_step` to the end of the run.
    fn run(mut self, first_step: usize) -> Result<()> {
        if first_step == 0 {
            self.render_references()?;
        }

        let batches = Arc::clone(&self.sampler).prefetch(self.config.sequences, self.config.seed, first_step, PREFETCH_DEPTH);
        let mut window = Window::default();

        for step in first_step..self.config.steps {
            let batch = batches.recv().context("the batch loader stopped")??;
            let started = Instant::now();

            self.vocoder.network.zero_gradients(self.gpu)?;
            self.pass.run(self.gpu, &mut self.vocoder, &batch.mel, &batch.target)?;
            let (loss, parts) = self.pass.loss(self.gpu)?;
            let report = self.optimizer.step(self.gpu, &mut self.vocoder.network)?;
            window.add(loss, &parts, report.gradient_norm, started.elapsed().as_secs_f64());

            let done = step + 1;
            if done % self.config.log_every == 0 {
                let (loss, parts, gradient_norm, seconds) = window.take();
                let line = json!({
                    "event": "progress", "step": done, "loss": loss, "loss_parts": parts,
                    "gradient_norm": gradient_norm, "learning_rate": report.learning_rate,
                    "milliseconds_per_step": 1000.0 * seconds,
                });
                self.emit(line)?;
            }
            if done % self.config.validate_every == 0 || done == self.config.steps {
                self.validate(done)?;
            }
            if done % self.config.checkpoint_every == 0 || done == self.config.steps {
                self.save("latest", done)?;
            }
            if done % self.config.render_every == 0 || done == self.config.steps {
                self.render(done)?;
            }
        }
        Ok(())
    }

    /// Mean loss over the held-out batches; saves `best` when it improves.
    fn validate(&mut self, step: usize) -> Result<()> {
        let mut total = 0.0;
        let mut parts_total = vec![0.0; TRAINING_RESOLUTIONS.len()];

        for batch in &self.validation {
            self.pass.run(self.gpu, &mut self.vocoder, &batch.mel, &batch.target)?;
            let (loss, parts) = self.pass.loss(self.gpu)?;
            total += loss;
            parts_total.iter_mut().zip(parts).for_each(|(sum, part)| *sum += part);
        }

        let count = self.validation.len().max(1) as f64;
        let loss = total / count;
        let parts: Vec<f64> = parts_total.iter().map(|sum| sum / count).collect();
        self.emit(json!({ "event": "validation", "step": step, "loss": loss, "loss_parts": parts }))?;

        if loss < self.best_validation {
            self.best_validation = loss;
            self.save("best", step)?;
        }
        Ok(())
    }

    /// Writes the checkpoint `name` (with the optimizer's moments) atomically.
    fn save(&mut self, name: &str, step: usize) -> Result<()> {
        let staging = self.run.join(format!("{name}.saving"));
        let destination = self.run.join(name);
        let _ = fs::remove_dir_all(&staging);

        self.vocoder.save(self.gpu, &staging, step)?;
        save_moments(self.gpu, &staging, &self.optimizer)?;

        let _ = fs::remove_dir_all(&destination);
        fs::rename(&staging, &destination)?;
        Ok(())
    }

    /// The listening tracks as recorded and through Griffin-Lim, written once.
    fn render_references(&mut self) -> Result<()> {
        let directory = self.run.join("samples").join("reference");
        fs::create_dir_all(&directory)?;

        let config = &self.config.vocoder;
        let analyzer = MelAnalyzer::new(MelConfig {
            sample_rate: config.sample_rate,
            fft_size: config.fft_size,
            hop_size: config.hop_size,
            mel_bands: config.mel_bands,
            min_frequency: 0.0,
            max_frequency: config.sample_rate as f32 / 2.0,
        });

        for id in self.sampler.listening_tracks(self.config.render_tracks) {
            let (log_mel, recording) = self.sampler.whole_track(&id, self.render_frames())?;
            let rendered = griffin_lim(&analyzer, &log_mel, GriffinLimConfig { iterations: GRIFFIN_LIM_ITERATIONS, seed: 0 });

            write_mono_pcm16(&directory.join(format!("{}-recording.wav", file_name(&id))), &recording, config.sample_rate)?;
            write_mono_pcm16(&directory.join(format!("{}-griffin-lim.wav", file_name(&id))), &rendered, config.sample_rate)?;
        }
        Ok(())
    }

    /// The listening tracks through the vocoder at `step`.
    fn render(&mut self, step: usize) -> Result<()> {
        let directory = self.run.join("samples").join(format!("step-{step:07}"));
        fs::create_dir_all(&directory)?;

        let mut written = Vec::new();
        for id in self.sampler.listening_tracks(self.config.render_tracks) {
            let (log_mel, _) = self.sampler.whole_track(&id, self.render_frames())?;
            let waveform = self.vocoder.synthesize(self.gpu, &log_mel)?;
            let path = directory.join(format!("{}-vocoder.wav", file_name(&id)));

            write_mono_pcm16(&path, &waveform, self.config.vocoder.sample_rate)?;
            written.push(path);
        }

        self.emit(json!({ "event": "samples", "step": step, "files": written }))
    }

    /// Frames in `render_seconds` of audio.
    fn render_frames(&self) -> usize {
        let config = &self.config.vocoder;
        (self.config.render_seconds * config.sample_rate as f32 / config.hop_size as f32) as usize + 1
    }

    /// Appends one line to `metrics.jsonl` and prints it.
    fn emit(&mut self, line: serde_json::Value) -> Result<()> {
        let text = line.to_string();
        writeln!(self.metrics, "{text}")?;
        eprintln!("{text}");
        Ok(())
    }
}

/// Running sums over the steps between two progress lines.
#[derive(Default)]
struct Window {
    /// Steps added.
    steps: usize,

    /// Sum of total losses.
    loss: f64,

    /// Sum of each part of the loss.
    parts: Vec<f64>,

    /// Sum of gradient norms.
    gradient_norm: f64,

    /// Sum of step durations, in seconds.
    seconds: f64,
}

impl Window {
    /// Adds one step.
    fn add(&mut self, loss: f64, parts: &[f64], gradient_norm: f32, seconds: f64) {
        self.parts.resize(parts.len(), 0.0);
        self.parts.iter_mut().zip(parts).for_each(|(sum, part)| *sum += part);

        self.steps += 1;
        self.loss += loss;
        self.gradient_norm += gradient_norm as f64;
        self.seconds += seconds;
    }

    /// Means of the loss, its parts, the gradient norm and the step time; then resets.
    fn take(&mut self) -> (f64, Vec<f64>, f64, f64) {
        let count = self.steps.max(1) as f64;
        let parts = self.parts.iter().map(|sum| sum / count).collect();
        let means = (self.loss / count, parts, self.gradient_norm / count, self.seconds / count);

        *self = Self::default();
        means
    }
}

/// The lowest validation loss recorded in `run/metrics.jsonl`, so that a
/// resumed run does not replace `best` with something worse.
fn best_validation_so_far(run: &Path) -> f64 {
    let Ok(text) = fs::read_to_string(run.join("metrics.jsonl")) else { return f64::INFINITY };

    text.lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|value| value["event"] == "validation")
        .filter_map(|value| value["loss"].as_f64())
        .fold(f64::INFINITY, f64::min)
}

/// A track id usable as a file name.
fn file_name(id: &str) -> String {
    id.replace(['/', '\\'], "_")
}

/// Writes every first moment, then every second moment, as little-endian f32.
fn save_moments(gpu: &Gpu, directory: &Path, optimizer: &AdamW) -> Result<()> {
    let mut file = std::io::BufWriter::new(fs::File::create(directory.join(OPTIMIZER_FILE))?);

    for moment in &optimizer.moments {
        for tensor in [&moment.first, &moment.second] {
            gpu.download(tensor)?.iter().try_for_each(|value| file.write_all(&value.to_le_bytes()))?;
        }
    }
    file.flush()?;
    Ok(())
}

/// Reads the moments written by [`save_moments`] into `optimizer`.
fn load_moments(gpu: &Gpu, directory: &Path, optimizer: &mut AdamW) -> Result<()> {
    let path = directory.join(OPTIMIZER_FILE);
    let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let values: Vec<f32> =
        bytes.chunks_exact(BYTES_PER_VALUE).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();

    let mut remaining = values.as_slice();
    for moment in &mut optimizer.moments {
        for tensor in [&mut moment.first, &mut moment.second] {
            ensure!(remaining.len() >= tensor.len(), "{} ends early", path.display());
            let (values, rest) = remaining.split_at(tensor.len());
            gpu.copy_from_host(tensor, values)?;
            remaining = rest;
        }
    }

    ensure!(remaining.is_empty(), "{} has more values than the optimizer", path.display());
    Ok(())
}
