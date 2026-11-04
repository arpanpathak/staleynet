//! The vocoder's training loop: steps, validation, checkpoints and listening
//! samples. Every step draws its segments from a generator seeded by the step
//! number, so a resumed run sees the batches it would have seen without
//! stopping, and `latest/` keeps the optimizer's moments for the same reason.

use super::config::VocoderTrainingConfig;
use super::progress::{Window, best_validation_so_far};
use super::segments::{Batch, SegmentSampler};
use anyhow::{Context, Result, ensure};
use serde_json::json;
use staleynet_audio::griffin_lim::{GriffinLimConfig, griffin_lim};
use staleynet_audio::wav::write_mono_pcm16;
use staleynet_audio::{MelAnalyzer, MelConfig};
use staleynet_cuda::Gpu;
use staleynet_data::store::CacheLayout;
use staleynet_model::optimizer::AdamW;
use staleynet_model::parameter::Parameters;
use staleynet_model::vocoder::checkpoint::{self, Restored};
use staleynet_model::vocoder::{LossBreakdown, TRAINING_RESOLUTIONS, TrainingPass, Vocoder};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Batches the loader keeps ready ahead of the GPU.
const PREFETCH_DEPTH: usize = 4;

/// Griffin-Lim iterations for the reference renderings.
const GRIFFIN_LIM_ITERATIONS: usize = 64;

/// Offset mixed into the seed for the held-out batches, so that they differ
/// from every training batch.
const VALIDATION_SEED_OFFSET: u64 = 1;

/// The checkpoint to synthesize with: `best` when there is one, else `latest`.
pub fn preferred_checkpoint(run: &Path) -> PathBuf {
    let best = run.join("best");
    if best.is_dir() { best } else { run.join("latest") }
}

/// Trains a vocoder on `cache` into `run`, resuming from `run/latest` when it exists.
pub fn train(gpu: &Gpu, cache: &Path, run: &Path, config: VocoderTrainingConfig) -> Result<()> {
    config.vocoder.validate();
    fs::create_dir_all(run)?;
    fs::write(run.join("config.json"), serde_json::to_string_pretty(&config)?)?;

    let sampler = SegmentSampler::new(&CacheLayout::new(cache), &config.vocoder, config.frames, config.levels)?;
    let counts = sampler.track_counts();
    eprintln!("vocoder: {} training and {} held-out tracks", counts.train, counts.held_out);

    let latest = run.join("latest");
    let resumed = checkpoint::exists(&latest);
    let restored = if resumed { checkpoint::load(gpu, &latest)? } else { fresh_vocoder(gpu, &sampler, &config)? };
    let Restored { mut vocoder, step } = restored;
    ensure!(vocoder.config == config.vocoder, "{} was trained with a different network", latest.display());

    let mut optimizer = AdamW::new(gpu, config.optimizer.clone(), &mut vocoder.network)?;
    if resumed {
        checkpoint::load_moments(gpu, &latest, &mut optimizer)?;
        eprintln!("vocoder: resuming at step {step}");
    }
    optimizer.steps_taken = step;

    let validation_seed = config.seed ^ VALIDATION_SEED_OFFSET;
    let session = Session {
        gpu,
        run,
        pass: TrainingPass::new(gpu, &vocoder, config.sequences, config.frames, &TRAINING_RESOLUTIONS)?,
        validation: sampler.validation_batches(config.sequences, config.validation_batches, validation_seed)?,
        sampler: Arc::new(sampler),
        vocoder,
        optimizer,
        best_validation: best_validation_so_far(run),
        metrics: fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(run.join("metrics.jsonl"))?,
        config: &config,
    };

    session.run(step)
}

/// A newly initialized vocoder at step 0, standardizing its input with the
/// tokenizer's statistics.
fn fresh_vocoder(gpu: &Gpu, sampler: &SegmentSampler, config: &VocoderTrainingConfig) -> Result<Restored> {
    let normalization = sampler.normalization();
    let mean = normalization.mean.clone();
    let std = normalization.std.clone();

    let vocoder = Vocoder::new(gpu, config.vocoder.clone(), mean, std, config.seed)?;
    Ok(Restored { vocoder, step: 0 })
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

impl Session<'_> {
    /// Steps from `first_step` to the end of the run.
    fn run(mut self, first_step: usize) -> Result<()> {
        if first_step == 0 {
            self.render_references()?;
        }

        let sequences = self.config.sequences;
        let batches = Arc::clone(&self.sampler).prefetch(sequences, self.config.seed, first_step, PREFETCH_DEPTH);
        let mut window = Window::default();

        for step in first_step..self.config.steps {
            let batch = batches.recv().context("the batch loader stopped")??;
            let started = Instant::now();

            self.vocoder.network.zero_gradients(self.gpu)?;
            self.pass
                .run(self.gpu, &mut self.vocoder, &batch.mel, &batch.target)?;
            let loss = self.pass.loss(self.gpu)?;
            let report = self.optimizer.step(self.gpu, &mut self.vocoder.network)?;
            window.add(&loss, report.gradient_norm, started.elapsed().as_secs_f64());

            let done = step + 1;
            let last = done == self.config.steps;

            if done.is_multiple_of(self.config.log_every) {
                let means = window.take();
                self.emit(json!({
                    "event": "progress",
                    "step": done,
                    "loss": means.loss,
                    "loss_parts": means.loss_parts,
                    "gradient_norm": means.gradient_norm,
                    "learning_rate": report.learning_rate,
                    "milliseconds_per_step": 1000.0 * means.seconds_per_step,
                }))?;
            }

            if done.is_multiple_of(self.config.validate_every) || last {
                self.validate(done)?;
            }

            if done.is_multiple_of(self.config.checkpoint_every) || last {
                self.save("latest", done)?;
            }

            if done.is_multiple_of(self.config.render_every) || last {
                self.render(done)?;
            }
        }

        Ok(())
    }

    /// Mean loss over the held-out batches; saves `best` when it improves.
    fn validate(&mut self, step: usize) -> Result<()> {
        let mut sum = LossBreakdown { total: 0.0, parts: vec![0.0; TRAINING_RESOLUTIONS.len()] };

        for batch in &self.validation {
            self.pass
                .run(self.gpu, &mut self.vocoder, &batch.mel, &batch.target)?;
            let loss = self.pass.loss(self.gpu)?;

            sum.total += loss.total;
            for (total, part) in sum.parts.iter_mut().zip(loss.parts) {
                *total += part;
            }
        }

        let count = self.validation.len().max(1) as f64;
        let loss = sum.total / count;
        let parts: Vec<f64> = sum.parts.iter().map(|total| total / count).collect();
        self.emit(json!({ "event": "validation", "step": step, "loss": loss, "loss_parts": parts }))?;

        if loss < self.best_validation {
            self.best_validation = loss;
            self.save("best", step)?;
        }

        Ok(())
    }

    /// Writes the checkpoint `name`, with the optimizer's moments, under a
    /// temporary name first, so that a reader never sees half of it.
    fn save(&mut self, name: &str, step: usize) -> Result<()> {
        let staging = self.run.join(format!("{name}.saving"));
        let destination = self.run.join(name);
        let _ = fs::remove_dir_all(&staging);

        checkpoint::save(self.gpu, &staging, &mut self.vocoder, Some(&self.optimizer), step)?;

        let _ = fs::remove_dir_all(&destination);
        fs::rename(&staging, &destination)?;
        Ok(())
    }

    /// The listening tracks as recorded and through Griffin-Lim, written once.
    fn render_references(&mut self) -> Result<()> {
        let directory = self.run.join("samples").join("reference");
        fs::create_dir_all(&directory)?;

        let vocoder = &self.config.vocoder;
        let analyzer = MelAnalyzer::new(MelConfig {
            sample_rate: vocoder.sample_rate,
            fft_size: vocoder.fft_size,
            hop_size: vocoder.hop_size,
            mel_bands: vocoder.mel_bands,
            min_frequency: 0.0,
            max_frequency: vocoder.sample_rate as f32 / 2.0,
        });
        let griffin_lim_config = GriffinLimConfig { iterations: GRIFFIN_LIM_ITERATIONS, seed: 0 };

        for id in self.sampler.listening_tracks(self.config.render_tracks) {
            let track = self.sampler.whole_track(&id, self.config.render_frames())?;
            let rendered = griffin_lim(&analyzer, &track.log_mel, griffin_lim_config);
            let stem = file_name(&id);

            write_mono_pcm16(&directory.join(format!("{stem}-recording.wav")), &track.recording, vocoder.sample_rate)?;
            write_mono_pcm16(&directory.join(format!("{stem}-griffin-lim.wav")), &rendered, vocoder.sample_rate)?;
        }

        Ok(())
    }

    /// The listening tracks through the vocoder at `step`.
    fn render(&mut self, step: usize) -> Result<()> {
        let directory = self.run.join("samples").join(format!("step-{step:07}"));
        fs::create_dir_all(&directory)?;

        let mut written = Vec::new();
        for id in self.sampler.listening_tracks(self.config.render_tracks) {
            let track = self.sampler.whole_track(&id, self.config.render_frames())?;
            let waveform = self.vocoder.synthesize(self.gpu, &track.log_mel)?;
            let path = directory.join(format!("{}-vocoder.wav", file_name(&id)));

            write_mono_pcm16(&path, &waveform, self.config.vocoder.sample_rate)?;
            written.push(path);
        }

        self.emit(json!({ "event": "samples", "step": step, "files": written }))
    }

    /// Appends one line to `metrics.jsonl` and prints it.
    fn emit(&mut self, line: serde_json::Value) -> Result<()> {
        let text = line.to_string();

        writeln!(self.metrics, "{text}")?;
        eprintln!("{text}");
        Ok(())
    }
}

/// A track id usable as a file name.
fn file_name(id: &str) -> String {
    id.replace(['/', '\\'], "_")
}
