//! `staleynet vocoder`: build the waveform cache, train the neural vocoder,
//! and render recordings through it.
//!
//! ```text
//! staleynet vocoder prepare --cache data/cache-fma-rock
//! staleynet vocoder train   --cache data/cache-fma-rock --run runs/vocoder --config configs/vocoder.json
//! staleynet vocoder render  --cache data/cache-fma-rock --run runs/vocoder --out samples/vocoder
//! ```

use super::open_gpu;
use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use staleynet_audio::wav::write_mono_pcm16;
use staleynet_data::store::CacheLayout;
use staleynet_model::vocoder::checkpoint;
use staleynet_train::vocoder::{SegmentSampler, VocoderTrainingConfig, WaveformOutcome, preferred_checkpoint, prepare_waveforms, train};
use std::path::{Path, PathBuf};

/// Tracks between progress lines while building the waveform cache.
const PREPARE_REPORT_EVERY: usize = 500;

/// Arguments of `staleynet vocoder`.
#[derive(Args)]
pub struct Arguments {
    /// What to do.
    #[command(subcommand)]
    task: Task,
}

/// The vocoder's tasks.
#[derive(Subcommand)]
enum Task {
    /// Write every track's samples into the cache, as the vocoder's targets.
    Prepare {
        /// Tokenized dataset cache.
        #[arg(long, default_value = "data/cache")]
        cache: PathBuf,
    },

    /// Train the vocoder, resuming from the run's latest checkpoint.
    Train {
        /// Tokenized dataset cache with waveforms.
        #[arg(long, default_value = "data/cache")]
        cache: PathBuf,

        /// Run directory.
        #[arg(long)]
        run: PathBuf,

        /// Training configuration (JSON); defaults where absent.
        #[arg(long)]
        config: Option<PathBuf>,
    },

    /// Render held-out tracks, decoded from their tokens, through a trained vocoder.
    Render {
        /// Tokenized dataset cache with waveforms.
        #[arg(long, default_value = "data/cache")]
        cache: PathBuf,

        /// Vocoder run (its best checkpoint, else its latest).
        #[arg(long)]
        run: PathBuf,

        /// Output directory.
        #[arg(long, default_value = "samples/vocoder")]
        out: PathBuf,

        /// Tracks to render.
        #[arg(long, default_value_t = 4)]
        tracks: usize,

        /// Seconds per track.
        #[arg(long, default_value_t = 12.0)]
        seconds: f32,
    },
}

/// Runs the chosen task.
pub fn run(arguments: Arguments) -> Result<()> {
    match arguments.task {
        Task::Prepare { cache } => prepare(cache),
        Task::Train { cache, run, config } => {
            let config = match config {
                Some(path) => read_config(&path)?,
                None => VocoderTrainingConfig::default(),
            };

            train(&open_gpu()?, &cache, &run, config)
        }
        Task::Render { cache, run, out, tracks, seconds } => render(cache, run, out, tracks, seconds),
    }
}

/// Reads a training configuration; fields it leaves out take their defaults.
fn read_config(path: &Path) -> Result<VocoderTrainingConfig> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;

    Ok(serde_json::from_str(&text)?)
}

/// Builds the waveform cache, printing progress and failures.
fn prepare(cache: PathBuf) -> Result<()> {
    let mut done = 0usize;
    let mut written = 0usize;
    let mut failed = 0usize;
    let mut seconds = 0.0f64;

    prepare_waveforms(&CacheLayout::new(&cache), |outcome| {
        done += 1;
        match outcome {
            WaveformOutcome::Written { seconds: length, .. } => {
                written += 1;
                seconds += *length as f64;
            }
            WaveformOutcome::Present(_) => {}
            WaveformOutcome::Failed { id, reason } => {
                failed += 1;
                eprintln!("skipped {id}: {reason}");
            }
        }

        if done.is_multiple_of(PREPARE_REPORT_EVERY) {
            eprintln!("{done} tracks: {written} written ({:.1} h), {failed} failed", seconds / 3600.0);
        }
    })?;

    eprintln!("done: {done} tracks, {written} written ({:.1} h), {failed} failed", seconds / 3600.0);
    Ok(())
}

/// Writes each held-out track's recording and its vocoder rendering.
fn render(cache: PathBuf, run: PathBuf, out: PathBuf, tracks: usize, seconds: f32) -> Result<()> {
    let gpu = open_gpu()?;
    let directory = preferred_checkpoint(&run);
    let restored = checkpoint::load(&gpu, &directory)?;
    let config = &restored.vocoder.config;

    let sampler = SegmentSampler::new(&CacheLayout::new(&cache), config, 1, None)?;
    let frames_per_second = config.sample_rate as f32 / config.hop_size as f32;
    let frames = (seconds * frames_per_second) as usize + 1;
    std::fs::create_dir_all(&out)?;

    for id in sampler.listening_tracks(tracks) {
        let track = sampler.whole_track(&id, frames)?;
        let waveform = restored.vocoder.synthesize(&gpu, &track.log_mel)?;
        let stem = id.replace(['/', '\\'], "_");

        write_mono_pcm16(&out.join(format!("{stem}-recording.wav")), &track.recording, config.sample_rate)?;
        write_mono_pcm16(&out.join(format!("{stem}-vocoder.wav")), &waveform, config.sample_rate)?;
    }

    eprintln!("rendered {tracks} tracks with {} (step {}) into {}", directory.display(), restored.step, out.display());
    Ok(())
}
