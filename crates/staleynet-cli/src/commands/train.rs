//! `staleynet train`: run (or resume) a training run.

use super::open_gpu;
use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use staleynet_cuda::MatmulKernel;
use staleynet_data::store::CacheLayout;
use staleynet_train::{StartPoint, TrainingConfig, TrainingEvent, train};
use std::path::PathBuf;

/// The matrix-product kernel a run trains with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum MatmulArgument {
    /// f32 on the CUDA cores, 64×64 tiles with 4×4 register blocks.
    Blocked,

    /// TF32 on the tensor cores: inputs rounded to a 10-bit mantissa, sums in f32.
    TensorCore,
}

impl From<MatmulArgument> for MatmulKernel {
    fn from(argument: MatmulArgument) -> Self {
        match argument {
            MatmulArgument::Blocked => MatmulKernel::Blocked,
            MatmulArgument::TensorCore => MatmulKernel::TensorCore,
        }
    }
}

/// Arguments of `staleynet train`.
#[derive(Args)]
pub struct Arguments {
    /// Tokenised dataset cache.
    #[arg(long, default_value = "data/cache")]
    cache: PathBuf,

    /// Run directory (created if missing).
    #[arg(long)]
    run: PathBuf,

    /// JSON TrainingConfig; defaults are used when omitted.
    #[arg(long)]
    config: Option<PathBuf>,

    /// Override the total number of steps.
    #[arg(long)]
    steps: Option<usize>,

    /// Continue from `<run>/latest`.
    #[arg(long, conflicts_with = "fine_tune_from")]
    resume: bool,

    /// Start from the weights of this checkpoint directory (e.g. runs/base/best),
    /// with a fresh optimiser. The configured model may add a lyrics field.
    #[arg(long)]
    fine_tune_from: Option<PathBuf>,

    /// Matrix-product kernel. A run may resume with a different one.
    #[arg(long, value_enum, default_value_t = MatmulArgument::Blocked)]
    matmul: MatmulArgument,
}

/// Prints one training event as a single human-readable line.
fn print_event(event: &TrainingEvent) {
    match event {
        TrainingEvent::Started {
            parameters,
            train_tracks,
            train_steps,
            validation_tracks,
            first_step,
            tracks_with_lyrics,
            best_validation,
        } => eprintln!(
            "model {:.2} M parameters | {train_tracks} train tracks ({train_steps} steps, {tracks_with_lyrics} with lyrics) | {validation_tracks} validation tracks | from step {first_step}, best so far {best_validation:?}",
            *parameters as f64 / 1e6
        ),
        TrainingEvent::Progress {
            step,
            loss,
            loss_per_codebook,
            gradient_norm,
            learning_rate,
            positions_per_second,
            milliseconds_per_step,
        } => eprintln!(
            "step {step:6}  loss {loss:.3} {loss_per_codebook:.2?}  |g| {gradient_norm:.2}  lr {learning_rate:.1e}  {positions_per_second:.0} pos/s  {milliseconds_per_step:.0} ms/step"
        ),
        TrainingEvent::Validation { step, loss, loss_per_codebook, improved } => {
            let marker = if *improved { "  ← best" } else { "" };
            eprintln!("step {step:6}  validation {loss:.3} {loss_per_codebook:.2?}{marker}");
        }
    }
}

pub fn run(arguments: Arguments) -> Result<()> {
    let mut config: TrainingConfig = match &arguments.config {
        Some(path) => serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?)?,
        None => TrainingConfig::default(),
    };
    if let Some(steps) = arguments.steps {
        config.steps = steps;
    }

    let start = match (arguments.resume, arguments.fine_tune_from) {
        (true, _) => StartPoint::ResumeLatest,
        (false, Some(source)) => StartPoint::FineTune(source),
        (false, None) => StartPoint::Fresh,
    };
    let mut gpu = open_gpu()?;
    gpu.matmul_kernel = arguments.matmul.into();

    train(&gpu, &CacheLayout::new(&arguments.cache), &arguments.run, config, start, print_event)
}
