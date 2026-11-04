//! The settings of a vocoder training run.

use serde::{Deserialize, Serialize};
use staleynet_model::optimizer::OptimizerConfig;
use staleynet_model::vocoder::VocoderConfig;

/// Optimizer steps of a default run.
const DEFAULT_STEPS: usize = 200_000;

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
    /// Sizes from Vocos (Siuzdak, 2023, <https://arxiv.org/abs/2306.00814>):
    /// 16 segments of about 0.7 s, AdamW with β = (0.8, 0.9) and a peak rate
    /// of 5·10⁻⁴, decayed by a cosine.
    fn default() -> Self {
        Self {
            vocoder: VocoderConfig::default(),
            optimizer: OptimizerConfig {
                peak_learning_rate: 5e-4,
                final_learning_rate: 5e-5,
                warmup_steps: 1_000,
                decay_steps: DEFAULT_STEPS,
                beta1: 0.8,
                beta2: 0.9,
                epsilon: 1e-8,
                weight_decay: 0.01,
                max_gradient_norm: 1.0,
            },
            sequences: 16,
            frames: 64,
            steps: DEFAULT_STEPS,
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

impl VocoderTrainingConfig {
    /// Frames in `render_seconds` of audio.
    pub fn render_frames(&self) -> usize {
        let frames_per_second = self.vocoder.sample_rate as f32 / self.vocoder.hop_size as f32;

        (self.render_seconds * frames_per_second) as usize + 1
    }
}
