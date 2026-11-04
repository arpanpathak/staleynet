//! One training step's forward and backward pass over a batch of segments.

use super::loss::{Resolution, SignalBatch, SpectralLoss};
use super::model::Vocoder;
use super::network::Activations;
use super::synthesis::Synthesis;
use staleynet_cuda::{Gpu, GpuResult, Shape, Tensor};

/// The loss of one pass: its total and its part at each resolution.
#[derive(Clone, Debug, PartialEq)]
pub struct LossBreakdown {
    /// Sum of the weighted parts.
    pub total: f64,

    /// The weighted loss at each resolution, in the order they were given.
    pub parts: Vec<f64>,
}

/// Everything one training step needs for a batch of `sequences` segments of
/// `time` frames: the network's activations, the synthesis buffers, and one
/// spectral loss per resolution.
pub struct TrainingPass {
    /// Network activations and gradient scratch.
    activations: Activations,

    /// Spectrum, signal and their gradients.
    synthesis: Synthesis,

    /// The spectral losses.
    losses: Vec<SpectralLoss>,

    /// Standardized input mels, `[rows, mel_bands]`.
    mel: Tensor,

    /// Target signals, `[sequences, padded samples]`.
    target: Tensor,
}

impl TrainingPass {
    /// Buffers for `sequences` segments of `time` frames, compared at `resolutions`.
    pub fn new(gpu: &Gpu, vocoder: &Vocoder, sequences: usize, time: usize, resolutions: &[Resolution]) -> GpuResult<Self> {
        let config = &vocoder.config;
        let length = config.padded_samples(time);
        let batch = SignalBatch { sequences, length, sample_rate: config.sample_rate, mel_bands: config.mel_bands };

        let losses = resolutions
            .iter()
            .map(|resolution| SpectralLoss::new(gpu, *resolution, batch))
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self {
            activations: Activations::new(gpu, config, sequences, time)?,
            synthesis: Synthesis::new(gpu, config, &vocoder.istft, sequences, time)?,
            losses,
            mel: gpu.zeros(Shape::new(sequences * time, config.mel_bands))?,
            target: gpu.zeros(Shape::new(sequences, length))?,
        })
    }

    /// Samples per target signal: `(time − 1) · hop + fft_size`.
    pub fn signal_length(&self) -> usize {
        self.target.shape().cols
    }

    /// One forward and backward pass: synthesizes the batch from `mel`
    /// (standardized, `[sequences · time, mel_bands]`), compares it with
    /// `target` (`[sequences, signal length]`), and adds every parameter's
    /// gradient into the network. Gradients are not zeroed here.
    pub fn run(&mut self, gpu: &Gpu, vocoder: &mut Vocoder, mel: &[f32], target: &[f32]) -> GpuResult {
        gpu.copy_from_host(&mut self.mel, mel)?;
        gpu.copy_from_host(&mut self.target, target)?;

        vocoder
            .network
            .forward(gpu, &self.mel, &mut self.activations)?;
        self.synthesis
            .forward(gpu, &vocoder.istft, &self.activations.head)?;

        gpu.fill_zero(&mut self.synthesis.signal_grad)?;
        for loss in &mut self.losses {
            loss.set_target(gpu, &self.target)?;
            loss.forward(gpu, &self.synthesis.signal)?;
            loss.backward(gpu, &mut self.synthesis.signal_grad)?;
        }

        self.synthesis
            .backward(gpu, &vocoder.istft, &self.activations.head)?;
        vocoder
            .network
            .backward(gpu, &self.synthesis.head_grad, &mut self.activations)
    }

    /// The loss of the last [`TrainingPass::run`].
    pub fn loss(&self, gpu: &Gpu) -> GpuResult<LossBreakdown> {
        let parts = self
            .losses
            .iter()
            .map(|loss| loss.value(gpu))
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(LossBreakdown { total: parts.iter().sum(), parts })
    }
}
