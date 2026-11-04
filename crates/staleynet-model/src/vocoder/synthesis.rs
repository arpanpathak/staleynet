//! From the network's head to samples, and back.
//!
//! The head holds a log-magnitude and a phase for every frequency of every
//! frame. [`Synthesis::forward`] turns them into a complex spectrum and the
//! inverse STFT turns that into samples; [`Synthesis::backward`] carries the
//! gradient of the samples back to the head.

use super::config::VocoderConfig;
use super::istft::InverseStft;
use staleynet_cuda::{Gpu, GpuResult, Shape, Tensor};

/// The predicted spectrum, the synthesized signal and their gradients, for
/// `sequences` sequences of `time` frames.
pub struct Synthesis {
    /// Frames per sequence.
    time: usize,

    /// Cap on log-magnitudes, from the configuration.
    max_log_magnitude: f32,

    /// Real part of the predicted spectrum, `[rows, bins]`.
    real: Tensor,

    /// Imaginary part of the predicted spectrum, `[rows, bins]`.
    imaginary: Tensor,

    /// Synthesized frames before overlap-add, `[rows, fft_size]`.
    frames: Tensor,

    /// One over the summed squared window at each sample of a sequence.
    normalization: Tensor,

    /// The synthesized signals, `[sequences, padded samples]`.
    pub signal: Tensor,

    /// Gradient of `signal`, filled by the losses.
    pub signal_grad: Tensor,

    /// Gradient of `frames`.
    frames_grad: Tensor,

    /// Gradient of `real`.
    real_grad: Tensor,

    /// Gradient of `imaginary`.
    imaginary_grad: Tensor,

    /// Gradient of the head output, `[rows, 2 · bins]`.
    pub head_grad: Tensor,
}

impl Synthesis {
    /// Buffers for `sequences` sequences of `time` frames.
    pub fn new(gpu: &Gpu, config: &VocoderConfig, istft: &InverseStft, sequences: usize, time: usize) -> GpuResult<Self> {
        let rows = sequences * time;
        let spectrum = Shape::new(rows, config.bins());
        let frames = Shape::new(rows, config.fft_size);
        let length = config.padded_samples(time);

        Ok(Self {
            time,
            max_log_magnitude: config.max_log_magnitude,
            real: gpu.zeros(spectrum)?,
            imaginary: gpu.zeros(spectrum)?,
            frames: gpu.zeros(frames)?,
            normalization: gpu.upload(Shape::vector(length), &istft.normalization(time))?,
            signal: gpu.zeros(Shape::new(sequences, length))?,
            signal_grad: gpu.zeros(Shape::new(sequences, length))?,
            frames_grad: gpu.zeros(frames)?,
            real_grad: gpu.zeros(spectrum)?,
            imaginary_grad: gpu.zeros(spectrum)?,
            head_grad: gpu.zeros(Shape::new(rows, 2 * config.bins()))?,
        })
    }

    /// The spectrum and the samples of the head output `head`.
    pub fn forward(&mut self, gpu: &Gpu, istft: &InverseStft, head: &Tensor) -> GpuResult {
        gpu.polar_spectrum(head, &mut self.real, &mut self.imaginary, self.max_log_magnitude)?;

        istft.forward(gpu, &self.real, &self.imaginary, &self.normalization, self.time, &mut self.frames, &mut self.signal)
    }

    /// The gradient of the head output, from `signal_grad`, into `head_grad`.
    pub fn backward(&mut self, gpu: &Gpu, istft: &InverseStft, head: &Tensor) -> GpuResult {
        istft.backward(
            gpu,
            &self.signal_grad,
            &self.normalization,
            self.time,
            &mut self.frames_grad,
            &mut self.real_grad,
            &mut self.imaginary_grad,
        )?;

        gpu.polar_spectrum_backward(
            head,
            &self.real,
            &self.imaginary,
            &self.real_grad,
            &self.imaginary_grad,
            &mut self.head_grad,
            self.max_log_magnitude,
        )
    }
}
