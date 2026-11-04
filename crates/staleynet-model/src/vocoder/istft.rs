//! The inverse short-time Fourier transform, computed as matrix products.
//!
//! Each frame's complex spectrum `X` becomes samples through the inverse
//! Fourier transform of a real signal,
//! `y[n] = (1/N) Σ_k c_k (Re X_k cos(2πkn/N) − Im X_k sin(2πkn/N))`, where
//! `c_k = 1` for the first and last frequency and 2 for the others, since each
//! of those stands for a pair of conjugate frequencies. The samples are
//! multiplied by the synthesis window, and overlap-add joins the frames,
//! dividing each sample by the summed squared window there, so that analysis
//! followed by synthesis returns the original signal (Griffin and Lim, 1984,
//! <https://doi.org/10.1109/TASSP.1984.1164317>; J. O. Smith, *Spectral Audio
//! Signal Processing*, <https://ccrma.stanford.edu/~jos/sasp/>).

use super::config::VocoderConfig;
use staleynet_cuda::{Gpu, GpuResult, MatmulSpec, Shape, Tensor};
use std::f64::consts::TAU;

/// Smallest summed squared window accepted at a sample; the two outermost
/// samples of the padded signal are covered by a vanishing window.
const WINDOW_SUM_FLOOR: f64 = 1e-8;

/// The constant matrices of the inverse transform and the window.
pub struct InverseStft {
    /// `[bins, fft_size]`: `c_k cos(2πkn/N) w[n] / N`.
    real_basis: Tensor,

    /// `[bins, fft_size]`: `−c_k sin(2πkn/N) w[n] / N`.
    imaginary_basis: Tensor,

    /// The synthesis window, one value per frame sample.
    window: Vec<f32>,

    /// Samples per frame, `N`.
    fft_size: usize,

    /// Samples between the starts of consecutive frames.
    hop_size: usize,
}

impl InverseStft {
    /// Builds the bases for `config`, with `window` (normally the analysis
    /// window, a periodic Hann window) as the synthesis window.
    pub fn new(gpu: &Gpu, config: &VocoderConfig, window: &[f32]) -> GpuResult<Self> {
        let fft_size = config.fft_size;
        let bins = config.bins();

        let mut real = vec![0.0f32; bins * fft_size];
        let mut imaginary = vec![0.0f32; bins * fft_size];

        for bin in 0..bins {
            let pair_weight = if bin == 0 || bin == fft_size / 2 { 1.0 } else { 2.0 };

            for sample in 0..fft_size {
                let angle = TAU * (bin * sample) as f64 / fft_size as f64;
                let scale = pair_weight * window[sample] as f64 / fft_size as f64;

                real[bin * fft_size + sample] = (angle.cos() * scale) as f32;
                imaginary[bin * fft_size + sample] = (-angle.sin() * scale) as f32;
            }
        }

        Ok(Self {
            real_basis: gpu.upload(Shape::new(bins, fft_size), &real)?,
            imaginary_basis: gpu.upload(Shape::new(bins, fft_size), &imaginary)?,
            window: window.to_vec(),
            fft_size,
            hop_size: config.hop_size,
        })
    }

    /// The factor that divides each sample of a signal joined from `frames`
    /// frames: one over the summed squared window at that sample.
    pub fn normalization(&self, frames: usize) -> Vec<f32> {
        let length = (frames - 1) * self.hop_size + self.fft_size;
        let mut window_sum = vec![0.0f64; length];

        for frame in 0..frames {
            for (offset, weight) in self.window.iter().enumerate() {
                window_sum[frame * self.hop_size + offset] += (*weight as f64).powi(2);
            }
        }

        window_sum
            .iter()
            .map(|sum| (1.0 / sum.max(WINDOW_SUM_FLOOR)) as f32)
            .collect()
    }

    /// Samples from a spectrum: `frames = real · Br + imaginary · Bi`, then
    /// overlap-add of every sequence of `frames_per_sequence` frames into its
    /// part of `signal`. `frames` is scratch of shape `[rows, fft_size]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        gpu: &Gpu,
        real: &Tensor,
        imaginary: &Tensor,
        normalization: &Tensor,
        frames_per_sequence: usize,
        frames: &mut Tensor,
        signal: &mut Tensor,
    ) -> GpuResult {
        let rows = real.shape().rows;
        let bins = real.shape().cols;
        let product = MatmulSpec::new(rows, self.fft_size, bins);

        gpu.matmul(&product, real, &self.real_basis, frames)?;
        gpu.matmul(&product.accumulate(), imaginary, &self.imaginary_basis, frames)?;

        gpu.overlap_add(frames, normalization, signal, frames_per_sequence, self.hop_size)
    }

    /// The gradient of [`InverseStft::forward`] with respect to the spectrum,
    /// written into `real_grad` and `imaginary_grad`.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        gpu: &Gpu,
        signal_grad: &Tensor,
        normalization: &Tensor,
        frames_per_sequence: usize,
        frames_grad: &mut Tensor,
        real_grad: &mut Tensor,
        imaginary_grad: &mut Tensor,
    ) -> GpuResult {
        gpu.overlap_add_backward(signal_grad, normalization, frames_grad, frames_per_sequence, self.hop_size)?;

        let rows = real_grad.shape().rows;
        let bins = real_grad.shape().cols;
        let product = MatmulSpec::new(rows, bins, self.fft_size).right_transposed();

        gpu.matmul(&product, frames_grad, &self.real_basis, real_grad)?;
        gpu.matmul(&product, frames_grad, &self.imaginary_basis, imaginary_grad)
    }
}
