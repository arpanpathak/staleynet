//! Spectral losses: how far a synthesized waveform is from its target.
//!
//! Both signals are analyzed the same way: cut into frames, transformed,
//! squared into power, optionally pooled into mel bands, and compressed by a
//! logarithm. The loss is the mean absolute difference of the two results.
//! Comparing at several resolutions (Yamamoto et al., 2020,
//! <https://arxiv.org/abs/1910.11480>) asks the waveform to be right both in
//! time (short frames) and in frequency (long frames); comparing log-mel
//! energies (Kong et al., 2020, <https://arxiv.org/abs/2010.05646>) weighs
//! errors the way hearing does.

use staleynet_audio::mel::LOG_ENERGY_FLOOR;
use staleynet_audio::{MelAnalyzer, MelConfig};
use staleynet_cuda::{Gpu, GpuResult, MatmulSpec, Shape, Tensor};

/// One analysis at which the waveform is compared.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Resolution {
    /// Samples per frame.
    pub fft_size: usize,

    /// Samples between the starts of consecutive frames.
    pub hop_size: usize,

    /// Whether frequencies are pooled into mel bands before the logarithm.
    pub mel: bool,

    /// Weight of this loss in the total.
    pub weight: f32,
}

/// The analyses used for training at 24 kHz with a 1024/256 vocoder: the mel
/// loss at the vocoder's own frame size, and plain log-magnitudes with shorter
/// frames (sharper timing) and longer frames (finer frequency). Each divides
/// a synthesized signal exactly into frames.
pub const TRAINING_RESOLUTIONS: [Resolution; 3] = [
    Resolution { fft_size: 1024, hop_size: 256, mel: true, weight: 1.0 },
    Resolution { fft_size: 512, hop_size: 128, mel: false, weight: 0.5 },
    Resolution { fft_size: 2048, hop_size: 256, mel: false, weight: 0.5 },
];

/// One spectral loss with every buffer it needs for a batch of signals.
pub struct SpectralLoss {
    /// The analysis.
    resolution: Resolution,

    /// Frames each signal is cut into.
    frames_per_sequence: usize,

    /// Windowed Fourier bases, `[fft_size, bins]`.
    real_basis: Tensor,

    /// See `real_basis`.
    imaginary_basis: Tensor,

    /// Mel filterbank `[bins, mel_bands]`, when the resolution pools into bands.
    filterbank: Option<Tensor>,

    /// All ones, one per sample of a signal: framing is overlap-add's adjoint
    /// with no normalization.
    ones: Tensor,

    /// The analyzed frames, `[rows, fft_size]`.
    frames: Tensor,

    /// Real and imaginary parts of the spectrum, `[rows, bins]`.
    real: Tensor,

    /// See `real`.
    imaginary: Tensor,

    /// Power per frequency, `[rows, bins]`.
    power: Tensor,

    /// Energies before the logarithm: mel bands or `power` itself.
    energies: Tensor,

    /// Log energies of the estimate.
    features: Tensor,

    /// Log energies of the target.
    target: Tensor,

    /// Gradient of `features`.
    features_grad: Tensor,

    /// Gradient of `energies`.
    energies_grad: Tensor,

    /// Gradient of `power` (aliases `energies_grad` when there is no pooling).
    power_grad: Tensor,

    /// Gradients of the two parts of the spectrum.
    real_grad: Tensor,

    /// See `real_grad`.
    imaginary_grad: Tensor,

    /// Gradient of the frames.
    frames_grad: Tensor,

    /// This loss's contribution to the signal gradient, `[sequences, length]`.
    signal_grad: Tensor,
}

impl SpectralLoss {
    /// Buffers for `sequences` signals of `length` samples at `sample_rate`.
    pub fn new(gpu: &Gpu, resolution: Resolution, sample_rate: u32, mel_bands: usize, sequences: usize, length: usize) -> GpuResult<Self> {
        let (n, hop) = (resolution.fft_size, resolution.hop_size);
        let frames_per_sequence = (length - n) / hop + 1;
        assert_eq!((frames_per_sequence - 1) * hop + n, length, "resolution {n}/{hop} must divide the signal exactly into frames");

        let analyzer = MelAnalyzer::new(MelConfig {
            sample_rate,
            fft_size: n,
            hop_size: hop,
            mel_bands,
            min_frequency: 0.0,
            max_frequency: sample_rate as f32 / 2.0,
        });
        let basis = analyzer.windowed_dft_basis();
        let bins = n / 2 + 1;
        let rows = sequences * frames_per_sequence;
        let features = if resolution.mel { mel_bands } else { bins };

        let filterbank = if resolution.mel { Some(gpu.upload(Shape::new(bins, mel_bands), &analyzer.filterbank)?) } else { None };
        let spectrum = Shape::new(rows, bins);
        let feature = Shape::new(rows, features);

        Ok(Self {
            resolution,
            frames_per_sequence,
            real_basis: gpu.upload(Shape::new(n, bins), &basis.real)?,
            imaginary_basis: gpu.upload(Shape::new(n, bins), &basis.imaginary)?,
            filterbank,
            ones: gpu.upload(Shape::vector(length), &vec![1.0; length])?,
            frames: gpu.zeros(Shape::new(rows, n))?,
            real: gpu.zeros(spectrum)?,
            imaginary: gpu.zeros(spectrum)?,
            power: gpu.zeros(spectrum)?,
            energies: gpu.zeros(feature)?,
            features: gpu.zeros(feature)?,
            target: gpu.zeros(feature)?,
            features_grad: gpu.zeros(feature)?,
            energies_grad: gpu.zeros(feature)?,
            power_grad: gpu.zeros(spectrum)?,
            real_grad: gpu.zeros(spectrum)?,
            imaginary_grad: gpu.zeros(spectrum)?,
            frames_grad: gpu.zeros(Shape::new(rows, n))?,
            signal_grad: gpu.zeros(Shape::new(sequences, length))?,
        })
    }

    /// Analyzes the target signals, which the next [`SpectralLoss::forward`] is compared with.
    pub fn set_target(&mut self, gpu: &Gpu, target: &Tensor) -> GpuResult {
        self.analyze(gpu, target)?;
        copy(gpu, &self.energies, &mut self.target)?;
        gpu.log_compress(&mut self.target, LOG_ENERGY_FLOOR)
    }

    /// Analyzes the estimated signals, keeping what the backward pass needs.
    pub fn forward(&mut self, gpu: &Gpu, estimate: &Tensor) -> GpuResult {
        self.analyze(gpu, estimate)?;
        copy(gpu, &self.energies, &mut self.features)?;
        gpu.log_compress(&mut self.features, LOG_ENERGY_FLOOR)
    }

    /// The weighted mean absolute difference between estimate and target.
    pub fn value(&self, gpu: &Gpu) -> GpuResult<f64> {
        let (estimate, target) = (gpu.download(&self.features)?, gpu.download(&self.target)?);
        let total: f64 = estimate.iter().zip(&target).map(|(a, b)| (a - b).abs() as f64).sum();

        Ok(self.resolution.weight as f64 * total / estimate.len() as f64)
    }

    /// Adds this loss's gradient with respect to the estimated signals into `signal_grad`.
    pub fn backward(&mut self, gpu: &Gpu, signal_grad: &mut Tensor) -> GpuResult {
        let scale = self.resolution.weight / self.features.len() as f32;
        gpu.absolute_error_gradient(&self.features, &self.target, &mut self.features_grad, scale)?;
        gpu.log_compress_backward(&self.energies, &self.features_grad, &mut self.energies_grad, LOG_ENERGY_FLOOR)?;

        let rows = self.power.shape().rows;
        let bins = self.power.shape().cols;
        match &self.filterbank {
            Some(filterbank) => {
                let bands = filterbank.shape().cols;
                gpu.matmul(&MatmulSpec::new(rows, bins, bands).right_transposed(), &self.energies_grad, filterbank, &mut self.power_grad)?;
            }
            None => copy(gpu, &self.energies_grad, &mut self.power_grad)?,
        }
        gpu.power_spectrum_backward(&self.real, &self.imaginary, &self.power_grad, &mut self.real_grad, &mut self.imaginary_grad)?;

        let n = self.resolution.fft_size;
        let product = MatmulSpec::new(rows, n, bins).right_transposed();
        gpu.matmul(&product, &self.real_grad, &self.real_basis, &mut self.frames_grad)?;
        gpu.matmul(&product.accumulate(), &self.imaginary_grad, &self.imaginary_basis, &mut self.frames_grad)?;

        gpu.overlap_add(&self.frames_grad, &self.ones, &mut self.signal_grad, self.frames_per_sequence, self.resolution.hop_size)?;
        gpu.add_scaled(1.0, &self.signal_grad, signal_grad)
    }

    /// Frames, transform, power and (mel) energies of `signals`.
    fn analyze(&mut self, gpu: &Gpu, signals: &Tensor) -> GpuResult {
        let (n, hop) = (self.resolution.fft_size, self.resolution.hop_size);
        gpu.overlap_add_backward(signals, &self.ones, &mut self.frames, self.frames_per_sequence, hop)?;

        let rows = self.frames.shape().rows;
        let bins = self.real.shape().cols;
        let product = MatmulSpec::new(rows, bins, n);
        gpu.matmul(&product, &self.frames, &self.real_basis, &mut self.real)?;
        gpu.matmul(&product, &self.frames, &self.imaginary_basis, &mut self.imaginary)?;
        gpu.power_spectrum(&self.real, &self.imaginary, &mut self.power)?;

        match &self.filterbank {
            Some(filterbank) => {
                gpu.matmul(&MatmulSpec::new(rows, filterbank.shape().cols, bins), &self.power, filterbank, &mut self.energies)
            }
            None => copy(gpu, &self.power, &mut self.energies),
        }
    }
}

/// `destination = source`, for tensors of equal length.
pub fn copy(gpu: &Gpu, source: &Tensor, destination: &mut Tensor) -> GpuResult {
    gpu.fill_zero(destination)?;
    gpu.add_scaled(1.0, source, destination)
}
