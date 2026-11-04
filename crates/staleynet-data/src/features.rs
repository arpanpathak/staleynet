//! Log-mel spectrograms on the GPU, computed as matrix products:
//!
//! ```text
//! frames  [F, N]      = frame(reflect_pad(signal))
//! real    [F, N/2+1]  = frames · W_real          (Hann window folded into W)
//! imag    [F, N/2+1]  = frames · W_imag
//! power               = real² + imag²
//! mel     [F, bands]  = power · filterbank
//! log-mel             = ln(max(mel, floor))
//! ```
//!
//! A DFT by matrix product costs O(N²) per frame instead of the FFT's
//! O(N log N), but it maps onto the GEMM kernel the model already uses and is
//! validated against the rustfft reference in `staleynet-audio`.

use staleynet_audio::mel::{LOG_ENERGY_FLOOR, LogMel, MelAnalyzer};
use staleynet_cuda::{Gpu, GpuResult, MatmulSpec, Shape, Tensor};

/// Frames processed per GPU pass; bounds device memory for long recordings
/// (4096 frames ≈ 44 s at 24 kHz / hop 256).
const FRAMES_PER_CHUNK: usize = 4096;

/// GPU log-mel extractor with its constant matrices resident on the device.
pub struct GpuMelExtractor {
    /// Windowed cosine basis `[fft_size, bins]`.
    real_basis: Tensor,

    /// Windowed negative-sine basis `[fft_size, bins]`.
    imaginary_basis: Tensor,

    /// Mel filterbank `[bins, bands]`.
    filterbank: Tensor,
}

impl GpuMelExtractor {
    /// Uploads the DFT bases and filterbank of `analyzer`.
    pub fn new(gpu: &Gpu, analyzer: &MelAnalyzer) -> GpuResult<Self> {
        let config = analyzer.config;
        let basis_shape = Shape::new(config.fft_size, config.frequency_bins());
        let basis = analyzer.windowed_dft_basis();

        Ok(Self {
            real_basis: gpu.upload(basis_shape, &basis.real)?,
            imaginary_basis: gpu.upload(basis_shape, &basis.imaginary)?,
            filterbank: gpu.upload(Shape::new(config.frequency_bins(), config.mel_bands), &analyzer.filterbank)?,
        })
    }

    /// Log-mel spectrogram of `signal`, framed exactly like
    /// [`MelAnalyzer::log_mel`].
    pub fn log_mel(&self, gpu: &Gpu, analyzer: &MelAnalyzer, signal: &[f32]) -> GpuResult<LogMel> {
        let config = analyzer.config;
        let padded = analyzer.reflect_pad(signal);
        let total_frames = config.frame_count(signal.len());
        let mut log_mel = Vec::with_capacity(total_frames * config.mel_bands);

        for first_frame in (0..total_frames).step_by(FRAMES_PER_CHUNK) {
            let frame_count = FRAMES_PER_CHUNK.min(total_frames - first_frame);
            let first_sample = first_frame * config.hop_size;
            let last_sample = first_sample + (frame_count - 1) * config.hop_size + config.fft_size;

            log_mel.extend(self.chunk(gpu, analyzer, &padded[first_sample..last_sample], frame_count)?);
        }

        Ok(log_mel)
    }

    /// Log-mel of `frame_count` consecutive frames of an already padded segment.
    fn chunk(&self, gpu: &Gpu, analyzer: &MelAnalyzer, segment: &[f32], frame_count: usize) -> GpuResult<Vec<f32>> {
        let config = analyzer.config;
        let bins = config.frequency_bins();
        let spectrum = Shape::new(frame_count, bins);

        let signal = gpu.upload(Shape::vector(segment.len()), segment)?;
        let mut frames = gpu.zeros(Shape::new(frame_count, config.fft_size))?;
        gpu.frame_signal(&signal, &mut frames, config.hop_size)?;

        let transform = MatmulSpec::new(frame_count, bins, config.fft_size);
        let mut real = gpu.zeros(spectrum)?;
        let mut imaginary = gpu.zeros(spectrum)?;
        gpu.matmul(&transform, &frames, &self.real_basis, &mut real)?;
        gpu.matmul(&transform, &frames, &self.imaginary_basis, &mut imaginary)?;

        let mut power = gpu.zeros(spectrum)?;
        gpu.power_spectrum(&real, &imaginary, &mut power)?;

        let mut mel = gpu.zeros(Shape::new(frame_count, config.mel_bands))?;
        gpu.matmul(&MatmulSpec::new(frame_count, config.mel_bands, bins), &power, &self.filterbank, &mut mel)?;
        gpu.log_compress(&mut mel, LOG_ENERGY_FLOOR)?;

        gpu.download(&mel)
    }
}
