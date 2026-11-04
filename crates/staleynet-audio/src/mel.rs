//! Short-time Fourier transform and log-mel spectrogram (CPU reference).
//!
//! A log-mel spectrogram is the magnitude-squared STFT, pooled into
//! perceptually spaced frequency bands and log-compressed. It is the acoustic
//! representation the whole model works in.
//!
//! References:
//! * Short-time Fourier transform: <https://en.wikipedia.org/wiki/Short-time_Fourier_transform>
//! * J. O. Smith, *Spectral Audio Signal Processing*, CCRMA:
//!   <https://ccrma.stanford.edu/~jos/sasp/>
//! * Mel scale: <https://en.wikipedia.org/wiki/Mel_scale>
//! * librosa `filters.mel` (the Slaney convention reproduced here):
//!   <https://librosa.org/doc/latest/generated/librosa.filters.mel.html>

use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::f32::consts::PI;
use std::sync::Arc;

/// One STFT frame: the non-negative-frequency bins `0..=n_fft/2`.
pub type SpectrumFrame = Vec<Complex32>;

/// Row-major `[frame][mel band]` log-mel values, flattened.
pub type LogMel = Vec<f32>;

/// Smallest mel energy before the logarithm, so silence maps to `ln(1e-5)`
/// instead of `-inf`. Same floor as the HiFi-GAN reference implementation
/// (Kong et al., 2020, <https://arxiv.org/abs/2010.05646>).
pub const LOG_ENERGY_FLOOR: f32 = 1e-5;

/// Frequency below which the Slaney mel scale is linear.
/// See M. Slaney, *Auditory Toolbox* v2, Interval Research TR 1998-010.
const SLANEY_LINEAR_LIMIT_HZ: f32 = 1000.0;

/// Width of one mel on the linear part of the Slaney scale (200/3 Hz).
const SLANEY_HZ_PER_MEL: f32 = 200.0 / 3.0;

/// Log step of the Slaney scale above 1 kHz: 27 mels per factor of 6.4.
const SLANEY_LOG_STEP: f32 = 0.068_751_78; // ln(6.4) / 27

/// STFT and mel-filterbank parameters.
///
/// The defaults follow Vocos (Siuzdak, 2023, <https://arxiv.org/abs/2306.00814>):
/// 24 kHz audio, 1024-point FFT, hop 256 (≈ 93.75 frames per second), but with
/// 128 mel bands instead of 100 to keep more resolution for full-band music.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MelConfig {
    /// Sampling rate the analysis assumes, in Hz.
    pub sample_rate: u32,

    /// FFT length and analysis window length, in samples.
    pub fft_size: usize,

    /// Distance between consecutive frames, in samples.
    pub hop_size: usize,

    /// Number of triangular mel bands.
    pub mel_bands: usize,

    /// Lower edge of the lowest mel band, in Hz.
    pub min_frequency: f32,

    /// Upper edge of the highest mel band, in Hz (at most Nyquist).
    pub max_frequency: f32,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self { sample_rate: 24_000, fft_size: 1024, hop_size: 256, mel_bands: 128, min_frequency: 0.0, max_frequency: 12_000.0 }
    }
}

impl MelConfig {
    /// Number of non-negative-frequency FFT bins, `fft_size / 2 + 1`.
    pub fn frequency_bins(&self) -> usize {
        self.fft_size / 2 + 1
    }

    /// Spectrogram frames per second of audio.
    pub fn frames_per_second(&self) -> f32 {
        self.sample_rate as f32 / self.hop_size as f32
    }

    /// Frames produced for `sample_count` samples with centred framing.
    pub fn frame_count(&self, sample_count: usize) -> usize {
        1 + sample_count / self.hop_size
    }

    /// Centre frequency of FFT bin `bin`, in Hz.
    fn bin_frequency(&self, bin: usize) -> f32 {
        bin as f32 * self.sample_rate as f32 / self.fft_size as f32
    }
}

/// Hz → mel on the Slaney scale (linear below 1 kHz, logarithmic above).
fn hz_to_mel(frequency: f32) -> f32 {
    let linear_mels = frequency / SLANEY_HZ_PER_MEL;
    let limit_mels = SLANEY_LINEAR_LIMIT_HZ / SLANEY_HZ_PER_MEL;

    if frequency < SLANEY_LINEAR_LIMIT_HZ { linear_mels } else { limit_mels + (frequency / SLANEY_LINEAR_LIMIT_HZ).ln() / SLANEY_LOG_STEP }
}

/// Inverse of [`hz_to_mel`].
fn mel_to_hz(mels: f32) -> f32 {
    let limit_mels = SLANEY_LINEAR_LIMIT_HZ / SLANEY_HZ_PER_MEL;

    if mels < limit_mels { mels * SLANEY_HZ_PER_MEL } else { SLANEY_LINEAR_LIMIT_HZ * (SLANEY_LOG_STEP * (mels - limit_mels)).exp() }
}

/// Periodic Hann window, the usual choice for STFT analysis with 75 % overlap.
/// See <https://en.wikipedia.org/wiki/Hann_function>
fn periodic_hann(length: usize) -> Vec<f32> {
    (0..length)
        .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f32 / length as f32).cos())
        .collect()
}

/// Area-normalised triangular filterbank, row-major `[frequency bin][mel band]`.
fn slaney_filterbank(config: &MelConfig) -> Vec<f32> {
    let band_count = config.mel_bands;
    let lowest_mel = hz_to_mel(config.min_frequency);
    let mel_span = hz_to_mel(config.max_frequency) - lowest_mel;

    // band_count + 2 edges: each band rises from edge b to b+1 and falls to b+2.
    let edges_hz: Vec<f32> = (0..band_count + 2)
        .map(|edge| mel_to_hz(lowest_mel + mel_span * edge as f32 / (band_count + 1) as f32))
        .collect();

    let mut weights = vec![0.0; config.frequency_bins() * band_count];
    for band in 0..band_count {
        let (lower, centre, upper) = (edges_hz[band], edges_hz[band + 1], edges_hz[band + 2]);
        let area_normaliser = 2.0 / (upper - lower);

        for bin in 0..config.frequency_bins() {
            let frequency = config.bin_frequency(bin);
            let rising = (frequency - lower) / (centre - lower);
            let falling = (upper - frequency) / (upper - centre);
            weights[bin * band_count + band] = rising.min(falling).max(0.0) * area_normaliser;
        }
    }

    weights
}

/// Precomputed window, filterbank and FFT plans for one [`MelConfig`].
pub struct MelAnalyzer {
    /// The parameters everything below was built from.
    pub config: MelConfig,

    /// Periodic Hann window of length `fft_size`.
    pub window: Vec<f32>,

    /// Mel filterbank, row-major `[frequency bin][mel band]`.
    pub filterbank: Vec<f32>,

    /// Forward FFT plan of length `fft_size`.
    forward_fft: Arc<dyn Fft<f32>>,

    /// Inverse FFT plan of length `fft_size`.
    inverse_fft: Arc<dyn Fft<f32>>,
}

impl MelAnalyzer {
    /// Builds the window, filterbank and FFT plans.
    pub fn new(config: MelConfig) -> Self {
        let mut planner = FftPlanner::new();

        Self {
            window: periodic_hann(config.fft_size),
            filterbank: slaney_filterbank(&config),
            forward_fft: planner.plan_fft_forward(config.fft_size),
            inverse_fft: planner.plan_fft_inverse(config.fft_size),
            config,
        }
    }

    /// Mirrors `fft_size / 2` samples onto each end so frame `f` is centred on
    /// sample `f * hop_size` (the `center=True` convention of librosa/PyTorch).
    pub fn reflect_pad(&self, signal: &[f32]) -> Vec<f32> {
        let padding = self.config.fft_size / 2;
        let last = signal.len() as isize - 1;
        let period = 2 * last.max(1);

        let mirrored_index = |index: isize| {
            let folded = index.rem_euclid(period);
            if folded > last { period - folded } else { folded }
        };

        (-(padding as isize)..(signal.len() + padding) as isize)
            .map(|index| signal[mirrored_index(index) as usize])
            .collect()
    }

    /// Complex STFT of `signal`, one [`SpectrumFrame`] per hop.
    pub fn stft(&self, signal: &[f32]) -> Vec<SpectrumFrame> {
        let padded = self.reflect_pad(signal);
        let MelConfig { fft_size, hop_size, .. } = self.config;

        (0..self.config.frame_count(signal.len()))
            .into_par_iter()
            .map(|frame| {
                let segment = &padded[frame * hop_size..frame * hop_size + fft_size];
                let mut spectrum: Vec<Complex32> = segment
                    .iter()
                    .zip(&self.window)
                    .map(|(sample, weight)| Complex32::new(sample * weight, 0.0))
                    .collect();

                self.forward_fft.process(&mut spectrum);
                spectrum.truncate(self.config.frequency_bins());
                spectrum
            })
            .collect()
    }

    /// Log-mel spectrogram of `signal`, `[frame][mel band]` flattened.
    pub fn log_mel(&self, signal: &[f32]) -> LogMel {
        let band_count = self.config.mel_bands;
        let mut log_mel = vec![0.0; self.config.frame_count(signal.len()) * band_count];

        log_mel
            .par_chunks_mut(band_count)
            .zip(self.stft(signal))
            .for_each(|(mel_row, spectrum)| self.pool_into_bands(&spectrum, mel_row));
        log_mel
    }

    /// Writes `ln(max(filterbankᵀ · |spectrum|², floor))` into `mel_row`.
    fn pool_into_bands(&self, spectrum: &[Complex32], mel_row: &mut [f32]) {
        let band_count = self.config.mel_bands;
        mel_row.fill(0.0);

        for (bin, value) in spectrum.iter().enumerate() {
            let power = value.norm_sqr();
            let bin_weights = &self.filterbank[bin * band_count..(bin + 1) * band_count];

            for (energy, weight) in mel_row.iter_mut().zip(bin_weights) {
                *energy += power * weight;
            }
        }

        for energy in mel_row.iter_mut() {
            *energy = energy.max(LOG_ENERGY_FLOOR).ln();
        }
    }

    /// Windowed DFT bases for computing the STFT as a matrix product
    /// `frames · basis`, each `[fft_size][frequency bin]`: real part
    /// `w[n]·cos(2πkn/N)` and imaginary part `−w[n]·sin(2πkn/N)`.
    /// This is how the GPU path computes the STFT. See
    /// <https://en.wikipedia.org/wiki/DFT_matrix>
    pub fn windowed_dft_basis(&self) -> DftBasis {
        let fft_size = self.config.fft_size;
        let bins = self.config.frequency_bins();
        let mut basis = DftBasis { real: vec![0.0; fft_size * bins], imaginary: vec![0.0; fft_size * bins] };

        for sample in 0..fft_size {
            for bin in 0..bins {
                // Reduce k·n modulo N before scaling so the angle stays exact.
                let turns = ((bin * sample) % fft_size) as f64 / fft_size as f64;
                let angle = 2.0 * std::f64::consts::PI * turns;
                let weight = self.window[sample] as f64;

                basis.real[sample * bins + bin] = (weight * angle.cos()) as f32;
                basis.imaginary[sample * bins + bin] = (-weight * angle.sin()) as f32;
            }
        }

        basis
    }

    /// Inverse STFT by weighted overlap-add, normalised by the summed squared
    /// window (Griffin & Lim, 1984, eq. 6; <https://doi.org/10.1109/TASSP.1984.1164317>).
    pub fn istft(&self, frames: &[SpectrumFrame], sample_count: usize) -> Vec<f32> {
        let MelConfig { fft_size, hop_size, .. } = self.config;
        let padded_length = (frames.len() - 1) * hop_size + fft_size;

        let time_frames: Vec<Vec<f32>> = frames
            .par_iter()
            .map(|half_spectrum| self.inverse_frame(half_spectrum))
            .collect();

        let mut signal = vec![0.0; padded_length];
        let mut window_energy = vec![0.0; padded_length];
        for (frame_index, frame) in time_frames.iter().enumerate() {
            let start = frame_index * hop_size;

            for (offset, (sample, weight)) in frame.iter().zip(&self.window).enumerate() {
                signal[start + offset] += sample;
                window_energy[start + offset] += weight * weight;
            }
        }

        let padding = fft_size / 2;
        (padding..padding + sample_count)
            .map(|index| match window_energy.get(index) {
                Some(&energy) if energy > f32::EPSILON => signal[index] / energy,
                _ => 0.0,
            })
            .collect()
    }

    /// One windowed time-domain frame from its non-negative-frequency half,
    /// restoring the negative frequencies by Hermitian symmetry.
    fn inverse_frame(&self, half_spectrum: &[Complex32]) -> Vec<f32> {
        let fft_size = self.config.fft_size;
        let mut spectrum = vec![Complex32::new(0.0, 0.0); fft_size];
        spectrum[..half_spectrum.len()].copy_from_slice(half_spectrum);

        for bin in 1..fft_size / 2 {
            spectrum[fft_size - bin] = half_spectrum[bin].conj();
        }

        self.inverse_fft.process(&mut spectrum);

        let scale = 1.0 / fft_size as f32;
        spectrum
            .iter()
            .zip(&self.window)
            .map(|(value, weight)| value.re * scale * weight)
            .collect()
    }
}

/// Real and imaginary windowed DFT matrices, each `[fft_size][frequency bin]`.
pub struct DftBasis {
    /// `w[n]·cos(2πkn/N)`.
    pub real: Vec<f32>,

    /// `−w[n]·sin(2πkn/N)`.
    pub imaginary: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_signal(length: usize) -> Vec<f32> {
        (0..length)
            .map(|index| (index as f32 * 0.05).sin() * 0.5 + (index as f32 * 0.31).cos() * 0.2)
            .collect()
    }

    #[test]
    fn istft_inverts_stft() {
        let analyzer = MelAnalyzer::new(MelConfig::default());
        let signal = test_signal(24_000);

        let reconstructed = analyzer.istft(&analyzer.stft(&signal), signal.len());

        let worst_error = signal
            .iter()
            .zip(&reconstructed)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(worst_error < 1e-4, "worst error {worst_error}");
    }

    #[test]
    fn dft_basis_matches_fft() {
        let analyzer = MelAnalyzer::new(MelConfig::default());
        let signal = test_signal(4096);

        let spectrum = analyzer.stft(&signal);
        let basis = analyzer.windowed_dft_basis();
        let padded = analyzer.reflect_pad(&signal);

        let MelConfig { fft_size, hop_size, .. } = analyzer.config;
        let bins = analyzer.config.frequency_bins();
        let frame = 3;

        for bin in [0, 1, 100, bins - 1] {
            let segment = &padded[frame * hop_size..frame * hop_size + fft_size];
            let real: f32 = segment
                .iter()
                .enumerate()
                .map(|(sample, value)| value * basis.real[sample * bins + bin])
                .sum();
            let imaginary: f32 = segment
                .iter()
                .enumerate()
                .map(|(sample, value)| value * basis.imaginary[sample * bins + bin])
                .sum();

            assert!((real - spectrum[frame][bin].re).abs() < 1e-3);
            assert!((imaginary - spectrum[frame][bin].im).abs() < 1e-3);
        }
    }

    #[test]
    fn slaney_scale_round_trips() {
        for frequency in [0.0, 440.0, 1000.0, 4000.0, 11_000.0] {
            assert!((mel_to_hz(hz_to_mel(frequency)) - frequency).abs() < 1e-2);
        }
    }

    #[test]
    fn a_1_khz_tone_peaks_in_the_band_containing_1_khz() {
        let config = MelConfig::default();
        let analyzer = MelAnalyzer::new(config);
        let tone: Vec<f32> = (0..24_000)
            .map(|index| (2.0 * PI * 1000.0 * index as f32 / 24_000.0).sin())
            .collect();

        let log_mel = analyzer.log_mel(&tone);
        let middle_frame = &log_mel[20 * config.mel_bands..21 * config.mel_bands];
        let loudest_band = (0..config.mel_bands)
            .max_by(|&a, &b| middle_frame[a].total_cmp(&middle_frame[b]))
            .unwrap();

        let bin_for_1khz = (1000.0 * config.fft_size as f32 / config.sample_rate as f32).round() as usize;
        assert!(analyzer.filterbank[bin_for_1khz * config.mel_bands + loudest_band] > 0.0);
    }
}

#[cfg(test)]
mod edge_tests {
    use super::*;

    #[test]
    fn config_helpers_match_the_defaults() {
        let config = MelConfig::default();

        assert_eq!(config.frequency_bins(), 513);
        assert!((config.frames_per_second() - 93.75).abs() < 1e-6);
        assert_eq!(config.frame_count(0), 1);
        assert_eq!(config.frame_count(2_560), 11);
        assert_eq!(config.bin_frequency(0), 0.0);
        assert!((config.bin_frequency(512) - 12_000.0).abs() < 1e-3);
    }

    #[test]
    fn reflect_pad_keeps_the_signal_centred() {
        let analyzer = MelAnalyzer::new(MelConfig::default());
        let signal = [1.0f32, 2.0, 3.0, 4.0];
        let padded = analyzer.reflect_pad(&signal);
        let padding = analyzer.config.fft_size / 2;

        assert_eq!(padded.len(), signal.len() + analyzer.config.fft_size);
        assert_eq!(&padded[padding..padding + signal.len()], &signal);
        assert!(padded.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn silence_uses_the_log_energy_floor() {
        let analyzer = MelAnalyzer::new(MelConfig::default());
        let log_mel = analyzer.log_mel(&vec![0.0f32; 2_400]);
        let expected = LOG_ENERGY_FLOOR.ln();

        assert_eq!(log_mel.len(), analyzer.config.frame_count(2_400) * analyzer.config.mel_bands);

        for value in log_mel {
            assert!((value - expected).abs() < 1e-4, "value {value} expected {expected}");
        }
    }

    #[test]
    fn stft_shape_matches_frames_and_bins() {
        let analyzer = MelAnalyzer::new(MelConfig::default());
        let frames = analyzer.stft(&vec![0.0f32; 2_400]);

        assert_eq!(frames.len(), analyzer.config.frame_count(2_400));
        assert!(
            frames
                .iter()
                .all(|frame| frame.len() == analyzer.config.frequency_bins())
        );
    }

    #[test]
    fn pooling_a_zero_spectrum_uses_the_floor() {
        let analyzer = MelAnalyzer::new(MelConfig::default());
        let mut row = vec![1.0f32; analyzer.config.mel_bands];
        let spectrum = vec![Complex32::new(0.0, 0.0); analyzer.config.frequency_bins()];

        analyzer.pool_into_bands(&spectrum, &mut row);

        let expected = LOG_ENERGY_FLOOR.ln();

        assert!(row.iter().all(|value| (value - expected).abs() < 1e-4));
    }

    #[test]
    fn dft_basis_and_filterbank_have_the_expected_shape() {
        let analyzer = MelAnalyzer::new(MelConfig::default());
        let basis = analyzer.windowed_dft_basis();
        let bins = analyzer.config.frequency_bins();

        assert_eq!(basis.real.len(), analyzer.config.fft_size * bins);
        assert_eq!(basis.imaginary.len(), analyzer.config.fft_size * bins);
        assert_eq!(analyzer.filterbank.len(), bins * analyzer.config.mel_bands);
    }

    #[test]
    fn istft_zero_pads_samples_beyond_the_window_energy() {
        let analyzer = MelAnalyzer::new(MelConfig::default());
        let frames = analyzer.stft(&vec![0.0f32; 100]);
        let requested = frames.len() * analyzer.config.hop_size + analyzer.config.fft_size;

        let waveform = analyzer.istft(&frames, requested);

        assert_eq!(waveform.len(), requested);
        assert!(waveform.iter().all(|value| value.is_finite()));
        assert_eq!(waveform[requested - 1], 0.0);
    }

    #[test]
    fn hz_to_mel_is_monotonic() {
        let mut previous = f32::NEG_INFINITY;

        for frequency in [0.0, 100.0, 999.0, 1_000.0, 2_000.0, 8_000.0] {
            let mel = hz_to_mel(frequency);

            assert!(mel > previous, "frequency {frequency}");

            previous = mel;
        }
    }
}
