//! Log-mel → waveform by Griffin-Lim phase reconstruction.
//!
//! Griffin-Lim alternates between the time domain and the STFT domain,
//! keeping the target magnitudes and only the *phase* of the re-analysed
//! signal, until magnitude and phase are consistent. The "fast" variant adds a
//! momentum term that converges in far fewer iterations.
//!
//! This is a signal-processing baseline, not a neural vocoder: it is audibly
//! "phasey", and a learned vocoder should beat it.
//!
//! References:
//! * D. Griffin, J. Lim, "Signal estimation from modified short-time Fourier
//!   transform", IEEE TASSP 1984: <https://doi.org/10.1109/TASSP.1984.1164317>
//! * N. Perraudin, P. Balazs, P. Søndergaard, "A fast Griffin-Lim algorithm",
//!   WASPAA 2013: <https://doi.org/10.1109/WASPAA.2013.6701851>

use crate::mel::{MelAnalyzer, SpectrumFrame};
use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use std::f32::consts::TAU;

/// Momentum of fast Griffin-Lim; Perraudin et al. recommend α = 0.99.
const FAST_GRIFFIN_LIM_MOMENTUM: f32 = 0.99;

/// Tikhonov regularisation of the filterbank pseudo-inverse, relative to the
/// mean diagonal of `FᵀF`. Keeps the inversion stable where mel bands overlap
/// only a few FFT bins. See <https://en.wikipedia.org/wiki/Ridge_regression>
const PSEUDO_INVERSE_RIDGE: f64 = 1e-3;

/// Magnitudes below this are treated as zero when taking a phase.
const PHASE_EPSILON: f32 = 1e-12;

/// Row-major `[rows][cols]` dense matrix used only for the small pseudo-inverse.
struct DenseMatrix {
    /// Number of rows.
    rows: usize,

    /// Number of columns.
    cols: usize,

    /// Row-major values.
    values: Vec<f64>,
}

impl DenseMatrix {
    /// Value at (`row`, `col`).
    fn at(&self, row: usize, col: usize) -> f64 {
        self.values[row * self.cols + col]
    }
}

/// Cholesky factor `L` of a symmetric positive-definite matrix (`A = L·Lᵀ`).
/// See <https://en.wikipedia.org/wiki/Cholesky_decomposition>
fn cholesky(matrix: &DenseMatrix) -> DenseMatrix {
    let size = matrix.rows;
    let mut lower = DenseMatrix { rows: size, cols: size, values: vec![0.0; size * size] };

    for row in 0..size {
        for col in 0..=row {
            let dot: f64 = (0..col).map(|k| lower.at(row, k) * lower.at(col, k)).sum();
            let residual = matrix.at(row, col) - dot;

            lower.values[row * size + col] = if row == col { residual.sqrt() } else { residual / lower.at(col, col) };
        }
    }

    lower
}

/// Solves `L·Lᵀ·x = rhs` for every column of `rhs` (forward then back substitution).
fn cholesky_solve(lower: &DenseMatrix, rhs: &mut DenseMatrix) {
    let size = lower.rows;

    for col in 0..rhs.cols {
        for row in 0..size {
            let dot: f64 = (0..row).map(|k| lower.at(row, k) * rhs.at(k, col)).sum();
            rhs.values[row * rhs.cols + col] = (rhs.at(row, col) - dot) / lower.at(row, row);
        }

        for row in (0..size).rev() {
            let dot: f64 = (row + 1..size).map(|k| lower.at(k, row) * rhs.at(k, col)).sum();
            rhs.values[row * rhs.cols + col] = (rhs.at(row, col) - dot) / lower.at(row, row);
        }
    }
}

/// Ridge-regularised pseudo-inverse of the filterbank, `[mel band][frequency bin]`:
/// `(FᵀF + λI)⁻¹ Fᵀ`, which maps mel energies back to linear-frequency power.
fn filterbank_pseudo_inverse(analyzer: &MelAnalyzer) -> Vec<f32> {
    let bins = analyzer.config.frequency_bins();
    let bands = analyzer.config.mel_bands;
    let weight = |bin: usize, band: usize| analyzer.filterbank[bin * bands + band] as f64;

    let mut gram = DenseMatrix { rows: bands, cols: bands, values: vec![0.0; bands * bands] };
    for (band_a, band_b) in (0..bands).flat_map(|a| (0..bands).map(move |b| (a, b))) {
        gram.values[band_a * bands + band_b] = (0..bins).map(|bin| weight(bin, band_a) * weight(bin, band_b)).sum();
    }

    let mean_diagonal = (0..bands).map(|band| gram.at(band, band)).sum::<f64>() / bands as f64;
    for band in 0..bands {
        gram.values[band * bands + band] += PSEUDO_INVERSE_RIDGE * mean_diagonal;
    }

    let mut transposed = DenseMatrix { rows: bands, cols: bins, values: vec![0.0; bands * bins] };
    for (band, bin) in (0..bands).flat_map(|band| (0..bins).map(move |bin| (band, bin))) {
        transposed.values[band * bins + bin] = weight(bin, band);
    }

    cholesky_solve(&cholesky(&gram), &mut transposed);
    transposed.values.into_iter().map(|value| value as f32).collect()
}

/// Linear-frequency magnitudes `[frame][bin]` implied by a log-mel spectrogram.
fn mel_to_magnitudes(analyzer: &MelAnalyzer, log_mel: &[f32]) -> Vec<Vec<f32>> {
    let bins = analyzer.config.frequency_bins();
    let bands = analyzer.config.mel_bands;
    let inverse = filterbank_pseudo_inverse(analyzer);

    log_mel
        .par_chunks(bands)
        .map(|mel_row| {
            let mut power = vec![0.0f32; bins];
            for (band, log_energy) in mel_row.iter().enumerate() {
                let energy = log_energy.exp();
                let band_row = &inverse[band * bins..(band + 1) * bins];

                for (bin_power, coefficient) in power.iter_mut().zip(band_row) {
                    *bin_power += energy * coefficient;
                }
            }

            power.into_iter().map(|value| value.max(0.0).sqrt()).collect()
        })
        .collect()
}

/// Deterministic xorshift generator for the initial random phases.
/// See G. Marsaglia, "Xorshift RNGs", JSS 2003: <https://doi.org/10.18637/jss.v008.i14>
struct PhaseGenerator(u64);

impl PhaseGenerator {
    /// Next phase, uniform in `[0, 2π)`.
    fn next_phase(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;

        let unit = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        unit * TAU
    }
}

/// Settings for [`griffin_lim`].
#[derive(Clone, Copy, Debug)]
pub struct GriffinLimConfig {
    /// Number of projection iterations; 32–100 is typical.
    pub iterations: usize,

    /// Seed of the random initial phase, for reproducible output.
    pub seed: u64,
}

/// Reconstructs a waveform from a `[frame][mel band]` log-mel spectrogram.
pub fn griffin_lim(analyzer: &MelAnalyzer, log_mel: &[f32], config: GriffinLimConfig) -> Vec<f32> {
    let magnitudes = mel_to_magnitudes(analyzer, log_mel);
    if magnitudes.len() <= 1 {
        return Vec::new();
    }

    let sample_count = (magnitudes.len() - 1) * analyzer.config.hop_size;
    let mut phases = PhaseGenerator(config.seed | 1);

    let mut estimate: Vec<SpectrumFrame> = magnitudes
        .iter()
        .map(|frame| frame.iter().map(|&magnitude| Complex32::from_polar(magnitude, phases.next_phase())).collect())
        .collect();
    let mut previous_projection: Option<Vec<SpectrumFrame>> = None;
    let mut waveform = analyzer.istft(&estimate, sample_count);

    for _ in 0..config.iterations {
        let projection = analyzer.stft(&waveform);

        for (frame_index, target_frame) in magnitudes.iter().enumerate() {
            for (bin, &target) in target_frame.iter().enumerate() {
                let current = projection[frame_index][bin];
                let accelerated = match &previous_projection {
                    Some(previous) => current + (current - previous[frame_index][bin]) * FAST_GRIFFIN_LIM_MOMENTUM,
                    None => current,
                };

                let norm = accelerated.norm();
                estimate[frame_index][bin] = if norm > PHASE_EPSILON { accelerated * (target / norm) } else { Complex32::new(target, 0.0) };
            }
        }

        previous_projection = Some(projection);
        waveform = analyzer.istft(&estimate, sample_count);
    }

    waveform
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mel::MelConfig;

    #[test]
    fn reconstruction_preserves_the_log_mel_spectrogram() {
        let analyzer = MelAnalyzer::new(MelConfig::default());
        let chord: Vec<f32> = (0..24_000)
            .map(|index| {
                let time = index as f32 / 24_000.0;
                0.3 * (TAU * 220.0 * time).sin() + 0.2 * (TAU * 440.0 * time).sin()
            })
            .collect();
        let original = analyzer.log_mel(&chord);

        let waveform = griffin_lim(&analyzer, &original, GriffinLimConfig { iterations: 32, seed: 7 });
        let reanalysed = analyzer.log_mel(&waveform);

        let compared = original.len().min(reanalysed.len());
        let mean_error =
            original[..compared].iter().zip(&reanalysed[..compared]).map(|(a, b)| (a - b).abs()).sum::<f32>() / compared as f32;
        assert!(mean_error < 1.0, "mean log-mel error {mean_error}");
    }
}

#[cfg(test)]
mod edge_tests {
    use super::*;
    use crate::mel::MelConfig;

    fn short_signal() -> Vec<f32> {
        (0..1_200).map(|index| (index as f32 * 0.05).sin() * 0.3).collect()
    }

    #[test]
    fn output_length_matches_the_frame_count() {
        let analyzer = MelAnalyzer::new(MelConfig::default());
        let log_mel = analyzer.log_mel(&short_signal());
        let frames = log_mel.len() / analyzer.config.mel_bands;

        let waveform = griffin_lim(&analyzer, &log_mel, GriffinLimConfig { iterations: 2, seed: 3 });

        assert_eq!(waveform.len(), (frames - 1) * analyzer.config.hop_size);
        assert!(waveform.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn the_seed_makes_reconstruction_deterministic() {
        let analyzer = MelAnalyzer::new(MelConfig::default());
        let log_mel = analyzer.log_mel(&short_signal());
        let config = GriffinLimConfig { iterations: 4, seed: 11 };

        let first = griffin_lim(&analyzer, &log_mel, config);
        let second = griffin_lim(&analyzer, &log_mel, config);
        let other = griffin_lim(&analyzer, &log_mel, GriffinLimConfig { iterations: 4, seed: 12 });

        assert_eq!(first, second);
        assert_ne!(first, other);
    }

    #[test]
    fn zero_iterations_returns_a_finite_waveform() {
        let analyzer = MelAnalyzer::new(MelConfig::default());
        let log_mel = analyzer.log_mel(&short_signal());

        let waveform = griffin_lim(&analyzer, &log_mel, GriffinLimConfig { iterations: 0, seed: 5 });

        assert!(!waveform.is_empty());
        assert!(waveform.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn a_single_frame_yields_an_empty_waveform() {
        let analyzer = MelAnalyzer::new(MelConfig::default());
        let log_mel = vec![0.0f32; analyzer.config.mel_bands];

        let waveform = griffin_lim(&analyzer, &log_mel, GriffinLimConfig { iterations: 1, seed: 1 });

        assert!(waveform.is_empty());
    }

    #[test]
    fn phase_generator_stays_in_range_and_varies() {
        let mut generator = PhaseGenerator(1);

        let first = generator.next_phase();
        let second = generator.next_phase();

        assert!((0.0..TAU).contains(&first));
        assert!((0.0..TAU).contains(&second));
        assert_ne!(first, second);
    }

    #[test]
    fn cholesky_solves_a_small_positive_definite_system() {
        let matrix = DenseMatrix { rows: 2, cols: 2, values: vec![4.0, 2.0, 2.0, 3.0] };
        let mut rhs = DenseMatrix { rows: 2, cols: 2, values: vec![1.0, 0.0, 0.0, 1.0] };

        cholesky_solve(&cholesky(&matrix), &mut rhs);

        assert!((rhs.at(0, 0) - 0.375).abs() < 1e-12);
        assert!((rhs.at(0, 1) + 0.25).abs() < 1e-12);
        assert!((rhs.at(1, 0) + 0.25).abs() < 1e-12);
        assert!((rhs.at(1, 1) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn small_filterbank_inverse_and_magnitudes_are_finite() {
        let config = MelConfig { sample_rate: 8_000, fft_size: 64, hop_size: 16, mel_bands: 8, min_frequency: 0.0, max_frequency: 4_000.0 };
        let analyzer = MelAnalyzer::new(config);

        let inverse = filterbank_pseudo_inverse(&analyzer);
        let magnitudes = mel_to_magnitudes(&analyzer, &vec![0.0f32; config.mel_bands]);

        assert_eq!(inverse.len(), config.mel_bands * config.frequency_bins());
        assert!(inverse.iter().all(|value| value.is_finite()));

        assert_eq!(magnitudes.len(), 1);
        assert_eq!(magnitudes[0].len(), config.frequency_bins());
        assert!(magnitudes[0].iter().all(|value| value.is_finite() && *value >= 0.0));
    }
}
