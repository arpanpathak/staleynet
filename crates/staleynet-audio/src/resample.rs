//! Band-limited sample-rate conversion with a Kaiser-windowed sinc kernel,
//! implemented as a polyphase filter bank.
//!
//! Each output sample is a weighted sum of nearby input samples, the weights
//! being an ideal low-pass (sinc) impulse response truncated by a Kaiser
//! window. Because the rate ratio is rational, `target / gcd` distinct
//! fractional offsets ("phases") occur; their taps are computed once, so each
//! output sample costs one short dot product.
//!
//! References:
//! * J. O. Smith, *Digital Audio Resampling Home Page*, CCRMA, Stanford:
//!   <https://ccrma.stanford.edu/~jos/resample/>
//! * Polyphase filter banks: <https://en.wikipedia.org/wiki/Polyphase_quadrature_filter>
//! * Whittaker–Shannon interpolation:
//!   <https://en.wikipedia.org/wiki/Whittaker%E2%80%93Shannon_interpolation_formula>

use rayon::prelude::*;
use std::f64::consts::PI;

/// Zero crossings of the sinc kernel kept on each side of the output sample.
/// Longer kernels give a sharper transition band at higher cost.
/// See Smith, "Implementation": <https://ccrma.stanford.edu/~jos/resample/Implementation.html>
const SINC_ZERO_CROSSINGS: f64 = 24.0;

/// Kaiser window shape parameter. Kaiser's empirical formula
/// `A ≈ β / 0.1102 + 8.7` gives ≈ 87 dB of stop-band attenuation for β = 8.6.
/// See <https://en.wikipedia.org/wiki/Kaiser_window> and Oppenheim & Schafer,
/// *Discrete-Time Signal Processing*, §7.6.
const KAISER_BETA: f64 = 8.6;

/// Fraction of the output Nyquist frequency kept by the anti-aliasing
/// low-pass; the remaining 3 % is the filter's transition band.
/// See <https://en.wikipedia.org/wiki/Nyquist_frequency>
const PASSBAND_FRACTION: f64 = 0.97;

/// Relative size at which the Bessel power series is truncated.
const BESSEL_SERIES_TOLERANCE: f64 = 1e-12;

/// Zeroth-order modified Bessel function of the first kind, `I₀(x)`, summed
/// from its power series `Σ ((x/2)^k / k!)²`. Needed to evaluate the Kaiser
/// window. See <https://en.wikipedia.org/wiki/Bessel_function#Modified_Bessel_functions:_I%CE%B1,_K%CE%B1>
fn bessel_i0(argument: f64) -> f64 {
    let quarter_square = argument * argument / 4.0;
    let mut sum = 1.0;
    let mut term = 1.0;
    let mut order = 1.0;

    while term > BESSEL_SERIES_TOLERANCE * sum {
        term *= quarter_square / (order * order);
        sum += term;
        order += 1.0;
    }

    sum
}

/// Greatest common divisor (Euclid).
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Precomputed taps for every fractional input offset of one rate pair.
struct PolyphaseFilter {
    /// Distinct fractional offsets, `target / gcd(source, target)`.
    phases: u64,

    /// Input samples advanced per output sample, in units of `1 / phases`.
    step: u64,

    /// Taps on each side of the centre sample.
    radius: usize,

    /// `phases × (2·radius + 1)` taps, row `p` for offset `p / phases`.
    taps: Vec<f32>,
}

impl PolyphaseFilter {
    /// Designs the filter bank for `source_rate → target_rate`.
    fn new(source_rate: u32, target_rate: u32) -> Self {
        let divisor = gcd(source_rate as u64, target_rate as u64);
        let phases = target_rate as u64 / divisor;
        let step = source_rate as u64 / divisor;

        let cutoff = (target_rate as f64 / source_rate as f64).min(1.0) * PASSBAND_FRACTION;
        let radius = (SINC_ZERO_CROSSINGS / cutoff).ceil() as usize;
        let width = 2 * radius + 1;
        let window_normaliser = bessel_i0(KAISER_BETA);

        let mut taps = vec![0.0f32; phases as usize * width];
        for phase in 0..phases as usize {
            let fraction = phase as f64 / phases as f64;

            for tap in 0..width {
                let distance = (tap as f64 - radius as f64 - fraction) * cutoff;
                let sinc = if distance == 0.0 { 1.0 } else { (PI * distance).sin() / (PI * distance) };
                let window_position = distance / SINC_ZERO_CROSSINGS;
                let window = bessel_i0(KAISER_BETA * (1.0 - window_position * window_position).max(0.0).sqrt()) / window_normaliser;

                taps[phase * width + tap] = (sinc * window * cutoff) as f32;
            }
        }

        Self { phases, step, radius, taps }
    }

    /// Output sample `index` of `input`.
    fn sample(&self, input: &[f32], index: usize) -> f32 {
        let position = index as u64 * self.step;
        let centre = (position / self.phases) as isize;
        let phase = (position % self.phases) as usize;

        let width = 2 * self.radius + 1;
        let phase_taps = &self.taps[phase * width..(phase + 1) * width];
        let first = centre - self.radius as isize;

        let mut accumulator = 0.0f32;
        for (offset, &weight) in phase_taps.iter().enumerate() {
            let input_index = first + offset as isize;
            if let Some(&sample) = usize::try_from(input_index).ok().and_then(|index| input.get(index)) {
                accumulator += sample * weight;
            }
        }
        accumulator
    }
}

/// Converts `input` from `source_rate` to `target_rate` Hz.
///
/// When downsampling, the kernel is stretched so its cutoff sits below the
/// *output* Nyquist frequency, which prevents aliasing.
pub fn resample(input: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate || input.is_empty() {
        return input.to_vec();
    }

    let filter = PolyphaseFilter::new(source_rate, target_rate);
    let output_length = (input.len() as u64 * target_rate as u64 / source_rate as u64) as usize;

    (0..output_length).into_par_iter().map(|index| filter.sample(input, index)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit sine of `frequency` Hz sampled at `rate`. The phase is computed
    /// in f64: an f32 phase of ~10⁵ rad carries enough rounding noise to
    /// masquerade as aliasing at −46 dB.
    fn tone(frequency: f64, rate: u32, length: usize) -> Vec<f32> {
        (0..length).map(|index| (std::f64::consts::TAU * frequency * index as f64 / rate as f64).sin() as f32).collect()
    }

    #[test]
    fn in_band_sines_survive_common_rate_changes() {
        for (source_rate, target_rate) in [(44_100u32, 24_000u32), (48_000, 24_000), (22_050, 24_000)] {
            let frequency = 440.0;
            let output = resample(&tone(frequency, source_rate, source_rate as usize), source_rate, target_rate);
            let expected = tone(frequency, target_rate, output.len());

            assert_eq!(output.len(), target_rate as usize);

            let edge = 1000;
            let worst = (edge..output.len() - edge).map(|index| (output[index] - expected[index]).abs()).fold(0.0f32, f32::max);
            assert!(worst < 2e-3, "{source_rate} → {target_rate}: worst error {worst}");
        }
    }

    #[test]
    fn content_above_the_new_nyquist_is_removed() {
        let (source_rate, target_rate) = (48_000u32, 24_000u32);
        let output = resample(&tone(15_000.0, source_rate, source_rate as usize), source_rate, target_rate);

        let edge = 1000;
        let peak = output[edge..output.len() - edge].iter().fold(0.0f32, |peak, sample| peak.max(sample.abs()));
        // The designed stop-band attenuation at 15 kHz is ≈ −107 dB; allow −80 dB.
        assert!(peak < 1e-4, "aliased peak {peak}");
    }
}

#[cfg(test)]
mod edge_tests {
    use super::*;

    #[test]
    fn identity_and_empty_input_are_passthrough() {
        let input = vec![0.1, -0.2, 0.3];

        assert_eq!(resample(&input, 24_000, 24_000), input);
        assert!(resample(&[], 44_100, 24_000).is_empty());
    }

    #[test]
    fn output_length_matches_the_rate_ratio() {
        let input = vec![0.0f32; 1_000];

        assert_eq!(resample(&input, 48_000, 24_000).len(), 500);
        assert_eq!(resample(&input, 24_000, 48_000).len(), 2_000);
        assert_eq!(resample(&input, 44_100, 24_000).len(), 544);
    }

    #[test]
    fn a_constant_signal_is_preserved_in_the_interior() {
        let input = vec![0.25f32; 4_800];
        let output = resample(&input, 48_000, 24_000);
        let edge = 100;

        for sample in &output[edge..output.len() - edge] {
            assert!((sample - 0.25).abs() < 1e-3, "sample {sample}");
        }
    }

    #[test]
    fn one_sample_input_does_not_panic() {
        let output = resample(&[0.5], 48_000, 24_000);

        assert!(output.is_empty() || output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn gcd_and_bessel_helpers_have_the_expected_values() {
        assert_eq!(gcd(48_000, 24_000), 24_000);
        assert_eq!(gcd(44_100, 24_000), 300);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(5, 0), 5);

        assert_eq!(bessel_i0(0.0), 1.0);
        assert!(bessel_i0(1.0) > 1.0);
        assert!((bessel_i0(2.0) - bessel_i0(-2.0)).abs() < 1e-12);
    }

    #[test]
    fn polyphase_filter_has_the_expected_shape() {
        let down = PolyphaseFilter::new(48_000, 24_000);

        assert_eq!(down.phases, 1);
        assert_eq!(down.step, 2);
        assert_eq!(down.taps.len(), 2 * down.radius + 1);

        let ratio = PolyphaseFilter::new(44_100, 24_000);

        assert_eq!(ratio.phases, 80);
        assert_eq!(ratio.step, 147);
        assert_eq!(ratio.taps.len(), 80 * (2 * ratio.radius + 1));
    }

    #[test]
    fn sampling_an_empty_buffer_yields_zero() {
        let filter = PolyphaseFilter::new(48_000, 24_000);

        assert_eq!(filter.sample(&[], 0), 0.0);
    }

    #[test]
    fn upsampling_is_finite() {
        let input: Vec<f32> = (0..100).map(|index| (index as f32 * 0.1).sin()).collect();
        let output = resample(&input, 24_000, 48_000);

        assert_eq!(output.len(), 200);
        assert!(output.iter().all(|value| value.is_finite()));
    }
}
