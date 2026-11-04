//! A small seeded random number generator, so initialisation, batch sampling
//! and generation are reproducible without external dependencies.

/// SplitMix64 (Steele, Lea & Flood, 2014, <https://doi.org/10.1145/2714064.2660195>).
/// Passes BigCrush and needs only 64 bits of state.
#[derive(Clone, Debug)]
pub struct Random {
    /// Generator state; advanced by the golden-ratio increment on every draw.
    state: u64,
}

/// 2⁶⁴ / φ, the SplitMix64 state increment.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// First SplitMix64 output multiplier.
const MIX_MULTIPLIER_1: u64 = 0xBF58_476D_1CE4_E5B9;

/// Second SplitMix64 output multiplier.
const MIX_MULTIPLIER_2: u64 = 0x94D0_49BB_1331_11EB;

/// Mantissa bits of an `f32`, used to build uniform floats from integers.
const F32_MANTISSA_BITS: u32 = 24;

/// Smallest uniform value fed to the logarithm in Box–Muller.
const MIN_UNIFORM: f32 = 1e-7;

impl Random {
    /// A generator whose whole stream is determined by `seed`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// A generator for sub-stream `stream` of `seed` (e.g. one per training step).
    pub fn derived(seed: u64, stream: u64) -> Self {
        let mut parent = Self::new(seed ^ stream.wrapping_mul(GOLDEN_GAMMA));
        Self::new(parent.next_u64())
    }

    /// Next 64 uniformly distributed bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);

        let mut mixed = self.state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(MIX_MULTIPLIER_1);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(MIX_MULTIPLIER_2);
        mixed ^ (mixed >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> (64 - F32_MANTISSA_BITS)) as f32 / (1u64 << F32_MANTISSA_BITS) as f32
    }

    /// Uniform integer in `0..bound` (bound > 0).
    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }

    /// True with probability `probability`.
    pub fn chance(&mut self, probability: f32) -> bool {
        self.uniform() < probability
    }

    /// Standard normal sample by the Box–Muller transform
    /// (<https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform>).
    pub fn normal(&mut self) -> f32 {
        let radius = (-2.0 * self.uniform().max(MIN_UNIFORM).ln()).sqrt();
        let angle = std::f32::consts::TAU * self.uniform();
        radius * angle.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_repeats_the_same_stream() {
        let mut first = Random::new(42);
        let mut second = Random::new(42);

        assert_eq!(first.next_u64(), second.next_u64());
        assert_eq!(first.uniform(), second.uniform());
        assert_eq!(first.normal(), second.normal());
    }

    #[test]
    fn different_seeds_diverge() {
        let mut first = Random::new(1);
        let mut second = Random::new(2);

        assert_ne!(first.next_u64(), second.next_u64());
    }

    #[test]
    fn derived_streams_are_deterministic_and_separate() {
        let mut first = Random::derived(7, 1);
        let mut second = Random::derived(7, 1);
        let mut other = Random::derived(7, 2);

        assert_eq!(first.next_u64(), second.next_u64());
        assert_ne!(first.next_u64(), other.next_u64());
    }

    #[test]
    fn uniform_stays_in_the_unit_interval_and_varies() {
        let mut random = Random::new(3);
        let values: Vec<f32> = (0..1_000).map(|_| random.uniform()).collect();

        assert!(values.iter().all(|value| (0.0..1.0).contains(value)));
        assert!(values.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn below_stays_inside_the_bound() {
        let mut random = Random::new(4);
        let values: Vec<usize> = (0..1_000).map(|_| random.below(10)).collect();

        assert!(values.iter().all(|value| *value < 10));
        assert!(values.contains(&0));
    }

    #[test]
    fn chance_respects_its_bounds() {
        let mut random = Random::new(5);

        assert!(!random.chance(0.0));
        assert!(random.chance(1.0));
    }

    #[test]
    fn normal_samples_are_finite_and_roughly_standard() {
        let mut random = Random::new(6);
        let values: Vec<f32> = (0..2_000).map(|_| random.normal()).collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / values.len() as f32;

        assert!(values.iter().all(|value| value.is_finite()));
        assert!(mean.abs() < 0.2, "mean {mean}");
        assert!((variance - 1.0).abs() < 0.2, "variance {variance}");
    }
}
