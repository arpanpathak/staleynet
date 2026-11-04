//! The GPU feature path and tokenizer against their CPU definitions.

use staleynet_audio::{MelAnalyzer, MelConfig};
use staleynet_cuda::Gpu;
use staleynet_data::features::GpuMelExtractor;
use staleynet_data::store::Normalization;
use staleynet_data::tokenizer::{AudioTokenizer, TokenizerConfig};
use staleynet_model::Random;

/// Log-mel values below this are dominated by the energy floor, where tiny
/// absolute differences in power become large differences in the logarithm.
const AUDIBLE_LOG_ENERGY: f32 = -8.0;

#[test]
fn gpu_log_mel_matches_the_fft_reference() {
    let gpu = Gpu::new(0).unwrap();
    let analyzer = MelAnalyzer::new(MelConfig::default());
    let signal: Vec<f32> = (0..100_000).map(|index| ((index as f32 * 0.031).sin() + 0.3 * (index as f32 * 0.17).cos()) * 0.5).collect();

    let expected = analyzer.log_mel(&signal);
    let actual = GpuMelExtractor::new(&gpu, &analyzer).unwrap().log_mel(&gpu, &analyzer, &signal).unwrap();

    assert_eq!(expected.len(), actual.len());
    let worst = expected
        .iter()
        .zip(&actual)
        .filter(|(reference, _)| **reference > AUDIBLE_LOG_ENERGY)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(worst < 1e-2, "worst log-mel difference {worst}");
}

/// Mean squared reconstruction error of random features after decoding with
/// 1, 2 and 3 residual levels, for a tokenizer with `band_groups` groups.
fn errors_by_level(band_groups: usize) -> Vec<f32> {
    let gpu = Gpu::new(0).unwrap();
    let mel_bands = 8;
    let levels = 3;
    let mut random = Random::new(3);
    let log_mel: Vec<f32> = (0..4000 * mel_bands).map(|_| random.normal()).collect();
    let normalization = Normalization { mean: vec![0.0; mel_bands], std: vec![1.0; mel_bands] };
    let config = TokenizerConfig {
        frames_per_step: 2,
        codebooks: levels * band_groups,
        band_groups,
        codebook_size: 32,
        iterations: 8,
        max_training_vectors: 10_000,
        seed: 1,
    };

    let vectors = AudioTokenizer::step_vectors(&log_mel, mel_bands, config.frames_per_step, &normalization);
    let tokenizer = AudioTokenizer::train(&gpu, config, mel_bands, normalization, vectors, |_| {}).unwrap();
    let codes = tokenizer.encode(&gpu, &log_mel).unwrap();

    (1..=levels)
        .map(|used| {
            let reconstruction = tokenizer.decode(&codes, used);
            reconstruction.iter().zip(&log_mel).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / log_mel.len() as f32
        })
        .collect()
}

#[test]
fn each_residual_level_reduces_reconstruction_error() {
    for band_groups in [1, 2] {
        let errors = errors_by_level(band_groups);
        assert!(errors[0] < 1.0 && errors[1] < errors[0] && errors[2] < errors[1], "{band_groups} groups: {errors:?}");
    }
}

#[test]
fn splitting_the_bands_lowers_the_error() {
    let whole = errors_by_level(1);
    let split = errors_by_level(2);
    assert!(split[2] < whole[2], "split {split:?} against whole {whole:?}");
}
