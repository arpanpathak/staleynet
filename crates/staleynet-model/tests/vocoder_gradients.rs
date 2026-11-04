//! The vocoder's whole gradient (network, spectrum, inverse STFT and the three
//! spectral losses) against finite differences of the loss, on a small
//! configuration. A wrong derivation anywhere along the path changes the
//! numbers here.

use staleynet_cuda::Gpu;
use staleynet_model::parameter::Parameters;
use staleynet_model::vocoder::{Resolution, TrainingPass, Vocoder, VocoderConfig, checkpoint};

/// Perturbations applied to one parameter value. Large steps suffer from the
/// curvature of the network and of the logarithm, small ones from f32 rounding
/// in the loss; a derivative is accepted when the central difference at any of
/// these steps agrees with it. A wrong derivation agrees at none of them.
const STEPS: [f32; 3] = [1e-2, 3e-3, 1e-3];

/// Largest accepted relative difference between the analytic and the numeric derivative.
const TOLERANCE: f64 = 0.05;

/// Smallest derivative magnitude a relative difference is measured against.
const DERIVATIVE_FLOOR: f64 = 1e-4;

/// Sequences in the batch.
const SEQUENCES: usize = 2;

/// Frames per sequence.
const TIME: usize = 8;

/// Amplitude of the target signals. Far louder than the untrained network's
/// output, so few terms of the absolute error change sign under a perturbation
/// and the loss stays close to smooth around the test point.
const TARGET_AMPLITUDE: f32 = 30.0;

/// Training step recorded in the round-trip checkpoint.
const SAVED_STEP: usize = 42;

/// One parameter value whose derivative is checked.
struct Probe {
    /// Position of the parameter in visitor order.
    parameter: usize,

    /// Index of the value within the parameter.
    index: usize,
}

/// One probe per kind of parameter. In visitor order: input weight 0 and bias
/// 1; block 0 depthwise weight 2, depthwise bias 3, norm gain 4, expand weight
/// 5 and bias 6, project weight 7 and bias 8, scale 9; block 1 at 10..=17;
/// final norm gain 18; head weight 19 and bias 20.
const PROBES: [Probe; 14] = [
    Probe { parameter: 0, index: 5 },
    Probe { parameter: 1, index: 3 },
    Probe { parameter: 2, index: 7 },
    Probe { parameter: 3, index: 2 },
    Probe { parameter: 4, index: 1 },
    Probe { parameter: 5, index: 11 },
    Probe { parameter: 7, index: 4 },
    Probe { parameter: 9, index: 6 },
    Probe { parameter: 13, index: 9 },
    Probe { parameter: 17, index: 2 },
    Probe { parameter: 18, index: 3 },
    Probe { parameter: 19, index: 17 },
    Probe { parameter: 19, index: 140 },
    Probe { parameter: 20, index: 40 },
];

/// A configuration small enough for finite differences.
fn config() -> VocoderConfig {
    VocoderConfig {
        sample_rate: 4_000,
        mel_bands: 8,
        fft_size: 64,
        hop_size: 16,
        width: 12,
        layers: 2,
        taps: 3,
        expansion: 2,
        max_log_magnitude: 4.6,
    }
}

/// Analyses that divide a signal of (TIME − 1) · 16 + 64 samples exactly.
fn resolutions() -> Vec<Resolution> {
    vec![
        Resolution { fft_size: 64, hop_size: 16, mel: true, weight: 1.0 },
        Resolution { fft_size: 32, hop_size: 8, mel: false, weight: 0.5 },
        Resolution { fft_size: 128, hop_size: 16, mel: false, weight: 0.5 },
    ]
}

/// Deterministic values in roughly [-1, 1].
fn random_values(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);

    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// The value of `probe`, replaced by `replacement` when one is given.
fn probe_value(gpu: &Gpu, vocoder: &mut Vocoder, probe: &Probe, replacement: Option<f32>) -> f32 {
    let mut position = 0;
    let mut found = 0.0;

    vocoder.network.visit_parameters(&mut |parameter| {
        if position == probe.parameter {
            let mut values = gpu.download(&parameter.value).unwrap();
            found = values[probe.index];

            if let Some(value) = replacement {
                values[probe.index] = value;
                gpu.copy_from_host(&mut parameter.value, &values).unwrap();
            }
        }

        position += 1;
    });

    found
}

/// The gradient at `probe`.
fn probe_gradient(gpu: &Gpu, vocoder: &mut Vocoder, probe: &Probe) -> f32 {
    let mut position = 0;
    let mut found = 0.0;

    vocoder.network.visit_parameters(&mut |parameter| {
        if position == probe.parameter {
            found = gpu.download(&parameter.gradient).unwrap()[probe.index];
        }

        position += 1;
    });

    found
}

#[test]
fn every_kind_of_parameter_matches_finite_differences() {
    let gpu = Gpu::new(0).expect("CUDA device 0");
    let config = config();
    let mean = vec![0.0; config.mel_bands];
    let std = vec![1.0; config.mel_bands];
    let mut vocoder = Vocoder::new(&gpu, config.clone(), mean, std, 3).unwrap();
    let mut pass = TrainingPass::new(&gpu, &vocoder, SEQUENCES, TIME, &resolutions()).unwrap();

    let mel = random_values(SEQUENCES * TIME * config.mel_bands, 1);
    let target: Vec<f32> = random_values(SEQUENCES * pass.signal_length(), 2)
        .iter()
        .map(|value| TARGET_AMPLITUDE * value)
        .collect();

    vocoder.network.zero_gradients(&gpu).unwrap();
    pass.run(&gpu, &mut vocoder, &mel, &target).unwrap();

    let analytic: Vec<f32> = PROBES
        .iter()
        .map(|probe| probe_gradient(&gpu, &mut vocoder, probe))
        .collect();

    for (probe, analytic) in PROBES.iter().zip(analytic) {
        let original = probe_value(&gpu, &mut vocoder, probe, None);

        let mut loss_at = |value: f32| {
            probe_value(&gpu, &mut vocoder, probe, Some(value));
            pass.run(&gpu, &mut vocoder, &mel, &target).unwrap();
            pass.loss(&gpu).unwrap().total
        };
        let numerics: Vec<f64> = STEPS
            .iter()
            .map(|&step| (loss_at(original + step) - loss_at(original - step)) / (2.0 * step as f64))
            .collect();

        probe_value(&gpu, &mut vocoder, probe, Some(original));

        let relative_error = |numeric: f64| {
            let scale = numeric
                .abs()
                .max(analytic.abs() as f64)
                .max(DERIVATIVE_FLOOR);
            (numeric - analytic as f64).abs() / scale
        };
        let best = numerics
            .iter()
            .map(|&numeric| relative_error(numeric))
            .reduce(f64::min)
            .unwrap_or(f64::INFINITY);

        assert!(best < TOLERANCE, "parameter {}[{}]: numeric {numerics:?}, analytic {analytic:.6e}", probe.parameter, probe.index);
    }
}

#[test]
fn a_checkpoint_round_trip_restores_the_network() {
    let gpu = Gpu::new(0).expect("CUDA device 0");
    let config = config();
    let mean = vec![0.5; config.mel_bands];
    let std = vec![2.0; config.mel_bands];
    let mut vocoder = Vocoder::new(&gpu, config.clone(), mean, std, 9).unwrap();

    let mel = random_values(TIME * config.mel_bands, 4);
    let before = vocoder.synthesize(&gpu, &mel).unwrap();

    let directory = std::env::temp_dir().join(format!("staleynet-vocoder-checkpoint-{}", std::process::id()));
    checkpoint::save(&gpu, &directory, &mut vocoder, None, SAVED_STEP).unwrap();
    let restored = checkpoint::load(&gpu, &directory).unwrap();

    assert_eq!(restored.step, SAVED_STEP);
    assert_eq!(restored.vocoder.synthesize(&gpu, &mel).unwrap(), before);
    assert_eq!(before.len(), (TIME - 1) * config.hop_size);

    std::fs::remove_dir_all(directory).unwrap();
}
