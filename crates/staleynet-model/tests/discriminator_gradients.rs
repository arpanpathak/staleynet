//! The discriminator's backward pass against central finite differences of
//! its own losses: the discriminator's gradient with respect to its weights,
//! and the vocoder's adversarial gradient with respect to the synthesized
//! samples. Run on a small discriminator with sizes that do not divide evenly.

use staleynet_cuda::{Gpu, Shape, Tensor};
use staleynet_model::Random;
use staleynet_model::parameter::{Parameter, Parameters};
use staleynet_model::vocoder::adversarial::{self, AdversarialWeights, Verdicts};
use staleynet_model::vocoder::{DiscriminatorConfig, LayerSize, MultiPeriodDiscriminator};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Signals in the test batch.
const SIGNALS: usize = 2;

/// Samples per signal, not a multiple of any period.
const LENGTH: usize = 47;

/// Step of the central finite differences for the weights.
const STEP: f32 = 1e-2;

/// Step for the samples: smaller, so that it rarely crosses a kink of the
/// absolute value in feature matching, where the derivative jumps.
const SAMPLE_STEP: f32 = 1e-4;

/// Relative tolerance of a finite-difference check.
const RELATIVE_TOLERANCE: f32 = 0.05;

/// Absolute tolerance, for gradients too small to compare relatively.
const ABSOLUTE_TOLERANCE: f32 = 3e-4;

/// Samples and weights checked per test.
const CHECKS: usize = 24;

/// One GPU shared by all tests; the harness runs tests on several threads.
fn gpu() -> MutexGuard<'static, Gpu> {
    static GPU: OnceLock<Mutex<Gpu>> = OnceLock::new();
    let guard = GPU
        .get_or_init(|| Mutex::new(Gpu::new(0).expect("CUDA device 0")))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    guard.bind_to_current_thread().expect("bind CUDA context");
    guard
}

/// A discriminator with two periods and two small layers before the scores.
fn small_config() -> DiscriminatorConfig {
    DiscriminatorConfig { periods: vec![2, 3], layers: vec![LayerSize { channels: 4, stride: 2 }, LayerSize { channels: 6, stride: 1 }] }
}

/// `SIGNALS` random signals of `LENGTH` samples on the device.
fn random_signals(gpu: &Gpu, random: &mut Random) -> Tensor {
    let values: Vec<f32> = (0..SIGNALS * LENGTH).map(|_| random.normal()).collect();

    let mut tensor = gpu.zeros(Shape::new(SIGNALS, LENGTH)).unwrap();
    gpu.copy_from_host(&mut tensor, &values).unwrap();
    tensor
}

/// Asserts that an analytic and a numeric derivative agree.
fn assert_close(analytic: f32, numeric: f32, what: &str) {
    let difference = (analytic - numeric).abs();
    let relative = difference / (analytic.abs() + numeric.abs()).max(1e-6);

    assert!(relative < RELATIVE_TOLERANCE || difference < ABSOLUTE_TOLERANCE, "{what}: analytic {analytic} vs numeric {numeric}");
}

/// The discriminator's loss on `real` and `synthesized`, judged afresh.
fn discriminator_loss(
    gpu: &Gpu,
    discriminator: &MultiPeriodDiscriminator,
    batches: &mut (Verdicts, Verdicts),
    signals: (&Tensor, &Tensor),
) -> f32 {
    let (real, synthesized) = batches;
    real.judge(gpu, discriminator, signals.0).unwrap();
    synthesized.judge(gpu, discriminator, signals.1).unwrap();

    adversarial::losses(gpu, real, synthesized, AdversarialWeights::default())
        .unwrap()
        .discriminator as f32
}

/// The vocoder's adversarial and feature-matching terms, judged afresh.
fn generator_loss(
    gpu: &Gpu,
    discriminator: &MultiPeriodDiscriminator,
    batches: &mut (Verdicts, Verdicts),
    signals: (&Tensor, &Tensor),
) -> f32 {
    let (real, synthesized) = batches;
    real.judge(gpu, discriminator, signals.0).unwrap();
    synthesized.judge(gpu, discriminator, signals.1).unwrap();

    let losses = adversarial::losses(gpu, real, synthesized, AdversarialWeights::default()).unwrap();
    (losses.adversarial + losses.feature_matching) as f32
}

/// Applies `edit` to element `element` of the parameter at visitor index `index`.
fn edit_parameter(gpu: &Gpu, discriminator: &mut MultiPeriodDiscriminator, index: usize, element: usize, edit: impl Fn(f32) -> f32) {
    let mut visited = 0;
    discriminator.visit_parameters(&mut |parameter: &mut Parameter| {
        if visited == index {
            let mut values = gpu.download(&parameter.value).unwrap();
            values[element] = edit(values[element]);
            gpu.copy_from_host(&mut parameter.value, &values).unwrap();
        }
        visited += 1;
    });
}

#[test]
fn the_discriminator_gradient_matches_finite_differences() {
    let gpu = gpu();
    let mut random = Random::new(5);
    let mut discriminator = MultiPeriodDiscriminator::new(&gpu, &small_config(), &mut random).unwrap();
    let real = random_signals(&gpu, &mut random);
    let synthesized = random_signals(&gpu, &mut random);

    let mut batches =
        (Verdicts::new(&gpu, &discriminator, SIGNALS, LENGTH).unwrap(), Verdicts::new(&gpu, &discriminator, SIGNALS, LENGTH).unwrap());
    discriminator_loss(&gpu, &discriminator, &mut batches, (&real, &synthesized));
    discriminator.zero_gradients(&gpu).unwrap();
    adversarial::discriminator_backward(&gpu, &mut discriminator, &mut batches.0, &mut batches.1).unwrap();

    let mut gradients: Vec<(String, Vec<f32>)> = Vec::new();
    discriminator.visit_parameters(&mut |parameter| gradients.push((parameter.name.clone(), gpu.download(&parameter.gradient).unwrap())));

    for check in 0..CHECKS {
        let index = random.below(gradients.len());
        let (name, gradient) = &gradients[index];
        let element = random.below(gradient.len());

        edit_parameter(&gpu, &mut discriminator, index, element, |value| value + STEP);
        let loss_up = discriminator_loss(&gpu, &discriminator, &mut batches, (&real, &synthesized));
        edit_parameter(&gpu, &mut discriminator, index, element, |value| value - 2.0 * STEP);
        let loss_down = discriminator_loss(&gpu, &discriminator, &mut batches, (&real, &synthesized));
        edit_parameter(&gpu, &mut discriminator, index, element, |value| value + STEP);

        let numeric = (loss_up - loss_down) / (2.0 * STEP);
        assert_close(gradient[element], numeric, &format!("check {check}: {name}[{element}]"));
    }
}

#[test]
fn the_adversarial_gradient_of_the_samples_matches_finite_differences() {
    let gpu = gpu();
    let mut random = Random::new(6);
    let mut discriminator = MultiPeriodDiscriminator::new(&gpu, &small_config(), &mut random).unwrap();
    let real = random_signals(&gpu, &mut random);
    let mut synthesized = random_signals(&gpu, &mut random);

    let mut batches =
        (Verdicts::new(&gpu, &discriminator, SIGNALS, LENGTH).unwrap(), Verdicts::new(&gpu, &discriminator, SIGNALS, LENGTH).unwrap());
    generator_loss(&gpu, &discriminator, &mut batches, (&real, &synthesized));

    let mut samples_grad = gpu.zeros(Shape::new(SIGNALS, LENGTH)).unwrap();
    let weights = AdversarialWeights::default();
    adversarial::generator_backward(&gpu, &mut discriminator, &batches.0, &mut batches.1, weights, &mut samples_grad).unwrap();
    let analytic = gpu.download(&samples_grad).unwrap();

    let mut samples = gpu.download(&synthesized).unwrap();
    for check in 0..CHECKS {
        let element = random.below(samples.len());
        let original = samples[element];

        samples[element] = original + SAMPLE_STEP;
        gpu.copy_from_host(&mut synthesized, &samples).unwrap();
        let loss_up = generator_loss(&gpu, &discriminator, &mut batches, (&real, &synthesized));

        samples[element] = original - SAMPLE_STEP;
        gpu.copy_from_host(&mut synthesized, &samples).unwrap();
        let loss_down = generator_loss(&gpu, &discriminator, &mut batches, (&real, &synthesized));

        samples[element] = original;
        let numeric = (loss_up - loss_down) / (2.0 * SAMPLE_STEP);
        assert_close(analytic[element], numeric, &format!("check {check}: sample {element}"));
    }
}
