//! The vocoder's kernels (overlap-add, predicted spectrum, spectral-loss
//! gradients, convolutions, column scales) against plain CPU loops, at sizes
//! that are not multiples of the block size. Gradients are checked against
//! the forward pass itself, by finite differences or as the adjoint of a
//! linear map, so that a wrong formula cannot pass.

use staleynet_cuda::{Convolution, Gpu, Shape, Tensor};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Relative tolerance for f32 results computed with different summation orders.
const TOLERANCE: f32 = 1e-4;

/// Relative tolerance for finite-difference checks, which lose precision to
/// cancellation.
const FINITE_DIFFERENCE_TOLERANCE: f64 = 2e-2;

/// Relative tolerance for the adjoint identity `<A x, g> = <x, Aᵀ g>`.
const ADJOINT_TOLERANCE: f64 = 1e-3;

/// Step of the central finite differences.
const FINITE_DIFFERENCE_STEP: f32 = 1e-3;

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

/// Deterministic values in roughly [-1, 1] (Knuth's MMIX LCG).
fn random_values(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);

    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Largest absolute difference divided by the largest expected magnitude.
fn relative_error(expected: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(expected.len(), actual.len());

    let magnitude = expected
        .iter()
        .fold(f32::EPSILON, |largest, value| largest.max(value.abs()));
    let worst = expected
        .iter()
        .zip(actual)
        .map(|(want, got)| (want - got).abs())
        .fold(0.0, f32::max);

    worst / magnitude
}

/// Copies `values` to the GPU as a `[rows, cols]` tensor.
fn upload(gpu: &Gpu, rows: usize, cols: usize, values: &[f32]) -> Tensor {
    gpu.upload(Shape::new(rows, cols), values).unwrap()
}

/// A `[rows, cols]` tensor of zeros on the GPU.
fn zeros(gpu: &Gpu, rows: usize, cols: usize) -> Tensor {
    gpu.zeros(Shape::new(rows, cols)).unwrap()
}

/// The inner product of two vectors, in f64.
fn dot(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(x, y)| *x as f64 * *y as f64)
        .sum()
}

/// Asserts the adjoint identity `<A x, g> = <x, Aᵀ g>` to [`ADJOINT_TOLERANCE`].
fn assert_adjoint(forward: f64, adjoint: f64, what: &str) {
    let allowed = ADJOINT_TOLERANCE * forward.abs().max(1.0);
    assert!((forward - adjoint).abs() < allowed, "{what}: {forward} vs {adjoint}");
}

// ---------------------------------------------------------------- overlap-add

/// One size of overlap-add to test.
struct OverlapCase {
    /// Signals joined side by side.
    sequences: usize,

    /// Frames per signal.
    frame_count: usize,

    /// Samples per frame.
    frame_length: usize,

    /// Samples between the starts of consecutive frames.
    hop: usize,
}

impl OverlapCase {
    /// Samples in one joined signal.
    fn length(&self) -> usize {
        (self.frame_count - 1) * self.hop + self.frame_length
    }
}

/// Overlap-add by plain loops over every frame sample.
fn overlap_add_reference(frames: &[f32], normalization: &[f32], case: &OverlapCase) -> Vec<f32> {
    let length = case.length();
    let mut signal = vec![0.0f32; case.sequences * length];

    for (row, values) in frames.chunks(case.frame_length).enumerate() {
        let sequence = row / case.frame_count;
        let frame = row % case.frame_count;

        for (offset, value) in values.iter().enumerate() {
            signal[sequence * length + frame * case.hop + offset] += value;
        }
    }

    signal
        .iter()
        .enumerate()
        .map(|(index, sum)| sum * normalization[index % length])
        .collect()
}

#[test]
fn overlap_add_matches_reference_and_its_gradient_is_the_adjoint() {
    let gpu = gpu();
    let cases = [
        OverlapCase { sequences: 1, frame_count: 1, frame_length: 8, hop: 2 },
        OverlapCase { sequences: 2, frame_count: 7, frame_length: 16, hop: 4 },
        OverlapCase { sequences: 3, frame_count: 13, frame_length: 37, hop: 9 },
        OverlapCase { sequences: 2, frame_count: 5, frame_length: 64, hop: 64 },
    ];

    for case in &cases {
        let length = case.length();
        let rows = case.sequences * case.frame_count;
        let frames = random_values(rows * case.frame_length, 1);
        let normalization: Vec<f32> = random_values(length, 2)
            .iter()
            .map(|value| 1.0 + value.abs())
            .collect();
        let signal_grad = random_values(case.sequences * length, 3);

        let frames_tensor = upload(&gpu, rows, case.frame_length, &frames);
        let normalization_tensor = upload(&gpu, 1, length, &normalization);
        let mut signal = zeros(&gpu, case.sequences, length);
        gpu.overlap_add(&frames_tensor, &normalization_tensor, &mut signal, case.frame_count, case.hop)
            .unwrap();

        let expected = overlap_add_reference(&frames, &normalization, case);
        let error = relative_error(&expected, &gpu.download(&signal).unwrap());
        assert!(error < TOLERANCE, "overlap_add {}x{}x{}/{}: {error}", case.sequences, case.frame_count, case.frame_length, case.hop);

        let grad_tensor = upload(&gpu, case.sequences, length, &signal_grad);
        let mut frames_grad = zeros(&gpu, rows, case.frame_length);
        gpu.overlap_add_backward(&grad_tensor, &normalization_tensor, &mut frames_grad, case.frame_count, case.hop)
            .unwrap();

        let forward = dot(&expected, &signal_grad);
        let adjoint = dot(&frames, &gpu.download(&frames_grad).unwrap());
        assert_adjoint(forward, adjoint, "overlap_add adjoint");
    }
}

// ----------------------------------------------------------- polar spectrum

#[test]
fn polar_spectrum_and_its_gradient_match_finite_differences() {
    let gpu = gpu();
    let frames = 3;
    let bins = 5;
    let max_log = 1.5f32;

    // The first log-magnitude lies above the cap, so its gradient must vanish.
    let mut head = random_values(frames * 2 * bins, 4);
    head[0] = 2.0;

    // The scalar loss L = Σ re · a + im · b has gradient (a, b) with respect to (re, im).
    let real_weights = random_values(frames * bins, 5);
    let imaginary_weights = random_values(frames * bins, 6);
    let loss = |head: &[f32]| -> f64 {
        let term = |element: usize| {
            let row = element / bins;
            let bin = element % bins;
            let magnitude = (head[row * 2 * bins + bin].min(max_log) as f64).exp();
            let phase = head[row * 2 * bins + bins + bin] as f64;

            magnitude * phase.cos() * real_weights[element] as f64 + magnitude * phase.sin() * imaginary_weights[element] as f64
        };

        (0..frames * bins).map(term).sum()
    };

    let head_tensor = upload(&gpu, frames, 2 * bins, &head);
    let mut real = zeros(&gpu, frames, bins);
    let mut imaginary = zeros(&gpu, frames, bins);
    gpu.polar_spectrum(&head_tensor, &mut real, &mut imaginary, max_log)
        .unwrap();

    let real_grad = upload(&gpu, frames, bins, &real_weights);
    let imaginary_grad = upload(&gpu, frames, bins, &imaginary_weights);
    let mut head_grad = zeros(&gpu, frames, 2 * bins);
    gpu.polar_spectrum_backward(&head_tensor, &real, &imaginary, &real_grad, &imaginary_grad, &mut head_grad, max_log)
        .unwrap();
    let analytic = gpu.download(&head_grad).unwrap();

    for index in 0..head.len() {
        let mut plus = head.clone();
        let mut minus = head.clone();
        plus[index] += FINITE_DIFFERENCE_STEP;
        minus[index] -= FINITE_DIFFERENCE_STEP;

        let numeric = (loss(&plus) - loss(&minus)) / (2.0 * FINITE_DIFFERENCE_STEP as f64);
        let difference = (numeric - analytic[index] as f64).abs();
        let allowed = FINITE_DIFFERENCE_TOLERANCE * numeric.abs().max(1e-2);
        assert!(difference < allowed, "head[{index}]: numeric {numeric}, analytic {}", analytic[index]);
    }

    assert_eq!(analytic[0], 0.0, "a capped log-magnitude receives no gradient");
}

// ------------------------------------------------------ spectral-loss gradients

#[test]
fn power_logarithm_and_absolute_error_gradients_match_their_formulas() {
    let gpu = gpu();
    let count = 1_025;
    let floor = 0.05f32;
    let real = random_values(count, 7);
    let imaginary = random_values(count, 8);
    let output_grad = random_values(count, 9);
    let energies: Vec<f32> = random_values(count, 10)
        .iter()
        .map(|value| value.abs())
        .collect();

    let real_tensor = upload(&gpu, 1, count, &real);
    let imaginary_tensor = upload(&gpu, 1, count, &imaginary);
    let grad_tensor = upload(&gpu, 1, count, &output_grad);

    let mut real_grad = zeros(&gpu, 1, count);
    let mut imaginary_grad = zeros(&gpu, 1, count);
    gpu.power_spectrum_backward(&real_tensor, &imaginary_tensor, &grad_tensor, &mut real_grad, &mut imaginary_grad)
        .unwrap();

    let expected_real: Vec<f32> = real
        .iter()
        .zip(&output_grad)
        .map(|(part, grad)| 2.0 * part * grad)
        .collect();
    let expected_imaginary: Vec<f32> = imaginary
        .iter()
        .zip(&output_grad)
        .map(|(part, grad)| 2.0 * part * grad)
        .collect();
    assert!(relative_error(&expected_real, &gpu.download(&real_grad).unwrap()) < TOLERANCE);
    assert!(relative_error(&expected_imaginary, &gpu.download(&imaginary_grad).unwrap()) < TOLERANCE);

    let energies_tensor = upload(&gpu, 1, count, &energies);
    let mut log_grad = zeros(&gpu, 1, count);
    gpu.log_compress_backward(&energies_tensor, &grad_tensor, &mut log_grad, floor)
        .unwrap();

    let expected_log: Vec<f32> = energies
        .iter()
        .zip(&output_grad)
        .map(|(energy, grad)| if *energy > floor { grad / energy } else { 0.0 })
        .collect();
    assert!(relative_error(&expected_log, &gpu.download(&log_grad).unwrap()) < TOLERANCE);

    let mut absolute_grad = zeros(&gpu, 1, count);
    gpu.absolute_error_gradient(&real_tensor, &imaginary_tensor, &mut absolute_grad, 0.25)
        .unwrap();

    let expected_absolute: Vec<f32> = real
        .iter()
        .zip(&imaginary)
        .map(|(estimate, target)| 0.25 * (estimate - target).signum())
        .collect();
    assert_eq!(gpu.download(&absolute_grad).unwrap(), expected_absolute);
}

// ----------------------------------------------------------- convolutions

/// A depthwise convolution by plain loops over every tap.
fn depthwise_reference(input: &[f32], weight: &[f32], bias: &[f32], shape: Convolution) -> Vec<f32> {
    let output = |element: usize| {
        let row = element / shape.channels;
        let channel = element % shape.channels;

        let mut sum = bias[channel];
        for tap in 0..shape.kernel {
            if let Some(source) = shape.source(row, tap) {
                sum += weight[tap * shape.channels + channel] * input[source * shape.channels + channel];
            }
        }

        sum
    };

    (0..shape.rows * shape.channels).map(output).collect()
}

#[test]
fn depthwise_convolution_and_its_gradients_match_reference() {
    let gpu = gpu();
    let shape = Convolution { rows: 3 * 11, time: 11, channels: 9, kernel: 7 };
    let input = random_values(shape.rows * shape.channels, 11);
    let weight = random_values(shape.kernel * shape.channels, 12);
    let bias = random_values(shape.channels, 13);
    let output_grad = random_values(shape.rows * shape.channels, 14);

    let input_tensor = upload(&gpu, shape.rows, shape.channels, &input);
    let weight_tensor = upload(&gpu, shape.kernel, shape.channels, &weight);
    let bias_tensor = upload(&gpu, 1, shape.channels, &bias);
    let mut output = zeros(&gpu, shape.rows, shape.channels);
    gpu.depthwise_convolution(&input_tensor, &weight_tensor, &bias_tensor, &mut output, shape)
        .unwrap();

    let expected = depthwise_reference(&input, &weight, &bias, shape);
    assert!(relative_error(&expected, &gpu.download(&output).unwrap()) < TOLERANCE, "forward");

    // Without its bias the convolution is linear in its input, and the input
    // gradient must be its adjoint.
    let grad_tensor = upload(&gpu, shape.rows, shape.channels, &output_grad);
    let mut input_grad = zeros(&gpu, shape.rows, shape.channels);
    gpu.depthwise_convolution_input_grad(&grad_tensor, &weight_tensor, &mut input_grad, shape)
        .unwrap();

    let zero_bias = vec![0.0; shape.channels];
    let forward = dot(&depthwise_reference(&input, &weight, &zero_bias, shape), &output_grad);
    let by_input = dot(&input, &gpu.download(&input_grad).unwrap());
    assert_adjoint(forward, by_input, "input gradient");

    // It is linear in its weights too, so the same identity checks the weight gradient.
    let mut weight_grad = zeros(&gpu, shape.kernel, shape.channels);
    gpu.depthwise_convolution_weight_grad(&grad_tensor, &input_tensor, &mut weight_grad, shape)
        .unwrap();

    let by_weights = dot(&weight, &gpu.download(&weight_grad).unwrap());
    assert_adjoint(forward, by_weights, "weight gradient");
}

#[test]
fn unfold_lays_out_each_neighborhood_and_respects_sequence_edges() {
    let gpu = gpu();
    let shape = Convolution { rows: 2 * 6, time: 6, channels: 4, kernel: 3 };
    let width = shape.kernel * shape.channels;
    let input = random_values(shape.rows * shape.channels, 15);

    let input_tensor = upload(&gpu, shape.rows, shape.channels, &input);
    let mut columns = zeros(&gpu, shape.rows, width);
    gpu.unfold(&input_tensor, &mut columns, shape).unwrap();

    let column = |element: usize| {
        let row = element / width;
        let tap = (element % width) / shape.channels;
        let channel = element % shape.channels;

        shape
            .source(row, tap)
            .map_or(0.0, |source| input[source * shape.channels + channel])
    };

    let expected: Vec<f32> = (0..shape.rows * width).map(column).collect();
    assert_eq!(gpu.download(&columns).unwrap(), expected);
}

// ----------------------------------------------------------- column scales

#[test]
fn column_scaling_and_its_gradient_match_reference() {
    let gpu = gpu();
    let rows = 37;
    let cols = 70;
    let values = random_values(rows * cols, 16);
    let scale = random_values(cols, 17);
    let output_grad = random_values(rows * cols, 18);

    let values_tensor = upload(&gpu, rows, cols, &values);
    let scale_tensor = upload(&gpu, 1, cols, &scale);
    let mut output = zeros(&gpu, rows, cols);
    gpu.scale_columns(&values_tensor, &scale_tensor, &mut output)
        .unwrap();

    let expected: Vec<f32> = values
        .iter()
        .enumerate()
        .map(|(index, value)| value * scale[index % cols])
        .collect();
    assert!(relative_error(&expected, &gpu.download(&output).unwrap()) < TOLERANCE);

    let grad_tensor = upload(&gpu, rows, cols, &output_grad);
    let mut scale_grad = zeros(&gpu, 1, cols);
    gpu.column_products(&grad_tensor, &values_tensor, &mut scale_grad)
        .unwrap();

    let column_sum = |col: usize| {
        (0..rows)
            .map(|row| output_grad[row * cols + col] * values[row * cols + col])
            .sum()
    };
    let expected_grad: Vec<f32> = (0..cols).map(column_sum).collect();
    assert!(relative_error(&expected_grad, &gpu.download(&scale_grad).unwrap()) < TOLERANCE);
}
