//! The vocoder kernels against plain CPU loops, at sizes that are not
//! multiples of the block size. Gradients are checked twice: against the
//! formula, and against the forward pass itself (by finite differences, or as
//! the adjoint of a linear map), so that a wrong formula cannot pass.

use staleynet_cuda::{Convolution, Gpu, Shape, Tensor};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Relative tolerance for f32 results computed with different summation orders.
const TOLERANCE: f32 = 1e-4;

/// Relative tolerance for finite-difference checks, which lose precision to
/// cancellation.
const FINITE_DIFFERENCE_TOLERANCE: f64 = 2e-2;

/// One GPU shared by all tests; the harness runs tests on several threads.
fn gpu() -> MutexGuard<'static, Gpu> {
    static GPU: OnceLock<Mutex<Gpu>> = OnceLock::new();
    let guard = GPU.get_or_init(|| Mutex::new(Gpu::new(0).expect("CUDA device 0"))).lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.bind_to_current_thread().expect("bind CUDA context");
    guard
}

/// Deterministic values in roughly [-1, 1] (Knuth's MMIX LCG).
fn random_values(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Largest absolute difference divided by the largest expected magnitude.
fn relative_error(expected: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(expected.len(), actual.len());
    let magnitude = expected.iter().fold(f32::EPSILON, |largest, value| largest.max(value.abs()));
    let worst = expected.iter().zip(actual).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    worst / magnitude
}

fn upload(gpu: &Gpu, rows: usize, cols: usize, values: &[f32]) -> Tensor {
    gpu.upload(Shape::new(rows, cols), values).unwrap()
}

fn zeros(gpu: &Gpu, rows: usize, cols: usize) -> Tensor {
    gpu.zeros(Shape::new(rows, cols)).unwrap()
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

// ---------------------------------------------------------------- overlap-add

fn overlap_add_reference(frames: &[f32], normalization: &[f32], frames_per_sequence: usize, frame_length: usize, hop: usize) -> Vec<f32> {
    let length = normalization.len();
    let sequences = frames.len() / (frames_per_sequence * frame_length);
    let mut signal = vec![0.0f32; sequences * length];

    for (row, values) in frames.chunks(frame_length).enumerate() {
        let (sequence, frame) = (row / frames_per_sequence, row % frames_per_sequence);
        for (offset, value) in values.iter().enumerate() {
            signal[sequence * length + frame * hop + offset] += value;
        }
    }
    signal.iter().enumerate().map(|(index, sum)| sum * normalization[index % length]).collect()
}

#[test]
fn overlap_add_matches_reference_and_its_gradient_is_the_adjoint() {
    let gpu = gpu();
    for (sequences, frame_count, frame_length, hop) in [(1, 1, 8, 2), (2, 7, 16, 4), (3, 13, 37, 9), (2, 5, 64, 64)] {
        let length = (frame_count - 1) * hop + frame_length;
        let rows = sequences * frame_count;
        let frames = random_values(rows * frame_length, 1);
        let normalization: Vec<f32> = random_values(length, 2).iter().map(|value| 1.0 + value.abs()).collect();
        let signal_grad = random_values(sequences * length, 3);

        let frames_tensor = upload(&gpu, rows, frame_length, &frames);
        let normalization_tensor = upload(&gpu, 1, length, &normalization);
        let mut signal = zeros(&gpu, sequences, length);
        gpu.overlap_add(&frames_tensor, &normalization_tensor, &mut signal, frame_count, hop).unwrap();

        let expected = overlap_add_reference(&frames, &normalization, frame_count, frame_length, hop);
        let error = relative_error(&expected, &gpu.download(&signal).unwrap());
        assert!(error < TOLERANCE, "overlap_add {sequences}x{frame_count}x{frame_length}/{hop}: {error}");

        // A linear map's gradient is its adjoint: <A x, g> = <x, Aᵀ g>.
        let grad_tensor = upload(&gpu, sequences, length, &signal_grad);
        let mut frames_grad = zeros(&gpu, rows, frame_length);
        gpu.overlap_add_backward(&grad_tensor, &normalization_tensor, &mut frames_grad, frame_count, hop).unwrap();
        let forward = dot(&expected, &signal_grad);
        let adjoint = dot(&frames, &gpu.download(&frames_grad).unwrap());
        assert!((forward - adjoint).abs() < 1e-3 * forward.abs().max(1.0), "overlap_add adjoint: {forward} vs {adjoint}");
    }
}

// ----------------------------------------------------------- polar spectrum

#[test]
fn polar_spectrum_and_its_gradient_match_finite_differences() {
    let gpu = gpu();
    let (frames, bins, max_log) = (3, 5, 1.5f32);
    let mut head = random_values(frames * 2 * bins, 4);
    head[0] = 2.0; // above the cap: its gradient must vanish
    let weights_re = random_values(frames * bins, 5);
    let weights_im = random_values(frames * bins, 6);

    // Scalar loss L = Σ re · a + im · b, whose gradient is (a, b) with respect to (re, im).
    let loss = |head: &[f32]| -> f64 {
        (0..frames * bins)
            .map(|element| {
                let (row, bin) = (element / bins, element % bins);
                let magnitude = (head[row * 2 * bins + bin].min(max_log) as f64).exp();
                let phase = head[row * 2 * bins + bins + bin] as f64;
                magnitude * phase.cos() * weights_re[element] as f64 + magnitude * phase.sin() * weights_im[element] as f64
            })
            .sum()
    };

    let head_tensor = upload(&gpu, frames, 2 * bins, &head);
    let mut real = zeros(&gpu, frames, bins);
    let mut imaginary = zeros(&gpu, frames, bins);
    gpu.polar_spectrum(&head_tensor, &mut real, &mut imaginary, max_log).unwrap();

    let real_grad = upload(&gpu, frames, bins, &weights_re);
    let imaginary_grad = upload(&gpu, frames, bins, &weights_im);
    let mut head_grad = zeros(&gpu, frames, 2 * bins);
    gpu.polar_spectrum_backward(&head_tensor, &real, &imaginary, &real_grad, &imaginary_grad, &mut head_grad, max_log).unwrap();
    let analytic = gpu.download(&head_grad).unwrap();

    let step = 1e-3f32;
    for index in 0..head.len() {
        let mut plus = head.clone();
        let mut minus = head.clone();
        plus[index] += step;
        minus[index] -= step;
        let numeric = (loss(&plus) - loss(&minus)) / (2.0 * step as f64);
        let difference = (numeric - analytic[index] as f64).abs();
        assert!(
            difference < FINITE_DIFFERENCE_TOLERANCE * numeric.abs().max(1e-2),
            "head[{index}]: numeric {numeric}, analytic {}",
            analytic[index]
        );
    }
    assert_eq!(analytic[0], 0.0, "a capped log-magnitude receives no gradient");
}

// ------------------------------------------------------- elementwise gradients

#[test]
fn power_logarithm_and_absolute_error_gradients_match_their_formulas() {
    let gpu = gpu();
    let count = 1_025;
    let real = random_values(count, 7);
    let imaginary = random_values(count, 8);
    let output_grad = random_values(count, 9);
    let floor = 0.05f32;
    let energies: Vec<f32> = random_values(count, 10).iter().map(|value| value.abs()).collect();

    let real_tensor = upload(&gpu, 1, count, &real);
    let imaginary_tensor = upload(&gpu, 1, count, &imaginary);
    let grad_tensor = upload(&gpu, 1, count, &output_grad);
    let mut real_grad = zeros(&gpu, 1, count);
    let mut imaginary_grad = zeros(&gpu, 1, count);
    gpu.power_spectrum_backward(&real_tensor, &imaginary_tensor, &grad_tensor, &mut real_grad, &mut imaginary_grad).unwrap();
    let expected_re: Vec<f32> = real.iter().zip(&output_grad).map(|(re, g)| 2.0 * re * g).collect();
    let expected_im: Vec<f32> = imaginary.iter().zip(&output_grad).map(|(im, g)| 2.0 * im * g).collect();
    assert!(relative_error(&expected_re, &gpu.download(&real_grad).unwrap()) < TOLERANCE);
    assert!(relative_error(&expected_im, &gpu.download(&imaginary_grad).unwrap()) < TOLERANCE);

    let energies_tensor = upload(&gpu, 1, count, &energies);
    let mut log_grad = zeros(&gpu, 1, count);
    gpu.log_compress_backward(&energies_tensor, &grad_tensor, &mut log_grad, floor).unwrap();
    let expected_log: Vec<f32> = energies.iter().zip(&output_grad).map(|(e, g)| if *e > floor { g / e } else { 0.0 }).collect();
    assert!(relative_error(&expected_log, &gpu.download(&log_grad).unwrap()) < TOLERANCE);

    let mut absolute_grad = zeros(&gpu, 1, count);
    gpu.absolute_error_gradient(&real_tensor, &imaginary_tensor, &mut absolute_grad, 0.25).unwrap();
    let expected_absolute: Vec<f32> = real.iter().zip(&imaginary).map(|(a, b)| 0.25 * (a - b).signum()).collect();
    assert_eq!(gpu.download(&absolute_grad).unwrap(), expected_absolute);
}

// ----------------------------------------------------------- convolutions

fn depthwise_reference(input: &[f32], weight: &[f32], bias: &[f32], shape: Convolution) -> Vec<f32> {
    (0..shape.rows * shape.channels)
        .map(|element| {
            let (row, channel) = (element / shape.channels, element % shape.channels);
            let taps = (0..shape.kernel).filter_map(|tap| shape.source(row, tap).map(|source| (tap, source)));
            bias[channel]
                + taps.map(|(tap, source)| weight[tap * shape.channels + channel] * input[source * shape.channels + channel]).sum::<f32>()
        })
        .collect()
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
    gpu.depthwise_convolution(&input_tensor, &weight_tensor, &bias_tensor, &mut output, shape).unwrap();

    let expected = depthwise_reference(&input, &weight, &bias, shape);
    assert!(relative_error(&expected, &gpu.download(&output).unwrap()) < TOLERANCE, "forward");

    // The input gradient is the adjoint of the (bias-free) convolution.
    let grad_tensor = upload(&gpu, shape.rows, shape.channels, &output_grad);
    let mut input_grad = zeros(&gpu, shape.rows, shape.channels);
    gpu.depthwise_convolution_input_grad(&grad_tensor, &weight_tensor, &mut input_grad, shape).unwrap();
    let zero_bias = vec![0.0; shape.channels];
    let linear = depthwise_reference(&input, &weight, &zero_bias, shape);
    let forward = dot(&linear, &output_grad);
    let adjoint = dot(&input, &gpu.download(&input_grad).unwrap());
    assert!((forward - adjoint).abs() < 1e-3 * forward.abs().max(1.0), "input gradient: {forward} vs {adjoint}");

    // The weight gradient, against the same identity with the roles swapped.
    let mut weight_grad = zeros(&gpu, shape.kernel, shape.channels);
    gpu.depthwise_convolution_weight_grad(&grad_tensor, &input_tensor, &mut weight_grad, shape).unwrap();
    let by_weights = dot(&weight, &gpu.download(&weight_grad).unwrap());
    assert!((forward - by_weights).abs() < 1e-3 * forward.abs().max(1.0), "weight gradient: {forward} vs {by_weights}");
}

#[test]
fn unfold_lays_out_each_neighborhood_and_respects_sequence_edges() {
    let gpu = gpu();
    let shape = Convolution { rows: 2 * 6, time: 6, channels: 4, kernel: 3 };
    let input = random_values(shape.rows * shape.channels, 15);

    let input_tensor = upload(&gpu, shape.rows, shape.channels, &input);
    let width = shape.kernel * shape.channels;
    let mut columns = zeros(&gpu, shape.rows, width);
    gpu.unfold(&input_tensor, &mut columns, shape).unwrap();

    let expected: Vec<f32> = (0..shape.rows * width)
        .map(|element| {
            let (row, tap, channel) = (element / width, (element % width) / shape.channels, element % shape.channels);
            shape.source(row, tap).map_or(0.0, |source| input[source * shape.channels + channel])
        })
        .collect();
    assert_eq!(gpu.download(&columns).unwrap(), expected);
}

#[test]
fn column_scaling_and_its_gradient_match_reference() {
    let gpu = gpu();
    let (rows, cols) = (37, 70);
    let values = random_values(rows * cols, 16);
    let scale = random_values(cols, 17);
    let output_grad = random_values(rows * cols, 18);

    let values_tensor = upload(&gpu, rows, cols, &values);
    let scale_tensor = upload(&gpu, 1, cols, &scale);
    let mut output = zeros(&gpu, rows, cols);
    gpu.scale_columns(&values_tensor, &scale_tensor, &mut output).unwrap();
    let expected: Vec<f32> = values.iter().enumerate().map(|(index, value)| value * scale[index % cols]).collect();
    assert!(relative_error(&expected, &gpu.download(&output).unwrap()) < TOLERANCE);

    let grad_tensor = upload(&gpu, rows, cols, &output_grad);
    let mut scale_grad = zeros(&gpu, 1, cols);
    gpu.column_products(&grad_tensor, &values_tensor, &mut scale_grad).unwrap();
    let expected_grad: Vec<f32> =
        (0..cols).map(|col| (0..rows).map(|row| output_grad[row * cols + col] * values[row * cols + col]).sum()).collect();
    assert!(relative_error(&expected_grad, &gpu.download(&scale_grad).unwrap()) < TOLERANCE);
}
