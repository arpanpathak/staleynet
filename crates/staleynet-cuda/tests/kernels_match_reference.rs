//! Every kernel against the CPU reference on random inputs, including ragged
//! sizes that are not multiples of the tile or warp size.

use staleynet_cuda::reference;
use staleynet_cuda::{
    AdamWHyperparameters, BatchStrides, CrossEntropyHead, EMPTY_TOKEN, Gpu, HeadLayout, MatmulKernel, MatmulSpec, Shape, TokenIndex,
    WriteMode,
};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Relative tolerance for f32 results computed with different summation orders.
const TOLERANCE: f32 = 1e-5;

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

#[test]
fn matmul_matches_reference_for_every_storage_and_kernel() {
    let mut gpu = gpu();
    let sizes = [(1, 1, 1), (17, 33, 9), (64, 48, 70), (5, 129, 31)];

    for (rows, cols, inner) in sizes {
        for batch in [1, 3] {
            let strides = BatchStrides { count: batch, left: rows * inner, right: inner * cols, output: rows * cols };
            let base = MatmulSpec::new(rows, cols, inner).batched(strides).scaled(0.7).accumulate();
            let variants = [base, base.left_transposed(), base.right_transposed(), base.left_transposed().right_transposed()];

            for spec in variants {
                let left = random_values(batch * rows * inner, 1);
                let right = random_values(batch * inner * cols, 2);
                let initial = random_values(batch * rows * cols, 3);

                let mut expected = initial.clone();
                reference::matmul(&spec, &left, &right, &mut expected);

                for kernel in [MatmulKernel::Naive, MatmulKernel::Tiled, MatmulKernel::Blocked] {
                    gpu.matmul_kernel = kernel;
                    let left_tensor = gpu.upload(Shape::vector(left.len()), &left).unwrap();
                    let right_tensor = gpu.upload(Shape::vector(right.len()), &right).unwrap();
                    let mut output = gpu.upload(Shape::vector(initial.len()), &initial).unwrap();

                    gpu.matmul(&spec, &left_tensor, &right_tensor, &mut output).unwrap();

                    let error = relative_error(&expected, &gpu.download(&output).unwrap());
                    assert!(error < TOLERANCE, "{kernel:?} {spec:?}: relative error {error}");
                }
            }
        }
    }
}

#[test]
fn rms_norm_forward_and_backward_match_reference() {
    let gpu = gpu();
    let shape = Shape::new(37, 70);
    let epsilon = 1e-5;
    let input = random_values(shape.len(), 4);
    let gain = random_values(shape.cols, 5);
    let output_grad = random_values(shape.len(), 6);

    let expected = reference::rms_norm_forward(&input, &gain, shape.cols, epsilon);
    let (expected_input_grad, expected_gain_grad) =
        reference::rms_norm_backward(&output_grad, &input, &gain, &expected.inv_rms, shape.cols);

    let input_tensor = gpu.upload(shape, &input).unwrap();
    let gain_tensor = gpu.upload(Shape::vector(shape.cols), &gain).unwrap();
    let mut output = gpu.zeros(shape).unwrap();
    let mut inv_rms = gpu.zeros(Shape::vector(shape.rows)).unwrap();
    gpu.rms_norm_forward(&input_tensor, &gain_tensor, &mut output, &mut inv_rms, epsilon).unwrap();

    let output_grad_tensor = gpu.upload(shape, &output_grad).unwrap();
    let mut input_grad = gpu.zeros(shape).unwrap();
    let mut gain_grad = gpu.zeros(Shape::vector(shape.cols)).unwrap();
    gpu.rms_norm_backward(&output_grad_tensor, &input_tensor, &gain_tensor, &inv_rms, &mut input_grad, &mut gain_grad).unwrap();

    assert!(relative_error(&expected.output, &gpu.download(&output).unwrap()) < TOLERANCE);
    assert!(relative_error(&expected.inv_rms, &gpu.download(&inv_rms).unwrap()) < TOLERANCE);
    assert!(relative_error(&expected_input_grad, &gpu.download(&input_grad).unwrap()) < 1e-4);
    assert!(relative_error(&expected_gain_grad, &gpu.download(&gain_grad).unwrap()) < 1e-4);
}

#[test]
fn elementwise_ops_match_reference() {
    let gpu = gpu();
    let shape = Shape::new(9, 50);
    let input = random_values(shape.len(), 7);
    let output_grad = random_values(shape.len(), 8);
    let bias = random_values(shape.cols, 9);
    let input_tensor = gpu.upload(shape, &input).unwrap();

    let mut activated = gpu.zeros(shape).unwrap();
    gpu.gelu_forward(&input_tensor, &mut activated).unwrap();
    let expected: Vec<f32> = input.iter().map(|&x| reference::gelu(x)).collect();
    assert!(relative_error(&expected, &gpu.download(&activated).unwrap()) < TOLERANCE);

    let mut input_grad = gpu.zeros(shape).unwrap();
    gpu.gelu_backward(&input_tensor, &gpu.upload(shape, &output_grad).unwrap(), &mut input_grad).unwrap();
    let expected: Vec<f32> = input.iter().zip(&output_grad).map(|(&x, &dy)| dy * reference::gelu_derivative(x)).collect();
    assert!(relative_error(&expected, &gpu.download(&input_grad).unwrap()) < TOLERANCE);

    let mut combined = gpu.upload(shape, &output_grad).unwrap();
    gpu.add_scaled(0.3, &input_tensor, &mut combined).unwrap();
    gpu.scale(&mut combined, 1.5).unwrap();
    gpu.add_row_bias(&mut combined, &gpu.upload(Shape::vector(shape.cols), &bias).unwrap()).unwrap();
    let expected: Vec<f32> =
        (0..shape.len()).map(|index| (output_grad[index] + 0.3 * input[index]) * 1.5 + bias[index % shape.cols]).collect();
    assert!(relative_error(&expected, &gpu.download(&combined).unwrap()) < TOLERANCE);

    let mut column_sums = gpu.upload(Shape::vector(shape.cols), &bias).unwrap();
    gpu.accumulate_column_sums(&input_tensor, &mut column_sums).unwrap();
    let expected: Vec<f32> =
        (0..shape.cols).map(|col| bias[col] + (0..shape.rows).map(|row| input[row * shape.cols + col]).sum::<f32>()).collect();
    assert!(relative_error(&expected, &gpu.download(&column_sums).unwrap()) < TOLERANCE);

    let squares: f64 = input.iter().map(|&value| (value as f64).powi(2)).sum();
    assert!((gpu.sum_of_squares(&input_tensor).unwrap() - squares).abs() / squares < 1e-5);
}

#[test]
fn causal_softmax_and_its_backward_match_reference() {
    let gpu = gpu();
    let (matrices, queries, keys, query_offset, scale) = (3, 5, 70, 40, 0.5);
    let shape = Shape::new(matrices * queries, keys);
    let scores = random_values(shape.len(), 10);
    let upstream = random_values(shape.len(), 11);

    let mut expected = scores.clone();
    reference::causal_softmax(&mut expected, queries, keys, scale, query_offset);
    let mut expected_grad = upstream.clone();
    reference::softmax_backward(&expected, &mut expected_grad, keys);

    let mut probabilities = gpu.upload(shape, &scores).unwrap();
    gpu.causal_softmax(&mut probabilities, queries, scale, query_offset).unwrap();
    let mut grad = gpu.upload(shape, &upstream).unwrap();
    gpu.softmax_backward(&probabilities, &mut grad).unwrap();

    assert!(relative_error(&expected, &gpu.download(&probabilities).unwrap()) < TOLERANCE);
    assert!(relative_error(&expected_grad, &gpu.download(&grad).unwrap()) < TOLERANCE);
}

#[test]
fn split_and_merge_heads_follow_the_layout_formula() {
    let gpu = gpu();
    let layout = HeadLayout::dense(2, 5, 3, 4, 3 * 12 + 2, 13).in_cache(9, 2);
    let packed = random_values(layout.batch * layout.time * layout.width, 12);

    let packed_tensor = gpu.upload(Shape::new(layout.batch * layout.time, layout.width), &packed).unwrap();
    let mut head_major = gpu.zeros(Shape::new(layout.batch * layout.heads * layout.time_capacity, layout.head_dim)).unwrap();
    gpu.split_heads(&packed_tensor, &mut head_major, &layout).unwrap();
    let split = gpu.download(&head_major).unwrap();

    for (batch, head, step, dim) in (0..layout.batch).flat_map(|b| {
        (0..layout.heads).flat_map(move |h| (0..layout.time).flat_map(move |t| (0..layout.head_dim).map(move |d| (b, h, t, d))))
    }) {
        let source = (batch * layout.time + step) * layout.width + layout.column_offset + head * layout.head_dim + dim;
        let target = ((batch * layout.heads + head) * layout.time_capacity + layout.time_offset + step) * layout.head_dim + dim;
        assert_eq!(split[target], packed[source]);
    }

    let mut merged = gpu.zeros(packed_tensor.shape()).unwrap();
    gpu.merge_heads(&head_major, &mut merged, &layout).unwrap();
    let merged = gpu.download(&merged).unwrap();
    let head_columns = layout.column_offset..layout.column_offset + layout.heads * layout.head_dim;
    for (index, (&original, &round_trip)) in packed.iter().zip(&merged).enumerate() {
        let expected = if head_columns.contains(&(index % layout.width)) { original } else { 0.0 };
        assert_eq!(round_trip, expected);
    }
}

#[test]
fn embedding_forward_and_backward_sum_over_slots() {
    let gpu = gpu();
    let (vocabulary, width, positions, slots) = (11, 6, 9, 2);
    let table = random_values(vocabulary * width, 13);
    let tokens: Vec<u32> =
        (0..positions * slots).map(|index| if index % 5 == 3 { EMPTY_TOKEN } else { (index * 7 % vocabulary) as u32 }).collect();
    let output_grad = random_values(positions * width, 14);

    let mut expected_output = vec![0.0; positions * width];
    let mut expected_table_grad = vec![0.0; vocabulary * width];
    for (flat_index, &token) in tokens.iter().enumerate().filter(|(_, token)| **token != EMPTY_TOKEN) {
        let position = flat_index / slots;
        for col in 0..width {
            expected_output[position * width + col] += table[token as usize * width + col];
            expected_table_grad[token as usize * width + col] += output_grad[position * width + col];
        }
    }

    let table_tensor = gpu.upload(Shape::new(vocabulary, width), &table).unwrap();
    let token_buffer = gpu.upload_tokens(&tokens).unwrap();
    let mut output = gpu.zeros(Shape::new(positions, width)).unwrap();
    gpu.embedding_forward(&table_tensor, &token_buffer, slots, &mut output, WriteMode::Overwrite).unwrap();

    let index = TokenIndex::build(&gpu, &tokens, slots, vocabulary).unwrap();
    let mut table_grad = gpu.zeros(Shape::new(vocabulary, width)).unwrap();
    gpu.embedding_backward(&gpu.upload(Shape::new(positions, width), &output_grad).unwrap(), &index, &mut table_grad).unwrap();

    assert!(relative_error(&expected_output, &gpu.download(&output).unwrap()) < TOLERANCE);
    assert!(relative_error(&expected_table_grad, &gpu.download(&table_grad).unwrap()) < TOLERANCE);
}

#[test]
fn cross_entropy_matches_reference_and_skips_empty_targets() {
    let gpu = gpu();
    let shape = Shape::new(13, 50);
    let head = CrossEntropyHead { column_offset: 10, classes: 37, target_stride: 2, target_offset: 1, grad_scale: 0.25 };
    let logits = random_values(shape.len(), 15);
    let targets: Vec<u32> =
        (0..shape.rows * 2).map(|index| if index % 7 == 0 { EMPTY_TOKEN } else { (index * 5 % head.classes) as u32 }).collect();

    let mut expected_losses = vec![0.0; shape.rows];
    let mut expected_grad = vec![0.0; shape.len()];
    reference::cross_entropy(&logits, &targets, shape.cols, &head, &mut expected_losses, &mut expected_grad);

    let mut losses = gpu.zeros(Shape::vector(shape.rows)).unwrap();
    let mut grad = gpu.zeros(shape).unwrap();
    gpu.cross_entropy(&gpu.upload(shape, &logits).unwrap(), &gpu.upload_tokens(&targets).unwrap(), &head, &mut losses, &mut grad).unwrap();

    assert!(relative_error(&expected_losses, &gpu.download(&losses).unwrap()) < TOLERANCE);
    assert!(relative_error(&expected_grad, &gpu.download(&grad).unwrap()) < TOLERANCE);
}

#[test]
fn adamw_step_matches_the_published_update() {
    let gpu = gpu();
    let shape = Shape::vector(300);
    let hyper = AdamWHyperparameters { learning_rate: 1e-3, beta1: 0.9, beta2: 0.95, epsilon: 1e-8, weight_decay: 0.1, step: 2 };
    let parameter = random_values(shape.len(), 16);
    let gradient = random_values(shape.len(), 17);
    let (initial_mean, initial_variance) = (0.01f32, 0.02f32);

    let expected: Vec<f32> = parameter
        .iter()
        .zip(&gradient)
        .map(|(&weight, &grad)| {
            let mean = hyper.beta1 * initial_mean + (1.0 - hyper.beta1) * grad;
            let variance = hyper.beta2 * initial_variance + (1.0 - hyper.beta2) * grad * grad;
            let mean_hat = mean / (1.0 - hyper.beta1.powi(2));
            let variance_hat = variance / (1.0 - hyper.beta2.powi(2));
            weight - hyper.learning_rate * (mean_hat / (variance_hat.sqrt() + hyper.epsilon) + hyper.weight_decay * weight)
        })
        .collect();

    let mut parameter_tensor = gpu.upload(shape, &parameter).unwrap();
    let mut first_moment = gpu.upload(shape, &vec![initial_mean; shape.len()]).unwrap();
    let mut second_moment = gpu.upload(shape, &vec![initial_variance; shape.len()]).unwrap();
    gpu.adamw_step(&mut parameter_tensor, &gpu.upload(shape, &gradient).unwrap(), &mut first_moment, &mut second_moment, &hyper).unwrap();

    assert!(relative_error(&expected, &gpu.download(&parameter_tensor).unwrap()) < TOLERANCE);
}

#[test]
fn signal_and_clustering_ops_match_definitions() {
    let gpu = gpu();
    let (frame_count, frame_length, hop) = (7, 16, 5);
    let signal = random_values((frame_count - 1) * hop + frame_length, 18);
    let mut frames = gpu.zeros(Shape::new(frame_count, frame_length)).unwrap();
    gpu.frame_signal(&gpu.upload(Shape::vector(signal.len()), &signal).unwrap(), &mut frames, hop).unwrap();
    let frames = gpu.download(&frames).unwrap();
    for (frame, sample) in (0..frame_count).flat_map(|f| (0..frame_length).map(move |n| (f, n))) {
        assert_eq!(frames[frame * frame_length + sample], signal[frame * hop + sample]);
    }

    let shape = Shape::new(4, 9);
    let (real, imaginary) = (random_values(shape.len(), 19), random_values(shape.len(), 20));
    let mut power = gpu.zeros(shape).unwrap();
    gpu.power_spectrum(&gpu.upload(shape, &real).unwrap(), &gpu.upload(shape, &imaginary).unwrap(), &mut power).unwrap();
    let floor = 0.05;
    gpu.log_compress(&mut power, floor).unwrap();
    let expected: Vec<f32> = real.iter().zip(&imaginary).map(|(re, im)| (re * re + im * im).max(floor).ln()).collect();
    assert!(relative_error(&expected, &gpu.download(&power).unwrap()) < TOLERANCE);

    let (points, centroids) = (40, 33);
    let dots = random_values(points * centroids, 21);
    let norms: Vec<f32> = random_values(centroids, 22).iter().map(|value| value.abs()).collect();
    let mut assignments = gpu.upload_tokens(&vec![0; points]).unwrap();
    gpu.nearest_centroid(
        &gpu.upload(Shape::new(points, centroids), &dots).unwrap(),
        &gpu.upload(Shape::vector(centroids), &norms).unwrap(),
        &mut assignments,
    )
    .unwrap();
    let expected: Vec<u32> = (0..points)
        .map(|point| {
            (0..centroids)
                .min_by(|&a, &b| (norms[a] - 2.0 * dots[point * centroids + a]).total_cmp(&(norms[b] - 2.0 * dots[point * centroids + b])))
                .unwrap() as u32
        })
        .collect();
    assert_eq!(expected, gpu.download_tokens(&assignments).unwrap());
}
