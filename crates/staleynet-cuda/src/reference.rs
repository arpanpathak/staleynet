//! Straightforward CPU implementations of the kernels' mathematics.
//!
//! They favour obviousness over speed and exist so every GPU kernel can be
//! checked against an independent implementation of the same formula.

use crate::kernels::EMPTY_TOKEN;
use crate::kernels::elementwise::{GELU_CUBIC, GELU_SQRT_2_OVER_PI};
use crate::ops::{CrossEntropyHead, MatmulSpec, Storage, WriteMode};

/// Element (`row`, `col`) of the logical `[rows, cols]` matrix `op(X)`.
fn logical(values: &[f32], base: usize, row: usize, col: usize, rows: usize, cols: usize, storage: Storage) -> f32 {
    match storage {
        Storage::RowMajor => values[base + row * cols + col],
        Storage::Transposed => values[base + col * rows + row],
    }
}

/// Reference for [`Gpu::matmul`](crate::Gpu::matmul).
pub fn matmul(spec: &MatmulSpec, left: &[f32], right: &[f32], output: &mut [f32]) {
    for batch in 0..spec.batch.count {
        for row in 0..spec.rows {
            for col in 0..spec.cols {
                let mut sum = 0.0f32;
                for step in 0..spec.inner {
                    let left_value = logical(left, batch * spec.batch.left, row, step, spec.rows, spec.inner, spec.left);
                    let right_value = logical(right, batch * spec.batch.right, step, col, spec.inner, spec.cols, spec.right);
                    sum += left_value * right_value;
                }

                let slot = &mut output[batch * spec.batch.output + row * spec.cols + col];
                *slot = match spec.write {
                    WriteMode::Overwrite => spec.scale * sum,
                    WriteMode::Accumulate => *slot + spec.scale * sum,
                };
            }
        }
    }
}

/// GELU, tanh approximation.
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (GELU_SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x)).tanh())
}

/// Derivative of [`gelu`].
pub fn gelu_derivative(x: f32) -> f32 {
    let tanh = (GELU_SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x)).tanh();
    0.5 * (1.0 + tanh) + 0.5 * x * (1.0 - tanh * tanh) * GELU_SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_CUBIC * x * x)
}

/// Output of [`rms_norm_forward`].
pub struct RmsNormForward {
    /// Normalised, gained rows.
    pub output: Vec<f32>,

    /// `1 / rms` per row.
    pub inv_rms: Vec<f32>,
}

/// Reference RMSNorm forward pass.
pub fn rms_norm_forward(input: &[f32], gain: &[f32], cols: usize, epsilon: f32) -> RmsNormForward {
    let mut output = Vec::with_capacity(input.len());
    let mut inv_rms = Vec::with_capacity(input.len() / cols);

    for row in input.chunks_exact(cols) {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / cols as f32;
        let inverse = 1.0 / (mean_square + epsilon).sqrt();

        output.extend(row.iter().zip(gain).map(|(value, scale)| value * inverse * scale));
        inv_rms.push(inverse);
    }

    RmsNormForward { output, inv_rms }
}

/// Reference RMSNorm input gradient and gain gradient.
pub fn rms_norm_backward(output_grad: &[f32], input: &[f32], gain: &[f32], inv_rms: &[f32], cols: usize) -> (Vec<f32>, Vec<f32>) {
    let mut input_grad = vec![0.0; input.len()];
    let mut gain_grad = vec![0.0; cols];

    for (row, &inverse) in inv_rms.iter().enumerate() {
        let span = row * cols..(row + 1) * cols;
        let (dy, x) = (&output_grad[span.clone()], &input[span.clone()]);
        let projection: f32 = (0..cols).map(|col| dy[col] * gain[col] * x[col]).sum();
        let correction = inverse.powi(3) * projection / cols as f32;

        for col in 0..cols {
            input_grad[span.start + col] = inverse * gain[col] * dy[col] - x[col] * correction;
            gain_grad[col] += dy[col] * x[col] * inverse;
        }
    }

    (input_grad, gain_grad)
}

/// Reference causal softmax over `[rows, keys]`, in place.
pub fn causal_softmax(scores: &mut [f32], queries: usize, keys: usize, scale: f32, query_offset: usize) {
    for (row, values) in scores.chunks_exact_mut(keys).enumerate() {
        let visible = (row % queries + query_offset + 1).min(keys);
        let maximum = values[..visible].iter().map(|value| value * scale).fold(f32::NEG_INFINITY, f32::max);

        for value in &mut values[..visible] {
            *value = (*value * scale - maximum).exp();
        }
        let total: f32 = values[..visible].iter().sum();

        for value in &mut values[..visible] {
            *value /= total;
        }
        values[visible..].fill(0.0);
    }
}

/// Reference softmax backward, in place on `grad`.
pub fn softmax_backward(probabilities: &[f32], grad: &mut [f32], cols: usize) {
    for (grad_row, probability_row) in grad.chunks_exact_mut(cols).zip(probabilities.chunks_exact(cols)) {
        let projection: f32 = grad_row.iter().zip(probability_row).map(|(g, p)| g * p).sum();

        for (g, p) in grad_row.iter_mut().zip(probability_row) {
            *g = p * (*g - projection);
        }
    }
}

/// Reference cross-entropy for one head: writes `losses` and the head's gradient columns.
pub fn cross_entropy(logits: &[f32], targets: &[u32], cols: usize, head: &CrossEntropyHead, losses: &mut [f32], logits_grad: &mut [f32]) {
    for (row, loss) in losses.iter_mut().enumerate() {
        let span = row * cols + head.column_offset..row * cols + head.column_offset + head.classes;
        let target = targets[row * head.target_stride + head.target_offset];

        if target == EMPTY_TOKEN {
            *loss = 0.0;
            logits_grad[span].fill(0.0);
            continue;
        }

        let row_logits = &logits[span.clone()];
        let maximum = row_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_partition = maximum + row_logits.iter().map(|logit| (logit - maximum).exp()).sum::<f32>().ln();

        *loss = log_partition - row_logits[target as usize];
        for (class, (grad, logit)) in logits_grad[span].iter_mut().zip(row_logits).enumerate() {
            let one_hot = if class as u32 == target { 1.0 } else { 0.0 };
            *grad = ((logit - log_partition).exp() - one_hot) * head.grad_scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::BatchStrides;

    fn close(actual: f32, expected: f32, tolerance: f32) {
        assert!((actual - expected).abs() <= tolerance, "actual {actual} expected {expected}");
    }

    #[test]
    fn matmul_overwrites_a_row_major_product() {
        let spec = MatmulSpec::new(2, 3, 2);
        let left = [1.0, 2.0, 3.0, 4.0];
        let right = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut output = [0.0f32; 6];

        matmul(&spec, &left, &right, &mut output);

        assert_eq!(output, [21.0, 24.0, 27.0, 47.0, 54.0, 61.0]);
    }

    #[test]
    fn matmul_scales_and_accumulates_into_the_existing_output() {
        let spec = MatmulSpec::new(2, 3, 2).scaled(0.5).accumulate();
        let left = [1.0, 2.0, 3.0, 4.0];
        let right = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut output = [1.0f32; 6];

        matmul(&spec, &left, &right, &mut output);

        assert_eq!(output, [11.5, 13.0, 14.5, 24.5, 28.0, 31.5]);
    }

    #[test]
    fn matmul_honours_transposed_storage() {
        let spec = MatmulSpec::new(2, 2, 2).left_transposed();
        let left = [1.0, 2.0, 3.0, 4.0];
        let right = [5.0, 6.0, 7.0, 8.0];
        let mut output = [0.0f32; 4];

        matmul(&spec, &left, &right, &mut output);

        assert_eq!(output, [26.0, 30.0, 38.0, 44.0]);
    }

    #[test]
    fn matmul_broadcasts_batched_operands() {
        let strides = BatchStrides { count: 2, left: 4, right: 0, output: 4 };
        let spec = MatmulSpec::new(2, 2, 2).batched(strides);
        let left = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let right = [1.0, 0.0, 0.0, 1.0];
        let mut output = [0.0f32; 8];

        matmul(&spec, &left, &right, &mut output);

        assert_eq!(output, left);
    }

    #[test]
    fn logical_reads_row_major_and_transposed_storage() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        assert_eq!(logical(&values, 0, 0, 1, 2, 3, Storage::RowMajor), 2.0);
        assert_eq!(logical(&values, 0, 0, 1, 2, 3, Storage::Transposed), 3.0);
    }

    #[test]
    fn gelu_matches_known_values_and_its_finite_difference_derivative() {
        close(gelu(0.0), 0.0, 1e-6);
        close(gelu(1.0), 0.841_191, 1e-4);
        close(gelu_derivative(0.0), 0.5, 1e-6);

        let h = 1e-3;
        for x in [-1.0f32, 0.0, 1.0] {
            let finite_difference = (gelu(x + h) - gelu(x - h)) / (2.0 * h);
            close(gelu_derivative(x), finite_difference, 2e-3);
        }
    }

    #[test]
    fn rms_norm_forward_matches_the_definition() {
        let result = rms_norm_forward(&[3.0, 4.0], &[1.0, 1.0], 2, 0.0);

        close(result.inv_rms[0], 0.282_842_7, 1e-6);
        close(result.output[0], 0.848_528_1, 1e-6);
        close(result.output[1], 1.131_370_8, 1e-6);
    }

    #[test]
    fn rms_norm_backward_matches_the_hand_derived_gradient() {
        let inv_rms = [0.282_842_7f32];
        let (input_grad, gain_grad) = rms_norm_backward(&[1.0, 0.0], &[3.0, 4.0], &[1.0, 1.0], &inv_rms, 2);

        close(input_grad[0], 0.181_0, 1e-3);
        close(input_grad[1], -0.135_8, 1e-3);
        close(gain_grad[0], 0.848_528_1, 1e-6);
        close(gain_grad[1], 0.0, 1e-6);
    }

    #[test]
    fn causal_softmax_masks_future_keys() {
        let mut scores = [1.0, 100.0, 100.0, 1.0, 2.0, 100.0];

        causal_softmax(&mut scores, 2, 3, 1.0, 0);

        close(scores[0], 1.0, 1e-6);
        close(scores[1], 0.0, 1e-6);
        close(scores[2], 0.0, 1e-6);
        close(scores[3], 0.268_941_4, 1e-6);
        close(scores[4], 0.731_058_6, 1e-6);
        close(scores[5], 0.0, 1e-6);
        close(scores[3] + scores[4] + scores[5], 1.0, 1e-6);
    }

    #[test]
    fn causal_softmax_applies_the_scale_and_query_offset() {
        let mut scores = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0];

        causal_softmax(&mut scores, 2, 3, 0.5, 1);

        close(scores[0] + scores[1] + scores[2], 1.0, 1e-6);
        close(scores[3] + scores[4] + scores[5], 1.0, 1e-6);
        assert!(scores[0] < scores[1]);
        assert_eq!(scores[2], 0.0);
        assert!(scores[3] < scores[4] && scores[4] < scores[5]);
    }

    #[test]
    fn softmax_backward_projects_out_the_weighted_sum() {
        let probabilities = [0.25, 0.25, 0.5];
        let mut grad = [1.0, 2.0, 3.0];

        softmax_backward(&probabilities, &mut grad, 3);

        close(grad[0], -0.3125, 1e-6);
        close(grad[1], -0.0625, 1e-6);
        close(grad[2], 0.375, 1e-6);
        close(grad[0] + grad[1] + grad[2], 0.0, 1e-6);
    }

    #[test]
    fn cross_entropy_writes_losses_and_gradients_for_the_target_head() {
        let logits = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let targets = [2u32, 0];
        let head = CrossEntropyHead { column_offset: 0, classes: 3, target_stride: 1, target_offset: 0, grad_scale: 1.0 };
        let mut losses = [0.0f32; 2];
        let mut grad = [7.0f32; 8];

        cross_entropy(&logits, &targets, 4, &head, &mut losses, &mut grad);

        close(losses[0], 0.407_606, 1e-5);
        close(losses[1], 1.098_612, 1e-5);

        close(grad[0], 0.090_03, 1e-4);
        close(grad[1], 0.244_73, 1e-4);
        close(grad[2], -0.334_76, 1e-4);
        assert_eq!(grad[3], 7.0);

        close(grad[4], -0.666_667, 1e-5);
        close(grad[5], 0.333_333, 1e-5);
        close(grad[6], 0.333_333, 1e-5);
        assert_eq!(grad[7], 7.0);
    }

    #[test]
    fn cross_entropy_skips_empty_targets() {
        let logits = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let targets = [EMPTY_TOKEN, 0];
        let head = CrossEntropyHead { column_offset: 0, classes: 3, target_stride: 1, target_offset: 0, grad_scale: 1.0 };
        let mut losses = [9.0f32; 2];
        let mut grad = [7.0f32; 8];

        cross_entropy(&logits, &targets, 4, &head, &mut losses, &mut grad);

        assert_eq!(losses[0], 0.0);
        assert_eq!(&grad[0..3], &[0.0, 0.0, 0.0]);
        assert_eq!(grad[3], 7.0);
        assert!(losses[1] > 0.0);
        assert!(grad[4] < 0.0 && grad[5] > 0.0 && grad[6] > 0.0);
    }

    #[test]
    fn cross_entropy_honours_head_offsets_and_gradient_scale() {
        let logits = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = [0u32, 1, 0, 0];
        let head = CrossEntropyHead { column_offset: 2, classes: 2, target_stride: 2, target_offset: 1, grad_scale: 0.5 };
        let mut losses = [0.0f32; 2];
        let mut grad = [7.0f32; 10];

        cross_entropy(&logits, &targets, 5, &head, &mut losses, &mut grad);

        assert_eq!(&grad[0..2], &[7.0, 7.0]);
        assert_eq!(grad[4], 7.0);
        assert_eq!(&grad[5..7], &[7.0, 7.0]);
        assert_eq!(grad[9], 7.0);
        assert!(losses.iter().all(|loss| *loss > 0.0));
        assert!(grad[2].is_finite() && grad[3].is_finite());
        assert!(grad[7].is_finite() && grad[8].is_finite());
    }
}
