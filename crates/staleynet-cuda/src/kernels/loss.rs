//! Softmax cross-entropy, computed together with its gradient.
//!
//! For logits `z` and target class `y`, `loss = logsumexp(z) − z_y` and
//! `∂loss/∂z = softmax(z) − onehot(y)` (Goodfellow, Bengio and Courville,
//! *Deep Learning*, §6.2.2.3, <https://www.deeplearningbook.org/contents/mlp.html>).

use cuda_device::cuda_module;

/// Which slice of a wide matrix of logits one classifier owns, passed to the
/// kernel by value. With `n` codebooks the output layer places `n`
/// classifiers side by side in each row, and each has its own column of targets.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Classifier {
    /// Logits per row of the matrix (all classifiers together).
    pub row_width: usize,

    /// First logit column of this classifier.
    pub column_offset: usize,

    /// Number of classes.
    pub classes: usize,

    /// Targets per row (one per classifier).
    pub target_stride: usize,

    /// This classifier's position within a row of targets.
    pub target_offset: usize,

    /// Factor applied to the gradient, usually `1 / (targets · classifiers)`
    /// so that the summed loss becomes a mean.
    pub grad_scale: f32,
}

#[cuda_module]
pub mod kernels {
    use super::Classifier;
    use crate::kernels::device::{WarpRow, store, warp_row};
    use crate::kernels::{EMPTY_TOKEN, NEGATIVE_LARGE, WARP_SIZE};
    use cuda_device::{DisjointSlice, kernel, warp};

    /// Computes the loss of one classifier for every row, and writes its
    /// gradient into that classifier's columns of `logits_grad` (the other
    /// columns are left untouched). Rows whose target is empty get zero loss
    /// and zero gradient. One warp per row.
    ///
    /// * `logits`: the scores, `[rows, row_width]`.
    /// * `targets`: `target_stride` class ids per row.
    /// * `losses`: one loss per row, written.
    /// * `logits_grad`: the gradient with respect to the logits, same shape as `logits`.
    /// * `rows`: the number of rows.
    /// * `classifier`: which columns and targets belong to this classifier.
    #[kernel]
    pub fn cross_entropy(
        logits: &[f32],
        targets: &[u32],
        mut losses: DisjointSlice<f32>,
        mut logits_grad: DisjointSlice<f32>,
        rows: usize,
        classifier: Classifier,
    ) {
        let WarpRow { row, lane } = warp_row();
        if row >= rows {
            return;
        }

        let base = row * classifier.row_width + classifier.column_offset;
        let target = targets[row * classifier.target_stride + classifier.target_offset];
        let has_target = target != EMPTY_TOKEN;

        let mut maximum = NEGATIVE_LARGE;
        for class in (lane..classifier.classes).step_by(WARP_SIZE) {
            maximum = maximum.max(logits[base + class]);
        }
        let maximum = warp::reduce_max_f32(maximum);

        let mut total = 0.0f32;
        for class in (lane..classifier.classes).step_by(WARP_SIZE) {
            total += libm::expf(logits[base + class] - maximum);
        }
        let log_partition = maximum + libm::logf(warp::reduce_sum_f32(total));

        for class in (lane..classifier.classes).step_by(WARP_SIZE) {
            let probability = libm::expf(logits[base + class] - log_partition);
            let one_hot = if class as u32 == target { 1.0 } else { 0.0 };
            let gradient = if has_target { (probability - one_hot) * classifier.grad_scale } else { 0.0 };
            store(&mut logits_grad, base + class, gradient);
        }

        if lane == 0 {
            let loss = if has_target { log_partition - logits[base + target as usize] } else { 0.0 };
            store(&mut losses, row, loss);
        }
    }
}
