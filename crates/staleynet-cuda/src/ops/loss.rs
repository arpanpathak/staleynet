//! Softmax cross-entropy, fused with its gradient.
//!
//! For logits `z` and target class `y`: `loss = logsumexp(z) − z_y` and
//! `∂loss/∂z = softmax(z) − one_hot(y)`. See Goodfellow, Bengio & Courville,
//! *Deep Learning*, §6.2.2.3: <https://www.deeplearningbook.org/contents/mlp.html>

use crate::device::{Gpu, GpuResult, Tensor, TokenBuffer};
use crate::launch;

/// Which slice of a wide logits matrix (and of the targets) one classifier
/// head owns. With `n` codebooks the output layer emits `n` heads side by side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossEntropyHead {
    /// First logit column of this head.
    pub column_offset: usize,

    /// Number of classes (codebook size).
    pub classes: usize,

    /// Targets per row (number of heads).
    pub target_stride: usize,

    /// This head's index within a target row.
    pub target_offset: usize,

    /// Multiplier on the gradient, typically `1 / (valid targets · heads)` so
    /// the summed loss is a mean.
    pub grad_scale: f32,
}

impl Gpu {
    /// Writes per-row losses into `losses` and this head's gradient columns
    /// into `logits_grad` (other columns untouched).
    pub fn cross_entropy(
        &self,
        logits: &Tensor,
        targets: &TokenBuffer,
        head: &CrossEntropyHead,
        losses: &mut Tensor,
        logits_grad: &mut Tensor,
    ) -> GpuResult {
        let shape = logits.shape();
        assert_eq!(logits_grad.shape(), shape, "cross_entropy: gradient shape");
        assert!(head.column_offset + head.classes <= shape.cols, "cross_entropy: head exceeds logits");
        assert!(losses.len() >= shape.rows, "cross_entropy: losses too small");
        assert!(targets.len() >= shape.rows * head.target_stride, "cross_entropy: too few targets");

        // SAFETY: warp per row; writes stay inside rows·cols and rows, asserted above.
        unsafe {
            self.kernels.cross_entropy(
                &self.stream,
                launch::warp_per_row(shape.rows),
                &logits.buffer,
                targets,
                &mut losses.buffer,
                &mut logits_grad.buffer,
                shape.rows,
                shape.cols,
                head.column_offset,
                head.classes,
                head.target_stride,
                head.target_offset,
                head.grad_scale,
            )
        }
    }
}
