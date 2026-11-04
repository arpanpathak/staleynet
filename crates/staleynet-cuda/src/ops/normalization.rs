//! RMSNorm (Zhang & Sennrich, 2019, <https://arxiv.org/abs/1910.07467>).

use crate::device::{Gpu, GpuResult, Tensor};
use crate::launch;

impl Gpu {
    /// Normalises each row of `input` by its root-mean-square and multiplies
    /// by `gain`. Saves `1/rms` per row in `inv_rms` for the backward pass.
    pub fn rms_norm_forward(&self, input: &Tensor, gain: &Tensor, output: &mut Tensor, inv_rms: &mut Tensor, epsilon: f32) -> GpuResult {
        let shape = input.shape();
        assert_eq!(output.shape(), shape, "rms_norm_forward: output shape");
        assert_eq!(gain.len(), shape.cols, "rms_norm_forward: gain length");
        assert!(inv_rms.len() >= shape.rows, "rms_norm_forward: inv_rms too small");

        // SAFETY: warp per row; kernel writes stay inside rows·cols and rows,
        // which the asserts above tie to the buffer sizes.
        unsafe {
            self.kernels.normalization.rms_norm_forward(
                &self.stream,
                launch::warp_per_row(shape.rows),
                &input.buffer,
                &gain.buffer,
                &mut output.buffer,
                &mut inv_rms.buffer,
                shape.rows,
                shape.cols,
                epsilon,
            )
        }
    }

    /// Writes `input_grad` and accumulates `gain_grad`.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_backward(
        &self,
        output_grad: &Tensor,
        input: &Tensor,
        gain: &Tensor,
        inv_rms: &Tensor,
        input_grad: &mut Tensor,
        gain_grad: &mut Tensor,
    ) -> GpuResult {
        let shape = input.shape();
        assert_eq!(output_grad.shape(), shape, "rms_norm_backward: output_grad shape");
        assert_eq!(input_grad.shape(), shape, "rms_norm_backward: input_grad shape");
        assert_eq!(gain_grad.len(), shape.cols, "rms_norm_backward: gain_grad length");

        // SAFETY: warp per row; writes stay inside rows·cols == input_grad.len().
        unsafe {
            self.kernels.normalization.rms_norm_backward_input(
                &self.stream,
                launch::warp_per_row(shape.rows),
                &output_grad.buffer,
                &input.buffer,
                &gain.buffer,
                &inv_rms.buffer,
                &mut input_grad.buffer,
                shape.rows,
                shape.cols,
            )?;
        }

        // SAFETY: one thread per column; out-of-range threads return early.
        unsafe {
            self.kernels.normalization.rms_norm_backward_gain(
                &self.stream,
                launch::elementwise(shape.cols),
                &output_grad.buffer,
                &input.buffer,
                &inv_rms.buffer,
                &mut gain_grad.buffer,
                shape.rows,
                shape.cols,
            )
        }
    }
}
