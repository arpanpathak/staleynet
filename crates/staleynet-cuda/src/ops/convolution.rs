//! Depthwise convolutions over time and the unfolding that turns a convolution
//! across channels into a matrix product. See [`crate::kernels::convolution`].

use crate::device::{Gpu, GpuResult, Tensor};
pub use crate::kernels::convolution::Convolution;
use crate::launch;

impl Gpu {
    /// A depthwise convolution over time with `weight` (`[taps, channels]`) and `bias`.
    pub fn depthwise_convolution(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        output: &mut Tensor,
        shape: Convolution,
    ) -> GpuResult {
        check_convolution(input, output, shape);
        assert_eq!(weight.len(), shape.kernel * shape.channels, "depthwise_convolution: weight size");
        assert_eq!(bias.len(), shape.channels, "depthwise_convolution: bias size");

        // SAFETY: one thread per output element; taps outside a sequence are skipped.
        unsafe {
            self.kernels.convolution.depthwise_convolution(
                &self.stream,
                launch::elementwise(output.len()),
                &input.buffer,
                &weight.buffer,
                &bias.buffer,
                &mut output.buffer,
                shape,
            )
        }
    }

    /// The gradient of [`Gpu::depthwise_convolution`] with respect to its input.
    pub fn depthwise_convolution_input_grad(
        &self,
        output_grad: &Tensor,
        weight: &Tensor,
        input_grad: &mut Tensor,
        shape: Convolution,
    ) -> GpuResult {
        check_convolution(output_grad, input_grad, shape);
        assert_eq!(weight.len(), shape.kernel * shape.channels, "depthwise_convolution_input_grad: weight size");

        // SAFETY: one thread per input element; taps outside a sequence are skipped.
        unsafe {
            self.kernels.convolution.depthwise_convolution_input_grad(
                &self.stream,
                launch::elementwise(input_grad.len()),
                &output_grad.buffer,
                &weight.buffer,
                &mut input_grad.buffer,
                shape,
            )
        }
    }

    /// The gradient of [`Gpu::depthwise_convolution`] with respect to its weights.
    pub fn depthwise_convolution_weight_grad(
        &self,
        output_grad: &Tensor,
        input: &Tensor,
        weight_grad: &mut Tensor,
        shape: Convolution,
    ) -> GpuResult {
        check_convolution(input, output_grad, shape);
        assert_eq!(weight_grad.len(), shape.kernel * shape.channels, "depthwise_convolution_weight_grad: weight size");

        // SAFETY: one thread per weight; every row read is within the checked shape.
        unsafe {
            self.kernels.convolution.depthwise_convolution_weight_grad(
                &self.stream,
                launch::elementwise(weight_grad.len()),
                &output_grad.buffer,
                &input.buffer,
                &mut weight_grad.buffer,
                shape,
            )
        }
    }

    /// Lays out each frame's `shape.kernel` neighbors side by side, so that a
    /// convolution across channels becomes one matrix product.
    pub fn unfold(&self, input: &Tensor, columns: &mut Tensor, shape: Convolution) -> GpuResult {
        assert_eq!(input.len(), shape.rows * shape.channels, "unfold: input size");
        assert_eq!(columns.len(), shape.rows * shape.kernel * shape.channels, "unfold: columns size");

        // SAFETY: one thread per output element; taps outside a sequence write zero.
        unsafe {
            self.kernels
                .convolution
                .unfold(&self.stream, launch::elementwise(columns.len()), &input.buffer, &mut columns.buffer, shape)
        }
    }
}

/// Asserts that `input` and `output` both hold `[rows, channels]` for `shape`,
/// whose rows are whole sequences of `time` frames.
fn check_convolution(input: &Tensor, output: &Tensor, shape: Convolution) {
    assert_eq!(input.len(), shape.rows * shape.channels, "convolution: input size");
    assert_eq!(output.len(), shape.rows * shape.channels, "convolution: output size");
    assert_eq!(shape.rows % shape.time, 0, "convolution: rows must be whole sequences");
    assert_eq!(shape.kernel % 2, 1, "convolution: the kernel must have an odd number of taps");
}
