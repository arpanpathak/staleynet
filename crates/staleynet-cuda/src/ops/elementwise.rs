//! Elementwise arithmetic, bias handling and the GELU activation.

use crate::device::{Gpu, GpuResult, Tensor};
use crate::launch;

impl Gpu {
    /// `output += scale · input` (residual connections, gradient sums).
    pub fn add_scaled(&self, scale: f32, input: &Tensor, output: &mut Tensor) -> GpuResult {
        assert_eq!(input.shape(), output.shape(), "add_scaled: shape mismatch");

        // SAFETY: one thread per output element; `get_mut` guards the tail.
        unsafe {
            self.kernels.elementwise.add_scaled(&self.stream, launch::elementwise(output.len()), scale, &input.buffer, &mut output.buffer)
        }
    }

    /// `values *= factor`.
    pub fn scale(&self, values: &mut Tensor, factor: f32) -> GpuResult {
        // SAFETY: one thread per element; `get_mut` guards the tail.
        unsafe { self.kernels.elementwise.scale_in_place(&self.stream, launch::elementwise(values.len()), &mut values.buffer, factor) }
    }

    /// `output[r, :] += bias` for every row.
    pub fn add_row_bias(&self, output: &mut Tensor, bias: &Tensor) -> GpuResult {
        let cols = output.shape().cols;
        assert_eq!(bias.len(), cols, "add_row_bias: bias length must equal output columns");

        // SAFETY: one thread per output element; `get_mut` guards the tail.
        unsafe {
            self.kernels.elementwise.add_row_bias(&self.stream, launch::elementwise(output.len()), &mut output.buffer, &bias.buffer, cols)
        }
    }

    /// `sums += Σ_rows input`: the gradient of a bias.
    pub fn accumulate_column_sums(&self, input: &Tensor, sums: &mut Tensor) -> GpuResult {
        let shape = input.shape();
        assert_eq!(sums.len(), shape.cols, "accumulate_column_sums: sums length must equal input columns");

        // SAFETY: one thread per column; out-of-range threads return early.
        unsafe {
            self.kernels.elementwise.accumulate_column_sums(
                &self.stream,
                launch::elementwise(shape.cols),
                &input.buffer,
                shape.rows,
                shape.cols,
                &mut sums.buffer,
            )
        }
    }

    /// `output = GELU(input)`.
    pub fn gelu_forward(&self, input: &Tensor, output: &mut Tensor) -> GpuResult {
        assert_eq!(input.shape(), output.shape(), "gelu_forward: shape mismatch");

        // SAFETY: one thread per element; `get_mut` guards the tail.
        unsafe { self.kernels.elementwise.gelu_forward(&self.stream, launch::elementwise(output.len()), &input.buffer, &mut output.buffer) }
    }

    /// `input_grad = output_grad ⊙ GELU'(input)`.
    pub fn gelu_backward(&self, input: &Tensor, output_grad: &Tensor, input_grad: &mut Tensor) -> GpuResult {
        assert_eq!(input.shape(), output_grad.shape(), "gelu_backward: shape mismatch");
        assert_eq!(input.shape(), input_grad.shape(), "gelu_backward: shape mismatch");

        // SAFETY: one thread per element; `get_mut` guards the tail.
        unsafe {
            self.kernels.elementwise.gelu_backward(
                &self.stream,
                launch::elementwise(input_grad.len()),
                &input.buffer,
                &output_grad.buffer,
                &mut input_grad.buffer,
            )
        }
    }
}
