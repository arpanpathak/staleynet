//! The operations of the neural vocoder: overlap-add for the inverse STFT,
//! the predicted spectrum, depthwise and unfolded convolutions, column scales,
//! and the gradients of the loss path (power spectrum, logarithm, absolute
//! error). Each wrapper checks the sizes its kernel depends on before launching.

use crate::device::{Gpu, GpuResult, Tensor};
pub use crate::kernels::Convolution;
use crate::launch;

impl Gpu {
    /// Overlap-adds the rows of `frames` (`[frame count, frame length]`), each
    /// starting `hop` samples after the previous one, into `signal`, and
    /// multiplies every sample by `normalization`.
    pub fn overlap_add(&self, frames: &Tensor, normalization: &Tensor, signal: &mut Tensor, hop: usize) -> GpuResult {
        let shape = frames.shape();
        let length = (shape.rows - 1) * hop + shape.cols;
        assert_eq!(signal.len(), length, "overlap_add: signal length");
        assert_eq!(normalization.len(), length, "overlap_add: normalization length");
        assert!(hop <= shape.cols, "overlap_add: frames must not leave gaps");

        // SAFETY: one thread per signal sample; every frame index read is below
        // rows · cols because first..=last only spans frames covering the sample.
        unsafe {
            self.kernels.synthesis.overlap_add(
                &self.stream,
                launch::elementwise(length),
                &frames.buffer,
                &normalization.buffer,
                &mut signal.buffer,
                shape.rows,
                shape.cols,
                hop,
            )
        }
    }

    /// The gradient of [`Gpu::overlap_add`] with respect to its frames.
    pub fn overlap_add_backward(&self, signal_grad: &Tensor, normalization: &Tensor, frames_grad: &mut Tensor, hop: usize) -> GpuResult {
        let shape = frames_grad.shape();
        let length = (shape.rows - 1) * hop + shape.cols;
        assert_eq!(signal_grad.len(), length, "overlap_add_backward: signal length");
        assert_eq!(normalization.len(), length, "overlap_add_backward: normalization length");

        // SAFETY: one thread per frame sample; the sample read is below `length`.
        unsafe {
            self.kernels.synthesis.overlap_add_backward(
                &self.stream,
                launch::elementwise(frames_grad.len()),
                &signal_grad.buffer,
                &normalization.buffer,
                &mut frames_grad.buffer,
                shape.cols,
                hop,
            )
        }
    }

    /// The complex spectrum predicted by the network's head (`[frames, 2 · bins]`,
    /// log-magnitudes then phases), with log-magnitudes capped at `max_log`.
    pub fn polar_spectrum(&self, head: &Tensor, real: &mut Tensor, imaginary: &mut Tensor, max_log: f32) -> GpuResult {
        let shape = real.shape();
        assert_eq!(imaginary.shape(), shape, "polar_spectrum: spectrum shapes");
        assert_eq!(head.shape().rows, shape.rows, "polar_spectrum: frames");
        assert_eq!(head.shape().cols, 2 * shape.cols, "polar_spectrum: head width");

        // SAFETY: one thread per frequency; reads stay within the checked head shape.
        unsafe {
            self.kernels.synthesis.polar_spectrum(
                &self.stream,
                launch::elementwise(real.len()),
                &head.buffer,
                &mut real.buffer,
                &mut imaginary.buffer,
                shape.cols,
                max_log,
            )
        }
    }

    /// The gradient of [`Gpu::polar_spectrum`] with respect to the head.
    #[allow(clippy::too_many_arguments)]
    pub fn polar_spectrum_backward(
        &self,
        head: &Tensor,
        real: &Tensor,
        imaginary: &Tensor,
        real_grad: &Tensor,
        imaginary_grad: &Tensor,
        head_grad: &mut Tensor,
        max_log: f32,
    ) -> GpuResult {
        let shape = real.shape();
        assert!(
            [imaginary.shape(), real_grad.shape(), imaginary_grad.shape()].iter().all(|other| *other == shape),
            "polar_spectrum_backward: shapes"
        );
        assert_eq!(head_grad.shape(), head.shape(), "polar_spectrum_backward: head shapes");
        assert_eq!(head.shape().cols, 2 * shape.cols, "polar_spectrum_backward: head width");

        // SAFETY: one thread per frequency; reads and writes stay within the checked shapes.
        unsafe {
            self.kernels.synthesis.polar_spectrum_backward(
                &self.stream,
                launch::elementwise(real.len()),
                &head.buffer,
                &real.buffer,
                &imaginary.buffer,
                &real_grad.buffer,
                &imaginary_grad.buffer,
                &mut head_grad.buffer,
                shape.cols,
                max_log,
            )
        }
    }

    /// The gradient of [`Gpu::power_spectrum`] with respect to both parts.
    pub fn power_spectrum_backward(
        &self,
        real: &Tensor,
        imaginary: &Tensor,
        power_grad: &Tensor,
        real_grad: &mut Tensor,
        imaginary_grad: &mut Tensor,
    ) -> GpuResult {
        let shape = real.shape();
        assert!(
            [imaginary.shape(), power_grad.shape(), real_grad.shape(), imaginary_grad.shape()].iter().all(|other| *other == shape),
            "power_spectrum_backward: shapes"
        );

        // SAFETY: one thread per element of tensors of equal length.
        unsafe {
            self.kernels.synthesis.power_spectrum_backward(
                &self.stream,
                launch::elementwise(real.len()),
                &real.buffer,
                &imaginary.buffer,
                &power_grad.buffer,
                &mut real_grad.buffer,
                &mut imaginary_grad.buffer,
            )
        }
    }

    /// The gradient of [`Gpu::log_compress`], given the energies before it.
    pub fn log_compress_backward(&self, energies: &Tensor, output_grad: &Tensor, input_grad: &mut Tensor, floor: f32) -> GpuResult {
        assert!(energies.shape() == output_grad.shape() && energies.shape() == input_grad.shape(), "log_compress_backward: shapes");

        // SAFETY: one thread per element of tensors of equal length.
        unsafe {
            self.kernels.synthesis.log_compress_backward(
                &self.stream,
                launch::elementwise(energies.len()),
                &energies.buffer,
                &output_grad.buffer,
                &mut input_grad.buffer,
                floor,
            )
        }
    }

    /// The gradient of a mean absolute error, `scale · sign(estimate − target)`.
    pub fn absolute_error_gradient(&self, estimate: &Tensor, target: &Tensor, gradient: &mut Tensor, scale: f32) -> GpuResult {
        assert!(estimate.shape() == target.shape() && estimate.shape() == gradient.shape(), "absolute_error_gradient: shapes");

        // SAFETY: one thread per element of tensors of equal length.
        unsafe {
            self.kernels.synthesis.absolute_error_gradient(
                &self.stream,
                launch::elementwise(estimate.len()),
                &estimate.buffer,
                &target.buffer,
                &mut gradient.buffer,
                scale,
            )
        }
    }

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
            self.kernels.synthesis.depthwise_convolution(
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
            self.kernels.synthesis.depthwise_convolution_input_grad(
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
            self.kernels.synthesis.depthwise_convolution_weight_grad(
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
            self.kernels.synthesis.unfold(&self.stream, launch::elementwise(columns.len()), &input.buffer, &mut columns.buffer, shape)
        }
    }

    /// `output[r, c] = values[r, c] · scale[c]`.
    pub fn scale_columns(&self, values: &Tensor, scale: &Tensor, output: &mut Tensor) -> GpuResult {
        let shape = values.shape();
        assert_eq!(scale.len(), shape.cols, "scale_columns: one factor per column");
        assert_eq!(output.shape(), shape, "scale_columns: output shape");

        // SAFETY: one thread per element of tensors of equal length.
        unsafe {
            self.kernels.synthesis.scale_columns(
                &self.stream,
                launch::elementwise(values.len()),
                &values.buffer,
                &scale.buffer,
                &mut output.buffer,
                shape.cols,
            )
        }
    }

    /// `scale_grad[c] = Σ_r output_grad[r, c] · values[r, c]`, the gradient of
    /// [`Gpu::scale_columns`] with respect to its factors.
    pub fn column_products(&self, output_grad: &Tensor, values: &Tensor, scale_grad: &mut Tensor) -> GpuResult {
        let shape = values.shape();
        assert_eq!(output_grad.shape(), shape, "column_products: shapes");
        assert_eq!(scale_grad.len(), shape.cols, "column_products: one sum per column");

        // SAFETY: one thread per column; each reads its own column of both tensors.
        unsafe {
            self.kernels.synthesis.column_products(
                &self.stream,
                launch::elementwise(shape.cols),
                &output_grad.buffer,
                &values.buffer,
                &mut scale_grad.buffer,
                shape.rows,
                shape.cols,
            )
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
