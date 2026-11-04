//! The vocoder's predicted spectrum and the gradients of its spectral loss.
//! See [`crate::kernels::spectrum`].

use crate::device::{Gpu, GpuResult, Tensor};
use crate::launch;

impl Gpu {
    /// The complex spectrum predicted by the network's head (`[frames, 2 · bins]`,
    /// log-magnitudes then phases), with log-magnitudes capped at `max_log`.
    pub fn polar_spectrum(&self, head: &Tensor, real: &mut Tensor, imaginary: &mut Tensor, max_log: f32) -> GpuResult {
        let shape = real.shape();
        assert_eq!(imaginary.shape(), shape, "polar_spectrum: spectrum shapes");
        assert_eq!(head.shape().rows, shape.rows, "polar_spectrum: frames");
        assert_eq!(head.shape().cols, 2 * shape.cols, "polar_spectrum: head width");

        // SAFETY: one thread per frequency; reads stay within the checked head shape.
        unsafe {
            self.kernels.spectrum.polar_spectrum(
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
        let spectrum_shapes = [imaginary.shape(), real_grad.shape(), imaginary_grad.shape()];
        assert!(spectrum_shapes.iter().all(|other| *other == shape), "polar_spectrum_backward: shapes");
        assert_eq!(head_grad.shape(), head.shape(), "polar_spectrum_backward: head shapes");
        assert_eq!(head.shape().cols, 2 * shape.cols, "polar_spectrum_backward: head width");

        // SAFETY: one thread per frequency; reads and writes stay within the checked shapes.
        unsafe {
            self.kernels.spectrum.polar_spectrum_backward(
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
        let other_shapes = [imaginary.shape(), power_grad.shape(), real_grad.shape(), imaginary_grad.shape()];
        assert!(other_shapes.iter().all(|other| *other == shape), "power_spectrum_backward: shapes");

        // SAFETY: one thread per element of tensors of equal length.
        unsafe {
            self.kernels.spectrum.power_spectrum_backward(
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
        assert_eq!(output_grad.shape(), energies.shape(), "log_compress_backward: output gradient shape");
        assert_eq!(input_grad.shape(), energies.shape(), "log_compress_backward: input gradient shape");

        // SAFETY: one thread per element of tensors of equal length.
        unsafe {
            self.kernels.spectrum.log_compress_backward(
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
        assert_eq!(target.shape(), estimate.shape(), "absolute_error_gradient: target shape");
        assert_eq!(gradient.shape(), estimate.shape(), "absolute_error_gradient: gradient shape");

        // SAFETY: one thread per element of tensors of equal length.
        unsafe {
            self.kernels.spectrum.absolute_error_gradient(
                &self.stream,
                launch::elementwise(estimate.len()),
                &estimate.buffer,
                &target.buffer,
                &mut gradient.buffer,
                scale,
            )
        }
    }
}
