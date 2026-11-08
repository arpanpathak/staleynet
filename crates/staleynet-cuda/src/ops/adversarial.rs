//! The discriminators' kernels: strided unfolding and its gradient, period
//! splitting and its gradient, the leaky ReLU, and the least-squares
//! gradient. See [`crate::kernels::adversarial`].

use crate::device::{Gpu, GpuResult, Tensor};
pub use crate::kernels::adversarial::{PeriodSplit, StridedConvolution};
use crate::launch;

/// Which side of the least-squares objective a batch of scores is pushed
/// towards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The audio is real: scores are pushed towards 1.
    Real,

    /// The audio is synthesized: scores are pushed towards 0.
    Synthesized,
}

impl Verdict {
    /// The score the objective pulls towards.
    pub fn target(self) -> f32 {
        match self {
            Verdict::Real => 1.0,
            Verdict::Synthesized => 0.0,
        }
    }
}

impl Gpu {
    /// Lays out each output frame's neighborhood as one row of `columns`, so
    /// that the strided convolution `shape` becomes one matrix product.
    pub fn unfold_strided(&self, input: &Tensor, columns: &mut Tensor, shape: StridedConvolution) -> GpuResult {
        assert_eq!(input.len(), shape.input_rows() * shape.channels, "unfold_strided: input size");
        assert_eq!(columns.len(), shape.output_rows() * shape.width(), "unfold_strided: columns size");

        // SAFETY: one thread per column element; padding taps write zero.
        unsafe {
            self.kernels.adversarial.unfold_strided(
                &self.stream,
                launch::elementwise(columns.len()),
                &input.buffer,
                &mut columns.buffer,
                shape,
            )
        }
    }

    /// The gradient of [`Gpu::unfold_strided`]: writes `input_grad` from `columns_grad`.
    pub fn unfold_strided_backward(&self, columns_grad: &Tensor, input_grad: &mut Tensor, shape: StridedConvolution) -> GpuResult {
        assert_eq!(columns_grad.len(), shape.output_rows() * shape.width(), "unfold_strided_backward: columns size");
        assert_eq!(input_grad.len(), shape.input_rows() * shape.channels, "unfold_strided_backward: input size");

        // SAFETY: one thread per input element, each gathering its own sum.
        unsafe {
            self.kernels.adversarial.unfold_strided_backward(
                &self.stream,
                launch::elementwise(input_grad.len()),
                &columns_grad.buffer,
                &mut input_grad.buffer,
                shape,
            )
        }
    }

    /// Splits each signal of `signals` into its `layout.period` interleaved sequences.
    pub fn split_periods(&self, signals: &Tensor, split: &mut Tensor, layout: PeriodSplit) -> GpuResult {
        assert_eq!(signals.len(), layout.signals * layout.length, "split_periods: signals size");
        assert_eq!(split.len(), layout.sequences() * layout.frames(), "split_periods: split size");

        // SAFETY: one thread per element of the split batch.
        unsafe {
            self.kernels.adversarial.split_periods(
                &self.stream,
                launch::elementwise(split.len()),
                &signals.buffer,
                &mut split.buffer,
                layout,
            )
        }
    }

    /// Adds the gradient of [`Gpu::split_periods`] into `signals_grad`.
    pub fn split_periods_backward(&self, split_grad: &Tensor, signals_grad: &mut Tensor, layout: PeriodSplit) -> GpuResult {
        assert_eq!(split_grad.len(), layout.sequences() * layout.frames(), "split_periods_backward: split size");
        assert_eq!(signals_grad.len(), layout.signals * layout.length, "split_periods_backward: signals size");

        // SAFETY: one thread per sample; `get_mut` guards the tail.
        unsafe {
            self.kernels.adversarial.split_periods_backward(
                &self.stream,
                launch::elementwise(signals_grad.len()),
                &split_grad.buffer,
                &mut signals_grad.buffer,
                layout,
            )
        }
    }

    /// `output = leaky_relu(input)`.
    pub fn leaky_relu(&self, input: &Tensor, output: &mut Tensor) -> GpuResult {
        assert_eq!(input.shape(), output.shape(), "leaky_relu: shape mismatch");

        // SAFETY: one thread per element.
        unsafe {
            self.kernels
                .adversarial
                .leaky_relu(&self.stream, launch::elementwise(output.len()), &input.buffer, &mut output.buffer)
        }
    }

    /// `input_grad = output_grad ⊙ leaky_relu'(input)`.
    pub fn leaky_relu_backward(&self, input: &Tensor, output_grad: &Tensor, input_grad: &mut Tensor) -> GpuResult {
        assert_eq!(input.shape(), output_grad.shape(), "leaky_relu_backward: shape mismatch");
        assert_eq!(input.shape(), input_grad.shape(), "leaky_relu_backward: shape mismatch");

        // SAFETY: one thread per element.
        unsafe {
            self.kernels.adversarial.leaky_relu_backward(
                &self.stream,
                launch::elementwise(input_grad.len()),
                &input.buffer,
                &output_grad.buffer,
                &mut input_grad.buffer,
            )
        }
    }

    /// The gradient of `scale · Σ (score − target)²` for the scores of
    /// `verdict`'s side, written into `gradient`.
    pub fn least_squares_gradient(&self, scores: &Tensor, gradient: &mut Tensor, verdict: Verdict, scale: f32) -> GpuResult {
        assert_eq!(scores.shape(), gradient.shape(), "least_squares_gradient: shape mismatch");

        // SAFETY: one thread per score.
        unsafe {
            self.kernels.adversarial.least_squares_gradient(
                &self.stream,
                launch::elementwise(gradient.len()),
                &scores.buffer,
                &mut gradient.buffer,
                verdict.target(),
                scale,
            )
        }
    }
}
