//! AdamW update and gradient-norm reduction.

use crate::device::{Gpu, GpuResult, Shape, Tensor};
use crate::kernels::optimizer::AdamWStep;
use crate::launch;

/// Per-step AdamW hyperparameters (Loshchilov & Hutter, 2019,
/// <https://arxiv.org/abs/1711.05101>).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWHyperparameters {
    /// Step size for this step (after scheduling).
    pub learning_rate: f32,

    /// Decay rate of the first-moment (mean) estimate, β₁.
    pub beta1: f32,

    /// Decay rate of the second-moment (uncentred variance) estimate, β₂.
    pub beta2: f32,

    /// Added to √v̂ to avoid division by zero.
    pub epsilon: f32,

    /// Decoupled weight decay coefficient (0 for gains, biases, embeddings).
    pub weight_decay: f32,

    /// 1-based index of this step, used for bias correction.
    pub step: u32,
}

impl Gpu {
    /// Updates `parameter` and its moment estimates in place.
    pub fn adamw_step(
        &self,
        parameter: &mut Tensor,
        gradient: &Tensor,
        first_moment: &mut Tensor,
        second_moment: &mut Tensor,
        hyper: &AdamWHyperparameters,
    ) -> GpuResult {
        let shape = parameter.shape();
        assert!(gradient.shape() == shape && first_moment.shape() == shape && second_moment.shape() == shape, "adamw_step: shape mismatch");

        let step = hyper.step as i32;
        let settings = AdamWStep {
            learning_rate: hyper.learning_rate,
            beta1: hyper.beta1,
            beta2: hyper.beta2,
            epsilon: hyper.epsilon,
            weight_decay: hyper.weight_decay,
            first_correction: 1.0 - hyper.beta1.powi(step),
            second_correction: 1.0 - hyper.beta2.powi(step),
        };

        // SAFETY: one thread per element; the four tensors have the same shape, asserted above.
        unsafe {
            self.kernels.optimizer.adamw_step(
                &self.stream,
                launch::elementwise(parameter.len()),
                &mut parameter.buffer,
                &gradient.buffer,
                &mut first_moment.buffer,
                &mut second_moment.buffer,
                settings,
            )
        }
    }

    /// `Σ values²`, reduced per block on the GPU and across blocks on the host.
    pub fn sum_of_squares(&self, values: &Tensor) -> GpuResult<f64> {
        let mut partial_sums = self.zeros(Shape::vector(launch::REDUCTION_BLOCKS))?;

        // SAFETY: fixed full-block grid matching the kernel's shared-memory size;
        // each block writes only its own partial sum.
        unsafe { self.kernels.optimizer.sum_of_squares(&self.stream, launch::reduction(), &values.buffer, &mut partial_sums.buffer)? };

        let partials = self.download(&partial_sums)?;
        Ok(partials.iter().map(|&partial| partial as f64).sum())
    }
}
