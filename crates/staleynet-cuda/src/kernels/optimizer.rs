//! The AdamW update and the sum of squares used for gradient clipping.

use cuda_device::cuda_module;

/// The settings of one AdamW step, passed to the kernel by value (Loshchilov
/// and Hutter, 2019, <https://arxiv.org/abs/1711.05101>, Algorithm 2).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdamWStep {
    /// Step size for this step, after scheduling.
    pub learning_rate: f32,

    /// Decay rate β₁ of the running mean of the gradient.
    pub beta1: f32,

    /// Decay rate β₂ of the running mean of the squared gradient.
    pub beta2: f32,

    /// Added to the square root of the second moment so it cannot divide by zero.
    pub epsilon: f32,

    /// Decoupled weight decay (0 for gains, biases and embeddings).
    pub weight_decay: f32,

    /// `1 − β₁ᵗ`, which removes the bias of the first moment at step `t`.
    pub first_correction: f32,

    /// `1 − β₂ᵗ`, which removes the bias of the second moment at step `t`.
    pub second_correction: f32,
}

#[cuda_module]
pub mod kernels {
    use super::AdamWStep;
    use crate::kernels::device::{global_thread, load_owned, store};
    use crate::kernels::{WARP_SIZE, WARPS_PER_BLOCK};
    use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, thread, warp};

    /// Updates every parameter and its two moment estimates. One thread per parameter.
    ///
    /// * `parameter`: the weights, updated in place.
    /// * `gradient`: their gradient.
    /// * `first_moment`: the running mean of the gradient, updated in place.
    /// * `second_moment`: the running mean of its square, updated in place.
    /// * `settings`: learning rate, decay rates and bias corrections.
    #[kernel]
    pub fn adamw_step(
        mut parameter: DisjointSlice<f32>,
        gradient: &[f32],
        mut first_moment: DisjointSlice<f32>,
        mut second_moment: DisjointSlice<f32>,
        settings: AdamWStep,
    ) {
        let index = global_thread();
        if index >= parameter.len() {
            return;
        }
        let grad = gradient[index];

        let mean = settings.beta1 * load_owned(&mut first_moment, index) + (1.0 - settings.beta1) * grad;
        let variance = settings.beta2 * load_owned(&mut second_moment, index) + (1.0 - settings.beta2) * grad * grad;
        store(&mut first_moment, index, mean);
        store(&mut second_moment, index, variance);

        let mean_hat = mean / settings.first_correction;
        let variance_hat = variance / settings.second_correction;
        let weight = load_owned(&mut parameter, index);
        let update = mean_hat / (variance_hat.sqrt() + settings.epsilon) + settings.weight_decay * weight;
        store(&mut parameter, index, weight - settings.learning_rate * update);
    }

    /// Partial sums of squares: the threads sweep `values` with a stride of
    /// the whole grid, each warp adds its threads' sums, and the first thread
    /// of block `b` writes the block's sum to `partial_sums[b]`. The host adds
    /// the partial sums.
    ///
    /// * `values`: the numbers whose squares are summed.
    /// * `partial_sums`: one result per block.
    #[kernel]
    #[launch_bounds(256)]
    pub fn sum_of_squares(values: &[f32], mut partial_sums: DisjointSlice<f32>) {
        static mut WARP_SUMS: SharedArray<f32, WARPS_PER_BLOCK> = SharedArray::UNINIT;

        let local = thread::threadIdx_x() as usize;
        let grid_stride = thread::gridDim_x() as usize * thread::blockDim_x() as usize;

        let mut sum = 0.0f32;
        for index in (global_thread()..values.len()).step_by(grid_stride) {
            sum += values[index] * values[index];
        }
        let warp_sum = warp::reduce_sum_f32(sum);

        if local % WARP_SIZE == 0 {
            // SAFETY: shared memory private to the block; local / 32 < 8 for a block of 256 threads.
            unsafe { WARP_SUMS[local / WARP_SIZE] = warp_sum };
        }
        thread::sync_threads();

        if local != 0 {
            return;
        }
        let mut block_sum = 0.0f32;
        for warp_index in 0..WARPS_PER_BLOCK {
            // SAFETY: warp_index < 8; every warp wrote its slot before the barrier above.
            block_sum += unsafe { WARP_SUMS[warp_index] };
        }
        store(&mut partial_sums, thread::blockIdx_x() as usize, block_sum);
    }
}
