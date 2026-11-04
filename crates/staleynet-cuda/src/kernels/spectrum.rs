//! The spectrum the neural vocoder predicts, and the gradients of the spectral
//! loss that trains it.
//!
//! Following Vocos (Siuzdak, 2023, <https://arxiv.org/abs/2306.00814>), the
//! network predicts a log-magnitude and a phase for every frequency of every
//! frame, and the two become a complex spectrum. The loss compares log
//! energies of the generated and the recorded audio by their mean absolute
//! difference; its forward pieces (power spectrum, logarithm) are the
//! tokenizer's kernels in [`super::signal`], and their gradients live here.

use cuda_device::cuda_module;

#[cuda_module]
pub mod kernels {
    use crate::kernels::device::{global_thread, store};
    use cuda_device::{DisjointSlice, kernel};

    /// The spectrum the network predicts: for frame `f` and frequency `k`,
    /// magnitude `exp(min(m, max_log))` and phase `p`, from `head = [m | p]`.
    /// `real = magnitude · cos p` and `imaginary = magnitude · sin p`. Capping
    /// the log-magnitude keeps an untrained network from producing overflows.
    ///
    /// * `head`: `[frames, 2 · bins]`, log-magnitudes then phases.
    /// * `real`, `imaginary`: `[frames, bins]`, written.
    #[kernel]
    pub fn polar_spectrum(head: &[f32], mut real: DisjointSlice<f32>, mut imaginary: DisjointSlice<f32>, bins: usize, max_log: f32) {
        let element = global_thread();
        if element >= real.len() {
            return;
        }

        let row = element / bins;
        let bin = element % bins;

        let magnitude = libm::expf(head[row * 2 * bins + bin].min(max_log));
        let phase = head[row * 2 * bins + bins + bin];

        store(&mut real, element, magnitude * libm::cosf(phase));
        store(&mut imaginary, element, magnitude * libm::sinf(phase));
    }

    /// The gradient of [`polar_spectrum`]. With `re = e^m cos p` and
    /// `im = e^m sin p`: `∂/∂m = re · ∂re + im · ∂im` (zero where `m` was
    /// capped) and `∂/∂p = −im · ∂re + re · ∂im`. One thread per frequency.
    ///
    /// * `head`: the network's output, as in [`polar_spectrum`].
    /// * `real`, `imaginary`: the spectrum [`polar_spectrum`] produced.
    /// * `real_grad`, `imaginary_grad`: their gradients.
    /// * `head_grad`: `[frames, 2 · bins]`, written.
    #[kernel]
    pub fn polar_spectrum_backward(
        head: &[f32],
        real: &[f32],
        imaginary: &[f32],
        real_grad: &[f32],
        imaginary_grad: &[f32],
        mut head_grad: DisjointSlice<f32>,
        bins: usize,
        max_log: f32,
    ) {
        let element = global_thread();
        if element >= real.len() {
            return;
        }

        let row = element / bins;
        let bin = element % bins;

        let magnitude_slot = row * 2 * bins + bin;
        let phase_slot = magnitude_slot + bins;

        let real_part = real[element];
        let imaginary_part = imaginary[element];
        let real_part_grad = real_grad[element];
        let imaginary_part_grad = imaginary_grad[element];

        let capped = head[magnitude_slot] > max_log;
        let magnitude_grad = if capped { 0.0 } else { real_part * real_part_grad + imaginary_part * imaginary_part_grad };
        let phase_grad = real_part * imaginary_part_grad - imaginary_part * real_part_grad;

        store(&mut head_grad, magnitude_slot, magnitude_grad);
        store(&mut head_grad, phase_slot, phase_grad);
    }

    /// The gradient of `power = re² + im²`: `∂re = 2 re ∂power` and
    /// `∂im = 2 im ∂power`. One thread per element.
    #[kernel]
    pub fn power_spectrum_backward(
        real: &[f32],
        imaginary: &[f32],
        power_grad: &[f32],
        mut real_grad: DisjointSlice<f32>,
        mut imaginary_grad: DisjointSlice<f32>,
    ) {
        let element = global_thread();
        if element >= real.len() {
            return;
        }

        store(&mut real_grad, element, 2.0 * real[element] * power_grad[element]);
        store(&mut imaginary_grad, element, 2.0 * imaginary[element] * power_grad[element]);
    }

    /// The gradient of `ln(max(x, floor))`: `∂x = ∂y / x` where `x` exceeded
    /// the floor, and 0 where the floor applied. One thread per element.
    ///
    /// * `energies`: the values before the logarithm.
    #[kernel]
    pub fn log_compress_backward(energies: &[f32], output_grad: &[f32], mut input_grad: DisjointSlice<f32>, floor: f32) {
        let element = global_thread();
        if element >= energies.len() {
            return;
        }

        let energy = energies[element];
        let gradient = if energy > floor { output_grad[element] / energy } else { 0.0 };

        store(&mut input_grad, element, gradient);
    }

    /// The gradient of a mean absolute error: `scale · sign(estimate − target)`,
    /// with a sign of 0 where the two agree. One thread per element.
    #[kernel]
    pub fn absolute_error_gradient(estimate: &[f32], target: &[f32], mut gradient: DisjointSlice<f32>, scale: f32) {
        let element = global_thread();
        if element >= estimate.len() {
            return;
        }

        let difference = estimate[element] - target[element];
        let sign = if difference > 0.0 {
            1.0
        } else if difference < 0.0 {
            -1.0
        } else {
            0.0
        };

        store(&mut gradient, element, scale * sign);
    }
}
