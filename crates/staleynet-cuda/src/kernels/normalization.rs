//! RMS normalisation and its gradients (Zhang and Sennrich, 2019,
//! <https://arxiv.org/abs/1910.07467>).
//!
//! For a row `x` of width `n`, the forward pass computes
//! `s = 1 / √(mean(x²) + ε)` and `y = x · s · g`, where `g` is a learned gain
//! per column. The row-wise kernels assign one warp to each row.

use cuda_device::cuda_module;

#[cuda_module]
pub mod kernels {
    use crate::kernels::WARP_SIZE;
    use crate::kernels::device::{WarpRow, store, warp_row};
    use cuda_device::{DisjointSlice, kernel, thread, warp};

    /// Normalises every row: `output = input · inv_rms · gain`, and saves
    /// `inv_rms = 1/√(mean(input²) + epsilon)` of each row for the backward pass.
    ///
    /// * `input`: the `[rows, cols]` matrix to normalise.
    /// * `gain`: one learned factor per column.
    /// * `output`: the normalised `[rows, cols]` matrix.
    /// * `inv_rms`: one value per row, written for the backward pass.
    /// * `rows`, `cols`: the size of the matrix.
    /// * `epsilon`: keeps the division finite for a row of zeros.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_forward(
        input: &[f32],
        gain: &[f32],
        mut output: DisjointSlice<f32>,
        mut inv_rms: DisjointSlice<f32>,
        rows: usize,
        cols: usize,
        epsilon: f32,
    ) {
        let WarpRow { row, lane } = warp_row();
        if row >= rows {
            return;
        }
        let base = row * cols;

        let mut square_sum = 0.0f32;
        for col in (lane..cols).step_by(WARP_SIZE) {
            square_sum += input[base + col] * input[base + col];
        }
        let inverse = 1.0 / (warp::reduce_sum_f32(square_sum) / cols as f32 + epsilon).sqrt();

        for col in (lane..cols).step_by(WARP_SIZE) {
            store(&mut output, base + col, input[base + col] * inverse * gain[col]);
        }
        if lane == 0 {
            store(&mut inv_rms, row, inverse);
        }
    }

    /// The gradient with respect to the input of every row. With `s` the saved
    /// `inv_rms` of the row, `dx = s · g ⊙ dy − x · s³ / cols · Σ(dy ⊙ g ⊙ x)`.
    ///
    /// * `output_grad`: the gradient `dy` with respect to the normalised output.
    /// * `input`: the input `x` of the forward pass.
    /// * `gain`: the learned gains `g`.
    /// * `inv_rms`: the values saved by [`rms_norm_forward`].
    /// * `input_grad`: the gradient `dx`, written.
    /// * `rows`, `cols`: the size of the matrix.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_backward_input(
        output_grad: &[f32],
        input: &[f32],
        gain: &[f32],
        inv_rms: &[f32],
        mut input_grad: DisjointSlice<f32>,
        rows: usize,
        cols: usize,
    ) {
        let WarpRow { row, lane } = warp_row();
        if row >= rows {
            return;
        }
        let base = row * cols;
        let inverse = inv_rms[row];

        let mut projection = 0.0f32;
        for col in (lane..cols).step_by(WARP_SIZE) {
            projection += output_grad[base + col] * gain[col] * input[base + col];
        }
        let correction = inverse * inverse * inverse * warp::reduce_sum_f32(projection) / cols as f32;

        for col in (lane..cols).step_by(WARP_SIZE) {
            store(&mut input_grad, base + col, inverse * gain[col] * output_grad[base + col] - input[base + col] * correction);
        }
    }

    /// Adds the gradient with respect to the gains:
    /// `gain_grad[c] += Σ_r dy[r, c] · x[r, c] · s[r]`. One thread per column.
    ///
    /// * `output_grad`: the gradient `dy` with respect to the normalised output.
    /// * `input`: the input `x` of the forward pass.
    /// * `inv_rms`: the values `s` saved by [`rms_norm_forward`].
    /// * `gain_grad`: one accumulator per column.
    /// * `rows`, `cols`: the size of the matrix.
    #[kernel]
    pub fn rms_norm_backward_gain(
        output_grad: &[f32],
        input: &[f32],
        inv_rms: &[f32],
        mut gain_grad: DisjointSlice<f32>,
        rows: usize,
        cols: usize,
    ) {
        let index = thread::index_1d();
        let col = index.get();
        if col >= cols {
            return;
        }

        let mut sum = 0.0f32;
        for row in 0..rows {
            sum += output_grad[row * cols + col] * input[row * cols + col] * inv_rms[row];
        }

        if let Some(slot) = gain_grad.get_mut(index) {
            *slot += sum;
        }
    }
}
