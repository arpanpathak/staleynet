//! Elementwise operations, one thread per output element: scaled addition,
//! scaling, bias, column sums, per-column scales and the GELU activation with
//! its derivative.

use cuda_device::cuda_module;

/// √(2/π), a constant of the tanh approximation of GELU (Hendrycks and
/// Gimpel, 2016, <https://arxiv.org/abs/1606.08415>, §2).
pub const GELU_SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// The cubic coefficient of the same approximation.
pub const GELU_CUBIC: f32 = 0.044_715;

/// GELU, tanh approximation: `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`.
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::tanhf(GELU_SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x)))
}

/// The derivative of [`gelu`] with respect to `x`.
pub fn gelu_derivative(x: f32) -> f32 {
    let tanh = libm::tanhf(GELU_SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x));
    let inner_derivative = GELU_SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_CUBIC * x * x);
    0.5 * (1.0 + tanh) + 0.5 * x * (1.0 - tanh * tanh) * inner_derivative
}

#[cuda_module]
pub mod kernels {
    use super::{gelu, gelu_derivative};
    use cuda_device::{DisjointSlice, kernel, thread};

    /// `output[i] += scale · input[i]`.
    ///
    /// * `scale`: the factor applied to `input`.
    /// * `input`: the values added; as long as `output`.
    /// * `output`: the values added to.
    #[kernel]
    pub fn add_scaled(scale: f32, input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let position = index.get();
        if let Some(slot) = output.get_mut(index) {
            *slot += scale * input[position];
        }
    }

    /// `values[i] *= factor`.
    ///
    /// * `values`: the values scaled in place.
    /// * `factor`: the multiplier.
    #[kernel]
    pub fn scale_in_place(mut values: DisjointSlice<f32>, factor: f32) {
        if let Some(slot) = values.get_mut(thread::index_1d()) {
            *slot *= factor;
        }
    }

    /// Adds a bias to every row of a row-major matrix: `output[r, c] += bias[c]`.
    ///
    /// * `output`: the `[rows, cols]` matrix.
    /// * `bias`: one value per column.
    /// * `cols`: the number of columns.
    #[kernel]
    pub fn add_row_bias(mut output: DisjointSlice<f32>, bias: &[f32], cols: usize) {
        let index = thread::index_1d();
        let position = index.get();
        if let Some(slot) = output.get_mut(index) {
            *slot += bias[position % cols];
        }
    }

    /// Adds the sum of every column of a matrix to `sums`:
    /// `sums[c] += Σ_r input[r, c]`. One thread per column, so the threads of a
    /// warp read consecutive addresses of each row.
    ///
    /// * `input`: the `[rows, cols]` matrix, row-major.
    /// * `rows`, `cols`: its size.
    /// * `sums`: one accumulator per column.
    #[kernel]
    pub fn accumulate_column_sums(input: &[f32], rows: usize, cols: usize, mut sums: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let col = index.get();
        if col >= cols {
            return;
        }

        let mut sum = 0.0f32;
        for row in 0..rows {
            sum += input[row * cols + col];
        }

        if let Some(slot) = sums.get_mut(index) {
            *slot += sum;
        }
    }

    /// Multiplies every column by its own factor, `output[r, c] = values[r, c] ·
    /// scale[c]`: the per-channel "layer scale" of ConvNeXt blocks (Touvron et
    /// al., 2021, <https://arxiv.org/abs/2103.17239>).
    ///
    /// * `values`, `output`: `[rows, cols]`, row-major.
    /// * `scale`: one factor per column.
    #[kernel]
    pub fn scale_columns(values: &[f32], scale: &[f32], mut output: DisjointSlice<f32>, cols: usize) {
        let index = thread::index_1d();
        let position = index.get();
        if let Some(slot) = output.get_mut(index) {
            *slot = values[position] * scale[position % cols];
        }
    }

    /// The gradient of [`scale_columns`] with respect to its factors:
    /// `scale_grad[c] = Σ_r output_grad[r, c] · values[r, c]`. One thread per
    /// column, summing its rows in a fixed order.
    ///
    /// * `output_grad`, `values`: `[rows, cols]`, row-major.
    /// * `scale_grad`: one sum per column, written.
    #[kernel]
    pub fn column_products(output_grad: &[f32], values: &[f32], rows: usize, cols: usize, mut scale_grad: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let col = index.get();
        if col >= cols {
            return;
        }

        let mut sum = 0.0f32;
        for row in 0..rows {
            sum += output_grad[row * cols + col] * values[row * cols + col];
        }

        if let Some(slot) = scale_grad.get_mut(index) {
            *slot = sum;
        }
    }

    /// `output[i] = GELU(input[i])`.
    ///
    /// * `input`: the pre-activations.
    /// * `output`: the activations; as long as `input`.
    #[kernel]
    pub fn gelu_forward(input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let position = index.get();
        if let Some(slot) = output.get_mut(index) {
            *slot = gelu(input[position]);
        }
    }

    /// The gradient of GELU: `input_grad[i] = output_grad[i] · GELU'(input[i])`.
    ///
    /// * `input`: the pre-activations of the forward pass.
    /// * `output_grad`: the gradient with respect to the activations.
    /// * `input_grad`: the gradient with respect to the pre-activations.
    #[kernel]
    pub fn gelu_backward(input: &[f32], output_grad: &[f32], mut input_grad: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let position = index.get();
        if let Some(slot) = input_grad.get_mut(index) {
            *slot = output_grad[position] * gelu_derivative(input[position]);
        }
    }
}
