//! One-dimensional convolutions over time, used by the neural vocoder.
//!
//! The vocoder's input layer mixes every channel of seven neighboring frames;
//! [`kernels::unfold`] lays each neighborhood out as one row, so that this
//! becomes a single matrix product (Chellapilla, Puri and Simard, 2006,
//! "High Performance Convolutional Neural Networks for Document Processing",
//! <https://inria.hal.science/inria-00112631>). Its ConvNeXt blocks filter each
//! channel on its own over time, a depthwise convolution (Liu et al., 2022,
//! <https://arxiv.org/abs/2201.03545>), which needs its own kernels and their
//! gradients.

use cuda_device::cuda_module;

/// The shape of a one-dimensional convolution over time, applied to a batch of
/// sequences stored one after another, `[sequences · time, channels]`.
/// Taps that would read outside a sequence read zero, so padding never mixes
/// two sequences.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Convolution {
    /// Rows of the input and the output: sequences times frames per sequence.
    pub rows: usize,

    /// Frames per sequence.
    pub time: usize,

    /// Channels per frame.
    pub channels: usize,

    /// Taps of the filter, an odd number; tap `k` reads frame `t + k − kernel / 2`.
    pub kernel: usize,
}

impl Convolution {
    /// The row read by tap `tap` for output row `row`, or `None` when that
    /// frame lies outside the row's sequence.
    pub fn source(&self, row: usize, tap: usize) -> Option<usize> {
        let frame = (row % self.time) as isize + tap as isize - (self.kernel / 2) as isize;
        if frame < 0 || frame as usize >= self.time {
            return None;
        }

        let first_row_of_sequence = row - row % self.time;
        Some(first_row_of_sequence + frame as usize)
    }
}

#[cuda_module]
pub mod kernels {
    use super::Convolution;
    use crate::kernels::device::{global_thread, store};
    use cuda_device::{DisjointSlice, kernel};

    /// A depthwise convolution over time: every channel is filtered with its
    /// own `kernel` taps, `output[t, c] = bias[c] + Σ_k weight[k, c] · input[t + k − kernel/2, c]`.
    /// One thread per output element.
    ///
    /// * `input`, `output`: `[rows, channels]`.
    /// * `weight`: `[kernel, channels]`; `bias`: `[channels]`.
    #[kernel]
    pub fn depthwise_convolution(input: &[f32], weight: &[f32], bias: &[f32], mut output: DisjointSlice<f32>, shape: Convolution) {
        let element = global_thread();
        if element >= shape.rows * shape.channels {
            return;
        }

        let row = element / shape.channels;
        let channel = element % shape.channels;

        let mut sum = bias[channel];
        for tap in 0..shape.kernel {
            if let Some(source) = shape.source(row, tap) {
                sum += weight[tap * shape.channels + channel] * input[source * shape.channels + channel];
            }
        }

        store(&mut output, element, sum);
    }

    /// The gradient of [`depthwise_convolution`] with respect to its input:
    /// each input frame collects the output gradients of every frame whose
    /// filter reached it. One thread per input element.
    #[kernel]
    pub fn depthwise_convolution_input_grad(output_grad: &[f32], weight: &[f32], mut input_grad: DisjointSlice<f32>, shape: Convolution) {
        let element = global_thread();
        if element >= shape.rows * shape.channels {
            return;
        }

        let row = element / shape.channels;
        let channel = element % shape.channels;

        let mut sum = 0.0f32;
        for tap in 0..shape.kernel {
            // Output row `r` read this row through tap `tap` when `r + tap − kernel/2 = row`,
            // that is, through the mirrored tap from this row's point of view.
            if let Some(reader) = shape.source(row, shape.kernel - 1 - tap) {
                sum += weight[tap * shape.channels + channel] * output_grad[reader * shape.channels + channel];
            }
        }

        store(&mut input_grad, element, sum);
    }

    /// The gradient of [`depthwise_convolution`] with respect to its weights:
    /// `∂weight[k, c] = Σ_t ∂output[t, c] · input[t + k − kernel/2, c]`. One
    /// thread per weight, which sums over every row in a fixed order.
    #[kernel]
    pub fn depthwise_convolution_weight_grad(output_grad: &[f32], input: &[f32], mut weight_grad: DisjointSlice<f32>, shape: Convolution) {
        let element = global_thread();
        if element >= shape.kernel * shape.channels {
            return;
        }

        let tap = element / shape.channels;
        let channel = element % shape.channels;

        let mut sum = 0.0f32;
        for row in 0..shape.rows {
            if let Some(source) = shape.source(row, tap) {
                sum += output_grad[row * shape.channels + channel] * input[source * shape.channels + channel];
            }
        }

        store(&mut weight_grad, element, sum);
    }

    /// Lays out every frame's neighborhood in one row, so that a convolution
    /// across channels becomes a matrix product:
    /// `columns[t, k · channels + c] = input[t + k − kernel/2, c]`, or 0 outside
    /// the sequence. One thread per output element.
    ///
    /// * `columns`: `[rows, kernel · channels]`, written.
    #[kernel]
    pub fn unfold(input: &[f32], mut columns: DisjointSlice<f32>, shape: Convolution) {
        let element = global_thread();
        let width = shape.kernel * shape.channels;
        if element >= shape.rows * width {
            return;
        }

        let row = element / width;
        let tap = (element % width) / shape.channels;
        let channel = element % shape.channels;

        let value = match shape.source(row, tap) {
            Some(source) => input[source * shape.channels + channel],
            None => 0.0,
        };

        store(&mut columns, element, value);
    }
}
