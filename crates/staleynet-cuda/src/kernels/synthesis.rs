//! Kernels of the neural vocoder: the inverse short-time Fourier transform,
//! the spectrum the network predicts, the one-dimensional convolutions of its
//! backbone, and the gradients of each.
//!
//! The vocoder follows Vocos (Siuzdak, 2023, <https://arxiv.org/abs/2306.00814>):
//! for every frame it predicts a log-magnitude and a phase per frequency, the
//! two become a complex spectrum, an inverse Fourier transform (a matrix
//! product) turns each frame into samples, and overlap-add joins the frames
//! into a waveform. Every kernel here writes each output element from exactly
//! one thread, so results are the same on every run.

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

        Some(row - row % self.time + frame as usize)
    }
}

#[cuda_module]
pub mod kernels {
    use super::Convolution;
    use crate::kernels::device::{global_thread, store};
    use cuda_device::{DisjointSlice, kernel};

    /// Overlap-add: joins frames into a signal, `signal[s] = normalization[s] ·
    /// Σ_f frames[f, s − f · hop]` over the frames that cover sample `s`. One
    /// thread per output sample, so no two threads add into the same place.
    ///
    /// * `frames`: `[frame count, frame_length]`, already multiplied by the
    ///   synthesis window.
    /// * `normalization`: one factor per output sample, the inverse of the
    ///   summed squared window at that sample.
    /// * `signal`: `(frame count − 1) · hop + frame_length` samples, written.
    #[kernel]
    pub fn overlap_add(
        frames: &[f32],
        normalization: &[f32],
        mut signal: DisjointSlice<f32>,
        frame_count: usize,
        frame_length: usize,
        hop: usize,
    ) {
        let sample = global_thread();
        if sample >= signal.len() {
            return;
        }

        let last = (sample / hop).min(frame_count - 1);
        let first = if sample >= frame_length { (sample - frame_length) / hop + 1 } else { 0 };

        let mut sum = 0.0f32;
        for frame in first..=last {
            sum += frames[frame * frame_length + sample - frame * hop];
        }
        store(&mut signal, sample, sum * normalization[sample]);
    }

    /// The gradient of [`overlap_add`]: every frame sample receives the
    /// gradient of the signal sample it was added into, times that sample's
    /// normalization. One thread per frame sample.
    ///
    /// * `signal_grad`: the gradient of the joined signal.
    /// * `normalization`: as in [`overlap_add`].
    /// * `frames_grad`: `[frame count, frame_length]`, written.
    #[kernel]
    pub fn overlap_add_backward(
        signal_grad: &[f32],
        normalization: &[f32],
        mut frames_grad: DisjointSlice<f32>,
        frame_length: usize,
        hop: usize,
    ) {
        let element = global_thread();
        let sample = (element / frame_length) * hop + element % frame_length;

        if element < frames_grad.len() {
            store(&mut frames_grad, element, signal_grad[sample] * normalization[sample]);
        }
    }

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
        let (re, im) = (real[element], imaginary[element]);
        let (d_re, d_im) = (real_grad[element], imaginary_grad[element]);

        let capped = head[row * 2 * bins + bin] > max_log;
        let magnitude_grad = if capped { 0.0 } else { re * d_re + im * d_im };
        store(&mut head_grad, row * 2 * bins + bin, magnitude_grad);
        store(&mut head_grad, row * 2 * bins + bins + bin, re * d_im - im * d_re);
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
        store(&mut input_grad, element, if energy > floor { output_grad[element] / energy } else { 0.0 });
    }

    /// The gradient of a mean absolute error: `scale · sign(estimate − target)`.
    /// One thread per element.
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

    /// Multiplies every column by its own factor, `output[r, c] = values[r, c] ·
    /// scale[c]`; the per-channel "layer scale" of ConvNeXt blocks. One thread
    /// per element.
    #[kernel]
    pub fn scale_columns(values: &[f32], scale: &[f32], mut output: DisjointSlice<f32>, columns: usize) {
        let element = global_thread();
        if element < values.len() {
            store(&mut output, element, values[element] * scale[element % columns]);
        }
    }

    /// The gradient of [`scale_columns`] with respect to the factors:
    /// `∂scale[c] = Σ_r ∂output[r, c] · values[r, c]`. One thread per column,
    /// summing its rows in a fixed order.
    #[kernel]
    pub fn column_products(output_grad: &[f32], values: &[f32], mut scale_grad: DisjointSlice<f32>, rows: usize, columns: usize) {
        let column = global_thread();
        if column >= columns {
            return;
        }

        let mut sum = 0.0f32;
        for row in 0..rows {
            sum += output_grad[row * columns + column] * values[row * columns + column];
        }
        store(&mut scale_grad, column, sum);
    }
}
