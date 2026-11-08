//! The kernels of the vocoder's discriminators (Kong, Kim and Bae, 2020,
//! "HiFi-GAN", <https://arxiv.org/abs/2010.05646>).
//!
//! A discriminator is a stack of strided convolutions with leaky ReLU
//! activations that ends in one score per frame: high where the audio sounds
//! real, low where it sounds synthesized. A strided convolution is computed as
//! in [`crate::kernels::convolution`]: [`kernels::unfold_strided`] lays out
//! each output frame's neighborhood as one row, and a matrix product applies
//! the filters. The multi-period discriminator first splits a signal into
//! `period` interleaved sequences ([`kernels::split_periods`]), so that a
//! convolution over each sequence sees samples exactly one period apart, which
//! exposes the periodic structure of pitched sound.
//!
//! The discriminators are trained with the least-squares objective of Mao et
//! al., 2017 (<https://arxiv.org/abs/1611.04076>), whose gradient is
//! [`kernels::least_squares_gradient`].

use cuda_device::cuda_module;

/// Slope of the leaky ReLU for negative inputs, HiFi-GAN's value.
pub const LEAKY_SLOPE: f32 = 0.1;

/// The shape of a strided one-dimensional convolution over a batch of
/// sequences stored one after another. The input is `[sequences · input_time,
/// channels]`; the unfolded columns are `[sequences · output_time, kernel ·
/// channels]`. Output frame `t` reads input frames `t · stride + k − padding`
/// for taps `k < kernel`, and frames outside the sequence read zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StridedConvolution {
    /// Sequences in the batch.
    pub sequences: usize,

    /// Input frames per sequence.
    pub input_time: usize,

    /// Output frames per sequence.
    pub output_time: usize,

    /// Input channels per frame.
    pub channels: usize,

    /// Taps of the filter.
    pub kernel: usize,

    /// Input frames between consecutive output frames.
    pub stride: usize,

    /// Zero frames added before each sequence: `kernel / 2`, so that the
    /// output has `ceil(input_time / stride)` frames.
    pub padding: usize,
}

impl StridedConvolution {
    /// A convolution of `kernel` taps and `stride` over `sequences` sequences
    /// of `input_time` frames with `channels` channels.
    pub fn new(sequences: usize, input_time: usize, channels: usize, kernel: usize, stride: usize) -> Self {
        let padding = kernel / 2;
        let output_time = (input_time + 2 * padding - kernel) / stride + 1;

        Self { sequences, input_time, output_time, channels, kernel, stride, padding }
    }

    /// Rows of the input, `sequences · input_time`.
    pub fn input_rows(&self) -> usize {
        self.sequences * self.input_time
    }

    /// Rows of the output, `sequences · output_time`.
    pub fn output_rows(&self) -> usize {
        self.sequences * self.output_time
    }

    /// Width of an unfolded row, `kernel · channels`.
    pub fn width(&self) -> usize {
        self.kernel * self.channels
    }

    /// The input frame that tap `tap` of output frame `frame` reads, or
    /// `None` when it falls in the padding.
    pub fn source(&self, frame: usize, tap: usize) -> Option<usize> {
        let position = (frame * self.stride + tap).checked_sub(self.padding)?;

        (position < self.input_time).then_some(position)
    }

    /// The output frame whose tap `tap` reads input frame `position`, if any:
    /// the inverse of [`StridedConvolution::source`].
    pub fn reader(&self, position: usize, tap: usize) -> Option<usize> {
        let shifted = (position + self.padding).checked_sub(tap)?;
        if shifted % self.stride != 0 {
            return None;
        }

        let frame = shifted / self.stride;
        (frame < self.output_time).then_some(frame)
    }
}

/// How a batch of signals is split into interleaved sequences, one per phase
/// of a period. Signal `s` of `length` samples becomes `period` sequences of
/// `frames()` samples; sequence `s · period + phase` holds samples `phase`,
/// `phase + period`, `phase + 2 · period`, …, with zeros past the end.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeriodSplit {
    /// Signals in the batch.
    pub signals: usize,

    /// Samples per signal.
    pub length: usize,

    /// Samples between consecutive entries of one sequence.
    pub period: usize,
}

impl PeriodSplit {
    /// Entries per sequence: the signal length divided by the period, rounded up.
    pub fn frames(&self) -> usize {
        self.length.div_ceil(self.period)
    }

    /// Sequences in the split batch, `signals · period`.
    pub fn sequences(&self) -> usize {
        self.signals * self.period
    }

    /// Where sample `sample` of signal `signal` lands in the split batch.
    pub fn split_index(&self, signal: usize, sample: usize) -> usize {
        let sequence = signal * self.period + sample % self.period;

        sequence * self.frames() + sample / self.period
    }
}

#[cuda_module]
pub mod kernels {
    use super::{LEAKY_SLOPE, PeriodSplit, StridedConvolution};
    use crate::kernels::device::{global_thread, store};
    use cuda_device::{DisjointSlice, kernel, thread};

    /// Lays out each output frame's neighborhood as one row, so that a strided
    /// convolution becomes a matrix product:
    /// `columns[t, k · channels + c] = input[t · stride + k − padding, c]`, or
    /// 0 in the padding. One thread per column element.
    ///
    /// * `input`: `[sequences · input_time, channels]`.
    /// * `columns`: `[sequences · output_time, kernel · channels]`, written.
    #[kernel]
    pub fn unfold_strided(input: &[f32], mut columns: DisjointSlice<f32>, shape: StridedConvolution) {
        let element = global_thread();
        let width = shape.width();
        if element >= shape.output_rows() * width {
            return;
        }

        let row = element / width;
        let sequence = row / shape.output_time;
        let frame = row % shape.output_time;
        let tap = (element % width) / shape.channels;
        let channel = element % shape.channels;

        let value = match shape.source(frame, tap) {
            Some(position) => input[(sequence * shape.input_time + position) * shape.channels + channel],
            None => 0.0,
        };
        store(&mut columns, element, value);
    }

    /// The gradient of [`unfold_strided`]: every input element sums the column
    /// gradients of the taps that read it. Each thread gathers for its own
    /// element, so the sum has a fixed order and no atomics are needed.
    ///
    /// * `columns_grad`: `[sequences · output_time, kernel · channels]`.
    /// * `input_grad`: `[sequences · input_time, channels]`, written.
    #[kernel]
    pub fn unfold_strided_backward(columns_grad: &[f32], mut input_grad: DisjointSlice<f32>, shape: StridedConvolution) {
        let element = global_thread();
        if element >= shape.input_rows() * shape.channels {
            return;
        }

        let row = element / shape.channels;
        let sequence = row / shape.input_time;
        let position = row % shape.input_time;
        let channel = element % shape.channels;

        let mut sum = 0.0f32;
        for tap in 0..shape.kernel {
            if let Some(frame) = shape.reader(position, tap) {
                sum += columns_grad[(sequence * shape.output_time + frame) * shape.width() + tap * shape.channels + channel];
            }
        }
        store(&mut input_grad, element, sum);
    }

    /// Splits every signal into its `period` interleaved sequences (see
    /// [`PeriodSplit`]). One thread per element of the split batch.
    ///
    /// * `signals`: `[signals, length]`.
    /// * `split`: `[signals · period, frames]`, written.
    #[kernel]
    pub fn split_periods(signals: &[f32], mut split: DisjointSlice<f32>, layout: PeriodSplit) {
        let element = global_thread();
        let frames = layout.frames();
        if element >= layout.sequences() * frames {
            return;
        }

        let sequence = element / frames;
        let signal = sequence / layout.period;
        let sample = (element % frames) * layout.period + sequence % layout.period;

        let value = if sample < layout.length { signals[signal * layout.length + sample] } else { 0.0 };
        store(&mut split, element, value);
    }

    /// The gradient of [`split_periods`], added into `signals_grad`, because
    /// several discriminators read the same signal. One thread per sample.
    ///
    /// * `split_grad`: `[signals · period, frames]`.
    /// * `signals_grad`: `[signals, length]`, accumulated.
    #[kernel]
    pub fn split_periods_backward(split_grad: &[f32], mut signals_grad: DisjointSlice<f32>, layout: PeriodSplit) {
        let index = thread::index_1d();
        let element = index.get();

        let signal = element / layout.length;
        let sample = element % layout.length;

        if let Some(slot) = signals_grad.get_mut(index) {
            *slot += split_grad[layout.split_index(signal, sample)];
        }
    }

    /// The leaky ReLU: `output = input` where positive, `LEAKY_SLOPE · input`
    /// elsewhere. One thread per element.
    #[kernel]
    pub fn leaky_relu(input: &[f32], mut output: DisjointSlice<f32>) {
        let element = global_thread();
        if element >= input.len() {
            return;
        }

        let value = input[element];
        store(&mut output, element, if value > 0.0 { value } else { LEAKY_SLOPE * value });
    }

    /// The gradient of [`leaky_relu`]:
    /// `input_grad = output_grad · (1 or LEAKY_SLOPE)` by the sign of `input`.
    #[kernel]
    pub fn leaky_relu_backward(input: &[f32], output_grad: &[f32], mut input_grad: DisjointSlice<f32>) {
        let element = global_thread();
        if element >= input.len() {
            return;
        }

        let slope = if input[element] > 0.0 { 1.0 } else { LEAKY_SLOPE };
        store(&mut input_grad, element, slope * output_grad[element]);
    }

    /// The gradient of the least-squares objective `scale · Σ (score − target)²`
    /// with respect to each score: `2 · scale · (score − target)`.
    ///
    /// * `target`: 1 for "real", 0 for "synthesized".
    /// * `scale`: the objective's weight divided by the number of scores.
    #[kernel]
    pub fn least_squares_gradient(scores: &[f32], mut gradient: DisjointSlice<f32>, target: f32, scale: f32) {
        let element = global_thread();
        if element >= scores.len() {
            return;
        }

        store(&mut gradient, element, 2.0 * scale * (scores[element] - target));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_strided_convolution_keeps_ceil_of_time_over_stride_frames() {
        let shape = StridedConvolution::new(2, 10, 1, 5, 3);

        assert_eq!(shape.padding, 2);
        assert_eq!(shape.output_time, 4);
        assert_eq!(shape.source(0, 0), None);
        assert_eq!(shape.source(0, 2), Some(0));
        assert_eq!(shape.source(3, 4), None);
    }

    #[test]
    fn reader_inverts_source() {
        let shape = StridedConvolution::new(1, 11, 1, 5, 3);

        for frame in 0..shape.output_time {
            for tap in 0..shape.kernel {
                if let Some(position) = shape.source(frame, tap) {
                    assert_eq!(shape.reader(position, tap), Some(frame));
                }
            }
        }
    }

    #[test]
    fn a_period_split_interleaves_the_samples() {
        let layout = PeriodSplit { signals: 2, length: 7, period: 3 };

        assert_eq!(layout.frames(), 3);
        assert_eq!(layout.sequences(), 6);
        assert_eq!(layout.split_index(0, 4), 4);
        assert_eq!(layout.split_index(1, 6), 11);
    }
}
