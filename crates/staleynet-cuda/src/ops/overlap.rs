//! Overlap-add and its gradient, which also frames signals for the spectral
//! losses. See [`crate::kernels::overlap`].

use crate::device::{Gpu, GpuResult, Tensor};
use crate::launch;

impl Gpu {
    /// Overlap-adds the frames of every sequence (`frames_per_sequence` rows
    /// of `frames` each, consecutive frames `hop` samples apart) into that
    /// sequence's part of `signal`, multiplying every sample by
    /// `normalization`. Sequences never mix.
    pub fn overlap_add(
        &self,
        frames: &Tensor,
        normalization: &Tensor,
        signal: &mut Tensor,
        frames_per_sequence: usize,
        hop: usize,
    ) -> GpuResult {
        check_overlap(frames, normalization, signal, frames_per_sequence, hop);
        let frame_length = frames.shape().cols;

        // SAFETY: one thread per signal sample; every frame index read lies within
        // its own sequence's rows, because first..=last only spans frames covering
        // the sample, and rows = sequences · frames_per_sequence was checked.
        unsafe {
            self.kernels.overlap.overlap_add(
                &self.stream,
                launch::elementwise(signal.len()),
                &frames.buffer,
                &normalization.buffer,
                &mut signal.buffer,
                frames_per_sequence,
                frame_length,
                hop,
            )
        }
    }

    /// The gradient of [`Gpu::overlap_add`] with respect to its frames. With
    /// a normalization of ones, this cuts every sequence of `signal_grad` into
    /// overlapping frames.
    pub fn overlap_add_backward(
        &self,
        signal_grad: &Tensor,
        normalization: &Tensor,
        frames_grad: &mut Tensor,
        frames_per_sequence: usize,
        hop: usize,
    ) -> GpuResult {
        check_overlap(frames_grad, normalization, signal_grad, frames_per_sequence, hop);
        let frame_length = frames_grad.shape().cols;

        // SAFETY: one thread per frame sample; the sample read lies within its
        // sequence's signal, by the sizes checked above.
        unsafe {
            self.kernels.overlap.overlap_add_backward(
                &self.stream,
                launch::elementwise(frames_grad.len()),
                &signal_grad.buffer,
                &normalization.buffer,
                &mut frames_grad.buffer,
                frames_per_sequence,
                frame_length,
                hop,
            )
        }
    }
}

/// Asserts that `frames` holds whole sequences of `frames_per_sequence` frames
/// whose joined signals fill `signal`.
fn check_overlap(frames: &Tensor, normalization: &Tensor, signal: &Tensor, frames_per_sequence: usize, hop: usize) {
    let shape = frames.shape();
    assert!(frames_per_sequence > 0 && shape.rows.is_multiple_of(frames_per_sequence), "overlap: rows must be whole sequences");
    assert!(hop <= shape.cols, "overlap: frames must not leave gaps");

    let length = (frames_per_sequence - 1) * hop + shape.cols;
    let sequences = shape.rows / frames_per_sequence;

    assert_eq!(normalization.len(), length, "overlap: one normalization factor per sample of a sequence");
    assert_eq!(signal.len(), sequences * length, "overlap: signal length");
}
