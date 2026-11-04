//! Overlap-add, the last step of the inverse short-time Fourier transform, and
//! its gradient.
//!
//! The inverse transform turns every frame's spectrum into a short run of
//! samples; overlap-add lays those runs down `hop` samples apart and sums them
//! (Smith, *Spectral Audio Signal Processing*, "Overlap-Add (OLA) STFT
//! Processing", <https://ccrma.stanford.edu/~jos/sasp/Overlap_Add_OLA_STFT_Processing.html>).
//! Its adjoint cuts a signal into overlapping frames, so the same pair of
//! kernels also frames signals for the spectral losses and carries their
//! gradients back.

use cuda_device::cuda_module;

#[cuda_module]
pub mod kernels {
    use crate::kernels::device::{global_thread, store};
    use cuda_device::{DisjointSlice, kernel};

    /// Joins the frames of each sequence into its signal,
    /// `signal[s] = normalization[s] · Σ_f frames[f, s − f · hop]` over that
    /// sequence's frames that cover sample `s`. Sequences are stored one after
    /// another and never mix. One thread per output sample, so no two threads
    /// add into the same place.
    ///
    /// With a normalization of ones this is also the adjoint of framing a
    /// signal, which the spectral losses use to send gradients back.
    ///
    /// * `frames`: `[sequences · frame_count, frame_length]`.
    /// * `normalization`: one factor per sample of one sequence's signal.
    /// * `signal`: `sequences · ((frame_count − 1) · hop + frame_length)` samples, written.
    #[kernel]
    pub fn overlap_add(
        frames: &[f32],
        normalization: &[f32],
        mut signal: DisjointSlice<f32>,
        frame_count: usize,
        frame_length: usize,
        hop: usize,
    ) {
        let element = global_thread();
        if element >= signal.len() {
            return;
        }

        let length = (frame_count - 1) * hop + frame_length;
        let first_frame_of_sequence = (element / length) * frame_count;
        let sample = element % length;

        let last = (sample / hop).min(frame_count - 1);
        let first = if sample >= frame_length { (sample - frame_length) / hop + 1 } else { 0 };

        let mut sum = 0.0f32;
        for frame in first..=last {
            sum += frames[(first_frame_of_sequence + frame) * frame_length + sample - frame * hop];
        }

        store(&mut signal, element, sum * normalization[sample]);
    }

    /// The gradient of [`overlap_add`]: every frame sample receives the
    /// gradient of the signal sample it was added into, times that sample's
    /// normalization. With a normalization of ones this cuts each sequence's
    /// signal into overlapping frames. One thread per frame sample.
    ///
    /// * `signal_grad`: the gradient of the joined signals.
    /// * `normalization`: as in [`overlap_add`].
    /// * `frames_grad`: `[sequences · frame_count, frame_length]`, written.
    #[kernel]
    pub fn overlap_add_backward(
        signal_grad: &[f32],
        normalization: &[f32],
        mut frames_grad: DisjointSlice<f32>,
        frame_count: usize,
        frame_length: usize,
        hop: usize,
    ) {
        let element = global_thread();
        if element >= frames_grad.len() {
            return;
        }

        let length = (frame_count - 1) * hop + frame_length;

        let row = element / frame_length;
        let sequence = row / frame_count;
        let sample = (row % frame_count) * hop + element % frame_length;

        store(&mut frames_grad, element, signal_grad[sequence * length + sample] * normalization[sample]);
    }
}
