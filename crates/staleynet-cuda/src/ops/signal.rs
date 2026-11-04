//! Signal-processing and clustering primitives used by feature extraction and
//! the audio tokenizer.

use crate::device::{Gpu, GpuResult, Tensor, TokenBuffer};
use crate::launch;

impl Gpu {
    /// Slices a pre-padded `signal` into overlapping rows of `frames`
    /// (`[frame count, frame length]`), advancing `hop` samples per frame.
    pub fn frame_signal(&self, signal: &Tensor, frames: &mut Tensor, hop: usize) -> GpuResult {
        let shape = frames.shape();
        assert!(signal.len() >= (shape.rows - 1) * hop + shape.cols, "frame_signal: signal too short");

        // SAFETY: one thread per frame element; reads were bounded by the assert.
        unsafe {
            self.kernels.signal.frame_signal(
                &self.stream,
                launch::elementwise(frames.len()),
                &signal.buffer,
                &mut frames.buffer,
                shape.cols,
                hop,
            )
        }
    }

    /// `power = real² + imaginary²`.
    pub fn power_spectrum(&self, real: &Tensor, imaginary: &Tensor, power: &mut Tensor) -> GpuResult {
        assert!(real.shape() == power.shape() && imaginary.shape() == power.shape(), "power_spectrum: shape mismatch");

        // SAFETY: one thread per element; `get_mut` guards the tail.
        unsafe {
            self.kernels.signal.power_spectrum(
                &self.stream,
                launch::elementwise(power.len()),
                &real.buffer,
                &imaginary.buffer,
                &mut power.buffer,
            )
        }
    }

    /// `values = ln(max(values, floor))`.
    pub fn log_compress(&self, values: &mut Tensor, floor: f32) -> GpuResult {
        // SAFETY: one thread per element; `get_mut` guards the tail.
        unsafe { self.kernels.signal.log_compress(&self.stream, launch::elementwise(values.len()), &mut values.buffer, floor) }
    }

    /// For each row of `dots` (`x·cᵀ`, `[points, centroids]`), the index of the
    /// centroid minimising `‖c‖² − 2·x·c`.
    pub fn nearest_centroid(&self, dots: &Tensor, centroid_norms: &Tensor, assignments: &mut TokenBuffer) -> GpuResult {
        let shape = dots.shape();
        assert_eq!(centroid_norms.len(), shape.cols, "nearest_centroid: norms length");
        assert!(assignments.len() >= shape.rows, "nearest_centroid: assignments too small");

        // SAFETY: one thread per point; out-of-range threads return early.
        unsafe {
            self.kernels.signal.nearest_centroid(
                &self.stream,
                launch::elementwise(shape.rows),
                &dots.buffer,
                &centroid_norms.buffer,
                assignments,
                shape.cols,
            )
        }
    }
}
