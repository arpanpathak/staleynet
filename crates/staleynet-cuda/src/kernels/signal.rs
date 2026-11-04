//! Signal processing for the spectrogram, and the nearest-prototype search of
//! the tokenizer.

use cuda_device::cuda_module;

#[cuda_module]
pub mod kernels {
    use crate::kernels::NEGATIVE_LARGE;
    use cuda_device::{DisjointSlice, kernel, thread};

    /// Cuts a padded signal into overlapping frames:
    /// `frames[f, n] = signal[f · hop + n]`. One thread per frame sample.
    ///
    /// * `signal`: the padded waveform.
    /// * `frames`: `[frame count, frame_length]`, written.
    /// * `frame_length`: samples per frame.
    /// * `hop`: samples between the starts of consecutive frames.
    #[kernel]
    pub fn frame_signal(signal: &[f32], mut frames: DisjointSlice<f32>, frame_length: usize, hop: usize) {
        let index = thread::index_1d();
        let element = index.get();
        let source = (element / frame_length) * hop + element % frame_length;

        if let Some(slot) = frames.get_mut(index) {
            *slot = signal[source];
        }
    }

    /// The power of each frequency: `power = re² + im²`.
    ///
    /// * `real`, `imaginary`: the two parts of the spectrum.
    /// * `power`: written; as long as each part.
    #[kernel]
    pub fn power_spectrum(real: &[f32], imaginary: &[f32], mut power: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let element = index.get();
        if let Some(slot) = power.get_mut(index) {
            *slot = real[element] * real[element] + imaginary[element] * imaginary[element];
        }
    }

    /// Logarithmic compression in place: `values = ln(max(values, floor))`.
    ///
    /// * `values`: the energies, replaced by their logarithms.
    /// * `floor`: the smallest energy kept, so that silence stays finite.
    #[kernel]
    pub fn log_compress(mut values: DisjointSlice<f32>, floor: f32) {
        if let Some(slot) = values.get_mut(thread::index_1d()) {
            *slot = libm::logf(slot.max(floor));
        }
    }

    /// The nearest prototype of every vector, from precomputed dot products.
    /// Since `‖x − e_k‖² = ‖x‖² − 2 x·e_k + ‖e_k‖²` and `‖x‖²` is the same for
    /// every `k`, the nearest prototype minimises `‖e_k‖² − 2 x·e_k`. One
    /// thread per vector.
    ///
    /// * `dots`: `x · e_k` for every vector and prototype, `[vectors, centroids]`.
    /// * `centroid_norms`: `‖e_k‖²` for every prototype.
    /// * `assignments`: the index of the nearest prototype of each vector, written.
    /// * `centroids`: the number of prototypes.
    #[kernel]
    pub fn nearest_centroid(dots: &[f32], centroid_norms: &[f32], mut assignments: DisjointSlice<u32>, centroids: usize) {
        let index = thread::index_1d();
        let row = index.get();
        if row >= assignments.len() {
            return;
        }

        let mut best_distance = -NEGATIVE_LARGE;
        let mut best_centroid = 0u32;
        for centroid in 0..centroids {
            let distance = centroid_norms[centroid] - 2.0 * dots[row * centroids + centroid];
            if distance < best_distance {
                best_distance = distance;
                best_centroid = centroid as u32;
            }
        }

        if let Some(slot) = assignments.get_mut(index) {
            *slot = best_centroid;
        }
    }
}
