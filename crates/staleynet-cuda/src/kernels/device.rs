//! Small helpers shared by the kernels: where a thread is, and how it writes.

use super::WARP_SIZE;
use cuda_device::{DisjointSlice, thread};

/// The global index of the calling thread in a one-dimensional launch.
pub fn global_thread() -> usize {
    thread::blockIdx_x() as usize * thread::blockDim_x() as usize + thread::threadIdx_x() as usize
}

/// The matrix row owned by the calling thread's warp, and the thread's lane
/// within the warp, for kernels that assign one warp to each row.
pub struct WarpRow {
    /// Matrix row assigned to the warp.
    pub row: usize,

    /// Lane in `0..WARP_SIZE`; also the first column the lane visits.
    pub lane: usize,
}

/// The row and lane of the calling thread in a warp-per-row launch.
pub fn warp_row() -> WarpRow {
    let thread = global_thread();
    WarpRow { row: thread / WARP_SIZE, lane: thread % WARP_SIZE }
}

/// Writes `value` to `output[index]`, or nothing if `index` is out of range.
///
/// `DisjointSlice` lets a thread write only through an index derived from its
/// own position. Kernels whose output index is computed (a row times a width
/// plus a column, for example) use this helper instead. Every caller maps
/// threads to output indices one to one, which is the condition under which
/// `get_unchecked_mut` is sound; the bound is checked here.
pub fn store(output: &mut DisjointSlice<f32>, index: usize, value: f32) {
    if index < output.len() {
        // SAFETY: the index is in range, and each caller writes each index from exactly one thread.
        unsafe { *output.get_unchecked_mut(index) = value };
    }
}

/// Reads `output[index]` for a read-modify-write by its single owning thread,
/// or returns 0 if `index` is out of range.
pub fn load_owned(output: &mut DisjointSlice<f32>, index: usize) -> f32 {
    if index >= output.len() {
        return 0.0;
    }
    // SAFETY: the index is in range, and only the owning thread touches it.
    unsafe { *output.get_unchecked_mut(index) }
}
