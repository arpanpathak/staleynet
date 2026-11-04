//! Matrix products: `C_b = scale · A_b · B_b (+ C_b)` for every matrix `b` of a
//! batch.
//!
//! Each operand is described by its strides (see [`Operand`]), so a transposed
//! matrix is simply the same data read with its two strides exchanged; the
//! kernels never test whether an operand is transposed. Three kernels compute
//! the same product with increasing reuse of fetched values:
//!
//! * [`kernels::matmul_naive`]: one thread per output, every operand fetched
//!   from global memory (the reference);
//! * [`kernels::matmul_tiled`]: `16 × 16` tiles shared through shared memory
//!   (Kirk, Hwu and El Hajj, *Programming Massively Parallel Processors*, 4th
//!   ed., ch. 5);
//! * [`kernels::matmul_blocked`]: `64 × 64` tiles in which each thread keeps a
//!   `4 × 4` block of outputs in registers (Volkov and Demmel, 2008,
//!   <https://doi.org/10.1109/SC.2008.5214359>; Boehm,
//!   <https://siboehm.com/articles/22/CUDA-MMM>).

use cuda_device::cuda_module;

/// Edge of the square output tile of the naive and tiled kernels: one block of
/// `16 × 16 = 256` threads, one output each.
pub const MATMUL_TILE: usize = 16;

/// Edge of the square output tile of one block of the blocked kernel.
pub const BLOCKED_TILE: usize = 64;

/// Depth of the slice of the inner dimension staged in shared memory per phase.
pub const BLOCKED_DEPTH: usize = 16;

/// Edge of the square of outputs each thread of the blocked kernel keeps in registers.
pub const BLOCKED_PER_THREAD: usize = 4;

/// Threads along each edge of a block of the blocked kernel (16, so 256 threads).
pub const BLOCKED_THREADS_PER_EDGE: usize = BLOCKED_TILE / BLOCKED_PER_THREAD;

/// Threads in one block of the blocked kernel.
const BLOCKED_THREADS: usize = BLOCKED_THREADS_PER_EDGE * BLOCKED_THREADS_PER_EDGE;

/// Values of each operand tile that every thread of the blocked kernel copies per phase.
const BLOCKED_LOADS_PER_THREAD: usize = BLOCKED_TILE * BLOCKED_DEPTH / BLOCKED_THREADS;

/// Where the elements of one matrix operand live: element `(row, col)` of
/// matrix `b` of a batch is `values[b · batch_stride + row · row_stride + col · col_stride]`.
///
/// A row-major `[rows, cols]` matrix has `row_stride = cols` and `col_stride = 1`;
/// its transpose is the same data with the two strides exchanged.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Operand {
    /// Distance between consecutive rows.
    pub row_stride: usize,

    /// Distance between consecutive columns.
    pub col_stride: usize,

    /// Distance between consecutive matrices of a batch (0 shares one matrix).
    pub batch_stride: usize,
}

impl Operand {
    /// Element `(row, col)` of matrix `batch`.
    pub fn at(&self, values: &[f32], batch: usize, row: usize, col: usize) -> f32 {
        values[batch * self.batch_stride + row * self.row_stride + col * self.col_stride]
    }
}

/// One batched matrix product, `C_b = scale · A_b · B_b (+ C_b)`, passed to
/// the kernels by value. `A_b` is `[rows, inner]`, `B_b` is `[inner, cols]`,
/// and `C_b` is stored row-major.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Product {
    /// Rows of `A` and `C`.
    pub rows: usize,

    /// Columns of `B` and `C`.
    pub cols: usize,

    /// Columns of `A` and rows of `B`.
    pub inner: usize,

    /// Layout of `A`.
    pub left: Operand,

    /// Layout of `B`.
    pub right: Operand,

    /// Distance between consecutive output matrices.
    pub output_batch_stride: usize,

    /// Factor applied to the product.
    pub scale: f32,

    /// 1 to add the product to `C`, 0 to overwrite `C`.
    pub accumulate: u32,
}

impl Product {
    /// Position of output `(row, col)` of matrix `batch`.
    pub fn output_index(&self, batch: usize, row: usize, col: usize) -> usize {
        batch * self.output_batch_stride + row * self.cols + col
    }

    /// The value to store at an output that held `previous`, given the dot product `sum`.
    pub fn combine(&self, previous: f32, sum: f32) -> f32 {
        if self.accumulate != 0 { previous + self.scale * sum } else { self.scale * sum }
    }

    /// Element `(row, k)` of `A`, or 0 outside the matrix (for partial tiles).
    pub fn left_or_zero(&self, values: &[f32], batch: usize, row: usize, k: usize) -> f32 {
        if row < self.rows && k < self.inner { self.left.at(values, batch, row, k) } else { 0.0 }
    }

    /// Element `(k, col)` of `B`, or 0 outside the matrix (for partial tiles).
    pub fn right_or_zero(&self, values: &[f32], batch: usize, k: usize, col: usize) -> f32 {
        if k < self.inner && col < self.cols { self.right.at(values, batch, k, col) } else { 0.0 }
    }
}

/// Splits the position `linear` of a value within a tile of `long × short`
/// values into `(long index, short index)`.
///
/// Consecutive threads copy consecutive values. When the operand is stored
/// with the short dimension contiguous, the short index varies fastest, and
/// otherwise the long index does, so that consecutive threads always read
/// consecutive addresses of global memory, which the hardware combines into
/// one access (coalescing).
pub fn tile_position(linear: usize, long: usize, short: usize, short_is_contiguous: bool) -> (usize, usize) {
    if short_is_contiguous { (linear / short, linear % short) } else { (linear % long, linear / long) }
}

#[cuda_module]
pub mod kernels {
    use super::{
        BLOCKED_DEPTH, BLOCKED_LOADS_PER_THREAD, BLOCKED_PER_THREAD, BLOCKED_THREADS, BLOCKED_THREADS_PER_EDGE, BLOCKED_TILE, MATMUL_TILE,
        Product, tile_position,
    };
    use crate::kernels::device::{load_owned, store};
    use cuda_device::{DisjointSlice, SharedArray, kernel, thread};

    /// The direct product: each thread computes one output from a row of `A`
    /// and a column of `B` read from global memory.
    ///
    /// Launch: blocks of `16 × 16` threads, `x` over output columns, `y` over
    /// output rows, `z` over the matrices of the batch.
    ///
    /// * `left`, `right`: the operands `A` and `B`, laid out as described by `product`.
    /// * `output`: the result `C`, row-major.
    /// * `product`: sizes, layouts, scale and write mode.
    #[kernel]
    pub fn matmul_naive(left: &[f32], right: &[f32], mut output: DisjointSlice<f32>, product: Product) {
        let batch = thread::blockIdx_z() as usize;
        let row = thread::blockIdx_y() as usize * MATMUL_TILE + thread::threadIdx_y() as usize;
        let col = thread::blockIdx_x() as usize * MATMUL_TILE + thread::threadIdx_x() as usize;

        if row >= product.rows || col >= product.cols {
            return;
        }

        let mut sum = 0.0f32;
        for k in 0..product.inner {
            sum += product.left.at(left, batch, row, k) * product.right.at(right, batch, k, col);
        }

        let index = product.output_index(batch, row, col);
        let previous = load_owned(&mut output, index);
        store(&mut output, index, product.combine(previous, sum));
    }

    /// The tiled product: a block of `16 × 16` threads copies a `16 × 16` tile
    /// of `A` and of `B` into shared memory, waits, multiplies the tiles, and
    /// moves on along the inner dimension. Each value is fetched from global
    /// memory once per block instead of once per thread.
    ///
    /// Launch: as [`matmul_naive`]. Arguments: as [`matmul_naive`].
    #[kernel]
    pub fn matmul_tiled(left: &[f32], right: &[f32], mut output: DisjointSlice<f32>, product: Product) {
        static mut LEFT_TILE: SharedArray<f32, { MATMUL_TILE * MATMUL_TILE }> = SharedArray::UNINIT;
        static mut RIGHT_TILE: SharedArray<f32, { MATMUL_TILE * MATMUL_TILE }> = SharedArray::UNINIT;

        let batch = thread::blockIdx_z() as usize;
        let tile_row = thread::blockIdx_y() as usize * MATMUL_TILE;
        let tile_col = thread::blockIdx_x() as usize * MATMUL_TILE;

        let local_x = thread::threadIdx_x() as usize;
        let local_y = thread::threadIdx_y() as usize;
        let linear = local_y * MATMUL_TILE + local_x;

        let mut sum = 0.0f32;
        for tile_start in (0..product.inner).step_by(MATMUL_TILE) {
            let (left_row, left_k) = tile_position(linear, MATMUL_TILE, MATMUL_TILE, product.left.col_stride == 1);
            let (right_col, right_k) = tile_position(linear, MATMUL_TILE, MATMUL_TILE, product.right.row_stride == 1);
            let left_value = product.left_or_zero(left, batch, tile_row + left_row, tile_start + left_k);
            let right_value = product.right_or_zero(right, batch, tile_start + right_k, tile_col + right_col);

            // SAFETY: shared tiles are private to the block; both indices are below 16 · 16.
            unsafe {
                LEFT_TILE[left_row * MATMUL_TILE + left_k] = left_value;
                RIGHT_TILE[right_k * MATMUL_TILE + right_col] = right_value;
            }
            thread::sync_threads();

            for k in 0..MATMUL_TILE {
                // SAFETY: indices below 16 · 16; every slot was written before the barrier above.
                sum += unsafe { LEFT_TILE[local_y * MATMUL_TILE + k] * RIGHT_TILE[k * MATMUL_TILE + local_x] };
            }
            thread::sync_threads();
        }

        let (row, col) = (tile_row + local_y, tile_col + local_x);
        if row < product.rows && col < product.cols {
            let index = product.output_index(batch, row, col);
            let previous = load_owned(&mut output, index);
            store(&mut output, index, product.combine(previous, sum));
        }
    }

    /// The register-blocked product: a block of `16 × 16` threads computes a
    /// `64 × 64` output tile, each thread a `4 × 4` group of outputs held in
    /// registers. Thread `(x, y)` owns rows `y + 16i` and columns `x + 16j`,
    /// so neighbouring threads read neighbouring words of the right-hand tile.
    /// Per step of the inner loop, a thread reads 4 values of each tile and
    /// performs 16 multiply-adds, four times the work per shared-memory read
    /// of [`matmul_tiled`].
    ///
    /// Launch: blocks of `16 × 16` threads, `x` over 64-column tiles, `y` over
    /// 64-row tiles, `z` over the matrices of the batch. Arguments: as
    /// [`matmul_naive`].
    #[kernel]
    pub fn matmul_blocked(left: &[f32], right: &[f32], mut output: DisjointSlice<f32>, product: Product) {
        static mut LEFT_TILE: SharedArray<f32, { BLOCKED_TILE * BLOCKED_DEPTH }> = SharedArray::UNINIT;
        static mut RIGHT_TILE: SharedArray<f32, { BLOCKED_DEPTH * BLOCKED_TILE }> = SharedArray::UNINIT;

        let batch = thread::blockIdx_z() as usize;
        let tile_row = thread::blockIdx_y() as usize * BLOCKED_TILE;
        let tile_col = thread::blockIdx_x() as usize * BLOCKED_TILE;

        let local_x = thread::threadIdx_x() as usize;
        let local_y = thread::threadIdx_y() as usize;
        let thread_index = local_y * BLOCKED_THREADS_PER_EDGE + local_x;

        let mut sums = [0.0f32; BLOCKED_PER_THREAD * BLOCKED_PER_THREAD];
        for tile_start in (0..product.inner).step_by(BLOCKED_DEPTH) {
            for load in 0..BLOCKED_LOADS_PER_THREAD {
                let linear = thread_index + load * BLOCKED_THREADS;
                let (left_row, left_k) = tile_position(linear, BLOCKED_TILE, BLOCKED_DEPTH, product.left.col_stride == 1);
                let (right_col, right_k) = tile_position(linear, BLOCKED_TILE, BLOCKED_DEPTH, product.right.row_stride == 1);
                let left_value = product.left_or_zero(left, batch, tile_row + left_row, tile_start + left_k);
                let right_value = product.right_or_zero(right, batch, tile_start + right_k, tile_col + right_col);

                // SAFETY: shared tiles are private to the block; both indices are below 64 · 16.
                unsafe {
                    LEFT_TILE[left_row * BLOCKED_DEPTH + left_k] = left_value;
                    RIGHT_TILE[right_k * BLOCKED_TILE + right_col] = right_value;
                }
            }
            thread::sync_threads();

            for k in 0..BLOCKED_DEPTH {
                let mut left_values = [0.0f32; BLOCKED_PER_THREAD];
                let mut right_values = [0.0f32; BLOCKED_PER_THREAD];
                for i in 0..BLOCKED_PER_THREAD {
                    // SAFETY: indices below 64 · 16; every slot was written before the barrier above.
                    unsafe {
                        left_values[i] = LEFT_TILE[(local_y + i * BLOCKED_THREADS_PER_EDGE) * BLOCKED_DEPTH + k];
                        right_values[i] = RIGHT_TILE[k * BLOCKED_TILE + local_x + i * BLOCKED_THREADS_PER_EDGE];
                    }
                }

                for i in 0..BLOCKED_PER_THREAD {
                    for j in 0..BLOCKED_PER_THREAD {
                        sums[i * BLOCKED_PER_THREAD + j] += left_values[i] * right_values[j];
                    }
                }
            }
            thread::sync_threads();
        }

        for i in 0..BLOCKED_PER_THREAD {
            for j in 0..BLOCKED_PER_THREAD {
                let row = tile_row + local_y + i * BLOCKED_THREADS_PER_EDGE;
                let col = tile_col + local_x + j * BLOCKED_THREADS_PER_EDGE;
                if row < product.rows && col < product.cols {
                    let index = product.output_index(batch, row, col);
                    let previous = load_owned(&mut output, index);
                    store(&mut output, index, product.combine(previous, sums[i * BLOCKED_PER_THREAD + j]));
                }
            }
        }
    }
}
