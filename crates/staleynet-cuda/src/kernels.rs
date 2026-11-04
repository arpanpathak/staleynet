//! Device code: every GPU kernel StaleyNET runs, compiled from Rust to PTX by
//! the cuda-oxide codegen backend (<https://github.com/NVlabs/cuda-oxide>).
//!
//! Three launch shapes are used throughout:
//!
//! * **Elementwise**: one thread per output element.
//! * **Warp per row**: one 32-lane warp owns one matrix row; lanes stride
//!   across the columns and combine partial results with butterfly shuffles.
//!   Every lane of a warp evaluates the same row bound, so shuffles always run
//!   with the full warp. See NVIDIA, *Using CUDA Warp-Level Primitives*:
//!   <https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/>
//! * **Matmul tiles**: a 2-D block computes one output tile, staging operand
//!   tiles through shared memory. See Kirk & Hwu, *Programming Massively
//!   Parallel Processors*, 4th ed., ch. 5 (tiling).
//!
//! Writes to positions not derived from the thread index go through
//! `get_unchecked_mut`; each one is preceded by an explicit bound check or by
//! a size invariant that the host wrapper asserts before launch.

use cuda_device::cuda_module;

/// Edge length of the square output tile computed by one matmul block.
/// 16 × 16 = 256 threads, one output element each.
pub const MATMUL_TILE: usize = 16;

/// Edge length of the output tile computed by one block of the blocked GEMM.
pub const BLOCKED_TILE: usize = 64;

/// Depth of the inner-dimension slice staged in shared memory per iteration.
pub const BLOCKED_DEPTH: usize = 16;

/// Edge length of the square of outputs each thread of the blocked GEMM keeps
/// in registers. A block is (BLOCKED_TILE / BLOCKED_PER_THREAD)² = 256 threads.
pub const BLOCKED_PER_THREAD: usize = 4;

/// Threads along each edge of a blocked-GEMM block.
pub const BLOCKED_THREADS_PER_EDGE: usize = BLOCKED_TILE / BLOCKED_PER_THREAD;

/// Threads per warp on every NVIDIA architecture to date.
/// See <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture>
pub const WARP_SIZE: usize = 32;

/// Threads per block for elementwise and warp-per-row kernels.
pub const BLOCK_THREADS: usize = 256;

/// Warps per block for warp-per-row kernels.
pub const WARPS_PER_BLOCK: usize = BLOCK_THREADS / WARP_SIZE;

/// Marks an empty token slot; such slots contribute no embedding and no loss.
pub const EMPTY_TOKEN: u32 = u32::MAX;

/// GELU tanh-approximation constant √(2/π) (Hendrycks & Gimpel, 2016,
/// <https://arxiv.org/abs/1606.08415>, §2).
pub const GELU_SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// GELU tanh-approximation cubic coefficient (same reference).
pub const GELU_CUBIC: f32 = 0.044_715;

/// Lower bound used as −∞ when taking a running maximum.
const NEGATIVE_LARGE: f32 = -3.0e38;

#[cuda_module]
pub mod kernels {
    use super::{
        BLOCKED_DEPTH, BLOCKED_PER_THREAD, BLOCKED_THREADS_PER_EDGE, BLOCKED_TILE, EMPTY_TOKEN, GELU_CUBIC, GELU_SQRT_2_OVER_PI,
        MATMUL_TILE, NEGATIVE_LARGE, WARP_SIZE,
    };
    use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, thread, warp};

    /// The row a warp owns and the lane's position inside the warp.
    struct WarpRow {
        /// Matrix row assigned to this warp.
        row: usize,

        /// Lane index in `0..WARP_SIZE`; also the lane's first column.
        lane: usize,
    }

    /// Flat global index of the calling thread in a 1-D launch.
    fn global_thread() -> usize {
        thread::blockIdx_x() as usize * thread::blockDim_x() as usize + thread::threadIdx_x() as usize
    }

    /// Row and lane for warp-per-row kernels.
    fn warp_row() -> WarpRow {
        let thread_index = global_thread();
        WarpRow { row: thread_index / WARP_SIZE, lane: thread_index % WARP_SIZE }
    }

    /// GELU, tanh approximation.
    fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + libm::tanhf(GELU_SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x)))
    }

    /// d GELU(x) / dx for the tanh approximation.
    fn gelu_derivative(x: f32) -> f32 {
        let tanh = libm::tanhf(GELU_SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x));
        let inner_derivative = GELU_SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_CUBIC * x * x);
        0.5 * (1.0 + tanh) + 0.5 * x * (1.0 - tanh * tanh) * inner_derivative
    }

    // ------------------------------------------------------------- matmul --

    /// Element (`row`, `inner`) of a row-major operand that may be stored transposed.
    fn operand(values: &[f32], base: usize, row: usize, inner: usize, rows: usize, inners: usize, transposed: u32) -> f32 {
        if transposed != 0 { values[base + inner * rows + row] } else { values[base + row * inners + inner] }
    }

    /// Reference GEMM: one thread per output element, all operands read from
    /// global memory. `output = scale · op(left) · op(right) (+ output)`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_naive(
        left: &[f32],
        right: &[f32],
        mut output: DisjointSlice<f32>,
        rows: usize,
        cols: usize,
        inner: usize,
        left_transposed: u32,
        right_transposed: u32,
        left_batch_stride: usize,
        right_batch_stride: usize,
        output_batch_stride: usize,
        scale: f32,
        accumulate: u32,
    ) {
        let batch = thread::blockIdx_z() as usize;
        let row = thread::blockIdx_y() as usize * MATMUL_TILE + thread::threadIdx_y() as usize;
        let col = thread::blockIdx_x() as usize * MATMUL_TILE + thread::threadIdx_x() as usize;
        if row >= rows || col >= cols {
            return;
        }

        let mut sum = 0.0f32;
        for step in 0..inner {
            let left_value = operand(left, batch * left_batch_stride, row, step, rows, inner, left_transposed);
            let right_value = operand(right, batch * right_batch_stride, col, step, cols, inner, right_transposed ^ 1);
            sum += left_value * right_value;
        }

        let index = batch * output_batch_stride + row * cols + col;
        if index >= output.len() {
            return;
        }

        // SAFETY: index < output.len(); (batch, row, col) ↦ index is injective.
        let slot = unsafe { output.get_unchecked_mut(index) };
        *slot = if accumulate != 0 { *slot + scale * sum } else { scale * sum };
    }

    /// Tiled GEMM: each 16×16 block stages a tile of each operand in shared
    /// memory, so every global value is read once per tile instead of once per
    /// multiply. Same contract as [`matmul_naive`].
    #[kernel]
    #[launch_bounds(256)]
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_tiled(
        left: &[f32],
        right: &[f32],
        mut output: DisjointSlice<f32>,
        rows: usize,
        cols: usize,
        inner: usize,
        left_transposed: u32,
        right_transposed: u32,
        left_batch_stride: usize,
        right_batch_stride: usize,
        output_batch_stride: usize,
        scale: f32,
        accumulate: u32,
    ) {
        static mut LEFT_TILE: SharedArray<f32, { MATMUL_TILE * MATMUL_TILE }> = SharedArray::UNINIT;
        static mut RIGHT_TILE: SharedArray<f32, { MATMUL_TILE * MATMUL_TILE }> = SharedArray::UNINIT;

        let batch = thread::blockIdx_z() as usize;
        let tile_row = thread::blockIdx_y() as usize * MATMUL_TILE;
        let tile_col = thread::blockIdx_x() as usize * MATMUL_TILE;
        let local_x = thread::threadIdx_x() as usize;
        let local_y = thread::threadIdx_y() as usize;
        let left_base = batch * left_batch_stride;
        let right_base = batch * right_batch_stride;

        let mut sum = 0.0f32;
        let mut tile_start = 0;
        while tile_start < inner {
            // When an operand is stored transposed, swap the thread roles so
            // consecutive threads still read consecutive addresses (coalescing).
            let (left_row, left_inner) = if left_transposed != 0 { (local_x, local_y) } else { (local_y, local_x) };
            let (right_inner, right_col) = if right_transposed != 0 { (local_x, local_y) } else { (local_y, local_x) };

            let global_left_row = tile_row + left_row;
            let global_left_inner = tile_start + left_inner;
            let left_value = if global_left_row < rows && global_left_inner < inner {
                operand(left, left_base, global_left_row, global_left_inner, rows, inner, left_transposed)
            } else {
                0.0
            };

            let global_right_inner = tile_start + right_inner;
            let global_right_col = tile_col + right_col;
            let right_value = if global_right_inner < inner && global_right_col < cols {
                operand(right, right_base, global_right_col, global_right_inner, cols, inner, right_transposed ^ 1)
            } else {
                0.0
            };

            // SAFETY: block-private shared memory; every index < MATMUL_TILE².
            unsafe {
                LEFT_TILE[left_row * MATMUL_TILE + left_inner] = left_value;
                RIGHT_TILE[right_inner * MATMUL_TILE + right_col] = right_value;
            }
            thread::sync_threads();

            for step in 0..MATMUL_TILE {
                // SAFETY: indices < MATMUL_TILE²; tiles were fully written before the barrier.
                sum += unsafe { LEFT_TILE[local_y * MATMUL_TILE + step] * RIGHT_TILE[step * MATMUL_TILE + local_x] };
            }
            thread::sync_threads();

            tile_start += MATMUL_TILE;
        }

        let row = tile_row + local_y;
        let col = tile_col + local_x;
        let index = batch * output_batch_stride + row * cols + col;
        if row >= rows || col >= cols || index >= output.len() {
            return;
        }

        // SAFETY: index < output.len(); (batch, row, col) ↦ index is injective.
        let slot = unsafe { output.get_unchecked_mut(index) };
        *slot = if accumulate != 0 { *slot + scale * sum } else { scale * sum };
    }

    /// Register-blocked GEMM. A block of 16×16 threads computes a 64×64 output
    /// tile; each thread accumulates a 4×4 square of it in registers. Per
    /// 16-deep slice of the inner dimension a thread reads 4 + 4 values from
    /// shared memory and performs 16 multiply-adds, four times the arithmetic
    /// per shared-memory read of [`matmul_tiled`]. Thread `(x, y)` owns rows
    /// `y + 16i` and columns `x + 16j`, so the 32 threads of a warp read
    /// consecutive shared-memory words of the right tile and broadcast words
    /// of the left tile, without bank conflicts. The technique is described by
    /// Volkov & Demmel (2008), "Benchmarking GPUs to tune dense linear algebra",
    /// <https://doi.org/10.1109/SC.2008.5214359>, and in Simon Boehm's
    /// walk-through <https://siboehm.com/articles/22/CUDA-MMM>.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_blocked(
        left: &[f32],
        right: &[f32],
        mut output: DisjointSlice<f32>,
        rows: usize,
        cols: usize,
        inner: usize,
        left_transposed: u32,
        right_transposed: u32,
        left_batch_stride: usize,
        right_batch_stride: usize,
        output_batch_stride: usize,
        scale: f32,
        accumulate: u32,
    ) {
        static mut LEFT_TILE: SharedArray<f32, { BLOCKED_TILE * BLOCKED_DEPTH }> = SharedArray::UNINIT;
        static mut RIGHT_TILE: SharedArray<f32, { BLOCKED_DEPTH * BLOCKED_TILE }> = SharedArray::UNINIT;

        let batch = thread::blockIdx_z() as usize;
        let tile_row = thread::blockIdx_y() as usize * BLOCKED_TILE;
        let tile_col = thread::blockIdx_x() as usize * BLOCKED_TILE;
        let local_x = thread::threadIdx_x() as usize;
        let local_y = thread::threadIdx_y() as usize;
        let thread_index = local_y * BLOCKED_THREADS_PER_EDGE + local_x;
        let block_threads = BLOCKED_THREADS_PER_EDGE * BLOCKED_THREADS_PER_EDGE;
        let left_base = batch * left_batch_stride;
        let right_base = batch * right_batch_stride;

        let mut sums = [0.0f32; BLOCKED_PER_THREAD * BLOCKED_PER_THREAD];
        let mut tile_start = 0;
        while tile_start < inner {
            // Each thread stages four values of each operand. The element a
            // thread loads is chosen so consecutive threads read consecutive
            // global addresses in either storage order (coalescing).
            let mut load = 0;
            while load < BLOCKED_TILE * BLOCKED_DEPTH / block_threads {
                let linear = thread_index + load * block_threads;
                let (left_row, left_inner) = if left_transposed != 0 {
                    (linear % BLOCKED_TILE, linear / BLOCKED_TILE)
                } else {
                    (linear / BLOCKED_DEPTH, linear % BLOCKED_DEPTH)
                };
                let (right_inner, right_col) = if right_transposed != 0 {
                    (linear % BLOCKED_DEPTH, linear / BLOCKED_DEPTH)
                } else {
                    (linear / BLOCKED_TILE, linear % BLOCKED_TILE)
                };

                let global_left_row = tile_row + left_row;
                let global_left_inner = tile_start + left_inner;
                let left_value = if global_left_row < rows && global_left_inner < inner {
                    operand(left, left_base, global_left_row, global_left_inner, rows, inner, left_transposed)
                } else {
                    0.0
                };

                let global_right_inner = tile_start + right_inner;
                let global_right_col = tile_col + right_col;
                let right_value = if global_right_inner < inner && global_right_col < cols {
                    operand(right, right_base, global_right_col, global_right_inner, cols, inner, right_transposed ^ 1)
                } else {
                    0.0
                };

                // SAFETY: block-private shared memory; both indices < 64 · 16.
                unsafe {
                    LEFT_TILE[left_row * BLOCKED_DEPTH + left_inner] = left_value;
                    RIGHT_TILE[right_inner * BLOCKED_TILE + right_col] = right_value;
                }
                load += 1;
            }
            thread::sync_threads();

            let mut step = 0;
            while step < BLOCKED_DEPTH {
                let mut left_column = [0.0f32; BLOCKED_PER_THREAD];
                let mut right_row = [0.0f32; BLOCKED_PER_THREAD];
                let mut i = 0;
                while i < BLOCKED_PER_THREAD {
                    // SAFETY: indices < 64 · 16; tiles were fully written before the barrier.
                    unsafe {
                        left_column[i] = LEFT_TILE[(local_y + i * BLOCKED_THREADS_PER_EDGE) * BLOCKED_DEPTH + step];
                        right_row[i] = RIGHT_TILE[step * BLOCKED_TILE + local_x + i * BLOCKED_THREADS_PER_EDGE];
                    }
                    i += 1;
                }

                let mut i = 0;
                while i < BLOCKED_PER_THREAD {
                    let mut j = 0;
                    while j < BLOCKED_PER_THREAD {
                        sums[i * BLOCKED_PER_THREAD + j] += left_column[i] * right_row[j];
                        j += 1;
                    }
                    i += 1;
                }
                step += 1;
            }
            thread::sync_threads();

            tile_start += BLOCKED_DEPTH;
        }

        let mut i = 0;
        while i < BLOCKED_PER_THREAD {
            let row = tile_row + local_y + i * BLOCKED_THREADS_PER_EDGE;
            let mut j = 0;
            while j < BLOCKED_PER_THREAD {
                let col = tile_col + local_x + j * BLOCKED_THREADS_PER_EDGE;
                let index = batch * output_batch_stride + row * cols + col;
                if row < rows && col < cols && index < output.len() {
                    let sum = sums[i * BLOCKED_PER_THREAD + j];
                    // SAFETY: index < output.len(); (batch, row, col) ↦ index is injective.
                    let slot = unsafe { output.get_unchecked_mut(index) };
                    *slot = if accumulate != 0 { *slot + scale * sum } else { scale * sum };
                }
                j += 1;
            }
            i += 1;
        }
    }

    // -------------------------------------------------------- elementwise --

    /// `output += scale · input`.
    #[kernel]
    pub fn add_scaled(scale: f32, input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let position = index.get();
        if let Some(slot) = output.get_mut(index) {
            *slot += scale * input[position];
        }
    }

    /// `values *= factor`.
    #[kernel]
    pub fn scale_in_place(mut values: DisjointSlice<f32>, factor: f32) {
        if let Some(slot) = values.get_mut(thread::index_1d()) {
            *slot *= factor;
        }
    }

    /// `output[r, c] += bias[c]` for a `[rows, cols]` matrix.
    #[kernel]
    pub fn add_row_bias(mut output: DisjointSlice<f32>, bias: &[f32], cols: usize) {
        let index = thread::index_1d();
        let position = index.get();
        if let Some(slot) = output.get_mut(index) {
            *slot += bias[position % cols];
        }
    }

    /// `sums[c] += Σ_r input[r, c]`; one thread per column, so consecutive
    /// threads read consecutive addresses in every row.
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

    /// `output = GELU(input)`.
    #[kernel]
    pub fn gelu_forward(input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let position = index.get();
        if let Some(slot) = output.get_mut(index) {
            *slot = gelu(input[position]);
        }
    }

    /// `input_grad = output_grad · GELU'(input)`.
    #[kernel]
    pub fn gelu_backward(input: &[f32], output_grad: &[f32], mut input_grad: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let position = index.get();
        if let Some(slot) = input_grad.get_mut(index) {
            *slot = output_grad[position] * gelu_derivative(input[position]);
        }
    }

    // ------------------------------------------------------ normalisation --

    /// RMSNorm (Zhang & Sennrich, 2019, <https://arxiv.org/abs/1910.07467>):
    /// `output = input · inv_rms · gain` with `inv_rms = 1/√(mean(input²) + ε)`.
    /// `inv_rms` is saved for the backward pass.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_forward(
        input: &[f32],
        gain: &[f32],
        mut output: DisjointSlice<f32>,
        mut inv_rms: DisjointSlice<f32>,
        rows: usize,
        cols: usize,
        epsilon: f32,
    ) {
        let WarpRow { row, lane } = warp_row();
        if row >= rows {
            return;
        }
        let base = row * cols;

        let mut square_sum = 0.0f32;
        let mut col = lane;
        while col < cols {
            square_sum += input[base + col] * input[base + col];
            col += WARP_SIZE;
        }
        let inverse = 1.0 / (warp::reduce_sum_f32(square_sum) / cols as f32 + epsilon).sqrt();

        let mut col = lane;
        while col < cols {
            // SAFETY: base + col < rows · cols, asserted equal to output.len() by the host.
            unsafe { *output.get_unchecked_mut(base + col) = input[base + col] * inverse * gain[col] };
            col += WARP_SIZE;
        }

        if lane == 0 {
            // SAFETY: row < rows, asserted ≤ inv_rms.len() by the host.
            unsafe { *inv_rms.get_unchecked_mut(row) = inverse };
        }
    }

    /// RMSNorm input gradient. With `s = inv_rms`:
    /// `dx = s·g⊙dy − x · s³/cols · Σ(dy⊙g⊙x)`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_backward_input(
        output_grad: &[f32],
        input: &[f32],
        gain: &[f32],
        inv_rms: &[f32],
        mut input_grad: DisjointSlice<f32>,
        rows: usize,
        cols: usize,
    ) {
        let WarpRow { row, lane } = warp_row();
        if row >= rows {
            return;
        }
        let base = row * cols;
        let inverse = inv_rms[row];

        let mut projection = 0.0f32;
        let mut col = lane;
        while col < cols {
            projection += output_grad[base + col] * gain[col] * input[base + col];
            col += WARP_SIZE;
        }
        let correction = inverse * inverse * inverse * warp::reduce_sum_f32(projection) / cols as f32;

        let mut col = lane;
        while col < cols {
            let gradient = inverse * gain[col] * output_grad[base + col] - input[base + col] * correction;
            // SAFETY: base + col < rows · cols, asserted equal to input_grad.len() by the host.
            unsafe { *input_grad.get_unchecked_mut(base + col) = gradient };
            col += WARP_SIZE;
        }
    }

    /// RMSNorm gain gradient, accumulated: `dg[c] += Σ_r dy[r,c] · x[r,c] · s[r]`.
    #[kernel]
    pub fn rms_norm_backward_gain(
        output_grad: &[f32],
        input: &[f32],
        inv_rms: &[f32],
        mut gain_grad: DisjointSlice<f32>,
        rows: usize,
        cols: usize,
    ) {
        let index = thread::index_1d();
        let col = index.get();
        if col >= cols {
            return;
        }

        let mut sum = 0.0f32;
        for row in 0..rows {
            sum += output_grad[row * cols + col] * input[row * cols + col] * inv_rms[row];
        }

        if let Some(slot) = gain_grad.get_mut(index) {
            *slot += sum;
        }
    }

    // ---------------------------------------------------------- attention --

    /// In-place causal softmax over rows of `scores` `[matrices · queries, keys]`.
    /// Query `q` may attend to keys `0..=q + query_offset`; later keys get 0.
    /// Uses the max-subtraction trick for numerical stability
    /// (<https://en.wikipedia.org/wiki/Softmax_function#Numerical_algorithms>).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn causal_softmax(mut scores: DisjointSlice<f32>, rows: usize, queries: usize, keys: usize, scale: f32, query_offset: usize) {
        let WarpRow { row, lane } = warp_row();
        if row >= rows {
            return;
        }
        let base = row * keys;
        let visible_keys = (row % queries + query_offset + 1).min(keys);

        let mut maximum = NEGATIVE_LARGE;
        let mut key = lane;
        while key < visible_keys {
            // SAFETY: base + key < rows · keys, asserted equal to scores.len() by the host.
            maximum = maximum.max(unsafe { *scores.get_unchecked_mut(base + key) } * scale);
            key += WARP_SIZE;
        }
        let maximum = warp::reduce_max_f32(maximum);

        let mut total = 0.0f32;
        let mut key = lane;
        while key < visible_keys {
            // SAFETY: as above.
            let slot = unsafe { scores.get_unchecked_mut(base + key) };
            *slot = libm::expf(*slot * scale - maximum);
            total += *slot;
            key += WARP_SIZE;
        }
        let normaliser = 1.0 / warp::reduce_sum_f32(total);

        let mut key = lane;
        while key < keys {
            // SAFETY: as above.
            let slot = unsafe { scores.get_unchecked_mut(base + key) };
            *slot = if key < visible_keys { *slot * normaliser } else { 0.0 };
            key += WARP_SIZE;
        }
    }

    /// Softmax backward in place: `grad ← p ⊙ (grad − Σ(grad ⊙ p))` per row.
    #[kernel]
    pub fn softmax_backward(probabilities: &[f32], mut grad: DisjointSlice<f32>, rows: usize, cols: usize) {
        let WarpRow { row, lane } = warp_row();
        if row >= rows {
            return;
        }
        let base = row * cols;

        let mut projection = 0.0f32;
        let mut col = lane;
        while col < cols {
            // SAFETY: base + col < rows · cols, asserted equal to grad.len() by the host.
            projection += unsafe { *grad.get_unchecked_mut(base + col) } * probabilities[base + col];
            col += WARP_SIZE;
        }
        let projection = warp::reduce_sum_f32(projection);

        let mut col = lane;
        while col < cols {
            // SAFETY: as above.
            let slot = unsafe { grad.get_unchecked_mut(base + col) };
            *slot = probabilities[base + col] * (*slot - projection);
            col += WARP_SIZE;
        }
    }

    /// Head-major index of element (`batch`, `head`, `time`, `dim`).
    #[allow(clippy::too_many_arguments)]
    fn head_major_index(
        batch: usize,
        head: usize,
        time: usize,
        dim: usize,
        heads: usize,
        head_dim: usize,
        time_capacity: usize,
        time_offset: usize,
    ) -> usize {
        ((batch * heads + head) * time_capacity + time_offset + time) * head_dim + dim
    }

    /// Packed `[batch · time, width]` → head-major `[batch, heads, capacity, head_dim]`.
    /// One thread per element of the `[batch, heads, time, head_dim]` view.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn split_heads(
        packed: &[f32],
        mut head_major: DisjointSlice<f32>,
        batch_size: usize,
        time: usize,
        heads: usize,
        head_dim: usize,
        width: usize,
        column_offset: usize,
        time_capacity: usize,
        time_offset: usize,
    ) {
        let element = global_thread();
        if element >= batch_size * heads * time * head_dim {
            return;
        }

        let dim = element % head_dim;
        let step = (element / head_dim) % time;
        let head = (element / (head_dim * time)) % heads;
        let batch = element / (head_dim * time * heads);

        let source = (batch * time + step) * width + column_offset + head * head_dim + dim;
        let target = head_major_index(batch, head, step, dim, heads, head_dim, time_capacity, time_offset);
        if target < head_major.len() {
            // SAFETY: bound-checked; the element ↦ target map is injective.
            unsafe { *head_major.get_unchecked_mut(target) = packed[source] };
        }
    }

    /// Inverse of [`split_heads`]: head-major → packed columns (others untouched).
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn merge_heads(
        head_major: &[f32],
        mut packed: DisjointSlice<f32>,
        batch_size: usize,
        time: usize,
        heads: usize,
        head_dim: usize,
        width: usize,
        column_offset: usize,
        time_capacity: usize,
        time_offset: usize,
    ) {
        let element = global_thread();
        if element >= batch_size * heads * time * head_dim {
            return;
        }

        let dim = element % head_dim;
        let step = (element / head_dim) % time;
        let head = (element / (head_dim * time)) % heads;
        let batch = element / (head_dim * time * heads);

        let source = head_major_index(batch, head, step, dim, heads, head_dim, time_capacity, time_offset);
        let target = (batch * time + step) * width + column_offset + head * head_dim + dim;
        if target < packed.len() {
            // SAFETY: bound-checked; the element ↦ target map is injective.
            unsafe { *packed.get_unchecked_mut(target) = head_major[source] };
        }
    }

    // --------------------------------------------------------- embeddings --

    /// `output[i] (+)= Σ_s table[tokens[i·slots + s]]`, skipping [`EMPTY_TOKEN`].
    #[kernel]
    pub fn embedding_forward(table: &[f32], tokens: &[u32], mut output: DisjointSlice<f32>, slots: usize, width: usize, accumulate: u32) {
        let index = thread::index_1d();
        let element = index.get();
        if element >= output.len() {
            return;
        }
        let position = element / width;
        let col = element % width;

        let mut sum = 0.0f32;
        for slot in 0..slots {
            let token = tokens[position * slots + slot];
            if token != EMPTY_TOKEN {
                sum += table[token as usize * width + col];
            }
        }

        if let Some(value) = output.get_mut(index) {
            *value = if accumulate != 0 { *value + sum } else { sum };
        }
    }

    /// Embedding gradient without atomics: thread (token, col) walks the rows
    /// that used `token`, listed by a host-built inverted index
    /// (`row_offsets[token]..row_offsets[token+1]` into `rows_by_token`).
    /// Deterministic, unlike an atomic scatter-add.
    #[kernel]
    pub fn embedding_backward(
        output_grad: &[f32],
        row_offsets: &[u32],
        rows_by_token: &[u32],
        mut table_grad: DisjointSlice<f32>,
        width: usize,
    ) {
        let index = thread::index_1d();
        let element = index.get();
        let token = element / width;
        let col = element % width;
        if token + 1 >= row_offsets.len() {
            return;
        }

        let mut sum = 0.0f32;
        for entry in row_offsets[token] as usize..row_offsets[token + 1] as usize {
            sum += output_grad[rows_by_token[entry] as usize * width + col];
        }

        if let Some(value) = table_grad.get_mut(index) {
            *value += sum;
        }
    }

    // --------------------------------------------------------------- loss --

    /// Softmax cross-entropy for one codebook head. Row `r` reads logits
    /// `[r·row_stride + column_offset, +classes)` and target
    /// `targets[r·target_stride + target_offset]`. Writes the per-row loss and
    /// `(softmax − one_hot) · grad_scale` into the same logit positions of
    /// `logits_grad`. Rows whose target is [`EMPTY_TOKEN`] get zero loss and gradient.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn cross_entropy(
        logits: &[f32],
        targets: &[u32],
        mut losses: DisjointSlice<f32>,
        mut logits_grad: DisjointSlice<f32>,
        rows: usize,
        row_stride: usize,
        column_offset: usize,
        classes: usize,
        target_stride: usize,
        target_offset: usize,
        grad_scale: f32,
    ) {
        let WarpRow { row, lane } = warp_row();
        if row >= rows {
            return;
        }
        let base = row * row_stride + column_offset;
        let target = targets[row * target_stride + target_offset];
        let has_target = target != EMPTY_TOKEN;

        let mut maximum = NEGATIVE_LARGE;
        let mut class = lane;
        while class < classes {
            maximum = maximum.max(logits[base + class]);
            class += WARP_SIZE;
        }
        let maximum = warp::reduce_max_f32(maximum);

        let mut total = 0.0f32;
        let mut class = lane;
        while class < classes {
            total += libm::expf(logits[base + class] - maximum);
            class += WARP_SIZE;
        }
        let log_partition = maximum + libm::logf(warp::reduce_sum_f32(total));

        let mut class = lane;
        while class < classes {
            let probability = libm::expf(logits[base + class] - log_partition);
            let one_hot = if class as u32 == target { 1.0 } else { 0.0 };
            let gradient = if has_target { (probability - one_hot) * grad_scale } else { 0.0 };
            // SAFETY: base + class < rows · row_stride, asserted ≤ logits_grad.len() by the host.
            unsafe { *logits_grad.get_unchecked_mut(base + class) = gradient };
            class += WARP_SIZE;
        }

        if lane == 0 {
            let loss = if has_target { log_partition - logits[base + target as usize] } else { 0.0 };
            // SAFETY: row < rows, asserted ≤ losses.len() by the host.
            unsafe { *losses.get_unchecked_mut(row) = loss };
        }
    }

    // ---------------------------------------------------------- optimiser --

    /// One AdamW step (Loshchilov & Hutter, 2019, <https://arxiv.org/abs/1711.05101>, Alg. 2).
    /// `first_correction = 1 − β₁ᵗ`, `second_correction = 1 − β₂ᵗ`.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_step(
        mut parameter: DisjointSlice<f32>,
        gradient: &[f32],
        mut first_moment: DisjointSlice<f32>,
        mut second_moment: DisjointSlice<f32>,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        first_correction: f32,
        second_correction: f32,
    ) {
        let index = global_thread();
        if index >= parameter.len() || index >= first_moment.len() || index >= second_moment.len() {
            return;
        }
        let grad = gradient[index];

        // SAFETY: index is bound-checked against all three slices; one thread per index.
        unsafe {
            let mean = first_moment.get_unchecked_mut(index);
            *mean = beta1 * *mean + (1.0 - beta1) * grad;
            let mean_hat = *mean / first_correction;

            let variance = second_moment.get_unchecked_mut(index);
            *variance = beta2 * *variance + (1.0 - beta2) * grad * grad;
            let variance_hat = *variance / second_correction;

            let weight = parameter.get_unchecked_mut(index);
            *weight -= learning_rate * (mean_hat / (variance_hat.sqrt() + epsilon) + weight_decay * *weight);
        }
    }

    /// Grid-stride sum of squares; block `b` writes its partial sum to `partial_sums[b]`.
    #[kernel]
    #[launch_bounds(256)]
    pub fn sum_of_squares(values: &[f32], mut partial_sums: DisjointSlice<f32>) {
        static mut WARP_SUMS: SharedArray<f32, { super::WARPS_PER_BLOCK }> = SharedArray::UNINIT;

        let local = thread::threadIdx_x() as usize;
        let grid_stride = thread::gridDim_x() as usize * thread::blockDim_x() as usize;

        let mut sum = 0.0f32;
        let mut index = global_thread();
        while index < values.len() {
            sum += values[index] * values[index];
            index += grid_stride;
        }
        let warp_sum = warp::reduce_sum_f32(sum);

        if local.is_multiple_of(WARP_SIZE) {
            // SAFETY: local / WARP_SIZE < WARPS_PER_BLOCK for a 256-thread block.
            unsafe { WARP_SUMS[local / WARP_SIZE] = warp_sum };
        }
        thread::sync_threads();

        let block = thread::blockIdx_x() as usize;
        if local != 0 || block >= partial_sums.len() {
            return;
        }

        let mut block_sum = 0.0f32;
        for warp_index in 0..super::WARPS_PER_BLOCK {
            // SAFETY: warp_index < WARPS_PER_BLOCK; written before the barrier.
            block_sum += unsafe { WARP_SUMS[warp_index] };
        }
        // SAFETY: block < partial_sums.len(); one writer per block.
        unsafe { *partial_sums.get_unchecked_mut(block) = block_sum };
    }

    // ------------------------------------------------------------- signal --

    /// Overlapping frames of a pre-padded signal: `frames[f, n] = signal[f·hop + n]`.
    #[kernel]
    pub fn frame_signal(signal: &[f32], mut frames: DisjointSlice<f32>, frame_length: usize, hop: usize) {
        let index = thread::index_1d();
        let element = index.get();
        let source = (element / frame_length) * hop + element % frame_length;
        if let Some(slot) = frames.get_mut(index) {
            *slot = signal[source];
        }
    }

    /// Power spectrum `re² + im²`.
    #[kernel]
    pub fn power_spectrum(real: &[f32], imaginary: &[f32], mut power: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let element = index.get();
        if let Some(slot) = power.get_mut(index) {
            *slot = real[element] * real[element] + imaginary[element] * imaginary[element];
        }
    }

    /// `values = ln(max(values, floor))`.
    #[kernel]
    pub fn log_compress(mut values: DisjointSlice<f32>, floor: f32) {
        if let Some(slot) = values.get_mut(thread::index_1d()) {
            *slot = libm::logf(slot.max(floor));
        }
    }

    /// Nearest centroid per row from precomputed `dots = x·cᵀ` and `‖c‖²`:
    /// `argmin_k ‖c_k‖² − 2·x·c_k` (the `‖x‖²` term is shared by all k).
    /// See k-means, <https://en.wikipedia.org/wiki/K-means_clustering>
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
