//! Device code: every GPU kernel StaleyNET runs, written in Rust and compiled
//! to PTX by the cuda-oxide backend (<https://github.com/NVlabs/cuda-oxide>).
//!
//! Each file holds one `#[cuda_module]` for one kind of computation, and each
//! module is loaded separately (see [`crate::device::Kernels`]). The kernels
//! divide their work in one of four ways:
//!
//! * **one thread per output element**, for elementwise operations;
//! * **one warp per matrix row**, for operations that need a whole row (the
//!   32 lanes of a warp stride across the row and combine their partial
//!   results by exchanging registers; see NVIDIA, *Using CUDA Warp-Level
//!   Primitives*, <https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/>);
//! * **one block per output tile**, for matrix products (Kirk, Hwu and
//!   El Hajj, *Programming Massively Parallel Processors*, 4th ed., ch. 5);
//! * **a fixed grid of blocks that sweeps the data**, for reductions.
//!
//! Every write through a computed index goes through [`device::store`], which
//! checks the index; the host wrappers in [`crate::ops`] check the sizes that
//! make each launch valid.

pub mod attention;
pub mod device;
pub mod elementwise;
pub mod embedding;
pub mod loss;
pub mod matmul;
pub mod normalization;
pub mod optimizer;
pub mod signal;
pub mod synthesis;

/// Threads per warp on every NVIDIA architecture to date.
/// See <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#simt-architecture>
pub const WARP_SIZE: usize = 32;

/// Threads per block for elementwise and warp-per-row kernels.
pub const BLOCK_THREADS: usize = 256;

/// Warps per block for warp-per-row kernels.
pub const WARPS_PER_BLOCK: usize = BLOCK_THREADS / WARP_SIZE;

/// Marks an empty token slot; such slots contribute no embedding and no loss.
pub const EMPTY_TOKEN: u32 = u32::MAX;

/// Stands in for minus infinity when taking a running maximum.
pub const NEGATIVE_LARGE: f32 = -3.0e38;

pub use matmul::{BLOCKED_THREADS_PER_EDGE, BLOCKED_TILE, MATMUL_TILE, Operand, Product};
pub use synthesis::Convolution;
