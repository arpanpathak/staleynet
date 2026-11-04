//! Launch geometry for the three kernel shapes described in [`crate::kernels`].
//! See *CUDA C++ Programming Guide*, "Thread Hierarchy":
//! <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy>

use crate::kernels::{BLOCK_THREADS, BLOCKED_THREADS_PER_EDGE, BLOCKED_TILE, MATMUL_TILE, WARPS_PER_BLOCK};
// The launch configuration taken by the launchers that `#[cuda_module]` generates.
use cuda_core::LaunchConfig;

/// Blocks a grid-stride reduction launches; enough to fill every SM on Orin
/// and Thor several times over, small enough that the host sum is trivial.
pub const REDUCTION_BLOCKS: usize = 128;

/// One thread per element over `count` elements.
pub fn elementwise(count: usize) -> LaunchConfig {
    LaunchConfig::for_num_elems(count.max(1) as u32)
}

/// One warp per row over `rows` rows.
pub fn warp_per_row(rows: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows.max(1).div_ceil(WARPS_PER_BLOCK) as u32, 1, 1),
        block_dim: (BLOCK_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// One 16×16 block per output tile, `z` enumerating the batch.
pub fn matmul_tiles(rows: usize, cols: usize, batch: usize) -> LaunchConfig {
    let tile = MATMUL_TILE as u32;

    LaunchConfig {
        grid_dim: ((cols as u32).div_ceil(tile), (rows as u32).div_ceil(tile), batch as u32),
        block_dim: (tile, tile, 1),
        shared_mem_bytes: 0,
    }
}

/// One 16×16-thread block per 64×64 output tile, for the blocked GEMM.
pub fn matmul_blocks(rows: usize, cols: usize, batch: usize) -> LaunchConfig {
    let tile = BLOCKED_TILE as u32;
    let edge = BLOCKED_THREADS_PER_EDGE as u32;

    LaunchConfig {
        grid_dim: ((cols as u32).div_ceil(tile), (rows as u32).div_ceil(tile), batch as u32),
        block_dim: (edge, edge, 1),
        shared_mem_bytes: 0,
    }
}

/// A fixed grid of [`REDUCTION_BLOCKS`] full blocks for grid-stride reductions.
pub fn reduction() -> LaunchConfig {
    LaunchConfig { grid_dim: (REDUCTION_BLOCKS as u32, 1, 1), block_dim: (BLOCK_THREADS as u32, 1, 1), shared_mem_bytes: 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elementwise_launches_at_least_one_block() {
        let one = elementwise(0);
        let many = elementwise(1_000_000);

        assert_eq!(one.grid_dim.0, 1);
        assert!(one.block_dim.0 > 0);
        assert!(many.grid_dim.0 > 1);
        assert_eq!(one.shared_mem_bytes, 0);
    }

    #[test]
    fn warp_per_row_uses_one_block_per_warp_group() {
        let empty = warp_per_row(0);
        let exact = warp_per_row(WARPS_PER_BLOCK);
        let ragged = warp_per_row(WARPS_PER_BLOCK * 2 + 1);

        assert_eq!(empty.grid_dim.0, 1);
        assert_eq!(exact.grid_dim.0, 1);
        assert_eq!(ragged.grid_dim.0, 3);
        assert_eq!(exact.block_dim.0, BLOCK_THREADS as u32);
        assert_eq!(exact.shared_mem_bytes, 0);
    }

    #[test]
    fn matmul_tiles_rounds_up_and_keeps_the_batch_axis() {
        let config = matmul_tiles(MATMUL_TILE + 1, MATMUL_TILE * 2, 3);

        assert_eq!(config.grid_dim.0, 2);
        assert_eq!(config.grid_dim.1, 2);
        assert_eq!(config.grid_dim.2, 3);
        assert_eq!(config.block_dim.0, MATMUL_TILE as u32);
        assert_eq!(config.block_dim.1, MATMUL_TILE as u32);
        assert_eq!(config.shared_mem_bytes, 0);
    }

    #[test]
    fn blocked_matmul_uses_the_blocked_tile_and_thread_edge() {
        let config = matmul_blocks(BLOCKED_TILE + 1, BLOCKED_TILE + 1, 2);

        assert_eq!(config.grid_dim.0, 2);
        assert_eq!(config.grid_dim.1, 2);
        assert_eq!(config.grid_dim.2, 2);
        assert_eq!(config.block_dim.0, BLOCKED_THREADS_PER_EDGE as u32);
        assert_eq!(config.block_dim.1, BLOCKED_THREADS_PER_EDGE as u32);
    }

    #[test]
    fn reduction_uses_the_fixed_grid() {
        let config = reduction();

        assert_eq!(config.grid_dim.0, REDUCTION_BLOCKS as u32);
        assert_eq!(config.block_dim.0, BLOCK_THREADS as u32);
        assert_eq!(config.grid_dim.1, 1);
        assert_eq!(config.grid_dim.2, 1);
    }
}
