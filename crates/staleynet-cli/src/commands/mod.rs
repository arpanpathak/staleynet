//! One module per subcommand. Each exposes `Arguments` and `run`.

pub mod bench;
pub mod evaluate;
pub mod prepare;
pub mod reconstruct;
pub mod sing;
pub mod tokenize;
pub mod train;
pub mod vocoder;

use anyhow::{Context, Result};
use staleynet_cuda::Gpu;

/// Opens CUDA device 0 with a readable error.
pub fn open_gpu() -> Result<Gpu> {
    Gpu::new(0).context("open CUDA device 0")
}
