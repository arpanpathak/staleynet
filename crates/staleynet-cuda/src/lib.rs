//! GPU compute for StaleyNET, written entirely in Rust with cuda-oxide.
//!
//! * [`kernels`]: device code (compiled to PTX).
//! * [`device`]: the [`Gpu`] handle and the [`Tensor`] type.
//! * [`ops`]: safe, shape-checked wrappers that launch the kernels.
//! * [`reference`]: plain CPU implementations every kernel is tested against.

pub mod device;
pub mod kernels;
pub mod launch;
pub mod ops;
pub mod reference;

pub use device::{Gpu, GpuInitError, GpuResult, MatmulKernel, Shape, Tensor, TokenBuffer};
pub use kernels::EMPTY_TOKEN;
pub use ops::{AdamWHyperparameters, BatchStrides, Convolution, CrossEntropyHead, HeadLayout, MatmulSpec, Storage, TokenIndex, WriteMode};
