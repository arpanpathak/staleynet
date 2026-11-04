//! Safe launch wrappers, grouped by the kind of computation. Every wrapper
//! checks the shapes it is given before launching, so the size invariants
//! the kernels rely on hold by construction.

mod attention;
mod elementwise;
mod embedding;
mod loss;
mod matmul;
mod normalization;
mod optimizer;
mod signal;

pub use attention::HeadLayout;
pub use embedding::TokenIndex;
pub use loss::CrossEntropyHead;
pub use matmul::{BatchStrides, MatmulSpec, Storage};
pub use optimizer::AdamWHyperparameters;

/// Whether an operation replaces its output or adds into it. Accumulation is
/// how gradients from several uses of one parameter are summed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteMode {
    /// `output = result`.
    Overwrite,

    /// `output += result`.
    Accumulate,
}

impl WriteMode {
    /// Kernel flag: 1 for accumulate, 0 for overwrite.
    pub(crate) fn flag(self) -> u32 {
        match self {
            WriteMode::Overwrite => 0,
            WriteMode::Accumulate => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_mode_flags_match_the_kernel_convention() {
        assert_eq!(WriteMode::Overwrite.flag(), 0);
        assert_eq!(WriteMode::Accumulate.flag(), 1);
    }
}
