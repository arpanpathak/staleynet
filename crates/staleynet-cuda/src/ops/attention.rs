//! Building blocks of causal multi-head attention (Vaswani et al., 2017,
//! <https://arxiv.org/abs/1706.03762>, §3.2): head layout permutes and the
//! masked softmax. The score and value products themselves are batched
//! [`MatmulSpec`](super::MatmulSpec)s.

use crate::device::{Gpu, GpuResult, Tensor};
pub use crate::kernels::attention::HeadLayout;
use crate::launch;

/// Asserts that both tensors are large enough for `layout`.
fn check_layout(layout: &HeadLayout, packed: &Tensor, head_major: &Tensor) {
    assert!(layout.time_offset + layout.time <= layout.time_capacity, "HeadLayout: time range exceeds capacity");
    assert!(packed.len() >= layout.batch * layout.time * layout.width, "HeadLayout: packed tensor too small");
    assert!(
        head_major.len() >= layout.batch * layout.heads * layout.time_capacity * layout.head_dim,
        "HeadLayout: head-major tensor too small"
    );
}

impl Gpu {
    /// Packed columns → head-major tensor.
    pub fn split_heads(&self, packed: &Tensor, head_major: &mut Tensor, layout: &HeadLayout) -> GpuResult {
        check_layout(layout, packed, head_major);

        // SAFETY: one thread per moved element; every write is bound-checked in the kernel.
        unsafe {
            self.kernels.attention.split_heads(
                &self.stream,
                launch::elementwise(layout.element_count()),
                &packed.buffer,
                &mut head_major.buffer,
                *layout,
            )
        }
    }

    /// Head-major tensor → packed columns (other columns are left untouched).
    pub fn merge_heads(&self, head_major: &Tensor, packed: &mut Tensor, layout: &HeadLayout) -> GpuResult {
        check_layout(layout, packed, head_major);

        // SAFETY: one thread per moved element; every write is bound-checked in the kernel.
        unsafe {
            self.kernels.attention.merge_heads(
                &self.stream,
                launch::elementwise(layout.element_count()),
                &head_major.buffer,
                &mut packed.buffer,
                *layout,
            )
        }
    }

    /// In-place causal softmax of `scores` `[matrices · queries, keys]`,
    /// applying `scale` before exponentiation. Query `q` sees keys
    /// `0..=q + query_offset`, which lets one kernel serve both training
    /// (offset 0) and cached decoding (offset = tokens already cached).
    pub fn causal_softmax(&self, scores: &mut Tensor, queries: usize, scale: f32, query_offset: usize) -> GpuResult {
        let shape = scores.shape();
        assert_eq!(shape.rows % queries, 0, "causal_softmax: rows must be a multiple of queries");

        // SAFETY: warp per row; writes stay inside rows·keys == scores.len().
        unsafe {
            self.kernels.attention.causal_softmax(
                &self.stream,
                launch::warp_per_row(shape.rows),
                &mut scores.buffer,
                shape.rows,
                queries,
                shape.cols,
                scale,
                query_offset,
            )
        }
    }

    /// In place on `grad`: the gradient with respect to the softmax input.
    pub fn softmax_backward(&self, probabilities: &Tensor, grad: &mut Tensor) -> GpuResult {
        let shape = probabilities.shape();
        assert_eq!(grad.shape(), shape, "softmax_backward: shape mismatch");

        // SAFETY: warp per row; writes stay inside rows·cols == grad.len().
        unsafe {
            self.kernels.attention.softmax_backward(
                &self.stream,
                launch::warp_per_row(shape.rows),
                &probabilities.buffer,
                &mut grad.buffer,
                shape.rows,
                shape.cols,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_layout_sets_capacity_and_zero_offset() {
        let layout = HeadLayout::dense(2, 5, 4, 8, 32, 16);

        assert_eq!(layout.batch, 2);
        assert_eq!(layout.time, 5);
        assert_eq!(layout.heads, 4);
        assert_eq!(layout.head_dim, 8);
        assert_eq!(layout.width, 32);
        assert_eq!(layout.column_offset, 16);
        assert_eq!(layout.time_capacity, 5);
        assert_eq!(layout.time_offset, 0);
    }

    #[test]
    fn in_cache_changes_only_the_cache_geometry() {
        let layout = HeadLayout::dense(2, 5, 4, 8, 32, 16).in_cache(9, 3);

        assert_eq!(layout.time, 5);
        assert_eq!(layout.time_capacity, 9);
        assert_eq!(layout.time_offset, 3);
        assert_eq!(layout.heads, 4);
        assert_eq!(layout.head_dim, 8);
    }

    #[test]
    fn element_count_multiplies_batch_heads_time_and_width() {
        let layout = HeadLayout::dense(2, 5, 4, 8, 32, 16);

        assert_eq!(layout.element_count(), 2 * 4 * 5 * 8);
    }
}
