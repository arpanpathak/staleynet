//! Building blocks of causal multi-head attention (Vaswani et al., 2017,
//! <https://arxiv.org/abs/1706.03762>, §3.2): head layout permutes and the
//! masked softmax. The score and value products themselves are batched
//! [`MatmulSpec`](super::MatmulSpec)s.

use crate::device::{Gpu, GpuResult, Tensor};
use crate::launch;

/// Where a group of heads lives in a packed activation and in a head-major tensor.
///
/// Packed layout: `[batch · time, width]`, head `h` occupying columns
/// `column_offset + h·head_dim ..`. Head-major layout:
/// `[batch, heads, time_capacity, head_dim]`, written at `time_offset ..`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadLayout {
    /// Sequences in the batch.
    pub batch: usize,

    /// Time steps being moved.
    pub time: usize,

    /// Attention heads.
    pub heads: usize,

    /// Width of one head.
    pub head_dim: usize,

    /// Row width of the packed activation.
    pub width: usize,

    /// First packed column of head 0 (selects Q, K or V inside a fused QKV row).
    pub column_offset: usize,

    /// Time length allocated in the head-major tensor (≥ `time_offset + time`).
    pub time_capacity: usize,

    /// First head-major time step written or read (non-zero when appending to a KV cache).
    pub time_offset: usize,
}

impl HeadLayout {
    /// Layout for dense activations: capacity equals `time`, offset 0.
    pub fn dense(batch: usize, time: usize, heads: usize, head_dim: usize, width: usize, column_offset: usize) -> Self {
        Self { batch, time, heads, head_dim, width, column_offset, time_capacity: time, time_offset: 0 }
    }

    /// Same layout, but addressing a cache of `capacity` steps starting at `offset`.
    pub fn in_cache(self, capacity: usize, offset: usize) -> Self {
        Self { time_capacity: capacity, time_offset: offset, ..self }
    }

    /// Elements moved by one permute.
    fn element_count(&self) -> usize {
        self.batch * self.heads * self.time * self.head_dim
    }

    /// Asserts both tensors are large enough for this layout.
    fn check(&self, packed: &Tensor, head_major: &Tensor) {
        assert!(self.time_offset + self.time <= self.time_capacity, "HeadLayout: time range exceeds capacity");
        assert!(packed.len() >= self.batch * self.time * self.width, "HeadLayout: packed tensor too small");
        assert!(
            head_major.len() >= self.batch * self.heads * self.time_capacity * self.head_dim,
            "HeadLayout: head-major tensor too small"
        );
    }
}

impl Gpu {
    /// Packed columns → head-major tensor.
    pub fn split_heads(&self, packed: &Tensor, head_major: &mut Tensor, layout: &HeadLayout) -> GpuResult {
        layout.check(packed, head_major);
        let HeadLayout { batch, time, heads, head_dim, width, column_offset, time_capacity, time_offset } = *layout;

        // SAFETY: one thread per moved element; every write is bound-checked in-kernel.
        unsafe {
            self.kernels.split_heads(
                &self.stream,
                launch::elementwise(layout.element_count()),
                &packed.buffer,
                &mut head_major.buffer,
                batch,
                time,
                heads,
                head_dim,
                width,
                column_offset,
                time_capacity,
                time_offset,
            )
        }
    }

    /// Head-major tensor → packed columns (other columns are left untouched).
    pub fn merge_heads(&self, head_major: &Tensor, packed: &mut Tensor, layout: &HeadLayout) -> GpuResult {
        layout.check(packed, head_major);
        let HeadLayout { batch, time, heads, head_dim, width, column_offset, time_capacity, time_offset } = *layout;

        // SAFETY: one thread per moved element; every write is bound-checked in-kernel.
        unsafe {
            self.kernels.merge_heads(
                &self.stream,
                launch::elementwise(layout.element_count()),
                &head_major.buffer,
                &mut packed.buffer,
                batch,
                time,
                heads,
                head_dim,
                width,
                column_offset,
                time_capacity,
                time_offset,
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
            self.kernels.causal_softmax(
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
            self.kernels.softmax_backward(
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
