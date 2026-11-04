//! Causal multi-head self-attention (Vaswani et al., 2017,
//! <https://arxiv.org/abs/1706.03762>, §3.2.2):
//! `softmax(Q·Kᵀ/√d_head + causal mask)·V`, with Q, K, V from one fused projection.
//!
//! Backward pass, per head, with `P = softmax(S)` and `S = scale·Q·Kᵀ`:
//! `∂V = Pᵀ·∂O`, `∂P = ∂O·Vᵀ`, `∂S = P ⊙ (∂P − rowsum(∂P ⊙ P))`,
//! `∂Q = scale·∂S·K`, `∂K = scale·∂Sᵀ·Q`.

use super::SequenceShape;
use super::linear::Linear;
use crate::parameter::{Parameter, Parameters};
use crate::random::Random;
use staleynet_cuda::{BatchStrides, Gpu, GpuResult, HeadLayout, MatmulSpec, Shape, Tensor};

/// Which third of a fused `[Q | K | V]` row a head group occupies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Projection {
    /// Queries, columns `0..width`.
    Query,

    /// Keys, columns `width..2·width`.
    Key,

    /// Values, columns `2·width..3·width`.
    Value,
}

impl Projection {
    /// First column of this projection inside a fused row.
    fn column_offset(self, width: usize) -> usize {
        match self {
            Projection::Query => 0,
            Projection::Key => width,
            Projection::Value => 2 * width,
        }
    }
}

/// The attention layer's parameters and head geometry.
pub struct CausalSelfAttention {
    /// Fused input projection `width → 3·width` producing Q, K and V.
    pub qkv: Linear,

    /// Output projection `width → width` back into the residual stream.
    pub output: Linear,

    /// Number of heads.
    heads: usize,

    /// Width of one head.
    head_dim: usize,
}

/// Every intermediate of one attention call over `[batch, time]`. One instance
/// stores the forward activations; a second, shared instance stores gradients.
pub struct AttentionTensors {
    /// Fused projections `[batch·time, 3·width]`.
    pub qkv: Tensor,

    /// Queries, head-major `[batch·heads·time, head_dim]`.
    pub queries: Tensor,

    /// Keys, head-major.
    pub keys: Tensor,

    /// Values, head-major.
    pub values: Tensor,

    /// Attention weights `[batch·heads·time, time]`.
    pub probabilities: Tensor,

    /// Weighted values, head-major `[batch·heads·time, head_dim]`.
    pub context_heads: Tensor,

    /// Weighted values, packed `[batch·time, width]`.
    pub context: Tensor,
}

impl AttentionTensors {
    /// Buffers for one sequence shape.
    pub fn new(gpu: &Gpu, shape: SequenceShape, width: usize, heads: usize) -> GpuResult<Self> {
        let rows = shape.rows();
        let head_rows = rows * heads;
        let head_dim = width / heads;

        Ok(Self {
            qkv: gpu.zeros(Shape::new(rows, 3 * width))?,
            queries: gpu.zeros(Shape::new(head_rows, head_dim))?,
            keys: gpu.zeros(Shape::new(head_rows, head_dim))?,
            values: gpu.zeros(Shape::new(head_rows, head_dim))?,
            probabilities: gpu.zeros(Shape::new(head_rows, shape.time))?,
            context_heads: gpu.zeros(Shape::new(head_rows, head_dim))?,
            context: gpu.zeros(Shape::new(rows, width))?,
        })
    }
}

/// Keys and values of all previous positions of one sequence, for one layer.
pub struct LayerCache {
    /// Keys `[heads·capacity, head_dim]`.
    keys: Tensor,

    /// Values `[heads·capacity, head_dim]`.
    values: Tensor,
}

impl LayerCache {
    /// An empty cache of `capacity` positions.
    pub fn new(gpu: &Gpu, capacity: usize, width: usize, heads: usize) -> GpuResult<Self> {
        let shape = Shape::new(heads * capacity, width / heads);
        Ok(Self { keys: gpu.zeros(shape)?, values: gpu.zeros(shape)? })
    }
}

/// Scratch for one incremental attention call over `new_positions` tokens.
pub struct CachedAttentionTensors {
    /// Fused projections `[new, 3·width]`.
    qkv: Tensor,

    /// Queries `[heads·new, head_dim]`.
    queries: Tensor,

    /// Attention weights over the whole cache, `[heads·new, capacity]`.
    probabilities: Tensor,

    /// Weighted values, head-major.
    context_heads: Tensor,

    /// Weighted values, packed `[new, width]`.
    context: Tensor,
}

impl CachedAttentionTensors {
    /// Scratch for `new_positions` tokens attending into a cache of `capacity`.
    pub fn new(gpu: &Gpu, new_positions: usize, capacity: usize, width: usize, heads: usize) -> GpuResult<Self> {
        let head_rows = new_positions * heads;

        Ok(Self {
            qkv: gpu.zeros(Shape::new(new_positions, 3 * width))?,
            queries: gpu.zeros(Shape::new(head_rows, width / heads))?,
            probabilities: gpu.zeros(Shape::new(head_rows, capacity))?,
            context_heads: gpu.zeros(Shape::new(head_rows, width / heads))?,
            context: gpu.zeros(Shape::new(new_positions, width))?,
        })
    }
}

impl CausalSelfAttention {
    /// A layer of `heads` heads over `width` features.
    pub fn new(
        gpu: &Gpu,
        name: &str,
        width: usize,
        heads: usize,
        projection_std: f32,
        output_std: f32,
        random: &mut Random,
    ) -> GpuResult<Self> {
        Ok(Self {
            qkv: Linear::new(gpu, &format!("{name}.qkv"), width, 3 * width, projection_std, random)?,
            output: Linear::new(gpu, &format!("{name}.output"), width, width, output_std, random)?,
            heads,
            head_dim: width / heads,
        })
    }

    /// Residual-stream width.
    fn width(&self) -> usize {
        self.heads * self.head_dim
    }

    /// `1/√d_head`, the logit scale that keeps dot products at unit variance.
    fn score_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Layout of one projection of a dense `[batch, time]` sequence.
    fn layout(&self, shape: SequenceShape, packed_width: usize, column_offset: usize) -> HeadLayout {
        HeadLayout::dense(shape.batch, shape.time, self.heads, self.head_dim, packed_width, column_offset)
    }

    /// Batch strides for products over `batch·heads` independent `[time, ·]` matrices.
    fn per_head(&self, shape: SequenceShape, left: usize, right: usize, output: usize) -> BatchStrides {
        BatchStrides { count: shape.batch * self.heads, left, right, output }
    }

    /// Full-sequence forward pass, keeping every activation in `tensors`.
    pub fn forward(
        &self,
        gpu: &Gpu,
        input: &Tensor,
        shape: SequenceShape,
        tensors: &mut AttentionTensors,
        output: &mut Tensor,
    ) -> GpuResult {
        let (width, time, head_dim) = (self.width(), shape.time, self.head_dim);
        let head_block = time * head_dim;
        let score_block = time * time;

        self.qkv.forward(gpu, input, &mut tensors.qkv)?;

        gpu.split_heads(&tensors.qkv, &mut tensors.queries, &self.layout(shape, 3 * width, Projection::Query.column_offset(width)))?;
        gpu.split_heads(&tensors.qkv, &mut tensors.keys, &self.layout(shape, 3 * width, Projection::Key.column_offset(width)))?;
        gpu.split_heads(&tensors.qkv, &mut tensors.values, &self.layout(shape, 3 * width, Projection::Value.column_offset(width)))?;

        let scores = MatmulSpec::new(time, time, head_dim)
            .right_transposed()
            .batched(self.per_head(shape, head_block, head_block, score_block));
        gpu.matmul(&scores, &tensors.queries, &tensors.keys, &mut tensors.probabilities)?;
        gpu.causal_softmax(&mut tensors.probabilities, time, self.score_scale(), 0)?;

        let mix = MatmulSpec::new(time, head_dim, time).batched(self.per_head(shape, score_block, head_block, head_block));
        gpu.matmul(&mix, &tensors.probabilities, &tensors.values, &mut tensors.context_heads)?;
        gpu.merge_heads(&tensors.context_heads, &mut tensors.context, &self.layout(shape, width, 0))?;

        self.output.forward(gpu, &tensors.context, output)
    }

    /// Accumulates parameter gradients and writes `input_grad`.
    pub fn backward(
        &mut self,
        gpu: &Gpu,
        input: &Tensor,
        shape: SequenceShape,
        activations: &AttentionTensors,
        output_grad: &Tensor,
        grads: &mut AttentionTensors,
        input_grad: &mut Tensor,
    ) -> GpuResult {
        let (width, time, head_dim) = (self.width(), shape.time, self.head_dim);
        let head_block = time * head_dim;
        let score_block = time * time;
        let scale = self.score_scale();

        self.output
            .backward(gpu, &activations.context, output_grad, &mut grads.context)?;
        gpu.split_heads(&grads.context, &mut grads.context_heads, &self.layout(shape, width, 0))?;

        let probability_grad = MatmulSpec::new(time, time, head_dim)
            .right_transposed()
            .batched(self.per_head(shape, head_block, head_block, score_block));
        gpu.matmul(&probability_grad, &grads.context_heads, &activations.values, &mut grads.probabilities)?;

        let value_grad = MatmulSpec::new(time, head_dim, time)
            .left_transposed()
            .batched(self.per_head(shape, score_block, head_block, head_block));
        gpu.matmul(&value_grad, &activations.probabilities, &grads.context_heads, &mut grads.values)?;

        gpu.softmax_backward(&activations.probabilities, &mut grads.probabilities)?;

        let query_grad = MatmulSpec::new(time, head_dim, time)
            .scaled(scale)
            .batched(self.per_head(shape, score_block, head_block, head_block));
        gpu.matmul(&query_grad, &grads.probabilities, &activations.keys, &mut grads.queries)?;

        let key_grad = MatmulSpec::new(time, head_dim, time)
            .left_transposed()
            .scaled(scale)
            .batched(self.per_head(shape, score_block, head_block, head_block));
        gpu.matmul(&key_grad, &grads.probabilities, &activations.queries, &mut grads.keys)?;

        gpu.merge_heads(&grads.queries, &mut grads.qkv, &self.layout(shape, 3 * width, Projection::Query.column_offset(width)))?;
        gpu.merge_heads(&grads.keys, &mut grads.qkv, &self.layout(shape, 3 * width, Projection::Key.column_offset(width)))?;
        gpu.merge_heads(&grads.values, &mut grads.qkv, &self.layout(shape, 3 * width, Projection::Value.column_offset(width)))?;

        self.qkv.backward(gpu, input, &grads.qkv, input_grad)
    }

    /// Incremental forward pass: the `new` rows of `input` are appended to
    /// `cache` at `cached_positions` and attend to everything cached so far.
    ///
    /// Scores are computed against the cache's full capacity; the causal mask
    /// zeroes the not-yet-written tail, so every tensor has a static shape.
    pub fn forward_cached(
        &self,
        gpu: &Gpu,
        input: &Tensor,
        cache: &mut LayerCache,
        cached_positions: usize,
        tensors: &mut CachedAttentionTensors,
        output: &mut Tensor,
    ) -> GpuResult {
        let (width, head_dim) = (self.width(), self.head_dim);
        let new_positions = input.shape().rows;
        let capacity = cache.keys.shape().rows / self.heads;
        let single = SequenceShape { batch: 1, time: new_positions };

        self.qkv.forward(gpu, input, &mut tensors.qkv)?;
        gpu.split_heads(&tensors.qkv, &mut tensors.queries, &self.layout(single, 3 * width, Projection::Query.column_offset(width)))?;

        let key_layout = self
            .layout(single, 3 * width, Projection::Key.column_offset(width))
            .in_cache(capacity, cached_positions);
        let value_layout = self
            .layout(single, 3 * width, Projection::Value.column_offset(width))
            .in_cache(capacity, cached_positions);

        gpu.split_heads(&tensors.qkv, &mut cache.keys, &key_layout)?;
        gpu.split_heads(&tensors.qkv, &mut cache.values, &value_layout)?;

        let cache_block = capacity * head_dim;
        let query_block = new_positions * head_dim;
        let score_block = new_positions * capacity;

        let heads = BatchStrides { count: self.heads, left: query_block, right: cache_block, output: score_block };

        let scores = MatmulSpec::new(new_positions, capacity, head_dim)
            .right_transposed()
            .batched(heads);
        gpu.matmul(&scores, &tensors.queries, &cache.keys, &mut tensors.probabilities)?;
        gpu.causal_softmax(&mut tensors.probabilities, new_positions, self.score_scale(), cached_positions)?;

        let mix_strides = BatchStrides { count: self.heads, left: score_block, right: cache_block, output: query_block };
        let mix = MatmulSpec::new(new_positions, head_dim, capacity).batched(mix_strides);

        gpu.matmul(&mix, &tensors.probabilities, &cache.values, &mut tensors.context_heads)?;
        gpu.merge_heads(&tensors.context_heads, &mut tensors.context, &self.layout(single, width, 0))?;

        self.output.forward(gpu, &tensors.context, output)
    }
}

impl Parameters for CausalSelfAttention {
    fn visit_parameters(&mut self, visit: &mut dyn FnMut(&mut Parameter)) {
        self.qkv.visit_parameters(visit);
        self.output.visit_parameters(visit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_offsets_split_a_fused_row_into_thirds() {
        assert_eq!(Projection::Query.column_offset(256), 0);
        assert_eq!(Projection::Key.column_offset(256), 256);
        assert_eq!(Projection::Value.column_offset(256), 512);
    }

    #[test]
    fn projection_offsets_scale_with_the_width() {
        assert_eq!(Projection::Key.column_offset(32), 32);
        assert_eq!(Projection::Value.column_offset(32), 64);
    }
}
