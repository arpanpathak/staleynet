//! One pre-norm Transformer block:
//! `h = x + Attention(Norm(x))`, `y = h + FFN(Norm(h))`.
//! Pre-norm keeps gradients well-behaved without warm-up tricks
//! (Xiong et al., 2020, <https://arxiv.org/abs/2002.04745>).

use super::SequenceShape;
use super::attention::{AttentionTensors, CachedAttentionTensors, CausalSelfAttention, LayerCache};
use super::feed_forward::{FeedForward, FeedForwardTensors};
use super::norm::RmsNorm;
use crate::config::ModelConfig;
use crate::parameter::{INIT_STD, Parameter, Parameters};
use crate::random::Random;
use staleynet_cuda::{Gpu, GpuResult, Shape, Tensor};

/// Attention and feed-forward sublayers with their norms.
pub struct Block {
    /// Norm in front of attention.
    pub attention_norm: RmsNorm,

    /// Causal self-attention.
    pub attention: CausalSelfAttention,

    /// Norm in front of the feed-forward network.
    pub feed_forward_norm: RmsNorm,

    /// Position-wise feed-forward network.
    pub feed_forward: FeedForward,
}

/// Forward activations of one block for a training sequence shape.
pub struct BlockActivations {
    /// `Norm(x)`.
    attention_input: Tensor,

    /// `1/rms` of each row of `x`.
    attention_inv_rms: Tensor,

    /// Attention intermediates.
    attention: AttentionTensors,

    /// `h = x + Attention(Norm(x))`.
    residual_mid: Tensor,

    /// `Norm(h)`.
    feed_forward_input: Tensor,

    /// `1/rms` of each row of `h`.
    feed_forward_inv_rms: Tensor,

    /// Feed-forward intermediates.
    feed_forward: FeedForwardTensors,
}

impl BlockActivations {
    /// Buffers for one block over `shape`.
    pub fn new(gpu: &Gpu, config: &ModelConfig, shape: SequenceShape) -> GpuResult<Self> {
        let rows = shape.rows();
        let stream = Shape::new(rows, config.width);

        Ok(Self {
            attention_input: gpu.zeros(stream)?,
            attention_inv_rms: gpu.zeros(Shape::vector(rows))?,
            attention: AttentionTensors::new(gpu, shape, config.width, config.heads)?,
            residual_mid: gpu.zeros(stream)?,
            feed_forward_input: gpu.zeros(stream)?,
            feed_forward_inv_rms: gpu.zeros(Shape::vector(rows))?,
            feed_forward: FeedForwardTensors::new(gpu, rows, config.feed_forward_width())?,
        })
    }
}

/// Gradient scratch shared by all blocks (they run backward one at a time).
pub struct BlockGradients {
    /// Gradient with respect to a norm's output.
    normed: Tensor,

    /// Gradient with respect to `h`.
    residual_mid: Tensor,

    /// Attention intermediate gradients.
    attention: AttentionTensors,

    /// Feed-forward intermediate gradients.
    feed_forward: FeedForwardTensors,
}

impl BlockGradients {
    /// Scratch for one block over `shape`.
    pub fn new(gpu: &Gpu, config: &ModelConfig, shape: SequenceShape) -> GpuResult<Self> {
        let stream = Shape::new(shape.rows(), config.width);

        Ok(Self {
            normed: gpu.zeros(stream)?,
            residual_mid: gpu.zeros(stream)?,
            attention: AttentionTensors::new(gpu, shape, config.width, config.heads)?,
            feed_forward: FeedForwardTensors::new(gpu, shape.rows(), config.feed_forward_width())?,
        })
    }
}

/// Scratch for one block during incremental decoding of `new_positions` tokens.
pub struct CachedBlockTensors {
    /// Norm output.
    normed: Tensor,

    /// `1/rms` per row.
    inv_rms: Tensor,

    /// Attention scratch.
    attention: CachedAttentionTensors,

    /// `h`.
    residual_mid: Tensor,

    /// Feed-forward scratch.
    feed_forward: FeedForwardTensors,
}

impl CachedBlockTensors {
    /// Scratch for `new_positions` rows attending into `capacity` cached positions.
    pub fn new(gpu: &Gpu, config: &ModelConfig, new_positions: usize, capacity: usize) -> GpuResult<Self> {
        let stream = Shape::new(new_positions, config.width);

        Ok(Self {
            normed: gpu.zeros(stream)?,
            inv_rms: gpu.zeros(Shape::vector(new_positions))?,
            attention: CachedAttentionTensors::new(gpu, new_positions, capacity, config.width, config.heads)?,
            residual_mid: gpu.zeros(stream)?,
            feed_forward: FeedForwardTensors::new(gpu, new_positions, config.feed_forward_width())?,
        })
    }
}

impl Block {
    /// Block `index` of a model with `config`. Residual projections start
    /// smaller by `1/√(2·layers)` so the residual stream's variance does not
    /// grow with depth (GPT-2, Radford et al., 2019, §2.3).
    pub fn new(gpu: &Gpu, config: &ModelConfig, index: usize, random: &mut Random) -> GpuResult<Self> {
        let name = format!("blocks.{index}");
        let residual_std = INIT_STD / (2.0 * config.layers as f32).sqrt();
        let width = config.width;

        Ok(Self {
            attention_norm: RmsNorm::new(gpu, &format!("{name}.attention_norm"), width, random)?,
            attention: CausalSelfAttention::new(gpu, &format!("{name}.attention"), width, config.heads, INIT_STD, residual_std, random)?,
            feed_forward_norm: RmsNorm::new(gpu, &format!("{name}.feed_forward_norm"), width, random)?,
            feed_forward: FeedForward::new(
                gpu,
                &format!("{name}.feed_forward"),
                width,
                config.feed_forward_width(),
                INIT_STD,
                residual_std,
                random,
            )?,
        })
    }

    /// `output = block(input)` over a full sequence, keeping activations.
    pub fn forward(
        &self,
        gpu: &Gpu,
        input: &Tensor,
        shape: SequenceShape,
        activations: &mut BlockActivations,
        output: &mut Tensor,
    ) -> GpuResult {
        let BlockActivations {
            attention_input,
            attention_inv_rms,
            attention,
            residual_mid,
            feed_forward_input,
            feed_forward_inv_rms,
            feed_forward,
        } = activations;

        self.attention_norm.forward(gpu, input, attention_input, attention_inv_rms)?;
        self.attention.forward(gpu, attention_input, shape, attention, residual_mid)?;
        gpu.add_scaled(1.0, input, residual_mid)?;

        self.feed_forward_norm.forward(gpu, residual_mid, feed_forward_input, feed_forward_inv_rms)?;
        self.feed_forward.forward(gpu, feed_forward_input, feed_forward, output)?;
        gpu.add_scaled(1.0, residual_mid, output)
    }

    /// Accumulates parameter gradients and writes `input_grad`.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &mut self,
        gpu: &Gpu,
        input: &Tensor,
        shape: SequenceShape,
        activations: &BlockActivations,
        output_grad: &Tensor,
        scratch: &mut BlockGradients,
        input_grad: &mut Tensor,
    ) -> GpuResult {
        let BlockGradients { normed, residual_mid, attention, feed_forward } = scratch;

        self.feed_forward.backward(gpu, &activations.feed_forward_input, &activations.feed_forward, output_grad, feed_forward, normed)?;
        self.feed_forward_norm.backward(gpu, normed, &activations.residual_mid, &activations.feed_forward_inv_rms, residual_mid)?;
        gpu.add_scaled(1.0, output_grad, residual_mid)?;

        self.attention.backward(gpu, &activations.attention_input, shape, &activations.attention, residual_mid, attention, normed)?;
        self.attention_norm.backward(gpu, normed, input, &activations.attention_inv_rms, input_grad)?;
        gpu.add_scaled(1.0, residual_mid, input_grad)
    }

    /// Incremental forward pass that appends to `cache`.
    pub fn forward_cached(
        &self,
        gpu: &Gpu,
        input: &Tensor,
        cache: &mut LayerCache,
        cached_positions: usize,
        scratch: &mut CachedBlockTensors,
        output: &mut Tensor,
    ) -> GpuResult {
        let CachedBlockTensors { normed, inv_rms, attention, residual_mid, feed_forward } = scratch;

        self.attention_norm.forward(gpu, input, normed, inv_rms)?;
        self.attention.forward_cached(gpu, normed, cache, cached_positions, attention, residual_mid)?;
        gpu.add_scaled(1.0, input, residual_mid)?;

        self.feed_forward_norm.forward(gpu, residual_mid, normed, inv_rms)?;
        self.feed_forward.forward(gpu, normed, feed_forward, output)?;
        gpu.add_scaled(1.0, residual_mid, output)
    }
}

impl Parameters for Block {
    fn visit_parameters(&mut self, visit: &mut dyn FnMut(&mut Parameter)) {
        self.attention_norm.visit_parameters(visit);
        self.attention.visit_parameters(visit);
        self.feed_forward_norm.visit_parameters(visit);
        self.feed_forward.visit_parameters(visit);
    }
}
