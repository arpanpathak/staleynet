//! One ConvNeXt block of the vocoder's backbone (Liu et al., 2022,
//! <https://arxiv.org/abs/2201.03545>), with its hand-derived gradient.
//!
//! A block takes the residual stream `x` (`[rows, width]`) and returns
//! `x + γ ⊙ project(GELU(expand(norm(depthwise(x)))))`:
//!
//! 1. a depthwise convolution mixes neighboring frames, every channel with its
//!    own filter;
//! 2. RMSNorm normalizes each frame;
//! 3. a feed-forward network expands each frame to `expansion · width`
//!    numbers, applies GELU, and projects back to `width`;
//! 4. the per-channel scale `γ` ("layer scale", Touvron et al., 2021,
//!    <https://arxiv.org/abs/2103.17239>) weighs the result before it is added
//!    to the input. It starts at `1 / layers`, so that a deep stack begins
//!    close to the identity and trains stably.

use super::config::VocoderConfig;
use crate::layers::linear::Linear;
use crate::layers::norm::RmsNorm;
use crate::parameter::{Decay, INIT_STD, Init, Parameter, Parameters};
use crate::random::Random;
use staleynet_cuda::{Convolution, Gpu, GpuResult, Shape, Tensor};

/// One ConvNeXt block's parameters.
pub struct Block {
    /// Depthwise filter, `[taps, width]`.
    depthwise_weight: Parameter,

    /// Depthwise bias, `[width]`.
    depthwise_bias: Parameter,

    /// Normalization before the feed-forward network.
    norm: RmsNorm,

    /// `width → expansion · width`.
    expand: Linear,

    /// `expansion · width → width`.
    project: Linear,

    /// Per-channel scale `γ` of the block's contribution, `[width]`.
    scale: Parameter,
}

/// What one block's forward pass keeps for its backward pass.
pub struct BlockActivations {
    /// After the depthwise convolution, `[rows, width]`.
    mixed: Tensor,

    /// After normalization, `[rows, width]`.
    normed: Tensor,

    /// One over the root mean square of each row of `mixed`, `[rows]`.
    inv_rms: Tensor,

    /// Before GELU, `[rows, expansion · width]`.
    pre_activation: Tensor,

    /// After GELU, `[rows, expansion · width]`.
    hidden: Tensor,

    /// The block's output before scaling, `[rows, width]`.
    projected: Tensor,
}

impl BlockActivations {
    /// Buffers for `rows` frames.
    pub fn new(gpu: &Gpu, config: &VocoderConfig, rows: usize) -> GpuResult<Self> {
        let stream = Shape::new(rows, config.width);
        let wide = Shape::new(rows, config.hidden_width());

        Ok(Self {
            mixed: gpu.zeros(stream)?,
            normed: gpu.zeros(stream)?,
            inv_rms: gpu.zeros(Shape::vector(rows))?,
            pre_activation: gpu.zeros(wide)?,
            hidden: gpu.zeros(wide)?,
            projected: gpu.zeros(stream)?,
        })
    }
}

/// Gradient buffers that every block reuses in turn during the backward pass.
pub struct BlockScratch {
    /// Gradient of a `[rows, width]` intermediate inside the block.
    width_grad: Tensor,

    /// Gradient of the depthwise convolution's output, `[rows, width]`.
    mixed_grad: Tensor,

    /// Gradient after GELU, `[rows, expansion · width]`.
    hidden_grad: Tensor,

    /// Gradient before GELU, `[rows, expansion · width]`.
    pre_activation_grad: Tensor,

    /// The gradient reaching the block's input through its convolution, `[rows, width]`.
    input_grad: Tensor,

    /// One block's layer-scale gradient before it is added in, `[width]`.
    scale_grad: Tensor,

    /// One block's depthwise-filter gradient before it is added in, `[taps, width]`.
    depthwise_grad: Tensor,
}

impl BlockScratch {
    /// Buffers for `rows` frames.
    pub fn new(gpu: &Gpu, config: &VocoderConfig, rows: usize) -> GpuResult<Self> {
        let stream = Shape::new(rows, config.width);
        let wide = Shape::new(rows, config.hidden_width());

        Ok(Self {
            width_grad: gpu.zeros(stream)?,
            mixed_grad: gpu.zeros(stream)?,
            hidden_grad: gpu.zeros(wide)?,
            pre_activation_grad: gpu.zeros(wide)?,
            input_grad: gpu.zeros(stream)?,
            scale_grad: gpu.zeros(Shape::vector(config.width))?,
            depthwise_grad: gpu.zeros(Shape::new(config.taps, config.width))?,
        })
    }
}

impl Block {
    /// A freshly initialized block whose parameters are named `name.*`.
    pub fn new(gpu: &Gpu, name: &str, config: &VocoderConfig, random: &mut Random) -> GpuResult<Self> {
        let width = config.width;
        let hidden = config.hidden_width();

        let depthwise_weight = Parameter::new(
            gpu,
            format!("{name}.depthwise.weight"),
            Shape::new(config.taps, width),
            Init::Normal(INIT_STD),
            Decay::Decayed,
            random,
        )?;
        let depthwise_bias =
            Parameter::new(gpu, format!("{name}.depthwise.bias"), Shape::vector(width), Init::Zeros, Decay::Exempt, random)?;

        let mut scale = Parameter::new(gpu, format!("{name}.scale"), Shape::vector(width), Init::Ones, Decay::Exempt, random)?;
        gpu.scale(&mut scale.value, 1.0 / config.layers as f32)?;

        Ok(Self {
            depthwise_weight,
            depthwise_bias,
            norm: RmsNorm::new(gpu, &format!("{name}.norm"), width, random)?,
            expand: Linear::new(gpu, &format!("{name}.expand"), width, hidden, INIT_STD, random)?,
            project: Linear::new(gpu, &format!("{name}.project"), hidden, width, INIT_STD, random)?,
            scale,
        })
    }

    /// `output = input + γ ⊙ project(GELU(expand(norm(depthwise(input)))))`,
    /// keeping in `saved` what the backward pass needs. `scaled` is scratch of
    /// the stream's shape.
    pub fn forward(
        &self,
        gpu: &Gpu,
        input: &Tensor,
        shape: Convolution,
        saved: &mut BlockActivations,
        scaled: &mut Tensor,
        output: &mut Tensor,
    ) -> GpuResult {
        gpu.depthwise_convolution(input, &self.depthwise_weight.value, &self.depthwise_bias.value, &mut saved.mixed, shape)?;
        self.norm
            .forward(gpu, &saved.mixed, &mut saved.normed, &mut saved.inv_rms)?;

        self.expand
            .forward(gpu, &saved.normed, &mut saved.pre_activation)?;
        gpu.gelu_forward(&saved.pre_activation, &mut saved.hidden)?;
        self.project
            .forward(gpu, &saved.hidden, &mut saved.projected)?;

        gpu.scale_columns(&saved.projected, &self.scale.value, scaled)?;
        gpu.fill_zero(output)?;
        gpu.add_scaled(1.0, input, output)?;
        gpu.add_scaled(1.0, scaled, output)
    }

    /// Accumulates this block's parameter gradients and turns `stream_grad`,
    /// the gradient of the block's output, into the gradient of its input.
    ///
    /// The residual connection passes `stream_grad` to the input unchanged;
    /// the block's own path adds the gradient that flows back through the
    /// scale, the feed-forward network, the norm and the convolution.
    pub fn backward(
        &mut self,
        gpu: &Gpu,
        input: &Tensor,
        shape: Convolution,
        saved: &BlockActivations,
        scratch: &mut BlockScratch,
        stream_grad: &mut Tensor,
    ) -> GpuResult {
        gpu.column_products(stream_grad, &saved.projected, &mut scratch.scale_grad)?;
        gpu.add_scaled(1.0, &scratch.scale_grad, &mut self.scale.gradient)?;
        gpu.scale_columns(stream_grad, &self.scale.value, &mut scratch.width_grad)?;

        self.project
            .backward(gpu, &saved.hidden, &scratch.width_grad, &mut scratch.hidden_grad)?;
        gpu.gelu_backward(&saved.pre_activation, &scratch.hidden_grad, &mut scratch.pre_activation_grad)?;
        self.expand
            .backward(gpu, &saved.normed, &scratch.pre_activation_grad, &mut scratch.width_grad)?;

        self.norm
            .backward(gpu, &scratch.width_grad, &saved.mixed, &saved.inv_rms, &mut scratch.mixed_grad)?;

        gpu.depthwise_convolution_weight_grad(&scratch.mixed_grad, input, &mut scratch.depthwise_grad, shape)?;
        gpu.add_scaled(1.0, &scratch.depthwise_grad, &mut self.depthwise_weight.gradient)?;
        gpu.accumulate_column_sums(&scratch.mixed_grad, &mut self.depthwise_bias.gradient)?;

        gpu.depthwise_convolution_input_grad(&scratch.mixed_grad, &self.depthwise_weight.value, &mut scratch.input_grad, shape)?;
        gpu.add_scaled(1.0, &scratch.input_grad, stream_grad)
    }
}

impl Parameters for Block {
    fn visit_parameters(&mut self, visit: &mut dyn FnMut(&mut Parameter)) {
        visit(&mut self.depthwise_weight);
        visit(&mut self.depthwise_bias);

        self.norm.visit_parameters(visit);
        self.expand.visit_parameters(visit);
        self.project.visit_parameters(visit);

        visit(&mut self.scale);
    }
}
