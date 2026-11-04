//! The vocoder's network: from standardized mel frames to a spectrum head.
//!
//! A convolution over time (`taps` neighboring frames, computed as an unfold
//! followed by a matrix product) lifts each frame's mel bands to `width`
//! numbers. A stack of ConvNeXt blocks (Liu et al., 2022,
//! <https://arxiv.org/abs/2201.03545>) then refines them. Each block:
//!
//! 1. mixes neighboring frames, every channel with its own filter (a
//!    depthwise convolution);
//! 2. normalizes each frame (RMSNorm);
//! 3. expands it to `expansion · width` numbers, applies GELU, and projects
//!    it back to `width`;
//! 4. scales the result per channel and adds it to its input, a residual
//!    connection whose scale starts small so that a deep stack trains stably.
//!
//! A final normalization and a linear head predict, for every frame, the
//! log-magnitude and the phase of each of `fft_size / 2 + 1` frequencies.

use crate::config::VocoderConfig;
use staleynet_cuda::{Convolution, Gpu, GpuResult, Shape, Tensor};
use staleynet_model::layers::linear::Linear;
use staleynet_model::layers::norm::RmsNorm;
use staleynet_model::parameter::{Decay, INIT_STD, Init, Parameter, Parameters};
use staleynet_model::random::Random;

/// One ConvNeXt block.
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

    /// Per-channel scale of the block's contribution, initialized to `1 / layers`.
    scale: Parameter,
}

impl Block {
    fn new(gpu: &Gpu, name: &str, config: &VocoderConfig, random: &mut Random) -> GpuResult<Self> {
        let (width, hidden) = (config.width, config.expansion * config.width);
        let mut scale = Parameter::new(gpu, format!("{name}.scale"), Shape::vector(width), Init::Ones, Decay::Exempt, random)?;
        gpu.scale(&mut scale.value, 1.0 / config.layers as f32)?;

        Ok(Self {
            depthwise_weight: Parameter::new(
                gpu,
                format!("{name}.depthwise.weight"),
                Shape::new(config.taps, width),
                Init::Normal(INIT_STD),
                Decay::Decayed,
                random,
            )?,
            depthwise_bias: Parameter::new(
                gpu,
                format!("{name}.depthwise.bias"),
                Shape::vector(width),
                Init::Zeros,
                Decay::Exempt,
                random,
            )?,
            norm: RmsNorm::new(gpu, &format!("{name}.norm"), width, random)?,
            expand: Linear::new(gpu, &format!("{name}.expand"), width, hidden, INIT_STD, random)?,
            project: Linear::new(gpu, &format!("{name}.project"), hidden, width, INIT_STD, random)?,
            scale,
        })
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

/// What one block's forward pass keeps for its backward pass.
struct BlockActivations {
    /// After the depthwise convolution, `[rows, width]`.
    mixed: Tensor,

    /// After normalization, `[rows, width]`.
    normed: Tensor,

    /// One over the root mean square of each row of `mixed`.
    inv_rms: Tensor,

    /// Before GELU, `[rows, expansion · width]`.
    pre_activation: Tensor,

    /// After GELU, `[rows, expansion · width]`.
    hidden: Tensor,

    /// The block's output before scaling, `[rows, width]`.
    projected: Tensor,
}

/// Every tensor of a forward and backward pass for a fixed number of rows.
pub struct Activations {
    /// The convolution shape over the mel input.
    input_shape: Convolution,

    /// The convolution shape over the backbone's width.
    stream_shape: Convolution,

    /// Each frame's neighborhood of mel frames, `[rows, taps · mel_bands]`.
    unfolded: Tensor,

    /// The residual stream before each block and after the last, `[rows, width]`.
    streams: Vec<Tensor>,

    /// Per-block activations.
    blocks: Vec<BlockActivations>,

    /// After the final normalization, `[rows, width]`.
    final_normed: Tensor,

    /// One over the root mean square of each row of the last stream.
    final_inv_rms: Tensor,

    /// The head's output, `[rows, 2 · bins]`: log-magnitudes, then phases.
    pub head: Tensor,

    /// Scratch of shape `[rows, width]`.
    scratch: Tensor,

    /// Gradient of the residual stream at the current depth, `[rows, width]`.
    stream_grad: Tensor,

    /// Gradient of a `[rows, width]` intermediate inside a block.
    width_grad: Tensor,

    /// Gradient of the depthwise convolution's output, `[rows, width]`.
    mixed_grad: Tensor,

    /// Gradient after GELU, `[rows, expansion · width]`.
    hidden_grad: Tensor,

    /// Gradient before GELU, `[rows, expansion · width]`.
    pre_activation_grad: Tensor,

    /// Gradient of the unfolded input, which nothing uses but the layer writes.
    unfolded_grad: Tensor,

    /// One block's layer-scale gradient before it is added in, `[width]`.
    scale_grad: Tensor,

    /// One block's depthwise-filter gradient before it is added in, `[taps, width]`.
    depthwise_grad: Tensor,
}

impl Activations {
    /// Buffers for `sequences` sequences of `time` frames each.
    pub fn new(gpu: &Gpu, config: &VocoderConfig, sequences: usize, time: usize) -> GpuResult<Self> {
        let rows = sequences * time;
        let (width, hidden) = (config.width, config.expansion * config.width);
        let stream = Shape::new(rows, width);
        let wide = Shape::new(rows, hidden);

        let mut streams = Vec::with_capacity(config.layers + 1);
        let mut blocks = Vec::with_capacity(config.layers);
        for _ in 0..config.layers {
            streams.push(gpu.zeros(stream)?);
            blocks.push(BlockActivations {
                mixed: gpu.zeros(stream)?,
                normed: gpu.zeros(stream)?,
                inv_rms: gpu.zeros(Shape::vector(rows))?,
                pre_activation: gpu.zeros(wide)?,
                hidden: gpu.zeros(wide)?,
                projected: gpu.zeros(stream)?,
            });
        }
        streams.push(gpu.zeros(stream)?);

        let unfolded = Shape::new(rows, config.taps * config.mel_bands);
        Ok(Self {
            input_shape: Convolution { rows, time, channels: config.mel_bands, kernel: config.taps },
            stream_shape: Convolution { rows, time, channels: width, kernel: config.taps },
            unfolded: gpu.zeros(unfolded)?,
            streams,
            blocks,
            final_normed: gpu.zeros(stream)?,
            final_inv_rms: gpu.zeros(Shape::vector(rows))?,
            head: gpu.zeros(Shape::new(rows, 2 * config.bins()))?,
            scratch: gpu.zeros(stream)?,
            stream_grad: gpu.zeros(stream)?,
            width_grad: gpu.zeros(stream)?,
            mixed_grad: gpu.zeros(stream)?,
            hidden_grad: gpu.zeros(wide)?,
            pre_activation_grad: gpu.zeros(wide)?,
            unfolded_grad: gpu.zeros(unfolded)?,
            scale_grad: gpu.zeros(Shape::vector(width))?,
            depthwise_grad: gpu.zeros(Shape::new(config.taps, width))?,
        })
    }

    /// Rows these buffers hold.
    pub fn rows(&self) -> usize {
        self.input_shape.rows
    }
}

/// The input convolution, the blocks, and the spectrum head.
pub struct Network {
    /// `taps · mel_bands → width`, applied to the unfolded input.
    input: Linear,

    /// The ConvNeXt blocks.
    blocks: Vec<Block>,

    /// Normalization before the head.
    final_norm: RmsNorm,

    /// `width → 2 · bins`: log-magnitudes, then phases.
    head: Linear,
}

impl Network {
    /// A freshly initialized network for `config`.
    pub fn new(gpu: &Gpu, config: &VocoderConfig, random: &mut Random) -> GpuResult<Self> {
        config.validate();
        let input = Linear::new(gpu, "input", config.taps * config.mel_bands, config.width, INIT_STD, random)?;
        let blocks =
            (0..config.layers).map(|layer| Block::new(gpu, &format!("blocks.{layer}"), config, random)).collect::<GpuResult<Vec<_>>>()?;

        Ok(Self {
            input,
            blocks,
            final_norm: RmsNorm::new(gpu, "final_norm", config.width, random)?,
            head: Linear::new(gpu, "head", config.width, 2 * config.bins(), INIT_STD, random)?,
        })
    }

    /// Runs the network on standardized mel frames (`[rows, mel_bands]`),
    /// leaving the spectrum head in `activations.head`.
    pub fn forward(&self, gpu: &Gpu, mel: &Tensor, activations: &mut Activations) -> GpuResult {
        gpu.unfold(mel, &mut activations.unfolded, activations.input_shape)?;
        self.input.forward(gpu, &activations.unfolded, &mut activations.streams[0])?;

        for (index, block) in self.blocks.iter().enumerate() {
            let (before, after) = activations.streams.split_at_mut(index + 1);
            let saved = &mut activations.blocks[index];
            let input = &before[index];
            let output = &mut after[0];

            gpu.depthwise_convolution(
                input,
                &block.depthwise_weight.value,
                &block.depthwise_bias.value,
                &mut saved.mixed,
                activations.stream_shape,
            )?;
            block.norm.forward(gpu, &saved.mixed, &mut saved.normed, &mut saved.inv_rms)?;
            block.expand.forward(gpu, &saved.normed, &mut saved.pre_activation)?;
            gpu.gelu_forward(&saved.pre_activation, &mut saved.hidden)?;
            block.project.forward(gpu, &saved.hidden, &mut saved.projected)?;

            // output = input + scale ⊙ projected
            gpu.scale_columns(&saved.projected, &block.scale.value, &mut activations.scratch)?;
            gpu.fill_zero(output)?;
            gpu.add_scaled(1.0, input, output)?;
            gpu.add_scaled(1.0, &activations.scratch, output)?;
        }

        let last = activations.streams.last().expect("at least one stream");
        self.final_norm.forward(gpu, last, &mut activations.final_normed, &mut activations.final_inv_rms)?;
        self.head.forward(gpu, &activations.final_normed, &mut activations.head)
    }

    /// Accumulates every parameter's gradient, given the gradient of the head
    /// output. The input mel needs no gradient.
    pub fn backward(&mut self, gpu: &Gpu, head_grad: &Tensor, activations: &mut Activations) -> GpuResult {
        let a = activations;
        self.head.backward(gpu, &a.final_normed, head_grad, &mut a.width_grad)?;
        let last = a.streams.last().expect("at least one stream");
        self.final_norm.backward(gpu, &a.width_grad, last, &a.final_inv_rms, &mut a.stream_grad)?;

        for (index, block) in self.blocks.iter_mut().enumerate().rev() {
            let saved = &a.blocks[index];
            let input = &a.streams[index];

            // Through output = input + scale ⊙ projected: the stream gradient
            // passes to the input unchanged, and is scaled into the block.
            gpu.column_products(&a.stream_grad, &saved.projected, &mut a.scale_grad)?;
            gpu.add_scaled(1.0, &a.scale_grad, &mut block.scale.gradient)?;
            gpu.scale_columns(&a.stream_grad, &block.scale.value, &mut a.width_grad)?;

            block.project.backward(gpu, &saved.hidden, &a.width_grad, &mut a.hidden_grad)?;
            gpu.gelu_backward(&saved.pre_activation, &a.hidden_grad, &mut a.pre_activation_grad)?;
            block.expand.backward(gpu, &saved.normed, &a.pre_activation_grad, &mut a.width_grad)?;
            block.norm.backward(gpu, &a.width_grad, &saved.mixed, &saved.inv_rms, &mut a.mixed_grad)?;

            gpu.depthwise_convolution_weight_grad(&a.mixed_grad, input, &mut a.depthwise_grad, a.stream_shape)?;
            gpu.add_scaled(1.0, &a.depthwise_grad, &mut block.depthwise_weight.gradient)?;
            gpu.accumulate_column_sums(&a.mixed_grad, &mut block.depthwise_bias.gradient)?;
            gpu.depthwise_convolution_input_grad(&a.mixed_grad, &block.depthwise_weight.value, &mut a.scratch, a.stream_shape)?;
            gpu.add_scaled(1.0, &a.scratch, &mut a.stream_grad)?;
        }

        self.input.backward(gpu, &a.unfolded, &a.stream_grad, &mut a.unfolded_grad)
    }
}

impl Parameters for Network {
    fn visit_parameters(&mut self, visit: &mut dyn FnMut(&mut Parameter)) {
        self.input.visit_parameters(visit);
        for block in &mut self.blocks {
            block.visit_parameters(visit);
        }

        self.final_norm.visit_parameters(visit);
        self.head.visit_parameters(visit);
    }
}
