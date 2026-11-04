//! The vocoder's network: from standardized mel frames to a spectrum head.
//!
//! A convolution over time (`taps` neighboring frames, computed as an unfold
//! followed by a matrix product) lifts each frame's mel bands to `width`
//! numbers. A stack of ConvNeXt [`Block`]s refines them. A final normalization
//! and a linear head then predict, for every frame, the log-magnitude and the
//! phase of each of `fft_size / 2 + 1` frequencies.

use super::block::{Block, BlockActivations, BlockScratch};
use super::config::VocoderConfig;
use crate::layers::linear::Linear;
use crate::layers::norm::RmsNorm;
use crate::parameter::{INIT_STD, Parameter, Parameters};
use crate::random::Random;
use staleynet_cuda::{Convolution, Gpu, GpuResult, Shape, Tensor};

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

    /// What each block keeps for its backward pass.
    blocks: Vec<BlockActivations>,

    /// After the final normalization, `[rows, width]`.
    final_normed: Tensor,

    /// One over the root mean square of each row of the last stream.
    final_inv_rms: Tensor,

    /// The head's output, `[rows, 2 · bins]`: log-magnitudes, then phases.
    pub head: Tensor,

    /// A block's scaled contribution before it is added to the stream, `[rows, width]`.
    scaled: Tensor,

    /// Gradient of the residual stream at the current depth, `[rows, width]`.
    stream_grad: Tensor,

    /// Gradient of the final normalization's output, `[rows, width]`.
    final_normed_grad: Tensor,

    /// Gradient of the unfolded input, which nothing uses but the layer writes.
    unfolded_grad: Tensor,

    /// Gradient buffers shared by the blocks.
    block_scratch: BlockScratch,
}

impl Activations {
    /// Buffers for `sequences` sequences of `time` frames each.
    pub fn new(gpu: &Gpu, config: &VocoderConfig, sequences: usize, time: usize) -> GpuResult<Self> {
        let rows = sequences * time;
        let stream = Shape::new(rows, config.width);
        let unfolded = Shape::new(rows, config.taps * config.mel_bands);

        let streams = (0..=config.layers)
            .map(|_| gpu.zeros(stream))
            .collect::<GpuResult<Vec<_>>>()?;
        let blocks = (0..config.layers)
            .map(|_| BlockActivations::new(gpu, config, rows))
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self {
            input_shape: Convolution { rows, time, channels: config.mel_bands, kernel: config.taps },
            stream_shape: Convolution { rows, time, channels: config.width, kernel: config.taps },
            unfolded: gpu.zeros(unfolded)?,
            streams,
            blocks,
            final_normed: gpu.zeros(stream)?,
            final_inv_rms: gpu.zeros(Shape::vector(rows))?,
            head: gpu.zeros(Shape::new(rows, 2 * config.bins()))?,
            scaled: gpu.zeros(stream)?,
            stream_grad: gpu.zeros(stream)?,
            final_normed_grad: gpu.zeros(stream)?,
            unfolded_grad: gpu.zeros(unfolded)?,
            block_scratch: BlockScratch::new(gpu, config, rows)?,
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
        let blocks = (0..config.layers)
            .map(|layer| Block::new(gpu, &format!("blocks.{layer}"), config, random))
            .collect::<GpuResult<Vec<_>>>()?;

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
        self.input
            .forward(gpu, &activations.unfolded, &mut activations.streams[0])?;

        for (index, block) in self.blocks.iter().enumerate() {
            let (before, after) = activations.streams.split_at_mut(index + 1);
            let input = &before[index];
            let output = &mut after[0];
            let saved = &mut activations.blocks[index];

            block.forward(gpu, input, activations.stream_shape, saved, &mut activations.scaled, output)?;
        }

        let last = activations.streams.last().expect("at least one stream");
        self.final_norm
            .forward(gpu, last, &mut activations.final_normed, &mut activations.final_inv_rms)?;
        self.head
            .forward(gpu, &activations.final_normed, &mut activations.head)
    }

    /// Accumulates every parameter's gradient, given the gradient of the head
    /// output. The input mel needs no gradient.
    pub fn backward(&mut self, gpu: &Gpu, head_grad: &Tensor, activations: &mut Activations) -> GpuResult {
        let last = activations.streams.last().expect("at least one stream");
        self.head
            .backward(gpu, &activations.final_normed, head_grad, &mut activations.final_normed_grad)?;
        self.final_norm
            .backward(gpu, &activations.final_normed_grad, last, &activations.final_inv_rms, &mut activations.stream_grad)?;

        for (index, block) in self.blocks.iter_mut().enumerate().rev() {
            let input = &activations.streams[index];
            let saved = &activations.blocks[index];

            block.backward(gpu, input, activations.stream_shape, saved, &mut activations.block_scratch, &mut activations.stream_grad)?;
        }

        self.input
            .backward(gpu, &activations.unfolded, &activations.stream_grad, &mut activations.unfolded_grad)
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
