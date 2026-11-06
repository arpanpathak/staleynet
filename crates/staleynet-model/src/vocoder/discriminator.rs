//! HiFi-GAN's multi-period discriminator (Kong, Kim and Bae, 2020,
//! <https://arxiv.org/abs/2010.05646>, §2.2).
//!
//! A spectral loss measures how far a waveform's spectrogram is from its
//! target's, but not whether the waveform sounds real, and a vocoder trained
//! with it alone sounds slightly muffled. A discriminator is a second network,
//! trained to tell real recordings from synthesized ones; the vocoder is then
//! trained to fool it, which restores the fine detail that hearing notices.
//!
//! Each sub-discriminator splits the signal by one period `p` into `p`
//! interleaved sequences and runs a stack of strided convolutions over each,
//! so it judges structure that repeats every `p` samples, such as the
//! harmonics of a voice or a string. The periods are primes (2, 3, 5, 7, 11)
//! so that the sub-discriminators overlap as little as possible. Every layer's
//! output is kept as a *feature* for the feature-matching loss (§2.3).

use crate::layers::linear::Linear;
use crate::parameter::{Parameter, Parameters};
use crate::random::Random;
use serde::{Deserialize, Serialize};
use staleynet_cuda::{Gpu, GpuResult, PeriodSplit, Shape, StridedConvolution, Tensor};

/// Taps of every convolution except the last, HiFi-GAN's value.
const KERNEL: usize = 5;

/// Taps of the last convolution, which turns features into scores.
const SCORE_KERNEL: usize = 3;

/// Channels of the scores: one judgement per frame.
const SCORE_CHANNELS: usize = 1;

/// Variance gain of He initialization for a ReLU-like activation (He et al.,
/// 2015, <https://arxiv.org/abs/1502.01852>): weights are drawn with standard
/// deviation `sqrt(HE_GAIN / fan_in)`.
const HE_GAIN: f32 = 2.0;

/// One downsampling layer: its output channels and its stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerSize {
    /// Output channels.
    pub channels: usize,

    /// Input frames between consecutive output frames.
    pub stride: usize,
}

/// The sizes of the multi-period discriminator.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscriminatorConfig {
    /// The period of each sub-discriminator, in samples.
    pub periods: Vec<usize>,

    /// The layers of every sub-discriminator, before the scoring layer.
    pub layers: Vec<LayerSize>,
}

impl Default for DiscriminatorConfig {
    /// HiFi-GAN's sizes: periods 2, 3, 5, 7 and 11, and four layers of stride
    /// 3 widening to 1 024 channels, then one of stride 1.
    fn default() -> Self {
        let layer = |channels, stride| LayerSize { channels, stride };

        Self { periods: vec![2, 3, 5, 7, 11], layers: vec![layer(32, 3), layer(128, 3), layer(512, 3), layer(1024, 3), layer(1024, 1)] }
    }
}

/// What follows a convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Activation {
    /// The leaky ReLU, after every layer but the last.
    LeakyRelu,

    /// Nothing: the scoring layer's outputs are the scores themselves.
    Identity,
}

/// One strided convolution, computed as an unfold followed by a [`Linear`]
/// map from `kernel · input channels` to the output channels.
struct ConvolutionLayer {
    /// The filters, `[kernel · input channels, output channels]`, and the bias.
    linear: Linear,

    /// Taps of the filter.
    kernel: usize,

    /// Input frames between consecutive output frames.
    stride: usize,

    /// What follows the convolution.
    activation: Activation,
}

impl ConvolutionLayer {
    /// A layer from `inputs` to `outputs` channels, He-initialized.
    fn new(gpu: &Gpu, name: &str, size: Size, activation: Activation, random: &mut Random) -> GpuResult<Self> {
        let fan_in = size.kernel * size.inputs;
        let weight_std = (HE_GAIN / fan_in as f32).sqrt();
        let linear = Linear::new(gpu, name, fan_in, size.outputs, weight_std, random)?;

        Ok(Self { linear, kernel: size.kernel, stride: size.stride, activation })
    }

    /// The convolution's shape over `sequences` sequences of `time` frames.
    fn shape(&self, sequences: usize, time: usize) -> StridedConvolution {
        StridedConvolution::new(sequences, time, self.linear.inputs() / self.kernel, self.kernel, self.stride)
    }
}

/// The sizes needed to build one [`ConvolutionLayer`].
#[derive(Clone, Copy)]
struct Size {
    /// Input channels.
    inputs: usize,

    /// Output channels.
    outputs: usize,

    /// Taps of the filter.
    kernel: usize,

    /// Input frames between consecutive output frames.
    stride: usize,
}

/// The buffers of one layer's forward pass over one batch.
pub struct LayerActivations {
    /// The convolution's shape for this batch.
    shape: StridedConvolution,

    /// The unfolded input, `[output rows, kernel · input channels]`.
    columns: Tensor,

    /// The convolution's output before the activation; absent for the
    /// scoring layer, whose output is written straight into `features`.
    pre_activation: Option<Tensor>,

    /// The layer's output, `[output rows, output channels]`: a feature for
    /// feature matching, and for the last layer the scores.
    features: Tensor,
}

/// The gradient buffers of one layer, matching [`LayerActivations`].
pub struct LayerGradients {
    /// Gradient with respect to the layer's output. The objective writes its
    /// own part here before the backward pass; the backward pass then adds
    /// the part that flows down from the layer above.
    pub features: Tensor,

    /// Gradient with respect to the convolution's output before the activation.
    pre_activation: Tensor,

    /// Gradient with respect to the unfolded input.
    columns: Tensor,

    /// Gradient with respect to the layer's input.
    input: Tensor,
}

/// Everything one sub-discriminator computes on one batch of signals.
pub struct Judgement {
    /// How the signals are split by this sub-discriminator's period.
    layout: PeriodSplit,

    /// The split signals, `[signals · period, frames]`.
    split: Tensor,

    /// Every layer's buffers, from the first to the scoring layer.
    layers: Vec<LayerActivations>,
}

impl Judgement {
    /// The scores, one per frame of every split sequence.
    pub fn scores(&self) -> &Tensor {
        &self
            .layers
            .last()
            .expect("a discriminator has layers")
            .features
    }

    /// Every layer's output, from the first layer to the scores.
    pub fn features(&self) -> impl Iterator<Item = &Tensor> {
        self.layers.iter().map(|layer| &layer.features)
    }
}

/// The gradient buffers of one [`Judgement`].
pub struct JudgementGradients {
    /// Every layer's gradients, from the first to the scoring layer.
    pub layers: Vec<LayerGradients>,
}

/// One sub-discriminator: a period and its convolutions.
pub struct PeriodDiscriminator {
    /// Samples between the entries of one split sequence.
    period: usize,

    /// The downsampling layers, then the scoring layer.
    layers: Vec<ConvolutionLayer>,
}

impl PeriodDiscriminator {
    /// A freshly initialized sub-discriminator of `period` with `sizes`.
    fn new(gpu: &Gpu, period: usize, sizes: &[LayerSize], random: &mut Random) -> GpuResult<Self> {
        let mut layers = Vec::with_capacity(sizes.len() + 1);
        let mut inputs = 1;

        for (index, layer) in sizes.iter().enumerate() {
            let size = Size { inputs, outputs: layer.channels, kernel: KERNEL, stride: layer.stride };
            layers.push(ConvolutionLayer::new(gpu, &format!("period{period}.layers.{index}"), size, Activation::LeakyRelu, random)?);
            inputs = layer.channels;
        }

        let score = Size { inputs, outputs: SCORE_CHANNELS, kernel: SCORE_KERNEL, stride: 1 };
        layers.push(ConvolutionLayer::new(gpu, &format!("period{period}.score"), score, Activation::Identity, random)?);

        Ok(Self { period, layers })
    }

    /// Buffers for judging `signals` signals of `length` samples.
    pub fn judgement(&self, gpu: &Gpu, signals: usize, length: usize) -> GpuResult<(Judgement, JudgementGradients)> {
        let layout = PeriodSplit { signals, length, period: self.period };
        let mut time = layout.frames();
        let mut activations = Vec::with_capacity(self.layers.len());
        let mut gradients = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let shape = layer.shape(layout.sequences(), time);
            let outputs = Shape::new(shape.output_rows(), layer.linear.outputs());
            let columns = Shape::new(shape.output_rows(), shape.width());
            let inputs = Shape::new(shape.input_rows(), shape.channels);

            let pre_activation = match layer.activation {
                Activation::LeakyRelu => Some(gpu.zeros(outputs)?),
                Activation::Identity => None,
            };
            activations.push(LayerActivations { shape, columns: gpu.zeros(columns)?, pre_activation, features: gpu.zeros(outputs)? });
            gradients.push(LayerGradients {
                features: gpu.zeros(outputs)?,
                pre_activation: gpu.zeros(outputs)?,
                columns: gpu.zeros(columns)?,
                input: gpu.zeros(inputs)?,
            });

            time = shape.output_time;
        }

        let split = gpu.zeros(Shape::new(layout.sequences(), layout.frames()))?;
        Ok((Judgement { layout, split, layers: activations }, JudgementGradients { layers: gradients }))
    }

    /// Judges `signals` (`[signals, length]`), filling `judgement`.
    pub fn forward(&self, gpu: &Gpu, signals: &Tensor, judgement: &mut Judgement) -> GpuResult {
        gpu.split_periods(signals, &mut judgement.split, judgement.layout)?;

        for index in 0..self.layers.len() {
            let (below, rest) = judgement.layers.split_at_mut(index);
            let input = below
                .last()
                .map_or(&judgement.split, |previous| &previous.features);
            let current = &mut rest[0];
            let layer = &self.layers[index];

            gpu.unfold_strided(input, &mut current.columns, current.shape)?;

            match current.pre_activation.as_mut() {
                Some(pre_activation) => {
                    layer
                        .linear
                        .forward(gpu, &current.columns, pre_activation)?;
                    gpu.leaky_relu(pre_activation, &mut current.features)?;
                }
                None => layer
                    .linear
                    .forward(gpu, &current.columns, &mut current.features)?,
            }
        }

        Ok(())
    }

    /// Accumulates every parameter's gradient from the gradients the
    /// objective wrote into `gradients.layers[_].features`, and, when
    /// `signals_grad` is given, adds the gradient with respect to the signals
    /// into it. The feature gradients are consumed: the backward pass adds
    /// into them.
    pub fn backward(
        &mut self,
        gpu: &Gpu,
        judgement: &Judgement,
        gradients: &mut JudgementGradients,
        signals_grad: Option<&mut Tensor>,
    ) -> GpuResult {
        for index in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[index];
            let saved = &judgement.layers[index];
            let (below, rest) = gradients.layers.split_at_mut(index);
            let current = &mut rest[0];

            let output_grad = match saved.pre_activation.as_ref() {
                Some(pre_activation) => {
                    gpu.leaky_relu_backward(pre_activation, &current.features, &mut current.pre_activation)?;
                    &current.pre_activation
                }
                None => &current.features,
            };
            layer
                .linear
                .backward(gpu, &saved.columns, output_grad, &mut current.columns)?;

            let Some(previous) = below.last_mut() else { break };
            gpu.unfold_strided_backward(&current.columns, &mut current.input, saved.shape)?;
            gpu.add_scaled(1.0, &current.input, &mut previous.features)?;
        }

        let Some(signals_grad) = signals_grad else { return Ok(()) };
        let first = &mut gradients.layers[0];
        gpu.unfold_strided_backward(&first.columns, &mut first.input, judgement.layers[0].shape)?;
        gpu.split_periods_backward(&first.input, signals_grad, judgement.layout)
    }
}

impl Parameters for PeriodDiscriminator {
    fn visit_parameters(&mut self, visit: &mut dyn FnMut(&mut Parameter)) {
        for layer in &mut self.layers {
            layer.linear.visit_parameters(visit);
        }
    }
}

/// One sub-discriminator per period.
pub struct MultiPeriodDiscriminator {
    /// The sub-discriminators, in the order of the configured periods.
    pub discriminators: Vec<PeriodDiscriminator>,
}

impl MultiPeriodDiscriminator {
    /// A freshly initialized discriminator for `config`.
    pub fn new(gpu: &Gpu, config: &DiscriminatorConfig, random: &mut Random) -> GpuResult<Self> {
        let discriminators = config
            .periods
            .iter()
            .map(|&period| PeriodDiscriminator::new(gpu, period, &config.layers, random))
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self { discriminators })
    }
}

impl Parameters for MultiPeriodDiscriminator {
    fn visit_parameters(&mut self, visit: &mut dyn FnMut(&mut Parameter)) {
        for discriminator in &mut self.discriminators {
            discriminator.visit_parameters(visit);
        }
    }
}
