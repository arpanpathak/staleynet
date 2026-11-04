//! Position-wise feed-forward network `FFN(x) = GELU(x·W₁ + b₁)·W₂ + b₂`
//! (Vaswani et al., 2017, <https://arxiv.org/abs/1706.03762>, §3.3, with GELU
//! as in GPT-2).

use super::linear::Linear;
use crate::parameter::{Parameter, Parameters};
use crate::random::Random;
use staleynet_cuda::{Gpu, GpuResult, Shape, Tensor};

/// Expansion to a wider hidden layer, GELU, and projection back.
pub struct FeedForward {
    /// `width → hidden`.
    pub expand: Linear,

    /// `hidden → width`.
    pub project: Linear,
}

/// Hidden-layer tensors for a fixed number of rows. One instance holds the
/// forward activations; a second, shared one holds their gradients.
pub struct FeedForwardTensors {
    /// Hidden layer before GELU.
    pub pre_activation: Tensor,

    /// Hidden layer after GELU.
    pub hidden: Tensor,
}

impl FeedForwardTensors {
    /// Buffers for `rows` positions.
    pub fn new(gpu: &Gpu, rows: usize, hidden_width: usize) -> GpuResult<Self> {
        Ok(Self { pre_activation: gpu.zeros(Shape::new(rows, hidden_width))?, hidden: gpu.zeros(Shape::new(rows, hidden_width))? })
    }
}

impl FeedForward {
    /// A network `width → hidden_width → width`; `project_std` is the
    /// (depth-scaled) initial std of the residual projection.
    pub fn new(
        gpu: &Gpu,
        name: &str,
        width: usize,
        hidden_width: usize,
        expand_std: f32,
        project_std: f32,
        random: &mut Random,
    ) -> GpuResult<Self> {
        Ok(Self {
            expand: Linear::new(gpu, &format!("{name}.expand"), width, hidden_width, expand_std, random)?,
            project: Linear::new(gpu, &format!("{name}.project"), hidden_width, width, project_std, random)?,
        })
    }

    /// `output = FFN(input)`.
    pub fn forward(&self, gpu: &Gpu, input: &Tensor, activations: &mut FeedForwardTensors, output: &mut Tensor) -> GpuResult {
        self.expand
            .forward(gpu, input, &mut activations.pre_activation)?;
        gpu.gelu_forward(&activations.pre_activation, &mut activations.hidden)?;
        self.project.forward(gpu, &activations.hidden, output)
    }

    /// Writes `input_grad`, using `hidden_grads` as scratch for the hidden-layer gradients.
    pub fn backward(
        &mut self,
        gpu: &Gpu,
        input: &Tensor,
        activations: &FeedForwardTensors,
        output_grad: &Tensor,
        hidden_grads: &mut FeedForwardTensors,
        input_grad: &mut Tensor,
    ) -> GpuResult {
        self.project
            .backward(gpu, &activations.hidden, output_grad, &mut hidden_grads.hidden)?;
        gpu.gelu_backward(&activations.pre_activation, &hidden_grads.hidden, &mut hidden_grads.pre_activation)?;
        self.expand
            .backward(gpu, input, &hidden_grads.pre_activation, input_grad)
    }
}

impl Parameters for FeedForward {
    fn visit_parameters(&mut self, visit: &mut dyn FnMut(&mut Parameter)) {
        self.expand.visit_parameters(visit);
        self.project.visit_parameters(visit);
    }
}
