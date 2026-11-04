//! Affine layer `y = x·W + b`.

use crate::parameter::{Decay, Init, Parameter, Parameters};
use crate::random::Random;
use staleynet_cuda::{Gpu, GpuResult, MatmulSpec, Shape, Tensor};

/// A fully connected layer with weight `[inputs, outputs]` and bias `[outputs]`.
pub struct Linear {
    /// Weight matrix, `[inputs, outputs]`.
    pub weight: Parameter,

    /// Bias vector, `[outputs]`.
    pub bias: Parameter,
}

impl Linear {
    /// A layer whose weight is drawn from `N(0, weight_std²)` and bias is zero.
    pub fn new(gpu: &Gpu, name: &str, inputs: usize, outputs: usize, weight_std: f32, random: &mut Random) -> GpuResult<Self> {
        Ok(Self {
            weight: Parameter::new(
                gpu,
                format!("{name}.weight"),
                Shape::new(inputs, outputs),
                Init::Normal(weight_std),
                Decay::Decayed,
                random,
            )?,
            bias: Parameter::new(gpu, format!("{name}.bias"), Shape::vector(outputs), Init::Zeros, Decay::Exempt, random)?,
        })
    }

    /// Input features.
    pub fn inputs(&self) -> usize {
        self.weight.value.shape().rows
    }

    /// Output features.
    pub fn outputs(&self) -> usize {
        self.weight.value.shape().cols
    }

    /// `output = input · W + b` for `input` `[rows, inputs]`.
    pub fn forward(&self, gpu: &Gpu, input: &Tensor, output: &mut Tensor) -> GpuResult {
        let rows = input.shape().rows;

        gpu.matmul(&MatmulSpec::new(rows, self.outputs(), self.inputs()), input, &self.weight.value, output)?;
        gpu.add_row_bias(output, &self.bias.value)
    }

    /// Accumulates `∂W += xᵀ·∂y` and `∂b += Σ_rows ∂y`, and writes `∂x = ∂y·Wᵀ`.
    pub fn backward(&mut self, gpu: &Gpu, input: &Tensor, output_grad: &Tensor, input_grad: &mut Tensor) -> GpuResult {
        let rows = input.shape().rows;
        let (inputs, outputs) = (self.inputs(), self.outputs());

        let weight_grad = MatmulSpec::new(inputs, outputs, rows)
            .left_transposed()
            .accumulate();
        gpu.matmul(&weight_grad, input, output_grad, &mut self.weight.gradient)?;
        gpu.accumulate_column_sums(output_grad, &mut self.bias.gradient)?;

        let input_grad_spec = MatmulSpec::new(rows, inputs, outputs).right_transposed();
        gpu.matmul(&input_grad_spec, output_grad, &self.weight.value, input_grad)
    }
}

impl Parameters for Linear {
    fn visit_parameters(&mut self, visit: &mut dyn FnMut(&mut Parameter)) {
        visit(&mut self.weight);
        visit(&mut self.bias);
    }
}
