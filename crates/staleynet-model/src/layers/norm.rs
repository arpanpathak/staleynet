//! RMSNorm layer (Zhang & Sennrich, 2019, <https://arxiv.org/abs/1910.07467>).

use crate::parameter::{Decay, Init, Parameter, Parameters};
use crate::random::Random;
use staleynet_cuda::{Gpu, GpuResult, Shape, Tensor};

/// Added to the mean square before the square root, as in the RMSNorm paper
/// and LLaMA (Touvron et al., 2023, <https://arxiv.org/abs/2302.13971>).
pub const RMS_NORM_EPSILON: f32 = 1e-5;

/// Root-mean-square normalisation with a learned per-feature gain.
pub struct RmsNorm {
    /// Per-feature gain, initialised to one.
    pub gain: Parameter,
}

impl RmsNorm {
    /// A norm over `width` features.
    pub fn new(gpu: &Gpu, name: &str, width: usize, random: &mut Random) -> GpuResult<Self> {
        Ok(Self { gain: Parameter::new(gpu, format!("{name}.gain"), Shape::vector(width), Init::Ones, Decay::Exempt, random)? })
    }

    /// Normalises every row of `input` into `output`, saving `1/rms` per row.
    pub fn forward(&self, gpu: &Gpu, input: &Tensor, output: &mut Tensor, inv_rms: &mut Tensor) -> GpuResult {
        gpu.rms_norm_forward(input, &self.gain.value, output, inv_rms, RMS_NORM_EPSILON)
    }

    /// Accumulates the gain gradient and writes the input gradient.
    pub fn backward(&mut self, gpu: &Gpu, output_grad: &Tensor, input: &Tensor, inv_rms: &Tensor, input_grad: &mut Tensor) -> GpuResult {
        gpu.rms_norm_backward(output_grad, input, &self.gain.value, inv_rms, input_grad, &mut self.gain.gradient)
    }
}

impl Parameters for RmsNorm {
    fn visit_parameters(&mut self, visit: &mut dyn FnMut(&mut Parameter)) {
        visit(&mut self.gain);
    }
}
