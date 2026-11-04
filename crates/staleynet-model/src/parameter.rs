//! Trainable tensors and the visitor that walks them.

use crate::random::Random;
use staleynet_cuda::{Gpu, GpuResult, Shape, Tensor};

/// Standard deviation of the normal initialisation of weight matrices and
/// embeddings, as in GPT-2 (Radford et al., 2019).
pub const INIT_STD: f32 = 0.02;

/// How a parameter is initialised.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Init {
    /// i.i.d. normal with the given standard deviation.
    Normal(f32),

    /// All ones (normalisation gains).
    Ones,

    /// All zeros (biases).
    Zeros,
}

/// Whether AdamW applies weight decay. Following common practice (e.g.
/// nanoGPT, <https://github.com/karpathy/nanoGPT>), only matrices that mix
/// features are decayed; gains, biases and embeddings are not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decay {
    /// Weight decay applies.
    Decayed,

    /// Weight decay does not apply.
    Exempt,
}

/// A tensor the optimiser updates, plus its accumulated gradient.
pub struct Parameter {
    /// Stable dotted name, e.g. `blocks.3.attention.qkv.weight`; used by checkpoints.
    pub name: String,

    /// Current value.
    pub value: Tensor,

    /// Gradient accumulated by backward passes since the last zeroing.
    pub gradient: Tensor,

    /// Weight-decay policy.
    pub decay: Decay,
}

impl Parameter {
    /// Allocates and initialises a parameter.
    pub fn new(gpu: &Gpu, name: String, shape: Shape, init: Init, decay: Decay, random: &mut Random) -> GpuResult<Self> {
        let values: Vec<f32> = match init {
            Init::Normal(std) => (0..shape.len()).map(|_| random.normal() * std).collect(),
            Init::Ones => vec![1.0; shape.len()],
            Init::Zeros => vec![0.0; shape.len()],
        };

        Ok(Self { value: gpu.upload(shape, &values)?, gradient: gpu.zeros(shape)?, decay, name })
    }
}

/// Anything that owns parameters. Implemented by every layer and by the model,
/// so optimisers and checkpoints can treat a whole network uniformly.
pub trait Parameters {
    /// Calls `visit` on every parameter, always in the same order.
    fn visit_parameters(&mut self, visit: &mut dyn FnMut(&mut Parameter));

    /// Zeroes every gradient.
    fn zero_gradients(&mut self, gpu: &Gpu) -> GpuResult {
        let mut result = Ok(());
        self.visit_parameters(&mut |parameter| {
            if result.is_ok() {
                result = gpu.fill_zero(&mut parameter.gradient);
            }
        });
        result
    }

    /// Total number of scalar parameters.
    fn parameter_count(&mut self) -> usize {
        let mut count = 0;
        self.visit_parameters(&mut |parameter| count += parameter.value.len());
        count
    }
}
