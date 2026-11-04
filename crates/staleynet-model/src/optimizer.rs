//! AdamW with global-norm gradient clipping and a warm-up + cosine schedule.

use crate::parameter::{Decay, Parameter, Parameters};
use serde::{Deserialize, Serialize};
use staleynet_cuda::{AdamWHyperparameters, Gpu, GpuResult, Tensor};

/// Optimiser settings. Defaults follow common GPT training recipes
/// (β₂ = 0.95 and decay 0.1 as in GPT-3, Brown et al., 2020,
/// <https://arxiv.org/abs/2005.14165>, App. B).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Learning rate after warm-up.
    pub peak_learning_rate: f32,

    /// Learning rate at the end of the cosine decay.
    pub final_learning_rate: f32,

    /// Steps of linear warm-up from 0 to the peak.
    pub warmup_steps: usize,

    /// Step at which the cosine reaches `final_learning_rate`.
    pub decay_steps: usize,

    /// First-moment decay β₁.
    pub beta1: f32,

    /// Second-moment decay β₂.
    pub beta2: f32,

    /// Denominator offset ε.
    pub epsilon: f32,

    /// Decoupled weight decay for [`Decay::Decayed`] parameters.
    pub weight_decay: f32,

    /// Gradients are rescaled when their global L2 norm exceeds this
    /// (Pascanu, Mikolov & Bengio, 2013, <https://arxiv.org/abs/1211.5063>).
    pub max_gradient_norm: f32,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            peak_learning_rate: 3e-4,
            final_learning_rate: 3e-5,
            warmup_steps: 500,
            decay_steps: 50_000,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            weight_decay: 0.1,
            max_gradient_norm: 1.0,
        }
    }
}

impl OptimizerConfig {
    /// Linear warm-up, then half-cosine decay (Loshchilov & Hutter, 2017,
    /// <https://arxiv.org/abs/1608.03983>), then constant.
    pub fn learning_rate(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            return self.peak_learning_rate * (step + 1) as f32 / self.warmup_steps as f32;
        }

        let decay_span = self.decay_steps.saturating_sub(self.warmup_steps).max(1);
        let progress = ((step - self.warmup_steps) as f32 / decay_span as f32).min(1.0);
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());

        self.final_learning_rate + (self.peak_learning_rate - self.final_learning_rate) * cosine
    }
}

/// AdamW's running moment estimates for one parameter.
pub struct Moments {
    /// Exponential average of gradients.
    pub first: Tensor,

    /// Exponential average of squared gradients.
    pub second: Tensor,
}

/// What one optimiser step did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepReport {
    /// Global gradient norm before clipping.
    pub gradient_norm: f32,

    /// Learning rate used.
    pub learning_rate: f32,
}

/// AdamW (Loshchilov & Hutter, 2019, <https://arxiv.org/abs/1711.05101>).
pub struct AdamW {
    /// Settings.
    pub config: OptimizerConfig,

    /// Moments, in [`Parameters::visit_parameters`] order.
    pub moments: Vec<Moments>,

    /// Steps taken so far.
    pub steps_taken: usize,
}

impl AdamW {
    /// Zero-initialised moments for every parameter of `model`.
    pub fn new(gpu: &Gpu, config: OptimizerConfig, model: &mut impl Parameters) -> GpuResult<Self> {
        let mut moments = Vec::new();
        let mut result = Ok(());

        model.visit_parameters(&mut |parameter| {
            let shape = parameter.value.shape();
            let allocated = gpu
                .zeros(shape)
                .and_then(|first| Ok(Moments { first, second: gpu.zeros(shape)? }));
            match allocated {
                Ok(moment) => moments.push(moment),
                Err(error) => result = Err(error),
            }
        });
        result?;

        Ok(Self { config, moments, steps_taken: 0 })
    }

    /// Clips the accumulated gradients and applies one update.
    pub fn step(&mut self, gpu: &Gpu, model: &mut impl Parameters) -> GpuResult<StepReport> {
        let gradient_norm = global_gradient_norm(gpu, model)? as f32;
        let clip_scale = if gradient_norm > self.config.max_gradient_norm { self.config.max_gradient_norm / gradient_norm } else { 1.0 };
        let learning_rate = self.config.learning_rate(self.steps_taken);

        self.steps_taken += 1;

        let config = &self.config;
        let step = self.steps_taken as u32;
        let mut moments = self.moments.iter_mut();
        let mut result = Ok(());

        model.visit_parameters(&mut |parameter: &mut Parameter| {
            let Some(moment) = moments.next() else { return };
            if result.is_err() {
                return;
            }

            let weight_decay = match parameter.decay {
                Decay::Decayed => config.weight_decay,
                Decay::Exempt => 0.0,
            };
            let hyper = AdamWHyperparameters {
                learning_rate,
                beta1: config.beta1,
                beta2: config.beta2,
                epsilon: config.epsilon,
                weight_decay,
                step,
            };

            result = gpu
                .scale(&mut parameter.gradient, clip_scale)
                .and_then(|()| gpu.adamw_step(&mut parameter.value, &parameter.gradient, &mut moment.first, &mut moment.second, &hyper));
        });
        result?;

        Ok(StepReport { gradient_norm, learning_rate })
    }
}

/// `√(Σ over all parameters of ‖gradient‖²)`.
pub fn global_gradient_norm(gpu: &Gpu, model: &mut impl Parameters) -> GpuResult<f64> {
    let mut sum = 0.0;
    let mut result = Ok(());

    model.visit_parameters(&mut |parameter| match gpu.sum_of_squares(&parameter.gradient) {
        Ok(squares) => sum += squares,
        Err(error) => result = Err(error),
    });

    result?;
    Ok(sum.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule() -> OptimizerConfig {
        OptimizerConfig {
            peak_learning_rate: 1.0,
            final_learning_rate: 0.1,
            warmup_steps: 10,
            decay_steps: 20,
            ..OptimizerConfig::default()
        }
    }

    #[test]
    fn defaults_match_the_published_recipe() {
        let config = OptimizerConfig::default();

        assert_eq!(config.peak_learning_rate, 3e-4);
        assert_eq!(config.final_learning_rate, 3e-5);
        assert_eq!(config.warmup_steps, 500);
        assert_eq!(config.decay_steps, 50_000);
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.95);
        assert_eq!(config.max_gradient_norm, 1.0);
    }

    #[test]
    fn warmup_rises_linearly_to_the_peak() {
        let config = schedule();

        assert_eq!(config.learning_rate(0), 0.1);
        assert_eq!(config.learning_rate(4), 0.5);
        assert_eq!(config.learning_rate(9), 1.0);
    }

    #[test]
    fn cosine_decay_reaches_the_final_rate() {
        let config = schedule();

        assert_eq!(config.learning_rate(10), 1.0);
        assert!((config.learning_rate(15) - 0.55).abs() < 1e-5);
        assert!((config.learning_rate(20) - 0.1).abs() < 1e-5);
        assert!((config.learning_rate(100) - 0.1).abs() < 1e-5);
    }
}
