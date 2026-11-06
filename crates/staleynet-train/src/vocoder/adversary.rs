//! The discriminator's side of adversarial vocoder training.
//!
//! Every step follows HiFi-GAN (Kong, Kim and Bae, 2020,
//! <https://arxiv.org/abs/2010.05646>): the vocoder synthesizes the batch; the
//! discriminator judges the recordings and the synthesized audio and takes
//! one optimizer step; it then judges both again, and the vocoder's gradient
//! from the adversarial and feature-matching terms is added to that of the
//! spectral losses before the vocoder takes its own step.

use super::config::AdversarialTraining;
use anyhow::Result;
use staleynet_cuda::{Gpu, GpuResult};
use staleynet_model::Random;
use staleynet_model::optimizer::AdamW;
use staleynet_model::parameter::Parameters;
use staleynet_model::vocoder::adversarial::{self, AdversarialLosses, AdversarialWeights, Verdicts};
use staleynet_model::vocoder::{MultiPeriodDiscriminator, TrainingPass, checkpoint};
use std::path::Path;

/// The discriminator, its optimizer, and its buffers for one batch shape.
pub struct Adversary {
    /// The multi-period discriminator.
    discriminator: MultiPeriodDiscriminator,

    /// The discriminator's AdamW state.
    optimizer: AdamW,

    /// Its judgement of the recordings.
    real: Verdicts,

    /// Its judgement of the synthesized audio.
    synthesized: Verdicts,

    /// How much the vocoder's adversarial terms weigh.
    weights: AdversarialWeights,
}

impl Adversary {
    /// A fresh discriminator for batches of `signals` signals of `length`
    /// samples, its weights drawn from `seed`.
    pub fn new(gpu: &Gpu, settings: &AdversarialTraining, signals: usize, length: usize, seed: u64) -> GpuResult<Self> {
        let mut random = Random::new(seed);
        let mut discriminator = MultiPeriodDiscriminator::new(gpu, &settings.discriminator, &mut random)?;
        let optimizer = AdamW::new(gpu, settings.optimizer.clone(), &mut discriminator)?;

        Ok(Self {
            real: Verdicts::new(gpu, &discriminator, signals, length)?,
            synthesized: Verdicts::new(gpu, &discriminator, signals, length)?,
            discriminator,
            optimizer,
            weights: settings.weights,
        })
    }

    /// Restores the discriminator and its optimizer from `directory`, which
    /// is at step `step`.
    pub fn resume(&mut self, gpu: &Gpu, directory: &Path, step: usize) -> Result<()> {
        checkpoint::load_discriminator(gpu, directory, &mut self.discriminator, &mut self.optimizer)?;
        self.optimizer.steps_taken = step;

        Ok(())
    }

    /// Saves the discriminator and its optimizer into `directory`.
    pub fn save(&mut self, gpu: &Gpu, directory: &Path) -> Result<()> {
        checkpoint::save_discriminator(gpu, directory, &mut self.discriminator, &self.optimizer)
    }

    /// One discriminator step on the batch of `pass`'s last forward pass:
    /// its scores on the recordings move towards 1, on the synthesized audio
    /// towards 0.
    pub fn train_discriminator(&mut self, gpu: &Gpu, pass: &TrainingPass) -> GpuResult {
        self.discriminator.zero_gradients(gpu)?;

        self.real.judge(gpu, &self.discriminator, pass.target())?;
        self.synthesized
            .judge(gpu, &self.discriminator, pass.synthesized())?;
        adversarial::discriminator_backward(gpu, &mut self.discriminator, &mut self.real, &mut self.synthesized)?;

        self.optimizer.step(gpu, &mut self.discriminator)?;
        Ok(())
    }

    /// Judges the batch again with the updated discriminator and adds the
    /// vocoder's adversarial and feature-matching gradient into `pass`'s
    /// signal gradient. Call after [`TrainingPass::spectral_backward`].
    pub fn add_vocoder_gradient(&mut self, gpu: &Gpu, pass: &mut TrainingPass) -> GpuResult {
        self.real.judge(gpu, &self.discriminator, pass.target())?;
        self.synthesized
            .judge(gpu, &self.discriminator, pass.synthesized())?;

        adversarial::generator_backward(gpu, &mut self.discriminator, &self.real, &mut self.synthesized, self.weights, pass.signal_grad())
    }

    /// The adversarial losses of the last [`Adversary::add_vocoder_gradient`].
    pub fn losses(&self, gpu: &Gpu) -> GpuResult<AdversarialLosses> {
        adversarial::losses(gpu, &self.real, &self.synthesized, self.weights)
    }
}
