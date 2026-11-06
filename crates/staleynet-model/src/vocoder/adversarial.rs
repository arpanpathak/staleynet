//! The adversarial objective of HiFi-GAN (Kong, Kim and Bae, 2020,
//! <https://arxiv.org/abs/2010.05646>, §2.3).
//!
//! The discriminator is trained with the least-squares objective of Mao et
//! al., 2017 (<https://arxiv.org/abs/1611.04076>): the scores it gives real
//! audio are pulled towards 1 and those of synthesized audio towards 0. The
//! vocoder is trained against it with two terms. The *adversarial* term pulls
//! the scores of its own audio towards 1, so that it sounds real. The
//! *feature-matching* term asks every layer of the discriminator to respond
//! to the synthesized audio as it responds to the real recording, which gives
//! the vocoder a dense signal about what differs, not only a verdict.

use super::discriminator::{Judgement, JudgementGradients, MultiPeriodDiscriminator};
use serde::{Deserialize, Serialize};
use staleynet_cuda::{Gpu, GpuResult, Tensor, Verdict};

/// How much each adversarial term weighs against the spectral losses.
///
/// HiFi-GAN weighs its mel-spectrogram loss 45 times the adversarial term and
/// 22.5 times the feature-matching term. Our spectral losses keep a weight of
/// 1, so the defaults are those ratios inverted.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdversarialWeights {
    /// Weight of the least-squares adversarial term.
    pub adversarial: f32,

    /// Weight of the feature-matching term.
    pub feature_matching: f32,
}

impl Default for AdversarialWeights {
    fn default() -> Self {
        Self { adversarial: 1.0 / 45.0, feature_matching: 2.0 / 45.0 }
    }
}

/// The adversarial losses of one step, for the progress log.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AdversarialLosses {
    /// The discriminator's least-squares loss on real and synthesized audio.
    pub discriminator: f64,

    /// The vocoder's weighted least-squares term.
    pub adversarial: f64,

    /// The vocoder's weighted feature-matching term.
    pub feature_matching: f64,
}

/// What every sub-discriminator computes on one batch of signals, with the
/// matching gradient buffers.
pub struct Verdicts {
    /// One judgement per sub-discriminator, in their order.
    judgements: Vec<Judgement>,

    /// One set of gradient buffers per judgement.
    gradients: Vec<JudgementGradients>,

    /// One buffer of each judgement's score shape, where the adversarial
    /// term's gradient is computed before it is added to feature matching's.
    score_scratch: Vec<Tensor>,
}

impl Verdicts {
    /// Buffers for judging `signals` signals of `length` samples.
    pub fn new(gpu: &Gpu, discriminator: &MultiPeriodDiscriminator, signals: usize, length: usize) -> GpuResult<Self> {
        let periods = discriminator.discriminators.len();
        let mut judgements = Vec::with_capacity(periods);
        let mut gradients = Vec::with_capacity(periods);
        let mut score_scratch = Vec::with_capacity(periods);

        for period in &discriminator.discriminators {
            let (judgement, judgement_gradients) = period.judgement(gpu, signals, length)?;
            score_scratch.push(gpu.zeros(judgement.scores().shape())?);
            judgements.push(judgement);
            gradients.push(judgement_gradients);
        }

        Ok(Self { judgements, gradients, score_scratch })
    }

    /// Judges `signals` with every sub-discriminator.
    pub fn judge(&mut self, gpu: &Gpu, discriminator: &MultiPeriodDiscriminator, signals: &Tensor) -> GpuResult {
        for (period, judgement) in discriminator
            .discriminators
            .iter()
            .zip(&mut self.judgements)
        {
            period.forward(gpu, signals, judgement)?;
        }

        Ok(())
    }
}

/// Accumulates the discriminator's gradient: its scores on `real` are pulled
/// towards 1 and its scores on `synthesized` towards 0. Both batches must
/// have been judged with the current discriminator.
pub fn discriminator_backward(
    gpu: &Gpu,
    discriminator: &mut MultiPeriodDiscriminator,
    real: &mut Verdicts,
    synthesized: &mut Verdicts,
) -> GpuResult {
    for (side, verdict) in [(real, Verdict::Real), (synthesized, Verdict::Synthesized)] {
        for (index, period) in discriminator.discriminators.iter_mut().enumerate() {
            let judgement = &side.judgements[index];
            let gradients = &mut side.gradients[index];
            let scoring_layer = gradients.layers.len() - 1;
            let (features, scores) = gradients.layers.split_at_mut(scoring_layer);

            for layer in features {
                gpu.fill_zero(&mut layer.features)?;
            }
            let scale = 1.0 / judgement.scores().len() as f32;
            gpu.least_squares_gradient(judgement.scores(), &mut scores[0].features, verdict, scale)?;

            period.backward(gpu, judgement, gradients, None)?;
        }
    }

    Ok(())
}

/// Adds the vocoder's adversarial gradient into `signals_grad`: the gradient
/// of the weighted least-squares term on `synthesized` and of the weighted
/// feature-matching term between `synthesized` and `real`. Both batches must
/// have been judged with the current discriminator. The discriminator's own
/// gradients are disturbed and must be zeroed before its next update.
pub fn generator_backward(
    gpu: &Gpu,
    discriminator: &mut MultiPeriodDiscriminator,
    real: &Verdicts,
    synthesized: &mut Verdicts,
    weights: AdversarialWeights,
    signals_grad: &mut Tensor,
) -> GpuResult {
    for (index, period) in discriminator.discriminators.iter_mut().enumerate() {
        let target = &real.judgements[index];
        let judgement = &synthesized.judgements[index];
        let gradients = &mut synthesized.gradients[index];
        let adversarial_grad = &mut synthesized.score_scratch[index];

        for ((estimate, reference), layer) in judgement
            .features()
            .zip(target.features())
            .zip(&mut gradients.layers)
        {
            let scale = weights.feature_matching / estimate.len() as f32;
            gpu.absolute_error_gradient(estimate, reference, &mut layer.features, scale)?;
        }

        let scores_grad = gradients
            .layers
            .last_mut()
            .expect("a discriminator has layers");
        let scale = weights.adversarial / judgement.scores().len() as f32;
        gpu.least_squares_gradient(judgement.scores(), adversarial_grad, Verdict::Real, scale)?;
        gpu.add_scaled(1.0, adversarial_grad, &mut scores_grad.features)?;

        period.backward(gpu, judgement, gradients, Some(signals_grad))?;
    }

    Ok(())
}

/// The losses behind the last gradients, for logging: the discriminator's
/// loss on `real` and `synthesized`, and the vocoder's two weighted terms.
/// Downloads every feature, so it is meant for occasional use.
pub fn losses(gpu: &Gpu, real: &Verdicts, synthesized: &Verdicts, weights: AdversarialWeights) -> GpuResult<AdversarialLosses> {
    let mut losses = AdversarialLosses::default();

    for (target, judgement) in real.judgements.iter().zip(&synthesized.judgements) {
        let real_scores = gpu.download(target.scores())?;
        let synthesized_scores = gpu.download(judgement.scores())?;

        losses.discriminator += squared_distance(&real_scores, Verdict::Real) + squared_distance(&synthesized_scores, Verdict::Synthesized);
        losses.adversarial += weights.adversarial as f64 * squared_distance(&synthesized_scores, Verdict::Real);

        for (estimate, reference) in judgement.features().zip(target.features()) {
            let distance = absolute_distance(&gpu.download(estimate)?, &gpu.download(reference)?);
            losses.feature_matching += weights.feature_matching as f64 * distance;
        }
    }

    Ok(losses)
}

/// The mean of `(score − target)²` over `scores`.
fn squared_distance(scores: &[f32], verdict: Verdict) -> f64 {
    let target = verdict.target() as f64;
    let sum: f64 = scores
        .iter()
        .map(|&score| (score as f64 - target).powi(2))
        .sum();

    sum / scores.len().max(1) as f64
}

/// The mean absolute difference of two equally long feature maps.
fn absolute_distance(estimate: &[f32], reference: &[f32]) -> f64 {
    let sum: f64 = estimate
        .iter()
        .zip(reference)
        .map(|(&left, &right)| (left as f64 - right as f64).abs())
        .sum();

    sum / estimate.len().max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distances_are_means_over_the_values() {
        assert_eq!(squared_distance(&[1.0, 0.0], Verdict::Real), 0.5);
        assert_eq!(squared_distance(&[1.0, 0.0], Verdict::Synthesized), 0.5);
        assert_eq!(absolute_distance(&[1.0, -1.0], &[0.0, 1.0]), 1.5);
    }
}
