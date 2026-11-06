//! A neural vocoder: turns a log-mel spectrogram into a waveform.
//!
//! Griffin-Lim estimates the phase that a spectrogram does not store by making
//! neighboring frames agree, without ever hearing real audio, and voices take
//! on a metallic smear. This vocoder learns phase, and part of the detail that
//! StaleyNET's tokens lose, from real recordings instead.
//!
//! It follows Vocos (Siuzdak, 2023, <https://arxiv.org/abs/2306.00814>). The
//! network runs at the frame rate: a convolution lifts each frame's mel bands
//! into a wider vector, a stack of ConvNeXt blocks (Liu et al., 2022,
//! <https://arxiv.org/abs/2201.03545>) mixes neighboring frames, and a head
//! predicts the log-magnitude and the phase of every frequency. An inverse
//! short-time Fourier transform turns that spectrum into samples.
//!
//! * [`config`]: sizes and settings.
//! * [`block`]: one ConvNeXt block, forward and backward.
//! * [`network`]: the input convolution, the blocks and the spectrum head.
//! * [`istft`]: the inverse Fourier transform and overlap-add.
//! * [`synthesis`]: from the head's output to samples, and back.
//! * [`loss`]: the spectral losses that compare a waveform with its target.
//! * [`model`]: the vocoder as a whole, mel spectrogram in, waveform out.
//! * [`training_pass`]: one forward and backward pass over a batch.
//! * [`checkpoint`]: saving and loading weights and optimizer state.
//! * [`discriminator`]: HiFi-GAN's multi-period discriminator.
//! * [`adversarial`]: the least-squares and feature-matching objective.
//!
//! The training loop and its data live in `staleynet-train`, as the language
//! model's do; the GPU kernels live in `staleynet-cuda`.

pub mod adversarial;
pub mod block;
pub mod checkpoint;
pub mod config;
pub mod discriminator;
pub mod istft;
pub mod loss;
pub mod model;
pub mod network;
pub mod synthesis;
pub mod training_pass;

pub use adversarial::{AdversarialLosses, AdversarialWeights, Verdicts};
pub use config::VocoderConfig;
pub use discriminator::{DiscriminatorConfig, LayerSize, MultiPeriodDiscriminator};
pub use loss::{Resolution, TRAINING_RESOLUTIONS};
pub use model::Vocoder;
pub use training_pass::{LossBreakdown, TrainingPass};
