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
//! * [`network`]: the backbone and the head, with hand-derived gradients.
//! * [`istft`]: the inverse Fourier transform and overlap-add.
//! * [`loss`]: the spectral losses that compare a waveform with its target.
//! * [`vocoder`]: the whole path from mel spectrogram to waveform and back.
//! * [`data`]: segments of decoded spectrograms paired with their recordings.
//! * [`train`]: the training loop, its checkpoints, metrics and samples.

pub mod config;
pub mod data;
pub mod istft;
pub mod loss;
pub mod network;
pub mod train;
pub mod vocoder;

pub use config::VocoderConfig;
pub use vocoder::Vocoder;
