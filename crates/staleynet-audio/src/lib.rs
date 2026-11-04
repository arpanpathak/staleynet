//! Audio input and output for StaleyNET.
//!
//! The crate turns arbitrary audio files into mono sample buffers at the model
//! sample rate, analyses them into log-mel spectrograms, and turns log-mel
//! spectrograms back into waveforms. The spectral code here is the CPU
//! reference against which the GPU feature extractor is validated.

pub mod decode;
pub mod griffin_lim;
pub mod mel;
pub mod resample;
pub mod wav;

pub use decode::{MonoAudio, decode_mono};
pub use mel::{MelAnalyzer, MelConfig};
