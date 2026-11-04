//! The vocoder's sizes and settings.

use serde::{Deserialize, Serialize};

/// Every size the vocoder depends on. The defaults are those of Vocos at
/// 24 kHz (Siuzdak, 2023), with StaleyNET's 128 mel bands as input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VocoderConfig {
    /// Sample rate of the waveform, in Hz.
    pub sample_rate: u32,

    /// Mel bands per input frame.
    pub mel_bands: usize,

    /// Samples per analysis and synthesis frame (the FFT length).
    pub fft_size: usize,

    /// Samples between the starts of consecutive frames.
    pub hop_size: usize,

    /// Width of the vector each frame carries through the backbone.
    pub width: usize,

    /// ConvNeXt blocks in the backbone.
    pub layers: usize,

    /// Taps of every convolution over time; frame `t` sees frames
    /// `t − taps/2 … t + taps/2`.
    pub taps: usize,

    /// The feed-forward network of each block expands to `expansion · width`.
    pub expansion: usize,

    /// Cap on predicted log-magnitudes, so that an untrained network cannot
    /// overflow `exp`; e^4.6 ≈ 100, as in Vocos.
    pub max_log_magnitude: f32,
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24_000,
            mel_bands: 128,
            fft_size: 1024,
            hop_size: 256,
            width: 512,
            layers: 8,
            taps: 7,
            expansion: 3,
            max_log_magnitude: 4.6,
        }
    }
}

impl VocoderConfig {
    /// Non-negative frequencies of one frame, `fft_size / 2 + 1`.
    pub fn bins(&self) -> usize {
        self.fft_size / 2 + 1
    }

    /// Samples of the waveform synthesized from `frames` frames, before the
    /// half-frame of padding at each end is trimmed.
    pub fn padded_samples(&self, frames: usize) -> usize {
        (frames - 1) * self.hop_size + self.fft_size
    }

    /// Panics with a clear message if the sizes cannot work together.
    pub fn validate(&self) {
        assert!(self.taps % 2 == 1, "taps must be odd, so that frame t is the center of its window");
        assert!(self.fft_size.is_multiple_of(self.hop_size), "the hop must divide the frame length");
        assert!(self.width > 0 && self.layers > 0 && self.expansion > 0, "the backbone must not be empty");
    }
}
