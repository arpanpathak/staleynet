//! The vocoder as a whole: mel spectrogram in, waveform out.
//!
//! The network predicts a log-magnitude and a phase for every frequency of
//! every frame, these become a complex spectrum, and the inverse STFT turns
//! the spectrum into samples. Its input is a log-mel spectrogram standardized
//! band by band with the statistics of the tokenizer, so that a spectrogram
//! decoded from tokens and one computed from a recording look alike to it.

use super::config::VocoderConfig;
use super::istft::InverseStft;
use super::network::{Activations, Network};
use super::synthesis::Synthesis;
use crate::random::Random;
use staleynet_audio::{MelAnalyzer, MelConfig};
use staleynet_cuda::{Gpu, GpuResult, Shape};

/// A trained (or freshly initialized) vocoder.
pub struct Vocoder {
    /// Sizes and settings.
    pub config: VocoderConfig,

    /// The network that predicts the spectrum.
    pub network: Network,

    /// The inverse STFT.
    pub istft: InverseStft,

    /// Mean log energy of each mel band, subtracted from the input.
    pub mel_mean: Vec<f32>,

    /// Standard deviation of each mel band, dividing the input.
    pub mel_std: Vec<f32>,
}

impl Vocoder {
    /// A freshly initialized vocoder whose input is standardized with
    /// `mel_mean` and `mel_std` (one value per mel band).
    pub fn new(gpu: &Gpu, config: VocoderConfig, mel_mean: Vec<f32>, mel_std: Vec<f32>, seed: u64) -> GpuResult<Self> {
        assert_eq!(mel_mean.len(), config.mel_bands, "one mean per mel band");
        assert_eq!(mel_std.len(), config.mel_bands, "one standard deviation per mel band");

        let network = Network::new(gpu, &config, &mut Random::new(seed))?;
        let istft = InverseStft::new(gpu, &config, &analyzer(&config).window)?;

        Ok(Self { config, network, istft, mel_mean, mel_std })
    }

    /// `log_mel` (`[frames, mel_bands]`, natural-log energies) standardized for the network.
    pub fn standardize(&self, log_mel: &[f32]) -> Vec<f32> {
        let bands = self.config.mel_bands;

        log_mel
            .iter()
            .enumerate()
            .map(|(index, value)| (value - self.mel_mean[index % bands]) / self.mel_std[index % bands])
            .collect()
    }

    /// The waveform of a log-mel spectrogram (`[frames, mel_bands]`, natural-log
    /// energies), trimmed to `(frames − 1) · hop` samples so that it lines up
    /// with the frames the spectrogram was computed from.
    pub fn synthesize(&self, gpu: &Gpu, log_mel: &[f32]) -> GpuResult<Vec<f32>> {
        let frames = log_mel.len() / self.config.mel_bands;
        let mut synthesis = Synthesis::new(gpu, &self.config, &self.istft, 1, frames)?;
        let mut activations = Activations::new(gpu, &self.config, 1, frames)?;

        let mel = gpu.upload(Shape::new(frames, self.config.mel_bands), &self.standardize(log_mel))?;

        self.network.forward(gpu, &mel, &mut activations)?;
        synthesis.forward(gpu, &self.istft, &activations.head)?;

        let signal = gpu.download(&synthesis.signal)?;
        let half_frame = self.config.fft_size / 2;

        Ok(signal[half_frame..signal.len() - half_frame].to_vec())
    }
}

/// The analysis settings that match `config`, for its window.
fn analyzer(config: &VocoderConfig) -> MelAnalyzer {
    MelAnalyzer::new(MelConfig {
        sample_rate: config.sample_rate,
        fft_size: config.fft_size,
        hop_size: config.hop_size,
        mel_bands: config.mel_bands,
        min_frequency: 0.0,
        max_frequency: config.sample_rate as f32 / 2.0,
    })
}
