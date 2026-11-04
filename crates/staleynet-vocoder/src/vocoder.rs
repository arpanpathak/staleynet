//! The whole vocoder: mel spectrogram in, waveform out, and the training pass
//! that compares the waveform with a recording.
//!
//! The network predicts a log-magnitude and a phase for every frequency of
//! every frame, these become a complex spectrum, and the inverse STFT turns
//! the spectrum into samples. Its input is a log-mel spectrogram standardized
//! band by band with the statistics of the tokenizer, so that a spectrogram
//! decoded from tokens and one computed from a recording look alike to it.

use crate::config::VocoderConfig;
use crate::istft::InverseStft;
use crate::loss::{Resolution, SpectralLoss};
use crate::network::{Activations, Network};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use staleynet_audio::{MelAnalyzer, MelConfig};
use staleynet_cuda::{Gpu, GpuResult, Shape, Tensor};
use staleynet_model::parameter::{Parameter, Parameters};
use staleynet_model::random::Random;
use std::fs;
use std::io::Write;
use std::path::Path;

/// Name of the metadata file inside a vocoder checkpoint directory.
const METADATA_FILE: &str = "vocoder.json";

/// Name of the tensor file inside a vocoder checkpoint directory.
const TENSOR_FILE: &str = "tensors.bin";

/// Bytes per stored value.
const BYTES_PER_VALUE: usize = std::mem::size_of::<f32>();

/// A trained (or freshly initialized) vocoder.
pub struct Vocoder {
    /// Sizes and settings.
    pub config: VocoderConfig,

    /// The network that predicts the spectrum.
    pub network: Network,

    /// The inverse STFT.
    istft: InverseStft,

    /// Mean log energy of each mel band, subtracted from the input.
    mel_mean: Vec<f32>,

    /// Standard deviation of each mel band, dividing the input.
    mel_std: Vec<f32>,
}

/// Contents of `vocoder.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Metadata {
    /// Sizes and settings.
    config: VocoderConfig,

    /// Input standardization: mean per band.
    mel_mean: Vec<f32>,

    /// Input standardization: standard deviation per band.
    mel_std: Vec<f32>,

    /// Stored parameters in file order: name and number of values.
    tensors: Vec<(String, usize)>,

    /// Training steps taken when saved.
    step: usize,
}

impl Vocoder {
    /// A freshly initialized vocoder whose input is standardized with
    /// `mel_mean` and `mel_std` (one value per mel band).
    pub fn new(gpu: &Gpu, config: VocoderConfig, mel_mean: Vec<f32>, mel_std: Vec<f32>, seed: u64) -> GpuResult<Self> {
        assert!(mel_mean.len() == config.mel_bands && mel_std.len() == config.mel_bands, "one statistic per mel band");
        let network = Network::new(gpu, &config, &mut Random::new(seed))?;
        let istft = InverseStft::new(gpu, &config, &analyzer(&config).window)?;

        Ok(Self { config, network, istft, mel_mean, mel_std })
    }

    /// `log_mel` (`[frames, mel_bands]`, natural-log energies) standardized for the network.
    pub fn standardize(&self, log_mel: &[f32]) -> Vec<f32> {
        let bands = self.config.mel_bands;

        log_mel.iter().enumerate().map(|(index, value)| (value - self.mel_mean[index % bands]) / self.mel_std[index % bands]).collect()
    }

    /// The waveform of a log-mel spectrogram (`[frames, mel_bands]`, natural-log
    /// energies), trimmed to `(frames − 1) · hop` samples so that it lines up
    /// with the frames the spectrogram was computed from.
    pub fn synthesize(&self, gpu: &Gpu, log_mel: &[f32]) -> GpuResult<Vec<f32>> {
        let frames = log_mel.len() / self.config.mel_bands;
        let mut spectrum = Spectrum::new(gpu, &self.config, &self.istft, 1, frames)?;
        let mut activations = Activations::new(gpu, &self.config, 1, frames)?;
        let mel = gpu.upload(Shape::new(frames, self.config.mel_bands), &self.standardize(log_mel))?;

        self.network.forward(gpu, &mel, &mut activations)?;
        spectrum.forward(gpu, self, &activations.head)?;
        let signal = gpu.download(&spectrum.signal)?;

        let half = self.config.fft_size / 2;
        Ok(signal[half..signal.len() - half].to_vec())
    }

    /// Saves the network and its standardization into `directory`.
    pub fn save(&mut self, gpu: &Gpu, directory: &Path, step: usize) -> Result<()> {
        fs::create_dir_all(directory)?;
        let mut tensors = Vec::new();
        let mut values = Vec::new();
        let mut failure = Ok(());

        self.network.visit_parameters(&mut |parameter| {
            if failure.is_err() {
                return;
            }
            match gpu.download(&parameter.value) {
                Ok(host) => {
                    tensors.push((parameter.name.clone(), host.len()));
                    values.extend(host);
                }
                Err(error) => failure = Err(error),
            }
        });
        failure?;

        let mut file = std::io::BufWriter::new(fs::File::create(directory.join(TENSOR_FILE))?);
        values.iter().try_for_each(|value| file.write_all(&value.to_le_bytes()))?;
        file.flush()?;

        let metadata =
            Metadata { config: self.config.clone(), mel_mean: self.mel_mean.clone(), mel_std: self.mel_std.clone(), tensors, step };
        fs::write(directory.join(METADATA_FILE), serde_json::to_string_pretty(&metadata)?)?;
        Ok(())
    }

    /// Loads a vocoder saved by [`Vocoder::save`]; returns it with its step.
    pub fn load(gpu: &Gpu, directory: &Path) -> Result<(Self, usize)> {
        let path = directory.join(METADATA_FILE);
        let metadata: Metadata = serde_json::from_str(&fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?)?;
        let bytes = fs::read(directory.join(TENSOR_FILE))?;
        let values: Vec<f32> =
            bytes.chunks_exact(BYTES_PER_VALUE).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();

        let mut vocoder = Self::new(gpu, metadata.config, metadata.mel_mean, metadata.mel_std, 0)?;
        let mut remaining = values.as_slice();
        let mut entries = metadata.tensors.iter();
        let mut failure: Result<()> = Ok(());

        vocoder.network.visit_parameters(&mut |parameter| {
            if failure.is_err() {
                return;
            }
            failure = restore(gpu, parameter, entries.next(), &mut remaining);
        });
        failure?;

        ensure!(remaining.is_empty(), "{} has more values than the network", directory.display());
        Ok((vocoder, metadata.step))
    }
}

/// Uploads the next stored tensor into `parameter`, after checking that its
/// name and size match.
fn restore(gpu: &Gpu, parameter: &mut Parameter, entry: Option<&(String, usize)>, remaining: &mut &[f32]) -> Result<()> {
    let (name, len) = entry.context("the checkpoint has fewer tensors than the network")?;
    ensure!(*name == parameter.name && *len == parameter.value.len(), "checkpoint tensor {name} ({len}) does not match {}", parameter.name);
    ensure!(remaining.len() >= *len, "tensors.bin ends early");

    let (values, rest) = remaining.split_at(*len);
    gpu.copy_from_host(&mut parameter.value, values)?;
    *remaining = rest;
    Ok(())
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

/// From the network's head to samples, and back: the predicted spectrum,
/// the inverse STFT, and their gradients, for `sequences` sequences of `time` frames.
struct Spectrum {
    /// Frames per sequence.
    time: usize,

    /// Real and imaginary parts of the predicted spectrum, `[rows, bins]`.
    real: Tensor,

    /// See `real`.
    imaginary: Tensor,

    /// Synthesized frames before overlap-add, `[rows, fft_size]`.
    frames: Tensor,

    /// One over the summed squared window at each sample of a sequence.
    normalization: Tensor,

    /// The synthesized signals, `[sequences, padded samples]`.
    signal: Tensor,

    /// Gradient of `signal`.
    signal_grad: Tensor,

    /// Gradient of `frames`.
    frames_grad: Tensor,

    /// Gradients of the two parts of the spectrum.
    real_grad: Tensor,

    /// See `real_grad`.
    imaginary_grad: Tensor,

    /// Gradient of the head output, `[rows, 2 · bins]`.
    head_grad: Tensor,
}

impl Spectrum {
    fn new(gpu: &Gpu, config: &VocoderConfig, istft: &InverseStft, sequences: usize, time: usize) -> GpuResult<Self> {
        let rows = sequences * time;
        let spectrum = Shape::new(rows, config.bins());
        let length = config.padded_samples(time);

        Ok(Self {
            time,
            real: gpu.zeros(spectrum)?,
            imaginary: gpu.zeros(spectrum)?,
            frames: gpu.zeros(Shape::new(rows, config.fft_size))?,
            normalization: gpu.upload(Shape::vector(length), &istft.normalization(time))?,
            signal: gpu.zeros(Shape::new(sequences, length))?,
            signal_grad: gpu.zeros(Shape::new(sequences, length))?,
            frames_grad: gpu.zeros(Shape::new(rows, config.fft_size))?,
            real_grad: gpu.zeros(spectrum)?,
            imaginary_grad: gpu.zeros(spectrum)?,
            head_grad: gpu.zeros(Shape::new(rows, 2 * config.bins()))?,
        })
    }

    /// Spectrum and samples from the head output.
    fn forward(&mut self, gpu: &Gpu, vocoder: &Vocoder, head: &Tensor) -> GpuResult {
        gpu.polar_spectrum(head, &mut self.real, &mut self.imaginary, vocoder.config.max_log_magnitude)?;
        vocoder.istft.forward(gpu, &self.real, &self.imaginary, &self.normalization, self.time, &mut self.frames, &mut self.signal)
    }

    /// The gradient of the head output, from `signal_grad`.
    fn backward(&mut self, gpu: &Gpu, vocoder: &Vocoder, head: &Tensor) -> GpuResult {
        vocoder.istft.backward(
            gpu,
            &self.signal_grad,
            &self.normalization,
            self.time,
            &mut self.frames_grad,
            &mut self.real_grad,
            &mut self.imaginary_grad,
        )?;
        gpu.polar_spectrum_backward(
            head,
            &self.real,
            &self.imaginary,
            &self.real_grad,
            &self.imaginary_grad,
            &mut self.head_grad,
            vocoder.config.max_log_magnitude,
        )
    }
}

/// Everything one training step needs for a batch of `sequences` segments of
/// `time` frames: the network's activations, the spectrum and its gradients,
/// and one spectral loss per resolution.
pub struct TrainingPass {
    /// Network activations and gradient scratch.
    activations: Activations,

    /// Spectrum, signal and their gradients.
    spectrum: Spectrum,

    /// The spectral losses.
    losses: Vec<SpectralLoss>,

    /// Standardized input mels, `[rows, mel_bands]`.
    mel: Tensor,

    /// Target signals, `[sequences, padded samples]`.
    target: Tensor,
}

impl TrainingPass {
    /// Buffers for `sequences` segments of `time` frames, compared at `resolutions`.
    pub fn new(gpu: &Gpu, vocoder: &Vocoder, sequences: usize, time: usize, resolutions: &[Resolution]) -> GpuResult<Self> {
        let config = &vocoder.config;
        let length = config.padded_samples(time);
        let losses = resolutions
            .iter()
            .map(|resolution| SpectralLoss::new(gpu, *resolution, config.sample_rate, config.mel_bands, sequences, length))
            .collect::<GpuResult<Vec<_>>>()?;

        Ok(Self {
            activations: Activations::new(gpu, config, sequences, time)?,
            spectrum: Spectrum::new(gpu, config, &vocoder.istft, sequences, time)?,
            losses,
            mel: gpu.zeros(Shape::new(sequences * time, config.mel_bands))?,
            target: gpu.zeros(Shape::new(sequences, length))?,
        })
    }

    /// Samples per target signal: `(time − 1) · hop + fft_size`.
    pub fn signal_length(&self) -> usize {
        self.target.shape().cols
    }

    /// One forward and backward pass: synthesizes the batch from `mel`
    /// (standardized, `[sequences · time, mel_bands]`), compares it with
    /// `target` (`[sequences, signal length]`), and adds every parameter's
    /// gradient into the network. Gradients are not zeroed here.
    pub fn run(&mut self, gpu: &Gpu, vocoder: &mut Vocoder, mel: &[f32], target: &[f32]) -> GpuResult {
        gpu.copy_from_host(&mut self.mel, mel)?;
        gpu.copy_from_host(&mut self.target, target)?;

        vocoder.network.forward(gpu, &self.mel, &mut self.activations)?;
        self.spectrum.forward(gpu, vocoder, &self.activations.head)?;

        gpu.fill_zero(&mut self.spectrum.signal_grad)?;
        for loss in &mut self.losses {
            loss.set_target(gpu, &self.target)?;
            loss.forward(gpu, &self.spectrum.signal)?;
            loss.backward(gpu, &mut self.spectrum.signal_grad)?;
        }

        self.spectrum.backward(gpu, vocoder, &self.activations.head)?;
        vocoder.network.backward(gpu, &self.spectrum.head_grad, &mut self.activations)
    }

    /// The total loss of the last [`TrainingPass::run`], and each part.
    pub fn loss(&self, gpu: &Gpu) -> GpuResult<(f64, Vec<f64>)> {
        let parts = self.losses.iter().map(|loss| loss.value(gpu)).collect::<GpuResult<Vec<_>>>()?;

        Ok((parts.iter().sum(), parts))
    }
}
