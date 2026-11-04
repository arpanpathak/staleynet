//! Training segments: a window of a spectrogram decoded from tokens, and the
//! stretch of recording it came from.
//!
//! The vocoder only ever hears spectrograms that the language model's tokens
//! decode to, so that is what it trains on. Each track's codes are decoded with
//! the tokenizer, a window of frames is cut out, and the matching samples are
//! read from the waveform cache. The vocoder thereby learns phase and also
//! part of the detail the tokens lost.
//!
//! Frame `t` of a log-mel spectrogram is centered on sample `t · hop`, so the
//! samples behind frames `t₀ … t₀ + T − 1`, with half a frame on each side,
//! start at sample `t₀ · hop − fft_size / 2` and number
//! `(T − 1) · hop + fft_size`: the padded signal the vocoder synthesizes.

use anyhow::{Context, Result, ensure};
use half::f16;
use rayon::prelude::*;
use staleynet_data::catalog::Split;
use staleynet_data::store::{CacheLayout, DatasetManifest, Normalization, PreparedTrack, read_codes, read_f16};
use staleynet_data::tokenizer::AudioTokenizer;
use staleynet_model::random::Random;
use staleynet_model::vocoder::VocoderConfig;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, sync_channel};

/// Bytes of one stored half-precision sample.
const F16_BYTES: usize = 2;

/// Bytes of one stored code.
const CODE_BYTES: usize = 2;

/// A batch ready for [`staleynet_model::vocoder::TrainingPass::run`].
pub struct Batch {
    /// Standardized log-mel frames, `[sequences · frames, mel_bands]`.
    pub mel: Vec<f32>,

    /// Target samples, `[sequences, padded samples]`.
    pub target: Vec<f32>,
}

/// How many tracks a sampler can cut segments from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrackCounts {
    /// Training tracks.
    pub train: usize,

    /// Held-out tracks.
    pub held_out: usize,
}

/// One whole track for listening: its decoded spectrogram and its recording.
pub struct WholeTrack {
    /// Log-mel frames decoded from the track's codes, `[frames, mel_bands]`.
    pub log_mel: Vec<f32>,

    /// The recorded samples behind those frames.
    pub recording: Vec<f32>,
}

/// One track that is long enough to cut segments from.
struct Clip {
    /// Catalog id.
    id: String,

    /// Last frame a segment may start at; the first is [`SegmentSampler::first_start`].
    last_start: usize,
}

/// Where one segment of a batch comes from.
struct Pick {
    /// Index of the clip.
    clip: usize,

    /// First frame of the segment.
    start: usize,
}

/// Cuts training segments out of a tokenized cache that also holds waveforms.
pub struct SegmentSampler {
    /// The cache.
    layout: CacheLayout,

    /// Decodes codes into log-mel frames, and standardizes them.
    tokenizer: AudioTokenizer,

    /// Residual levels used when decoding.
    levels: usize,

    /// Frames per segment.
    frames: usize,

    /// Samples between frames.
    hop_size: usize,

    /// Samples per frame.
    fft_size: usize,

    /// Training tracks.
    train: Vec<Clip>,

    /// Held-out tracks.
    validation: Vec<Clip>,
}

impl SegmentSampler {
    /// A sampler of `frames`-frame segments from `layout`, decoding with the
    /// first `levels` residual levels (all of them when `None`).
    pub fn new(layout: &CacheLayout, config: &VocoderConfig, frames: usize, levels: Option<usize>) -> Result<Self> {
        let manifest = DatasetManifest::load(layout)?;
        let tokenizer = AudioTokenizer::load(&layout.tokenizer())?;
        let mel = &manifest.mel;

        ensure!(mel.sample_rate == config.sample_rate, "cache sample rate {} ≠ vocoder {}", mel.sample_rate, config.sample_rate);
        ensure!(mel.hop_size == config.hop_size && mel.fft_size == config.fft_size, "cache framing differs from the vocoder's");
        ensure!(mel.mel_bands == config.mel_bands, "cache has {} mel bands, the vocoder {}", mel.mel_bands, config.mel_bands);

        let mut sampler = Self {
            layout: layout.clone(),
            levels: levels.unwrap_or(tokenizer.config.levels()),
            tokenizer,
            frames,
            hop_size: config.hop_size,
            fft_size: config.fft_size,
            train: Vec::new(),
            validation: Vec::new(),
        };

        for entry in &manifest.tracks {
            let Some(clip) = sampler.clip(entry) else { continue };

            match entry.track.split {
                Split::Train => sampler.train.push(clip),
                Split::Validation => sampler.validation.push(clip),
            }
        }

        ensure!(!sampler.train.is_empty(), "no training track in {} has codes, a waveform and {frames} frames", layout.root.display());
        Ok(sampler)
    }

    /// Training and held-out tracks usable for segments.
    pub fn track_counts(&self) -> TrackCounts {
        TrackCounts { train: self.train.len(), held_out: self.validation.len() }
    }

    /// The tokenizer's standardization, which the vocoder must share.
    pub fn normalization(&self) -> &Normalization {
        &self.tokenizer.normalization
    }

    /// A batch of training segments, the same for the same `seed` and `step`.
    pub fn training_batch(&self, sequences: usize, seed: u64, step: usize) -> Result<Batch> {
        self.batch(&self.train, sequences, &mut Random::derived(seed, step as u64))
    }

    /// `count` fixed batches of held-out segments (training segments when
    /// the cache holds no held-out track).
    pub fn validation_batches(&self, sequences: usize, count: usize, seed: u64) -> Result<Vec<Batch>> {
        let mut random = Random::new(seed);

        (0..count)
            .map(|_| self.batch(self.held_out(), sequences, &mut random))
            .collect()
    }

    /// Ids of the first `count` held-out tracks, for listening tests.
    pub fn listening_tracks(&self, count: usize) -> Vec<String> {
        self.held_out()
            .iter()
            .take(count)
            .map(|clip| clip.id.clone())
            .collect()
    }

    /// The log-mel spectrogram of track `id` decoded from its codes, and its
    /// recorded samples, each cut to at most `frames` frames.
    pub fn whole_track(&self, id: &str, frames: usize) -> Result<WholeTrack> {
        let bands = self.tokenizer.mel_bands;
        let codes = read_codes(&self.layout.codes(id))?;

        let mut log_mel = self.tokenizer.decode(&codes, self.levels);
        let kept_frames = frames.min(log_mel.len() / bands);
        log_mel.truncate(kept_frames * bands);

        let mut recording = read_f16(&self.layout.waveform(id))?;
        recording.truncate(kept_frames.saturating_sub(1) * self.hop_size);

        Ok(WholeTrack { log_mel, recording })
    }

    /// Training batches for steps `first_step, first_step + 1, …`, prepared on
    /// a background thread `depth` batches ahead of the GPU.
    pub fn prefetch(self: Arc<Self>, sequences: usize, seed: u64, first_step: usize, depth: usize) -> Receiver<Result<Batch>> {
        let (sender, receiver) = sync_channel(depth);

        std::thread::spawn(move || {
            for step in first_step.. {
                if sender
                    .send(self.training_batch(sequences, seed, step))
                    .is_err()
                {
                    return;
                }
            }
        });

        receiver
    }

    /// Held-out clips, or training clips when the cache holds none.
    fn held_out(&self) -> &[Clip] {
        if self.validation.is_empty() { &self.train } else { &self.validation }
    }

    /// First frame a segment may start at: its first sample, half a frame
    /// before the frame's center, must not fall before the recording.
    fn first_start(&self) -> usize {
        (self.fft_size / 2).div_ceil(self.hop_size)
    }

    /// Samples behind a segment, padding included.
    fn segment_samples(&self) -> usize {
        (self.frames - 1) * self.hop_size + self.fft_size
    }

    /// The track as a [`Clip`], when its codes and waveform exist and hold a whole segment.
    fn clip(&self, entry: &PreparedTrack) -> Option<Clip> {
        let id = &entry.track.id;
        let code_bytes = std::fs::metadata(self.layout.codes(id)).ok()?.len() as usize;
        let sample_bytes = std::fs::metadata(self.layout.waveform(id)).ok()?.len() as usize;

        let config = &self.tokenizer.config;
        let mel_frames = code_bytes / CODE_BYTES / config.codebooks * config.frames_per_step;
        let samples = sample_bytes / F16_BYTES;

        let last_by_frames = mel_frames.checked_sub(self.frames)?;
        let last_by_samples = (samples + self.fft_size / 2).checked_sub(self.segment_samples())? / self.hop_size;
        let last_start = last_by_frames.min(last_by_samples);

        (last_start >= self.first_start()).then(|| Clip { id: id.clone(), last_start })
    }

    /// Fills `mel` and `target` with the segment of `clip` that starts at frame `start`.
    fn segment(&self, clip: &Clip, start: usize, mel: &mut [f32], target: &mut [f32]) -> Result<()> {
        self.decoded_frames(clip, start, mel)?;
        self.recorded_samples(clip, start, target)
    }

    /// The standardized log-mel frames of the segment, decoded from the clip's codes.
    fn decoded_frames(&self, clip: &Clip, start: usize, mel: &mut [f32]) -> Result<()> {
        let codes = read_codes(&self.layout.codes(&clip.id))?;
        let decoded = self.tokenizer.decode(&codes, self.levels);

        let bands = self.tokenizer.mel_bands;
        let frames = &decoded[start * bands..(start + self.frames) * bands];
        let normalization = &self.tokenizer.normalization;

        for (index, (value, decoded_value)) in mel.iter_mut().zip(frames).enumerate() {
            let band = index % bands;
            *value = (decoded_value - normalization.mean[band]) / normalization.std[band];
        }

        Ok(())
    }

    /// The recorded samples behind the segment, read directly from the waveform cache.
    fn recorded_samples(&self, clip: &Clip, start: usize, target: &mut [f32]) -> Result<()> {
        let first_sample = start * self.hop_size - self.fft_size / 2;
        let path = self.layout.waveform(&clip.id);
        let mut bytes = vec![0u8; target.len() * F16_BYTES];

        let mut file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        file.seek(SeekFrom::Start((first_sample * F16_BYTES) as u64))?;
        file.read_exact(&mut bytes)
            .with_context(|| format!("read {}", path.display()))?;

        for (sample, pair) in target.iter_mut().zip(bytes.chunks_exact(F16_BYTES)) {
            *sample = f16::from_le_bytes([pair[0], pair[1]]).to_f32();
        }

        Ok(())
    }

    /// `sequences` segments from random places in `clips`.
    fn batch(&self, clips: &[Clip], sequences: usize, random: &mut Random) -> Result<Batch> {
        let bands = self.tokenizer.mel_bands;
        let length = self.segment_samples();
        let first = self.first_start();

        let picks: Vec<Pick> = (0..sequences)
            .map(|_| {
                let clip = random.below(clips.len());
                let start = first + random.below(clips[clip].last_start - first + 1);
                Pick { clip, start }
            })
            .collect();

        let mut batch = Batch { mel: vec![0.0; sequences * self.frames * bands], target: vec![0.0; sequences * length] };
        batch
            .mel
            .par_chunks_mut(self.frames * bands)
            .zip(batch.target.par_chunks_mut(length))
            .zip(&picks)
            .try_for_each(|((mel, target), pick)| self.segment(&clips[pick.clip], pick.start, mel, target))?;

        Ok(batch)
    }
}
