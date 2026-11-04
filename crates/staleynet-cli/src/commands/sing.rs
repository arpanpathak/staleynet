//! `staleynet sing`: text prompt → WAV.

use super::open_gpu;
use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use serde::Serialize;
use staleynet_audio::griffin_lim::{GriffinLimConfig, griffin_lim};
use staleynet_audio::wav::write_mono_pcm16;
use staleynet_audio::{MelAnalyzer, MelConfig};
use staleynet_data::store::{CacheLayout, DatasetManifest, write_codes};
use staleynet_data::tokenizer::AudioTokenizer;
use staleynet_model::generate::{AttentionMode, GenerationConfig, GenerationStats, Guidance, Sampling, generate};
use staleynet_model::{Conditioning, checkpoint};
use staleynet_train::{RUN_TOKENIZER, preferred_checkpoint};
use staleynet_vocoder::Vocoder;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Command-line spelling of [`AttentionMode`].
#[derive(Clone, Copy, Debug, ValueEnum)]
enum AttentionArgument {
    /// Reuse cached keys and values.
    Cached,

    /// Recompute the whole context every step.
    Recompute,
}

/// Arguments of `staleynet sing`.
#[derive(Args)]
pub struct Arguments {
    /// Dataset cache, for the analysis settings and, when the run has no
    /// tokenizer of its own, the tokenizer.
    #[arg(long, default_value = "data/cache")]
    cache: PathBuf,

    /// Training run to load (its best checkpoint, else its latest).
    #[arg(long)]
    run: PathBuf,

    /// What to generate, in plain language.
    #[arg(long)]
    prompt: String,

    /// Words to sing, one sung line per text line. Needs a model trained
    /// with a lyrics field; other models ignore them.
    #[arg(long, default_value = "", conflicts_with = "lyrics_file")]
    lyrics: String,

    /// Read the lyrics from this file instead.
    #[arg(long)]
    lyrics_file: Option<PathBuf>,

    /// Length of the output.
    #[arg(long, default_value_t = 15.0)]
    seconds: f32,

    /// Output WAV path; a `.json` report is written beside it.
    #[arg(long, default_value = "out.wav")]
    out: PathBuf,

    /// Sampling temperature (0 = greedy).
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    /// Candidates kept per codebook when sampling (MusicGen uses 250).
    #[arg(long, default_value_t = 250)]
    top_k: usize,

    /// Classifier-free guidance scale (1 = off).
    #[arg(long, default_value_t = 3.0)]
    guidance: f32,

    /// KV caching.
    #[arg(long, value_enum, default_value_t = AttentionArgument::Cached)]
    attention: AttentionArgument,

    /// Sampling seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Also write the sampled codes (`[frames, codebooks]`, little-endian u16),
    /// for comparison with a reference.
    #[arg(long)]
    save_codes: Option<PathBuf>,

    /// Griffin-Lim iterations for the waveform, when no neural vocoder is given.
    #[arg(long, default_value_t = 64)]
    griffin_lim_iterations: usize,

    /// A trained neural vocoder run (`staleynet vocoder train`) to turn the
    /// spectrogram into samples instead of Griffin-Lim.
    #[arg(long)]
    vocoder: Option<PathBuf>,
}

/// What `sing` writes beside the WAV.
#[derive(Serialize)]
struct SingReport {
    /// The prompt.
    prompt: String,

    /// The lyrics, as given.
    lyrics: String,

    /// Whether the model has a lyrics field to read them with.
    model_reads_lyrics: bool,

    /// Checkpoint used.
    checkpoint: PathBuf,

    /// Optimiser step of that checkpoint.
    checkpoint_step: usize,

    /// Seconds of audio produced.
    audio_seconds: f32,

    /// Token sampling timing.
    generation: GenerationStats,

    /// The vocoder checkpoint used, or `None` for Griffin-Lim.
    vocoder: Option<PathBuf>,

    /// Seconds spent turning the spectrogram into samples.
    vocoder_seconds: f64,

    /// (sampling + vocoding time) / audio duration; below 1 is faster than real time.
    real_time_factor: f64,
}

/// Generates audio for the prompt and writes the WAV file and its report.
pub fn run(arguments: Arguments) -> Result<()> {
    let gpu = open_gpu()?;
    let layout = CacheLayout::new(&arguments.cache);
    let mel = MelConfig::from(DatasetManifest::load(&layout)?.mel);
    let tokenizer = run_tokenizer(&arguments.run, &layout)?;

    let checkpoint_directory = preferred_checkpoint(&arguments.run);
    let (model, metadata) = checkpoint::load(&gpu, &checkpoint_directory, None)?;

    let lyrics = arguments.lyrics()?;
    let model_reads_lyrics = model.config.lyrics_bytes > 0;
    if !lyrics.trim().is_empty() && !model_reads_lyrics {
        eprintln!("warning: this model has no lyrics field; the lyrics are ignored");
    }

    let frames = (arguments.seconds * mel.frames_per_second() / tokenizer.config.frames_per_step as f32).ceil() as usize;
    eprintln!("{} (step {}): generating {frames} frames for {:.1}s", checkpoint_directory.display(), metadata.step, arguments.seconds);

    let prompt = Conditioning { caption: &arguments.prompt, lyrics: &lyrics };
    let generation = generate(&gpu, &model, prompt, &arguments.generation_config(frames))?;

    if let Some(path) = &arguments.save_codes {
        write_codes(path, &generation.codes)?;
    }

    let log_mel = tokenizer.decode(&generation.codes, tokenizer.config.codebooks);
    let vocoder_checkpoint = arguments.vocoder.as_deref().map(vocoder_checkpoint);
    let vocoder_started = Instant::now();
    let waveform = match &vocoder_checkpoint {
        Some(checkpoint) => Vocoder::load(&gpu, checkpoint)?.0.synthesize(&gpu, &log_mel)?,
        None => {
            let griffin = GriffinLimConfig { iterations: arguments.griffin_lim_iterations, seed: arguments.seed };
            griffin_lim(&MelAnalyzer::new(mel), &log_mel, griffin)
        }
    };
    let vocoder_seconds = vocoder_started.elapsed().as_secs_f64();

    write_mono_pcm16(&arguments.out, &waveform, mel.sample_rate)?;

    let audio_seconds = waveform.len() as f32 / mel.sample_rate as f32;
    let report = SingReport {
        prompt: arguments.prompt.clone(),
        lyrics,
        model_reads_lyrics,
        checkpoint: checkpoint_directory,
        checkpoint_step: metadata.step,
        vocoder: vocoder_checkpoint,
        audio_seconds,
        generation: generation.stats,
        vocoder_seconds,
        real_time_factor: (generation.stats.seconds + vocoder_seconds) / audio_seconds as f64,
    };
    std::fs::write(arguments.out.with_extension("json"), serde_json::to_string_pretty(&report)?)?;

    eprintln!(
        "wrote {} ({audio_seconds:.1}s): sampling {:.1}s, vocoder {vocoder_seconds:.1}s, RTF {:.2}",
        arguments.out.display(),
        report.generation.seconds,
        report.real_time_factor
    );
    Ok(())
}

/// The checkpoint of a vocoder run to synthesize with.
fn vocoder_checkpoint(run: &Path) -> PathBuf {
    staleynet_vocoder::train::preferred_checkpoint(run)
}

/// The tokenizer saved with the run, or the cache's when the run has none.
fn run_tokenizer(run: &Path, layout: &CacheLayout) -> Result<AudioTokenizer> {
    let saved_with_run = run.join(RUN_TOKENIZER);
    let path = if saved_with_run.is_file() { saved_with_run } else { layout.tokenizer() };

    AudioTokenizer::load(&path)
}

impl Arguments {
    /// The lyrics from `--lyrics-file` when given, otherwise from `--lyrics`.
    fn lyrics(&self) -> Result<String> {
        let Some(path) = &self.lyrics_file else { return Ok(self.lyrics.clone()) };

        std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))
    }

    /// Sampling, guidance and attention settings for `frames` frames.
    /// A temperature of 0 means greedy decoding, and a guidance scale of 1
    /// means no guidance.
    fn generation_config(&self, frames: usize) -> GenerationConfig {
        let sampling =
            if self.temperature <= 0.0 { Sampling::Greedy } else { Sampling::TopK { temperature: self.temperature, k: self.top_k } };
        let guidance = if self.guidance == 1.0 { Guidance::Off } else { Guidance::ClassifierFree { scale: self.guidance } };
        let attention = match self.attention {
            AttentionArgument::Cached => AttentionMode::KvCache,
            AttentionArgument::Recompute => AttentionMode::Recompute,
        };

        GenerationConfig { frames, sampling, guidance, attention, seed: self.seed }
    }
}
