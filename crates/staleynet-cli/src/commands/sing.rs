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
use std::path::PathBuf;
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

    /// Griffin-Lim iterations for the waveform.
    #[arg(long, default_value_t = 64)]
    griffin_lim_iterations: usize,
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

    /// Seconds spent in Griffin-Lim.
    vocoder_seconds: f64,

    /// (sampling + vocoding time) / audio duration; below 1 is faster than real time.
    real_time_factor: f64,
}

pub fn run(arguments: Arguments) -> Result<()> {
    let gpu = open_gpu()?;
    let layout = CacheLayout::new(&arguments.cache);
    let manifest = DatasetManifest::load(&layout)?;
    let run_tokenizer = arguments.run.join(RUN_TOKENIZER);
    let tokenizer_path = if run_tokenizer.is_file() { run_tokenizer } else { layout.tokenizer() };
    let tokenizer = AudioTokenizer::load(&tokenizer_path)?;
    let checkpoint_directory = preferred_checkpoint(&arguments.run);
    let (model, metadata) = checkpoint::load(&gpu, &checkpoint_directory, None)?;

    let mel = MelConfig::from(manifest.mel);
    let frames = (arguments.seconds * mel.frames_per_second() / tokenizer.config.frames_per_step as f32).ceil() as usize;
    let sampling = if arguments.temperature <= 0.0 {
        Sampling::Greedy
    } else {
        Sampling::TopK { temperature: arguments.temperature, k: arguments.top_k }
    };
    let guidance = if arguments.guidance == 1.0 { Guidance::Off } else { Guidance::ClassifierFree { scale: arguments.guidance } };
    let attention = match arguments.attention {
        AttentionArgument::Cached => AttentionMode::KvCache,
        AttentionArgument::Recompute => AttentionMode::Recompute,
    };

    let lyrics = match &arguments.lyrics_file {
        Some(path) => std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?,
        None => arguments.lyrics.clone(),
    };
    let model_reads_lyrics = model.config.lyrics_bytes > 0;
    if !lyrics.trim().is_empty() && !model_reads_lyrics {
        eprintln!("warning: this model has no lyrics field; the lyrics are ignored");
    }

    eprintln!("{} (step {}): generating {frames} frames for {:.1}s", checkpoint_directory.display(), metadata.step, arguments.seconds);
    let prompt = Conditioning { caption: &arguments.prompt, lyrics: &lyrics };
    let generation = generate(&gpu, &model, prompt, &GenerationConfig { frames, sampling, guidance, attention, seed: arguments.seed })?;

    if let Some(path) = &arguments.save_codes {
        write_codes(path, &generation.codes)?;
    }

    let vocoder_started = Instant::now();
    let analyzer = MelAnalyzer::new(mel);
    let log_mel = tokenizer.decode(&generation.codes, tokenizer.config.codebooks);
    let waveform =
        griffin_lim(&analyzer, &log_mel, GriffinLimConfig { iterations: arguments.griffin_lim_iterations, seed: arguments.seed });
    let vocoder_seconds = vocoder_started.elapsed().as_secs_f64();
    write_mono_pcm16(&arguments.out, &waveform, mel.sample_rate)?;

    let audio_seconds = waveform.len() as f32 / mel.sample_rate as f32;
    let report = SingReport {
        prompt: arguments.prompt,
        lyrics,
        model_reads_lyrics,
        checkpoint: checkpoint_directory,
        checkpoint_step: metadata.step,
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
