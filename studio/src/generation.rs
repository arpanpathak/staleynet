//! Turning a browser request into a `staleynet sing` invocation.
//!
//! The prompt is passed as a separate process argument, never through a
//! shell, so no prompt text can be interpreted as a command.

use crate::library::STUDIO_SUBDIRECTORY;
use crate::runs::has_checkpoint;
use serde::Deserialize;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// Longest prompt accepted; the model reads at most 126 bytes anyway.
const MAX_PROMPT_CHARACTERS: usize = 400;

/// Longest lyrics accepted, in characters (a long song runs to about 3,000).
const MAX_LYRICS_CHARACTERS: usize = 8_000;

/// Allowed output length, in seconds.
const SECONDS_RANGE: std::ops::RangeInclusive<f32> = 2.0..=60.0;

/// Allowed classifier-free guidance scale (1 disables guidance).
const GUIDANCE_RANGE: std::ops::RangeInclusive<f32> = 1.0..=10.0;

/// Allowed sampling temperature (0 is greedy).
const TEMPERATURE_RANGE: std::ops::RangeInclusive<f32> = 0.0..=2.0;

/// Allowed top-k candidates per codebook.
const TOP_K_RANGE: std::ops::RangeInclusive<usize> = 1..=1024;

/// The generation form as the browser submits it.
#[derive(Deserialize)]
pub struct GenerationRequest {
    /// Text description of the music.
    pub prompt: String,

    /// Words to sing; only models with a lyrics field read them.
    #[serde(default)]
    pub lyrics: String,

    /// Run directory name under `runs/`.
    pub run: String,

    /// Output length in seconds.
    pub seconds: f32,

    /// Guidance scale.
    pub guidance: f32,

    /// Sampling temperature.
    pub temperature: f32,

    /// Top-k candidates.
    pub top_k: usize,

    /// Sampling seed.
    pub seed: u64,
}

/// Why a request was refused or failed.
pub enum GenerationError {
    /// The request itself is invalid (HTTP 400).
    Invalid(String),

    /// The generator ran and failed (HTTP 500).
    Failed(String),
}

impl GenerationRequest {
    /// Checks every field against its allowed range.
    fn validate(&self, project: &Path) -> Result<(), GenerationError> {
        let invalid = |message: String| Err(GenerationError::Invalid(message));

        if self.prompt.trim().is_empty() {
            return invalid("the prompt is empty".into());
        }
        if self.prompt.chars().count() > MAX_PROMPT_CHARACTERS {
            return invalid(format!("the prompt is longer than {MAX_PROMPT_CHARACTERS} characters"));
        }
        if self.lyrics.chars().count() > MAX_LYRICS_CHARACTERS {
            return invalid(format!("the lyrics are longer than {MAX_LYRICS_CHARACTERS} characters"));
        }

        let is_plain_name = self.run.chars().all(|character| character.is_ascii_alphanumeric() || "-_".contains(character));
        if self.run.is_empty() || !is_plain_name {
            return invalid("invalid run name".into());
        }
        if !has_checkpoint(&project.join("runs").join(&self.run)) {
            return invalid(format!("run `{}` has no checkpoint", self.run));
        }

        if !SECONDS_RANGE.contains(&self.seconds) {
            return invalid(format!("seconds must be in {SECONDS_RANGE:?}"));
        }

        if !GUIDANCE_RANGE.contains(&self.guidance) {
            return invalid(format!("guidance must be in {GUIDANCE_RANGE:?}"));
        }

        if !TEMPERATURE_RANGE.contains(&self.temperature) {
            return invalid(format!("temperature must be in {TEMPERATURE_RANGE:?}"));
        }

        if !TOP_K_RANGE.contains(&self.top_k) {
            return invalid(format!("top_k must be in {TOP_K_RANGE:?}"));
        }

        Ok(())
    }
}

/// A short, file-safe slug of `prompt` for file names.
pub(crate) fn slug(prompt: &str) -> String {
    const SLUG_WORDS: usize = 5;
    prompt
        .split_whitespace()
        .take(SLUG_WORDS)
        .map(|word| word.chars().filter(char::is_ascii_alphanumeric).collect::<String>().to_lowercase())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// Runs `staleynet sing` and returns the new clip's path relative to `samples/`.
pub fn generate(project: &Path, binary: &Path, request: &GenerationRequest) -> Result<String, GenerationError> {
    request.validate(project)?;

    let output_directory = project.join("samples").join(STUDIO_SUBDIRECTORY);
    std::fs::create_dir_all(&output_directory).map_err(|error| GenerationError::Failed(error.to_string()))?;

    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).map(|elapsed| elapsed.as_secs()).unwrap_or_default();
    let file_name = format!("{timestamp}-{}-{}.wav", request.run, slug(&request.prompt));
    let output = output_directory.join(&file_name);

    let result = Command::new(binary)
        .current_dir(project)
        .arg("sing")
        .args(["--run", &format!("runs/{}", request.run)])
        .args(["--prompt", &request.prompt])
        .args(["--lyrics", &request.lyrics])
        .args(["--seconds", &request.seconds.to_string()])
        .args(["--guidance", &request.guidance.to_string()])
        .args(["--temperature", &request.temperature.to_string()])
        .args(["--top-k", &request.top_k.to_string()])
        .args(["--seed", &request.seed.to_string()])
        .arg("--out")
        .arg(&output)
        .output()
        .map_err(|error| GenerationError::Failed(format!("could not start {}: {error}", binary.display())))?;

    if !result.status.success() {
        return Err(GenerationError::Failed(String::from_utf8_lossy(&result.stderr).into_owned()));
    }

    Ok(format!("{STUDIO_SUBDIRECTORY}/{file_name}"))
}
