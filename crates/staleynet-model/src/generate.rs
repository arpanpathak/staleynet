//! Prompt → audio codes by autoregressive sampling.
//!
//! Lyrics longer than the model's lyrics field are cut into consecutive
//! windows at line breaks. The output is divided evenly between the windows,
//! and when one window ends the next replaces it in the prompt while the most
//! recent audio stays in context, so the voice carries on into the next lines.

use crate::decode::Decoder;
use crate::gpt::Gpt;
use crate::random::Random;
use crate::vocabulary::{Conditioning, DelayPattern};
use staleynet_cuda::{EMPTY_TOKEN, Gpu, GpuResult};
use std::time::Instant;

/// How a code is drawn from a codebook's logits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sampling {
    /// Always the most likely code.
    Greedy,

    /// Sample from the `k` most likely codes after dividing logits by
    /// `temperature` (Fan, Lewis & Dauphin, 2018, <https://arxiv.org/abs/1805.04833>).
    TopK {
        /// Softmax temperature; below 1 sharpens, above 1 flattens.
        temperature: f32,

        /// Number of candidates kept.
        k: usize,
    },
}

/// Whether and how strongly to push samples towards the prompt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Guidance {
    /// Use the prompted model alone.
    Off,

    /// Classifier-free guidance (Ho & Salimans, 2022, <https://arxiv.org/abs/2207.12598>):
    /// `logits = unconditional + scale · (conditional − unconditional)`.
    /// Needs a model trained with randomly dropped prompts.
    ClassifierFree {
        /// Guidance strength; 1 equals [`Guidance::Off`], 3 is MusicGen's default.
        scale: f32,
    },
}

/// Whether past keys and values are cached between steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionMode {
    /// Feed only the newest step; reuse cached keys and values.
    KvCache,

    /// Re-run the whole visible context every step (the baseline the cache is
    /// measured against).
    Recompute,
}

/// Everything that controls one generation.
#[derive(Clone, Copy, Debug)]
pub struct GenerationConfig {
    /// Audio frames (tokenizer steps) to produce.
    pub frames: usize,

    /// Code selection rule.
    pub sampling: Sampling,

    /// Prompt adherence.
    pub guidance: Guidance,

    /// KV caching on or off.
    pub attention: AttentionMode,

    /// Seed of the sampling noise.
    pub seed: u64,
}

/// Timing of one generation.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct GenerationStats {
    /// Delayed steps sampled.
    pub steps: usize,

    /// Forward calls made (two per step under guidance).
    pub model_calls: usize,

    /// Wall-clock seconds spent sampling.
    pub seconds: f64,
}

/// Result of [`generate`].
pub struct Generation {
    /// Undelayed codes, `[frames, codebooks]`.
    pub codes: Vec<u32>,

    /// Timing.
    pub stats: GenerationStats,
}

/// One conditioning (the prompt, or the empty prompt) with its own caches.
struct ConditionedStream {
    /// Token slots of the prompt prefix.
    prompt_slots: Vec<u32>,

    /// KV caches for this conditioning.
    decoder: Decoder,
}

impl ConditionedStream {
    /// Swaps the prompt. The caches are dropped, so the next call rebuilds
    /// them from the new prompt and the most recent audio.
    fn condition_on(&mut self, prompt_slots: Vec<u32>) {
        self.prompt_slots = prompt_slots;
        self.decoder.reset();
    }

    /// Logits for the next step, given every audio step generated so far.
    fn next_logits(&mut self, gpu: &Gpu, model: &Gpt, history: &[u32], attention: AttentionMode) -> GpuResult<Vec<f32>> {
        let codebooks = model.config.codebooks;
        let can_extend = matches!(attention, AttentionMode::KvCache) && self.decoder.cached_positions() > 0 && self.decoder.remaining() > 0;

        if can_extend {
            let newest_step = &history[history.len() - codebooks..];
            return self.decoder.feed(gpu, model, newest_step);
        }

        // Rebuild from the prompt plus as much recent history as fits. With a
        // cache, keep only half the window so rebuilds stay rare.
        let audio_room = model.config.sequence_capacity() - self.prompt_slots.len() / codebooks;
        let window = match attention {
            AttentionMode::KvCache => audio_room / 2,
            AttentionMode::Recompute => audio_room,
        };
        let history_steps = history.len() / codebooks;
        let kept_steps = history_steps.min(window);

        let mut slots = self.prompt_slots.clone();
        slots.extend_from_slice(&history[(history_steps - kept_steps) * codebooks..]);

        self.decoder.reset();
        if attention == AttentionMode::Recompute {
            self.decoder.release_scratch_except(slots.len() / codebooks);
        }
        self.decoder.feed(gpu, model, &slots)
    }
}

/// Splits `lyrics` into consecutive windows of at most `window_bytes` bytes,
/// breaking between lines where possible and between words otherwise.
/// Returns one empty window for empty lyrics or a model without a lyrics field.
pub fn lyrics_windows(lyrics: &str, window_bytes: usize) -> Vec<String> {
    let words_per_line: Vec<Vec<&str>> =
        lyrics.lines().map(|line| line.split_whitespace().collect::<Vec<_>>()).filter(|words| !words.is_empty()).collect();
    if window_bytes == 0 || words_per_line.is_empty() {
        return vec![String::new()];
    }

    let mut windows = Vec::new();
    let mut current = String::new();
    for words in words_per_line {
        let line = words.join(" ");
        if joined_length(&current, &line) <= window_bytes {
            push_with_separator(&mut current, &line, '\n');
            continue;
        }

        if !current.is_empty() {
            windows.push(std::mem::take(&mut current));
        }
        for word in words {
            if !current.is_empty() && joined_length(&current, word) > window_bytes {
                windows.push(std::mem::take(&mut current));
            }
            push_with_separator(&mut current, word, ' ');
        }
    }

    if !current.is_empty() {
        windows.push(current);
    }
    windows
}

/// Length of `buffer` after appending `text` with a one-byte separator.
fn joined_length(buffer: &str, text: &str) -> usize {
    if buffer.is_empty() { text.len() } else { buffer.len() + 1 + text.len() }
}

/// Appends `text` to `buffer`, preceded by `separator` unless `buffer` is empty.
fn push_with_separator(buffer: &mut String, text: &str, separator: char) {
    if !buffer.is_empty() {
        buffer.push(separator);
    }
    buffer.push_str(text);
}

/// Draws one code from `logits` according to `sampling`.
fn sample(logits: &[f32], sampling: Sampling, random: &mut Random) -> u32 {
    let (temperature, k) = match sampling {
        Sampling::Greedy => {
            let best = (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap_or(0);
            return best as u32;
        }
        Sampling::TopK { temperature, k } => (temperature, k.clamp(1, logits.len())),
    };

    let mut candidates: Vec<usize> = (0..logits.len()).collect();
    candidates.select_nth_unstable_by(k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
    candidates.truncate(k);

    let best_logit = candidates.iter().map(|&code| logits[code]).fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = candidates.iter().map(|&code| ((logits[code] - best_logit) / temperature).exp()).collect();

    let mut threshold = random.uniform() * weights.iter().sum::<f32>();
    for (&code, &weight) in candidates.iter().zip(&weights) {
        threshold -= weight;
        if threshold <= 0.0 {
            return code as u32;
        }
    }
    candidates[k - 1] as u32
}

/// Generates `config.frames` frames of audio codes for `prompt`; its lyrics
/// are sung in order across the whole output.
pub fn generate(gpu: &Gpu, model: &Gpt, prompt: Conditioning, config: &GenerationConfig) -> GpuResult<Generation> {
    let vocabulary = model.vocabulary();
    let codebooks = vocabulary.codebooks;
    let codebook_size = vocabulary.codebook_size;
    let pattern = DelayPattern { codebooks };
    let mut random = Random::new(config.seed);

    let windows = lyrics_windows(prompt.lyrics, vocabulary.lyrics_bytes);
    let window_prompt = |index: usize| vocabulary.prompt_slots(Conditioning { caption: prompt.caption, lyrics: &windows[index] });
    let mut current_window = 0;

    let mut conditional = ConditionedStream { prompt_slots: window_prompt(current_window), decoder: Decoder::new(gpu, model)? };
    let mut unconditional = match config.guidance {
        Guidance::Off => None,
        Guidance::ClassifierFree { .. } => {
            Some(ConditionedStream { prompt_slots: vocabulary.prompt_slots(Conditioning::NONE), decoder: Decoder::new(gpu, model)? })
        }
    };

    let steps = pattern.delayed_steps(config.frames);
    let mut history: Vec<u32> = Vec::with_capacity(steps * codebooks);
    let mut delayed_codes: Vec<u32> = Vec::with_capacity(steps * codebooks);
    let mut model_calls = 0;
    let started = Instant::now();

    for step in 0..steps {
        let window = step * windows.len() / steps;
        if window != current_window {
            current_window = window;
            conditional.condition_on(window_prompt(window));
        }

        let mut logits = conditional.next_logits(gpu, model, &history, config.attention)?;
        model_calls += 1;

        if let (Some(stream), Guidance::ClassifierFree { scale }) = (unconditional.as_mut(), config.guidance) {
            let unconditional_logits = stream.next_logits(gpu, model, &history, config.attention)?;
            model_calls += 1;

            for (guided, &free) in logits.iter_mut().zip(&unconditional_logits) {
                *guided = free + scale * (*guided - free);
            }
        }

        for codebook in 0..codebooks {
            let inside_pattern = step >= codebook && step - codebook < config.frames;
            if !inside_pattern {
                history.push(EMPTY_TOKEN);
                delayed_codes.push(EMPTY_TOKEN);
                continue;
            }

            let head_logits = &logits[codebook * codebook_size..(codebook + 1) * codebook_size];
            let code = sample(head_logits, config.sampling, &mut random);
            history.push(vocabulary.audio_token(codebook, code));
            delayed_codes.push(code);
        }
    }

    gpu.synchronize()?;
    let stats = GenerationStats { steps, model_calls, seconds: started.elapsed().as_secs_f64() };
    Ok(Generation { codes: pattern.undelay(&delayed_codes), stats })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lyrics_are_packed_into_windows_at_line_breaks() {
        let lyrics = "first line here\nsecond line\n\nthird";
        assert_eq!(lyrics_windows(lyrics, 30), vec!["first line here\nsecond line", "third"]);
        assert_eq!(lyrics_windows(lyrics, 0), vec![""]);
        assert_eq!(lyrics_windows("", 30), vec![""]);
    }

    #[test]
    fn overlong_lines_break_between_words() {
        assert_eq!(lyrics_windows("one two three four", 9), vec!["one two", "three", "four"]);
    }
}
