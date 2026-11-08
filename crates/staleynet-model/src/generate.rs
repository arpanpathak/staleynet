//! Prompt → audio codes by autoregressive sampling.
//!
//! Lyrics longer than the model's lyrics field are cut into consecutive
//! windows at line breaks. The output is divided evenly between the windows,
//! and when one window ends the next replaces it in the prompt while the most
//! recent audio stays in context, so the voice carries on into the next lines.

use crate::decode::Decoder;
use crate::gpt::Gpt;
use crate::random::Random;
use crate::vocabulary::{Conditioning, DelayPattern, Vocabulary, sung_byte};
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

    /// Duration of one code step, which sets how fast the lyric cursor moves.
    pub seconds_per_step: f32,
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

    /// Lyric cursor of every delayed step; empty for the unprompted stream.
    cursors: Vec<u32>,

    /// First step whose cursor points into the current prompt; earlier steps,
    /// sung from a previous window, get no cursor after a prompt swap.
    cursor_from: usize,
}

impl ConditionedStream {
    /// Swaps the prompt from `step` on. The caches are dropped, so the next
    /// call rebuilds them from the new prompt and the most recent audio.
    fn condition_on(&mut self, prompt_slots: Vec<u32>, step: usize) {
        self.prompt_slots = prompt_slots;
        self.cursor_from = step;
        self.decoder.reset();
    }

    /// The lyric cursor for the position that predicts `step`.
    fn cursor(&self, step: usize) -> u32 {
        if step < self.cursor_from {
            return EMPTY_TOKEN;
        }
        self.cursors.get(step).copied().unwrap_or(EMPTY_TOKEN)
    }

    /// Logits for the next step, given every audio step generated so far.
    /// The position holding step `s` predicts step `s + 1`, so it carries
    /// that step's cursor; the last prompt position predicts the first kept step.
    fn next_logits(&mut self, gpu: &Gpu, model: &Gpt, history: &[u32], attention: AttentionMode) -> GpuResult<Vec<f32>> {
        let codebooks = model.config.codebooks;
        let history_steps = history.len() / codebooks;
        let can_extend = matches!(attention, AttentionMode::KvCache) && self.decoder.cached_positions() > 0 && self.decoder.remaining() > 0;

        if can_extend {
            let newest_step = &history[history.len() - codebooks..];
            return self
                .decoder
                .feed_with_cursors(gpu, model, newest_step, &[self.cursor(history_steps)]);
        }

        // Rebuild from the prompt plus as much recent history as fits. With a
        // cache, keep only half the window so rebuilds stay rare.
        let audio_room = model.config.sequence_capacity() - self.prompt_slots.len() / codebooks;
        let window = match attention {
            AttentionMode::KvCache => audio_room / 2,
            AttentionMode::Recompute => audio_room,
        };

        let kept_steps = history_steps.min(window);
        let first_kept = history_steps - kept_steps;

        let mut slots = self.prompt_slots.clone();
        slots.extend_from_slice(&history[first_kept * codebooks..]);

        let prompt_positions = self.prompt_slots.len() / codebooks;
        let mut cursors = vec![EMPTY_TOKEN; prompt_positions - 1];
        cursors.extend((first_kept..=history_steps).map(|step| self.cursor(step)));

        self.decoder.reset();
        if attention == AttentionMode::Recompute {
            self.decoder.release_scratch_except(slots.len() / codebooks);
        }
        self.decoder.feed_with_cursors(gpu, model, &slots, &cursors)
    }
}

/// Splits `lyrics` into consecutive windows of at most `window_bytes` bytes,
/// breaking between lines where possible and between words otherwise.
/// Returns one empty window for empty lyrics or a model without a lyrics field.
pub fn lyrics_windows(lyrics: &str, window_bytes: usize) -> Vec<String> {
    let words_per_line: Vec<Vec<&str>> = lyrics
        .lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .filter(|words| !words.is_empty())
        .collect();
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
            let best = (0..logits.len())
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                .unwrap_or(0);
            return best as u32;
        }
        Sampling::TopK { temperature, k } => (temperature, k.clamp(1, logits.len())),
    };

    let mut candidates: Vec<usize> = (0..logits.len()).collect();
    candidates.select_nth_unstable_by(k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
    candidates.truncate(k);

    let best_logit = candidates
        .iter()
        .map(|&code| logits[code])
        .fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = candidates
        .iter()
        .map(|&code| ((logits[code] - best_logit) / temperature).exp())
        .collect();

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
///
/// Each delayed step asks the model for the next codes (twice under
/// classifier-free guidance), samples one code per codebook, and appends them
/// to the history the model reads at the following step.
pub fn generate(gpu: &Gpu, model: &Gpt, prompt: Conditioning, config: &GenerationConfig) -> GpuResult<Generation> {
    let vocabulary = model.vocabulary();
    let pattern = DelayPattern { codebooks: vocabulary.codebooks };
    let steps = pattern.delayed_steps(config.frames);

    let mut lyrics = LyricsSchedule::new(prompt, vocabulary.lyrics_bytes, steps);
    let cursors = lyrics.cursors(&vocabulary, config.seconds_per_step);
    let mut streams = GuidedStreams::new(gpu, model, lyrics.prompt_slots(&vocabulary), cursors, config.guidance)?;
    let mut codes = DelayedCodes::with_capacity(steps * vocabulary.codebooks);

    let mut random = Random::new(config.seed);
    let started = Instant::now();

    for step in 0..steps {
        if lyrics.advance_to(step) {
            streams
                .conditional
                .condition_on(lyrics.prompt_slots(&vocabulary), step);
        }

        let logits = streams.next_logits(gpu, model, &codes.history, config.attention)?;
        codes.append_step(step, &logits, &vocabulary, config, &mut random);
    }

    gpu.synchronize()?;
    let stats = GenerationStats { steps, model_calls: streams.model_calls, seconds: started.elapsed().as_secs_f64() };
    Ok(Generation { codes: pattern.undelay(&codes.delayed), stats })
}

/// Which window of the lyrics is in the prompt at each step. The output is
/// divided evenly between the windows, so each is sung for the same time.
struct LyricsSchedule<'a> {
    /// The caption, which stays the same in every window.
    caption: &'a str,

    /// The lyrics cut into windows that fit the model's lyrics field.
    windows: Vec<String>,

    /// Index of the window currently in the prompt.
    current: usize,

    /// Delayed steps in the whole generation.
    total_steps: usize,
}

impl<'a> LyricsSchedule<'a> {
    /// Cuts the lyrics of `prompt` into windows of at most `lyrics_bytes` bytes
    /// and starts at the first one.
    fn new(prompt: Conditioning<'a>, lyrics_bytes: usize, total_steps: usize) -> Self {
        Self { caption: prompt.caption, windows: lyrics_windows(prompt.lyrics, lyrics_bytes), current: 0, total_steps }
    }

    /// Moves to the window that `step` falls in; true when the window changed.
    fn advance_to(&mut self, step: usize) -> bool {
        let window = step * self.windows.len() / self.total_steps;
        if window == self.current {
            return false;
        }

        self.current = window;
        true
    }

    /// Prompt slots for the caption and the current window.
    fn prompt_slots(&self, vocabulary: &Vocabulary) -> Vec<u32> {
        vocabulary.prompt_slots(Conditioning { caption: self.caption, lyrics: &self.windows[self.current] })
    }

    /// The lyric cursor of every delayed step. Each window's lines are sung one
    /// after another across the window's steps, each given time in proportion
    /// to its length, and swept with [`sung_byte`] as in training.
    fn cursors(&self, vocabulary: &Vocabulary, seconds_per_step: f32) -> Vec<u32> {
        let windows = self.windows.len();
        let first_step = |window: usize| (window * self.total_steps).div_ceil(windows);

        (0..self.total_steps)
            .map(|step| {
                let window = step * windows / self.total_steps;
                let first = first_step(window);
                let steps = first_step(window + 1) - first;
                window_cursor(&self.windows[window], step - first, steps, seconds_per_step, vocabulary)
            })
            .collect()
    }
}

/// The lyric cursor `step` steps into a window of `steps` steps that sings
/// `text`, one line after another, each for a share of the time proportional
/// to its bytes (counting its line break).
fn window_cursor(text: &str, step: usize, steps: usize, seconds_per_step: f32, vocabulary: &Vocabulary) -> u32 {
    let bytes = text.len() + 1;
    let seconds_per_byte = steps as f32 * seconds_per_step / bytes as f32;
    let elapsed = step as f32 * seconds_per_step;

    let mut line_start = 0;
    for line in text.split('\n') {
        let line_end = line_start + line.len() + 1;
        if elapsed < line_end as f32 * seconds_per_byte {
            let into_line = elapsed - line_start as f32 * seconds_per_byte;
            let duration = (line.len() + 1) as f32 * seconds_per_byte;
            return sung_byte(into_line, duration, line.len()).map_or(EMPTY_TOKEN, |byte| vocabulary.cursor_row(line_start + byte));
        }
        line_start = line_end;
    }

    EMPTY_TOKEN
}

/// The prompted stream and, under classifier-free guidance, the unprompted
/// one, each with its own key/value caches.
struct GuidedStreams {
    /// The stream conditioned on the caption and the current lyrics.
    conditional: ConditionedStream,

    /// The stream with an empty prompt; present only under guidance.
    unconditional: Option<ConditionedStream>,

    /// How the two streams are combined.
    guidance: Guidance,

    /// Forward calls made so far, for the statistics.
    model_calls: usize,
}

impl GuidedStreams {
    /// Opens the prompted stream on `prompt_slots` with the lyric `cursors` of
    /// every step, and the unprompted one (with no cursor) when
    /// `guidance` needs it.
    fn new(gpu: &Gpu, model: &Gpt, prompt_slots: Vec<u32>, cursors: Vec<u32>, guidance: Guidance) -> GpuResult<Self> {
        let conditional = ConditionedStream { prompt_slots, decoder: Decoder::new(gpu, model)?, cursors, cursor_from: 0 };
        let unconditional = match guidance {
            Guidance::Off => None,
            Guidance::ClassifierFree { .. } => Some(ConditionedStream {
                prompt_slots: model.vocabulary().prompt_slots(Conditioning::NONE),
                decoder: Decoder::new(gpu, model)?,
                cursors: Vec::new(),
                cursor_from: 0,
            }),
        };

        Ok(Self { conditional, unconditional, guidance, model_calls: 0 })
    }

    /// Logits for the next step. Under guidance they are moved away from the
    /// unprompted logits: `unconditional + scale · (conditional − unconditional)`.
    fn next_logits(&mut self, gpu: &Gpu, model: &Gpt, history: &[u32], attention: AttentionMode) -> GpuResult<Vec<f32>> {
        let mut logits = self
            .conditional
            .next_logits(gpu, model, history, attention)?;
        self.model_calls += 1;

        let (Some(stream), Guidance::ClassifierFree { scale }) = (self.unconditional.as_mut(), self.guidance) else {
            return Ok(logits);
        };
        let unconditional_logits = stream.next_logits(gpu, model, history, attention)?;
        self.model_calls += 1;

        for (guided, &free) in logits.iter_mut().zip(&unconditional_logits) {
            *guided = free + scale * (*guided - free);
        }
        Ok(logits)
    }
}

/// The codes sampled so far, in delayed order: as vocabulary tokens for the
/// model to read, and as plain codes for the tokenizer to decode.
struct DelayedCodes {
    /// Vocabulary tokens, `[delayed step, codebook]`, fed back to the model.
    history: Vec<u32>,

    /// Codes in `[0, codebook_size)`, `[delayed step, codebook]`.
    delayed: Vec<u32>,
}

impl DelayedCodes {
    /// Empty buffers with room for `capacity` codes.
    fn with_capacity(capacity: usize) -> Self {
        Self { history: Vec::with_capacity(capacity), delayed: Vec::with_capacity(capacity) }
    }

    /// Samples one code per codebook for delayed `step`. Codebook `c` runs `c`
    /// steps behind codebook 0, so near the start and the end some codebooks
    /// have no frame to describe and receive the empty token instead.
    fn append_step(&mut self, step: usize, logits: &[f32], vocabulary: &Vocabulary, config: &GenerationConfig, random: &mut Random) {
        let codebook_size = vocabulary.codebook_size;

        for codebook in 0..vocabulary.codebooks {
            let inside_pattern = step >= codebook && step - codebook < config.frames;
            if !inside_pattern {
                self.history.push(EMPTY_TOKEN);
                self.delayed.push(EMPTY_TOKEN);
                continue;
            }

            let head_logits = &logits[codebook * codebook_size..(codebook + 1) * codebook_size];
            let code = sample(head_logits, config.sampling, random);

            self.history.push(vocabulary.audio_token(codebook, code));
            self.delayed.push(code);
        }
    }
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

    #[test]
    fn the_cursor_sings_each_line_of_a_window_in_turn() {
        let vocabulary = Vocabulary { codebooks: 1, codebook_size: 8, prompt_bytes: 2, lyrics_bytes: 8, context_length: 16 };
        let cursor = |step| window_cursor("ab\ncd", step, 10, 0.1, &vocabulary);

        assert_eq!(cursor(0), vocabulary.cursor_row(0));
        assert_eq!(cursor(3), vocabulary.cursor_row(1));
        assert_eq!(cursor(6), vocabulary.cursor_row(3));
        assert_eq!(cursor(8), vocabulary.cursor_row(4));
        assert_eq!(window_cursor("", 0, 10, 0.1, &vocabulary), EMPTY_TOKEN);
    }
}
