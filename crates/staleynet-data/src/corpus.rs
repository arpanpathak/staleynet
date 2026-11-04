//! Tokenised tracks in memory, and the sampler that turns them into
//! [`TrainingBatch`]es.

use crate::catalog::{LineTiming, LyricLine, Split};
use crate::store::{CacheLayout, DatasetManifest, read_codes};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use staleynet_cuda::EMPTY_TOKEN;
use staleynet_model::vocabulary::Conditioning;
use staleynet_model::{DelayPattern, Random, TrainingBatch, Vocabulary};
use std::ops::Range;

/// A lyric line with a time span, measured or estimated.
#[derive(Clone, Debug, PartialEq)]
pub struct TimedLine {
    /// The words.
    pub text: String,

    /// When it is sung.
    pub timing: LineTiming,
}

/// Gives every line a time span. Measured timings are kept. When none are
/// known, the lines are assumed to be spread evenly over the recording: rough,
/// but right for a short take of continuous singing, which is what the studio
/// asks for.
fn timed_lines(lyrics: &[LyricLine], seconds: f32) -> Vec<TimedLine> {
    let share = seconds / lyrics.len().max(1) as f32;

    lyrics
        .iter()
        .enumerate()
        .map(|(index, line)| {
            let estimated = LineTiming { start: index as f32 * share, end: (index + 1) as f32 * share };
            TimedLine { text: line.text.clone(), timing: line.timing.unwrap_or(estimated) }
        })
        .collect()
}

/// One tokenised recording.
pub struct TokenTrack {
    /// Descriptions to condition on.
    pub captions: Vec<String>,

    /// The words sung, each with a time span; empty without lyrics.
    pub lyrics: Vec<TimedLine>,

    /// Codes `[steps, codebooks]`.
    pub codes: Vec<u32>,

    /// Duration of one code step.
    pub seconds_per_step: f32,
}

impl TokenTrack {
    /// Lyrics sung during steps `first_step .. first_step + steps`: every line
    /// that overlaps that span, one per text line.
    pub fn lyrics_between(&self, first_step: usize, steps: usize) -> String {
        let start = first_step as f32 * self.seconds_per_step;
        let end = (first_step + steps) as f32 * self.seconds_per_step;

        let sung: Vec<&str> = self
            .lyrics
            .iter()
            .filter(|line| line.timing.end > start && line.timing.start < end)
            .map(|line| line.text.as_str())
            .collect();
        sung.join("\n")
    }

    /// Lyrics of a `steps`-long window starting at a randomly chosen line, so
    /// the window is never empty for a track with lyrics. Always makes one draw.
    pub fn lyrics_from_random_line(&self, steps: usize, random: &mut Random) -> String {
        let choice = random.below(self.lyrics.len().max(1));
        let Some(line) = self.lyrics.get(choice) else { return String::new() };

        let first_step = (line.timing.start / self.seconds_per_step) as usize;
        self.lyrics_between(first_step, steps)
    }
}

/// Selects tracks by the words in their captions. Matching is on whole words,
/// ignoring case. An empty filter keeps every track.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrackFilter {
    /// Keep a track only if at least one caption contains one of these words.
    /// Empty means no requirement.
    #[serde(default)]
    pub require_any: Vec<String>,

    /// Drop a track if any caption contains one of these words or phrases.
    #[serde(default)]
    pub exclude_any: Vec<String>,
}

impl TrackFilter {
    /// True if a track with these captions passes the filter.
    pub fn accepts(&self, captions: &[String]) -> bool {
        let mentions = |terms: &[String]| {
            captions
                .iter()
                .any(|caption| terms.iter().any(|term| contains_phrase(caption, term)))
        };

        let required = self.require_any.is_empty() || mentions(&self.require_any);
        let excluded = mentions(&self.exclude_any);
        required && !excluded
    }
}

/// True if `phrase` occurs in `text` as whole words, ignoring case.
fn contains_phrase(text: &str, phrase: &str) -> bool {
    let words = |value: &str| {
        value
            .split(|character: char| !character.is_alphanumeric())
            .filter(|word| !word.is_empty())
            .map(str::to_lowercase)
            .collect::<Vec<_>>()
    };

    let text_words = words(text);
    let phrase_words = words(phrase);

    if phrase_words.is_empty() {
        return false;
    }

    text_words
        .windows(phrase_words.len())
        .any(|window| window == phrase_words.as_slice())
}

/// Tokenised tracks drawn from one or more datasets, each track with a
/// sampling weight.
#[derive(Default)]
pub struct TokenCorpus {
    /// Tracks with at least one code step.
    pub tracks: Vec<TokenTrack>,

    /// Running sum of the track weights: entry `i` is the total weight of
    /// tracks `0..=i`. Sampling searches it, so drawing a track is `O(log n)`.
    cumulative_weight: Vec<f64>,
}

impl TokenCorpus {
    /// Loads every track of `split` listed in the manifest that `filter`
    /// accepts, each with sampling weight 1.
    pub fn load(layout: &CacheLayout, manifest: &DatasetManifest, split: Split, filter: &TrackFilter, codebooks: usize) -> Result<Self> {
        let mut corpus = Self::default();
        for entry in manifest
            .tracks
            .iter()
            .filter(|entry| entry.track.split == split)
        {
            let captions: Vec<String> = entry
                .track
                .captions
                .iter()
                .map(|caption| caption.text.clone())
                .collect();
            if !filter.accepts(&captions) {
                continue;
            }

            let codes = read_codes(&layout.codes(&entry.track.id))?;
            let steps = codes.len() / codebooks;
            if steps == 0 {
                continue;
            }

            let track = TokenTrack {
                captions,
                lyrics: timed_lines(&entry.track.lyrics, entry.seconds),
                codes,
                seconds_per_step: entry.seconds / steps as f32,
            };
            corpus.push(track, 1.0);
        }

        Ok(corpus)
    }

    /// Adds one track drawn with relative probability `weight`.
    pub fn push(&mut self, track: TokenTrack, weight: f64) {
        let total = self.cumulative_weight.last().copied().unwrap_or(0.0);
        self.tracks.push(track);
        self.cumulative_weight.push(total + weight);
    }

    /// Moves every track of `other` into this corpus, scaling their weights by
    /// `weight`. A weight of 3 makes each of those tracks three times as likely
    /// to be drawn as a weight-1 track.
    pub fn absorb(&mut self, other: TokenCorpus, weight: f64) {
        let mut previous = 0.0;
        for (track, cumulative) in other.tracks.into_iter().zip(other.cumulative_weight) {
            self.push(track, (cumulative - previous) * weight);
            previous = cumulative;
        }
    }

    /// A track drawn in proportion to its weight.
    pub fn pick(&self, random: &mut Random) -> &TokenTrack {
        &self.tracks[self.pick_index(random)]
    }

    /// Index of a track drawn in proportion to its weight.
    fn pick_index(&self, random: &mut Random) -> usize {
        let total = self.cumulative_weight.last().copied().unwrap_or(0.0);
        let target = random.uniform() as f64 * total;
        let index = self
            .cumulative_weight
            .partition_point(|&cumulative| cumulative <= target);
        index.min(self.tracks.len() - 1)
    }

    /// Keeps only the tracks that come with lyrics.
    pub fn retain_tracks_with_lyrics(self) -> Self {
        let mut kept = Self::default();
        let mut previous = 0.0;

        for (track, cumulative) in self.tracks.into_iter().zip(self.cumulative_weight) {
            if !track.lyrics.is_empty() {
                kept.push(track, cumulative - previous);
            }
            previous = cumulative;
        }

        kept
    }

    /// Total code steps across tracks.
    pub fn total_steps(&self, codebooks: usize) -> usize {
        self.tracks
            .iter()
            .map(|track| track.codes.len() / codebooks)
            .sum()
    }

    /// Tracks that come with lyrics.
    pub fn tracks_with_lyrics(&self) -> usize {
        self.tracks
            .iter()
            .filter(|track| !track.lyrics.is_empty())
            .count()
    }
}

/// Which lyrics go into a sequence's prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LyricsCondition {
    /// The lines sung during the crop (training and normal validation).
    Aligned,

    /// None at all.
    Removed,

    /// Lines from a different track, as a control: a model that reads the
    /// lyrics predicts the audio worse with these than with the right ones.
    Mismatched,
}

/// Batch sampling settings.
#[derive(Clone, Copy, Debug)]
pub struct SamplerOptions {
    /// Probability of replacing the caption with the empty prompt. Training on
    /// both lets one model serve classifier-free guidance (Ho & Salimans, 2022,
    /// <https://arxiv.org/abs/2207.12598>); MusicGen uses 0.2 for text (Copet et
    /// al., 2023, §4.1), we use a smaller value for a small dataset.
    pub prompt_dropout: f32,

    /// Probability of starting the crop at the beginning of the track, so the
    /// model also learns how pieces open.
    pub start_probability: f32,

    /// Which lyrics accompany the crop.
    pub lyrics: LyricsCondition,
}

impl SamplerOptions {
    /// Settings for training.
    pub const TRAINING: Self = Self { prompt_dropout: 0.1, start_probability: 0.2, lyrics: LyricsCondition::Aligned };

    /// Settings for validation: always conditioned.
    pub const VALIDATION: Self = Self { prompt_dropout: 0.0, start_probability: 0.0, lyrics: LyricsCondition::Aligned };
}

/// Fills [`TrainingBatch`]es with random crops of a corpus.
pub struct BatchSampler {
    /// Token layout of the model being trained.
    vocabulary: Vocabulary,

    /// Delay pattern applied to every track.
    pattern: DelayPattern,

    /// Sampling settings.
    options: SamplerOptions,
}

impl BatchSampler {
    /// A sampler for a model with `vocabulary`.
    pub fn new(vocabulary: Vocabulary, options: SamplerOptions) -> Self {
        Self { pattern: DelayPattern { codebooks: vocabulary.codebooks }, vocabulary, options }
    }

    /// Overwrites `batch` with fresh sequences drawn from `corpus`.
    pub fn fill(&self, corpus: &TokenCorpus, batch: &mut TrainingBatch, random: &mut Random) {
        batch.tokens.fill(EMPTY_TOKEN);
        batch.targets.fill(EMPTY_TOKEN);

        for sequence in 0..batch.shape.batch {
            // Every draw is made under every condition, so a seed gives the
            // same crops whichever lyrics are attached to them.
            let index = corpus.pick_index(random);
            let donor = &corpus.tracks[other_track(index, corpus.tracks.len(), random)];
            let audio_positions = batch.shape.time - self.vocabulary.prompt_positions();
            let donor_lyrics = donor.lyrics_from_random_line(audio_positions, random);
            let mismatched = (self.options.lyrics == LyricsCondition::Mismatched).then_some(donor_lyrics.as_str());
            self.fill_sequence(&corpus.tracks[index], mismatched, sequence, batch, random);
        }
    }

    /// Writes one sequence: the prompt (a caption and the lyrics sung during
    /// the crop), then a crop of the delayed code stream, with each position's
    /// target being the next position's codes. Prompt dropout removes caption
    /// and lyrics together, giving the unconditional sequence.
    /// `mismatched` holds another track's lines under [`LyricsCondition::Mismatched`].
    ///
    /// The random draws happen in a fixed order (crop start, dropout, caption),
    /// so a seed always gives the same crops.
    fn fill_sequence(&self, track: &TokenTrack, mismatched: Option<&str>, sequence: usize, batch: &mut TrainingBatch, random: &mut Random) {
        let rows = SequenceRows { sequence, time: batch.shape.time, codebooks: self.vocabulary.codebooks };
        let prompt_positions = self.vocabulary.prompt_positions();
        let audio_positions = rows.time - prompt_positions;

        let delayed_steps = self
            .pattern
            .delayed_steps(track.codes.len() / rows.codebooks);
        let first_step = self.crop_start(delayed_steps, audio_positions, random);
        let lyrics = self.lyrics_for(track, mismatched, first_step, audio_positions);
        let conditioning = self.conditioning(track, &lyrics, random);

        self.vocabulary
            .write_prompt(conditioning, &mut batch.tokens[rows.slot(0)..rows.slot(prompt_positions)]);
        self.write_crop(track, first_step..delayed_steps, rows, prompt_positions, batch);
    }

    /// The delayed step at which the crop begins: the start of the track with
    /// probability `start_probability` (and whenever the track is no longer
    /// than the crop), otherwise a uniformly random step.
    fn crop_start(&self, delayed_steps: usize, audio_positions: usize, random: &mut Random) -> usize {
        let from_start = random.chance(self.options.start_probability) || delayed_steps <= audio_positions;
        if from_start { 0 } else { random.below(delayed_steps - audio_positions) }
    }

    /// The lyrics placed in the prompt under the sampler's lyrics condition.
    fn lyrics_for(&self, track: &TokenTrack, mismatched: Option<&str>, first_step: usize, audio_positions: usize) -> String {
        match (self.options.lyrics, mismatched) {
            (LyricsCondition::Removed, _) => String::new(),
            (LyricsCondition::Mismatched, Some(other)) => other.to_string(),
            (LyricsCondition::Aligned | LyricsCondition::Mismatched, _) => track.lyrics_between(first_step, audio_positions),
        }
    }

    /// The prompt: nothing with probability `prompt_dropout`, otherwise one of
    /// the track's captions, chosen uniformly, with `lyrics`.
    fn conditioning<'a>(&self, track: &'a TokenTrack, lyrics: &'a str, random: &mut Random) -> Conditioning<'a> {
        if random.chance(self.options.prompt_dropout) {
            return Conditioning::NONE;
        }

        Conditioning { caption: &track.captions[random.below(track.captions.len())], lyrics }
    }

    /// Writes the delayed codes of `steps` after the prompt: position `offset`
    /// receives step `first + offset` as input, and the position before it
    /// receives the same codes as its target. The position after the last
    /// input has no input slot; it only supplies the final target.
    fn write_crop(&self, track: &TokenTrack, steps: Range<usize>, rows: SequenceRows, prompt_positions: usize, batch: &mut TrainingBatch) {
        let audio_positions = rows.time - prompt_positions;

        for (offset, step) in steps.take(audio_positions + 1).enumerate() {
            for codebook in 0..rows.codebooks {
                let Some(code) = self.pattern.delayed_code(&track.codes, step, codebook) else { continue };

                if offset < audio_positions {
                    batch.tokens[rows.slot(prompt_positions + offset) + codebook] = self.vocabulary.audio_token(codebook, code);
                }
                batch.targets[rows.slot(prompt_positions - 1 + offset) + codebook] = code;
            }
        }
    }
}

/// Where one sequence of a batch lives in the flat `[sequence, position,
/// codebook]` token and target arrays.
#[derive(Clone, Copy)]
struct SequenceRows {
    /// Index of the sequence within the batch.
    sequence: usize,

    /// Positions per sequence.
    time: usize,

    /// Codes per position.
    codebooks: usize,
}

impl SequenceRows {
    /// Index of the first code of `position` in this sequence.
    fn slot(&self, position: usize) -> usize {
        (self.sequence * self.time + position) * self.codebooks
    }
}

/// A uniformly chosen index in `0..count` other than `excluded` (or
/// `excluded` itself when it is the only one).
fn other_track(excluded: usize, count: usize, random: &mut Random) -> usize {
    if count < 2 {
        return excluded;
    }
    let candidate = random.below(count - 1);
    if candidate >= excluded { candidate + 1 } else { candidate }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_matches_whole_words_and_honours_exclusions() {
        let filter = TrackFilter { require_any: vec!["vocals".into(), "singer".into()], exclude_any: vec!["no vocals".into()] };
        let captions = |text: &str| vec![text.to_string()];

        assert!(filter.accepts(&captions("Pop song with female Vocals.")));
        assert!(filter.accepts(&captions("a male singer and guitar")));
        assert!(!filter.accepts(&captions("instrumental with no vocals at all")));
        assert!(!filter.accepts(&captions("singers' choir"))); // "singers" is not the word "singer"
        assert!(TrackFilter::default().accepts(&captions("anything")));
    }
    use staleynet_model::{ModelConfig, SequenceShape};

    #[test]
    fn crops_get_the_lines_sung_during_them() {
        let line = |text: &str, start: f32, end: f32| LyricLine { text: text.into(), timing: Some(LineTiming { start, end }) };
        let lyrics = [line("one", 0.0, 1.0), line("two", 1.0, 2.0), line("three", 2.5, 3.0)];
        let track = TokenTrack { captions: Vec::new(), lyrics: timed_lines(&lyrics, 3.0), codes: Vec::new(), seconds_per_step: 0.1 };

        assert_eq!(track.lyrics_between(5, 10), "one\ntwo");
        assert_eq!(track.lyrics_between(20, 10), "three");
        assert_eq!(track.lyrics_between(21, 3), "");
    }

    #[test]
    fn untimed_lyrics_are_spread_evenly() {
        let lyrics = [LyricLine { text: "a".into(), timing: None }, LyricLine { text: "b".into(), timing: None }];
        let timed = timed_lines(&lyrics, 10.0);
        assert_eq!(timed[1].timing, LineTiming { start: 5.0, end: 10.0 });
    }

    #[test]
    fn weights_bias_sampling() {
        let track =
            |caption: &str| TokenTrack { captions: vec![caption.into()], lyrics: Vec::new(), codes: vec![0; 4], seconds_per_step: 0.1 };

        let mut corpus = TokenCorpus::default();
        corpus.push(track("light"), 1.0);

        let mut heavy = TokenCorpus::default();
        heavy.push(track("heavy"), 1.0);
        corpus.absorb(heavy, 3.0);

        let mut random = Random::new(7);
        let heavy_draws = (0..4000)
            .filter(|_| corpus.pick(&mut random).captions[0] == "heavy")
            .count();
        assert!((2800..3200).contains(&heavy_draws), "{heavy_draws} of 4000");
    }

    #[test]
    fn every_target_is_the_next_positions_input() {
        let config = ModelConfig { prompt_bytes: 4, codebooks: 3, codebook_size: 10, context_length: 32, ..ModelConfig::default() };
        let vocabulary = Vocabulary::new(&config);

        let mut corpus = TokenCorpus::default();
        corpus.push(
            TokenTrack {
                captions: vec!["calm piano".into()],
                lyrics: Vec::new(),
                codes: (0..90).map(|value| value % 10).collect(),
                seconds_per_step: 0.02,
            },
            1.0,
        );

        let shape = SequenceShape { batch: 4, time: 20 };
        let mut batch = TrainingBatch::empty(shape, config.codebooks);

        BatchSampler::new(vocabulary, SamplerOptions::TRAINING).fill(&corpus, &mut batch, &mut Random::new(1));

        for position in 1..shape.rows() {
            if position % shape.time == 0 {
                continue;
            }
            for codebook in 0..config.codebooks {
                let input = batch.tokens[position * config.codebooks + codebook];
                let target = batch.targets[(position - 1) * config.codebooks + codebook];
                if position % shape.time >= vocabulary.prompt_positions() && input != EMPTY_TOKEN {
                    assert_eq!(input, vocabulary.audio_token(codebook, target));
                }
            }
        }
    }
}
