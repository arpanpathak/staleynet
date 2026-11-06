//! The shared token space and the sequence layout.
//!
//! Text and audio live in one embedding table:
//!
//! ```text
//! 0..256                     UTF-8 bytes of the prompt
//! 256, 257, 258              BEGIN, SEPARATOR, PADDING
//! 259 + c·K + code           audio code `code` of codebook `c` (K = codebook size)
//! ```
//!
//! Byte-level text needs no learned tokenizer (cf. ByT5, Xue et al., 2022,
//! <https://arxiv.org/abs/2105.13626>). Every position carries `codebooks`
//! token slots; text positions use only the first slot. A sequence is a
//! fixed-length prompt followed by audio steps:
//!
//! ```text
//! region     BEGIN  caption bytes   lyrics bytes          SEPARATOR  audio …
//! physical   0      1 ..= P         P+1 ..= P+L           P+L+1      P+L+2 …
//! position   0      1 ..= P         C ..< C+L             P+1        P+2 …
//! ```
//!
//! `P` is `prompt_bytes`, `L` is `lyrics_bytes` and `C` is `context_length`.
//! The lyrics take position-table rows *after* the original ones, so every
//! other position keeps the row it had in a model without lyrics. That is
//! what lets such a model be extended with a lyrics field and keep its
//! training (see `checkpoint::load_weights`).

use crate::config::ModelConfig;
use staleynet_cuda::EMPTY_TOKEN;

/// Distinct byte values.
const BYTE_VALUES: u32 = 256;

/// Marks the start of every sequence.
pub const BEGIN: u32 = BYTE_VALUES;

/// Ends the prompt; the first audio step is predicted from this position.
pub const SEPARATOR: u32 = BYTE_VALUES + 1;

/// Fills unused prompt bytes so the prompt always has the same length.
pub const PADDING: u32 = BYTE_VALUES + 2;

/// Number of text token ids (bytes + the three specials).
pub const TEXT_VOCABULARY: usize = BYTE_VALUES as usize + 3;

/// Longest time one lyrics byte is assumed to be sung. A line's cursor moves
/// across its bytes at the line's own pace, but never slower than this, so a
/// line followed by a long instrumental stretch does not stretch across it.
pub const MAX_SECONDS_PER_LYRIC_BYTE: f32 = 0.25;

/// Which byte of a `bytes`-long line is being sung `elapsed` seconds into a
/// line that lasts `duration` seconds, or `None` before it and once it is over.
pub fn sung_byte(elapsed: f32, duration: f32, bytes: usize) -> Option<usize> {
    let sweep = duration.min(bytes as f32 * MAX_SECONDS_PER_LYRIC_BYTE);
    if bytes == 0 || elapsed < 0.0 || elapsed >= sweep {
        return None;
    }

    let byte = (elapsed / sweep * bytes as f32) as usize;
    Some(byte.min(bytes - 1))
}

/// What a sequence is conditioned on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Conditioning<'a> {
    /// Description of the sound, in the style of the training captions.
    pub caption: &'a str,

    /// Words to be sung in this stretch of audio.
    pub lyrics: &'a str,
}

impl Conditioning<'_> {
    /// No caption and no lyrics: the unconditional sequence used for
    /// classifier-free guidance.
    pub const NONE: Conditioning<'static> = Conditioning { caption: "", lyrics: "" };
}

/// The part of a sequence a physical position belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Region {
    /// The first position.
    Begin,

    /// A caption byte or caption padding.
    Caption,

    /// A lyrics byte or lyrics padding.
    Lyrics,

    /// The last prompt position; the first audio step is predicted from it.
    Separator,

    /// An audio step.
    Audio,
}

/// Index arithmetic for one [`ModelConfig`]'s vocabulary and sequence layout.
#[derive(Clone, Copy, Debug)]
pub struct Vocabulary {
    /// Token slots per position.
    pub codebooks: usize,

    /// Entries per codebook.
    pub codebook_size: usize,

    /// Caption bytes per sequence.
    pub prompt_bytes: usize,

    /// Lyrics bytes per sequence.
    pub lyrics_bytes: usize,

    /// Position rows used by everything except the lyrics.
    pub context_length: usize,
}

impl Vocabulary {
    /// The layout implied by `config`.
    pub fn new(config: &ModelConfig) -> Self {
        Self {
            codebooks: config.codebooks,
            codebook_size: config.codebook_size,
            prompt_bytes: config.prompt_bytes,
            lyrics_bytes: config.lyrics_bytes,
            context_length: config.context_length,
        }
    }

    /// Rows of the token embedding table.
    pub fn size(&self) -> usize {
        TEXT_VOCABULARY + self.codebooks * self.codebook_size
    }

    /// Positions occupied by the prompt: BEGIN, caption, lyrics, SEPARATOR.
    pub fn prompt_positions(&self) -> usize {
        self.prompt_bytes + self.lyrics_bytes + 2
    }

    /// The region physical position `position` falls in.
    pub fn region(&self, position: usize) -> Region {
        let caption_end = 1 + self.prompt_bytes;
        let lyrics_end = caption_end + self.lyrics_bytes;

        match position {
            0 => Region::Begin,
            caption if caption < caption_end => Region::Caption,
            lyrics if lyrics < lyrics_end => Region::Lyrics,
            separator if separator == lyrics_end => Region::Separator,
            _ => Region::Audio,
        }
    }

    /// Row of the position table used at physical position `position`.
    pub fn position_id(&self, position: usize) -> u32 {
        let row = match self.region(position) {
            Region::Begin | Region::Caption => position,
            Region::Lyrics => self.context_length + position - 1 - self.prompt_bytes,
            Region::Separator | Region::Audio => position - self.lyrics_bytes,
        };
        row as u32
    }

    /// Position-table rows of physical positions `first .. first + count`.
    pub fn position_ids(&self, first: usize, count: usize) -> Vec<u32> {
        (first..first + count)
            .map(|position| self.position_id(position))
            .collect()
    }

    /// The lyric cursor for lyrics byte `byte`: the position row that byte
    /// occupies in the prompt, or [`EMPTY_TOKEN`] (no cursor) past the field.
    pub fn cursor_row(&self, byte: usize) -> u32 {
        if byte >= self.lyrics_bytes {
            return EMPTY_TOKEN;
        }
        (self.context_length + byte) as u32
    }

    /// Position slots of physical positions `first .. first + cursors.len()`
    /// for a model with a lyric cursor: each position's own row, then its
    /// cursor row ([`EMPTY_TOKEN`] where nothing is sung).
    pub fn cursor_position_slots(&self, first: usize, cursors: &[u32]) -> Vec<u32> {
        cursors
            .iter()
            .enumerate()
            .flat_map(|(offset, &cursor)| [self.position_id(first + offset), cursor])
            .collect()
    }

    /// Embedding row of `code` in `codebook`.
    pub fn audio_token(&self, codebook: usize, code: u32) -> u32 {
        (TEXT_VOCABULARY + codebook * self.codebook_size) as u32 + code
    }

    /// Writes the prompt's token slots (`prompt_positions × codebooks`) into
    /// `slots`. Text past either field's length is dropped, and so are lyrics
    /// for a model without a lyrics field.
    pub fn write_prompt(&self, conditioning: Conditioning, slots: &mut [u32]) {
        assert_eq!(slots.len(), self.prompt_positions() * self.codebooks, "prompt slot buffer has the wrong size");
        slots.fill(EMPTY_TOKEN);

        let mut caption = conditioning.caption.bytes();
        let mut lyrics = conditioning.lyrics.bytes();
        for position in 0..self.prompt_positions() {
            let token = match self.region(position) {
                Region::Begin => BEGIN,
                Region::Caption => caption.next().map_or(PADDING, u32::from),
                Region::Lyrics => lyrics.next().map_or(PADDING, u32::from),
                Region::Separator | Region::Audio => SEPARATOR,
            };
            slots[position * self.codebooks] = token;
        }
    }

    /// The prompt's token slots as a new vector.
    pub fn prompt_slots(&self, conditioning: Conditioning) -> Vec<u32> {
        let mut slots = vec![EMPTY_TOKEN; self.prompt_positions() * self.codebooks];
        self.write_prompt(conditioning, &mut slots);
        slots
    }
}

/// The *delay pattern* of MusicGen (Copet et al., 2023,
/// <https://arxiv.org/abs/2306.05284>, §2.2): codebook `c` is shifted right
/// by `c` steps, so at each step the model predicts the coarse code of frame
/// `t` together with finer codes of earlier frames whose coarse codes are
/// already in context.
///
/// ```text
/// step:       0      1      2      3
/// codebook 0  a₀     a₁     a₂     a₃
/// codebook 1  ·      b₀     b₁     b₂
/// codebook 2  ·      ·      c₀     c₁
/// ```
#[derive(Clone, Copy, Debug)]
pub struct DelayPattern {
    /// Number of codebooks.
    pub codebooks: usize,
}

impl DelayPattern {
    /// Steps in the delayed stream for `frames` undelayed frames.
    pub fn delayed_steps(&self, frames: usize) -> usize {
        frames + self.codebooks - 1
    }

    /// The code of delayed step `step`, codebook `codebook`, or `None` in the
    /// empty triangles at either end.
    pub fn delayed_code(&self, codes: &[u32], step: usize, codebook: usize) -> Option<u32> {
        let frames = codes.len() / self.codebooks;
        let frame = step.checked_sub(codebook)?;
        (frame < frames).then(|| codes[frame * self.codebooks + codebook])
    }

    /// Undelays a stream `[delayed steps, codebooks]` back to `[frames, codebooks]`.
    pub fn undelay(&self, delayed: &[u32]) -> Vec<u32> {
        let frames = (delayed.len() / self.codebooks).saturating_sub(self.codebooks - 1);
        let mut codes = vec![0; frames * self.codebooks];

        for frame in 0..frames {
            for codebook in 0..self.codebooks {
                codes[frame * self.codebooks + codebook] = delayed[(frame + codebook) * self.codebooks + codebook];
            }
        }

        codes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_then_undelay_is_identity() {
        let pattern = DelayPattern { codebooks: 4 };
        let codes: Vec<u32> = (0..20).collect();
        let steps = pattern.delayed_steps(5);

        let mut delayed = vec![EMPTY_TOKEN; steps * 4];
        for step in 0..steps {
            for codebook in 0..4 {
                if let Some(code) = pattern.delayed_code(&codes, step, codebook) {
                    delayed[step * 4 + codebook] = code;
                }
            }
        }

        assert_eq!(delayed[1], EMPTY_TOKEN);
        assert_eq!(pattern.undelay(&delayed), codes);
    }

    #[test]
    fn prompt_is_framed_and_padded() {
        let vocabulary = Vocabulary { codebooks: 2, codebook_size: 8, prompt_bytes: 4, lyrics_bytes: 0, context_length: 16 };
        let slots = vocabulary.prompt_slots(Conditioning { caption: "hi", lyrics: "ignored" });
        let first_slots: Vec<u32> = slots.iter().step_by(2).copied().collect();

        assert_eq!(first_slots, vec![BEGIN, b'h' as u32, b'i' as u32, PADDING, PADDING, SEPARATOR]);
        assert!(
            slots
                .iter()
                .skip(1)
                .step_by(2)
                .all(|&slot| slot == EMPTY_TOKEN)
        );
    }

    #[test]
    fn lyrics_sit_between_caption_and_separator() {
        let vocabulary = Vocabulary { codebooks: 1, codebook_size: 8, prompt_bytes: 2, lyrics_bytes: 3, context_length: 16 };
        let slots = vocabulary.prompt_slots(Conditioning { caption: "a", lyrics: "la la" });

        assert_eq!(slots, vec![BEGIN, b'a' as u32, PADDING, b'l' as u32, b'a' as u32, b' ' as u32, SEPARATOR]);
    }

    #[test]
    fn lyrics_use_new_position_rows_and_keep_the_others() {
        let with_lyrics = Vocabulary { codebooks: 1, codebook_size: 8, prompt_bytes: 2, lyrics_bytes: 3, context_length: 16 };
        let without_lyrics = Vocabulary { lyrics_bytes: 0, ..with_lyrics };

        assert_eq!(with_lyrics.position_ids(0, 9), vec![0, 1, 2, 16, 17, 18, 3, 4, 5]);
        assert_eq!(without_lyrics.position_ids(0, 6), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn the_cursor_points_at_the_lyrics_rows() {
        let vocabulary = Vocabulary { codebooks: 1, codebook_size: 8, prompt_bytes: 2, lyrics_bytes: 3, context_length: 16 };

        assert_eq!(vocabulary.cursor_row(0), 16);
        assert_eq!(vocabulary.cursor_row(2), 18);
        assert_eq!(vocabulary.cursor_row(3), EMPTY_TOKEN);
        assert_eq!(vocabulary.cursor_position_slots(6, &[17, EMPTY_TOKEN]), vec![3, 17, 4, EMPTY_TOKEN]);
    }

    #[test]
    fn a_line_is_swept_at_its_pace_but_not_slower_than_the_cap() {
        assert_eq!(sung_byte(0.0, 2.0, 4), Some(0));
        assert_eq!(sung_byte(0.6, 2.0, 4), Some(2));
        assert_eq!(sung_byte(-0.1, 2.0, 4), None);
        assert_eq!(sung_byte(2.0, 2.0, 4), None);

        let capped = 4.0 * MAX_SECONDS_PER_LYRIC_BYTE;
        assert_eq!(sung_byte(capped - 0.01, 30.0, 4), Some(3));
        assert_eq!(sung_byte(capped, 30.0, 4), None);
        assert_eq!(sung_byte(0.0, 2.0, 0), None);
    }
}
