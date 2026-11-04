//! Model hyperparameters.

use serde::{Deserialize, Serialize};

/// Architecture of the Transformer. Everything else (vocabulary layout, tensor
/// shapes) is derived from these numbers.
///
/// The default is a small model in the spirit of GPT-2 small scaled down
/// (Radford et al., 2019), about 8 M parameters, sized for an edge GPU.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Width of the residual stream, `d_model`.
    pub width: usize,

    /// Attention heads per layer; must divide `width`.
    pub heads: usize,

    /// Number of Transformer blocks.
    pub layers: usize,

    /// Hidden width of the feed-forward network as a multiple of `width`
    /// (4 in Vaswani et al., 2017, <https://arxiv.org/abs/1706.03762>, §3.3).
    pub feed_forward_multiplier: usize,

    /// Maximum length of the caption part plus the audio part of a sequence.
    /// The lyrics field comes on top; see [`ModelConfig::sequence_capacity`].
    pub context_length: usize,

    /// UTF-8 bytes of caption text the model reads; longer captions are truncated.
    pub prompt_bytes: usize,

    /// UTF-8 bytes of lyrics the model reads per sequence, 0 for a model
    /// without lyrics. Absent from older checkpoints, hence the default.
    #[serde(default)]
    pub lyrics_bytes: usize,

    /// Residual codebooks per audio step (tokens per position).
    pub codebooks: usize,

    /// Entries per codebook.
    pub codebook_size: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            width: 256,
            heads: 8,
            layers: 6,
            feed_forward_multiplier: 4,
            context_length: 768,
            prompt_bytes: 126,
            lyrics_bytes: 0,
            codebooks: 4,
            codebook_size: 1024,
        }
    }
}

impl ModelConfig {
    /// Width of one attention head.
    pub fn head_dim(&self) -> usize {
        self.width / self.heads
    }

    /// Hidden width of the feed-forward network.
    pub fn feed_forward_width(&self) -> usize {
        self.width * self.feed_forward_multiplier
    }

    /// Longest physical sequence: caption, lyrics and audio together. Sizes
    /// the position table and the KV caches.
    pub fn sequence_capacity(&self) -> usize {
        self.context_length + self.lyrics_bytes
    }

    /// Output logits per position: one block of `codebook_size` per codebook.
    pub fn logits_width(&self) -> usize {
        self.codebooks * self.codebook_size
    }

    /// Panics with a clear message if the configuration is inconsistent.
    pub fn validate(&self) {
        assert!(self.width.is_multiple_of(self.heads), "width {} is not divisible by heads {}", self.width, self.heads);
        assert!(self.prompt_bytes + 2 < self.context_length, "prompt does not fit in the context");
        assert!(self.codebooks >= 1 && self.codebook_size >= 2, "need at least one codebook of two entries");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_small_model() {
        let config = ModelConfig::default();

        assert_eq!(config.width, 256);
        assert_eq!(config.heads, 8);
        assert_eq!(config.layers, 6);
        assert_eq!(config.feed_forward_multiplier, 4);
        assert_eq!(config.context_length, 768);
        assert_eq!(config.prompt_bytes, 126);
        assert_eq!(config.lyrics_bytes, 0);
        assert_eq!(config.codebooks, 4);
        assert_eq!(config.codebook_size, 1024);
    }

    #[test]
    fn derived_widths_are_computed_from_the_config() {
        let config = ModelConfig::default();

        assert_eq!(config.head_dim(), 32);
        assert_eq!(config.feed_forward_width(), 1024);
        assert_eq!(config.sequence_capacity(), 768);
        assert_eq!(config.logits_width(), 4096);

        let with_lyrics = ModelConfig { lyrics_bytes: 256, ..config };

        assert_eq!(with_lyrics.sequence_capacity(), 1024);
    }

    #[test]
    fn validate_accepts_the_default() {
        ModelConfig::default().validate();
    }

    #[test]
    #[should_panic(expected = "not divisible by heads")]
    fn validate_rejects_an_indivisible_width() {
        ModelConfig { width: 10, heads: 4, ..ModelConfig::default() }.validate();
    }

    #[test]
    #[should_panic(expected = "prompt does not fit")]
    fn validate_rejects_a_prompt_that_does_not_fit() {
        ModelConfig { context_length: 10, prompt_bytes: 9, ..ModelConfig::default() }.validate();
    }

    #[test]
    #[should_panic(expected = "at least one codebook")]
    fn validate_rejects_an_empty_codebook() {
        ModelConfig { codebooks: 0, ..ModelConfig::default() }.validate();
    }
}
