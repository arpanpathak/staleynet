//! Incremental inference with a key/value cache (Pope et al., 2022,
//! "Efficiently Scaling Transformer Inference", <https://arxiv.org/abs/2211.05102>).
//!
//! Each call feeds new positions, appends their keys and values to the
//! per-layer caches, and returns the logits of the last fed position. Without
//! a cache every new token would recompute attention over the whole prefix.

use crate::gpt::Gpt;
use crate::layers::attention::LayerCache;
use crate::layers::block::CachedBlockTensors;
use staleynet_cuda::{Gpu, GpuResult, Shape, Tensor, TokenBuffer};
use std::collections::HashMap;
use std::collections::hash_map::Entry;

/// Buffers for feeding a fixed number of new positions.
struct FeedScratch {
    /// Token slots of the new positions.
    tokens: TokenBuffer,

    /// Position-table rows of the new positions.
    positions: TokenBuffer,

    /// Residual stream entering the current block.
    residual: Tensor,

    /// Residual stream leaving the current block.
    next_residual: Tensor,

    /// Per-block scratch, shared by all blocks (they run one after another).
    block: CachedBlockTensors,

    /// Final-norm output.
    final_normed: Tensor,

    /// Final-norm `1/rms`.
    final_inv_rms: Tensor,

    /// Logits of every new position.
    logits: Tensor,
}

impl FeedScratch {
    /// Scratch for feeding `new_positions` positions to `model`.
    fn new(gpu: &Gpu, model: &Gpt, new_positions: usize) -> GpuResult<Self> {
        let config = &model.config;
        let stream = Shape::new(new_positions, config.width);
        let placeholder = vec![0; new_positions * config.codebooks];

        Ok(Self {
            tokens: gpu.upload_tokens(&placeholder)?,
            positions: gpu.upload_tokens(&placeholder[..new_positions])?,
            residual: gpu.zeros(stream)?,
            next_residual: gpu.zeros(stream)?,
            block: CachedBlockTensors::new(gpu, config, new_positions, config.sequence_capacity())?,
            final_normed: gpu.zeros(stream)?,
            final_inv_rms: gpu.zeros(Shape::vector(new_positions))?,
            logits: gpu.zeros(Shape::new(new_positions, config.logits_width()))?,
        })
    }
}

/// Decoding state of one sequence: KV caches plus reusable scratch.
pub struct Decoder {
    /// One cache per block.
    caches: Vec<LayerCache>,

    /// Positions already in the caches.
    cached_positions: usize,

    /// Maximum positions the caches hold (the model's sequence capacity).
    capacity: usize,

    /// Scratch keyed by how many positions a call feeds (typically the
    /// prefill length and 1), so steady-state decoding allocates nothing.
    scratch: HashMap<usize, FeedScratch>,
}

impl Decoder {
    /// Empty caches for `model`.
    pub fn new(gpu: &Gpu, model: &Gpt) -> GpuResult<Self> {
        let config = &model.config;
        let mut caches = Vec::with_capacity(config.layers);
        for _ in 0..config.layers {
            caches.push(LayerCache::new(gpu, config.sequence_capacity(), config.width, config.heads)?);
        }

        Ok(Self { caches, cached_positions: 0, capacity: config.sequence_capacity(), scratch: HashMap::new() })
    }

    /// Positions currently cached.
    pub fn cached_positions(&self) -> usize {
        self.cached_positions
    }

    /// Positions that can still be fed before the context is full.
    pub fn remaining(&self) -> usize {
        self.capacity - self.cached_positions
    }

    /// Forgets the sequence. Stale cache contents stay in memory but are
    /// never attended to: the causal mask only exposes rewritten positions.
    pub fn reset(&mut self) {
        self.cached_positions = 0;
    }

    /// Frees scratch buffers that are not for `keep_positions`-sized feeds.
    pub fn release_scratch_except(&mut self, keep_positions: usize) {
        self.scratch.retain(|&positions, _| positions == keep_positions);
    }

    /// Feeds `token_slots` (`[new positions, codebooks]`) and returns the
    /// logits of the last new position.
    pub fn feed(&mut self, gpu: &Gpu, model: &Gpt, token_slots: &[u32]) -> GpuResult<Vec<f32>> {
        let codebooks = model.config.codebooks;
        let new_positions = token_slots.len() / codebooks;
        assert!(new_positions > 0 && new_positions <= self.remaining(), "feed exceeds the context window");

        let scratch = match self.scratch.entry(new_positions) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(FeedScratch::new(gpu, model, new_positions)?),
        };

        let positions = model.vocabulary().position_ids(self.cached_positions, new_positions);
        gpu.copy_tokens_from_host(&mut scratch.tokens, token_slots)?;
        gpu.copy_tokens_from_host(&mut scratch.positions, &positions)?;

        model.embed(gpu, &scratch.tokens, &scratch.positions, &mut scratch.residual)?;

        for (block, cache) in model.blocks.iter().zip(&mut self.caches) {
            block.forward_cached(gpu, &scratch.residual, cache, self.cached_positions, &mut scratch.block, &mut scratch.next_residual)?;
            std::mem::swap(&mut scratch.residual, &mut scratch.next_residual);
        }
        self.cached_positions += new_positions;

        model.final_norm.forward(gpu, &scratch.residual, &mut scratch.final_normed, &mut scratch.final_inv_rms)?;
        model.head.forward(gpu, &scratch.final_normed, &mut scratch.logits)?;

        let logits_width = model.config.logits_width();
        let mut all_logits = gpu.download(&scratch.logits)?;
        Ok(all_logits.split_off((new_positions - 1) * logits_width))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_decoder(capacity: usize) -> Decoder {
        Decoder { caches: Vec::new(), cached_positions: 0, capacity, scratch: HashMap::new() }
    }

    #[test]
    fn cached_positions_and_remaining_track_the_window() {
        let mut decoder = empty_decoder(10);

        assert_eq!(decoder.cached_positions(), 0);
        assert_eq!(decoder.remaining(), 10);

        decoder.cached_positions = 4;

        assert_eq!(decoder.cached_positions(), 4);
        assert_eq!(decoder.remaining(), 6);
    }

    #[test]
    fn reset_returns_to_the_start_of_the_window() {
        let mut decoder = empty_decoder(10);
        decoder.cached_positions = 7;

        decoder.reset();

        assert_eq!(decoder.cached_positions(), 0);
        assert_eq!(decoder.remaining(), 10);
    }

    #[test]
    fn releasing_scratch_on_an_empty_cache_is_a_noop() {
        let mut decoder = empty_decoder(10);

        decoder.release_scratch_except(1);

        assert!(decoder.scratch.is_empty());
    }
}
