//! Embedding lookup and its gradient.
//!
//! A position may carry several tokens (one per audio codebook); their
//! embeddings are summed, as in MusicGen (Copet et al., 2023,
//! <https://arxiv.org/abs/2306.05284>, §2.2).

use super::WriteMode;
use crate::device::{Gpu, GpuResult, Tensor, TokenBuffer};
use crate::kernels::EMPTY_TOKEN;
use crate::launch;

/// Inverted index from token id to the positions that used it, in CSR form
/// (<https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>).
///
/// It lets the embedding gradient be computed by gathering instead of
/// scattering, which needs no atomics and is bit-for-bit deterministic.
pub struct TokenIndex {
    /// `offsets[t]..offsets[t+1]` is token `t`'s range in `positions`; length `vocabulary + 1`.
    offsets: TokenBuffer,

    /// Positions grouped by token.
    positions: TokenBuffer,

    /// Number of rows in the embedding table.
    vocabulary: usize,
}

impl TokenIndex {
    /// Builds the index for `tokens` `[positions, slots]` on the host and uploads it.
    pub fn build(gpu: &Gpu, tokens: &[u32], slots: usize, vocabulary: usize) -> GpuResult<Self> {
        let (offsets, positions) = invert_tokens(tokens, slots, vocabulary);

        Ok(Self { offsets: gpu.upload_tokens(&offsets)?, positions: gpu.upload_tokens(&positions)?, vocabulary })
    }
}

/// Counting sort of positions by token: returns (offsets, positions).
pub fn invert_tokens(tokens: &[u32], slots: usize, vocabulary: usize) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = vec![0u32; vocabulary + 1];
    for &token in tokens.iter().filter(|&&token| token != EMPTY_TOKEN) {
        offsets[token as usize + 1] += 1;
    }
    for token in 0..vocabulary {
        offsets[token + 1] += offsets[token];
    }

    let mut next_free = offsets.clone();
    // At least one element so the device buffer is never empty.
    let mut positions = vec![0u32; (offsets[vocabulary] as usize).max(1)];
    for (flat_index, &token) in tokens.iter().enumerate() {
        if token == EMPTY_TOKEN {
            continue;
        }

        let cursor = &mut next_free[token as usize];
        positions[*cursor as usize] = (flat_index / slots) as u32;
        *cursor += 1;
    }

    (offsets, positions)
}

impl Gpu {
    /// `output[p] (+)= Σ_s table[tokens[p·slots + s]]`, skipping empty slots.
    pub fn embedding_forward(
        &self,
        table: &Tensor,
        tokens: &TokenBuffer,
        slots: usize,
        output: &mut Tensor,
        write: WriteMode,
    ) -> GpuResult {
        let width = table.shape().cols;
        assert_eq!(output.shape().cols, width, "embedding_forward: width mismatch");
        assert!(tokens.len() >= output.shape().rows * slots, "embedding_forward: too few tokens");

        // SAFETY: one thread per output element; out-of-range threads return early.
        unsafe {
            self.kernels.embedding_forward(
                &self.stream,
                launch::elementwise(output.len()),
                &table.buffer,
                tokens,
                &mut output.buffer,
                slots,
                width,
                write.flag(),
            )
        }
    }

    /// `table_grad[t] += Σ_{p uses t} output_grad[p]`.
    pub fn embedding_backward(&self, output_grad: &Tensor, index: &TokenIndex, table_grad: &mut Tensor) -> GpuResult {
        let width = table_grad.shape().cols;
        assert_eq!(table_grad.shape().rows, index.vocabulary, "embedding_backward: vocabulary mismatch");
        assert_eq!(output_grad.shape().cols, width, "embedding_backward: width mismatch");

        // SAFETY: one thread per table element; out-of-range threads return early.
        unsafe {
            self.kernels.embedding_backward(
                &self.stream,
                launch::elementwise(table_grad.len()),
                &output_grad.buffer,
                &index.offsets,
                &index.positions,
                &mut table_grad.buffer,
                width,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invert_tokens_groups_positions_by_token() {
        let tokens = [0, 1, 0, EMPTY_TOKEN, 2, 0];
        let (offsets, positions) = invert_tokens(&tokens, 2, 3);

        assert_eq!(offsets, [0, 3, 4, 5]);
        assert_eq!(positions, [0, 1, 2, 0, 2]);
    }

    #[test]
    fn empty_tokens_produce_empty_ranges() {
        let tokens = [EMPTY_TOKEN, EMPTY_TOKEN];
        let (offsets, positions) = invert_tokens(&tokens, 1, 2);

        assert_eq!(offsets, [0, 0, 0]);
        assert_eq!(positions, [0]);
    }

    #[test]
    fn repeated_tokens_share_one_bucket() {
        let tokens = [1, 1, 1, 1];
        let (offsets, positions) = invert_tokens(&tokens, 2, 2);

        assert_eq!(offsets, [0, 0, 4]);
        assert_eq!(positions, [0, 0, 1, 1]);
    }

    #[test]
    fn zero_vocabulary_still_returns_a_device_sized_position_buffer() {
        let (offsets, positions) = invert_tokens(&[], 1, 0);

        assert_eq!(offsets, [0]);
        assert_eq!(positions, [0]);
    }
}
