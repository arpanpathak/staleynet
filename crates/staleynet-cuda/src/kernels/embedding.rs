//! Embedding lookup and its gradient.
//!
//! A position may carry several token slots (one per codebook); its embedding
//! is the sum of the rows of its tokens. The gradient is gathered through an
//! index built on the host, so that it is summed in the same order on every
//! run (see [`crate::ops::TokenIndex`]).

use cuda_device::cuda_module;

#[cuda_module]
pub mod kernels {
    use crate::kernels::EMPTY_TOKEN;
    use cuda_device::{DisjointSlice, kernel, thread};

    /// Sums the embeddings of every position's tokens:
    /// `output[p, c] (+)= Σ_s table[tokens[p · slots + s], c]`, skipping empty slots.
    /// One thread per output element.
    ///
    /// * `table`: the embedding table, `[vocabulary, width]`.
    /// * `tokens`: `slots` token ids per position; `EMPTY_TOKEN` marks an empty slot.
    /// * `output`: `[positions, width]`.
    /// * `slots`: token slots per position.
    /// * `width`: the embedding width.
    /// * `accumulate`: 1 to add to `output`, 0 to overwrite it.
    #[kernel]
    pub fn embedding_forward(table: &[f32], tokens: &[u32], mut output: DisjointSlice<f32>, slots: usize, width: usize, accumulate: u32) {
        let index = thread::index_1d();
        let element = index.get();
        let position = element / width;
        let col = element % width;

        let mut sum = 0.0f32;
        for slot in 0..slots {
            let token = tokens
                .get(position * slots + slot)
                .copied()
                .unwrap_or(EMPTY_TOKEN);
            if token != EMPTY_TOKEN {
                sum += table[token as usize * width + col];
            }
        }

        if let Some(value) = output.get_mut(index) {
            *value = if accumulate != 0 { *value + sum } else { sum };
        }
    }

    /// Adds the gradient of the embedding table without atomic additions:
    /// the thread for `(token, col)` sums the gradients of the positions that
    /// used `token`, in the order listed by the host-built index. One thread
    /// per element of the table.
    ///
    /// * `output_grad`: the gradient with respect to the embeddings, `[positions, width]`.
    /// * `row_offsets`: for each token, where its positions start in `rows_by_token`
    ///   (one more entry than there are tokens).
    /// * `rows_by_token`: positions grouped by token.
    /// * `table_grad`: the gradient of the table, `[vocabulary, width]`, added to.
    /// * `width`: the embedding width.
    #[kernel]
    pub fn embedding_backward(
        output_grad: &[f32],
        row_offsets: &[u32],
        rows_by_token: &[u32],
        mut table_grad: DisjointSlice<f32>,
        width: usize,
    ) {
        let index = thread::index_1d();
        let element = index.get();
        let token = element / width;
        let col = element % width;

        if token + 1 >= row_offsets.len() {
            return;
        }

        let mut sum = 0.0f32;
        for entry in row_offsets[token] as usize..row_offsets[token + 1] as usize {
            sum += output_grad[rows_by_token[entry] as usize * width + col];
        }

        if let Some(value) = table_grad.get_mut(index) {
            *value += sum;
        }
    }
}
