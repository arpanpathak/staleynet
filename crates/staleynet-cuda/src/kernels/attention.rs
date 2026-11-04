//! Building blocks of causal multi-head attention (Vaswani et al., 2017,
//! <https://arxiv.org/abs/1706.03762>, §3.2): moving heads between the packed
//! and the head-major layout, the masked softmax and its gradient. The two
//! products of attention are batched matrix products (see `matmul`).

use cuda_device::cuda_module;

/// Where a group of attention heads lives in two layouts, passed to the
/// permute kernels by value.
///
/// In the **packed** layout `[batch · time, width]`, head `h` of a row occupies
/// the columns `column_offset + h · head_dim ..`; the offset selects the
/// queries, keys or values inside a fused projection. In the **head-major**
/// layout `[batch, heads, time_capacity, head_dim]`, each head's time steps
/// are contiguous, and the steps moved are written from `time_offset` on,
/// which lets the same kernels append to a key/value cache.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadLayout {
    /// Sequences in the batch.
    pub batch: usize,

    /// Time steps moved.
    pub time: usize,

    /// Attention heads.
    pub heads: usize,

    /// Width of one head.
    pub head_dim: usize,

    /// Row width of the packed layout.
    pub width: usize,

    /// First packed column of head 0.
    pub column_offset: usize,

    /// Time steps allocated per head in the head-major layout.
    pub time_capacity: usize,

    /// First head-major time step written or read.
    pub time_offset: usize,
}

/// The coordinates of one moved element.
#[derive(Clone, Copy, Debug)]
pub struct HeadElement {
    /// Sequence.
    pub batch: usize,

    /// Head.
    pub head: usize,

    /// Time step, counted from the first step moved.
    pub step: usize,

    /// Position within the head.
    pub dim: usize,
}

impl HeadLayout {
    /// A layout for dense activations: the capacity equals `time` and the offset is 0.
    pub fn dense(batch: usize, time: usize, heads: usize, head_dim: usize, width: usize, column_offset: usize) -> Self {
        Self { batch, time, heads, head_dim, width, column_offset, time_capacity: time, time_offset: 0 }
    }

    /// The same layout, addressing a cache of `capacity` steps from `offset` on.
    pub fn in_cache(self, capacity: usize, offset: usize) -> Self {
        Self { time_capacity: capacity, time_offset: offset, ..self }
    }

    /// Elements moved by one permute: `batch · heads · time · head_dim`.
    pub fn element_count(&self) -> usize {
        self.batch * self.heads * self.time * self.head_dim
    }

    /// The coordinates of moved element `element`, enumerated with `dim`
    /// varying fastest, then `step`, `head` and `batch`.
    pub fn element(&self, element: usize) -> HeadElement {
        HeadElement {
            dim: element % self.head_dim,
            step: (element / self.head_dim) % self.time,
            head: (element / (self.head_dim * self.time)) % self.heads,
            batch: element / (self.head_dim * self.time * self.heads),
        }
    }

    /// Position of an element in the packed layout.
    pub fn packed_index(&self, at: HeadElement) -> usize {
        (at.batch * self.time + at.step) * self.width + self.column_offset + at.head * self.head_dim + at.dim
    }

    /// Position of an element in the head-major layout.
    pub fn head_major_index(&self, at: HeadElement) -> usize {
        ((at.batch * self.heads + at.head) * self.time_capacity + self.time_offset + at.step) * self.head_dim + at.dim
    }
}

#[cuda_module]
pub mod kernels {
    use super::HeadLayout;
    use crate::kernels::device::{WarpRow, global_thread, load_owned, store, warp_row};
    use crate::kernels::{NEGATIVE_LARGE, WARP_SIZE};
    use cuda_device::{DisjointSlice, kernel, warp};

    /// Copies heads from the packed layout into the head-major layout.
    /// One thread per moved element.
    ///
    /// * `packed`: the source, `[batch · time, width]`.
    /// * `head_major`: the destination, `[batch, heads, time_capacity, head_dim]`.
    /// * `layout`: where the heads live in both.
    #[kernel]
    pub fn split_heads(packed: &[f32], mut head_major: DisjointSlice<f32>, layout: HeadLayout) {
        let element = global_thread();
        if element >= layout.element_count() {
            return;
        }
        let at = layout.element(element);
        store(&mut head_major, layout.head_major_index(at), packed[layout.packed_index(at)]);
    }

    /// The inverse of [`split_heads`]: copies heads from the head-major layout
    /// back into their packed columns, leaving other columns untouched.
    ///
    /// * `head_major`: the source.
    /// * `packed`: the destination.
    /// * `layout`: where the heads live in both.
    #[kernel]
    pub fn merge_heads(head_major: &[f32], mut packed: DisjointSlice<f32>, layout: HeadLayout) {
        let element = global_thread();
        if element >= layout.element_count() {
            return;
        }
        let at = layout.element(element);
        store(&mut packed, layout.packed_index(at), head_major[layout.head_major_index(at)]);
    }

    /// Causal softmax in place, one warp per row of scores. Query `q` of a
    /// matrix may attend to keys `0 ..= q + query_offset`; later keys receive
    /// probability 0. The row maximum is subtracted before exponentiating, so
    /// that large scores cannot overflow
    /// (<https://en.wikipedia.org/wiki/Softmax_function#Numerical_algorithms>).
    ///
    /// * `scores`: `[matrices · queries, keys]`, replaced by probabilities.
    /// * `rows`: `matrices · queries`.
    /// * `queries`: rows per matrix.
    /// * `keys`: columns.
    /// * `scale`: applied to the scores before the softmax (1/√head width).
    /// * `query_offset`: positions already cached before the first query.
    #[kernel]
    pub fn causal_softmax(mut scores: DisjointSlice<f32>, rows: usize, queries: usize, keys: usize, scale: f32, query_offset: usize) {
        let WarpRow { row, lane } = warp_row();
        if row >= rows {
            return;
        }
        let base = row * keys;
        let visible_keys = (row % queries + query_offset + 1).min(keys);

        let mut maximum = NEGATIVE_LARGE;
        for key in (lane..visible_keys).step_by(WARP_SIZE) {
            maximum = maximum.max(load_owned(&mut scores, base + key) * scale);
        }
        let maximum = warp::reduce_max_f32(maximum);

        let mut total = 0.0f32;
        for key in (lane..visible_keys).step_by(WARP_SIZE) {
            let exponential = libm::expf(load_owned(&mut scores, base + key) * scale - maximum);
            store(&mut scores, base + key, exponential);
            total += exponential;
        }
        let normaliser = 1.0 / warp::reduce_sum_f32(total);

        for key in (lane..keys).step_by(WARP_SIZE) {
            let probability = if key < visible_keys { load_owned(&mut scores, base + key) * normaliser } else { 0.0 };
            store(&mut scores, base + key, probability);
        }
    }

    /// The gradient of a row-wise softmax, in place:
    /// `grad ← p ⊙ (grad − Σ(grad ⊙ p))` for every row. One warp per row.
    ///
    /// * `probabilities`: the softmax output `p`, `[rows, cols]`.
    /// * `grad`: on entry the gradient with respect to `p`; on exit the
    ///   gradient with respect to the softmax input.
    /// * `rows`, `cols`: the size of both matrices.
    #[kernel]
    pub fn softmax_backward(probabilities: &[f32], mut grad: DisjointSlice<f32>, rows: usize, cols: usize) {
        let WarpRow { row, lane } = warp_row();
        if row >= rows {
            return;
        }
        let base = row * cols;

        let mut projection = 0.0f32;
        for col in (lane..cols).step_by(WARP_SIZE) {
            projection += load_owned(&mut grad, base + col) * probabilities[base + col];
        }
        let projection = warp::reduce_sum_f32(projection);

        for col in (lane..cols).step_by(WARP_SIZE) {
            let gradient = probabilities[base + col] * (load_owned(&mut grad, base + col) - projection);
            store(&mut grad, base + col, gradient);
        }
    }
}
