//! Network layers. Each owns its [`Parameter`](crate::parameter::Parameter)s,
//! computes its forward pass into caller-provided buffers, and accumulates
//! parameter gradients in its backward pass.

pub mod attention;
pub mod block;
pub mod feed_forward;
pub mod linear;
pub mod norm;

/// Batch size and sequence length of a training step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceShape {
    /// Independent sequences.
    pub batch: usize,

    /// Positions per sequence.
    pub time: usize,
}

impl SequenceShape {
    /// Positions across the batch, i.e. rows of every activation matrix.
    pub fn rows(&self) -> usize {
        self.batch * self.time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_multiply_batch_and_time() {
        assert_eq!(SequenceShape { batch: 2, time: 3 }.rows(), 6);
        assert_eq!(SequenceShape { batch: 0, time: 7 }.rows(), 0);
        assert_eq!(SequenceShape { batch: 5, time: 0 }.rows(), 0);
    }
}
