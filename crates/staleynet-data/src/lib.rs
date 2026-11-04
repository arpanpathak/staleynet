//! From raw recordings to training batches.
//!
//! * [`catalog`]: which recordings exist, their captions and split.
//! * [`prepare`]: decode, resample, normalise and analyse into log-mels (GPU).
//! * [`store`]: the on-disk cache layout and manifest.
//! * [`tokenizer`]: residual k-means quantiser: log-mel ↔ discrete codes.
//! * [`corpus`]: tokenised tracks and the batch sampler.

pub mod catalog;
pub mod corpus;
pub mod features;
pub mod prepare;
pub mod store;
pub mod tokenizer;
