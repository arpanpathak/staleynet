//! StaleyNET's acoustic language model: a decoder-only Transformer that reads a
//! text prompt as bytes and continues it with audio tokens.
//!
//! * [`gpt`]: the network. [`layers`]: its building blocks.
//! * [`training`]: teacher-forced loss and backward pass.
//! * [`optimizer`]: AdamW and the learning-rate schedule.
//! * [`decode`], [`generate`]: cached incremental inference and sampling.
//! * [`checkpoint`]: saving and restoring weights and optimiser state.

pub mod checkpoint;
pub mod config;
pub mod decode;
pub mod generate;
pub mod gpt;
pub mod layers;
pub mod optimizer;
pub mod parameter;
pub mod random;
pub mod training;
pub mod vocabulary;

pub use config::ModelConfig;
pub use gpt::Gpt;
pub use layers::SequenceShape;
pub use parameter::Parameters;
pub use random::Random;
pub use training::{LossReport, TrainingBatch, TrainingWorkspace};
pub use vocabulary::{Conditioning, DelayPattern, Vocabulary};
