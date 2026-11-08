//! Training the neural vocoder of [`staleynet_model::vocoder`].
//!
//! The vocoder learns from pairs of a spectrogram decoded from tokens and the
//! stretch of recording it came from, so a run needs a tokenized cache whose
//! recordings were also stored as samples ([`waveforms`]).
//!
//! ```text
//! <run>/
//!   config.json        the training configuration
//!   metrics.jsonl      one JSON object per progress, validation or sample event
//!   latest/            the last checkpoint, with the optimizer's moments
//!   best/              the checkpoint with the lowest validation loss
//!   samples/reference/ recordings and their Griffin-Lim rendering, for comparison
//!   samples/step-N/    the vocoder's rendering of the same tracks at step N
//! ```
//!
//! * [`waveforms`]: writes every track's samples into the cache.
//! * [`segments`]: cuts training segments out of the cache.
//! * [`config`]: the settings of a run.
//! * [`adversary`]: the discriminator's side of adversarial training.
//! * [`progress`]: running averages and the best validation loss so far.
//! * [`session`]: the training loop, checkpoints and listening samples.

pub mod adversary;
pub mod config;
pub mod progress;
pub mod segments;
pub mod session;
pub mod waveforms;

pub use config::{AdversarialTraining, VocoderTrainingConfig};
pub use segments::SegmentSampler;
pub use session::{preferred_checkpoint, train};
pub use waveforms::{WaveformOutcome, prepare_waveforms};
