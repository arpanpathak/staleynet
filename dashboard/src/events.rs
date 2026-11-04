//! The events a training run appends to its `metrics.jsonl`.
//!
//! Both trainers write one JSON object per line, tagged by `"event"`. The
//! language model's lines carry a loss per codebook; the vocoder's carry a
//! loss per spectral resolution. Fields one trainer does not write are
//! optional here, and lines that are not events (or not yet complete, while
//! the trainer is writing them) are skipped.

use serde::Deserialize;
use std::path::Path;

/// One line of `metrics.jsonl`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    /// The trainer started, or resumed.
    Started(Started),

    /// Means over the steps since the previous progress line.
    Progress(Progress),

    /// The loss on held-out data.
    Validation(Validation),

    /// Listening samples were written (vocoder runs).
    Samples(Samples),

    /// An event this dashboard does not chart.
    #[serde(other)]
    Other,
}

/// The start of a training session.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct Started {
    /// The step training starts from: 0, or the step of the checkpoint resumed from.
    pub first_step: Option<u64>,

    /// Trainable parameters of the model.
    pub parameters: Option<u64>,

    /// Training tracks the run samples from.
    pub train_tracks: Option<u64>,

    /// Held-out tracks.
    pub validation_tracks: Option<u64>,

    /// Training tracks that carry lyrics.
    pub tracks_with_lyrics: Option<u64>,
}

/// The training loss and the optimizer's state over a span of steps.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Progress {
    /// The last step of the span.
    pub step: u64,

    /// Mean training loss over the span, in nats per code (language model)
    /// or in the weighted spectral loss (vocoder).
    pub loss: f64,

    /// The language model's loss for each codebook.
    #[serde(default)]
    pub loss_per_codebook: Vec<f64>,

    /// The vocoder's loss at each spectral resolution.
    #[serde(default)]
    pub loss_parts: Vec<f64>,

    /// Norm of the gradient before clipping.
    pub gradient_norm: Option<f64>,

    /// The learning rate at the last step.
    pub learning_rate: Option<f64>,

    /// Mean wall-clock time of one step.
    pub milliseconds_per_step: Option<f64>,
}

/// The loss on held-out data at one step.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Validation {
    /// The step it was measured at.
    pub step: u64,

    /// Mean held-out loss.
    pub loss: f64,

    /// The language model's held-out loss for each codebook.
    #[serde(default)]
    pub loss_per_codebook: Vec<f64>,

    /// The vocoder's held-out loss at each spectral resolution.
    #[serde(default)]
    pub loss_parts: Vec<f64>,
}

/// Listening samples written at one step.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Samples {
    /// The step they were rendered at.
    pub step: u64,

    /// Paths of the rendered files.
    #[serde(default)]
    pub files: Vec<String>,
}

/// Every event of a run, in the order they were written.
pub fn read(metrics: &Path) -> Vec<Event> {
    let Ok(text) = std::fs::read_to_string(metrics) else { return Vec::new() };

    text.lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_model_lines_parse() {
        let progress = r#"{"event":"progress","step":25,"loss":6.9,"loss_per_codebook":[6.8,6.9],"gradient_norm":0.99,"learning_rate":1e-5,"positions_per_second":3341.4,"milliseconds_per_step":3370.9}"#;
        let Ok(Event::Progress(parsed)) = serde_json::from_str::<Event>(progress) else { panic!("not progress") };

        assert_eq!(parsed.step, 25);
        assert_eq!(parsed.loss_per_codebook, vec![6.8, 6.9]);
        assert!(parsed.loss_parts.is_empty());
        assert_eq!(parsed.milliseconds_per_step, Some(3370.9));
    }

    #[test]
    fn vocoder_lines_parse() {
        let validation = r#"{"event":"validation","loss":1.76,"loss_parts":[0.55,0.53,0.67],"step":1000}"#;
        let Ok(Event::Validation(parsed)) = serde_json::from_str::<Event>(validation) else { panic!("not validation") };

        assert_eq!(parsed.step, 1000);
        assert_eq!(parsed.loss_parts.len(), 3);
    }

    #[test]
    fn unknown_events_and_broken_lines_are_skipped() {
        let directory = std::env::temp_dir().join(format!("dashboard-events-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("metrics.jsonl");

        let lines = [
            r#"{"event":"started","first_step":0,"parameters":42858496}"#,
            r#"{"event":"checkpoint","step":250}"#,
            r#"not json"#,
            r#"{"event":"validation","step":500,"loss":6.25}"#,
            r#"{"event":"progress","step":5"#,
        ];
        std::fs::write(&path, lines.join("\n")).unwrap();

        let events = read(&path);
        assert_eq!(events.len(), 3);
        assert!(matches!(events[0], Event::Started(_)));
        assert_eq!(events[1], Event::Other);
        assert!(matches!(events[2], Event::Validation(_)));

        std::fs::remove_dir_all(directory).unwrap();
    }
}
