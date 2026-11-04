//! Training runs: where they are, and a summary of each.
//!
//! A run is any folder with a `metrics.jsonl`, inside one of the runs folders
//! the dashboard was given. Each folder is a *source* with a short label (for
//! example the machine it lives on), so runs of the same name on two machines
//! stay apart: a run's id is `<label>/<name>`.

use crate::events::{self, Event, Started};
use crate::history::{History, Point, recent_training_loss};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// File every trainer appends its events to.
const METRICS_FILE: &str = "metrics.jsonl";

/// File holding a run's configuration, including its total steps.
const CONFIG_FILE: &str = "config.json";

/// A run whose metrics changed within this many seconds is training now.
/// The slowest trainer logs every 25 steps of up to 4 seconds each.
const LIVE_SECONDS: f64 = 300.0;

/// Validations without a new best after which a run whose gap has grown since
/// its best is flagged, however flat its recent trend.
const STALLED_VALIDATIONS: usize = 4;

/// The overfitting watch looks at this many most recent validations.
const RECENT_VALIDATIONS: usize = 5;

/// Over those validations, a held-out loss that fell by less than this has
/// stalled, in loss units: it may still creep down, but by too little to matter.
const MIN_IMPROVEMENT: f64 = 0.01;

/// Over the same validations, a generalization gap that grew by more than this
/// means the training loss kept falling without the held-out loss following.
const GAP_GROWTH: f64 = 0.05;

/// A runs folder and the label its runs are shown under.
#[derive(Clone, Debug, PartialEq)]
pub struct Source {
    /// Short label, such as the machine's name.
    pub label: String,

    /// The folder that holds the runs.
    pub folder: PathBuf,
}

/// Whether a run is still training.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Its metrics changed within the last few minutes.
    Live,

    /// It reached its configured number of steps.
    Finished,

    /// It stopped before reaching them.
    Stopped,
}

/// Which trainer wrote the run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    /// The acoustic language model; losses are in nats per code.
    LanguageModel,

    /// The neural vocoder; losses are weighted spectral distances.
    Vocoder,
}

/// One run at a glance.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Summary {
    /// `<label>/<name>`, unique across sources.
    pub id: String,

    /// The run folder's name.
    pub name: String,

    /// The label of the source it was found in.
    pub source: String,

    /// Which trainer wrote it.
    pub kind: Kind,

    /// Whether it is still training.
    pub status: Status,

    /// The last step logged.
    pub step: Option<u64>,

    /// The step the run is configured to end at.
    pub target_steps: Option<u64>,

    /// Mean training loss over the most recent steps.
    pub training_loss: Option<f64>,

    /// The lowest held-out loss and its step.
    pub best_held_out: Option<Point>,

    /// The most recent held-out loss and its step.
    pub latest_held_out: Option<Point>,

    /// The most recent generalization gap.
    pub gap: Option<f64>,

    /// Validations since the best one.
    pub validations_since_best: usize,

    /// Held-out loss has stalled while the gap grows.
    pub overfitting: bool,

    /// The most recent time per step.
    pub milliseconds_per_step: Option<f64>,

    /// Seconds until `target_steps` at the current speed, while live.
    pub eta_seconds: Option<f64>,

    /// Seconds since the metrics last changed.
    pub seconds_since_update: f64,

    /// Facts from the most recent start: model size and data.
    pub started: Option<StartedFacts>,
}

/// What the trainer reported when it last started.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StartedFacts {
    /// Trainable parameters.
    pub parameters: Option<u64>,

    /// Training tracks.
    pub train_tracks: Option<u64>,

    /// Held-out tracks.
    pub validation_tracks: Option<u64>,

    /// Training tracks with lyrics.
    pub tracks_with_lyrics: Option<u64>,
}

/// One run's summary together with its chart series.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Detail {
    /// The run at a glance.
    pub summary: Summary,

    /// The run's configuration as written by its trainer.
    pub config: serde_json::Value,

    /// The series its charts draw.
    pub history: History,
}

/// Every run in every source, live runs first, then the most recently updated.
pub fn discover(sources: &[Source]) -> Vec<Summary> {
    let mut summaries = Vec::new();
    for source in sources {
        for folder in run_folders(source) {
            if let Some(detail) = load(source, &folder) {
                summaries.push(detail.summary);
            }
        }
    }

    summaries.sort_by(|left, right| {
        let left_live = left.status == Status::Live;
        let right_live = right.status == Status::Live;

        right_live.cmp(&left_live).then(
            left.seconds_since_update
                .total_cmp(&right.seconds_since_update),
        )
    });
    summaries
}

/// The run with id `<label>/<name>`, if it exists.
pub fn find(sources: &[Source], id: &str) -> Option<Detail> {
    let (label, name) = id.split_once('/')?;
    let is_plain_name = !name.is_empty() && !name.contains(['/', '\\']) && name != "." && name != "..";
    if !is_plain_name {
        return None;
    }

    let source = sources.iter().find(|source| source.label == label)?;
    load(source, &source.folder.join(name))
}

/// The folders of a source that hold a metrics file.
fn run_folders(source: &Source) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(&source.folder) else { return Vec::new() };

    entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.join(METRICS_FILE).is_file())
        .collect()
}

/// A run's summary, configuration and history, read from its folder.
fn load(source: &Source, folder: &Path) -> Option<Detail> {
    let metrics = folder.join(METRICS_FILE);
    let modified = std::fs::metadata(&metrics).ok()?.modified().ok()?;
    let name = folder.file_name()?.to_string_lossy().into_owned();

    let events = events::read(&metrics);
    let history = History::from_events(&events);
    let config = read_config(folder);

    let summary = summarize(SummaryInputs {
        id: format!("{}/{name}", source.label),
        name,
        source: source.label.clone(),
        kind: kind_of(&events, &config),
        target_steps: config.get("steps").and_then(|steps| steps.as_u64()),
        seconds_since_update: SystemTime::now()
            .duration_since(modified)
            .map_or(0.0, |elapsed| elapsed.as_secs_f64()),
        started: latest_start(&events),
        history: &history,
    });

    Some(Detail { summary, config, history })
}

/// The run's `config.json`, or `null` when it has none.
fn read_config(folder: &Path) -> serde_json::Value {
    std::fs::read_to_string(folder.join(CONFIG_FILE))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(serde_json::Value::Null)
}

/// A vocoder run reports loss parts and configures a `vocoder`; anything else is a language model.
fn kind_of(events: &[Event], config: &serde_json::Value) -> Kind {
    let reports_parts = events.iter().any(|event| match event {
        Event::Progress(progress) => !progress.loss_parts.is_empty(),
        Event::Validation(validation) => !validation.loss_parts.is_empty(),
        _ => false,
    });

    if reports_parts || config.get("vocoder").is_some() { Kind::Vocoder } else { Kind::LanguageModel }
}

/// The most recent start event, if the trainer writes them.
fn latest_start(events: &[Event]) -> Option<StartedFacts> {
    events.iter().rev().find_map(|event| match event {
        Event::Started(Started { parameters, train_tracks, validation_tracks, tracks_with_lyrics, .. }) => Some(StartedFacts {
            parameters: *parameters,
            train_tracks: *train_tracks,
            validation_tracks: *validation_tracks,
            tracks_with_lyrics: *tracks_with_lyrics,
        }),
        _ => None,
    })
}

/// What a summary is computed from.
struct SummaryInputs<'a> {
    /// `<label>/<name>`.
    id: String,

    /// The run folder's name.
    name: String,

    /// The source's label.
    source: String,

    /// Which trainer wrote it.
    kind: Kind,

    /// Configured total steps.
    target_steps: Option<u64>,

    /// Seconds since the metrics changed.
    seconds_since_update: f64,

    /// Facts from the latest start.
    started: Option<StartedFacts>,

    /// The run's series.
    history: &'a History,
}

/// The summary of a run from its series and files.
fn summarize(inputs: SummaryInputs<'_>) -> Summary {
    let history = inputs.history;
    let step = history.training.last().map(|point| point.step);
    let reached_target = step.is_some_and(|step| inputs.target_steps.is_some_and(|target| step >= target));

    let status = if reached_target {
        Status::Finished
    } else if inputs.seconds_since_update < LIVE_SECONDS {
        Status::Live
    } else {
        Status::Stopped
    };

    let best = lowest(&history.held_out);
    let validations_since_best = best.map_or(0, |best| {
        history
            .held_out
            .iter()
            .filter(|point| point.step > best.step)
            .count()
    });

    let gap = history.gap.last().map(|point| point.value);

    let milliseconds_per_step = history
        .milliseconds_per_step
        .last()
        .map(|point| point.value);
    let eta_seconds = if status == Status::Live { eta(step, inputs.target_steps, milliseconds_per_step) } else { None };

    Summary {
        id: inputs.id,
        name: inputs.name,
        source: inputs.source,
        kind: inputs.kind,
        status,
        step,
        target_steps: inputs.target_steps,
        training_loss: step.and_then(|step| recent_training_loss(&history.training, step)),
        best_held_out: best,
        latest_held_out: history.held_out.last().copied(),
        gap,
        validations_since_best,
        overfitting: is_overfitting(&history.held_out, &history.gap, best, validations_since_best),
        milliseconds_per_step,
        eta_seconds,
        seconds_since_update: inputs.seconds_since_update,
        started: inputs.started,
    }
}

/// The point with the lowest value; the earliest one when several tie.
fn lowest(points: &[Point]) -> Option<Point> {
    let mut lowest: Option<Point> = None;
    for point in points {
        if lowest.is_none_or(|current| point.value < current.value) {
            lowest = Some(*point);
        }
    }

    lowest
}

/// Whether the run is overfitting, by either of two signs:
///
/// * over the last [`RECENT_VALIDATIONS`] validations, the held-out loss has
///   stalled while the generalization gap kept growing (the run is overfitting now);
/// * the held-out loss has not improved for [`STALLED_VALIDATIONS`]
///   validations and the gap has grown since its best (it overfitted earlier
///   and never recovered).
fn is_overfitting(held_out: &[Point], gap: &[Point], best: Option<Point>, validations_since_best: usize) -> bool {
    let stalled_recently = change_over_recent(held_out).is_some_and(|change| -change < MIN_IMPROVEMENT);
    let gap_grew_recently = change_over_recent(gap).is_some_and(|change| change > GAP_GROWTH);

    let gap_now = gap.last().map(|point| point.value);
    let gap_at_best = best.and_then(|best| {
        gap.iter()
            .find(|point| point.step == best.step)
            .map(|point| point.value)
    });
    let gap_grew_since_best = gap_now.is_some_and(|now| gap_at_best.is_some_and(|then| now - then > GAP_GROWTH));

    let overfitting_now = stalled_recently && gap_grew_recently;
    let never_recovered = validations_since_best >= STALLED_VALIDATIONS && gap_grew_since_best;
    overfitting_now || never_recovered
}

/// The change across the last [`RECENT_VALIDATIONS`] points, from the first of
/// them to the last, once there are that many.
fn change_over_recent(points: &[Point]) -> Option<f64> {
    let recent = points.get(points.len().checked_sub(RECENT_VALIDATIONS)?..)?;
    let first = recent.first()?.value;
    let last = recent.last()?.value;

    Some(last - first)
}

/// Seconds left until `target` steps, at `milliseconds` per step.
fn eta(step: Option<u64>, target: Option<u64>, milliseconds: Option<f64>) -> Option<f64> {
    let remaining = target?.saturating_sub(step?);
    Some(remaining as f64 * milliseconds? / 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(history: &History, seconds_since_update: f64) -> SummaryInputs<'_> {
        SummaryInputs {
            id: "thor/run".into(),
            name: "run".into(),
            source: "thor".into(),
            kind: Kind::LanguageModel,
            target_steps: Some(1_000),
            seconds_since_update,
            started: None,
            history,
        }
    }

    fn points(values: &[(u64, f64)]) -> Vec<Point> {
        values
            .iter()
            .map(|&(step, value)| Point { step, value })
            .collect()
    }

    #[test]
    fn a_live_run_has_an_eta_from_its_speed() {
        let history = History { training: points(&[(400, 5.0)]), milliseconds_per_step: points(&[(400, 2_000.0)]), ..History::default() };
        let summary = summarize(inputs(&history, 10.0));

        assert_eq!(summary.status, Status::Live);
        assert_eq!(summary.eta_seconds, Some(1_200.0));
    }

    #[test]
    fn a_quiet_run_short_of_its_target_has_stopped() {
        let history = History { training: points(&[(400, 5.0)]), ..History::default() };

        assert_eq!(summarize(inputs(&history, 3_600.0)).status, Status::Stopped);
    }

    #[test]
    fn a_held_out_loss_that_only_creeps_down_while_the_gap_grows_is_flagged() {
        // pq-large's last validations: a new best by thousandths, while the gap widens.
        let history = History {
            training: points(&[(1_000, 4.0)]),
            held_out: points(&[(100, 5.10), (200, 5.068), (300, 5.067), (400, 5.066), (500, 5.065), (600, 5.064)]),
            gap: points(&[(100, 0.6), (200, 0.7), (300, 0.75), (400, 0.8), (500, 0.9), (600, 0.97)]),
            ..History::default()
        };
        let summary = summarize(inputs(&history, 10.0));

        assert_eq!(summary.best_held_out, Some(Point { step: 600, value: 5.064 }));
        assert_eq!(summary.validations_since_best, 0);
        assert!(summary.overfitting);
    }

    #[test]
    fn a_run_that_overfitted_early_and_never_recovered_is_flagged() {
        // vocal-finetune: best at its first validation, flat ever since, with a wide gap.
        let history = History {
            training: points(&[(1_000, 3.2)]),
            held_out: points(&[(100, 4.23), (200, 4.28), (300, 4.30), (400, 4.30), (500, 4.30), (600, 4.30)]),
            gap: points(&[(100, 0.30), (200, 0.95), (300, 1.0), (400, 1.0), (500, 1.0), (600, 1.0)]),
            ..History::default()
        };

        assert!(summarize(inputs(&history, 10.0)).overfitting);
    }

    #[test]
    fn a_held_out_loss_still_falling_is_not_flagged() {
        let history = History {
            training: points(&[(1_000, 4.0)]),
            held_out: points(&[(100, 6.0), (200, 5.8), (300, 5.6), (400, 5.5), (500, 5.4), (600, 5.3)]),
            gap: points(&[(100, 0.1), (200, 0.2), (300, 0.3), (400, 0.4), (500, 0.5), (600, 0.6)]),
            ..History::default()
        };

        assert!(!summarize(inputs(&history, 10.0)).overfitting);
    }

    #[test]
    fn find_rejects_ids_that_leave_the_source() {
        let sources = [Source { label: "thor".into(), folder: std::env::temp_dir() }];

        assert!(find(&sources, "thor/..").is_none());
        assert!(find(&sources, "thor/a/b").is_none());
        assert!(find(&sources, "elsewhere/run").is_none());
    }
}
