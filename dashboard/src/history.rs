//! A run's events turned into the series its charts draw.
//!
//! The one derived quantity is the *generalization gap*: at each validation,
//! the held-out loss minus the mean training loss over the preceding
//! [`GAP_WINDOW_STEPS`] steps. A gap that keeps growing while the held-out
//! loss stops falling means the model is learning its training songs rather
//! than music in general, that is, overfitting.

use crate::events::{Event, Progress, Validation};
use serde::Serialize;

/// Steps of training loss averaged for the generalization gap; the same
/// window the paper's overfitting table uses.
pub const GAP_WINDOW_STEPS: u64 = 1_000;

/// Band groups per residual level in the band-group tokenizer. A language
/// model with more codebooks than this numbers them level by level, so
/// codebook `c` belongs to level `c / BAND_GROUPS`; with this many or fewer,
/// every codebook is its own level.
pub const BAND_GROUPS: usize = 4;

/// Names of the vocoder's spectral resolutions, in the order its metrics list them.
pub const VOCODER_PARTS: [&str; 3] = ["log-mel, 1024 / 256", "log-magnitude, 512 / 128", "log-magnitude, 2048 / 256"];

/// One value at one training step.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Point {
    /// The training step.
    pub step: u64,

    /// The value at that step.
    pub value: f64,
}

/// A named series, for charts that draw several lines.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Series {
    /// Legend label.
    pub name: String,

    /// The points, by increasing step.
    pub points: Vec<Point>,
}

/// Everything the charts of one run draw.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct History {
    /// Training loss at each progress line.
    pub training: Vec<Point>,

    /// Held-out loss at each validation.
    pub held_out: Vec<Point>,

    /// Held-out loss minus the recent training loss, at each validation.
    pub gap: Vec<Point>,

    /// Held-out loss by residual level (language model) or by resolution (vocoder).
    pub held_out_parts: Vec<Series>,

    /// Learning rate at each progress line.
    pub learning_rate: Vec<Point>,

    /// Gradient norm before clipping at each progress line.
    pub gradient_norm: Vec<Point>,

    /// Milliseconds per step at each progress line.
    pub milliseconds_per_step: Vec<Point>,

    /// Steps the trainer resumed from, after its first start.
    pub resumes: Vec<u64>,

    /// Steps at which listening samples were rendered.
    pub samples: Vec<u64>,
}

impl History {
    /// The series of a run, from its events in the order they were written.
    pub fn from_events(events: &[Event]) -> Self {
        let mut history = History::default();
        let mut validations = Vec::new();
        let mut starts = 0;

        for event in events {
            match event {
                Event::Started(started) => {
                    starts += 1;
                    if starts > 1 {
                        history.resumes.push(started.first_step.unwrap_or(0));
                    }
                }
                Event::Progress(progress) => history.add_progress(progress),
                Event::Validation(validation) => validations.push(validation.clone()),
                Event::Samples(samples) => history.samples.push(samples.step),
                Event::Other => {}
            }
        }

        // A resumed run repeats the steps after its last checkpoint; the later
        // line for a step is the one the run continued from.
        history.training = latest_per_step(&history.training);
        history.held_out = latest_per_step(&validations.iter().map(held_out_point).collect::<Vec<_>>());
        history.gap = history
            .held_out
            .iter()
            .filter_map(|point| gap_at(&history.training, *point))
            .collect();
        history.held_out_parts = parts(&validations);

        history
    }

    /// Adds the series of one progress line.
    fn add_progress(&mut self, progress: &Progress) {
        let step = progress.step;
        self.training.push(Point { step, value: progress.loss });

        if let Some(value) = progress.learning_rate {
            self.learning_rate.push(Point { step, value });
        }

        if let Some(value) = progress.gradient_norm {
            self.gradient_norm.push(Point { step, value });
        }

        if let Some(value) = progress.milliseconds_per_step {
            self.milliseconds_per_step.push(Point { step, value });
        }
    }
}

/// The held-out loss of one validation as a point.
fn held_out_point(validation: &Validation) -> Point {
    Point { step: validation.step, value: validation.loss }
}

/// `points` sorted by step, keeping only the last point written for each step.
fn latest_per_step(points: &[Point]) -> Vec<Point> {
    let mut kept: Vec<Point> = Vec::with_capacity(points.len());

    for point in points {
        kept.retain(|earlier| earlier.step != point.step);
        kept.push(*point);
    }

    kept.sort_by_key(|point| point.step);
    kept
}

/// The mean training loss over the window before `step`, if any was logged there.
pub fn recent_training_loss(training: &[Point], step: u64) -> Option<f64> {
    let window_start = step.saturating_sub(GAP_WINDOW_STEPS);
    let window: Vec<f64> = training
        .iter()
        .filter(|point| point.step > window_start && point.step <= step)
        .map(|point| point.value)
        .collect();

    if window.is_empty() {
        return None;
    }

    Some(window.iter().sum::<f64>() / window.len() as f64)
}

/// The generalization gap at a held-out point.
fn gap_at(training: &[Point], held_out: Point) -> Option<Point> {
    let training_loss = recent_training_loss(training, held_out.step)?;

    Some(Point { step: held_out.step, value: held_out.value - training_loss })
}

/// The held-out loss of every residual level (language model) or every
/// spectral resolution (vocoder), over all validations.
fn parts(validations: &[Validation]) -> Vec<Series> {
    let Some(last) = validations.last() else { return Vec::new() };

    if !last.loss_parts.is_empty() {
        return (0..last.loss_parts.len())
            .map(|part| Series {
                name: VOCODER_PARTS
                    .get(part)
                    .map_or(format!("part {}", part + 1), |name| name.to_string()),
                points: validations
                    .iter()
                    .filter_map(|validation| {
                        validation
                            .loss_parts
                            .get(part)
                            .map(|&value| Point { step: validation.step, value })
                    })
                    .collect(),
            })
            .collect();
    }

    let codebooks = last.loss_per_codebook.len();
    let groups = if codebooks > BAND_GROUPS && codebooks.is_multiple_of(BAND_GROUPS) { BAND_GROUPS } else { 1 };
    let levels = codebooks / groups;

    (0..levels)
        .map(|level| Series {
            name: level_name(level, levels),
            points: validations
                .iter()
                .filter(|validation| validation.loss_per_codebook.len() == codebooks)
                .map(|validation| Point { step: validation.step, value: level_mean(&validation.loss_per_codebook, level, groups) })
                .collect(),
        })
        .collect()
}

/// The mean loss of one level's codebooks.
fn level_mean(losses: &[f64], level: usize, groups: usize) -> f64 {
    let codebooks = &losses[level * groups..(level + 1) * groups];
    codebooks.iter().sum::<f64>() / groups as f64
}

/// "Level 1 (coarsest)" … "Level 4 (finest)".
fn level_name(level: usize, levels: usize) -> String {
    let detail = match level {
        0 => " (coarsest)",
        last if last + 1 == levels => " (finest)",
        _ => "",
    };

    format!("Level {}{detail}", level + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::Started;

    fn progress(step: u64, loss: f64) -> Event {
        Event::Progress(Progress {
            step,
            loss,
            loss_per_codebook: Vec::new(),
            loss_parts: Vec::new(),
            gradient_norm: Some(0.5),
            learning_rate: Some(1e-4),
            milliseconds_per_step: Some(2000.0),
        })
    }

    fn validation(step: u64, loss: f64, codebooks: Vec<f64>) -> Event {
        Event::Validation(Validation { step, loss, loss_per_codebook: codebooks, loss_parts: Vec::new() })
    }

    #[test]
    fn the_gap_is_held_out_minus_the_recent_training_mean() {
        let events = [progress(500, 9.0), progress(1_500, 4.0), progress(2_000, 3.0), validation(2_000, 5.0, Vec::new())];
        let history = History::from_events(&events);

        // Only the progress lines within the 1 000 steps before step 2 000 count.
        assert_eq!(history.gap, vec![Point { step: 2_000, value: 1.5 }]);
    }

    #[test]
    fn a_resume_replaces_the_repeated_steps_and_is_marked() {
        let events = [
            Event::Started(Started { first_step: Some(0), ..Started::default() }),
            progress(100, 6.0),
            progress(200, 5.0),
            Event::Started(Started { first_step: Some(100), ..Started::default() }),
            progress(200, 4.8),
        ];
        let history = History::from_events(&events);

        assert_eq!(history.resumes, vec![100]);
        assert_eq!(history.training, vec![Point { step: 100, value: 6.0 }, Point { step: 200, value: 4.8 }]);
    }

    #[test]
    fn sixteen_codebooks_group_into_four_levels() {
        let codebooks: Vec<f64> = (0..16).map(|codebook| (codebook / 4) as f64).collect();
        let history = History::from_events(&[validation(500, 1.0, codebooks)]);

        assert_eq!(history.held_out_parts.len(), 4);
        assert_eq!(history.held_out_parts[0].name, "Level 1 (coarsest)");
        assert_eq!(history.held_out_parts[3].name, "Level 4 (finest)");
        assert_eq!(history.held_out_parts[2].points, vec![Point { step: 500, value: 2.0 }]);
    }
}
