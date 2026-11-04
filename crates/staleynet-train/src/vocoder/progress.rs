//! Running averages between progress lines, and the best validation loss a
//! run has recorded.

use staleynet_model::vocoder::LossBreakdown;
use std::path::Path;

/// Means over the steps since the last progress line.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowMeans {
    /// Mean total loss.
    pub loss: f64,

    /// Mean of each part of the loss.
    pub loss_parts: Vec<f64>,

    /// Mean gradient norm before clipping.
    pub gradient_norm: f64,

    /// Mean step duration, in seconds.
    pub seconds_per_step: f64,
}

/// Running sums over the steps between two progress lines.
#[derive(Default)]
pub struct Window {
    /// Steps added.
    steps: usize,

    /// Sum of total losses.
    loss: f64,

    /// Sum of each part of the loss.
    loss_parts: Vec<f64>,

    /// Sum of gradient norms.
    gradient_norm: f64,

    /// Sum of step durations, in seconds.
    seconds: f64,
}

impl Window {
    /// Adds one step.
    pub fn add(&mut self, loss: &LossBreakdown, gradient_norm: f32, seconds: f64) {
        self.loss_parts.resize(loss.parts.len(), 0.0);
        for (sum, part) in self.loss_parts.iter_mut().zip(&loss.parts) {
            *sum += part;
        }

        self.steps += 1;
        self.loss += loss.total;
        self.gradient_norm += gradient_norm as f64;
        self.seconds += seconds;
    }

    /// The means since the last call, after which the window starts empty.
    pub fn take(&mut self) -> WindowMeans {
        let count = self.steps.max(1) as f64;
        let means = WindowMeans {
            loss: self.loss / count,
            loss_parts: self.loss_parts.iter().map(|sum| sum / count).collect(),
            gradient_norm: self.gradient_norm / count,
            seconds_per_step: self.seconds / count,
        };

        *self = Self::default();
        means
    }
}

/// The lowest validation loss recorded in `run/metrics.jsonl`, so that a
/// resumed run does not replace `best` with something worse.
pub fn best_validation_so_far(run: &Path) -> f64 {
    let Ok(text) = std::fs::read_to_string(run.join("metrics.jsonl")) else { return f64::INFINITY };

    text.lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|value| value["event"] == "validation")
        .filter_map(|value| value["loss"].as_f64())
        .fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_averages_its_steps_and_then_resets() {
        let mut window = Window::default();
        window.add(&LossBreakdown { total: 3.0, parts: vec![1.0, 2.0] }, 0.5, 0.2);
        window.add(&LossBreakdown { total: 5.0, parts: vec![3.0, 2.0] }, 1.5, 0.4);

        let means = window.take();
        assert_eq!(means.loss, 4.0);
        assert_eq!(means.loss_parts, vec![2.0, 2.0]);
        assert_eq!(means.gradient_norm, 1.0);
        assert!((means.seconds_per_step - 0.3).abs() < 1e-12);

        assert_eq!(window.take().loss, 0.0);
    }
}
