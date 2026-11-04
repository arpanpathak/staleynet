//! Discovery of trained runs under `<project>/runs/`.

use serde::Serialize;
use serde_json::Value;
use std::path::Path;

/// File every checkpoint directory contains.
const CHECKPOINT_METADATA: &str = "model.json";

/// Metrics log written by the trainer.
const METRICS_LOG: &str = "metrics.jsonl";

/// A run that can be sampled from, as shown in the run picker.
#[derive(Serialize)]
pub struct RunSummary {
    /// Directory name under `runs/`.
    pub name: String,

    /// Lowest validation loss logged so far, in nats.
    pub best_validation_loss: Option<f64>,

    /// Step at which that loss was reached (the step `best/` holds).
    pub best_step: Option<u64>,

    /// Last step that appears in the metrics log.
    pub latest_step: Option<u64>,

    /// Bytes of lyrics the model reads; 0 for a model without a lyrics field.
    pub lyrics_bytes: u64,
}

/// Every run with at least one checkpoint, sorted by best validation loss.
pub fn discover(project: &Path) -> Vec<RunSummary> {
    let Ok(entries) = std::fs::read_dir(project.join("runs")) else { return Vec::new() };

    let mut runs: Vec<RunSummary> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| has_checkpoint(path))
        .filter_map(|path| path.file_name().map(|name| summarise(&path, name.to_string_lossy().into_owned())))
        .collect();

    runs.sort_by(|a, b| a.best_validation_loss.unwrap_or(f64::MAX).total_cmp(&b.best_validation_loss.unwrap_or(f64::MAX)));
    runs
}

/// True when `run` holds a `best/` or `latest/` checkpoint.
pub fn has_checkpoint(run: &Path) -> bool {
    ["best", "latest"].iter().any(|kind| run.join(kind).join(CHECKPOINT_METADATA).is_file())
}

/// Bytes of lyrics the run's model reads, from its checkpoint metadata.
fn lyrics_bytes(run: &Path) -> u64 {
    let metadata = ["best", "latest"].iter().find_map(|kind| std::fs::read_to_string(run.join(kind).join(CHECKPOINT_METADATA)).ok());
    let Some(model) = metadata.and_then(|text| serde_json::from_str::<Value>(&text).ok()) else { return 0 };
    model["model"]["lyrics_bytes"].as_u64().unwrap_or(0)
}

/// Reads the run's metrics log for its best and latest steps.
fn summarise(run: &Path, name: String) -> RunSummary {
    let mut summary = RunSummary { name, best_validation_loss: None, best_step: None, latest_step: None, lyrics_bytes: lyrics_bytes(run) };
    let Ok(log) = std::fs::read_to_string(run.join(METRICS_LOG)) else { return summary };

    for event in log.lines().filter_map(|line| serde_json::from_str::<Value>(line).ok()) {
        let step = event["step"].as_u64();
        summary.latest_step = summary.latest_step.max(step);

        let is_improved_validation = event["event"] == "validation" && event["improved"] == true;
        if is_improved_validation {
            summary.best_validation_loss = event["loss"].as_f64();
            summary.best_step = step;
        }
    }

    summary
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn temp_project(tag: &str) -> std::path::PathBuf {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);

        std::env::temp_dir().join(format!("staleynet-studio-runs-{tag}-{}-{id}", std::process::id()))
    }

    fn write_run(project: &Path, name: &str, kind: &str, lyrics: u64, metrics: &str) {
        let run = project.join("runs").join(name);
        let checkpoint = run.join(kind);
        std::fs::create_dir_all(&checkpoint).unwrap();
        std::fs::write(checkpoint.join("model.json"), format!(r#"{{"model":{{"lyrics_bytes":{lyrics}}}}}"#)).unwrap();
        std::fs::write(run.join("metrics.jsonl"), metrics).unwrap();
    }

    #[test]
    fn discovers_and_sorts_runs_by_best_validation() {
        let project = temp_project("discover");
        write_run(&project, "a", "best", 256, "{\"event\":\"validation\",\"loss\":2.0,\"step\":100,\"improved\":true}\n{\"event\":\"progress\",\"step\":150}\n");
        write_run(&project, "b", "latest", 0, "{\"event\":\"validation\",\"loss\":3.0,\"step\":50,\"improved\":true}\n");
        std::fs::create_dir_all(project.join("runs").join("empty")).unwrap();

        let runs = discover(&project);

        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].name, "a");
        assert_eq!(runs[0].best_validation_loss, Some(2.0));
        assert_eq!(runs[0].best_step, Some(100));
        assert_eq!(runs[0].latest_step, Some(150));
        assert_eq!(runs[0].lyrics_bytes, 256);
        assert_eq!(runs[1].name, "b");
        assert_eq!(runs[1].lyrics_bytes, 0);

        std::fs::remove_dir_all(project).unwrap();
    }

    #[test]
    fn only_directories_with_a_checkpoint_are_kept() {
        let project = temp_project("checkpoint");
        write_run(&project, "run", "best", 0, "");

        assert!(has_checkpoint(&project.join("runs").join("run")));
        assert!(!has_checkpoint(&project.join("runs").join("missing")));

        std::fs::remove_dir_all(project).unwrap();
    }

    #[test]
    fn a_missing_runs_directory_is_empty() {
        let project = temp_project("missing");

        assert!(discover(&project).is_empty());
    }
}
