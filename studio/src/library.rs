//! The audio clips the studio can play: everything under `<project>/samples/`.

use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Sub-directory of `samples/` the studio writes its own generations into.
pub const STUDIO_SUBDIRECTORY: &str = "studio";

/// One playable clip.
#[derive(Serialize)]
pub struct Clip {
    /// Path relative to `samples/`, also the key used in `/audio/<path>`.
    pub path: String,

    /// Directory the clip lives in (`studio`, `tokenizer`, or `` for the root).
    pub group: String,

    /// The JSON report `staleynet sing` wrote beside the WAV, if any.
    pub report: Option<Value>,

    /// Seconds since the Unix epoch at which the file was last written.
    pub modified: u64,
}

/// Every WAV under `samples/`, newest first.
pub fn list(samples: &Path) -> Vec<Clip> {
    let mut found = Vec::new();
    collect_wavs(samples, &mut found);

    let mut clips: Vec<Clip> = found.into_iter().filter_map(|path| describe(samples, &path)).collect();
    clips.sort_by(|a, b| b.modified.cmp(&a.modified));
    clips
}

/// Depth-first collection of `.wav` files.
fn collect_wavs(directory: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(directory) else { return };

    for path in entries.filter_map(|entry| entry.ok().map(|entry| entry.path())) {
        if path.is_dir() {
            collect_wavs(&path, found);
        } else if path.extension().is_some_and(|extension| extension == "wav") {
            found.push(path);
        }
    }
}

/// A [`Clip`] for `path`, or `None` if it is not readable.
fn describe(samples: &Path, path: &Path) -> Option<Clip> {
    let relative = path.strip_prefix(samples).ok()?;
    let modified = std::fs::metadata(path).ok()?.modified().ok()?.duration_since(SystemTime::UNIX_EPOCH).ok()?.as_secs();
    let report = std::fs::read_to_string(path.with_extension("json")).ok().and_then(|text| serde_json::from_str(&text).ok());
    let group = relative.parent().map(|parent| parent.to_string_lossy().into_owned()).unwrap_or_default();

    Some(Clip { path: relative.to_string_lossy().into_owned(), group, report, modified })
}

/// Resolves a client-supplied relative path to a WAV inside `directory`,
/// refusing anything that escapes it (e.g. `../`).
pub fn resolve(directory: &Path, relative: &str) -> Option<PathBuf> {
    let root = directory.canonicalize().ok()?;
    let candidate = root.join(relative).canonicalize().ok()?;

    let inside = candidate.starts_with(&root);
    let is_wav = candidate.extension().is_some_and(|extension| extension == "wav");
    (inside && is_wav && candidate.is_file()).then_some(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn temp_samples(tag: &str) -> PathBuf {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);

        std::env::temp_dir().join(format!("staleynet-studio-library-{tag}-{}-{id}", std::process::id()))
    }

    #[test]
    fn lists_wavs_with_reports_and_groups() {
        let samples = temp_samples("list");
        std::fs::create_dir_all(samples.join("studio")).unwrap();

        std::fs::write(samples.join("a.wav"), b"wav").unwrap();
        std::fs::write(samples.join("a.json"), r#"{"prompt":"calm"}"#).unwrap();
        std::fs::write(samples.join("studio").join("b.wav"), b"wav").unwrap();

        let clips = list(&samples);

        assert_eq!(clips.len(), 2);

        let root = clips.iter().find(|clip| clip.path == "a.wav").unwrap();
        assert_eq!(root.group, "");
        assert_eq!(root.report.as_ref().unwrap()["prompt"], "calm");

        let studio = clips.iter().find(|clip| clip.path == "studio/b.wav").unwrap();
        assert_eq!(studio.group, "studio");
        assert!(studio.report.is_none());

        std::fs::remove_dir_all(samples).unwrap();
    }

    #[test]
    fn resolve_accepts_files_inside_the_directory_only() {
        let samples = temp_samples("resolve");
        std::fs::create_dir_all(&samples).unwrap();

        std::fs::write(samples.join("a.wav"), b"wav").unwrap();
        std::fs::write(samples.join("a.txt"), b"text").unwrap();

        assert!(resolve(&samples, "a.wav").is_some());
        assert!(resolve(&samples, "../a.wav").is_none());
        assert!(resolve(&samples, "a.txt").is_none());
        assert!(resolve(&samples, "missing.wav").is_none());

        std::fs::remove_dir_all(samples).unwrap();
    }
}
