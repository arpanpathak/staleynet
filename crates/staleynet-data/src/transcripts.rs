//! Machine-transcribed lyrics, merged into a cache's manifest.
//!
//! The Whisper pass (`deploy/lyrics/transcribe.py`) writes one JSON line per
//! song:
//!
//! ```text
//! {"id": ..., "mean_logprob": -0.38, "language": "en",
//!  "lines": [{"text": ..., "start": 90.0, "end": 94.74}, ...]}
//! ```
//!
//! Merging gives each track without lyrics its transcribed lines, timed in
//! seconds from the start of the recording. The manifest as it was before the
//! first merge is kept as `manifest.before-lyrics.json` and is always the
//! starting point, so merging again with other filters gives a clean result.
//! The codes are not touched, so nothing needs to be tokenized again.

use crate::catalog::{LineTiming, LyricLine};
use crate::store::{CacheLayout, DatasetManifest};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// The copy of the manifest taken before the first merge.
const ORIGINAL_MANIFEST: &str = "manifest.before-lyrics.json";

/// One transcribed line.
#[derive(Clone, Debug, Deserialize)]
struct TranscribedLine {
    /// The words.
    text: String,

    /// Start, in seconds from the start of the recording.
    start: f32,

    /// End, in seconds from the start of the recording.
    end: f32,
}

/// One song's transcription.
#[derive(Clone, Debug, Deserialize)]
pub struct Transcript {
    /// Track id in the manifest.
    pub id: String,

    /// Whisper's mean token log-probability over the windows that kept lines.
    #[serde(default)]
    pub mean_logprob: Option<f32>,

    /// The language most windows were detected as, when any kept lines.
    #[serde(default)]
    pub language: Option<String>,

    /// The kept lines, in order.
    lines: Vec<TranscribedLine>,
}

impl Transcript {
    /// The lines in the manifest's lyrics format.
    fn lyric_lines(&self) -> Vec<LyricLine> {
        self.lines
            .iter()
            .map(|line| LyricLine { text: line.text.clone(), timing: Some(LineTiming { start: line.start, end: line.end }) })
            .collect()
    }
}

/// Which transcripts are trusted enough to train on.
#[derive(Clone, Debug, Default)]
pub struct TranscriptFilter {
    /// Lowest mean log-probability kept; `None` keeps every confidence.
    pub min_logprob: Option<f32>,

    /// Languages kept, as Whisper codes such as `en`; empty keeps every language.
    pub languages: Vec<String>,
}

impl TranscriptFilter {
    /// Whether `transcript` has lines and passes the filter.
    pub fn accepts(&self, transcript: &Transcript) -> bool {
        if transcript.lines.is_empty() {
            return false;
        }

        let confident = match (self.min_logprob, transcript.mean_logprob) {
            (None, _) => true,
            (Some(floor), Some(logprob)) => logprob >= floor,
            (Some(_), None) => false,
        };

        let language_kept = match &transcript.language {
            _ if self.languages.is_empty() => true,
            Some(language) => self.languages.contains(language),
            None => false,
        };

        confident && language_kept
    }
}

/// Reads every transcript of a JSON-lines file.
pub fn read_transcripts(path: &Path) -> Result<Vec<Transcript>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;

    text.lines()
        .filter(|row| !row.trim().is_empty())
        .enumerate()
        .map(|(index, row)| serde_json::from_str(row).with_context(|| format!("{} row {}", path.display(), index + 1)))
        .collect()
}

/// What a merge did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MergeReport {
    /// Transcripts read.
    pub transcripts: usize,

    /// Transcripts that passed the filter.
    pub accepted: usize,

    /// Tracks that gained lyrics.
    pub added: usize,

    /// Tracks that now have lyrics, including ones that had them before.
    pub with_lyrics: usize,

    /// Tracks in the manifest.
    pub tracks: usize,
}

/// The manifest as it was before any lyrics were merged, copied aside on the first merge.
fn original_manifest(layout: &CacheLayout, root: &Path) -> Result<DatasetManifest> {
    let original = root.join(ORIGINAL_MANIFEST);
    if !original.exists() {
        fs::copy(layout.manifest(), &original).with_context(|| format!("copy manifest to {}", original.display()))?;
    }

    let text = fs::read_to_string(&original)?;
    Ok(serde_json::from_str(&text)?)
}

/// Gives each track without lyrics the lines of its accepted transcript; returns how many gained some.
fn add_lyrics(manifest: &mut DatasetManifest, lyrics: &HashMap<&str, Vec<LyricLine>>) -> usize {
    let mut added = 0;

    for entry in &mut manifest.tracks {
        let track = &mut entry.track;
        let Some(lines) = lyrics.get(track.id.as_str()) else { continue };
        if !track.lyrics.is_empty() {
            continue;
        }

        track.lyrics = lines.clone();
        added += 1;
    }

    added
}

/// Merges the accepted transcripts into the manifest of the cache at `root`.
pub fn merge_lyrics(root: &Path, transcripts: &[Transcript], filter: &TranscriptFilter) -> Result<MergeReport> {
    let layout = CacheLayout::new(root);
    let mut manifest = original_manifest(&layout, root)?;

    let lyrics: HashMap<&str, Vec<LyricLine>> = transcripts
        .iter()
        .filter(|transcript| filter.accepts(transcript))
        .map(|transcript| (transcript.id.as_str(), transcript.lyric_lines()))
        .collect();

    let added = add_lyrics(&mut manifest, &lyrics);
    manifest.save(&layout)?;

    Ok(MergeReport {
        transcripts: transcripts.len(),
        accepted: lyrics.len(),
        added,
        with_lyrics: manifest
            .tracks
            .iter()
            .filter(|entry| !entry.track.lyrics.is_empty())
            .count(),
        tracks: manifest.tracks.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript(row: &str) -> Transcript {
        serde_json::from_str(row).unwrap()
    }

    #[test]
    fn rejects_transcripts_without_lines() {
        let empty = transcript(r#"{"id": "a", "mean_logprob": -0.2, "lines": [], "language": null}"#);

        assert!(!TranscriptFilter::default().accepts(&empty));
    }

    #[test]
    fn filters_by_confidence_and_language() {
        let french =
            transcript(r#"{"id": "a", "mean_logprob": -0.38, "language": "fr", "lines": [{"text": "x", "start": 90.0, "end": 94.7}]}"#);
        let english_only = TranscriptFilter { min_logprob: None, languages: vec!["en".to_string()] };
        let confident = TranscriptFilter { min_logprob: Some(-0.4), languages: Vec::new() };
        let very_confident = TranscriptFilter { min_logprob: Some(-0.3), languages: Vec::new() };

        assert!(!english_only.accepts(&french));
        assert!(confident.accepts(&french));
        assert!(!very_confident.accepts(&french));
    }

    #[test]
    fn keeps_whole_song_timing() {
        let song =
            transcript(r#"{"id": "a", "mean_logprob": -0.3, "language": "en", "lines": [{"text": "x", "start": 210.0, "end": 219.88}]}"#);
        let timing = song.lyric_lines()[0].timing.unwrap();

        assert_eq!((timing.start, timing.end), (210.0, 219.88));
    }
}
