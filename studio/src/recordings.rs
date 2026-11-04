//! Takes recorded in the browser: your own singing, saved as a training set.
//!
//! Every take is four files under `<project>/data/recordings/`:
//!
//! ```text
//! <id>.wav          mono 16-bit PCM, as the browser captured it
//! <id>.txt          the description, one caption per line
//! <id>.lyrics.txt   the words sung, if any
//! <id>.json         a `Recording` record for the studio
//! ```
//!
//! The `.wav` and `.txt` pair is the layout `staleynet prepare --folder` reads
//! (see `crates/staleynet-data/src/catalog.rs`), so the directory is a dataset
//! as it stands.

use base64::Engine;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Where takes live, relative to the project root. `data/` is git-ignored, so
/// recordings of your voice never end up in the repository.
pub const RECORDINGS_DIRECTORY: &str = "data/recordings";

/// Size of the canonical RIFF/WAVE header for 16-bit PCM, in bytes.
/// Format reference: <https://ccrma.stanford.edu/courses/422-winter-2014/projects/WaveFormat/>
const WAV_HEADER_BYTES: usize = 44;

/// Sample rates accepted from the browser. Every audio interface in use
/// records at one of these; anything else suggests a corrupted upload.
const ACCEPTED_SAMPLE_RATES: [u32; 4] = [16_000, 22_050, 44_100, 48_000];

/// Bytes per sample of 16-bit PCM.
const BYTES_PER_SAMPLE: usize = 2;

/// Shortest take worth keeping, in seconds.
const MIN_SECONDS: f32 = 1.0;

/// Longest take accepted, in seconds. Longer songs can be recorded in parts.
pub const MAX_SECONDS: f32 = 300.0;

/// Longest description accepted, in characters.
const MAX_CAPTION_CHARACTERS: usize = 400;

/// Longest lyrics accepted, in characters (a long song runs to about 3,000).
const MAX_LYRICS_CHARACTERS: usize = 8_000;

/// What plays under the voice. Recorded so the two kinds of take can be
/// selected separately for training.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Accompaniment {
    /// Voice alone (a cappella).
    VoiceOnly,

    /// Voice with guitar played at the same time.
    VoiceAndGuitar,

    /// Guitar or other instruments, no voice.
    InstrumentOnly,
}

/// One saved take, as stored in `<id>.json` and listed to the browser.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Recording {
    /// File stem shared by the take's files.
    pub id: String,

    /// What the take sounds like, in the style of the training captions.
    pub caption: String,

    /// The words sung; empty for humming or instrumental takes.
    pub lyrics: String,

    /// What plays under the voice.
    pub accompaniment: Accompaniment,

    /// Length of the audio.
    pub seconds: f32,

    /// Sample rate the browser recorded at.
    pub sample_rate: u32,

    /// Seconds since the Unix epoch when the take was saved.
    pub created: u64,
}

/// A take as the browser uploads it.
#[derive(Deserialize)]
pub struct Upload {
    /// Description of the take.
    pub caption: String,

    /// Words sung.
    #[serde(default)]
    pub lyrics: String,

    /// What plays under the voice.
    pub accompaniment: Accompaniment,

    /// The WAV file, base64-encoded (RFC 4648 §4).
    pub wav_base64: String,
}

/// Facts read from a WAV header.
struct WavFacts {
    /// Samples per second.
    sample_rate: u32,

    /// Duration of the PCM data.
    seconds: f32,
}

/// Reads and checks the header the browser wrote: mono, 16-bit PCM, a known
/// sample rate. Returns `None` for anything else.
fn inspect_wav(bytes: &[u8]) -> Option<WavFacts> {
    let header = bytes.get(..WAV_HEADER_BYTES)?;
    let u16_at = |offset: usize| u16::from_le_bytes([header[offset], header[offset + 1]]);
    let u32_at = |offset: usize| u32::from_le_bytes([header[offset], header[offset + 1], header[offset + 2], header[offset + 3]]);

    let is_riff_wave = &header[0..4] == b"RIFF" && &header[8..12] == b"WAVE" && &header[36..40] == b"data";
    let is_mono_pcm16 = u16_at(20) == 1 && u16_at(22) == 1 && u16_at(34) == 16;
    let sample_rate = u32_at(24);

    if !is_riff_wave || !is_mono_pcm16 || !ACCEPTED_SAMPLE_RATES.contains(&sample_rate) {
        return None;
    }

    let samples = (bytes.len() - WAV_HEADER_BYTES) / BYTES_PER_SAMPLE;
    Some(WavFacts { sample_rate, seconds: samples as f32 / sample_rate as f32 })
}

/// The recordings directory of `project`.
pub fn directory(project: &Path) -> PathBuf {
    project.join(RECORDINGS_DIRECTORY)
}

/// True if `id` is a stem this module could have produced, so it is safe to
/// join onto the recordings directory.
fn is_valid_id(id: &str) -> bool {
    !id.is_empty() && id.chars().all(|character| character.is_ascii_alphanumeric() || character == '-')
}

/// Checks and stores one take; returns its record.
pub fn save(project: &Path, upload: Upload) -> Result<Recording, String> {
    let caption = upload.caption.trim();
    let lyrics = upload.lyrics.trim();

    if caption.is_empty() {
        return Err("describe the take (for example \"male voice singing softly with acoustic guitar\")".into());
    }

    if caption.chars().count() > MAX_CAPTION_CHARACTERS {
        return Err(format!("the description is longer than {MAX_CAPTION_CHARACTERS} characters"));
    }

    if lyrics.chars().count() > MAX_LYRICS_CHARACTERS {
        return Err(format!("the lyrics are longer than {MAX_LYRICS_CHARACTERS} characters"));
    }

    let wav = base64::engine::general_purpose::STANDARD
        .decode(upload.wav_base64.as_bytes())
        .map_err(|error| format!("audio is not valid base64: {error}"))?;
    let Some(facts) = inspect_wav(&wav) else {
        return Err("audio must be a mono 16-bit PCM WAV at 16, 22.05, 44.1 or 48 kHz".into());
    };
    if !(MIN_SECONDS..=MAX_SECONDS).contains(&facts.seconds) {
        return Err(format!("a take must last between {MIN_SECONDS} and {MAX_SECONDS} seconds"));
    }

    let created = SystemTime::now().duration_since(UNIX_EPOCH).map(|elapsed| elapsed.as_secs()).unwrap_or_default();
    let id = format!("{created}-{}", crate::generation::slug(caption));
    let recording = Recording {
        id,
        caption: caption.to_string(),
        lyrics: lyrics.to_string(),
        accompaniment: upload.accompaniment,
        seconds: facts.seconds,
        sample_rate: facts.sample_rate,
        created,
    };

    let root = directory(project);
    let write = |extension: &str, contents: &[u8]| std::fs::write(root.join(format!("{}.{extension}", recording.id)), contents);

    std::fs::create_dir_all(&root).map_err(|error| error.to_string())?;
    let record = serde_json::to_vec_pretty(&recording).map_err(|error| error.to_string())?;

    write("wav", &wav)
        .and_then(|()| write("txt", format!("{caption}\n").as_bytes()))
        .and_then(|()| write("lyrics.txt", lyrics.as_bytes()))
        .and_then(|()| write("json", &record))
        .map_err(|error| format!("could not save the take: {error}"))?;

    Ok(recording)
}

/// Every saved take, newest first.
pub fn list(project: &Path) -> Vec<Recording> {
    let Ok(entries) = std::fs::read_dir(directory(project)) else { return Vec::new() };

    let mut recordings: Vec<Recording> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|extension| extension == "json"))
        .filter_map(|path| std::fs::read_to_string(path).ok())
        .filter_map(|text| serde_json::from_str(&text).ok())
        .collect();
    recordings.sort_by(|a, b| b.created.cmp(&a.created));
    recordings
}

/// Deletes every file of take `id`. Returns false if there was no such take.
pub fn delete(project: &Path, id: &str) -> bool {
    if !is_valid_id(id) {
        return false;
    }

    let root = directory(project);
    let record = root.join(format!("{id}.json"));
    if !record.is_file() {
        return false;
    }

    for extension in ["wav", "txt", "lyrics.txt", "json"] {
        let _ = std::fs::remove_file(root.join(format!("{id}.{extension}")));
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal mono 16-bit WAV with `samples` zero samples.
    fn silent_wav(sample_rate: u32, samples: usize) -> Vec<u8> {
        let data_bytes = (samples * BYTES_PER_SAMPLE) as u32;
        let mut bytes = Vec::new();

        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_bytes).to_le_bytes());

        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());

        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_bytes.to_le_bytes());
        bytes.resize(bytes.len() + samples * BYTES_PER_SAMPLE, 0);

        bytes
    }

    #[test]
    fn header_inspection_accepts_mono_pcm16_only() {
        let facts = inspect_wav(&silent_wav(48_000, 96_000)).expect("valid header");
        assert_eq!(facts.sample_rate, 48_000);
        assert!((facts.seconds - 2.0).abs() < 1e-6);

        let mut stereo = silent_wav(48_000, 100);
        stereo[22] = 2;

        assert!(inspect_wav(&stereo).is_none());
        assert!(inspect_wav(&silent_wav(12_345, 100)).is_none());
        assert!(inspect_wav(b"RIFF").is_none());
    }

    #[test]
    fn ids_cannot_escape_the_directory() {
        assert!(is_valid_id("1793755194-voice-and-guitar"));
        assert!(!is_valid_id("../runs"));
        assert!(!is_valid_id("a/b"));
        assert!(!is_valid_id(""));
    }
}
