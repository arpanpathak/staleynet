//! What audio exists and how it is described: the catalogue a dataset is built from.
//!
//! Three sources are supported:
//! * the **Song Describer Dataset** (Manco et al., 2023,
//!   <https://arxiv.org/abs/2311.10057>): CC-licensed Jamendo tracks with
//!   human-written captions;
//! * **JamendoLyrics MultiLang** (Durand, Stoller & Ewert, 2023,
//!   <https://arxiv.org/abs/2306.07744>): 78 CC-licensed Jamendo songs in
//!   four languages whose lyrics are time-aligned line by line;
//! * **a folder of your own recordings**, each with an optional `.txt` file of
//!   the same name holding one description per line and an optional
//!   `.lyrics.txt` file holding the words sung.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use staleynet_audio::decode::is_audio_file;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Caption file of the Song Describer release.
const SONG_DESCRIBER_CAPTIONS: &str = "song_describer.csv";

/// Directory the Song Describer audio archive is extracted into.
const SONG_DESCRIBER_AUDIO: &str = "audio";

/// Metadata file of the JamendoLyrics release.
const JAMENDO_LYRICS_METADATA: &str = "JamendoLyrics.csv";

/// JamendoLyrics directory of the audio files.
const JAMENDO_LYRICS_AUDIO: &str = "mp3";

/// JamendoLyrics directory of the line-level annotations, one CSV per song.
const JAMENDO_LYRICS_LINES: &str = "annotations/lines";

/// Suffix of the lyrics sidecar in a folder of recordings.
const LYRICS_SIDECAR_SUFFIX: &str = "lyrics.txt";

/// Percentage of split groups held out for validation.
pub const VALIDATION_PERCENT: u64 = 10;

/// Where a caption came from. Recorded so that automatically filled labels
/// are never mistaken for human annotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaptionSource {
    /// Written by a person for this recording (Song Describer annotators).
    HumanAnnotated,

    /// Supplied by the dataset owner in a sidecar `.txt` file.
    Sidecar,

    /// Derived from the file name because nothing else was available.
    FileName,

    /// Assembled from catalogue metadata such as genre and language.
    Metadata,
}

/// When a lyric line is sung, in seconds from the start of the recording.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LineTiming {
    /// Start of the line.
    pub start: f32,

    /// End of the line.
    pub end: f32,
}

/// One sung line.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LyricLine {
    /// The words.
    pub text: String,

    /// Measured timing, or `None` when only the order of lines is known.
    pub timing: Option<LineTiming>,
}

/// One description of a recording.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Caption {
    /// Free-form text.
    pub text: String,

    /// Provenance.
    pub source: CaptionSource,
}

/// Train/validation membership.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Split {
    /// Used for gradient updates.
    Train,

    /// Held out for measuring generalisation.
    Validation,
}

/// One recording and its descriptions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Track {
    /// File-system-safe unique id.
    pub id: String,

    /// Location of the audio file.
    pub audio_path: PathBuf,

    /// All descriptions; the sampler picks one at random each time.
    pub captions: Vec<Caption>,

    /// The words sung, in order; empty for instrumental or unannotated audio.
    /// Absent from manifests written before lyrics were supported.
    #[serde(default)]
    pub lyrics: Vec<LyricLine>,

    /// Recordings sharing this key always land in the same split (the artist
    /// for Song Describer), which prevents leakage between partitions.
    pub split_group: String,

    /// Partition derived from `split_group`.
    pub split: Split,
}

/// FNV-1a, a fixed hash so splits are identical on every machine
/// (<http://www.isthe.com/chongo/tech/comp/fnv/>).
fn stable_hash(text: &str) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    text.bytes()
        .fold(OFFSET_BASIS, |hash, byte| (hash ^ byte as u64).wrapping_mul(PRIME))
}

/// The split of a group: a deterministic ~[`VALIDATION_PERCENT`] % hold-out.
pub fn split_for(group: &str) -> Split {
    if stable_hash(group) % 100 < VALIDATION_PERCENT { Split::Validation } else { Split::Train }
}

/// Keeps ASCII letters, digits, `-` and `_`; replaces everything else.
fn file_safe(text: &str) -> String {
    text.chars()
        .map(|character| if character.is_ascii_alphanumeric() || character == '-' { character } else { '_' })
        .collect()
}

/// One row of `song_describer.csv` (only the columns used).
#[derive(Deserialize)]
struct SongDescriberRow {
    /// Jamendo track id.
    track_id: String,

    /// Human-written caption.
    caption: String,

    /// Jamendo artist id, used as the split group.
    artist_id: String,

    /// Audio path relative to the audio directory, e.g. `34/1004034.mp3`.
    path: String,
}

/// Reads the Song Describer Dataset extracted under `root`. A track's audio
/// may be stored as `NN/ID.mp3` or `NN/ID.2min.mp3`; both are accepted.
pub fn song_describer(root: &Path) -> Result<Vec<Track>> {
    let captions_path = root.join(SONG_DESCRIBER_CAPTIONS);
    let mut reader = csv::Reader::from_path(&captions_path).with_context(|| format!("open {}", captions_path.display()))?;
    let audio_root = find_audio_root(&root.join(SONG_DESCRIBER_AUDIO))?;

    let mut tracks: BTreeMap<String, Track> = BTreeMap::new();
    for row in reader.deserialize() {
        let row: SongDescriberRow = row?;
        let caption = Caption { text: row.caption.trim().to_string(), source: CaptionSource::HumanAnnotated };

        if let Some(track) = tracks.get_mut(&row.track_id) {
            track.captions.push(caption);
            continue;
        }

        let Some(audio_path) = locate_song_describer_audio(&audio_root, &row.path) else { continue };
        let split_group = format!("artist-{}", row.artist_id);
        tracks.insert(
            row.track_id.clone(),
            Track {
                id: format!("sdd-{}", file_safe(&row.track_id)),
                audio_path,
                captions: vec![caption],
                lyrics: Vec::new(),
                split: split_for(&split_group),
                split_group,
            },
        );
    }

    Ok(tracks.into_values().collect())
}

/// The directory that directly contains the two-digit shard folders. Archives
/// sometimes add one wrapping directory, so descend through single-child chains.
fn find_audio_root(directory: &Path) -> Result<PathBuf> {
    let mut current = directory.to_path_buf();
    loop {
        let children: Vec<PathBuf> = std::fs::read_dir(&current)
            .with_context(|| format!("list {}", current.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.is_dir())
            .collect();

        match children.as_slice() {
            [only] if !is_shard_directory(only) => current = only.clone(),
            _ => return Ok(current),
        }
    }
}

/// True for the two-digit shard folders (`00` … `99`) of the Jamendo layout.
fn is_shard_directory(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.len() == 2 && name.bytes().all(|byte| byte.is_ascii_digit()))
}

/// Resolves a caption-file path to an existing audio file, trying the
/// two-minute-clip naming as well.
fn locate_song_describer_audio(audio_root: &Path, relative: &str) -> Option<PathBuf> {
    let exact = audio_root.join(relative);
    let clip = audio_root.join(relative.replace(".mp3", ".2min.mp3"));
    [exact, clip]
        .into_iter()
        .find(|candidate| candidate.is_file())
}

/// Every audio file under `root` (recursively), described by its sidecar
/// `.txt` (one caption per non-empty line) or, failing that, its file name.
///
/// A file directly in `root` is its own split group. Files inside a subfolder
/// share that subfolder as their group, so a folder per artist keeps all of an
/// artist's songs on the same side of the train/validation split, and the
/// validation loss measures how well the model handles voices it never heard.
pub fn folder(root: &Path) -> Result<Vec<Track>> {
    let mut audio_files = Vec::new();
    collect_audio_files(root, &mut audio_files)?;
    audio_files.sort();

    let mut tracks = Vec::with_capacity(audio_files.len());
    for audio_path in audio_files {
        let stem = audio_path
            .file_stem()
            .map(|stem| stem.to_string_lossy().to_string())
            .unwrap_or_default();
        let relative = audio_path
            .strip_prefix(root)
            .unwrap_or(&audio_path)
            .to_string_lossy()
            .to_string();
        let id = format!("own-{}", file_safe(&relative));

        let sidecar = audio_path.with_extension("txt");
        let captions = match std::fs::read_to_string(&sidecar) {
            Ok(text) => text
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(|line| Caption { text: line.to_string(), source: CaptionSource::Sidecar })
                .collect(),
            Err(_) => vec![Caption { text: stem.replace(['_', '-'], " "), source: CaptionSource::FileName }],
        };

        let lyrics = std::fs::read_to_string(audio_path.with_extension(LYRICS_SIDECAR_SUFFIX))
            .map(|text| untimed_lines(&text))
            .unwrap_or_default();
        let split_group = folder_split_group(root, &audio_path).unwrap_or_else(|| id.clone());
        tracks.push(Track { split: split_for(&split_group), split_group, id, audio_path, captions, lyrics });
    }

    Ok(tracks)
}

/// The first subfolder of `root` that contains `audio_path`, as a split group,
/// or `None` for a file directly in `root`.
fn folder_split_group(root: &Path, audio_path: &Path) -> Option<String> {
    let relative = audio_path.strip_prefix(root).ok()?;
    let mut components = relative.components();
    let first = components.next()?;
    components.next()?;

    Some(format!("folder-{}", first.as_os_str().to_string_lossy()))
}

/// Non-empty lines of `text` as lyrics with unknown timing.
fn untimed_lines(text: &str) -> Vec<LyricLine> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| LyricLine { text: line.to_string(), timing: None })
        .collect()
}

/// One row of `JamendoLyrics.csv` (only the columns used).
#[derive(Deserialize)]
struct JamendoLyricsRow {
    /// Audio file name inside `mp3/`, e.g. `HILA_-_Give_Me_the_Same.mp3`.
    #[serde(rename = "Filepath")]
    file_path: String,

    /// Artist name, used as the split group.
    #[serde(rename = "Artist")]
    artist: String,

    /// Genre label, e.g. `Rock`.
    #[serde(rename = "Genre")]
    genre: String,

    /// Language of the lyrics, e.g. `English`.
    #[serde(rename = "Language")]
    language: String,
}

/// One row of an `annotations/lines/*.csv` file.
#[derive(Deserialize)]
struct JamendoLyricsLine {
    /// Start of the line in seconds.
    start_time: f32,

    /// End of the line in seconds.
    end_time: f32,

    /// The words of the line.
    lyrics_line: String,
}

/// Reads JamendoLyrics MultiLang cloned under `root`. The dataset has no
/// free-text descriptions, so each song gets one assembled from its genre and
/// language, marked [`CaptionSource::Metadata`].
pub fn jamendo_lyrics(root: &Path) -> Result<Vec<Track>> {
    let metadata_path = root.join(JAMENDO_LYRICS_METADATA);
    let mut reader = csv::Reader::from_path(&metadata_path).with_context(|| format!("open {}", metadata_path.display()))?;

    let mut tracks = Vec::new();
    for row in reader.deserialize() {
        let row: JamendoLyricsRow = row?;
        let stem = row.file_path.trim_end_matches(".mp3");
        let lines_path = root.join(JAMENDO_LYRICS_LINES).join(format!("{stem}.csv"));

        let mut lines_reader = csv::Reader::from_path(&lines_path).with_context(|| format!("open {}", lines_path.display()))?;
        let mut lyrics = Vec::new();
        for line in lines_reader.deserialize() {
            let line: JamendoLyricsLine = line?;
            lyrics.push(LyricLine {
                text: line.lyrics_line.trim().to_string(),
                timing: Some(LineTiming { start: line.start_time, end: line.end_time }),
            });
        }

        let caption = format!("{} song with vocals sung in {}", row.genre.trim().to_lowercase(), row.language.trim());
        let split_group = format!("jamendolyrics-artist-{}", row.artist.trim().to_lowercase());
        tracks.push(Track {
            id: format!("jl-{}", file_safe(stem)),
            audio_path: root.join(JAMENDO_LYRICS_AUDIO).join(&row.file_path),
            captions: vec![Caption { text: caption, source: CaptionSource::Metadata }],
            lyrics,
            split: split_for(&split_group),
            split_group,
        });
    }

    Ok(tracks)
}

/// Depth-first collection of audio files.
fn collect_audio_files(directory: &Path, found: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(directory).with_context(|| format!("list {}", directory.display()))? {
        let path = entry?.path();
        if path.is_dir() {
            collect_audio_files(&path, found)?;
        } else if is_audio_file(&path) {
            found.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_are_deterministic_and_roughly_ten_percent() {
        let validation = (0..10_000)
            .filter(|index| split_for(&format!("artist-{index}")) == Split::Validation)
            .count();
        assert!((800..1200).contains(&validation), "{validation} validation groups");
        assert_eq!(split_for("artist-42"), split_for("artist-42"));
    }

    #[test]
    fn lyrics_sidecars_become_untimed_lines() {
        let lines = untimed_lines("first line\n\n  second line  \n");
        assert_eq!(
            lines,
            vec![LyricLine { text: "first line".into(), timing: None }, LyricLine { text: "second line".into(), timing: None }]
        );
    }

    #[test]
    fn subfolders_are_split_groups_and_root_files_are_their_own() {
        let root = Path::new("/music");

        assert_eq!(folder_split_group(root, Path::new("/music/artist-7/song.mp3")).as_deref(), Some("folder-artist-7"));
        assert_eq!(folder_split_group(root, Path::new("/music/artist-7/live/song.mp3")).as_deref(), Some("folder-artist-7"));
        assert_eq!(folder_split_group(root, Path::new("/music/take.wav")), None);
    }

    #[test]
    fn ids_are_file_safe() {
        assert_eq!(file_safe("a b/c.mp3"), "a_b_c_mp3");
    }
}
