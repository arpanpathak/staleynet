//! The rock songs of MTG-Jamendo (Bogdanov et al., 2019,
//! <https://github.com/MTG/mtg-jamendo-dataset>): selecting them from the tag
//! file, downloading each complete song, and arranging the songs into the
//! folder layout that `staleynet prepare --folder` reads.
//!
//! The dataset's own downloader only offers whole tar archives covering every
//! genre, so the rock tracks are fetched one by one from Jamendo's storage.

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// Genres kept, as written after `genre---` in the tag file.
pub const ROCK_GENRES: [&str; 19] = [
    "rock",
    "punkrock",
    "grunge",
    "alternative",
    "hardrock",
    "metal",
    "punk",
    "indie",
    "poprock",
    "rocknroll",
    "progressive",
    "heavymetal",
    "alternativerock",
    "classicrock",
    "garage",
    "postrock",
    "emo",
    "psychedelic",
    "stoner",
];

/// Columns of the tag file before the tags begin: track, artist, album, path, duration.
const LEADING_COLUMNS: usize = 5;

/// Separator between a tag's family and its name, as in `genre---rock`.
const TAG_SEPARATOR: &str = "---";

/// Tag family of genres.
const GENRE_FAMILY: &str = "genre";

/// Tag families in caption order: genres first, then moods, then instruments.
const CAPTION_FAMILIES: [&str; 3] = [GENRE_FAMILY, "mood/theme", "instrument"];

/// Tag spellings that read better split into words.
const SPELLING: [(&str, &str); 11] = [
    ("punkrock", "punk rock"),
    ("hardrock", "hard rock"),
    ("poprock", "pop rock"),
    ("rocknroll", "rock and roll"),
    ("heavymetal", "heavy metal"),
    ("alternativerock", "alternative rock"),
    ("classicrock", "classic rock"),
    ("postrock", "post-rock"),
    ("electricguitar", "electric guitar"),
    ("acousticguitar", "acoustic guitar"),
    ("drummachine", "drum machine"),
];

/// Prefix of a track id in the tag file, as in `track_0000214`.
const TRACK_PREFIX: &str = "track_";

/// Prefix of an artist id in the tag file, as in `artist_000005`.
const ARTIST_PREFIX: &str = "artist_";

/// Jamendo storage address of one track; `mp32` is the high-quality variable-bitrate MP3.
const DOWNLOAD_URL: &str = "https://prod-1.storage.jamendo.com/?format=mp32&trackid=";

/// Attempts per track before giving up on it.
const ATTEMPTS: usize = 3;

/// Pause between attempts.
const RETRY_PAUSE: Duration = Duration::from_secs(5);

/// Files smaller than this are treated as failed downloads.
const MIN_BYTES: u64 = 50_000;

/// Downloads between progress lines.
const REPORT_EVERY: usize = 100;

/// The track list written beside the downloads.
pub const TRACK_LIST: &str = "tracks.tsv";

/// Extension of a downloaded song.
const AUDIO_EXTENSION: &str = "mp3";

/// Extension of an unfinished download.
const PARTIAL_EXTENSION: &str = "part";

/// Extension of a caption sidecar.
const CAPTION_EXTENSION: &str = "txt";

/// One rock track of the tag file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JamendoTrack {
    /// Numeric track id without prefix or leading zeros, as Jamendo's storage expects.
    pub id: String,

    /// Artist id as written in the tag file (`artist_000005`).
    pub artist: String,

    /// Every tag, as `family---name`.
    pub tags: Vec<String>,
}

impl JamendoTrack {
    /// Parses one row of the tag file.
    fn parse(row: &str) -> Option<Self> {
        let columns: Vec<&str> = row.split('\t').collect();
        if columns.len() <= LEADING_COLUMNS {
            return None;
        }

        let id = columns[0]
            .strip_prefix(TRACK_PREFIX)?
            .trim_start_matches('0')
            .to_string();
        let artist = columns[1].to_string();
        let tags = columns[LEADING_COLUMNS..]
            .iter()
            .map(|tag| tag.to_string())
            .collect();

        Some(Self { id, artist, tags })
    }

    /// The names of the tags in `family`, in file order.
    fn names_in<'a>(&'a self, family: &'a str) -> impl Iterator<Item = &'a str> {
        self.tags.iter().filter_map(move |tag| {
            let (kind, name) = tag.split_once(TAG_SEPARATOR)?;
            (kind == family).then_some(name)
        })
    }

    /// Whether any of the track's genres is a rock genre.
    pub fn is_rock(&self) -> bool {
        self.names_in(GENRE_FAMILY)
            .any(|genre| ROCK_GENRES.contains(&genre))
    }

    /// A comma-separated caption in the style of the FMA captions, such as
    /// `rock, grunge, dark, voice`.
    pub fn caption(&self) -> String {
        let words: Vec<&str> = CAPTION_FAMILIES
            .iter()
            .flat_map(|family| self.names_in(family))
            .map(spelled)
            .collect();

        words.join(", ")
    }

    /// One line of the track list: id, artist and comma-joined tags.
    fn list_row(&self) -> String {
        format!("{}\t{}\t{}", self.id, self.artist, self.tags.join(","))
    }

    /// Parses one line of the track list.
    fn from_list_row(row: &str) -> Option<Self> {
        let mut columns = row.split('\t');
        let id = columns.next()?.to_string();
        let artist = columns.next()?.to_string();
        let tags = columns.next()?.split(',').map(str::to_string).collect();

        Some(Self { id, artist, tags })
    }
}

/// The readable spelling of a tag name.
fn spelled(name: &str) -> &str {
    let Some((_, spelling)) = SPELLING.iter().find(|(tag, _)| *tag == name) else { return name };
    spelling
}

/// Every rock track of the tag file (`raw_30s_cleantags.tsv`).
pub fn rock_tracks(tag_file: &Path) -> Result<Vec<JamendoTrack>> {
    let text = fs::read_to_string(tag_file).with_context(|| format!("read {}", tag_file.display()))?;

    let tracks = text
        .lines()
        .skip(1)
        .filter_map(JamendoTrack::parse)
        .filter(JamendoTrack::is_rock)
        .collect();

    Ok(tracks)
}

/// What happened to one track's download.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fetched {
    /// Already on disk.
    Skipped,

    /// Downloaded now.
    Downloaded,

    /// Every attempt failed.
    Failed,
}

/// Download counts so far.
#[derive(Default)]
struct Tally {
    done: AtomicUsize,
    skipped: AtomicUsize,
    downloaded: AtomicUsize,
    failed: AtomicUsize,
}

impl Tally {
    /// Counts one outcome; returns how many tracks are done.
    fn count(&self, fetched: Fetched) -> usize {
        let counter = match fetched {
            Fetched::Skipped => &self.skipped,
            Fetched::Downloaded => &self.downloaded,
            Fetched::Failed => &self.failed,
        };
        counter.fetch_add(1, Ordering::Relaxed);

        self.done.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// `downloaded … skipped … failed …`.
    fn summary(&self) -> String {
        format!(
            "downloaded {}, skipped {}, failed {}",
            self.downloaded.load(Ordering::Relaxed),
            self.skipped.load(Ordering::Relaxed),
            self.failed.load(Ordering::Relaxed),
        )
    }
}

/// Where a track's song is stored in the download folder.
fn download_path(folder: &Path, id: &str) -> PathBuf {
    folder.join(id).with_extension(AUDIO_EXTENSION)
}

/// Downloads one track into `target` unless it is already there.
fn fetch(target: &Path, id: &str) -> Fetched {
    let present = fs::metadata(target).is_ok_and(|metadata| metadata.len() >= MIN_BYTES);
    if present {
        return Fetched::Skipped;
    }

    let partial = target.with_extension(PARTIAL_EXTENSION);
    for attempt in 0..ATTEMPTS {
        if attempt > 0 {
            std::thread::sleep(RETRY_PAUSE);
        }

        let Ok(bytes) = download(&format!("{DOWNLOAD_URL}{id}"), &partial) else { continue };
        if bytes >= MIN_BYTES && fs::rename(&partial, target).is_ok() {
            return Fetched::Downloaded;
        }
    }

    Fetched::Failed
}

/// Streams `url` into `path`; returns the bytes written.
fn download(url: &str, path: &Path) -> Result<u64> {
    let mut response = ureq::get(url).call()?;
    let mut file = fs::File::create(path)?;
    let bytes = std::io::copy(&mut response.body_mut().as_reader(), &mut file)?;
    file.flush()?;

    Ok(bytes)
}

/// Writes the track list into `folder`, then downloads every track with
/// `workers` parallel connections. Resumable: songs already on disk are skipped.
/// Returns how many tracks failed.
pub fn download_tracks(tracks: &[JamendoTrack], folder: &Path, workers: usize) -> Result<usize> {
    fs::create_dir_all(folder)?;

    let list: Vec<String> = tracks.iter().map(JamendoTrack::list_row).collect();
    fs::write(folder.join(TRACK_LIST), list.join("\n") + "\n")?;

    let tally = Tally::default();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;

    pool.install(|| {
        tracks.par_iter().for_each(|track| {
            let done = tally.count(fetch(&download_path(folder, &track.id), &track.id));
            if done % REPORT_EVERY == 0 || done == tracks.len() {
                println!("{done}/{} {}", tracks.len(), tally.summary());
            }
        });
    });

    Ok(tally.failed.load(Ordering::Relaxed))
}

/// Hard-links every downloaded song into `output/artist-<id>/<track>.mp3`
/// beside a caption `<track>.txt`. One folder per artist keeps every artist
/// on one side of the train/validation split, and hard links take no extra
/// disk space. Returns how many songs were arranged.
pub fn arrange(downloads: &Path, output: &Path) -> Result<usize> {
    let list_path = downloads.join(TRACK_LIST);
    let list = fs::read_to_string(&list_path).with_context(|| format!("read {} (download first)", list_path.display()))?;

    let mut arranged = 0;
    for track in list.lines().filter_map(JamendoTrack::from_list_row) {
        let source = download_path(downloads, &track.id);
        if !source.exists() {
            continue;
        }

        let artist = track
            .artist
            .strip_prefix(ARTIST_PREFIX)
            .unwrap_or(&track.artist);
        let folder = output.join(format!("artist-{artist}"));
        fs::create_dir_all(&folder)?;

        let target = download_path(&folder, &track.id);
        if !target.exists() {
            fs::hard_link(&source, &target).with_context(|| format!("link {}", target.display()))?;
        }

        fs::write(folder.join(&track.id).with_extension(CAPTION_EXTENSION), track.caption() + "\n")?;
        arranged += 1;
    }

    if arranged == 0 {
        bail!("no downloaded songs found in {}", downloads.display());
    }

    Ok(arranged)
}

/// How many artist folders `output` holds.
pub fn artist_folders(output: &Path) -> Result<usize> {
    let folders = fs::read_dir(output)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_dir())
        .count();

    Ok(folders)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROW: &str =
        "track_0000214\tartist_000014\talbum_000031\t14/214.mp3\t124.6\tgenre---punkrock\tinstrument---electricguitar\tmood/theme---dark";

    #[test]
    fn parses_a_tag_row() {
        let track = JamendoTrack::parse(ROW).unwrap();

        assert_eq!(track.id, "214");
        assert_eq!(track.artist, "artist_000014");
        assert!(track.is_rock());
    }

    #[test]
    fn captions_genres_then_moods_then_instruments() {
        let track = JamendoTrack::parse(ROW).unwrap();

        assert_eq!(track.caption(), "punk rock, dark, electric guitar");
    }

    #[test]
    fn track_list_rows_round_trip() {
        let track = JamendoTrack::parse(ROW).unwrap();

        assert_eq!(JamendoTrack::from_list_row(&track.list_row()), Some(track));
    }
}
