//! A rock corpus built from a list of bands: their studio songs from
//! MusicBrainz (<https://musicbrainz.org/doc/MusicBrainz_API>), timed lyrics
//! from LRCLIB (<https://lrclib.net/docs>), and the audio from YouTube through
//! `yt-dlp` (<https://github.com/yt-dlp/yt-dlp>).
//!
//! Every song is matched by duration three times: MusicBrainz gives the album
//! length, LRCLIB only returns lyrics timed against a recording of about that
//! length, and only a YouTube upload within [`DURATION_TOLERANCE_SECONDS`] of
//! it is downloaded. That keeps the lyric timing on the right version of the
//! song: not a live take, not a video with a spoken intro.
//!
//! The output is the folder layout `staleynet prepare --folder` reads:
//!
//! ```text
//! <out>/artist-<band>/<song>.m4a   audio as uploaded (AAC, not re-encoded)
//! <out>/artist-<band>/<song>.lrc   timed lyrics
//! <out>/artist-<band>/<song>.txt   caption from the band's MusicBrainz genres
//! ```

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// MusicBrainz asks for at most one request per second from each client.
const MUSICBRAINZ_INTERVAL: Duration = Duration::from_millis(1100);

/// MusicBrainz and LRCLIB ask clients to identify themselves.
const USER_AGENT: &str = "staleynet-research/0.1 (https://github.com/arpanpathak/staleynet)";

/// MusicBrainz search address.
const MUSICBRAINZ: &str = "https://musicbrainz.org/ws/2";

/// LRCLIB lyrics lookup address.
const LRCLIB_GET: &str = "https://lrclib.net/api/get";

/// HTTP statuses that mean "too busy, try again later".
const BUSY_STATUSES: [u16; 2] = [429, 503];

/// Retries of a busy answer before giving up.
const BUSY_RETRIES: u32 = 6;

/// Pause before the first retry of a busy answer; later retries wait longer.
const BUSY_PAUSE: Duration = Duration::from_secs(2);

/// Recordings per MusicBrainz page, the service's maximum.
const PAGE_SIZE: usize = 100;

/// Recordings read per band at most.
const MAX_RECORDINGS: usize = 1000;

/// Genres kept in a band's caption at most.
const CAPTION_GENRES: usize = 4;

/// Words in a recording's MusicBrainz comment that mark a version other than
/// the studio one.
const OTHER_VERSIONS: [&str; 9] = ["live", "demo", "remix", "acoustic", "instrumental", "edit", "mix", "rehearsal", "karaoke"];

/// How far a YouTube upload's length may be from the album length.
const DURATION_TOLERANCE_SECONDS: f32 = 3.0;

/// YouTube search results considered per song.
const SEARCH_RESULTS: usize = 5;

/// Milliseconds in a second, for MusicBrainz lengths.
const MILLISECONDS_PER_SECOND: f32 = 1000.0;

/// Shortest song kept, in seconds; shorter recordings are intros and skits.
const MIN_SONG_SECONDS: f32 = 90.0;

/// Longest song kept, in seconds.
const MAX_SONG_SECONDS: f32 = 600.0;

/// Marker written when a song has no timed lyrics or no matching audio, so a
/// rerun does not ask again.
const MISSING_EXTENSION: &str = "missing";

/// Spaces MusicBrainz requests at least [`MUSICBRAINZ_INTERVAL`] apart across threads.
struct RateLimit {
    last: Mutex<Option<Instant>>,
}

impl RateLimit {
    /// Waits until the next request is allowed.
    fn wait(&self) {
        let mut last = self
            .last
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(previous) = *last {
            std::thread::sleep(MUSICBRAINZ_INTERVAL.saturating_sub(previous.elapsed()));
        }
        *last = Some(Instant::now());
    }
}

/// A MusicBrainz tag or genre.
#[derive(Deserialize)]
struct Tag {
    name: String,
    count: i64,
}

/// A MusicBrainz artist search hit.
#[derive(Deserialize)]
struct Artist {
    id: String,
    #[serde(default)]
    tags: Vec<Tag>,
}

/// A page of MusicBrainz artist search results.
#[derive(Deserialize)]
struct ArtistPage {
    artists: Vec<Artist>,
}

/// A MusicBrainz recording.
#[derive(Deserialize)]
struct Recording {
    title: String,
    length: Option<u64>,
    #[serde(default)]
    disambiguation: String,
}

/// A page of MusicBrainz recording search results.
#[derive(Deserialize)]
struct RecordingPage {
    count: usize,
    recordings: Vec<Recording>,
}

/// LRCLIB's answer for one song.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Lyrics {
    #[serde(default)]
    instrumental: bool,
    synced_lyrics: Option<String>,
}

/// One studio song of a band.
#[derive(Clone, Debug)]
pub struct Song {
    /// Title as MusicBrainz spells it.
    pub title: String,

    /// Album length in seconds.
    pub seconds: f32,
}

/// One band's studio songs and caption.
#[derive(Clone, Debug)]
pub struct Band {
    /// Name as given in the band list.
    pub name: String,

    /// Comma-separated genres, then `voice`.
    pub caption: String,

    /// Distinct studio songs.
    pub songs: Vec<Song>,
}

/// What happened to the songs of one band.
#[derive(Clone, Copy, Debug, Default)]
pub struct BandTally {
    /// Songs on disk with audio and timed lyrics.
    pub ready: usize,

    /// Songs without timed lyrics or matching audio.
    pub missing: usize,
}

/// A GET request with the client's identification. A busy answer (429 or
/// 503, how MusicBrainz signals its rate limit) is retried after a pause that
/// grows with each attempt.
fn get(url: &str, query: &[(&str, &str)]) -> Result<String> {
    let mut attempt = 0;

    loop {
        let mut request = ureq::get(url).header("User-Agent", USER_AGENT);
        for (name, value) in query {
            request = request.query(*name, *value);
        }

        match request.call() {
            Ok(mut response) => return Ok(response.body_mut().read_to_string()?),
            Err(ureq::Error::StatusCode(code)) if BUSY_STATUSES.contains(&code) && attempt < BUSY_RETRIES => {
                attempt += 1;
                std::thread::sleep(BUSY_PAUSE * attempt);
            }
            Err(error) => return Err(error.into()),
        }
    }
}

/// `title` in lower case without bracketed suffixes, so "Rooster (2022 Remaster)"
/// and "Rooster" count as one song.
fn song_key(title: &str) -> String {
    let base = title.split(['(', '[']).next().unwrap_or(title);
    base.trim().to_lowercase()
}

/// `text` reduced to lower-case letters, digits and dashes, for file names.
fn slug(text: &str) -> String {
    let words: Vec<String> = text
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect();

    words.join("-")
}

/// Whether a recording's comment, or the bracketed part of its title as in
/// "Down in a Hole (Radio Edit)", marks it as other than the studio version.
fn is_other_version(title: &str, comment: &str) -> bool {
    let bracketed = title.find(['(', '[']).map_or("", |start| &title[start..]);
    let notes = format!("{bracketed} {comment}").to_lowercase();

    OTHER_VERSIONS.iter().any(|word| notes.contains(word))
}

/// The band's MusicBrainz id and caption.
fn find_band(name: &str, limit: &RateLimit) -> Result<(String, String)> {
    limit.wait();
    let query = format!("artist:\"{name}\"");
    let page: ArtistPage =
        serde_json::from_str(&get(&format!("{MUSICBRAINZ}/artist"), &[("query", &query), ("limit", "1"), ("fmt", "json")])?)?;
    let Some(mut artist) = page.artists.into_iter().next() else { bail!("{name} is not on MusicBrainz") };

    artist
        .tags
        .sort_by(|left, right| right.count.cmp(&left.count));
    let mut words: Vec<String> = artist
        .tags
        .iter()
        .filter(|tag| tag.count > 0)
        .take(CAPTION_GENRES)
        .map(|tag| tag.name.clone())
        .collect();
    words.push("voice".to_string());

    Ok((artist.id, words.join(", ")))
}

/// The band's distinct studio songs of a sensible length.
fn studio_songs(artist_id: &str, limit: &RateLimit) -> Result<Vec<Song>> {
    let query = format!("arid:{artist_id} AND status:official AND primarytype:album");
    let mut seen = BTreeSet::new();
    let mut songs = Vec::new();
    let mut offset = 0;

    loop {
        limit.wait();
        let offset_text = offset.to_string();
        let page_size = PAGE_SIZE.to_string();
        let body = get(
            &format!("{MUSICBRAINZ}/recording"),
            &[("query", &query), ("limit", &page_size), ("offset", &offset_text), ("fmt", "json")],
        )?;
        let page: RecordingPage = serde_json::from_str(&body)?;

        for recording in &page.recordings {
            let Some(length) = recording.length else { continue };
            let seconds = length as f32 / MILLISECONDS_PER_SECOND;
            let sensible = (MIN_SONG_SECONDS..=MAX_SONG_SECONDS).contains(&seconds);
            if !sensible || is_other_version(&recording.title, &recording.disambiguation) || !seen.insert(song_key(&recording.title)) {
                continue;
            }

            songs.push(Song { title: recording.title.clone(), seconds });
        }

        offset += page.recordings.len();
        if page.recordings.is_empty() || offset >= page.count.min(MAX_RECORDINGS) {
            break;
        }
    }

    Ok(songs)
}

/// The band's songs and caption from MusicBrainz.
fn band(name: &str, limit: &RateLimit) -> Result<Band> {
    let (id, caption) = find_band(name, limit)?;
    let songs = studio_songs(&id, limit)?;

    Ok(Band { name: name.to_string(), caption, songs })
}

/// LRCLIB's timed lyrics for a song, or `None` when it has none or is instrumental.
fn timed_lyrics(band: &str, song: &Song) -> Option<String> {
    let duration = format!("{:.0}", song.seconds);
    let body = get(LRCLIB_GET, &[("artist_name", band), ("track_name", &song.title), ("duration", &duration)]).ok()?;
    let lyrics: Lyrics = serde_json::from_str(&body).ok()?;
    if lyrics.instrumental {
        return None;
    }

    lyrics.synced_lyrics.filter(|text| !text.trim().is_empty())
}

/// Downloads the first YouTube upload of the song within the duration
/// tolerance into `target` (an `.m4a` path); returns whether one was found.
fn youtube_audio(yt_dlp: &Path, band: &str, song: &Song, target: &Path) -> Result<bool> {
    let shortest = song.seconds - DURATION_TOLERANCE_SECONDS;
    let longest = song.seconds + DURATION_TOLERANCE_SECONDS;
    let filter = format!("duration>{shortest:.0} & duration<{longest:.0}");
    let search = format!("ytsearch{SEARCH_RESULTS}:{band} {} audio", song.title);

    // yt-dlp exits with 101 when it stops at --max-downloads, so its status
    // says nothing; whether the file exists does.
    Command::new(yt_dlp)
        .args(["--quiet", "--no-warnings", "--no-playlist", "--max-downloads", "1"])
        .args(["--format", "bestaudio[ext=m4a]", "--match-filter", &filter])
        .arg("--output")
        .arg(target)
        .arg(&search)
        .status()
        .with_context(|| format!("run {}", yt_dlp.display()))?;

    Ok(target.exists())
}

/// Fetches lyrics and audio for every song of `band` not already on disk.
fn fetch_band(band: &Band, out: &Path, yt_dlp: &Path) -> Result<BandTally> {
    let folder = out.join(format!("artist-{}", slug(&band.name)));
    fs::create_dir_all(&folder)?;

    let mut tally = BandTally::default();
    for song in &band.songs {
        let stem = folder.join(slug(&song.title));
        let audio = stem.with_extension("m4a");
        let missing = stem.with_extension(MISSING_EXTENSION);

        if audio.exists() {
            tally.ready += 1;
            continue;
        }
        if missing.exists() {
            tally.missing += 1;
            continue;
        }

        let Some(lyrics) = timed_lyrics(&band.name, song) else {
            fs::write(&missing, "no timed lyrics\n")?;
            tally.missing += 1;
            continue;
        };

        if !youtube_audio(yt_dlp, &band.name, song, &audio)? {
            fs::write(&missing, "no upload of matching length\n")?;
            tally.missing += 1;
            continue;
        }

        fs::write(stem.with_extension("lrc"), lyrics)?;
        fs::write(stem.with_extension("txt"), format!("{}\n", band.caption))?;
        tally.ready += 1;
    }

    Ok(tally)
}

/// Builds the corpus for every band named in `bands` (one per line; `#`
/// starts a comment) with `workers` bands in parallel. Resumable: songs
/// already fetched or already found missing are skipped.
pub fn fetch_bands(bands: &Path, out: &Path, yt_dlp: &Path, workers: usize) -> Result<BandTally> {
    let text = fs::read_to_string(bands).with_context(|| format!("read {}", bands.display()))?;
    let names: Vec<&str> = text
        .lines()
        .map(|line| line.split('#').next().unwrap_or("").trim())
        .filter(|line| !line.is_empty())
        .collect();

    let limit = RateLimit { last: Mutex::new(None) };
    let total = Mutex::new(BandTally::default());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;

    pool.install(|| {
        names.par_iter().for_each(|name| {
            let result = band(name, &limit).and_then(|band| fetch_band(&band, out, yt_dlp).map(|tally| (band, tally)));
            let Ok((band, tally)) = result.inspect_err(|error| eprintln!("{name}: {error:#}")) else { return };

            let mut total = total
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            total.ready += tally.ready;
            total.missing += tally.missing;
            println!(
                "{name}: {} songs, {} ready, {} missing | total {} ready, {} missing",
                band.songs.len(),
                tally.ready,
                tally.missing,
                total.ready,
                total.missing
            );
        });
    });

    Ok(total
        .into_inner()
        .unwrap_or_else(|poisoned| poisoned.into_inner()))
}

/// The default location of `yt-dlp`: `~/.local/bin/yt-dlp`.
pub fn default_yt_dlp() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    home.join(".local/bin/yt-dlp")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remasters_are_the_same_song() {
        assert_eq!(song_key("Rooster (2022 Remaster)"), song_key("rooster"));
    }

    #[test]
    fn slugs_are_file_safe() {
        assert_eq!(slug("Would? (Live)"), "would-live");
        assert_eq!(slug("Alice in Chains"), "alice-in-chains");
    }

    #[test]
    fn live_takes_are_other_versions() {
        assert!(is_other_version("Would?", "live, 1996-07-03: Kemper Arena"));
        assert!(is_other_version("Down in a Hole (Radio Edit)", ""));
        assert!(!is_other_version("Would?", ""));
    }
}
