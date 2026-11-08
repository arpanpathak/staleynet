//! `staleynet data`: build training corpora: download and arrange the
//! MTG-Jamendo rock songs, and merge transcribed lyrics into a cache.
//!
//! ```text
//! staleynet data jamendo-download --tags data/raw/mtg-jamendo-dataset/data/raw_30s_cleantags.tsv --out data/raw/jamendo-rock
//! staleynet data jamendo-arrange  --downloads data/raw/jamendo-rock --out data/jamendo-rock
//! staleynet data merge-lyrics     --cache data/cache-jamendo-rock-full --transcripts data/jamendo-lyrics/transcripts.jsonl --language en
//! ```

use anyhow::{Result, bail};
use clap::{Args, Subcommand};
use staleynet_data::jamendo::{arrange, artist_folders, download_tracks, rock_tracks};
use staleynet_data::transcripts::{TranscriptFilter, merge_lyrics, read_transcripts};
use std::path::PathBuf;

/// Parallel downloads; kept low to be polite to Jamendo's servers.
const DEFAULT_WORKERS: usize = 6;

/// Arguments of `staleynet data`.
#[derive(Args)]
pub struct Arguments {
    /// What to do.
    #[command(subcommand)]
    task: Task,
}

/// The data tasks.
#[derive(Subcommand)]
enum Task {
    /// Download every complete MTG-Jamendo rock song (resumable).
    JamendoDownload {
        /// The dataset's tag file, `raw_30s_cleantags.tsv`.
        #[arg(long, default_value = "data/raw/mtg-jamendo-dataset/data/raw_30s_cleantags.tsv")]
        tags: PathBuf,

        /// Download folder.
        #[arg(long, default_value = "data/raw/jamendo-rock")]
        out: PathBuf,

        /// Parallel downloads.
        #[arg(long, default_value_t = DEFAULT_WORKERS)]
        workers: usize,
    },

    /// Link the downloaded songs into one folder per artist, with tag captions.
    JamendoArrange {
        /// Download folder.
        #[arg(long, default_value = "data/raw/jamendo-rock")]
        downloads: PathBuf,

        /// Folder for `staleynet prepare --folder`.
        #[arg(long, default_value = "data/jamendo-rock")]
        out: PathBuf,
    },

    /// Merge transcribed lyrics into a cache's manifest.
    MergeLyrics {
        /// Cache whose manifest receives the lyrics.
        #[arg(long)]
        cache: PathBuf,

        /// JSON lines written by the Whisper pass.
        #[arg(long)]
        transcripts: PathBuf,

        /// Lowest mean log-probability kept; every confidence when absent.
        #[arg(long, allow_hyphen_values = true)]
        min_logprob: Option<f32>,

        /// Language to keep, such as `en`; repeat for several; every language when absent.
        #[arg(long = "language")]
        languages: Vec<String>,
    },
}

/// Runs the chosen task.
pub fn run(arguments: Arguments) -> Result<()> {
    match arguments.task {
        Task::JamendoDownload { tags, out, workers } => download(tags, out, workers),
        Task::JamendoArrange { downloads, out } => arrange_songs(downloads, out),
        Task::MergeLyrics { cache, transcripts, min_logprob, languages } => {
            merge(cache, transcripts, TranscriptFilter { min_logprob, languages })
        }
    }
}

/// Selects the rock tracks and downloads them.
fn download(tags: PathBuf, out: PathBuf, workers: usize) -> Result<()> {
    let tracks = rock_tracks(&tags)?;
    println!("{} rock tracks", tracks.len());

    let failed = download_tracks(&tracks, &out, workers)?;
    if failed > 0 {
        bail!("{failed} tracks failed to download; run again to retry them");
    }

    Ok(())
}

/// Arranges the downloads into artist folders.
fn arrange_songs(downloads: PathBuf, out: PathBuf) -> Result<()> {
    let arranged = arrange(&downloads, &out)?;
    println!("arranged {arranged} songs into {} artist folders", artist_folders(&out)?);

    Ok(())
}

/// Merges the accepted transcripts into the cache.
fn merge(cache: PathBuf, transcripts: PathBuf, filter: TranscriptFilter) -> Result<()> {
    let report = merge_lyrics(&cache, &read_transcripts(&transcripts)?, &filter)?;

    println!(
        "{} transcripts, {} accepted; added lyrics to {} tracks; {} of {} now have lyrics",
        report.transcripts, report.accepted, report.added, report.with_lyrics, report.tracks,
    );

    Ok(())
}
