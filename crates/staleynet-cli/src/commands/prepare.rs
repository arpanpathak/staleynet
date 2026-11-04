//! `staleynet prepare`: catalogue → feature cache.

use super::open_gpu;
use anyhow::Result;
use clap::Args;
use staleynet_audio::MelConfig;
use staleynet_data::catalog::{self, CaptionSource, Split};
use staleynet_data::prepare::{PrepareOptions, PrepareOutcome, prepare};
use staleynet_data::store::CacheLayout;
use std::path::PathBuf;

/// Where recordings come from. Exactly one flag must be given.
#[derive(Args)]
#[group(required = true, multiple = false)]
struct SourceFlags {
    /// Extracted Song Describer Dataset (contains song_describer.csv and audio/).
    #[arg(long)]
    song_describer: Option<PathBuf>,

    /// A clone of JamendoLyrics MultiLang (contains JamendoLyrics.csv, mp3/ and annotations/).
    #[arg(long)]
    jamendo_lyrics: Option<PathBuf>,

    /// A folder of your own recordings with optional same-named .txt captions
    /// and .lyrics.txt lyrics (the studio's data/recordings is one).
    #[arg(long)]
    folder: Option<PathBuf>,
}

/// A dataset source, resolved from [`SourceFlags`].
enum DatasetSource {
    /// The Song Describer Dataset at this path.
    SongDescriber(PathBuf),

    /// JamendoLyrics MultiLang at this path.
    JamendoLyrics(PathBuf),

    /// A folder of recordings at this path.
    Folder(PathBuf),
}

impl SourceFlags {
    /// The single source clap guarantees is present.
    fn resolve(self) -> DatasetSource {
        match (self.song_describer, self.jamendo_lyrics, self.folder) {
            (Some(root), _, _) => DatasetSource::SongDescriber(root),
            (None, Some(root), _) => DatasetSource::JamendoLyrics(root),
            (None, None, Some(root)) => DatasetSource::Folder(root),
            (None, None, None) => unreachable!("clap requires one source"),
        }
    }
}

/// Arguments of `staleynet prepare`.
#[derive(Args)]
pub struct Arguments {
    /// Dataset source.
    #[command(flatten)]
    source: SourceFlags,

    /// Output cache directory.
    #[arg(long, default_value = "data/cache")]
    cache: PathBuf,

    /// Keep at most this many seconds of each recording.
    #[arg(long)]
    max_seconds: Option<f32>,
}

pub fn run(arguments: Arguments) -> Result<()> {
    let tracks = match arguments.source.resolve() {
        DatasetSource::SongDescriber(root) => catalog::song_describer(&root)?,
        DatasetSource::JamendoLyrics(root) => catalog::jamendo_lyrics(&root)?,
        DatasetSource::Folder(root) => catalog::folder(&root)?,
    };

    let with_lyrics = tracks
        .iter()
        .filter(|track| !track.lyrics.is_empty())
        .count();
    let unlabelled = tracks
        .iter()
        .filter(|track| {
            track
                .captions
                .iter()
                .all(|caption| caption.source == CaptionSource::FileName)
        })
        .count();

    eprintln!("catalogue: {} recordings ({unlabelled} described only by file name, {with_lyrics} with lyrics)", tracks.len());

    let gpu = open_gpu()?;
    let layout = CacheLayout::new(&arguments.cache);
    let options = PrepareOptions { mel: MelConfig::default(), max_seconds: arguments.max_seconds };

    let total = tracks.len();
    let mut finished = 0;

    let manifest = prepare(&gpu, tracks, &layout, &options, |outcome| {
        finished += 1;
        match outcome {
            PrepareOutcome::Prepared(entry) => {
                eprintln!("[{finished}/{total}] {} {:.1}s {:?}", entry.track.id, entry.seconds, entry.track.split)
            }
            PrepareOutcome::Skipped { id, reason } => eprintln!("[{finished}/{total}] skipped {id}: {reason}"),
        }
    })?;

    let hours = manifest
        .tracks
        .iter()
        .map(|entry| entry.seconds)
        .sum::<f32>()
        / 3600.0;
    let validation = manifest
        .tracks
        .iter()
        .filter(|entry| entry.track.split == Split::Validation)
        .count();

    eprintln!("prepared {} recordings, {hours:.2} h ({validation} validation) → {}", manifest.tracks.len(), arguments.cache.display());
    Ok(())
}
