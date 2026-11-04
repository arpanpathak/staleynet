//! The waveform cache: every track's samples, stored beside its tokens.
//!
//! The vocoder's targets must be the exact samples its input spectrograms were
//! computed from, so each recording is decoded, resampled and level-normalized
//! exactly as it was for analysis, and stored as half-precision samples in
//! `<cache>/waveforms/<id>.f16`.

use anyhow::Result;
use rayon::prelude::*;
use staleynet_audio::MelConfig;
use staleynet_data::prepare::{PrepareOptions, load_signal};
use staleynet_data::store::{CacheLayout, DatasetManifest, PreparedTrack, write_f16};

/// Recordings decoded at once while building the waveform cache.
const DECODE_GROUP_SIZE: usize = 32;

/// Directory of the waveform cache inside a dataset cache.
const WAVEFORM_DIRECTORY: &str = "waveforms";

/// What happened to one track while building the waveform cache.
pub enum WaveformOutcome {
    /// Samples written.
    Written {
        /// Catalog id.
        id: String,

        /// Seconds of audio.
        seconds: f32,
    },

    /// Already in the cache from an earlier run.
    Present(String),

    /// Could not be decoded.
    Failed {
        /// Catalog id.
        id: String,

        /// Why.
        reason: String,
    },
}

/// Writes the samples of every track in the cache's manifest. Tracks already
/// present are kept, so an interrupted run can resume.
pub fn prepare_waveforms(layout: &CacheLayout, mut report: impl FnMut(&WaveformOutcome)) -> Result<()> {
    let manifest = DatasetManifest::load(layout)?;
    std::fs::create_dir_all(layout.root.join(WAVEFORM_DIRECTORY))?;

    let options = PrepareOptions { mel: MelConfig::from(manifest.mel), max_seconds: None };

    for group in manifest.tracks.chunks(DECODE_GROUP_SIZE) {
        let outcomes: Vec<WaveformOutcome> = group
            .par_iter()
            .map(|entry| prepare_one(layout, entry, &options))
            .collect();

        outcomes.iter().for_each(&mut report);
    }

    Ok(())
}

/// Writes one track's samples, unless they are already there.
fn prepare_one(layout: &CacheLayout, entry: &PreparedTrack, options: &PrepareOptions) -> WaveformOutcome {
    let id = entry.track.id.clone();
    let path = layout.waveform(&id);
    if path.is_file() {
        return WaveformOutcome::Present(id);
    }

    let written = load_signal(&entry.track, options).and_then(|signal| {
        write_f16(&path, &signal)?;
        Ok(signal.len())
    });

    match written {
        Ok(samples) => WaveformOutcome::Written { id, seconds: samples as f32 / options.mel.sample_rate as f32 },
        Err(error) => WaveformOutcome::Failed { id, reason: format!("{error:#}") },
    }
}
