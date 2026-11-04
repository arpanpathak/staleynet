//! On-disk layout of a prepared dataset.
//!
//! ```text
//! <cache>/
//!   manifest.json        tracks, mel configuration, normalisation statistics
//!   tokenizer.json       trained residual codebooks
//!   mel/<id>.f16         log-mel frames, IEEE half precision, [frames, bands]
//!   codes/<id>.u16       tokenizer codes, [steps, codebooks]
//!   waveforms/<id>.f16   resampled, level-normalized samples (for the vocoder)
//! ```
//!
//! Half precision halves the cache with an error (≈ 0.001 relative) far
//! below the tokenizer's quantisation error.
//! See <https://en.wikipedia.org/wiki/Half-precision_floating-point_format>

use crate::catalog::Track;
use anyhow::{Context, Result};
use half::f16;
use serde::{Deserialize, Serialize};
use staleynet_audio::MelConfig;
use std::path::{Path, PathBuf};

/// Bytes of one stored half-precision value.
const F16_BYTES: usize = 2;

/// Bytes of one stored code.
const U16_BYTES: usize = 2;

/// Paths inside a dataset cache directory.
#[derive(Clone, Debug)]
pub struct CacheLayout {
    /// Cache root.
    pub root: PathBuf,
}

impl CacheLayout {
    /// The layout rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// `manifest.json`.
    pub fn manifest(&self) -> PathBuf {
        self.root.join("manifest.json")
    }

    /// `tokenizer.json`.
    pub fn tokenizer(&self) -> PathBuf {
        self.root.join("tokenizer.json")
    }

    /// Log-mel file of track `id`.
    pub fn mel(&self, id: &str) -> PathBuf {
        self.root.join("mel").join(format!("{id}.f16"))
    }

    /// Code file of track `id`.
    pub fn codes(&self, id: &str) -> PathBuf {
        self.root.join("codes").join(format!("{id}.u16"))
    }

    /// Waveform file of track `id`: the samples its log-mel was computed from.
    pub fn waveform(&self, id: &str) -> PathBuf {
        self.root.join("waveforms").join(format!("{id}.f16"))
    }

    /// Creates the sub-directories.
    pub fn create(&self) -> Result<()> {
        std::fs::create_dir_all(self.root.join("mel"))?;
        std::fs::create_dir_all(self.root.join("codes"))?;
        Ok(())
    }
}

/// Serializable copy of [`MelConfig`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MelSettings {
    /// See [`MelConfig::sample_rate`].
    pub sample_rate: u32,

    /// See [`MelConfig::fft_size`].
    pub fft_size: usize,

    /// See [`MelConfig::hop_size`].
    pub hop_size: usize,

    /// See [`MelConfig::mel_bands`].
    pub mel_bands: usize,

    /// See [`MelConfig::min_frequency`].
    pub min_frequency: f32,

    /// See [`MelConfig::max_frequency`].
    pub max_frequency: f32,
}

impl From<MelConfig> for MelSettings {
    fn from(config: MelConfig) -> Self {
        let MelConfig { sample_rate, fft_size, hop_size, mel_bands, min_frequency, max_frequency } = config;
        Self { sample_rate, fft_size, hop_size, mel_bands, min_frequency, max_frequency }
    }
}

impl From<MelSettings> for MelConfig {
    fn from(settings: MelSettings) -> Self {
        let MelSettings { sample_rate, fft_size, hop_size, mel_bands, min_frequency, max_frequency } = settings;
        Self { sample_rate, fft_size, hop_size, mel_bands, min_frequency, max_frequency }
    }
}

/// Per-band mean and standard deviation of training log-mels, used to whiten
/// features before tokenisation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Normalization {
    /// Mean log energy of each band.
    pub mean: Vec<f32>,

    /// Standard deviation of each band.
    pub std: Vec<f32>,
}

/// Running sums for [`Normalization`], accumulated in `f64`.
pub struct BandStatistics {
    /// Σ x per band.
    sum: Vec<f64>,

    /// Σ x² per band.
    sum_of_squares: Vec<f64>,

    /// Frames seen.
    frames: usize,
}

/// Lower bound on a band's variance, so silent bands do not divide by zero.
const MIN_BAND_VARIANCE: f64 = 1e-4;

impl BandStatistics {
    /// Empty statistics over `bands` bands.
    pub fn new(bands: usize) -> Self {
        Self { sum: vec![0.0; bands], sum_of_squares: vec![0.0; bands], frames: 0 }
    }

    /// Adds every frame of a `[frames, bands]` log-mel.
    pub fn add(&mut self, log_mel: &[f32]) {
        for frame in log_mel.chunks_exact(self.sum.len()) {
            for (band, &value) in frame.iter().enumerate() {
                self.sum[band] += value as f64;
                self.sum_of_squares[band] += (value as f64).powi(2);
            }
            self.frames += 1;
        }
    }

    /// Mean and standard deviation per band.
    pub fn finish(&self) -> Normalization {
        let count = self.frames.max(1) as f64;
        let mean: Vec<f64> = self.sum.iter().map(|sum| sum / count).collect();
        let std = self
            .sum_of_squares
            .iter()
            .zip(&mean)
            .map(|(squares, mean)| (squares / count - mean * mean).max(MIN_BAND_VARIANCE).sqrt() as f32)
            .collect();

        Normalization { mean: mean.into_iter().map(|value| value as f32).collect(), std }
    }
}

/// A track that was successfully prepared.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreparedTrack {
    /// Catalogue entry.
    pub track: Track,

    /// Duration after resampling, in seconds.
    pub seconds: f32,

    /// Log-mel frames stored.
    pub frames: usize,
}

/// Contents of `manifest.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetManifest {
    /// Analysis parameters every stored log-mel was computed with.
    pub mel: MelSettings,

    /// Statistics over the training split.
    pub normalization: Normalization,

    /// Every prepared track.
    pub tracks: Vec<PreparedTrack>,
}

impl DatasetManifest {
    /// Reads `manifest.json`.
    pub fn load(layout: &CacheLayout) -> Result<Self> {
        let text = std::fs::read_to_string(layout.manifest())
            .with_context(|| format!("read {} (run `staleynet prepare` first)", layout.manifest().display()))?;
        Ok(serde_json::from_str(&text)?)
    }

    /// Writes `manifest.json`.
    pub fn save(&self, layout: &CacheLayout) -> Result<()> {
        std::fs::write(layout.manifest(), serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}

/// Writes `values` as little-endian half floats.
pub fn write_f16(path: &Path, values: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|&value| f16::from_f32(value).to_le_bytes()).collect();
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

/// Reads a file written by [`write_f16`].
pub fn read_f16(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(bytes.chunks_exact(F16_BYTES).map(|pair| f16::from_le_bytes([pair[0], pair[1]]).to_f32()).collect())
}

/// Writes codes as little-endian `u16` (codebooks never exceed 65 536 entries).
pub fn write_codes(path: &Path, codes: &[u32]) -> Result<()> {
    let bytes: Vec<u8> = codes.iter().flat_map(|&code| u16::try_from(code).expect("code exceeds u16").to_le_bytes()).collect();
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

/// Reads a file written by [`write_codes`].
pub fn read_codes(path: &Path) -> Result<Vec<u32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(bytes.chunks_exact(U16_BYTES).map(|pair| u16::from_le_bytes([pair[0], pair[1]]) as u32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn temp_root(tag: &str) -> PathBuf {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);

        std::env::temp_dir().join(format!("staleynet-store-{tag}-{}-{id}", std::process::id()))
    }

    #[test]
    fn cache_layout_names_every_artifact() {
        let root = temp_root("paths");
        let layout = CacheLayout::new(&root);

        assert_eq!(layout.manifest(), root.join("manifest.json"));
        assert_eq!(layout.tokenizer(), root.join("tokenizer.json"));
        assert_eq!(layout.mel("song"), root.join("mel").join("song.f16"));
        assert_eq!(layout.codes("song"), root.join("codes").join("song.u16"));
    }

    #[test]
    fn create_makes_the_mel_and_code_directories() {
        let root = temp_root("create");
        let layout = CacheLayout::new(&root);

        layout.create().unwrap();

        assert!(root.join("mel").is_dir());
        assert!(root.join("codes").is_dir());

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn f16_values_round_trip_with_half_precision() {
        let root = temp_root("f16");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("values.f16");

        let values = [0.0, 1.0, -0.5, 0.1];

        write_f16(&path, &values).unwrap();
        let decoded = read_f16(&path).unwrap();

        assert_eq!(decoded.len(), values.len());
        for (actual, expected) in decoded.iter().zip(values) {
            assert!((actual - expected).abs() < 1e-3);
        }

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn codes_round_trip_within_u16() {
        let root = temp_root("codes");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("codes.u16");

        let codes = [0, 1, 1_024, 65_535];

        write_codes(&path, &codes).unwrap();

        assert_eq!(read_codes(&path).unwrap(), codes);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[should_panic(expected = "code exceeds u16")]
    fn codes_above_u16_are_rejected() {
        let root = temp_root("overflow");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("codes.u16");

        write_codes(&path, &[65_536]).unwrap();
    }

    #[test]
    fn band_statistics_compute_population_mean_and_std() {
        let mut statistics = BandStatistics::new(2);

        statistics.add(&[1.0, 2.0, 3.0, 4.0]);

        let normalization = statistics.finish();

        assert_eq!(normalization.mean, vec![2.0, 3.0]);
        assert_eq!(normalization.std, vec![1.0, 1.0]);
    }

    #[test]
    fn empty_band_statistics_use_the_variance_floor() {
        let statistics = BandStatistics::new(2);

        let normalization = statistics.finish();

        assert_eq!(normalization.mean, vec![0.0, 0.0]);
        assert!(normalization.std.iter().all(|std| (*std - 0.01).abs() < 1e-6));
    }

    #[test]
    fn manifest_round_trips_through_disk() {
        let root = temp_root("manifest");
        let layout = CacheLayout::new(&root);
        layout.create().unwrap();

        let manifest = DatasetManifest {
            mel: MelSettings::from(MelConfig::default()),
            normalization: Normalization { mean: vec![0.0, 1.0], std: vec![1.0, 2.0] },
            tracks: Vec::new(),
        };

        manifest.save(&layout).unwrap();
        let decoded = DatasetManifest::load(&layout).unwrap();

        assert_eq!(decoded.mel, manifest.mel);
        assert_eq!(decoded.normalization, manifest.normalization);
        assert!(decoded.tracks.is_empty());

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mel_settings_convert_back_to_the_audio_config() {
        let config = MelConfig::default();
        let settings = MelSettings::from(config);

        assert_eq!(MelConfig::from(settings), config);
    }

    #[test]
    fn loading_a_missing_manifest_reports_an_error() {
        let layout = CacheLayout::new(temp_root("missing"));

        assert!(DatasetManifest::load(&layout).is_err());
    }
}
