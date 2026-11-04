//! The audio tokenizer: residual vector quantisation of log-mel patches,
//! trained from scratch with k-means.
//!
//! A *step* is `frames_per_step` consecutive normalised log-mel frames. The
//! mel bands are divided into `band_groups` equal groups, and each group of a
//! step is quantised on its own: its first codebook names the nearest
//! prototype, its second quantises what the first missed, and so on. Splitting
//! the bands is product quantisation; a 32-band piece is described far more
//! precisely by 1 024 prototypes than a whole 128-band frame.
//!
//! Codes are ordered level by level: the coarse codes of all band groups come
//! first, then the first corrections of all groups, and so on, which is the
//! coarse-to-fine order the [`DelayPattern`](staleynet_model::DelayPattern)
//! relies on.
//!
//! References:
//! * Residual VQ in SoundStream (Zeghidour et al., 2021, <https://arxiv.org/abs/2107.03312>, §3.3).
//! * Product quantisation (Jégou, Douze and Schmid, 2011, <https://doi.org/10.1109/TPAMI.2010.57>).
//! * Lloyd's algorithm for k-means (Lloyd, 1982, <https://doi.org/10.1109/TIT.1982.1056489>).

use crate::store::Normalization;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use staleynet_cuda::{Gpu, GpuResult, MatmulSpec, Shape, Tensor};
use staleynet_model::Random;
use std::path::Path;

/// Points sent to the GPU per nearest-centroid search; bounds the
/// `[points, centroids]` distance matrix (8192 × 1024 floats = 32 MiB).
const SEARCH_CHUNK: usize = 8192;

/// Tokenizer hyperparameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Log-mel frames per step (2 frames at hop 256 / 24 kHz ≈ 21 ms).
    pub frames_per_step: usize,

    /// Codes per step: residual levels times band groups.
    pub codebooks: usize,

    /// Equal groups of mel bands, each quantised with its own codebooks.
    /// Tokenizers saved before band groups existed have one group.
    #[serde(default = "TokenizerConfig::single_group")]
    pub band_groups: usize,

    /// Centroids per level.
    pub codebook_size: usize,

    /// Lloyd iterations per level.
    pub iterations: usize,

    /// Training vectors are subsampled to at most this many.
    pub max_training_vectors: usize,

    /// Seed for initialisation and subsampling.
    pub seed: u64,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            frames_per_step: 2,
            codebooks: 4,
            band_groups: 1,
            codebook_size: 1024,
            iterations: 20,
            max_training_vectors: 200_000,
            seed: 0,
        }
    }
}

impl TokenizerConfig {
    /// The band-group count of tokenizers saved before band groups existed.
    fn single_group() -> usize {
        1
    }

    /// Residual levels within each band group.
    pub fn levels(&self) -> usize {
        self.codebooks / self.band_groups
    }

    /// Position of the code of `group` at residual `level` within a step.
    pub fn code_index(&self, level: usize, group: usize) -> usize {
        level * self.band_groups + group
    }
}

/// Progress of one Lloyd iteration, for logging.
#[derive(Clone, Copy, Debug)]
pub struct KMeansProgress {
    /// Residual level being trained.
    pub level: usize,

    /// Iteration within the level.
    pub iteration: usize,

    /// Mean squared quantisation error per dimension.
    pub mean_squared_error: f64,

    /// Centroids that received no points and were re-seeded.
    pub reseeded: usize,
}

/// A trained residual quantiser.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioTokenizer {
    /// Hyperparameters it was trained with.
    pub config: TokenizerConfig,

    /// Mel bands per frame.
    pub mel_bands: usize,

    /// Feature whitening applied before quantisation.
    pub normalization: Normalization,

    /// Centroids of each codebook, `[codebook_size, group_dim]` row-major,
    /// in code order (see [`TokenizerConfig::code_index`]).
    pub codebooks: Vec<Vec<f32>>,
}

/// Centroids resident on the GPU, ready for nearest-neighbour search.
struct CentroidSearch {
    /// `[centroids, dim]`.
    centroids: Tensor,

    /// `‖c‖²` per centroid.
    norms: Tensor,

    /// Vector dimension.
    dim: usize,
}

impl CentroidSearch {
    /// Uploads `centroids` (`[count, dim]`).
    fn new(gpu: &Gpu, centroids: &[f32], dim: usize) -> GpuResult<Self> {
        let count = centroids.len() / dim;
        let norms: Vec<f32> = centroids.chunks_exact(dim).map(|centroid| centroid.iter().map(|value| value * value).sum()).collect();

        Ok(Self { centroids: gpu.upload(Shape::new(count, dim), centroids)?, norms: gpu.upload(Shape::vector(count), &norms)?, dim })
    }

    /// Index of the nearest centroid for every `dim`-vector in `points`.
    fn assign(&self, gpu: &Gpu, points: &[f32]) -> GpuResult<Vec<u32>> {
        let centroid_count = self.norms.len();
        let mut assignments = Vec::with_capacity(points.len() / self.dim);

        for chunk in points.chunks(SEARCH_CHUNK * self.dim) {
            let rows = chunk.len() / self.dim;
            let chunk_points = gpu.upload(Shape::new(rows, self.dim), chunk)?;
            let mut dots = gpu.zeros(Shape::new(rows, centroid_count))?;
            let mut nearest = gpu.upload_tokens(&vec![0; rows])?;

            gpu.matmul(&MatmulSpec::new(rows, centroid_count, self.dim).right_transposed(), &chunk_points, &self.centroids, &mut dots)?;
            gpu.nearest_centroid(&dots, &self.norms, &mut nearest)?;
            assignments.extend(gpu.download_tokens(&nearest)?);
        }

        Ok(assignments)
    }
}

/// Copies `count` randomly chosen rows of `points` (`dim` wide).
fn random_rows(points: &[f32], dim: usize, count: usize, random: &mut Random) -> Vec<f32> {
    let available = points.len() / dim;
    let mut rows = Vec::with_capacity(count * dim);
    for _ in 0..count {
        let row = random.below(available);
        rows.extend_from_slice(&points[row * dim..(row + 1) * dim]);
    }
    rows
}

/// One Lloyd update: each centroid moves to the mean of its points; empty
/// centroids are re-seeded from random points. Returns how many were re-seeded.
fn update_centroids(centroids: &mut [f32], points: &[f32], assignments: &[u32], dim: usize, random: &mut Random) -> usize {
    let centroid_count = centroids.len() / dim;
    let mut sums = vec![0.0f64; centroids.len()];
    let mut members = vec![0usize; centroid_count];

    for (point, &centroid) in points.chunks_exact(dim).zip(assignments) {
        let centroid = centroid as usize;
        members[centroid] += 1;
        for (sum, &value) in sums[centroid * dim..(centroid + 1) * dim].iter_mut().zip(point) {
            *sum += value as f64;
        }
    }

    let mut reseeded = 0;
    for (centroid, &member_count) in members.iter().enumerate() {
        let span = centroid * dim..(centroid + 1) * dim;
        if member_count == 0 {
            centroids[span].copy_from_slice(&random_rows(points, dim, 1, random));
            reseeded += 1;
            continue;
        }

        let count = member_count as f64;
        for (value, sum) in centroids[span.clone()].iter_mut().zip(&sums[span]) {
            *value = (sum / count) as f32;
        }
    }

    reseeded
}

/// Subtracts each point's assigned centroid, leaving the residual for the next level.
fn subtract_assigned(points: &mut [f32], centroids: &[f32], assignments: &[u32], dim: usize) {
    for (point, &centroid) in points.chunks_exact_mut(dim).zip(assignments) {
        let centroid = &centroids[centroid as usize * dim..(centroid as usize + 1) * dim];
        for (value, center) in point.iter_mut().zip(centroid) {
            *value -= center;
        }
    }
}

/// Mean squared error per dimension between points and their assigned centroids.
fn quantisation_error(points: &[f32], centroids: &[f32], assignments: &[u32], dim: usize) -> f64 {
    let mut total = 0.0f64;
    for (point, &centroid) in points.chunks_exact(dim).zip(assignments) {
        let center = &centroids[centroid as usize * dim..(centroid as usize + 1) * dim];
        total += point.iter().zip(center).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>();
    }
    total / points.len().max(1) as f64
}

/// How a step vector `[frames_per_step, bands]` is divided into band groups.
#[derive(Clone, Copy, Debug)]
struct BandGroups {
    /// Mel bands per frame.
    bands: usize,

    /// Frames per step.
    frames: usize,

    /// Bands in each group.
    group_bands: usize,
}

impl BandGroups {
    /// Divides `bands` into `count` equal groups.
    fn new(bands: usize, frames: usize, count: usize) -> Self {
        assert!(count > 0 && bands.is_multiple_of(count), "{bands} mel bands cannot be divided into {count} equal groups");
        Self { bands, frames, group_bands: bands / count }
    }

    /// Width of one group's piece of a step: its bands in every frame.
    fn group_dim(&self) -> usize {
        self.frames * self.group_bands
    }

    /// The pieces `[steps, group_dim]` of `group` from step vectors `[steps, frames·bands]`.
    fn extract(&self, steps: &[f32], group: usize) -> Vec<f32> {
        let first_band = group * self.group_bands;
        steps.chunks_exact(self.bands).flat_map(|frame| &frame[first_band..first_band + self.group_bands]).copied().collect()
    }

    /// Writes one group's `piece` back into its place in `step`.
    fn insert(&self, step: &mut [f32], group: usize, piece: &[f32]) {
        let first_band = group * self.group_bands;
        for (frame, values) in step.chunks_exact_mut(self.bands).zip(piece.chunks_exact(self.group_bands)) {
            frame[first_band..first_band + self.group_bands].copy_from_slice(values);
        }
    }
}

/// Fits one codebook to `points` with Lloyd's algorithm, reporting each
/// iteration's error and re-seeded centroids to `progress`.
fn fit_codebook(
    gpu: &Gpu,
    config: &TokenizerConfig,
    points: &[f32],
    dim: usize,
    random: &mut Random,
    mut progress: impl FnMut(usize, f64, usize),
) -> GpuResult<Vec<f32>> {
    let mut centroids = random_rows(points, dim, config.codebook_size, random);

    for iteration in 0..config.iterations {
        let assignments = CentroidSearch::new(gpu, &centroids, dim)?.assign(gpu, points)?;
        let mean_squared_error = quantisation_error(points, &centroids, &assignments, dim);
        let reseeded = update_centroids(&mut centroids, points, &assignments, dim, random);
        progress(iteration, mean_squared_error, reseeded);
    }

    Ok(centroids)
}

impl AudioTokenizer {
    /// Width of one step vector.
    pub fn step_dim(&self) -> usize {
        self.config.frames_per_step * self.mel_bands
    }

    /// Normalised step vectors `[steps, step_dim]` of a `[frames, bands]` log-mel.
    /// Trailing frames that do not fill a whole step are dropped.
    pub fn step_vectors(log_mel: &[f32], mel_bands: usize, frames_per_step: usize, normalization: &Normalization) -> Vec<f32> {
        let usable_frames = (log_mel.len() / mel_bands) / frames_per_step * frames_per_step;
        let mut vectors = Vec::with_capacity(usable_frames * mel_bands);

        for frame in log_mel.chunks_exact(mel_bands).take(usable_frames) {
            for ((value, mean), std) in frame.iter().zip(&normalization.mean).zip(&normalization.std) {
                vectors.push((value - mean) / std);
            }
        }

        vectors
    }

    /// Trains every codebook on `vectors` (already from [`Self::step_vectors`]).
    pub fn train(
        gpu: &Gpu,
        config: TokenizerConfig,
        mel_bands: usize,
        normalization: Normalization,
        mut vectors: Vec<f32>,
        mut progress: impl FnMut(KMeansProgress),
    ) -> GpuResult<Self> {
        let step_dim = config.frames_per_step * mel_bands;
        let groups = BandGroups::new(mel_bands, config.frames_per_step, config.band_groups);
        let mut random = Random::new(config.seed);
        if vectors.len() / step_dim > config.max_training_vectors {
            vectors = random_rows(&vectors, step_dim, config.max_training_vectors, &mut random);
        }

        let mut codebooks = vec![Vec::new(); config.codebooks];
        for group in 0..config.band_groups {
            let mut residual = groups.extract(&vectors, group);

            for level in 0..config.levels() {
                let centroids =
                    fit_codebook(gpu, &config, &residual, groups.group_dim(), &mut random, |iteration, mean_squared_error, reseeded| {
                        progress(KMeansProgress { level: config.code_index(level, group), iteration, mean_squared_error, reseeded })
                    })?;

                let assignments = CentroidSearch::new(gpu, &centroids, groups.group_dim())?.assign(gpu, &residual)?;
                subtract_assigned(&mut residual, &centroids, &assignments, groups.group_dim());
                codebooks[config.code_index(level, group)] = centroids;
            }
        }

        Ok(Self { config, mel_bands, normalization, codebooks })
    }

    /// Codes `[steps, codebooks]` of a `[frames, bands]` log-mel.
    pub fn encode(&self, gpu: &Gpu, log_mel: &[f32]) -> GpuResult<Vec<u32>> {
        let config = &self.config;
        let groups = BandGroups::new(self.mel_bands, config.frames_per_step, config.band_groups);
        let vectors = Self::step_vectors(log_mel, self.mel_bands, config.frames_per_step, &self.normalization);
        let steps = vectors.len() / self.step_dim();
        let mut codes = vec![0u32; steps * config.codebooks];

        for group in 0..config.band_groups {
            let mut residual = groups.extract(&vectors, group);

            for level in 0..config.levels() {
                let index = config.code_index(level, group);
                let centroids = &self.codebooks[index];
                let assignments = CentroidSearch::new(gpu, centroids, groups.group_dim())?.assign(gpu, &residual)?;
                subtract_assigned(&mut residual, centroids, &assignments, groups.group_dim());

                for (step, &code) in assignments.iter().enumerate() {
                    codes[step * config.codebooks + index] = code;
                }
            }
        }

        Ok(codes)
    }

    /// Log-mel `[steps·frames_per_step, bands]` reconstructed from the first
    /// `levels` residual levels of every band group in `codes`.
    pub fn decode(&self, codes: &[u32], levels: usize) -> Vec<f32> {
        let config = &self.config;
        let groups = BandGroups::new(self.mel_bands, config.frames_per_step, config.band_groups);
        let group_dim = groups.group_dim();
        let mut log_mel = vec![0.0f32; codes.len() / config.codebooks * self.step_dim()];
        let mut piece = vec![0.0f32; group_dim];

        for (step_codes, step_values) in codes.chunks_exact(config.codebooks).zip(log_mel.chunks_exact_mut(self.step_dim())) {
            for group in 0..config.band_groups {
                piece.fill(0.0);
                for level in 0..levels.min(config.levels()) {
                    let index = config.code_index(level, group);
                    let code = step_codes[index] as usize;
                    let centroid = &self.codebooks[index][code * group_dim..(code + 1) * group_dim];
                    piece.iter_mut().zip(centroid).for_each(|(value, center)| *value += center);
                }
                groups.insert(step_values, group, &piece);
            }
        }

        for frame in log_mel.chunks_exact_mut(self.mel_bands) {
            for ((value, mean), std) in frame.iter_mut().zip(&self.normalization.mean).zip(&self.normalization.std) {
                *value = *value * std + mean;
            }
        }

        log_mel
    }

    /// Writes the tokenizer as JSON.
    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, serde_json::to_vec(self)?)?;
        Ok(())
    }

    /// Reads a tokenizer written by [`Self::save`].
    pub fn load(path: &Path) -> Result<Self> {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_groups_split_and_rejoin_a_step() {
        let groups = BandGroups::new(4, 2, 2);
        let step: Vec<f32> = (0..8).map(|value| value as f32).collect();

        let low = groups.extract(&step, 0);
        let high = groups.extract(&step, 1);
        assert_eq!(low, vec![0.0, 1.0, 4.0, 5.0]);
        assert_eq!(high, vec![2.0, 3.0, 6.0, 7.0]);

        let mut rebuilt = vec![0.0; 8];
        groups.insert(&mut rebuilt, 0, &low);
        groups.insert(&mut rebuilt, 1, &high);
        assert_eq!(rebuilt, step);
    }

    #[test]
    fn codes_are_ordered_level_by_level() {
        let config = TokenizerConfig { codebooks: 8, band_groups: 4, ..TokenizerConfig::default() };
        assert_eq!(config.levels(), 2);
        assert_eq!((0..4).map(|group| config.code_index(0, group)).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        assert_eq!(config.code_index(1, 2), 6);
    }
}

#[cfg(test)]
mod edge_tests {
    use super::*;

    #[test]
    fn defaults_match_the_residual_quantiser() {
        let config = TokenizerConfig::default();

        assert_eq!(config.frames_per_step, 2);
        assert_eq!(config.codebooks, 4);
        assert_eq!(config.band_groups, 1);
        assert_eq!(config.codebook_size, 1024);
        assert_eq!(config.iterations, 20);
        assert_eq!(config.max_training_vectors, 200_000);
        assert_eq!(config.seed, 0);
    }

    #[test]
    fn step_vectors_whiten_and_drop_partial_steps() {
        let normalization = Normalization { mean: vec![1.0, 2.0], std: vec![2.0, 1.0] };
        let log_mel = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let vectors = AudioTokenizer::step_vectors(&log_mel, 2, 2, &normalization);

        assert_eq!(vectors, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn step_dim_multiplies_frames_by_mel_bands() {
        let tokenizer = AudioTokenizer {
            config: TokenizerConfig { frames_per_step: 2, ..TokenizerConfig::default() },
            mel_bands: 128,
            normalization: Normalization { mean: Vec::new(), std: Vec::new() },
            codebooks: Vec::new(),
        };

        assert_eq!(tokenizer.step_dim(), 256);
    }
}
