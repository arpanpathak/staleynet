//! Vocoder checkpoints.
//!
//! A checkpoint directory holds `vocoder.json` (the configuration, the input
//! standardization, the tensor manifest and the training step), `tensors.bin`
//! (every parameter, little-endian `f32`, in visitor order) and, for a
//! resumable checkpoint, `optimizer.bin` (each parameter's first AdamW moment
//! followed by its second, in the same order).

use super::config::VocoderConfig;
use super::model::Vocoder;
use crate::checkpoint::{TensorEntry, read_f32_file, try_for_each_parameter, upload_next, write_atomically};
use crate::optimizer::AdamW;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use staleynet_cuda::Gpu;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Name of the metadata file inside a vocoder checkpoint directory.
pub const METADATA_FILE: &str = "vocoder.json";

/// Name of the parameter file inside a vocoder checkpoint directory.
const TENSOR_FILE: &str = "tensors.bin";

/// Name of the optimizer-state file inside a resumable checkpoint directory.
const OPTIMIZER_FILE: &str = "optimizer.bin";

/// Contents of `vocoder.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Metadata {
    /// Sizes and settings.
    config: VocoderConfig,

    /// Input standardization: mean per band.
    mel_mean: Vec<f32>,

    /// Input standardization: standard deviation per band.
    mel_std: Vec<f32>,

    /// Stored parameters, in file order.
    tensors: Vec<TensorEntry>,

    /// Training steps taken when saved.
    step: usize,
}

/// A vocoder read back from a checkpoint.
pub struct Restored {
    /// The vocoder with its saved weights.
    pub vocoder: Vocoder,

    /// Training steps taken when it was saved.
    pub step: usize,
}

/// Whether `directory` holds a vocoder checkpoint.
pub fn exists(directory: &Path) -> bool {
    directory.join(METADATA_FILE).is_file()
}

/// Saves the vocoder's weights and standardization into `directory`, with
/// the optimizer's moments when `optimizer` is given. Each file is written
/// under a temporary name and then renamed.
pub fn save(gpu: &Gpu, directory: &Path, vocoder: &mut Vocoder, optimizer: Option<&AdamW>, step: usize) -> Result<()> {
    fs::create_dir_all(directory)?;

    let mut tensors = Vec::new();
    let mut values: Vec<Vec<f32>> = Vec::new();
    try_for_each_parameter(&mut vocoder.network, |parameter| {
        let host = gpu.download(&parameter.value)?;
        tensors.push(TensorEntry { name: parameter.name.clone(), len: host.len() });
        values.push(host);
        Ok(())
    })?;

    write_atomically(&directory.join(TENSOR_FILE), |writer| write_values(writer, &values))?;

    if let Some(optimizer) = optimizer {
        save_moments(gpu, directory, optimizer)?;
    }

    let metadata =
        Metadata { config: vocoder.config.clone(), mel_mean: vocoder.mel_mean.clone(), mel_std: vocoder.mel_std.clone(), tensors, step };
    let text = serde_json::to_string_pretty(&metadata)?;

    write_atomically(&directory.join(METADATA_FILE), |writer| writer.write_all(text.as_bytes()))
}

/// Loads the vocoder saved in `directory` by [`save`].
pub fn load(gpu: &Gpu, directory: &Path) -> Result<Restored> {
    let path = directory.join(METADATA_FILE);
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let metadata: Metadata = serde_json::from_str(&text)?;

    let values = read_f32_file(&directory.join(TENSOR_FILE))?;
    let mut vocoder = Vocoder::new(gpu, metadata.config, metadata.mel_mean, metadata.mel_std, 0)?;

    let mut remaining = values.as_slice();
    let mut entries = metadata.tensors.iter();
    try_for_each_parameter(&mut vocoder.network, |parameter| {
        let entry = entries
            .next()
            .context("the checkpoint has fewer tensors than the network")?;
        let matches = entry.name == parameter.name && entry.len == parameter.value.len();
        ensure!(matches, "checkpoint tensor {} ({}) does not match {}", entry.name, entry.len, parameter.name);

        upload_next(gpu, &mut remaining, &mut parameter.value)
    })?;

    ensure!(remaining.is_empty(), "{} has more values than the network", directory.display());
    Ok(Restored { vocoder, step: metadata.step })
}

/// Restores the optimizer's moments saved in `directory` by [`save`].
pub fn load_moments(gpu: &Gpu, directory: &Path, optimizer: &mut AdamW) -> Result<()> {
    let path = directory.join(OPTIMIZER_FILE);
    let values = read_f32_file(&path)?;
    let mut remaining = values.as_slice();

    for moment in &mut optimizer.moments {
        upload_next(gpu, &mut remaining, &mut moment.first)?;
        upload_next(gpu, &mut remaining, &mut moment.second)?;
    }

    ensure!(remaining.is_empty(), "{} has more values than the optimizer", path.display());
    Ok(())
}

/// Writes each parameter's first moment followed by its second.
fn save_moments(gpu: &Gpu, directory: &Path, optimizer: &AdamW) -> Result<()> {
    let mut values = Vec::with_capacity(2 * optimizer.moments.len());
    for moment in &optimizer.moments {
        values.push(gpu.download(&moment.first)?);
        values.push(gpu.download(&moment.second)?);
    }

    write_atomically(&directory.join(OPTIMIZER_FILE), |writer| write_values(writer, &values))
}

/// Writes every value of `tensors`, in order, as little-endian `f32`.
fn write_values(writer: &mut BufWriter<fs::File>, tensors: &[Vec<f32>]) -> std::io::Result<()> {
    tensors
        .iter()
        .flatten()
        .try_for_each(|value| writer.write_all(&value.to_le_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifests_written_before_the_move_still_parse() {
        let old_manifest = r#"[["input.weight", 6], ["input.bias", 2]]"#;
        let entries: Vec<TensorEntry> = serde_json::from_str(old_manifest).unwrap();

        assert_eq!(entries[0], TensorEntry { name: "input.weight".into(), len: 6 });
        assert_eq!(entries[1], TensorEntry { name: "input.bias".into(), len: 2 });
    }
}
