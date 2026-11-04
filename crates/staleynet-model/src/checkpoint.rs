//! Checkpoints: `model.json` (configuration, tensor manifest, training step)
//! and `tensors.bin` (little-endian `f32` values, parameters then, optionally,
//! AdamW moments, all in visitor order).

use crate::config::ModelConfig;
use crate::gpt::Gpt;
use crate::optimizer::AdamW;
use crate::parameter::{Parameter, Parameters};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use staleynet_cuda::{Gpu, Tensor};
use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Name of the metadata file inside a checkpoint directory.
const METADATA_FILE: &str = "model.json";

/// Name of the tensor file inside a checkpoint directory.
const TENSOR_FILE: &str = "tensors.bin";

/// Bytes per stored value.
const BYTES_PER_VALUE: usize = std::mem::size_of::<f32>();

/// Name and size of one stored tensor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorEntry {
    /// Parameter name.
    pub name: String,

    /// Number of `f32` values.
    pub len: usize,
}

/// Contents of `model.json`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Architecture.
    pub model: ModelConfig,

    /// Stored parameters, in file order.
    pub tensors: Vec<TensorEntry>,

    /// Optimiser steps taken when saved.
    pub step: usize,

    /// Whether AdamW moments follow the parameters in `tensors.bin`.
    pub has_optimizer_state: bool,

    /// Free-form training configuration, for provenance.
    pub training: serde_json::Value,
}

/// Downloads every parameter (and optionally the optimiser moments) and writes a checkpoint.
/// `step` is recorded explicitly because weight-only checkpoints carry no optimiser.
///
/// Each file is written under a temporary name and then renamed, so a reader
/// never sees a half-written file.
pub fn save(
    gpu: &Gpu,
    directory: &Path,
    model: &mut Gpt,
    optimizer: Option<&AdamW>,
    step: usize,
    training: serde_json::Value,
) -> Result<()> {
    fs::create_dir_all(directory)?;

    let mut tensors = Vec::new();
    let mut values: Vec<Vec<f32>> = Vec::new();
    try_for_each_parameter(model, |parameter| {
        let host = gpu.download(&parameter.value)?;
        tensors.push(TensorEntry { name: parameter.name.clone(), len: host.len() });
        values.push(host);
        Ok(())
    })?;

    if let Some(optimizer) = optimizer {
        values.extend(download_moments(gpu, optimizer)?);
    }

    write_atomically(&directory.join(TENSOR_FILE), |writer| {
        values
            .iter()
            .flatten()
            .try_for_each(|value| writer.write_all(&value.to_le_bytes()))
    })?;

    let metadata = CheckpointMetadata { model: model.config.clone(), tensors, step, has_optimizer_state: optimizer.is_some(), training };
    let text = serde_json::to_string_pretty(&metadata)?;
    write_atomically(&directory.join(METADATA_FILE), |writer| writer.write_all(text.as_bytes()))
}

/// What [`load_weights`] did.
#[derive(Clone, Debug, Default)]
pub struct WeightTransfer {
    /// Parameters copied whole.
    pub copied: Vec<String>,

    /// Parameters that are larger in the model than in the checkpoint: their
    /// leading rows were copied and the new rows keep their fresh values.
    pub grown: Vec<String>,

    /// Parameters the checkpoint does not have; they keep their fresh values.
    pub fresh: Vec<String>,
}

/// Copies the weights of the checkpoint in `directory` into `model`, matching
/// tensors by name. A tensor stored smaller than the parameter is copied into
/// its leading values, which is how a position table grows when a lyrics field
/// is added (the new rows come last; see `vocabulary`). Any other mismatch is
/// an error. The optimiser is not touched: this starts a new run from old
/// weights.
pub fn load_weights(gpu: &Gpu, directory: &Path, model: &mut Gpt) -> Result<WeightTransfer> {
    let metadata = read_metadata(directory)?;
    let values = read_values(directory)?;
    let stored = stored_tensors(&metadata, &values);

    let mut transfer = WeightTransfer::default();
    try_for_each_parameter(model, |parameter| {
        let Some(&source) = stored.get(&parameter.name) else {
            transfer.fresh.push(parameter.name.clone());
            return Ok(());
        };

        let grows_by_rows = copy_leading_values(gpu, source, parameter)?;

        let list = if grows_by_rows { &mut transfer.grown } else { &mut transfer.copied };
        list.push(parameter.name.clone());
        Ok(())
    })?;

    Ok(transfer)
}

/// Copies `source` into the leading values of `parameter`. Returns whether the
/// parameter is larger than `source` by whole rows (a grown table); any other
/// size mismatch is an error.
fn copy_leading_values(gpu: &Gpu, source: &[f32], parameter: &mut Parameter) -> Result<bool> {
    let row_width = parameter.value.shape().cols;
    let grows_by_rows = source.len() < parameter.value.len() && source.len().is_multiple_of(row_width);
    ensure!(
        source.len() == parameter.value.len() || grows_by_rows,
        "{} has {} values in the checkpoint but {} in the model",
        parameter.name,
        source.len(),
        parameter.value.len()
    );

    let mut current = gpu.download(&parameter.value)?;
    current[..source.len()].copy_from_slice(source);
    gpu.copy_from_host(&mut parameter.value, &current)?;

    Ok(grows_by_rows)
}

/// Reads a checkpoint's metadata only.
pub fn read_metadata(directory: &Path) -> Result<CheckpointMetadata> {
    let path = directory.join(METADATA_FILE);
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    Ok(serde_json::from_str(&text)?)
}

/// Builds a model from a checkpoint and, if `optimizer` is given and the
/// checkpoint has optimiser state, restores its moments and step count.
pub fn load(gpu: &Gpu, directory: &Path, optimizer: Option<&mut AdamW>) -> Result<(Gpt, CheckpointMetadata)> {
    let metadata = read_metadata(directory)?;
    let values = read_values(directory)?;
    let mut model = Gpt::new(gpu, metadata.model.clone(), 0)?;

    let mut remaining = values.as_slice();
    let mut entries = metadata.tensors.iter();
    try_for_each_parameter(&mut model, |parameter| {
        let entry = entries
            .next()
            .context("checkpoint has fewer tensors than the model")?;
        ensure!(
            entry.name == parameter.name && entry.len == parameter.value.len(),
            "checkpoint tensor {} ({}) does not match parameter {}",
            entry.name,
            entry.len,
            parameter.name
        );

        upload_next(gpu, &mut remaining, &mut parameter.value)
    })?;

    let Some(optimizer) = optimizer else { return Ok((model, metadata)) };
    ensure!(metadata.has_optimizer_state, "checkpoint {} has no optimiser state", directory.display());

    restore_moments(gpu, optimizer, remaining)?;
    optimizer.steps_taken = metadata.step;
    Ok((model, metadata))
}

/// Calls `action` on every parameter of `model` in visitor order and stops at
/// the first error, which it returns. Shared with the vocoder's checkpoints.
pub(crate) fn try_for_each_parameter(model: &mut impl Parameters, mut action: impl FnMut(&mut Parameter) -> Result<()>) -> Result<()> {
    let mut outcome = Ok(());
    model.visit_parameters(&mut |parameter| {
        if outcome.is_ok() {
            outcome = action(parameter);
        }
    });

    outcome
}

/// All values of `tensors.bin`, decoded from little-endian `f32`.
fn read_values(directory: &Path) -> Result<Vec<f32>> {
    read_f32_file(&directory.join(TENSOR_FILE))
}

/// Every value of a file of little-endian `f32`. Shared with the vocoder's checkpoints.
pub(crate) fn read_f32_file(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;

    Ok(bytes
        .chunks_exact(BYTES_PER_VALUE)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// The stored parameters by name, each a slice of `values` in file order.
fn stored_tensors<'a>(metadata: &CheckpointMetadata, values: &'a [f32]) -> HashMap<String, &'a [f32]> {
    let mut stored = HashMap::new();
    let mut remaining = values;

    for entry in &metadata.tensors {
        let (tensor, rest) = remaining.split_at(entry.len);
        stored.insert(entry.name.clone(), tensor);
        remaining = rest;
    }
    stored
}

/// Uploads the next `tensor.len()` values of `remaining` into `tensor` and
/// advances `remaining` past them. Shared with the vocoder's checkpoints.
pub(crate) fn upload_next(gpu: &Gpu, remaining: &mut &[f32], tensor: &mut Tensor) -> Result<()> {
    ensure!(remaining.len() >= tensor.len(), "tensors.bin ends before every tensor was read");

    let (values, rest) = remaining.split_at(tensor.len());
    gpu.copy_from_host(tensor, values)?;
    *remaining = rest;

    Ok(())
}

/// The AdamW moments in file order: every first moment, then every second moment.
fn download_moments(gpu: &Gpu, optimizer: &AdamW) -> Result<Vec<Vec<f32>>> {
    let firsts = optimizer.moments.iter().map(|moment| &moment.first);
    let seconds = optimizer.moments.iter().map(|moment| &moment.second);

    Ok(firsts
        .chain(seconds)
        .map(|tensor| gpu.download(tensor))
        .collect::<Result<_, _>>()?)
}

/// Uploads the AdamW moments stored after the parameters, in the order that
/// [`download_moments`] wrote them.
fn restore_moments(gpu: &Gpu, optimizer: &mut AdamW, mut remaining: &[f32]) -> Result<()> {
    for moment in optimizer.moments.iter_mut() {
        upload_next(gpu, &mut remaining, &mut moment.first)?;
    }
    for moment in optimizer.moments.iter_mut() {
        upload_next(gpu, &mut remaining, &mut moment.second)?;
    }
    Ok(())
}

/// Lets `write` fill a temporary file next to `path` and then renames it to
/// `path`, so that the file at `path` is always either the old one or the
/// complete new one. Writing through a buffer keeps large tensors streaming
/// to disk instead of being copied into memory first. Shared with the
/// vocoder's checkpoints.
pub(crate) fn write_atomically(path: &Path, write: impl FnOnce(&mut BufWriter<fs::File>) -> std::io::Result<()>) -> Result<()> {
    let mut staging_name = path.as_os_str().to_owned();
    staging_name.push(".partial");
    let staging = Path::new(&staging_name);

    let mut writer = BufWriter::new(fs::File::create(staging)?);
    write(&mut writer)?;
    writer.flush()?;
    drop(writer);

    fs::rename(staging, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_entry_round_trips_json() {
        let entry = TensorEntry { name: "blocks.0.attention.qkv.weight".into(), len: 196_608 };
        let encoded = serde_json::to_string(&entry).unwrap();
        let decoded: TensorEntry = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded, entry);
    }

    #[test]
    fn checkpoint_metadata_round_trips_json() {
        let metadata = CheckpointMetadata {
            model: ModelConfig::default(),
            tensors: vec![TensorEntry { name: "token_embedding".into(), len: 1_114_880 }],
            step: 7,
            has_optimizer_state: false,
            training: serde_json::json!({ "seed": 1, "batch_size": 4 }),
        };
        let encoded = serde_json::to_string(&metadata).unwrap();
        let decoded: CheckpointMetadata = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded.model, metadata.model);
        assert_eq!(decoded.tensors, metadata.tensors);
        assert_eq!(decoded.step, metadata.step);
        assert_eq!(decoded.has_optimizer_state, metadata.has_optimizer_state);
        assert_eq!(decoded.training, metadata.training);
    }

    #[test]
    fn older_metadata_defaults_the_lyrics_field_to_zero() {
        let encoded = r#"{
            "model": {
                "width": 256,
                "heads": 8,
                "layers": 6,
                "feed_forward_multiplier": 4,
                "context_length": 768,
                "prompt_bytes": 126,
                "codebooks": 4,
                "codebook_size": 1024
            },
            "tensors": [],
            "step": 0,
            "has_optimizer_state": false,
            "training": {}
        }"#;

        let decoded: CheckpointMetadata = serde_json::from_str(encoded).unwrap();

        assert_eq!(decoded.model.lyrics_bytes, 0);
        assert_eq!(decoded.step, 0);
    }
}
