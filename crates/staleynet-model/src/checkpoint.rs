//! Checkpoints: `model.json` (configuration, tensor manifest, training step)
//! and `tensors.bin` (little-endian `f32` values, parameters then, optionally,
//! AdamW moments, all in visitor order).

use crate::config::ModelConfig;
use crate::gpt::Gpt;
use crate::optimizer::AdamW;
use crate::parameter::Parameters;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use staleynet_cuda::Gpu;
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
    let mut result = Ok(());

    model.visit_parameters(&mut |parameter| match gpu.download(&parameter.value) {
        Ok(host) => {
            tensors.push(TensorEntry { name: parameter.name.clone(), len: host.len() });
            values.push(host);
        }
        Err(error) => result = Err(error),
    });
    result?;

    if let Some(optimizer) = optimizer {
        for moment in &optimizer.moments {
            values.push(gpu.download(&moment.first)?);
        }
        for moment in &optimizer.moments {
            values.push(gpu.download(&moment.second)?);
        }
    }

    let staging = directory.join(format!("{TENSOR_FILE}.partial"));
    {
        let mut writer = BufWriter::new(fs::File::create(&staging)?);
        for value in values.iter().flatten() {
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.flush()?;
    }
    fs::rename(&staging, directory.join(TENSOR_FILE))?;

    let metadata = CheckpointMetadata { model: model.config.clone(), tensors, step, has_optimizer_state: optimizer.is_some(), training };
    let staging = directory.join(format!("{METADATA_FILE}.partial"));
    fs::write(&staging, serde_json::to_string_pretty(&metadata)?)?;
    fs::rename(&staging, directory.join(METADATA_FILE))?;
    Ok(())
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
    let bytes = fs::read(directory.join(TENSOR_FILE)).with_context(|| format!("read {}", directory.join(TENSOR_FILE).display()))?;
    let values: Vec<f32> =
        bytes.chunks_exact(BYTES_PER_VALUE).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();

    let mut stored = std::collections::HashMap::new();
    let mut cursor = 0;
    for entry in &metadata.tensors {
        stored.insert(entry.name.clone(), &values[cursor..cursor + entry.len]);
        cursor += entry.len;
    }

    let mut transfer = WeightTransfer::default();
    let mut problem: Option<anyhow::Error> = None;
    model.visit_parameters(&mut |parameter| {
        if problem.is_some() {
            return;
        }
        let Some(&source) = stored.get(&parameter.name) else {
            transfer.fresh.push(parameter.name.clone());
            return;
        };

        let row_width = parameter.value.shape().cols;
        let grows_by_rows = source.len() < parameter.value.len() && source.len() % row_width == 0;
        if source.len() != parameter.value.len() && !grows_by_rows {
            problem = Some(anyhow::anyhow!(
                "{} has {} values in the checkpoint but {} in the model",
                parameter.name,
                source.len(),
                parameter.value.len()
            ));
            return;
        }

        let result = gpu.download(&parameter.value).and_then(|mut current| {
            current[..source.len()].copy_from_slice(source);
            gpu.copy_from_host(&mut parameter.value, &current)
        });
        if let Err(error) = result {
            problem = Some(error.into());
            return;
        }

        let list = if grows_by_rows { &mut transfer.grown } else { &mut transfer.copied };
        list.push(parameter.name.clone());
    });

    match problem {
        Some(error) => Err(error),
        None => Ok(transfer),
    }
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
    let bytes = fs::read(directory.join(TENSOR_FILE)).with_context(|| format!("read {}", directory.join(TENSOR_FILE).display()))?;
    let values: Vec<f32> =
        bytes.chunks_exact(BYTES_PER_VALUE).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();

    let mut model = Gpt::new(gpu, metadata.model.clone(), 0)?;
    let mut cursor = 0;
    let mut problem: Option<anyhow::Error> = None;

    let mut entries = metadata.tensors.iter();
    model.visit_parameters(&mut |parameter| {
        if problem.is_some() {
            return;
        }
        let Some(entry) = entries.next() else {
            problem = Some(anyhow::anyhow!("checkpoint has fewer tensors than the model"));
            return;
        };
        if entry.name != parameter.name || entry.len != parameter.value.len() {
            problem = Some(anyhow::anyhow!("checkpoint tensor {} ({}) does not match parameter {}", entry.name, entry.len, parameter.name));
            return;
        }

        let slice = &values[cursor..cursor + entry.len];
        cursor += entry.len;
        if let Err(error) = gpu.copy_from_host(&mut parameter.value, slice) {
            problem = Some(error.into());
        }
    });
    if let Some(error) = problem {
        return Err(error);
    }

    let Some(optimizer) = optimizer else { return Ok((model, metadata)) };
    ensure!(metadata.has_optimizer_state, "checkpoint {} has no optimiser state", directory.display());

    for moment in optimizer.moments.iter_mut() {
        let len = moment.first.len();
        gpu.copy_from_host(&mut moment.first, &values[cursor..cursor + len])?;
        cursor += len;
    }
    for moment in optimizer.moments.iter_mut() {
        let len = moment.second.len();
        gpu.copy_from_host(&mut moment.second, &values[cursor..cursor + len])?;
        cursor += len;
    }
    optimizer.steps_taken = metadata.step;

    Ok((model, metadata))
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
