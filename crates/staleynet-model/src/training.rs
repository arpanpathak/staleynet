//! Teacher-forced training: one forward pass over whole sequences, the
//! per-codebook cross-entropy, and the backward pass.

use crate::config::ModelConfig;
use crate::gpt::Gpt;
use crate::layers::SequenceShape;
use crate::layers::block::{BlockActivations, BlockGradients};
use crate::parameter::Parameters;
use crate::vocabulary::Vocabulary;
use staleynet_cuda::{CrossEntropyHead, EMPTY_TOKEN, Gpu, GpuResult, Shape, Tensor, TokenBuffer, TokenIndex};

/// One batch of sequences on the host.
///
/// Position `p` of a sequence holds `codebooks` token slots in `tokens`, and
/// `targets` holds, per codebook, the code the model must predict *at* `p`
/// (the content of position `p + 1`), or [`EMPTY_TOKEN`] where there is none.
pub struct TrainingBatch {
    /// Batch size and sequence length.
    pub shape: SequenceShape,

    /// Input token slots, `[batch·time, codebooks]`.
    pub tokens: Vec<u32>,

    /// Target codes local to each codebook, `[batch·time, codebooks]`.
    pub targets: Vec<u32>,
}

impl TrainingBatch {
    /// An all-empty batch, to be filled by a sampler and reused every step.
    pub fn empty(shape: SequenceShape, codebooks: usize) -> Self {
        let slots = shape.rows() * codebooks;
        Self { shape, tokens: vec![EMPTY_TOKEN; slots], targets: vec![EMPTY_TOKEN; slots] }
    }
}

/// Losses of one forward pass.
#[derive(Clone, Debug, PartialEq)]
pub struct LossReport {
    /// Mean of the per-codebook mean cross-entropies, in nats.
    pub mean: f32,

    /// Mean cross-entropy of each codebook.
    pub per_codebook: Vec<f32>,

    /// Number of (position, codebook) targets that contributed.
    pub target_count: usize,
}

/// Every device buffer one training step touches, allocated once for a fixed
/// [`SequenceShape`] and reused for every step.
pub struct TrainingWorkspace {
    /// Shape this workspace was built for.
    shape: SequenceShape,

    /// Device copy of [`TrainingBatch::tokens`].
    tokens: TokenBuffer,

    /// Device copy of [`TrainingBatch::targets`].
    targets: TokenBuffer,

    /// Position-table rows of `0..time`, repeated for every sequence.
    positions: TokenBuffer,

    /// Inverted index of `positions`; constant, so built once.
    position_index: TokenIndex,

    /// Residual streams: entry `l` enters block `l`, the last leaves the stack.
    residuals: Vec<Tensor>,

    /// Saved activations of every block.
    blocks: Vec<BlockActivations>,

    /// Backward scratch shared by all blocks.
    block_gradients: BlockGradients,

    /// Output of the final norm.
    final_normed: Tensor,

    /// `1/rms` of the final norm.
    final_inv_rms: Tensor,

    /// Logits `[batch·time, codebooks·codebook_size]`.
    logits: Tensor,

    /// Gradient of the loss with respect to `logits`.
    logits_grad: Tensor,

    /// Per-row loss of the head currently being evaluated.
    row_losses: Tensor,

    /// Gradient with respect to `final_normed`.
    final_normed_grad: Tensor,

    /// Gradient with respect to the residual stream entering the current block.
    residual_grad: Tensor,

    /// Gradient with respect to the residual stream leaving the current block.
    next_residual_grad: Tensor,
}

impl TrainingWorkspace {
    /// Allocates every buffer for `shape`.
    pub fn new(gpu: &Gpu, config: &ModelConfig, shape: SequenceShape) -> GpuResult<Self> {
        assert!(shape.time <= config.sequence_capacity(), "sequence longer than the context");

        let rows = shape.rows();
        let stream = Shape::new(rows, config.width);
        let logits = Shape::new(rows, config.logits_width());
        let empty_slots = vec![EMPTY_TOKEN; rows * config.codebooks];

        let positions: Vec<u32> = Vocabulary::new(config).position_ids(0, shape.time).repeat(shape.batch);

        let mut residuals = Vec::with_capacity(config.layers + 1);
        let mut blocks = Vec::with_capacity(config.layers);
        for _ in 0..config.layers {
            residuals.push(gpu.zeros(stream)?);
            blocks.push(BlockActivations::new(gpu, config, shape)?);
        }
        residuals.push(gpu.zeros(stream)?);

        Ok(Self {
            shape,
            tokens: gpu.upload_tokens(&empty_slots)?,
            targets: gpu.upload_tokens(&empty_slots)?,
            position_index: TokenIndex::build(gpu, &positions, 1, config.sequence_capacity())?,
            positions: gpu.upload_tokens(&positions)?,
            residuals,
            blocks,
            block_gradients: BlockGradients::new(gpu, config, shape)?,
            final_normed: gpu.zeros(stream)?,
            final_inv_rms: gpu.zeros(Shape::vector(rows))?,
            logits: gpu.zeros(logits)?,
            logits_grad: gpu.zeros(logits)?,
            row_losses: gpu.zeros(Shape::vector(rows))?,
            final_normed_grad: gpu.zeros(stream)?,
            residual_grad: gpu.zeros(stream)?,
            next_residual_grad: gpu.zeros(stream)?,
        })
    }
}

impl TrainingWorkspace {
    /// Logits of the last forward pass, `[batch·time, codebooks·codebook_size]`.
    pub fn logits(&self) -> &Tensor {
        &self.logits
    }
}

impl Gpt {
    /// Forward pass and loss. Leaves activations and `∂loss/∂logits` in
    /// `workspace`, ready for [`Gpt::backward`].
    pub fn forward_loss(&self, gpu: &Gpu, workspace: &mut TrainingWorkspace, batch: &TrainingBatch) -> GpuResult<LossReport> {
        let shape = workspace.shape;
        assert_eq!(batch.shape, shape, "batch shape differs from workspace shape");

        gpu.copy_tokens_from_host(&mut workspace.tokens, &batch.tokens)?;
        gpu.copy_tokens_from_host(&mut workspace.targets, &batch.targets)?;
        self.embed(gpu, &workspace.tokens, &workspace.positions, &mut workspace.residuals[0])?;

        for (index, block) in self.blocks.iter().enumerate() {
            let (inputs, outputs) = workspace.residuals.split_at_mut(index + 1);
            block.forward(gpu, &inputs[index], shape, &mut workspace.blocks[index], &mut outputs[0])?;
        }

        let last_residual = &workspace.residuals[self.config.layers];
        self.final_norm.forward(gpu, last_residual, &mut workspace.final_normed, &mut workspace.final_inv_rms)?;
        self.head.forward(gpu, &workspace.final_normed, &mut workspace.logits)?;

        self.codebook_losses(gpu, workspace, batch)
    }

    /// Cross-entropy of every codebook head; also writes the logits gradient.
    fn codebook_losses(&self, gpu: &Gpu, workspace: &mut TrainingWorkspace, batch: &TrainingBatch) -> GpuResult<LossReport> {
        let codebooks = self.config.codebooks;
        let mut per_codebook = Vec::with_capacity(codebooks);
        let mut target_count = 0;

        for codebook in 0..codebooks {
            let valid_targets = batch.targets.iter().skip(codebook).step_by(codebooks).filter(|&&target| target != EMPTY_TOKEN).count();
            let grad_scale = if valid_targets == 0 { 0.0 } else { 1.0 / (valid_targets * codebooks) as f32 };
            let head = CrossEntropyHead {
                column_offset: codebook * self.config.codebook_size,
                classes: self.config.codebook_size,
                target_stride: codebooks,
                target_offset: codebook,
                grad_scale,
            };

            gpu.cross_entropy(&workspace.logits, &workspace.targets, &head, &mut workspace.row_losses, &mut workspace.logits_grad)?;

            let loss_sum: f32 = gpu.download(&workspace.row_losses)?.iter().sum();
            per_codebook.push(loss_sum / valid_targets.max(1) as f32);
            target_count += valid_targets;
        }

        let mean = per_codebook.iter().sum::<f32>() / codebooks as f32;
        Ok(LossReport { mean, per_codebook, target_count })
    }

    /// Backward pass for the batch last given to [`Gpt::forward_loss`];
    /// accumulates into every parameter's gradient.
    pub fn backward(&mut self, gpu: &Gpu, workspace: &mut TrainingWorkspace, batch: &TrainingBatch) -> GpuResult {
        let shape = workspace.shape;
        let last_residual = &workspace.residuals[self.config.layers];

        self.head.backward(gpu, &workspace.final_normed, &workspace.logits_grad, &mut workspace.final_normed_grad)?;
        self.final_norm.backward(
            gpu,
            &workspace.final_normed_grad,
            last_residual,
            &workspace.final_inv_rms,
            &mut workspace.next_residual_grad,
        )?;

        for (index, block) in self.blocks.iter_mut().enumerate().rev() {
            block.backward(
                gpu,
                &workspace.residuals[index],
                shape,
                &workspace.blocks[index],
                &workspace.next_residual_grad,
                &mut workspace.block_gradients,
                &mut workspace.residual_grad,
            )?;
            std::mem::swap(&mut workspace.residual_grad, &mut workspace.next_residual_grad);
        }

        let embedding_grad = &workspace.next_residual_grad;
        let token_index = TokenIndex::build(gpu, &batch.tokens, self.config.codebooks, self.token_embedding.value.shape().rows)?;

        gpu.embedding_backward(embedding_grad, &token_index, &mut self.token_embedding.gradient)?;
        gpu.embedding_backward(embedding_grad, &workspace.position_index, &mut self.position_embedding.gradient)
    }

    /// Zero gradients, forward, backward: everything but the optimiser step.
    pub fn accumulate_gradients(&mut self, gpu: &Gpu, workspace: &mut TrainingWorkspace, batch: &TrainingBatch) -> GpuResult<LossReport> {
        self.zero_gradients(gpu)?;
        let report = self.forward_loss(gpu, workspace, batch)?;
        self.backward(gpu, workspace, batch)?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_batch_fills_every_slot() {
        let batch = TrainingBatch::empty(SequenceShape { batch: 2, time: 3 }, 4);

        assert_eq!(batch.tokens.len(), 24);
        assert_eq!(batch.targets.len(), 24);
        assert!(batch.tokens.iter().all(|token| *token == staleynet_cuda::EMPTY_TOKEN));
        assert!(batch.targets.iter().all(|token| *token == staleynet_cuda::EMPTY_TOKEN));
    }
}
