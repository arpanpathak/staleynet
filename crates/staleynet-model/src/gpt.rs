//! The decoder-only Transformer: embeddings, a stack of [`Block`]s, a final
//! norm and one classifier head per codebook. It models
//!
//! `P(audio | prompt) = Π_t Π_c P(a_t^c | prompt, a_<t)`
//!
//! with the codebooks of one step predicted in parallel from the same hidden
//! state (made sound by the [`DelayPattern`](crate::vocabulary::DelayPattern)).

use crate::config::ModelConfig;
use crate::layers::block::Block;
use crate::layers::linear::Linear;
use crate::layers::norm::RmsNorm;
use crate::parameter::{Decay, INIT_STD, Init, Parameter, Parameters};
use crate::random::Random;
use crate::vocabulary::Vocabulary;
use staleynet_cuda::{Gpu, GpuResult, Shape, Tensor, TokenBuffer, WriteMode};

/// The full network.
pub struct Gpt {
    /// Architecture this instance was built with.
    pub config: ModelConfig,

    /// Embedding of every text and audio token, `[vocabulary, width]`.
    pub token_embedding: Parameter,

    /// Learned absolute position embedding, `[sequence_capacity, width]`;
    /// rows are assigned by [`Vocabulary::position_id`].
    pub position_embedding: Parameter,

    /// Transformer blocks, applied in order.
    pub blocks: Vec<Block>,

    /// Norm applied to the last residual stream.
    pub final_norm: RmsNorm,

    /// Projection to `codebooks × codebook_size` logits.
    pub head: Linear,
}

impl Gpt {
    /// A freshly initialised model; identical seeds give identical weights.
    pub fn new(gpu: &Gpu, config: ModelConfig, seed: u64) -> GpuResult<Self> {
        config.validate();

        let mut random = Random::new(seed);
        let vocabulary = Vocabulary::new(&config);
        let embedding_init = Init::Normal(INIT_STD);

        let token_embedding = Parameter::new(
            gpu,
            "token_embedding".into(),
            Shape::new(vocabulary.size(), config.width),
            embedding_init,
            Decay::Exempt,
            &mut random,
        )?;
        let position_embedding = Parameter::new(
            gpu,
            "position_embedding".into(),
            Shape::new(config.sequence_capacity(), config.width),
            embedding_init,
            Decay::Exempt,
            &mut random,
        )?;

        let mut blocks = Vec::with_capacity(config.layers);
        for index in 0..config.layers {
            blocks.push(Block::new(gpu, &config, index, &mut random)?);
        }

        let final_norm = RmsNorm::new(gpu, "final_norm", config.width, &mut random)?;
        let head = Linear::new(gpu, "head", config.width, config.logits_width(), INIT_STD, &mut random)?;

        Ok(Self { config, token_embedding, position_embedding, blocks, final_norm, head })
    }

    /// The vocabulary layout of this model.
    pub fn vocabulary(&self) -> Vocabulary {
        Vocabulary::new(&self.config)
    }

    /// `output = Σ_slots token_embedding[tokens] + position_embedding[positions]`.
    pub fn embed(&self, gpu: &Gpu, tokens: &TokenBuffer, positions: &TokenBuffer, output: &mut Tensor) -> GpuResult {
        gpu.embedding_forward(&self.token_embedding.value, tokens, self.config.codebooks, output, WriteMode::Overwrite)?;
        gpu.embedding_forward(&self.position_embedding.value, positions, 1, output, WriteMode::Accumulate)
    }
}

impl Parameters for Gpt {
    fn visit_parameters(&mut self, visit: &mut dyn FnMut(&mut Parameter)) {
        visit(&mut self.token_embedding);
        visit(&mut self.position_embedding);

        for block in &mut self.blocks {
            block.visit_parameters(visit);
        }

        self.final_norm.visit_parameters(visit);
        self.head.visit_parameters(visit);
    }
}
