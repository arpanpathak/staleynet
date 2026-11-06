//! End-to-end correctness of the model on the GPU:
//! analytic gradients match finite differences, cached decoding matches the
//! full forward pass (with and without a lyrics field), a fixed batch can be
//! memorised, checkpoints round-trip with their optimiser state, and a model
//! without lyrics can be warm-started into one with lyrics.

use staleynet_cuda::Gpu;
use staleynet_model::checkpoint;
use staleynet_model::decode::Decoder;
use staleynet_model::optimizer::{AdamW, OptimizerConfig};
use staleynet_model::parameter::Parameters;
use staleynet_model::{Conditioning, Gpt, ModelConfig, Random, SequenceShape, TrainingBatch, TrainingWorkspace};
use std::sync::{Mutex, MutexGuard, OnceLock};

fn gpu() -> MutexGuard<'static, Gpu> {
    static GPU: OnceLock<Mutex<Gpu>> = OnceLock::new();
    let guard = GPU
        .get_or_init(|| Mutex::new(Gpu::new(0).expect("CUDA device 0")))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.bind_to_current_thread().expect("bind CUDA context");
    guard
}

fn tiny_config() -> ModelConfig {
    ModelConfig {
        width: 32,
        heads: 4,
        layers: 2,
        feed_forward_multiplier: 2,
        context_length: 16,
        prompt_bytes: 3,
        lyrics_bytes: 0,
        lyric_cursor: false,
        codebooks: 2,
        codebook_size: 7,
    }
}

/// [`tiny_config`] with a four-byte lyrics field.
fn tiny_lyrics_config() -> ModelConfig {
    ModelConfig { lyrics_bytes: 4, ..tiny_config() }
}

/// [`tiny_lyrics_config`] with a lyric cursor.
fn tiny_cursor_config() -> ModelConfig {
    ModelConfig { lyric_cursor: true, ..tiny_lyrics_config() }
}

/// A batch whose sequences are a short prompt followed by random audio steps,
/// with next-position targets, in the same layout the data pipeline produces.
fn random_batch(config: &ModelConfig, shape: SequenceShape, seed: u64) -> TrainingBatch {
    let vocabulary = staleynet_model::Vocabulary::new(config);
    let codebooks = config.codebooks;
    let prompt_positions = vocabulary.prompt_positions();
    let mut random = Random::new(seed);
    let mut batch = TrainingBatch::empty(shape, codebooks);

    for sequence in 0..shape.batch {
        let row = |position: usize| (sequence * shape.time + position) * codebooks;
        vocabulary.write_prompt(Conditioning { caption: "hi", lyrics: "la" }, &mut batch.tokens[row(0)..row(prompt_positions)]);

        for position in prompt_positions..shape.time {
            for codebook in 0..codebooks {
                let code = random.below(config.codebook_size) as u32;
                batch.tokens[row(position) + codebook] = vocabulary.audio_token(codebook, code);
                batch.targets[row(position - 1) + codebook] = code;
            }
        }

        // Random cursors, some empty, from the last prompt position on.
        for position in prompt_positions - 1..shape.time {
            let byte = random.below(config.lyrics_bytes + 1);
            batch.cursors[sequence * shape.time + position] = vocabulary.cursor_row(byte);
        }
    }

    batch
}

/// Reads one scalar of a parameter, by visitor index and element.
fn read_parameter(gpu: &Gpu, model: &mut Gpt, parameter_index: usize, element: usize) -> f32 {
    let mut value = 0.0;
    let mut index = 0;
    model.visit_parameters(&mut |parameter| {
        if index == parameter_index {
            value = gpu.download(&parameter.value).unwrap()[element];
        }
        index += 1;
    });
    value
}

/// Overwrites one scalar of a parameter.
fn write_parameter(gpu: &Gpu, model: &mut Gpt, parameter_index: usize, element: usize, value: f32) {
    let mut index = 0;
    model.visit_parameters(&mut |parameter| {
        if index == parameter_index {
            let mut host = gpu.download(&parameter.value).unwrap();
            host[element] = value;
            gpu.copy_from_host(&mut parameter.value, &host).unwrap();
        }
        index += 1;
    });
}

#[test]
fn analytic_gradients_match_finite_differences() {
    assert_gradients_match(tiny_config());
}

#[test]
fn analytic_gradients_match_finite_differences_with_a_lyric_cursor() {
    assert_gradients_match(tiny_cursor_config());
}

/// Compares backpropagated gradients with central finite differences.
fn assert_gradients_match(config: ModelConfig) {
    let gpu = gpu();
    let shape = SequenceShape { batch: 2, time: 12 + config.lyrics_bytes };
    let batch = random_batch(&config, shape, 1);
    let mut model = Gpt::new(&gpu, config.clone(), 3).unwrap();
    let mut workspace = TrainingWorkspace::new(&gpu, &config, shape).unwrap();

    model
        .accumulate_gradients(&gpu, &mut workspace, &batch)
        .unwrap();
    let mut analytic: Vec<(String, Vec<f32>)> = Vec::new();
    model.visit_parameters(&mut |parameter| analytic.push((parameter.name.clone(), gpu.download(&parameter.gradient).unwrap())));

    let step = 1e-2;
    let mut random = Random::new(9);
    let mut checked = 0;
    for (parameter_index, (name, gradient)) in analytic.iter().enumerate() {
        for _ in 0..3 {
            let element = random.below(gradient.len());
            let original = read_parameter(&gpu, &mut model, parameter_index, element);

            write_parameter(&gpu, &mut model, parameter_index, element, original + step);
            let loss_up = model
                .forward_loss(&gpu, &mut workspace, &batch)
                .unwrap()
                .mean;
            write_parameter(&gpu, &mut model, parameter_index, element, original - step);
            let loss_down = model
                .forward_loss(&gpu, &mut workspace, &batch)
                .unwrap()
                .mean;
            write_parameter(&gpu, &mut model, parameter_index, element, original);

            let numeric = (loss_up - loss_down) / (2.0 * step);
            let difference = (numeric - gradient[element]).abs();
            let relative = difference / (numeric.abs() + gradient[element].abs()).max(1e-3);
            assert!(relative < 0.05 || difference < 3e-4, "{name}[{element}]: numeric {numeric} vs analytic {}", gradient[element]);
            checked += 1;
        }
    }
    assert!(checked > 30);
}

#[test]
fn cached_decoding_reproduces_the_full_forward_pass() {
    assert_cached_decoding_matches(tiny_config());
}

#[test]
fn cached_decoding_reproduces_the_full_forward_pass_with_lyrics() {
    assert_cached_decoding_matches(tiny_lyrics_config());
}

#[test]
fn cached_decoding_reproduces_the_full_forward_pass_with_a_lyric_cursor() {
    assert_cached_decoding_matches(tiny_cursor_config());
}

/// Feeds a sequence position by position through the KV cache and compares
/// every step's logits with one full forward pass.
fn assert_cached_decoding_matches(config: ModelConfig) {
    let gpu = gpu();
    let shape = SequenceShape { batch: 1, time: 14 + config.lyrics_bytes };
    let batch = random_batch(&config, shape, 2);
    let model = Gpt::new(&gpu, config.clone(), 4).unwrap();
    let mut workspace = TrainingWorkspace::new(&gpu, &config, shape).unwrap();

    model.forward_loss(&gpu, &mut workspace, &batch).unwrap();
    let full_logits = gpu.download(workspace.logits()).unwrap();
    let logits_width = config.logits_width();
    let codebooks = config.codebooks;

    let mut decoder = Decoder::new(&gpu, &model).unwrap();
    let prefill = 7;
    let mut fed = prefill;
    let mut logits = decoder
        .feed_with_cursors(&gpu, &model, &batch.tokens[..prefill * codebooks], &batch.cursors[..prefill])
        .unwrap();
    loop {
        let expected = &full_logits[(fed - 1) * logits_width..fed * logits_width];
        let worst = expected
            .iter()
            .zip(&logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(worst < 1e-4, "position {}: max difference {worst}", fed - 1);

        if fed == shape.time {
            break;
        }
        logits = decoder
            .feed_with_cursors(&gpu, &model, &batch.tokens[fed * codebooks..(fed + 1) * codebooks], &batch.cursors[fed..fed + 1])
            .unwrap();
        fed += 1;
    }
}

#[test]
fn a_fixed_batch_is_memorised() {
    let gpu = gpu();
    let config = tiny_config();
    let shape = SequenceShape { batch: 2, time: 16 };
    let batch = random_batch(&config, shape, 5);
    let mut model = Gpt::new(&gpu, config.clone(), 6).unwrap();
    let mut workspace = TrainingWorkspace::new(&gpu, &config, shape).unwrap();
    let optimizer_config =
        OptimizerConfig { peak_learning_rate: 1e-2, final_learning_rate: 1e-2, warmup_steps: 1, weight_decay: 0.0, ..Default::default() };
    let mut optimizer = AdamW::new(&gpu, optimizer_config, &mut model).unwrap();

    let initial = model
        .accumulate_gradients(&gpu, &mut workspace, &batch)
        .unwrap()
        .mean;
    let mut last = initial;
    for _ in 0..300 {
        last = model
            .accumulate_gradients(&gpu, &mut workspace, &batch)
            .unwrap()
            .mean;
        optimizer.step(&gpu, &mut model).unwrap();
    }

    // ln(7) ≈ 1.95 nats is chance level for 7 codes.
    assert!(initial > 1.7, "initial loss {initial}");
    assert!(last < 0.1, "loss after training {last}");
}

#[test]
fn checkpoints_round_trip_exactly() {
    let gpu = gpu();
    let config = tiny_config();
    let mut model = Gpt::new(&gpu, config.clone(), 7).unwrap();
    let mut optimizer = AdamW::new(&gpu, OptimizerConfig::default(), &mut model).unwrap();
    optimizer.steps_taken = 42;
    let directory = std::env::temp_dir().join(format!("staleynet-checkpoint-test-{}", std::process::id()));

    checkpoint::save(&gpu, &directory, &mut model, Some(&optimizer), optimizer.steps_taken, serde_json::json!({})).unwrap();
    let mut restored_optimizer = AdamW::new(&gpu, OptimizerConfig::default(), &mut model).unwrap();
    let (mut restored, metadata) = checkpoint::load(&gpu, &directory, Some(&mut restored_optimizer)).unwrap();

    let mut original_values = Vec::new();
    model.visit_parameters(&mut |parameter| original_values.push(gpu.download(&parameter.value).unwrap()));
    let mut restored_values = Vec::new();
    restored.visit_parameters(&mut |parameter| restored_values.push(gpu.download(&parameter.value).unwrap()));

    assert_eq!(original_values, restored_values);
    assert_eq!(metadata.step, 42);
    assert_eq!(restored_optimizer.steps_taken, 42);
    for (saved, loaded) in optimizer.moments.iter().zip(&restored_optimizer.moments) {
        assert_eq!(gpu.download(&saved.first).unwrap(), gpu.download(&loaded.first).unwrap());
        assert_eq!(gpu.download(&saved.second).unwrap(), gpu.download(&loaded.second).unwrap());
    }
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn a_model_without_lyrics_warm_starts_one_with_lyrics() {
    let gpu = gpu();
    let mut source = Gpt::new(&gpu, tiny_config(), 8).unwrap();
    let directory = std::env::temp_dir().join(format!("staleynet-warm-start-test-{}", std::process::id()));
    checkpoint::save(&gpu, &directory, &mut source, None, 0, serde_json::json!({})).unwrap();

    let mut target = Gpt::new(&gpu, tiny_lyrics_config(), 9).unwrap();
    let transfer = checkpoint::load_weights(&gpu, &directory, &mut target).unwrap();
    assert_eq!(transfer.grown, vec!["position_embedding".to_string()]);
    assert!(transfer.fresh.is_empty());

    let old_rows = gpu.download(&source.position_embedding.value).unwrap();
    let new_rows = gpu.download(&target.position_embedding.value).unwrap();
    assert_eq!(&new_rows[..old_rows.len()], old_rows.as_slice());
    assert_eq!(gpu.download(&source.head.weight.value).unwrap(), gpu.download(&target.head.weight.value).unwrap());
    std::fs::remove_dir_all(directory).unwrap();
}
