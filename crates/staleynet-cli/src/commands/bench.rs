//! `staleynet bench`: the performance numbers reported in the book.
//!
//! GPU times bracket the work with a device synchronisation, so they include
//! kernel launch overhead but not host–device copies unless stated.

use super::open_gpu;
use anyhow::Result;
use clap::Args;
use serde::Serialize;
use staleynet_audio::{MelAnalyzer, MelConfig};
use staleynet_cuda::{Gpu, MatmulKernel, MatmulSpec, Shape};
use staleynet_data::features::GpuMelExtractor;
use staleynet_model::generate::{AttentionMode, GenerationConfig, Guidance, Sampling, generate};
use staleynet_model::{Conditioning, Gpt, ModelConfig, Random, SequenceShape, TrainingBatch, TrainingWorkspace};
use std::path::PathBuf;
use std::time::Instant;

/// Timed repetitions per measurement (after one warm-up run).
const REPETITIONS: usize = 5;

/// Seconds of synthetic audio for the feature benchmark.
const FEATURE_SECONDS: usize = 60;

/// Frames generated in the decoding benchmark.
const GENERATED_FRAMES: usize = 200;

/// Arguments of `staleynet bench`.
#[derive(Args)]
pub struct Arguments {
    /// Where to write the JSON results.
    #[arg(long, default_value = "benchmarks/results.json")]
    out: PathBuf,
}

/// One GEMM shape measured with both kernels.
#[derive(Serialize)]
struct MatmulResult {
    /// What the shape corresponds to in the model.
    role: &'static str,

    /// `rows × cols × inner`.
    shape: [usize; 3],

    /// Throughput of the naive kernel, GFLOP/s.
    naive_gflops: f64,

    /// Throughput of the tiled kernel, GFLOP/s.
    tiled_gflops: f64,

    /// Throughput of the register-blocked kernel, GFLOP/s.
    blocked_gflops: f64,
}

/// Everything `bench` measures.
#[derive(Serialize)]
struct BenchResults {
    /// Board model string.
    device: String,

    /// GEMM kernels at model shapes.
    matmul: Vec<MatmulResult>,

    /// Log-mel of 60 s of audio: GPU milliseconds.
    log_mel_gpu_ms: f64,

    /// Log-mel of 60 s of audio: CPU (rustfft + rayon) milliseconds.
    log_mel_cpu_ms: f64,

    /// Training step (forward + backward + no optimiser) with each GEMM kernel, ms.
    train_step_naive_ms: f64,

    /// See `train_step_naive_ms`.
    train_step_tiled_ms: f64,

    /// See `train_step_naive_ms`.
    train_step_blocked_ms: f64,

    /// Positions per second with the register-blocked kernel, the default.
    train_positions_per_second: f64,

    /// Generation latency per step with the KV cache, ms.
    decode_cached_ms_per_step: f64,

    /// Generation latency per step recomputing the full context, ms.
    decode_recompute_ms_per_step: f64,
}

/// Mean seconds of `work` over [`REPETITIONS`] runs after a warm-up.
fn time(gpu: &Gpu, mut work: impl FnMut() -> Result<()>) -> Result<f64> {
    work()?;
    gpu.synchronize()?;

    let started = Instant::now();
    for _ in 0..REPETITIONS {
        work()?;
    }
    gpu.synchronize()?;
    Ok(started.elapsed().as_secs_f64() / REPETITIONS as f64)
}

/// GFLOP/s of one GEMM shape with the current kernel.
fn matmul_gflops(gpu: &Gpu, rows: usize, cols: usize, inner: usize) -> Result<f64> {
    let left = gpu.zeros(Shape::new(rows, inner))?;
    let right = gpu.zeros(Shape::new(inner, cols))?;
    let mut output = gpu.zeros(Shape::new(rows, cols))?;
    let spec = MatmulSpec::new(rows, cols, inner);

    let seconds = time(gpu, || Ok(gpu.matmul(&spec, &left, &right, &mut output)?))?;
    Ok(2.0 * (rows * cols * inner) as f64 / seconds / 1e9)
}

/// A batch of random audio tokens shaped like real training data.
fn random_batch(config: &ModelConfig, shape: SequenceShape) -> TrainingBatch {
    let vocabulary = staleynet_model::Vocabulary::new(config);
    let mut random = Random::new(0);
    let mut batch = TrainingBatch::empty(shape, config.codebooks);

    for (slot, token) in batch.tokens.iter_mut().enumerate() {
        let codebook = slot % config.codebooks;
        *token = vocabulary.audio_token(codebook, random.below(config.codebook_size) as u32);
    }
    for target in batch.targets.iter_mut() {
        *target = random.below(config.codebook_size) as u32;
    }

    batch
}

pub fn run(arguments: Arguments) -> Result<()> {
    let mut gpu = open_gpu()?;
    let device = std::fs::read_to_string("/proc/device-tree/model").unwrap_or_default().trim_end_matches('\0').to_string();
    eprintln!("device: {device}");

    let config = ModelConfig { context_length: 512, ..ModelConfig::default() };
    let positions = 4 * 512;
    let gemm_shapes = [
        ("qkv projection", positions, 3 * config.width, config.width),
        ("feed-forward expand", positions, config.feed_forward_width(), config.width),
        ("codebook heads", positions, config.logits_width(), config.width),
        ("STFT as GEMM", 4096, MelConfig::default().frequency_bins(), MelConfig::default().fft_size),
    ];

    let mut matmul = Vec::new();
    for (role, rows, cols, inner) in gemm_shapes {
        gpu.matmul_kernel = MatmulKernel::Naive;
        let naive_gflops = matmul_gflops(&gpu, rows, cols, inner)?;
        gpu.matmul_kernel = MatmulKernel::Tiled;
        let tiled_gflops = matmul_gflops(&gpu, rows, cols, inner)?;
        gpu.matmul_kernel = MatmulKernel::Blocked;
        let blocked_gflops = matmul_gflops(&gpu, rows, cols, inner)?;

        eprintln!(
            "matmul {role:20} {rows}×{cols}×{inner}: naive {naive_gflops:6.1}, tiled {tiled_gflops:6.1}, blocked {blocked_gflops:6.1} GFLOP/s"
        );
        matmul.push(MatmulResult { role, shape: [rows, cols, inner], naive_gflops, tiled_gflops, blocked_gflops });
    }

    let analyzer = MelAnalyzer::new(MelConfig::default());
    let signal: Vec<f32> =
        (0..analyzer.config.sample_rate as usize * FEATURE_SECONDS).map(|index| (index as f32 * 0.013).sin() * 0.3).collect();
    let extractor = GpuMelExtractor::new(&gpu, &analyzer)?;
    let log_mel_gpu_ms = time(&gpu, || Ok(extractor.log_mel(&gpu, &analyzer, &signal).map(drop)?))? * 1e3;
    let cpu_started = Instant::now();
    let _ = analyzer.log_mel(&signal);
    let log_mel_cpu_ms = cpu_started.elapsed().as_secs_f64() * 1e3;
    eprintln!("log-mel of {FEATURE_SECONDS} s: GPU {log_mel_gpu_ms:.1} ms, CPU {log_mel_cpu_ms:.1} ms");

    let shape = SequenceShape { batch: 4, time: 512 };
    let batch = random_batch(&config, shape);
    let mut model = Gpt::new(&gpu, config.clone(), 0)?;
    let mut workspace = TrainingWorkspace::new(&gpu, &config, shape)?;
    let mut step_ms = Vec::new();
    for kernel in [MatmulKernel::Naive, MatmulKernel::Tiled, MatmulKernel::Blocked] {
        gpu.matmul_kernel = kernel;
        let seconds = time(&gpu, || Ok(model.accumulate_gradients(&gpu, &mut workspace, &batch).map(drop)?))?;
        eprintln!("train step B={} T={} ({kernel:?}): {:.0} ms", shape.batch, shape.time, seconds * 1e3);
        step_ms.push(seconds * 1e3);
    }

    let mut decode_ms = Vec::new();
    for attention in [AttentionMode::KvCache, AttentionMode::Recompute] {
        let generation_config = GenerationConfig {
            frames: GENERATED_FRAMES,
            sampling: Sampling::TopK { temperature: 1.0, k: 250 },
            guidance: Guidance::Off,
            attention,
            seed: 0,
        };
        let stats = generate(&gpu, &model, Conditioning { caption: "benchmark", lyrics: "" }, &generation_config)?.stats;
        let per_step = stats.seconds * 1e3 / stats.steps as f64;
        eprintln!("decode {attention:?}: {per_step:.2} ms/step");
        decode_ms.push(per_step);
    }

    let results = BenchResults {
        device,
        matmul,
        log_mel_gpu_ms,
        log_mel_cpu_ms,
        train_step_naive_ms: step_ms[0],
        train_step_tiled_ms: step_ms[1],
        train_step_blocked_ms: step_ms[2],
        train_positions_per_second: shape.rows() as f64 / (step_ms[2] / 1e3),
        decode_cached_ms_per_step: decode_ms[0],
        decode_recompute_ms_per_step: decode_ms[1],
    };

    if let Some(parent) = arguments.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&arguments.out, serde_json::to_string_pretty(&results)?)?;
    eprintln!("wrote {}", arguments.out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ModelConfig {
        ModelConfig { codebooks: 3, codebook_size: 10, ..ModelConfig::default() }
    }

    #[test]
    fn random_batch_fills_every_slot_with_an_audio_token() {
        let config = config();
        let vocabulary = staleynet_model::Vocabulary::new(&config);
        let batch = random_batch(&config, SequenceShape { batch: 2, time: 4 });

        assert_eq!(batch.tokens.len(), 24);
        assert_eq!(batch.targets.len(), 24);
        assert!(batch.targets.iter().all(|target| *target < 10));

        for (slot, token) in batch.tokens.iter().enumerate() {
            let codebook = slot % config.codebooks;
            let first = vocabulary.audio_token(codebook, 0);
            let last = vocabulary.audio_token(codebook, 9);

            assert!((first..=last).contains(token));
        }
    }

    #[test]
    fn random_batch_is_deterministic() {
        let config = config();
        let first = random_batch(&config, SequenceShape { batch: 2, time: 4 });
        let second = random_batch(&config, SequenceShape { batch: 2, time: 4 });

        assert_eq!(first.tokens, second.tokens);
        assert_eq!(first.targets, second.targets);
    }
}
