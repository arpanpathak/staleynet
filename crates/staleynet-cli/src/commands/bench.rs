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
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Timed repetitions per measurement (after one warm-up run).
const REPETITIONS: usize = 5;

/// Seconds of synthetic audio for the feature benchmark.
const FEATURE_SECONDS: usize = 60;

/// Frames generated in the decoding benchmark.
const GENERATED_FRAMES: usize = 200;

/// Sequences per batch in the training-step benchmark.
const BENCH_BATCH: usize = 4;

/// Positions per sequence in the training-step benchmark, and the model's context.
const BENCH_SEQUENCE: usize = 512;

/// Spectrogram frames in the STFT-as-GEMM shape.
const STFT_FRAMES: usize = 4096;

/// Candidates kept by top-k sampling in the decoding benchmark.
const BENCH_TOP_K: usize = 250;

/// Duration of a code step while benchmarking decoding; it only paces the
/// lyric cursor, which the empty benchmark lyrics never use.
const BENCH_SECONDS_PER_STEP: f32 = 0.05;

/// Number of GEMM kernels compared.
const KERNEL_COUNT: usize = 4;

/// The GEMM kernels compared, in the order the results report them.
const KERNELS: [MatmulKernel; KERNEL_COUNT] = [MatmulKernel::Naive, MatmulKernel::Tiled, MatmulKernel::Blocked, MatmulKernel::TensorCore];

/// Arguments of `staleynet bench`.
#[derive(Args)]
pub struct Arguments {
    /// Where to write the JSON results.
    #[arg(long, default_value = "benchmarks/results.json")]
    out: PathBuf,
}

/// One GEMM shape measured with every kernel.
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

    /// Throughput of the TF32 tensor-core kernel, GFLOP/s.
    tensor_core_gflops: f64,
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

    /// See `train_step_naive_ms`.
    train_step_tensor_core_ms: f64,

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

/// Runs every measurement and writes the results as JSON.
pub fn run(arguments: Arguments) -> Result<()> {
    let mut gpu = open_gpu()?;
    let device = std::fs::read_to_string("/proc/device-tree/model")
        .unwrap_or_default()
        .trim_end_matches('\0')
        .to_string();
    eprintln!("device: {device}");

    let config = ModelConfig { context_length: BENCH_SEQUENCE, ..ModelConfig::default() };
    let shape = SequenceShape { batch: BENCH_BATCH, time: BENCH_SEQUENCE };

    let matmul = measure_matmuls(&mut gpu, &config, shape.rows())?;
    let (log_mel_gpu_ms, log_mel_cpu_ms) = measure_log_mel(&gpu)?;

    let mut model = Gpt::new(&gpu, config.clone(), 0)?;
    let [naive_ms, tiled_ms, blocked_ms, tensor_core_ms] = measure_train_steps(&mut gpu, &mut model, shape)?;
    let [cached_ms, recompute_ms] = measure_decoding(&gpu, &model)?;

    let results = BenchResults {
        device,
        matmul,
        log_mel_gpu_ms,
        log_mel_cpu_ms,
        train_step_naive_ms: naive_ms,
        train_step_tiled_ms: tiled_ms,
        train_step_blocked_ms: blocked_ms,
        train_step_tensor_core_ms: tensor_core_ms,
        train_positions_per_second: shape.rows() as f64 / (blocked_ms / 1e3),
        decode_cached_ms_per_step: cached_ms,
        decode_recompute_ms_per_step: recompute_ms,
    };
    write_results(&arguments.out, &results)
}

/// Every GEMM kernel on the matrix shapes the model actually uses, for
/// `positions` rows of activations.
fn measure_matmuls(gpu: &mut Gpu, config: &ModelConfig, positions: usize) -> Result<Vec<MatmulResult>> {
    let mel = MelConfig::default();
    let shapes = [
        ("qkv projection", positions, 3 * config.width, config.width),
        ("feed-forward expand", positions, config.feed_forward_width(), config.width),
        ("codebook heads", positions, config.logits_width(), config.width),
        ("STFT as GEMM", STFT_FRAMES, mel.frequency_bins(), mel.fft_size),
    ];

    let mut results = Vec::with_capacity(shapes.len());
    for (role, rows, cols, inner) in shapes {
        let mut gflops = [0.0; KERNEL_COUNT];
        for (slot, kernel) in gflops.iter_mut().zip(KERNELS) {
            *slot = with_kernel(gpu, kernel, |gpu| matmul_gflops(gpu, rows, cols, inner))?;
        }
        let [naive_gflops, tiled_gflops, blocked_gflops, tensor_core_gflops] = gflops;

        eprintln!(
            "matmul {role:20} {rows}×{cols}×{inner}: naive {naive_gflops:6.1}, tiled {tiled_gflops:6.1}, blocked {blocked_gflops:6.1}, tensor core {tensor_core_gflops:6.1} GFLOP/s"
        );
        results.push(MatmulResult { role, shape: [rows, cols, inner], naive_gflops, tiled_gflops, blocked_gflops, tensor_core_gflops });
    }
    Ok(results)
}

/// Milliseconds to compute the log-mel spectrogram of [`FEATURE_SECONDS`] of
/// audio on the GPU (as a matrix product) and on the CPU (FFT), in that order.
fn measure_log_mel(gpu: &Gpu) -> Result<(f64, f64)> {
    let analyzer = MelAnalyzer::new(MelConfig::default());
    let samples = analyzer.config.sample_rate as usize * FEATURE_SECONDS;
    let signal: Vec<f32> = (0..samples)
        .map(|index| (index as f32 * 0.013).sin() * 0.3)
        .collect();

    let extractor = GpuMelExtractor::new(gpu, &analyzer)?;
    let gpu_ms = time(gpu, || Ok(extractor.log_mel(gpu, &analyzer, &signal).map(drop)?))? * 1e3;

    let cpu_started = Instant::now();
    let _ = analyzer.log_mel(&signal);
    let cpu_ms = cpu_started.elapsed().as_secs_f64() * 1e3;

    eprintln!("log-mel of {FEATURE_SECONDS} s: GPU {gpu_ms:.1} ms, CPU {cpu_ms:.1} ms");
    Ok((gpu_ms, cpu_ms))
}

/// Milliseconds of one forward and backward pass with each GEMM kernel, in
/// the order of [`KERNELS`].
fn measure_train_steps(gpu: &mut Gpu, model: &mut Gpt, shape: SequenceShape) -> Result<[f64; KERNEL_COUNT]> {
    let batch = random_batch(&model.config, shape);
    let mut workspace = TrainingWorkspace::new(gpu, &model.config, shape)?;

    let mut milliseconds = [0.0; KERNEL_COUNT];
    for (slot, kernel) in milliseconds.iter_mut().zip(KERNELS) {
        let seconds = with_kernel(gpu, kernel, |gpu| {
            time(gpu, || {
                Ok(model
                    .accumulate_gradients(gpu, &mut workspace, &batch)
                    .map(drop)?)
            })
        })?;
        eprintln!("train step B={} T={} ({kernel:?}): {:.0} ms", shape.batch, shape.time, seconds * 1e3);
        *slot = seconds * 1e3;
    }
    Ok(milliseconds)
}

/// Milliseconds per generated step with the key/value cache and with full
/// recomputation, in that order.
fn measure_decoding(gpu: &Gpu, model: &Gpt) -> Result<[f64; 2]> {
    let mut milliseconds = [0.0; 2];

    for (slot, attention) in milliseconds
        .iter_mut()
        .zip([AttentionMode::KvCache, AttentionMode::Recompute])
    {
        let config = GenerationConfig {
            frames: GENERATED_FRAMES,
            sampling: Sampling::TopK { temperature: 1.0, k: BENCH_TOP_K },
            guidance: Guidance::Off,
            attention,
            seed: 0,
            seconds_per_step: BENCH_SECONDS_PER_STEP,
        };
        let stats = generate(gpu, model, Conditioning { caption: "benchmark", lyrics: "" }, &config)?.stats;

        *slot = stats.seconds * 1e3 / stats.steps as f64;
        eprintln!("decode {attention:?}: {:.2} ms/step", *slot);
    }
    Ok(milliseconds)
}

/// Runs `work` with `kernel` selected for every matrix product.
fn with_kernel<T>(gpu: &mut Gpu, kernel: MatmulKernel, work: impl FnOnce(&Gpu) -> T) -> T {
    gpu.matmul_kernel = kernel;
    work(gpu)
}

/// Writes `results` to `path` as pretty-printed JSON, creating its folder.
fn write_results(path: &Path, results: &BenchResults) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(path, serde_json::to_string_pretty(results)?)?;
    eprintln!("wrote {}", path.display());
    Ok(())
}
