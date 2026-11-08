//! The GPU handle and the device-resident tensor types.

use crate::kernels::{attention, convolution, elementwise, embedding, loss, matmul, normalization, optimizer, overlap, signal, spectrum};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError};
use cuda_host::EmbeddedModuleError;
use std::fmt;
use std::sync::Arc;

/// Result of any operation that talks to the CUDA driver.
pub type GpuResult<T = ()> = Result<T, DriverError>;

/// Why a [`Gpu`] could not be created.
#[derive(Debug)]
pub enum GpuInitError {
    /// The driver refused to create a context (no device, driver mismatch, …).
    Driver(DriverError),

    /// The embedded PTX module could not be found or loaded.
    Module(EmbeddedModuleError),
}

impl fmt::Display for GpuInitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuInitError::Driver(error) => write!(formatter, "CUDA driver error: {error:?}"),
            GpuInitError::Module(error) => write!(formatter, "kernel module error: {error:?}"),
        }
    }
}

impl std::error::Error for GpuInitError {}

/// Token ids in device memory (`u32`, [`crate::kernels::EMPTY_TOKEN`] = empty slot).
pub type TokenBuffer = DeviceBuffer<u32>;

/// Which GEMM kernel [`Gpu::matmul`](crate::ops) launches. Kept switchable so
/// the two can be benchmarked against each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulKernel {
    /// One thread per output, operands read straight from global memory.
    Naive,

    /// 16×16 shared-memory tiles.
    Tiled,

    /// 64×64 shared-memory tiles with a 4×4 register block per thread.
    Blocked,

    /// The tiles of [`MatmulKernel::Blocked`] multiplied on the tensor cores
    /// in TF32: inputs rounded to a 10-bit mantissa, sums kept in f32.
    TensorCore,
}

impl MatmulKernel {
    /// Largest relative error against an exact f32 product that the kernel's
    /// arithmetic allows. TF32 rounds each input to 2⁻¹¹ relative precision,
    /// so its products differ by a few parts in 10⁴; the f32 kernels differ
    /// only by summation order.
    pub fn tolerance(self) -> f32 {
        match self {
            MatmulKernel::Naive | MatmulKernel::Tiled | MatmulKernel::Blocked => 1e-5,
            MatmulKernel::TensorCore => 2e-3,
        }
    }
}

/// The loaded kernel modules, one per kind of computation (see [`crate::kernels`]).
pub struct Kernels {
    /// Matrix products.
    pub(crate) matmul: matmul::kernels::LoadedModule,

    /// Scaled addition, bias, column sums and GELU.
    pub(crate) elementwise: elementwise::kernels::LoadedModule,

    /// RMS normalisation and its gradients.
    pub(crate) normalization: normalization::kernels::LoadedModule,

    /// Head permutes and the causal softmax.
    pub(crate) attention: attention::kernels::LoadedModule,

    /// Embedding lookup and gradient.
    pub(crate) embedding: embedding::kernels::LoadedModule,

    /// Cross-entropy.
    pub(crate) loss: loss::kernels::LoadedModule,

    /// AdamW and the sum of squares.
    pub(crate) optimizer: optimizer::kernels::LoadedModule,

    /// Spectrogram helpers and the nearest-prototype search.
    pub(crate) signal: signal::kernels::LoadedModule,

    /// Overlap-add for the inverse STFT, and framing for the spectral losses.
    pub(crate) overlap: overlap::kernels::LoadedModule,

    /// The vocoder's predicted spectrum and its spectral-loss gradients.
    pub(crate) spectrum: spectrum::kernels::LoadedModule,

    /// Depthwise convolutions over time and the unfolding for convolution products.
    pub(crate) convolution: convolution::kernels::LoadedModule,
}

impl Kernels {
    /// Loads every module into `context`.
    fn load(context: &Arc<CudaContext>) -> Result<Self, EmbeddedModuleError> {
        Ok(Self {
            matmul: matmul::kernels::load(context)?,
            elementwise: elementwise::kernels::load(context)?,
            normalization: normalization::kernels::load(context)?,
            attention: attention::kernels::load(context)?,
            embedding: embedding::kernels::load(context)?,
            loss: loss::kernels::load(context)?,
            optimizer: optimizer::kernels::load(context)?,
            signal: signal::kernels::load(context)?,
            overlap: overlap::kernels::load(context)?,
            spectrum: spectrum::kernels::load(context)?,
            convolution: convolution::kernels::load(context)?,
        })
    }
}

/// A CUDA context, its stream, and the loaded StaleyNET kernels.
///
/// All work is issued on one stream, so operations execute in program order
/// and need no explicit synchronisation between them.
pub struct Gpu {
    /// Driver context owning all allocations.
    pub(crate) context: Arc<CudaContext>,

    /// The stream every kernel and copy is issued on.
    pub(crate) stream: Arc<CudaStream>,

    /// Handles to the compiled kernels.
    pub(crate) kernels: Kernels,

    /// GEMM implementation used by matmul.
    pub matmul_kernel: MatmulKernel,
}

impl Gpu {
    /// Opens device `ordinal` and loads the kernel module.
    pub fn new(ordinal: usize) -> Result<Self, GpuInitError> {
        let context = CudaContext::new(ordinal).map_err(GpuInitError::Driver)?;
        let stream = context.default_stream();
        let kernels = Kernels::load(&context).map_err(GpuInitError::Module)?;

        Ok(Self { context, stream, kernels, matmul_kernel: MatmulKernel::Blocked })
    }

    /// Makes this context current on the calling thread. Needed when a `Gpu`
    /// is shared across threads (e.g. by the test harness).
    pub fn bind_to_current_thread(&self) -> GpuResult {
        self.context.bind_to_thread()
    }

    /// Blocks until all queued work has finished.
    pub fn synchronize(&self) -> GpuResult {
        self.stream.synchronize()
    }

    /// Allocates a zero-filled tensor.
    pub fn zeros(&self, shape: Shape) -> GpuResult<Tensor> {
        Ok(Tensor { buffer: DeviceBuffer::zeroed(&self.stream, shape.len())?, shape })
    }

    /// Copies host values into a new tensor of `shape`.
    pub fn upload(&self, shape: Shape, values: &[f32]) -> GpuResult<Tensor> {
        assert_eq!(values.len(), shape.len(), "upload: {} values for shape {shape:?}", values.len());
        Ok(Tensor { buffer: DeviceBuffer::from_host(&self.stream, values)?, shape })
    }

    /// Copies a tensor back to the host.
    pub fn download(&self, tensor: &Tensor) -> GpuResult<Vec<f32>> {
        tensor.buffer.to_host_vec(&self.stream)
    }

    /// Overwrites an existing tensor with host values, reusing its allocation.
    pub fn copy_from_host(&self, tensor: &mut Tensor, values: &[f32]) -> GpuResult {
        tensor.buffer.copy_from_host(&self.stream, values)
    }

    /// Fills a tensor with zeros in place.
    pub fn fill_zero(&self, tensor: &mut Tensor) -> GpuResult {
        tensor.buffer.zero_async(&self.stream)
    }

    /// Copies token ids into a new device buffer.
    pub fn upload_tokens(&self, tokens: &[u32]) -> GpuResult<TokenBuffer> {
        DeviceBuffer::from_host(&self.stream, tokens)
    }

    /// Overwrites an existing token buffer with host ids, reusing its allocation.
    pub fn copy_tokens_from_host(&self, buffer: &mut TokenBuffer, tokens: &[u32]) -> GpuResult {
        buffer.copy_from_host(&self.stream, tokens)
    }

    /// Copies token ids back to the host.
    pub fn download_tokens(&self, tokens: &TokenBuffer) -> GpuResult<Vec<u32>> {
        tokens.to_host_vec(&self.stream)
    }
}

/// Rows × columns of a row-major matrix.
///
/// Higher-rank tensors are stored as matrices whose rows enumerate all leading
/// axes, e.g. attention scores `[batch·heads·queries, keys]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// Number of rows.
    pub rows: usize,

    /// Number of columns (the contiguous axis).
    pub cols: usize,
}

impl Shape {
    /// A `rows × cols` shape.
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    /// A single row of `cols` values (vectors such as biases and gains).
    pub const fn vector(cols: usize) -> Self {
        Self { rows: 1, cols }
    }

    /// Number of elements.
    pub const fn len(&self) -> usize {
        self.rows * self.cols
    }

    /// True when the shape holds no elements.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A row-major `f32` matrix in device memory.
pub struct Tensor {
    /// Device allocation of exactly `shape.len()` values.
    pub(crate) buffer: DeviceBuffer<f32>,

    /// Logical shape of `buffer`.
    shape: Shape,
}

impl Tensor {
    /// Logical shape.
    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.shape.len()
    }

    /// True when the tensor holds no elements.
    pub fn is_empty(&self) -> bool {
        self.shape.is_empty()
    }

    /// Reinterprets the same memory with a new shape of equal size (no copy).
    pub fn reshaped(self, shape: Shape) -> Self {
        assert_eq!(shape.len(), self.shape.len(), "reshape {:?} → {shape:?} changes size", self.shape);
        Self { buffer: self.buffer, shape }
    }
}
