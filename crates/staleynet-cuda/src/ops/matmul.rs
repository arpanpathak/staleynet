//! Strided, batched matrix multiplication:
//! `output[b] = scale · op(left[b]) · op(right[b])  (+ output[b])`.
//!
//! `op(X)` is `X` or `Xᵀ` depending on how the operand is stored. Expressing
//! transposes as a storage flag (as BLAS does) means no layer ever has to
//! materialise a transposed copy. See the BLAS `?gemm` interface:
//! <https://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3.html>

use super::WriteMode;
use crate::device::{Gpu, GpuResult, MatmulKernel, Tensor};
use crate::kernels::{Operand, Product};
use crate::launch;

/// How an operand is laid out relative to the logical matrix `op(X)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    /// `op(X) = X`, stored row-major as `[rows, cols]`.
    RowMajor,

    /// `op(X) = Xᵀ`: the logical `[rows, cols]` matrix is stored as `[cols, rows]`.
    Transposed,
}

impl Storage {
    /// The strides of a logical `[rows, cols]` matrix stored this way, with
    /// `batch_stride` between consecutive matrices.
    fn operand(self, rows: usize, cols: usize, batch_stride: usize) -> Operand {
        match self {
            Storage::RowMajor => Operand { row_stride: cols, col_stride: 1, batch_stride },
            Storage::Transposed => Operand { row_stride: 1, col_stride: rows, batch_stride },
        }
    }
}

/// Element offsets between consecutive matrices of a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchStrides {
    /// Number of independent products.
    pub count: usize,

    /// Elements between consecutive left operands (0 broadcasts one matrix).
    pub left: usize,

    /// Elements between consecutive right operands (0 broadcasts one matrix).
    pub right: usize,

    /// Elements between consecutive outputs.
    pub output: usize,
}

impl BatchStrides {
    /// A single, unbatched product.
    pub const SINGLE: Self = Self { count: 1, left: 0, right: 0, output: 0 };
}

/// Full description of one (batched) GEMM.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatmulSpec {
    /// Rows of `op(left)` and of the output.
    pub rows: usize,

    /// Columns of `op(right)` and of the output.
    pub cols: usize,

    /// Shared inner dimension: columns of `op(left)`, rows of `op(right)`.
    pub inner: usize,

    /// Storage of the left operand.
    pub left: Storage,

    /// Storage of the right operand.
    pub right: Storage,

    /// Batch layout.
    pub batch: BatchStrides,

    /// Multiplier applied to the product.
    pub scale: f32,

    /// Overwrite or accumulate into the output.
    pub write: WriteMode,
}

impl MatmulSpec {
    /// `[rows, inner] · [inner, cols]`, row-major, unbatched, overwrite, scale 1.
    pub fn new(rows: usize, cols: usize, inner: usize) -> Self {
        Self {
            rows,
            cols,
            inner,
            left: Storage::RowMajor,
            right: Storage::RowMajor,
            batch: BatchStrides::SINGLE,
            scale: 1.0,
            write: WriteMode::Overwrite,
        }
    }

    /// Use the left operand transposed.
    pub fn left_transposed(self) -> Self {
        Self { left: Storage::Transposed, ..self }
    }

    /// Use the right operand transposed.
    pub fn right_transposed(self) -> Self {
        Self { right: Storage::Transposed, ..self }
    }

    /// Add into the output instead of overwriting it.
    pub fn accumulate(self) -> Self {
        Self { write: WriteMode::Accumulate, ..self }
    }

    /// Multiply the product by `scale`.
    pub fn scaled(self, scale: f32) -> Self {
        Self { scale, ..self }
    }

    /// Run `strides.count` products with the given offsets.
    pub fn batched(self, strides: BatchStrides) -> Self {
        Self { batch: strides, ..self }
    }

    /// The description of this product that the kernels receive.
    fn product(&self) -> Product {
        Product {
            rows: self.rows,
            cols: self.cols,
            inner: self.inner,
            left: self.left.operand(self.rows, self.inner, self.batch.left),
            right: self.right.operand(self.inner, self.cols, self.batch.right),
            output_batch_stride: self.batch.output,
            scale: self.scale,
            accumulate: self.write.flag(),
        }
    }

    /// Elements an operand of `per_matrix` elements must hold for this batch.
    fn required_len(&self, stride: usize, per_matrix: usize) -> usize {
        (self.batch.count - 1) * stride + per_matrix
    }
}

impl Gpu {
    /// Runs `spec` with the kernel selected by [`Gpu::matmul_kernel`].
    pub fn matmul(&self, spec: &MatmulSpec, left: &Tensor, right: &Tensor, output: &mut Tensor) -> GpuResult {
        assert!(left.len() >= spec.required_len(spec.batch.left, spec.rows * spec.inner), "matmul: left operand too small");
        assert!(right.len() >= spec.required_len(spec.batch.right, spec.inner * spec.cols), "matmul: right operand too small");
        assert!(output.len() >= spec.required_len(spec.batch.output, spec.rows * spec.cols), "matmul: output too small");

        let config = match self.matmul_kernel {
            MatmulKernel::Naive | MatmulKernel::Tiled => launch::matmul_tiles(spec.rows, spec.cols, spec.batch.count),
            MatmulKernel::Blocked | MatmulKernel::TensorCore => launch::matmul_blocks(spec.rows, spec.cols, spec.batch.count),
        };

        let product = spec.product();
        let kernels = &self.kernels.matmul;
        let (left, right, output) = (&left.buffer, &right.buffer, &mut output.buffer);

        // SAFETY: the grid is the tile grid each kernel indexes by (x: column tile,
        // y: row tile, z: matrix of the batch); the operand sizes were asserted
        // above, and every write is bound-checked in the kernel.
        unsafe {
            match self.matmul_kernel {
                MatmulKernel::Naive => kernels.matmul_naive(&self.stream, config, left, right, output, product),
                MatmulKernel::Tiled => kernels.matmul_tiled(&self.stream, config, left, right, output, product),
                MatmulKernel::Blocked => kernels.matmul_blocked(&self.stream, config, left, right, output, product),
                MatmulKernel::TensorCore => kernels.matmul_tensor_core(&self.stream, config, left, right, output, product),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_unbatched_row_major_overwrite_with_unit_scale() {
        let spec = MatmulSpec::new(2, 3, 4);

        assert_eq!((spec.rows, spec.cols, spec.inner), (2, 3, 4));
        assert_eq!(spec.left, Storage::RowMajor);
        assert_eq!(spec.right, Storage::RowMajor);
        assert_eq!(spec.batch, BatchStrides::SINGLE);
        assert_eq!(spec.scale, 1.0);
        assert_eq!(spec.write, WriteMode::Overwrite);
    }

    #[test]
    fn builders_set_the_expected_fields() {
        let spec = MatmulSpec::new(2, 3, 4)
            .left_transposed()
            .right_transposed()
            .accumulate()
            .scaled(0.25)
            .batched(BatchStrides { count: 3, left: 8, right: 12, output: 6 });

        assert_eq!(spec.left, Storage::Transposed);
        assert_eq!(spec.right, Storage::Transposed);
        assert_eq!(spec.write, WriteMode::Accumulate);
        assert_eq!(spec.scale, 0.25);
        assert_eq!(spec.batch.count, 3);
        assert_eq!(spec.batch.left, 8);
        assert_eq!(spec.batch.right, 12);
        assert_eq!(spec.batch.output, 6);
    }

    #[test]
    fn required_len_accounts_for_the_batch_stride() {
        let spec = MatmulSpec::new(2, 3, 4).batched(BatchStrides { count: 3, left: 8, right: 0, output: 6 });

        assert_eq!(spec.required_len(8, 8), 24);
        assert_eq!(spec.required_len(0, 8), 8);
        assert_eq!(spec.required_len(6, 6), 18);
    }

    #[test]
    fn single_batch_stride_describes_one_matrix() {
        assert_eq!(BatchStrides::SINGLE.count, 1);
        assert_eq!(BatchStrides::SINGLE.left, 0);
        assert_eq!(BatchStrides::SINGLE.right, 0);
        assert_eq!(BatchStrides::SINGLE.output, 0);
    }
}
