#![forbid(unsafe_code)]

use numpy::ndarray::Array2;
use numpy::{Element, IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use sketch_spgemm as core;
use std::ops::AddAssign;
use std::sync::Arc;

type PyCsrArrays = (Py<PyAny>, Py<PyAny>, Py<PyAny>);

fn value_error(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}

fn nonnegative_index(name: &str, position: usize, value: i64) -> PyResult<usize> {
    usize::try_from(value)
        .map_err(|_| value_error(format!("{name}[{position}] must be non-negative")))
}

fn map_build_error(error: core::CsrBuildError) -> PyErr {
    match error {
        core::CsrBuildError::ArithmeticOverflow { .. } => {
            PyOverflowError::new_err(error.to_string())
        }
        core::CsrBuildError::RowOutOfBounds { .. }
        | core::CsrBuildError::ColumnOutOfBounds { .. }
        | core::CsrBuildError::OutOfOrder { .. } => value_error(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

fn collect_triplets<T>(
    data: PyReadonlyArray1<'_, T>,
    row_indices: PyReadonlyArray1<'_, i64>,
    column_indices: PyReadonlyArray1<'_, i64>,
) -> PyResult<Vec<(usize, usize, T)>>
where
    T: Element + Copy,
{
    let data = data
        .as_slice()
        .map_err(|_| value_error("data must be a contiguous one-dimensional array"))?;
    let row_indices = row_indices
        .as_slice()
        .map_err(|_| value_error("row_indices must be a contiguous one-dimensional array"))?;
    let column_indices = column_indices
        .as_slice()
        .map_err(|_| value_error("column_indices must be a contiguous one-dimensional array"))?;
    if data.len() != row_indices.len() || data.len() != column_indices.len() {
        return Err(value_error(
            "data, row_indices, and column_indices must have equal lengths",
        ));
    }

    let mut triplets = Vec::with_capacity(data.len());
    for position in 0..data.len() {
        triplets.push((
            nonnegative_index("row_indices", position, row_indices[position])?,
            nonnegative_index("column_indices", position, column_indices[position])?,
            data[position],
        ));
    }
    Ok(triplets)
}

fn csr_from_arrays<T>(
    data: PyReadonlyArray1<'_, T>,
    indices: PyReadonlyArray1<'_, i64>,
    indptr: PyReadonlyArray1<'_, i64>,
    shape: (usize, usize),
) -> PyResult<core::CsrMatrix<T>>
where
    T: Element + Copy + Default + PartialEq,
{
    let data = data
        .as_slice()
        .map_err(|_| value_error("data must be a contiguous one-dimensional array"))?;
    let indices = indices
        .as_slice()
        .map_err(|_| value_error("indices must be a contiguous one-dimensional array"))?;
    let indptr = indptr
        .as_slice()
        .map_err(|_| value_error("indptr must be a contiguous one-dimensional array"))?;

    let (rows, cols) = shape;
    if rows > i64::MAX as usize || cols > i64::MAX as usize {
        return Err(PyOverflowError::new_err(
            "shape dimensions must fit in signed 64-bit integers",
        ));
    }
    if data.len() > i64::MAX as usize {
        return Err(PyOverflowError::new_err(
            "the number of stored values must fit in a signed 64-bit integer",
        ));
    }
    if data.len() != indices.len() {
        return Err(value_error("data and indices must have equal lengths"));
    }
    let expected_indptr = rows
        .checked_add(1)
        .ok_or_else(|| PyOverflowError::new_err("row count is too large"))?;
    if indptr.len() != expected_indptr {
        return Err(value_error(format!(
            "indptr length must equal rows + 1 ({expected_indptr})"
        )));
    }
    if indptr.first().copied() != Some(0) {
        return Err(value_error("indptr must start at zero"));
    }

    let mut row_ptr = Vec::with_capacity(indptr.len());
    let mut previous = 0usize;
    for (position, &pointer) in indptr.iter().enumerate() {
        let pointer = usize::try_from(pointer)
            .map_err(|_| value_error(format!("indptr[{position}] must be non-negative")))?;
        if pointer < previous {
            return Err(value_error("indptr must be monotonically non-decreasing"));
        }
        if pointer > data.len() {
            return Err(value_error("indptr entries cannot exceed nnz"));
        }
        row_ptr.push(pointer);
        previous = pointer;
    }
    if previous != data.len() {
        return Err(value_error("the final indptr entry must equal nnz"));
    }

    let mut col_idx = Vec::with_capacity(indices.len());
    for (position, &column) in indices.iter().enumerate() {
        let column = usize::try_from(column)
            .map_err(|_| value_error(format!("indices[{position}] must be non-negative")))?;
        if column >= cols {
            return Err(value_error(format!(
                "indices[{position}]={column} is outside matrix width {cols}"
            )));
        }
        col_idx.push(column);
    }

    let zero = T::default();
    for row in 0..rows {
        let start = row_ptr[row];
        let end = row_ptr[row + 1];
        for position in start..end {
            if data[position] == zero {
                return Err(value_error(format!(
                    "explicit zero at data[{position}] is not canonical CSR"
                )));
            }
            if position > start && col_idx[position - 1] >= col_idx[position] {
                return Err(value_error(format!(
                    "column indices in row {row} must be strictly increasing"
                )));
            }
        }
    }

    Ok(core::CsrMatrix {
        rows,
        cols,
        row_ptr,
        col_idx,
        values: data.to_vec(),
    })
}

fn extend_builder<T>(
    builder: &mut core::CsrBuilder<T>,
    data: PyReadonlyArray1<'_, T>,
    row_indices: PyReadonlyArray1<'_, i64>,
    column_indices: PyReadonlyArray1<'_, i64>,
) -> PyResult<()>
where
    T: Element + Copy + Default + PartialEq + core::CheckedAddScalar,
{
    let triplets = collect_triplets(data, row_indices, column_indices)?;
    builder.try_extend(triplets).map_err(map_build_error)?;
    Ok(())
}

fn dtype_error() -> PyErr {
    PyTypeError::new_err("data dtype must be one of: int32, int64, float32, float64")
}

fn parse_dtype(dtype: &str) -> PyResult<&'static str> {
    match dtype {
        "int32" | "i32" => Ok("int32"),
        "int64" | "i64" => Ok("int64"),
        "float32" | "f32" => Ok("float32"),
        "float64" | "f64" => Ok("float64"),
        _ => Err(dtype_error()),
    }
}

fn finite_nonnegative(name: &str, value: f64) -> PyResult<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(value_error(format!(
            "{name} must be finite and non-negative"
        )))
    }
}

fn finite_positive(name: &str, value: f64) -> PyResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(value_error(format!("{name} must be finite and positive")))
    }
}

fn parse_rectangular_policy(value: &str) -> PyResult<core::RectangularPolicy> {
    match value {
        "auto" => Ok(core::RectangularPolicy::Auto),
        "dense" => Ok(core::RectangularPolicy::Dense),
        "sparse_left" => Ok(core::RectangularPolicy::SparseLeft),
        "sparse_right" => Ok(core::RectangularPolicy::SparseRight),
        "sparse_sparse" => Ok(core::RectangularPolicy::SparseSparse),
        _ => Err(value_error(
            "rectangular_policy must be one of: auto, dense, sparse_left, sparse_right, sparse_sparse",
        )),
    }
}

fn rectangular_policy_name(value: core::RectangularPolicy) -> &'static str {
    match value {
        core::RectangularPolicy::Auto => "auto",
        core::RectangularPolicy::Dense => "dense",
        core::RectangularPolicy::SparseLeft => "sparse_left",
        core::RectangularPolicy::SparseRight => "sparse_right",
        core::RectangularPolicy::SparseSparse => "sparse_sparse",
    }
}

fn rectangular_kernel_name(value: core::RectangularKernel) -> &'static str {
    match value {
        core::RectangularKernel::DenseBlocked => "dense_blocked",
        core::RectangularKernel::SparseLeft => "sparse_left",
        core::RectangularKernel::SparseRight => "sparse_right",
        core::RectangularKernel::SparseSparse => "sparse_sparse",
    }
}

#[derive(Clone)]
enum CsrStorage {
    I32(Arc<core::CsrMatrix<i32>>),
    I64(Arc<core::CsrMatrix<i64>>),
    F32(Arc<core::CsrMatrix<f32>>),
    F64(Arc<core::CsrMatrix<f64>>),
}

impl CsrStorage {
    fn rows(&self) -> usize {
        match self {
            Self::I32(matrix) => matrix.rows,
            Self::I64(matrix) => matrix.rows,
            Self::F32(matrix) => matrix.rows,
            Self::F64(matrix) => matrix.rows,
        }
    }

    fn cols(&self) -> usize {
        match self {
            Self::I32(matrix) => matrix.cols,
            Self::I64(matrix) => matrix.cols,
            Self::F32(matrix) => matrix.cols,
            Self::F64(matrix) => matrix.cols,
        }
    }

    fn nnz(&self) -> usize {
        match self {
            Self::I32(matrix) => matrix.nnz(),
            Self::I64(matrix) => matrix.nnz(),
            Self::F32(matrix) => matrix.nnz(),
            Self::F64(matrix) => matrix.nnz(),
        }
    }

    fn dtype(&self) -> &'static str {
        match self {
            Self::I32(_) => "int32",
            Self::I64(_) => "int64",
            Self::F32(_) => "float32",
            Self::F64(_) => "float64",
        }
    }
}

fn arrays_to_python<T>(py: Python<'_>, matrix: &core::CsrMatrix<T>) -> PyCsrArrays
where
    T: Element + Clone,
{
    let data = matrix.values.clone().into_pyarray(py).into_any().unbind();
    let indices = matrix
        .col_idx
        .iter()
        .map(|&value| value as i64)
        .collect::<Vec<_>>()
        .into_pyarray(py)
        .into_any()
        .unbind();
    let indptr = matrix
        .row_ptr
        .iter()
        .map(|&value| value as i64)
        .collect::<Vec<_>>()
        .into_pyarray(py)
        .into_any()
        .unbind();
    (data, indices, indptr)
}

fn dense_to_python<T>(py: Python<'_>, matrix: &core::CsrMatrix<T>) -> PyResult<Py<PyAny>>
where
    T: Element + Copy + Default + PartialEq + AddAssign,
{
    let dense = matrix.to_dense();
    let array = Array2::from_shape_vec((dense.rows, dense.cols), dense.data)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py).into_any().unbind())
}

/// Immutable canonical CSR matrix with a NumPy-compatible numeric dtype.
#[pyclass(name = "CsrMatrix", frozen, module = "sketch_spgemm._sketch_spgemm")]
struct PyCsrMatrix {
    inner: CsrStorage,
}

impl PyCsrMatrix {
    fn from_i32(inner: core::CsrMatrix<i32>) -> Self {
        Self {
            inner: CsrStorage::I32(Arc::new(inner)),
        }
    }

    fn from_i64(inner: core::CsrMatrix<i64>) -> Self {
        Self {
            inner: CsrStorage::I64(Arc::new(inner)),
        }
    }

    fn from_f32(inner: core::CsrMatrix<f32>) -> Self {
        Self {
            inner: CsrStorage::F32(Arc::new(inner)),
        }
    }

    fn from_f64(inner: core::CsrMatrix<f64>) -> Self {
        Self {
            inner: CsrStorage::F64(Arc::new(inner)),
        }
    }
}

#[pymethods]
impl PyCsrMatrix {
    #[new]
    fn new(
        data: &Bound<'_, PyAny>,
        indices: PyReadonlyArray1<'_, i64>,
        indptr: PyReadonlyArray1<'_, i64>,
        shape: (usize, usize),
    ) -> PyResult<Self> {
        if let Ok(values) = data.extract::<PyReadonlyArray1<'_, i32>>() {
            return csr_from_arrays(values, indices, indptr, shape).map(Self::from_i32);
        }
        if let Ok(values) = data.extract::<PyReadonlyArray1<'_, i64>>() {
            return csr_from_arrays(values, indices, indptr, shape).map(Self::from_i64);
        }
        if let Ok(values) = data.extract::<PyReadonlyArray1<'_, f32>>() {
            return csr_from_arrays(values, indices, indptr, shape).map(Self::from_f32);
        }
        if let Ok(values) = data.extract::<PyReadonlyArray1<'_, f64>>() {
            return csr_from_arrays(values, indices, indptr, shape).map(Self::from_f64);
        }
        Err(dtype_error())
    }

    /// Build a canonical matrix from COO-style triplets.
    #[staticmethod]
    #[pyo3(signature = (data, row_indices, column_indices, shape, *, sorted=false))]
    fn from_triplets(
        data: &Bound<'_, PyAny>,
        row_indices: PyReadonlyArray1<'_, i64>,
        column_indices: PyReadonlyArray1<'_, i64>,
        shape: (usize, usize),
        sorted: bool,
    ) -> PyResult<Self> {
        let (rows, cols) = shape;
        if rows > i64::MAX as usize || cols > i64::MAX as usize {
            return Err(PyOverflowError::new_err(
                "shape dimensions must fit in signed 64-bit integers",
            ));
        }
        macro_rules! build {
            ($scalar:ty, $constructor:ident) => {
                if let Ok(values) = data.extract::<PyReadonlyArray1<'_, $scalar>>() {
                    let triplets = collect_triplets(values, row_indices, column_indices)?;
                    let matrix = if sorted {
                        core::CsrMatrix::try_from_sorted_triplets(rows, cols, triplets)
                    } else {
                        core::CsrMatrix::try_from_triplets(rows, cols, &triplets)
                    }
                    .map_err(map_build_error)?;
                    return Ok(Self::$constructor(matrix));
                }
            };
        }
        build!(i32, from_i32);
        build!(i64, from_i64);
        build!(f32, from_f32);
        build!(f64, from_f64);
        Err(dtype_error())
    }

    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows()
    }

    #[getter]
    fn cols(&self) -> usize {
        self.inner.cols()
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.rows(), self.inner.cols())
    }

    #[getter]
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype()
    }

    fn to_arrays(&self, py: Python<'_>) -> PyCsrArrays {
        match &self.inner {
            CsrStorage::I32(matrix) => arrays_to_python(py, matrix),
            CsrStorage::I64(matrix) => arrays_to_python(py, matrix),
            CsrStorage::F32(matrix) => arrays_to_python(py, matrix),
            CsrStorage::F64(matrix) => arrays_to_python(py, matrix),
        }
    }

    fn to_dense(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner {
            CsrStorage::I32(matrix) => dense_to_python(py, matrix),
            CsrStorage::I64(matrix) => dense_to_python(py, matrix),
            CsrStorage::F32(matrix) => dense_to_python(py, matrix),
            CsrStorage::F64(matrix) => dense_to_python(py, matrix),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CsrMatrix(shape=({}, {}), nnz={}, dtype='{}')",
            self.inner.rows(),
            self.inner.cols(),
            self.inner.nnz(),
            self.inner.dtype()
        )
    }
}

enum CsrBuilderStorage {
    I32(core::CsrBuilder<i32>),
    I64(core::CsrBuilder<i64>),
    F32(core::CsrBuilder<f32>),
    F64(core::CsrBuilder<f64>),
}

impl CsrBuilderStorage {
    fn finish(self) -> PyCsrMatrix {
        match self {
            Self::I32(builder) => PyCsrMatrix::from_i32(builder.finish()),
            Self::I64(builder) => PyCsrMatrix::from_i64(builder.finish()),
            Self::F32(builder) => PyCsrMatrix::from_f32(builder.finish()),
            Self::F64(builder) => PyCsrMatrix::from_f64(builder.finish()),
        }
    }
}

/// Incremental canonical CSR construction from row-major sorted triplets.
#[pyclass(name = "CsrBuilder", module = "sketch_spgemm._sketch_spgemm")]
struct PyCsrBuilder {
    inner: Option<CsrBuilderStorage>,
    rows: usize,
    cols: usize,
    dtype: &'static str,
}

#[pymethods]
impl PyCsrBuilder {
    #[new]
    #[pyo3(signature = (rows, cols, capacity=0, *, dtype="int64"))]
    fn new(rows: usize, cols: usize, capacity: usize, dtype: &str) -> PyResult<Self> {
        if rows > i64::MAX as usize || cols > i64::MAX as usize {
            return Err(PyOverflowError::new_err(
                "shape dimensions must fit in signed 64-bit integers",
            ));
        }
        let dtype = parse_dtype(dtype)?;
        let inner = match dtype {
            "int32" => {
                CsrBuilderStorage::I32(core::CsrBuilder::with_capacity(rows, cols, capacity))
            }
            "int64" => {
                CsrBuilderStorage::I64(core::CsrBuilder::with_capacity(rows, cols, capacity))
            }
            "float32" => {
                CsrBuilderStorage::F32(core::CsrBuilder::with_capacity(rows, cols, capacity))
            }
            "float64" => {
                CsrBuilderStorage::F64(core::CsrBuilder::with_capacity(rows, cols, capacity))
            }
            _ => unreachable!(),
        };
        Ok(Self {
            inner: Some(inner),
            rows,
            cols,
            dtype,
        })
    }

    #[getter]
    fn rows(&self) -> usize {
        self.rows
    }

    #[getter]
    fn cols(&self) -> usize {
        self.cols
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.dtype
    }

    /// Push one row-major sorted triplet.
    fn push(&mut self, row: i64, column: i64, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let row = nonnegative_index("row", 0, row)?;
        let column = nonnegative_index("column", 0, column)?;
        match self.active()? {
            CsrBuilderStorage::I32(builder) => builder.try_push(row, column, value.extract()?),
            CsrBuilderStorage::I64(builder) => builder.try_push(row, column, value.extract()?),
            CsrBuilderStorage::F32(builder) => builder.try_push(row, column, value.extract()?),
            CsrBuilderStorage::F64(builder) => builder.try_push(row, column, value.extract()?),
        }
        .map_err(map_build_error)
    }

    /// Extend the builder with one contiguous NumPy chunk.
    fn extend(
        &mut self,
        data: &Bound<'_, PyAny>,
        row_indices: PyReadonlyArray1<'_, i64>,
        column_indices: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<()> {
        // Validate the lifecycle even for an empty chunk. Keeping the builder
        // reference also avoids repeating the check for every coordinate.
        match self.active()? {
            CsrBuilderStorage::I32(builder) => extend_builder(
                builder,
                data.extract().map_err(|_| dtype_error())?,
                row_indices,
                column_indices,
            ),
            CsrBuilderStorage::I64(builder) => extend_builder(
                builder,
                data.extract().map_err(|_| dtype_error())?,
                row_indices,
                column_indices,
            ),
            CsrBuilderStorage::F32(builder) => extend_builder(
                builder,
                data.extract().map_err(|_| dtype_error())?,
                row_indices,
                column_indices,
            ),
            CsrBuilderStorage::F64(builder) => extend_builder(
                builder,
                data.extract().map_err(|_| dtype_error())?,
                row_indices,
                column_indices,
            ),
        }
    }

    /// Consume the builder and return its canonical matrix.
    fn finish(&mut self) -> PyResult<PyCsrMatrix> {
        let builder = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("CsrBuilder has already been finished"))?;
        Ok(builder.finish())
    }

    fn __repr__(&self) -> String {
        let state = if self.inner.is_some() {
            "active"
        } else {
            "finished"
        };
        format!(
            "CsrBuilder(shape=({}, {}), dtype='{}', state='{state}')",
            self.rows, self.cols, self.dtype
        )
    }
}

impl PyCsrBuilder {
    fn active(&mut self) -> PyResult<&mut CsrBuilderStorage> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("CsrBuilder has already been finished"))
    }
}

/// Moment/IBLT sparse-recovery settings.
#[pyclass(
    name = "MomentConfig",
    frozen,
    skip_from_py_object,
    module = "sketch_spgemm._sketch_spgemm"
)]
#[derive(Clone)]
struct PyMomentConfig {
    inner: core::MomentConfig,
}

#[pymethods]
impl PyMomentConfig {
    #[new]
    #[pyo3(signature = (*, degree=3, oversampling=3.0, seed=0x4D4F_4D45_4E54_0001, identity_fallback=true, guaranteed_correction=true))]
    fn new(
        degree: usize,
        oversampling: f64,
        seed: u64,
        identity_fallback: bool,
        guaranteed_correction: bool,
    ) -> PyResult<Self> {
        if degree == 0 {
            return Err(value_error("degree must be positive"));
        }
        finite_positive("oversampling", oversampling)?;
        Ok(Self {
            inner: core::MomentConfig {
                degree,
                oversampling,
                seed,
                identity_fallback,
                guaranteed_correction,
            },
        })
    }

    #[getter]
    fn degree(&self) -> usize {
        self.inner.degree
    }
    #[getter]
    fn oversampling(&self) -> f64 {
        self.inner.oversampling
    }
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }
    #[getter]
    fn identity_fallback(&self) -> bool {
        self.inner.identity_fallback
    }
    #[getter]
    fn guaranteed_correction(&self) -> bool {
        self.inner.guaranteed_correction
    }
}

/// Residual-fingerprint settings.
#[pyclass(
    name = "FingerprintConfig",
    frozen,
    skip_from_py_object,
    module = "sketch_spgemm._sketch_spgemm"
)]
#[derive(Clone)]
struct PyFingerprintConfig {
    inner: core::FingerprintConfig,
}

#[pymethods]
impl PyFingerprintConfig {
    #[new]
    #[pyo3(signature = (*, lanes=3, seed=0))]
    fn new(lanes: usize, seed: u64) -> PyResult<Self> {
        if lanes == 0 {
            return Err(value_error("lanes must be positive"));
        }
        Ok(Self {
            inner: core::FingerprintConfig { lanes, seed },
        })
    }

    #[getter]
    fn lanes(&self) -> usize {
        self.inner.lanes
    }
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }
}

/// Automatic workload-selection and multiplication settings.
#[pyclass(
    name = "AutoSpGemmConfig",
    frozen,
    skip_from_py_object,
    module = "sketch_spgemm._sketch_spgemm"
)]
#[derive(Clone)]
struct PyAutoSpGemmConfig {
    inner: core::AutoSpGemmConfig,
}

#[pymethods]
impl PyAutoSpGemmConfig {
    #[new]
    #[pyo3(signature = (*, sample_rows=8, initial_sample_rows=2, min_structural_amplification=64.0, k_safety_factor=1.5, min_estimated_rho=256.0, max_estimated_avg_column_nnz=64.0, max_estimated_output_density=0.10, max_moment_row_ratio=0.80, rectangular_policy="auto", moment=None, fingerprint=None, exact_dense_cell_limit=16_000_000, exact_fallback=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        sample_rows: usize,
        initial_sample_rows: usize,
        min_structural_amplification: f64,
        k_safety_factor: f64,
        min_estimated_rho: f64,
        max_estimated_avg_column_nnz: f64,
        max_estimated_output_density: f64,
        max_moment_row_ratio: f64,
        rectangular_policy: &str,
        moment: Option<PyRef<'_, PyMomentConfig>>,
        fingerprint: Option<PyRef<'_, PyFingerprintConfig>>,
        exact_dense_cell_limit: usize,
        exact_fallback: bool,
    ) -> PyResult<Self> {
        if sample_rows == 0 {
            return Err(value_error("sample_rows must be positive"));
        }
        if initial_sample_rows == 0 || initial_sample_rows > sample_rows {
            return Err(value_error(
                "initial_sample_rows must be positive and no greater than sample_rows",
            ));
        }
        finite_nonnegative("min_structural_amplification", min_structural_amplification)?;
        if !k_safety_factor.is_finite() || k_safety_factor < 1.0 {
            return Err(value_error(
                "k_safety_factor must be finite and at least 1.0",
            ));
        }
        finite_nonnegative("min_estimated_rho", min_estimated_rho)?;
        finite_nonnegative("max_estimated_avg_column_nnz", max_estimated_avg_column_nnz)?;
        for (name, value) in [
            ("max_estimated_output_density", max_estimated_output_density),
            ("max_moment_row_ratio", max_moment_row_ratio),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(value_error(format!("{name} must be between 0.0 and 1.0")));
            }
        }

        Ok(Self {
            inner: core::AutoSpGemmConfig {
                sample_rows,
                initial_sample_rows,
                min_structural_amplification,
                k_safety_factor,
                min_estimated_rho,
                max_estimated_avg_column_nnz,
                max_estimated_output_density,
                max_moment_row_ratio,
                rectangular_policy: parse_rectangular_policy(rectangular_policy)?,
                moment: moment
                    .as_ref()
                    .map(|value| value.inner.clone())
                    .unwrap_or_default(),
                fingerprint: fingerprint
                    .as_ref()
                    .map(|value| value.inner)
                    .unwrap_or_default(),
                exact_dense_cell_limit,
                exact_fallback,
            },
        })
    }

    #[getter]
    fn sample_rows(&self) -> usize {
        self.inner.sample_rows
    }
    #[getter]
    fn initial_sample_rows(&self) -> usize {
        self.inner.initial_sample_rows
    }
    #[getter]
    fn min_structural_amplification(&self) -> f64 {
        self.inner.min_structural_amplification
    }
    #[getter]
    fn k_safety_factor(&self) -> f64 {
        self.inner.k_safety_factor
    }
    #[getter]
    fn min_estimated_rho(&self) -> f64 {
        self.inner.min_estimated_rho
    }
    #[getter]
    fn max_estimated_avg_column_nnz(&self) -> f64 {
        self.inner.max_estimated_avg_column_nnz
    }
    #[getter]
    fn max_estimated_output_density(&self) -> f64 {
        self.inner.max_estimated_output_density
    }
    #[getter]
    fn max_moment_row_ratio(&self) -> f64 {
        self.inner.max_moment_row_ratio
    }
    #[getter]
    fn rectangular_policy(&self) -> &'static str {
        rectangular_policy_name(self.inner.rectangular_policy)
    }
    #[getter]
    fn moment(&self) -> PyMomentConfig {
        PyMomentConfig {
            inner: self.inner.moment.clone(),
        }
    }
    #[getter]
    fn fingerprint(&self) -> PyFingerprintConfig {
        PyFingerprintConfig {
            inner: self.inner.fingerprint,
        }
    }
    #[getter]
    fn exact_dense_cell_limit(&self) -> usize {
        self.inner.exact_dense_cell_limit
    }
    #[getter]
    fn exact_fallback(&self) -> bool {
        self.inner.exact_fallback
    }
}

/// Workload analysis returned by `analyze_workload`.
#[pyclass(
    name = "WorkloadEstimate",
    frozen,
    get_all,
    skip_from_py_object,
    module = "sketch_spgemm._sketch_spgemm"
)]
#[derive(Clone)]
struct PyWorkloadEstimate {
    candidate_products: u128,
    structural_amplification: f64,
    sampled_rows: usize,
    sampled_output_nnz: usize,
    sampled_unique_columns: usize,
    estimated_output_nnz: usize,
    estimated_active_columns: usize,
    estimated_avg_nnz_per_active_column: f64,
    estimated_output_density: f64,
    estimated_rho: f64,
    target_q: usize,
    estimated_moment_rows: usize,
    choose_sketch: bool,
    reason: String,
    structural_prefilter_only: bool,
}

impl From<core::WorkloadEstimate> for PyWorkloadEstimate {
    fn from(value: core::WorkloadEstimate) -> Self {
        Self {
            candidate_products: value.candidate_products,
            structural_amplification: value.structural_amplification,
            sampled_rows: value.sampled_rows,
            sampled_output_nnz: value.sampled_output_nnz,
            sampled_unique_columns: value.sampled_unique_columns,
            estimated_output_nnz: value.estimated_output_nnz,
            estimated_active_columns: value.estimated_active_columns,
            estimated_avg_nnz_per_active_column: value.estimated_avg_nnz_per_active_column,
            estimated_output_density: value.estimated_output_density,
            estimated_rho: value.estimated_rho,
            target_q: value.target_q,
            estimated_moment_rows: value.estimated_moment_rows,
            choose_sketch: value.choose_sketch,
            reason: value.reason,
            structural_prefilter_only: value.structural_prefilter_only,
        }
    }
}

/// Wall-clock timings in seconds for automatic multiplication.
#[pyclass(
    name = "AutoTimingStats",
    frozen,
    get_all,
    skip_from_py_object,
    module = "sketch_spgemm._sketch_spgemm"
)]
#[derive(Clone)]
struct PyAutoTimingStats {
    total: f64,
    analysis_total: f64,
    candidate_count: f64,
    row_sampling: f64,
    nested: f64,
    fingerprint_setup: f64,
    fingerprint_checks: f64,
    exact: f64,
}

impl From<core::AutoTimingStats> for PyAutoTimingStats {
    fn from(value: core::AutoTimingStats) -> Self {
        Self {
            total: value.total.as_secs_f64(),
            analysis_total: value.analysis_total.as_secs_f64(),
            candidate_count: value.candidate_count.as_secs_f64(),
            row_sampling: value.row_sampling.as_secs_f64(),
            nested: value.nested.as_secs_f64(),
            fingerprint_setup: value.fingerprint_setup.as_secs_f64(),
            fingerprint_checks: value.fingerprint_checks.as_secs_f64(),
            exact: value.exact.as_secs_f64(),
        }
    }
}

/// Residual-fingerprint counters.
#[pyclass(
    name = "FingerprintStats",
    frozen,
    get_all,
    skip_from_py_object,
    module = "sketch_spgemm._sketch_spgemm"
)]
#[derive(Clone)]
struct PyFingerprintStats {
    checks: usize,
    passes: usize,
    failures: usize,
    lanes: usize,
    seed: u64,
}

impl From<core::FingerprintStats> for PyFingerprintStats {
    fn from(value: core::FingerprintStats) -> Self {
        Self {
            checks: value.checks,
            passes: value.passes,
            failures: value.failures,
            lanes: value.lanes,
            seed: value.seed,
        }
    }
}

/// Recovery schedule parameters for one round.
#[pyclass(
    name = "RoundParams",
    frozen,
    get_all,
    skip_from_py_object,
    module = "sketch_spgemm._sketch_spgemm"
)]
#[derive(Clone)]
struct PyRoundParams {
    i: usize,
    t: usize,
    q: usize,
    p: usize,
    q_accumulated_after: usize,
}

impl From<core::RoundParams> for PyRoundParams {
    fn from(value: core::RoundParams) -> Self {
        Self {
            i: value.i,
            t: value.t,
            q: value.q,
            p: value.p,
            q_accumulated_after: value.q_accumulated_after,
        }
    }
}

/// Statistics for an exact residual-correction pass.
#[pyclass(
    name = "CorrectionPassStats",
    frozen,
    get_all,
    skip_from_py_object,
    module = "sketch_spgemm._sketch_spgemm"
)]
#[derive(Clone)]
struct PyCorrectionPassStats {
    residual_columns: usize,
    residual_nnz: usize,
    elapsed: f64,
}

impl From<core::CorrectionPassStats> for PyCorrectionPassStats {
    fn from(value: core::CorrectionPassStats) -> Self {
        Self {
            residual_columns: value.residual_columns,
            residual_nnz: value.residual_nnz,
            elapsed: value.elapsed.as_secs_f64(),
        }
    }
}

/// Statistics for a rectangular multiplication kernel.
#[pyclass(
    name = "RectangularStats",
    frozen,
    get_all,
    skip_from_py_object,
    module = "sketch_spgemm._sketch_spgemm"
)]
#[derive(Clone)]
struct PyRectangularStats {
    kernel: Option<String>,
    a_nnz: usize,
    b_nnz: usize,
    a_density: f64,
    b_density: f64,
    dense_ops: u128,
    dense_estimated_cost: u128,
    sparse_left_estimated_cost: u128,
    sparse_right_estimated_cost: u128,
    sparse_sparse_estimated_cost: u128,
    sparse_candidate_products: u128,
    scalar_multiplications: u128,
}

impl From<core::RectangularStats> for PyRectangularStats {
    fn from(value: core::RectangularStats) -> Self {
        Self {
            kernel: value
                .kernel
                .map(|item| rectangular_kernel_name(item).to_string()),
            a_nnz: value.a_nnz,
            b_nnz: value.b_nnz,
            a_density: value.a_density,
            b_density: value.b_density,
            dense_ops: value.dense_ops,
            dense_estimated_cost: value.dense_estimated_cost,
            sparse_left_estimated_cost: value.sparse_left_estimated_cost,
            sparse_right_estimated_cost: value.sparse_right_estimated_cost,
            sparse_sparse_estimated_cost: value.sparse_sparse_estimated_cost,
            sparse_candidate_products: value.sparse_candidate_products,
            scalar_multiplications: value.scalar_multiplications,
        }
    }
}

/// Detailed statistics for one nested recovery round.
#[pyclass(
    name = "NestedRoundStats",
    frozen,
    get_all,
    skip_from_py_object,
    module = "sketch_spgemm._sketch_spgemm"
)]
#[derive(Clone)]
struct PyNestedRoundStats {
    params: PyRoundParams,
    h_rows: usize,
    g_rows: usize,
    h_kind: String,
    g_kind: String,
    w_nnz: usize,
    outer_recovered: usize,
    inner_updates: usize,
    d_nnz_after: usize,
    left_time: f64,
    right_time: f64,
    rectangular_time: f64,
    rectangular_kernel: String,
    rectangular_scalar_multiplications: u128,
    rectangular_candidate_products: u128,
    ha_density: f64,
    bgt_density: f64,
    h_matrix_cache_hit: bool,
    g_matrix_cache_hit: bool,
    ha_cache_hit: bool,
    bgt_cache_hit: bool,
    product_cache_hit: bool,
    residual_cache_hit: bool,
    masked_residual: bool,
    active_mask_columns: usize,
    scheduler_target_q: usize,
    residual_measure_time: f64,
    decode_time: f64,
}

impl From<core::NestedRoundStats> for PyNestedRoundStats {
    fn from(value: core::NestedRoundStats) -> Self {
        Self {
            params: value.params.into(),
            h_rows: value.h_rows,
            g_rows: value.g_rows,
            h_kind: value.h_kind.to_string(),
            g_kind: value.g_kind.to_string(),
            w_nnz: value.w_nnz,
            outer_recovered: value.outer_recovered,
            inner_updates: value.inner_updates,
            d_nnz_after: value.d_nnz_after,
            left_time: value.left_time.as_secs_f64(),
            right_time: value.right_time.as_secs_f64(),
            rectangular_time: value.rectangular_time.as_secs_f64(),
            rectangular_kernel: rectangular_kernel_name(value.rectangular_kernel).to_string(),
            rectangular_scalar_multiplications: value.rectangular_scalar_multiplications,
            rectangular_candidate_products: value.rectangular_candidate_products,
            ha_density: value.ha_density,
            bgt_density: value.bgt_density,
            h_matrix_cache_hit: value.h_matrix_cache_hit,
            g_matrix_cache_hit: value.g_matrix_cache_hit,
            ha_cache_hit: value.ha_cache_hit,
            bgt_cache_hit: value.bgt_cache_hit,
            product_cache_hit: value.product_cache_hit,
            residual_cache_hit: value.residual_cache_hit,
            masked_residual: value.masked_residual,
            active_mask_columns: value.active_mask_columns,
            scheduler_target_q: value.scheduler_target_q,
            residual_measure_time: value.residual_measure_time.as_secs_f64(),
            decode_time: value.decode_time.as_secs_f64(),
        }
    }
}

/// Detailed statistics for nested sparse recovery.
#[pyclass(
    name = "NestedSpGemmStats",
    frozen,
    get_all,
    skip_from_py_object,
    module = "sketch_spgemm._sketch_spgemm"
)]
#[derive(Clone)]
struct PyNestedSpGemmStats {
    rounds: Vec<PyNestedRoundStats>,
    correction_pass: Option<PyCorrectionPassStats>,
    terminated_early: bool,
    termination_reason: Option<String>,
    scheduler_skipped_rounds: usize,
    fingerprint: Option<PyFingerprintStats>,
    fingerprint_setup_time: f64,
    fingerprint_check_time: f64,
    fingerprint_verified: bool,
    deterministic_verified: bool,
}

impl From<core::NestedSpGemmStats> for PyNestedSpGemmStats {
    fn from(value: core::NestedSpGemmStats) -> Self {
        Self {
            rounds: value.rounds.into_iter().map(Into::into).collect(),
            correction_pass: value.correction_pass.map(Into::into),
            terminated_early: value.terminated_early,
            termination_reason: value.termination_reason,
            scheduler_skipped_rounds: value.scheduler_skipped_rounds,
            fingerprint: value.fingerprint.map(Into::into),
            fingerprint_setup_time: value.fingerprint_setup_time.as_secs_f64(),
            fingerprint_check_time: value.fingerprint_check_time.as_secs_f64(),
            fingerprint_verified: value.fingerprint_verified,
            deterministic_verified: value.deterministic_verified,
        }
    }
}

/// Complete statistics returned by `auto_spgemm`.
#[pyclass(
    name = "AutoSpGemmStats",
    frozen,
    get_all,
    skip_from_py_object,
    module = "sketch_spgemm._sketch_spgemm"
)]
#[derive(Clone)]
struct PyAutoSpGemmStats {
    choice: String,
    estimate: PyWorkloadEstimate,
    k_bound_used: usize,
    nested: Option<PyNestedSpGemmStats>,
    exact_stats: Option<PyRectangularStats>,
    exact_method: Option<String>,
    fallback_reason: Option<String>,
    timing: PyAutoTimingStats,
}

impl From<core::AutoSpGemmStats> for PyAutoSpGemmStats {
    fn from(value: core::AutoSpGemmStats) -> Self {
        let choice = match value.choice {
            core::AutoChoice::Exact => "exact",
            core::AutoChoice::Sketch => "sketch",
            core::AutoChoice::ExactFallback => "exact_fallback",
        };
        let exact_method = value.exact_method.map(|method| match method {
            core::ExactMethod::AdaptiveDense => "adaptive_dense".to_string(),
            core::ExactMethod::HashSparse => "hash_sparse".to_string(),
        });
        Self {
            choice: choice.to_string(),
            estimate: value.estimate.into(),
            k_bound_used: value.k_bound_used,
            nested: value.nested.map(Into::into),
            exact_stats: value.exact_stats.map(Into::into),
            exact_method,
            fallback_reason: value.fallback_reason,
            timing: value.timing.into(),
        }
    }
}

/// Statistics returned by exact checked sparse multiplication.
#[pyclass(name = "SpGemmStats", frozen, module = "sketch_spgemm._sketch_spgemm")]
struct PySpGemmStats {
    #[pyo3(get)]
    candidate_products: u128,
}

impl From<core::SpGemmStats> for PySpGemmStats {
    fn from(value: core::SpGemmStats) -> Self {
        Self {
            candidate_products: value.candidate_products,
        }
    }
}

fn map_core_error(error: core::SpGemmError) -> PyErr {
    match error {
        core::SpGemmError::DimensionMismatch { .. } => value_error(error.to_string()),
        core::SpGemmError::IndexOverflow { .. } | core::SpGemmError::ArithmeticOverflow { .. } => {
            PyOverflowError::new_err(error.to_string())
        }
        core::SpGemmError::NonCsrStorage { .. } | core::SpGemmError::InvalidOutputStructure(_) => {
            PyRuntimeError::new_err(error.to_string())
        }
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

#[pyfunction]
/// Multiply two CSR matrices with overflow-detecting exact arithmetic.
fn checked_spgemm(
    py: Python<'_>,
    left: PyRef<'_, PyCsrMatrix>,
    right: PyRef<'_, PyCsrMatrix>,
) -> PyResult<(PyCsrMatrix, PySpGemmStats)> {
    match (left.inner.clone(), right.inner.clone()) {
        (CsrStorage::I32(left), CsrStorage::I32(right)) => {
            let (product, stats) = py
                .detach(move || core::try_spgemm_checked(left.as_ref(), right.as_ref()))
                .map_err(map_core_error)?;
            Ok((PyCsrMatrix::from_i32(product), stats.into()))
        }
        (CsrStorage::I64(left), CsrStorage::I64(right)) => {
            let (product, stats) = py
                .detach(move || core::try_spgemm_checked(left.as_ref(), right.as_ref()))
                .map_err(map_core_error)?;
            Ok((PyCsrMatrix::from_i64(product), stats.into()))
        }
        (CsrStorage::F32(left), CsrStorage::F32(right)) => {
            let (product, stats) = py
                .detach(move || core::try_spgemm_checked(left.as_ref(), right.as_ref()))
                .map_err(map_core_error)?;
            Ok((PyCsrMatrix::from_f32(product), stats.into()))
        }
        (CsrStorage::F64(left), CsrStorage::F64(right)) => {
            let (product, stats) = py
                .detach(move || core::try_spgemm_checked(left.as_ref(), right.as_ref()))
                .map_err(map_core_error)?;
            Ok((PyCsrMatrix::from_f64(product), stats.into()))
        }
        _ => Err(PyTypeError::new_err(
            "left and right matrices must have the same dtype",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (left, right, config=None))]
/// Multiply two canonical CSR matrices and return the product and execution statistics.
fn auto_spgemm(
    py: Python<'_>,
    left: PyRef<'_, PyCsrMatrix>,
    right: PyRef<'_, PyCsrMatrix>,
    config: Option<PyRef<'_, PyAutoSpGemmConfig>>,
) -> PyResult<(PyCsrMatrix, PyAutoSpGemmStats)> {
    let left = match &left.inner {
        CsrStorage::I64(matrix) => Arc::clone(matrix),
        _ => {
            return Err(PyTypeError::new_err(
                "auto_spgemm requires int64 matrices; use checked_spgemm for other dtypes",
            ));
        }
    };
    let right = match &right.inner {
        CsrStorage::I64(matrix) => Arc::clone(matrix),
        _ => {
            return Err(PyTypeError::new_err(
                "auto_spgemm requires int64 matrices; use checked_spgemm for other dtypes",
            ));
        }
    };
    let config = config
        .as_ref()
        .map(|value| value.inner.clone())
        .unwrap_or_default();
    let (product, stats) = py
        .detach(move || core::try_auto_spgemm(left.as_ref(), right.as_ref(), config))
        .map_err(map_core_error)?;
    Ok((PyCsrMatrix::from_i64(product), stats.into()))
}

#[pyfunction]
#[pyo3(signature = (left, right, config=None))]
/// Analyze a multiplication workload without computing the complete product.
fn analyze_workload(
    py: Python<'_>,
    left: PyRef<'_, PyCsrMatrix>,
    right: PyRef<'_, PyCsrMatrix>,
    config: Option<PyRef<'_, PyAutoSpGemmConfig>>,
) -> PyResult<PyWorkloadEstimate> {
    let left = match &left.inner {
        CsrStorage::I64(matrix) => Arc::clone(matrix),
        _ => {
            return Err(PyTypeError::new_err(
                "analyze_workload requires int64 matrices",
            ))
        }
    };
    let right = match &right.inner {
        CsrStorage::I64(matrix) => Arc::clone(matrix),
        _ => {
            return Err(PyTypeError::new_err(
                "analyze_workload requires int64 matrices",
            ))
        }
    };
    let config = config
        .as_ref()
        .map(|value| value.inner.clone())
        .unwrap_or_default();
    py.detach(move || core::try_analyze_workload(left.as_ref(), right.as_ref(), &config))
        .map(Into::into)
        .map_err(map_core_error)
}

#[pymodule]
#[pyo3(name = "_sketch_spgemm")]
fn py_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCsrMatrix>()?;
    m.add_class::<PyCsrBuilder>()?;
    m.add_class::<PyMomentConfig>()?;
    m.add_class::<PyFingerprintConfig>()?;
    m.add_class::<PyAutoSpGemmConfig>()?;
    m.add_class::<PyWorkloadEstimate>()?;
    m.add_class::<PyAutoTimingStats>()?;
    m.add_class::<PyFingerprintStats>()?;
    m.add_class::<PyRoundParams>()?;
    m.add_class::<PyCorrectionPassStats>()?;
    m.add_class::<PyRectangularStats>()?;
    m.add_class::<PyNestedRoundStats>()?;
    m.add_class::<PyNestedSpGemmStats>()?;
    m.add_class::<PyAutoSpGemmStats>()?;
    m.add_class::<PySpGemmStats>()?;
    m.add_function(wrap_pyfunction!(checked_spgemm, m)?)?;
    m.add_function(wrap_pyfunction!(auto_spgemm, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_workload, m)?)?;
    m.add("__version__", "0.11.0")?;
    Ok(())
}
