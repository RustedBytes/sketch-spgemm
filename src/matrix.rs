use std::collections::BTreeMap;
use std::ops::{AddAssign, Index, IndexMut};

/// Exact scalar used by the sketch, recovery, and fingerprint algorithms.
///
/// The matrix containers are generic, but the current algorithms intentionally
/// keep exact `i64` arithmetic through this default.
pub type Scalar = i64;

/// Representation-independent matrix metadata.
///
/// This trait deliberately exposes only operations that are cheap for both
/// dense and CSR storage. Kernels should continue to accept the concrete
/// representation whose access pattern they require.
pub trait MatrixLike {
    type Scalar;

    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn nnz(&self) -> usize;

    #[inline]
    fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.rows() == 0 || self.cols() == 0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsrMatrix<T = Scalar> {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<T>,
}

impl<T> CsrMatrix<T> {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_ptr: vec![0; rows + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
        }
    }

    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

impl<T> CsrMatrix<T>
where
    T: Copy,
{
    #[inline]
    pub fn row(&self, r: usize) -> impl Iterator<Item = (usize, T)> + '_ {
        let start = self.row_ptr[r];
        let end = self.row_ptr[r + 1];
        self.col_idx[start..end]
            .iter()
            .copied()
            .zip(self.values[start..end].iter().copied())
    }
}

impl<T> CsrMatrix<T>
where
    T: Copy + Default + PartialEq + AddAssign,
{
    /// Build CSR storage from triplets, combining duplicate coordinates.
    ///
    /// `T::default()` is treated as the additive zero.
    pub fn from_triplets(rows: usize, cols: usize, triplets: &[(usize, usize, T)]) -> Self {
        let zero = T::default();
        let mut per_row: Vec<BTreeMap<usize, T>> = (0..rows).map(|_| BTreeMap::new()).collect();

        for &(r, c, v) in triplets {
            assert!(r < rows, "triplet row {r} out of range {rows}");
            assert!(c < cols, "triplet col {c} out of range {cols}");
            if v == zero {
                continue;
            }
            *per_row[r].entry(c).or_default() += v;
        }

        let mut row_ptr = Vec::with_capacity(rows + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);

        for row in per_row {
            for (c, v) in row {
                if v != zero {
                    col_idx.push(c);
                    values.push(v);
                }
            }
            row_ptr.push(col_idx.len());
        }

        Self {
            rows,
            cols,
            row_ptr,
            col_idx,
            values,
        }
    }

    pub fn to_dense(&self) -> DenseMatrix<T> {
        let mut out = DenseMatrix::zeros(self.rows, self.cols);
        for r in 0..self.rows {
            for (c, v) in self.row(r) {
                out[(r, c)] = v;
            }
        }
        out
    }
}

impl<T> MatrixLike for CsrMatrix<T> {
    type Scalar = T;

    #[inline]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    fn nnz(&self) -> usize {
        self.nnz()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DenseMatrix<T = Scalar> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

impl<T> DenseMatrix<T>
where
    T: Clone + Default,
{
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::default(); rows * cols],
        }
    }
}

impl<T> DenseMatrix<T>
where
    T: Default + PartialEq,
{
    pub fn nnz(&self) -> usize {
        let zero = T::default();
        self.data.iter().filter(|value| **value != zero).count()
    }
}

impl<T> DenseMatrix<T>
where
    T: Copy + Default + PartialEq + AddAssign,
{
    pub fn to_csr(&self) -> CsrMatrix<T> {
        let zero = T::default();
        let mut triplets = Vec::with_capacity(self.nnz());
        for i in 0..self.rows {
            let base = i * self.cols;
            for j in 0..self.cols {
                let v = self.data[base + j];
                if v != zero {
                    triplets.push((i, j, v));
                }
            }
        }
        CsrMatrix::from_triplets(self.rows, self.cols, &triplets)
    }
}

impl<T> MatrixLike for DenseMatrix<T>
where
    T: Default + PartialEq,
{
    type Scalar = T;

    #[inline]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    fn nnz(&self) -> usize {
        self.nnz()
    }
}

impl<T> Index<(usize, usize)> for DenseMatrix<T> {
    type Output = T;

    #[inline]
    fn index(&self, (r, c): (usize, usize)) -> &Self::Output {
        &self.data[r * self.cols + c]
    }
}

impl<T> IndexMut<(usize, usize)> for DenseMatrix<T> {
    #[inline]
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut Self::Output {
        &mut self.data[r * self.cols + c]
    }
}

/// Owning boundary type for code that accepts either CSR or dense storage.
///
/// Performance-sensitive kernels should match this enum once and then call a
/// concrete CSR or dense implementation, rather than branching in inner loops.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Matrix<T = Scalar> {
    Csr(CsrMatrix<T>),
    Dense(DenseMatrix<T>),
}

impl<T> Matrix<T> {
    #[inline]
    pub fn rows(&self) -> usize {
        match self {
            Self::Csr(matrix) => matrix.rows,
            Self::Dense(matrix) => matrix.rows,
        }
    }

    #[inline]
    pub fn cols(&self) -> usize {
        match self {
            Self::Csr(matrix) => matrix.cols,
            Self::Dense(matrix) => matrix.cols,
        }
    }

    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    #[inline]
    pub fn as_csr(&self) -> Option<&CsrMatrix<T>> {
        match self {
            Self::Csr(matrix) => Some(matrix),
            Self::Dense(_) => None,
        }
    }

    #[inline]
    pub fn as_dense(&self) -> Option<&DenseMatrix<T>> {
        match self {
            Self::Csr(_) => None,
            Self::Dense(matrix) => Some(matrix),
        }
    }
}

impl<T> Matrix<T>
where
    T: Default + PartialEq,
{
    #[inline]
    pub fn nnz(&self) -> usize {
        match self {
            Self::Csr(matrix) => matrix.nnz(),
            Self::Dense(matrix) => matrix.nnz(),
        }
    }
}

impl<T> Matrix<T>
where
    T: Copy + Default + PartialEq + AddAssign,
{
    pub fn into_csr(self) -> CsrMatrix<T> {
        match self {
            Self::Csr(matrix) => matrix,
            Self::Dense(matrix) => matrix.to_csr(),
        }
    }

    pub fn into_dense(self) -> DenseMatrix<T> {
        match self {
            Self::Csr(matrix) => matrix.to_dense(),
            Self::Dense(matrix) => matrix,
        }
    }
}

impl<T> MatrixLike for Matrix<T>
where
    T: Default + PartialEq,
{
    type Scalar = T;

    #[inline]
    fn rows(&self) -> usize {
        self.rows()
    }

    #[inline]
    fn cols(&self) -> usize {
        self.cols()
    }

    #[inline]
    fn nnz(&self) -> usize {
        self.nnz()
    }
}

impl<T> From<CsrMatrix<T>> for Matrix<T> {
    fn from(matrix: CsrMatrix<T>) -> Self {
        Self::Csr(matrix)
    }
}

impl<T> From<DenseMatrix<T>> for Matrix<T> {
    fn from(matrix: DenseMatrix<T>) -> Self {
        Self::Dense(matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata<M: MatrixLike>(matrix: &M) -> ((usize, usize), usize) {
        (matrix.shape(), matrix.nnz())
    }

    #[test]
    fn default_scalar_remains_i64() {
        let matrix: CsrMatrix = CsrMatrix::from_triplets(1, 1, &[(0, 0, 7)]);
        let value: Scalar = matrix.values[0];
        assert_eq!(value, 7i64);
    }

    #[test]
    fn generic_dense_csr_round_trip_and_shared_metadata() {
        let mut dense = DenseMatrix::<i32>::zeros(2, 3);
        dense[(0, 1)] = 4;
        dense[(1, 2)] = -7;

        let csr = dense.to_csr();
        assert_eq!(metadata(&dense), ((2, 3), 2));
        assert_eq!(metadata(&csr), ((2, 3), 2));
        assert_eq!(csr.to_dense(), dense);
    }

    #[test]
    fn unified_matrix_converts_at_api_boundary() {
        let csr = CsrMatrix::<i32>::from_triplets(2, 2, &[(0, 0, 3), (0, 0, -1), (1, 1, 5)]);
        let matrix: Matrix<i32> = csr.into();

        assert_eq!(matrix.shape(), (2, 2));
        assert_eq!(matrix.nnz(), 2);
        assert!(matrix.as_csr().is_some());
        assert_eq!(matrix.into_dense().data, vec![2, 0, 0, 5]);
    }
}
