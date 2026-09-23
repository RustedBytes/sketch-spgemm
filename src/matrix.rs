use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::ops::{AddAssign, Index, IndexMut, Mul};
use std::slice;

/// Exact scalar used by the sketch, recovery, and fingerprint algorithms.
///
/// The matrix containers are generic, but the current algorithms intentionally
/// keep exact `i64` arithmetic through this default.
pub type Scalar = i64;

/// Read-only access to a canonical CSR matrix over [`Scalar`].
///
/// Implementations must yield every row in strictly increasing column order,
/// without duplicate columns or explicitly stored zero values. Every yielded
/// column must be smaller than [`cols`](Self::cols), and [`nnz`](Self::nnz)
/// must equal the total number of yielded entries. Algorithms may rely on
/// these invariants without rescanning the complete input.
pub trait CsrInput {
    /// Scalar stored by the matrix.
    type Scalar: Copy;

    /// Iterator returned for a borrowed matrix row.
    type RowIter<'a>: Iterator<Item = (usize, Self::Scalar)>
    where
        Self: 'a;

    /// Number of matrix rows.
    fn rows(&self) -> usize;

    /// Number of matrix columns.
    fn cols(&self) -> usize;

    /// Number of explicitly stored nonzero values.
    fn nnz(&self) -> usize;

    /// Iterate over `(column, value)` pairs in `row`.
    ///
    /// # Panics
    ///
    /// May panic when `row >= self.rows()`.
    fn row(&self, row: usize) -> Self::RowIter<'_>;

    /// Materialize this sparse input as a row-major dense matrix.
    fn to_dense(&self) -> DenseMatrix<Self::Scalar>
    where
        Self::Scalar: Default,
    {
        let mut dense = DenseMatrix::zeros(self.rows(), self.cols());
        for row in 0..self.rows() {
            for (column, value) in self.row(row) {
                dense[(row, column)] = value;
            }
        }
        dense
    }
}

/// Borrowing row iterator for [`CsrMatrix`] and [`CsrView`].
#[derive(Clone, Debug)]
pub struct CsrRowIter<'a, T = Scalar> {
    columns: slice::Iter<'a, usize>,
    values: slice::Iter<'a, T>,
}

impl<T: Copy> Iterator for CsrRowIter<'_, T> {
    type Item = (usize, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.columns
            .next()
            .zip(self.values.next())
            .map(|(&column, &value)| (column, value))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.columns.size_hint()
    }
}

impl<T: Copy> ExactSizeIterator for CsrRowIter<'_, T> {}

/// Error returned when borrowed CSR buffers are not canonical.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsrStructureError {
    reason: String,
}

impl CsrStructureError {
    fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    /// Human-readable validation failure.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

impl fmt::Display for CsrStructureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid canonical CSR buffers: {}", self.reason)
    }
}

impl Error for CsrStructureError {}

/// Checked zero-copy view over canonical CSR buffers.
///
/// This adapter is useful for external matrix types that expose ordinary
/// `usize` index buffers but cannot implement this crate's trait directly.
#[derive(Clone, Copy, Debug)]
pub struct CsrView<'a, T = Scalar> {
    rows: usize,
    cols: usize,
    row_ptr: &'a [usize],
    col_idx: &'a [usize],
    values: &'a [T],
}

impl<'a, T> CsrView<'a, T>
where
    T: Copy + Default + PartialEq,
{
    /// Validate and borrow CSR buffers without copying them.
    pub fn try_new(
        rows: usize,
        cols: usize,
        row_ptr: &'a [usize],
        col_idx: &'a [usize],
        values: &'a [T],
    ) -> Result<Self, CsrStructureError> {
        if row_ptr.len() != rows + 1 {
            return Err(CsrStructureError::new(format!(
                "row_ptr has length {}, expected {}",
                row_ptr.len(),
                rows + 1
            )));
        }
        if col_idx.len() != values.len() {
            return Err(CsrStructureError::new(format!(
                "col_idx has length {}, values has length {}",
                col_idx.len(),
                values.len()
            )));
        }
        if row_ptr.first().copied() != Some(0) {
            return Err(CsrStructureError::new("row_ptr must start at zero"));
        }
        if row_ptr.last().copied() != Some(values.len()) {
            return Err(CsrStructureError::new(
                "final row pointer must equal the number of values",
            ));
        }

        let zero = T::default();
        for row in 0..rows {
            let start = row_ptr[row];
            let end = row_ptr[row + 1];
            if start > end || end > values.len() {
                return Err(CsrStructureError::new(format!(
                    "invalid pointer range {start}..{end} for row {row}"
                )));
            }
            let mut previous = None;
            for position in start..end {
                let column = col_idx[position];
                if column >= cols {
                    return Err(CsrStructureError::new(format!(
                        "column {column} in row {row} is outside 0..{cols}"
                    )));
                }
                if previous.is_some_and(|value| value >= column) {
                    return Err(CsrStructureError::new(format!(
                        "columns in row {row} are not strictly increasing"
                    )));
                }
                if values[position] == zero {
                    return Err(CsrStructureError::new(format!(
                        "explicit zero at position {position}"
                    )));
                }
                previous = Some(column);
            }
        }

        Ok(Self {
            rows,
            cols,
            row_ptr,
            col_idx,
            values,
        })
    }
}

impl<T: Copy> CsrInput for CsrView<'_, T> {
    type Scalar = T;
    type RowIter<'a>
        = CsrRowIter<'a, T>
    where
        Self: 'a;

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn nnz(&self) -> usize {
        self.values.len()
    }

    fn row(&self, row: usize) -> Self::RowIter<'_> {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        CsrRowIter {
            columns: self.col_idx[start..end].iter(),
            values: self.values[start..end].iter(),
        }
    }
}

/// Scalar operations required by the representation-independent exact kernels.
///
/// `Default::default()` is interpreted as additive zero. This deliberately
/// keeps the exact baseline independent from the stronger integer assumptions
/// made by sketch recovery and residual fingerprints.
pub trait SpGemmScalar: Copy + Default + PartialEq + AddAssign + Mul<Output = Self> {}

impl<T> SpGemmScalar for T where T: Copy + Default + PartialEq + AddAssign + Mul<Output = Self> {}

/// Scalar addition required when duplicate coordinates are combined by
/// overflow-detecting matrix construction APIs.
///
/// Implementations must return `None` when the exact result cannot be
/// represented by the scalar. Every intermediate addition is checked in input
/// order, so a temporarily unrepresentable sum is an error even if later terms
/// would bring it back into range. Floating-point implementations additionally
/// reject non-finite results.
pub trait CheckedAddScalar: Copy {
    /// Add two values when the result is representable.
    fn checked_add_value(self, rhs: Self) -> Option<Self>;
}

/// Scalar operations required by overflow-detecting multiplication APIs.
pub trait CheckedSpGemmScalar: SpGemmScalar + CheckedAddScalar {
    /// Multiply two values when the result is representable.
    fn checked_mul_value(self, rhs: Self) -> Option<Self>;
}

macro_rules! impl_checked_integer_scalar {
    ($($scalar:ty),+ $(,)?) => {
        $(
            impl CheckedAddScalar for $scalar {
                #[inline]
                fn checked_add_value(self, rhs: Self) -> Option<Self> {
                    <$scalar>::checked_add(self, rhs)
                }
            }

            impl CheckedSpGemmScalar for $scalar {
                #[inline]
                fn checked_mul_value(self, rhs: Self) -> Option<Self> {
                    <$scalar>::checked_mul(self, rhs)
                }
            }
        )+
    };
}

impl_checked_integer_scalar!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize,);

macro_rules! impl_checked_float_scalar {
    ($($scalar:ty),+ $(,)?) => {
        $(
            impl CheckedAddScalar for $scalar {
                #[inline]
                fn checked_add_value(self, rhs: Self) -> Option<Self> {
                    let result = self + rhs;
                    result.is_finite().then_some(result)
                }
            }

            impl CheckedSpGemmScalar for $scalar {
                #[inline]
                fn checked_mul_value(self, rhs: Self) -> Option<Self> {
                    let result = self * rhs;
                    result.is_finite().then_some(result)
                }
            }
        )+
    };
}

impl_checked_float_scalar!(f32, f64);

/// Error returned while constructing canonical CSR storage.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CsrBuildError {
    /// A triplet row lies outside the declared matrix shape.
    RowOutOfBounds { row: usize, rows: usize },
    /// A triplet column lies outside the declared matrix shape.
    ColumnOutOfBounds { column: usize, columns: usize },
    /// A streaming builder received a coordinate before its predecessor.
    OutOfOrder {
        previous: (usize, usize),
        current: (usize, usize),
    },
    /// Duplicate values could not be added without overflow.
    ArithmeticOverflow { row: usize, column: usize },
}

impl fmt::Display for CsrBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RowOutOfBounds { row, rows } => {
                write!(f, "triplet row {row} is outside 0..{rows}")
            }
            Self::ColumnOutOfBounds { column, columns } => {
                write!(f, "triplet column {column} is outside 0..{columns}")
            }
            Self::OutOfOrder { previous, current } => write!(
                f,
                "triplet coordinate ({}, {}) follows ({}, {}) but row-major order is required",
                current.0, current.1, previous.0, previous.1
            ),
            Self::ArithmeticOverflow { row, column } => write!(
                f,
                "arithmetic overflow while combining triplets at ({row}, {column})"
            ),
        }
    }
}

impl Error for CsrBuildError {}

/// Streaming builder for canonical CSR matrices from row-major sorted
/// triplets.
///
/// Coordinates must be pushed in nondecreasing `(row, column)` order.
/// Duplicate coordinates are combined with checked addition, resulting zeros
/// are removed, and empty rows are emitted without buffering their entries.
#[derive(Clone, Debug)]
pub struct CsrBuilder<T = Scalar> {
    rows: usize,
    cols: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<T>,
    last_coordinate: Option<(usize, usize)>,
    pending: Option<(usize, usize, T)>,
}

impl<T> CsrBuilder<T> {
    /// Create an empty streaming builder.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::with_capacity(rows, cols, 0)
    }

    /// Create a builder with space reserved for approximately `capacity`
    /// output entries.
    pub fn with_capacity(rows: usize, cols: usize, capacity: usize) -> Self {
        Self {
            rows,
            cols,
            row_ptr: Vec::with_capacity(rows.saturating_add(1)),
            col_idx: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
            last_coordinate: None,
            pending: None,
        }
    }
}

impl<T> CsrBuilder<T>
where
    T: CheckedAddScalar + Default + PartialEq,
{
    /// Push one row-major sorted triplet into the builder.
    pub fn try_push(&mut self, row: usize, column: usize, value: T) -> Result<(), CsrBuildError> {
        if row >= self.rows {
            return Err(CsrBuildError::RowOutOfBounds {
                row,
                rows: self.rows,
            });
        }
        if column >= self.cols {
            return Err(CsrBuildError::ColumnOutOfBounds {
                column,
                columns: self.cols,
            });
        }

        let coordinate = (row, column);
        if let Some(previous) = self.last_coordinate {
            if coordinate < previous {
                return Err(CsrBuildError::OutOfOrder {
                    previous,
                    current: coordinate,
                });
            }
            if coordinate == previous {
                if value == T::default() {
                    return Ok(());
                }
                if let Some((_, _, current)) = self.pending.as_mut() {
                    let sum = current
                        .checked_add_value(value)
                        .ok_or(CsrBuildError::ArithmeticOverflow { row, column })?;
                    if sum == T::default() {
                        self.pending = None;
                    } else {
                        *current = sum;
                    }
                } else {
                    self.pending = Some((row, column, value));
                }
                return Ok(());
            }
        }

        self.flush_pending();
        while self.row_ptr.len() <= row {
            self.row_ptr.push(self.values.len());
        }
        self.last_coordinate = Some(coordinate);
        if value != T::default() {
            self.pending = Some((row, column, value));
        }
        Ok(())
    }

    /// Extend the builder from row-major sorted triplets.
    ///
    /// This is a streaming, non-transactional operation. If a later triplet
    /// returns an error, all earlier triplets from the same iterator remain in
    /// the builder and construction may continue from that retained prefix.
    pub fn try_extend<I>(&mut self, triplets: I) -> Result<&mut Self, CsrBuildError>
    where
        I: IntoIterator<Item = (usize, usize, T)>,
    {
        for (row, column, value) in triplets {
            self.try_push(row, column, value)?;
        }
        Ok(self)
    }

    /// Finish construction and return canonical CSR storage.
    pub fn finish(mut self) -> CsrMatrix<T> {
        self.flush_pending();
        while self.row_ptr.len() <= self.rows {
            self.row_ptr.push(self.values.len());
        }
        CsrMatrix {
            rows: self.rows,
            cols: self.cols,
            row_ptr: self.row_ptr,
            col_idx: self.col_idx,
            values: self.values,
        }
    }

    fn flush_pending(&mut self) {
        if let Some((_, column, value)) = self.pending.take() {
            self.col_idx.push(column);
            self.values.push(value);
        }
    }
}

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

    /// Convert canonical or non-canonical CSC buffers into canonical CSR.
    ///
    /// Rows within a column may be unsorted and duplicated; duplicates are
    /// summed and resulting zeros are removed, just as in [`Self::from_triplets`].
    /// This conversion allocates because row-oriented kernels cannot traverse
    /// CSC storage efficiently in place.
    ///
    /// # Panics
    ///
    /// Panics when the pointer/data lengths are inconsistent, pointers are not
    /// monotonic, or a row index is outside `0..rows`.
    pub fn from_csc(
        rows: usize,
        cols: usize,
        col_ptr: &[usize],
        row_idx: &[usize],
        values: &[T],
    ) -> Self {
        assert_eq!(col_ptr.len(), cols + 1, "CSC column-pointer length");
        assert_eq!(row_idx.len(), values.len(), "CSC index/value lengths");
        assert_eq!(col_ptr[0], 0, "CSC column pointers must start at zero");
        assert_eq!(
            col_ptr[cols],
            values.len(),
            "CSC final column pointer must equal nnz"
        );

        let mut triplets = Vec::with_capacity(values.len());
        for column in 0..cols {
            let start = col_ptr[column];
            let end = col_ptr[column + 1];
            assert!(start <= end, "CSC column pointers must be monotonic");
            assert!(end <= values.len(), "CSC column pointer exceeds nnz");
            for position in start..end {
                let row = row_idx[position];
                assert!(row < rows, "CSC row {row} out of range {rows}");
                triplets.push((row, column, values[position]));
            }
        }
        Self::from_triplets(rows, cols, &triplets)
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

impl<T> CsrMatrix<T>
where
    T: CheckedAddScalar + Default + PartialEq,
{
    /// Build canonical CSR storage from unsorted triplets without panicking or
    /// silently overflowing while duplicate coordinates are combined.
    /// Duplicate additions are checked in input order; the function does not
    /// use a wider accumulator to permit temporary overflow and later
    /// cancellation.
    pub fn try_from_triplets(
        rows: usize,
        cols: usize,
        triplets: &[(usize, usize, T)],
    ) -> Result<Self, CsrBuildError> {
        let zero = T::default();
        let mut per_row: Vec<BTreeMap<usize, T>> = (0..rows).map(|_| BTreeMap::new()).collect();

        for &(row, column, value) in triplets {
            if row >= rows {
                return Err(CsrBuildError::RowOutOfBounds { row, rows });
            }
            if column >= cols {
                return Err(CsrBuildError::ColumnOutOfBounds {
                    column,
                    columns: cols,
                });
            }
            if value == zero {
                continue;
            }

            if let Some(current) = per_row[row].get(&column).copied() {
                let sum = current
                    .checked_add_value(value)
                    .ok_or(CsrBuildError::ArithmeticOverflow { row, column })?;
                if sum == zero {
                    per_row[row].remove(&column);
                } else {
                    per_row[row].insert(column, sum);
                }
            } else {
                per_row[row].insert(column, value);
            }
        }

        let mut row_ptr = Vec::with_capacity(rows.saturating_add(1));
        let mut col_idx = Vec::with_capacity(triplets.len());
        let mut values = Vec::with_capacity(triplets.len());
        row_ptr.push(0);
        for row in per_row {
            for (column, value) in row {
                col_idx.push(column);
                values.push(value);
            }
            row_ptr.push(col_idx.len());
        }

        Ok(Self {
            rows,
            cols,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Build canonical CSR storage from a row-major sorted triplet stream.
    ///
    /// Unlike [`Self::try_from_triplets`], this path does not retain all input
    /// triplets or allocate a map for each matrix row.
    pub fn try_from_sorted_triplets<I>(
        rows: usize,
        cols: usize,
        triplets: I,
    ) -> Result<Self, CsrBuildError>
    where
        I: IntoIterator<Item = (usize, usize, T)>,
    {
        let iterator = triplets.into_iter();
        let capacity = iterator.size_hint().0;
        let mut builder = CsrBuilder::with_capacity(rows, cols, capacity);
        builder.try_extend(iterator)?;
        Ok(builder.finish())
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

impl<T: Copy> CsrInput for CsrMatrix<T> {
    type Scalar = T;
    type RowIter<'a>
        = CsrRowIter<'a, T>
    where
        T: 'a;

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
        self.values.len()
    }

    #[inline]
    fn row(&self, row: usize) -> Self::RowIter<'_> {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        CsrRowIter {
            columns: self.col_idx[start..end].iter(),
            values: self.values[start..end].iter(),
        }
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

    #[test]
    fn csc_conversion_sorts_rows_combines_duplicates_and_removes_zeros() {
        let csr = CsrMatrix::<i32>::from_csc(2, 2, &[0, 3, 4], &[1, 0, 1, 0], &[2, 5, -2, 7]);
        assert_eq!(csr.row_ptr, vec![0, 2, 2]);
        assert_eq!(csr.col_idx, vec![0, 1]);
        assert_eq!(csr.values, vec![5, 7]);
    }

    #[test]
    fn checked_csr_view_is_zero_copy_and_scalar_generic() {
        let row_ptr = [0, 2, 3];
        let col_idx = [0, 2, 1];
        let values = [2_i32, 3, 4];
        let view = CsrView::try_new(2, 3, &row_ptr, &col_idx, &values).unwrap();

        assert_eq!(view.row(0).collect::<Vec<_>>(), vec![(0, 2), (2, 3)]);
        assert_eq!(view.to_dense().data, vec![2, 0, 3, 0, 4, 0]);
    }

    #[test]
    fn checked_csr_view_rejects_noncanonical_rows() {
        let error = CsrView::try_new(1, 3, &[0, 2], &[2, 1], &[4_i32, 5]).unwrap_err();
        assert!(error.reason().contains("strictly increasing"));
    }

    #[test]
    fn streaming_builder_combines_duplicates_and_preserves_empty_rows() {
        let matrix = CsrMatrix::<i64>::try_from_sorted_triplets(
            4,
            4,
            [
                (0, 2, 4),
                (0, 2, -1),
                (1, 0, 0),
                (2, 1, 5),
                (2, 3, 2),
                (2, 3, -2),
                (3, 0, 7),
            ],
        )
        .unwrap();

        assert_eq!(matrix.row_ptr, vec![0, 1, 1, 2, 3]);
        assert_eq!(matrix.col_idx, vec![2, 1, 0]);
        assert_eq!(matrix.values, vec![3, 5, 7]);
    }

    #[test]
    fn streaming_builder_rejects_out_of_order_coordinates() {
        let mut builder = CsrBuilder::<i32>::new(2, 3);
        builder.try_push(1, 0, 4).unwrap();
        let error = builder.try_push(0, 2, 5).unwrap_err();
        assert_eq!(
            error,
            CsrBuildError::OutOfOrder {
                previous: (1, 0),
                current: (0, 2),
            }
        );
    }

    #[test]
    fn checked_triplet_construction_reports_duplicate_overflow() {
        let error =
            CsrMatrix::<i8>::try_from_triplets(1, 1, &[(0, 0, 127), (0, 0, 1)]).unwrap_err();
        assert_eq!(
            error,
            CsrBuildError::ArithmeticOverflow { row: 0, column: 0 }
        );

        let mut builder = CsrBuilder::<i8>::new(1, 1);
        builder.try_push(0, 0, 127).unwrap();
        assert_eq!(
            builder.try_push(0, 0, 1).unwrap_err(),
            CsrBuildError::ArithmeticOverflow { row: 0, column: 0 }
        );
    }

    #[test]
    fn checked_triplet_construction_reports_invalid_coordinates() {
        assert_eq!(
            CsrMatrix::<i64>::try_from_triplets(1, 2, &[(1, 0, 3)]).unwrap_err(),
            CsrBuildError::RowOutOfBounds { row: 1, rows: 1 }
        );
        assert_eq!(
            CsrMatrix::<i64>::try_from_triplets(1, 2, &[(0, 2, 3)]).unwrap_err(),
            CsrBuildError::ColumnOutOfBounds {
                column: 2,
                columns: 2,
            }
        );
    }
}
