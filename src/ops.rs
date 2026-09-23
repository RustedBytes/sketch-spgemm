//! Canonical CSR transformations and checked element-wise operations.

use crate::{CheckedAddScalar, CsrBuildError, CsrMatrix, SpGemmError};

impl<T> CsrMatrix<T>
where
    T: Copy,
{
    /// Return the canonical CSR transpose in `O(rows + cols + nnz)` time.
    pub fn transpose(&self) -> Self {
        let mut row_ptr = vec![0usize; self.cols + 1];
        for &column in &self.col_idx {
            row_ptr[column + 1] += 1;
        }
        for row in 0..self.cols {
            row_ptr[row + 1] += row_ptr[row];
        }

        let mut next = row_ptr[..self.cols].to_vec();
        let mut col_idx = vec![0usize; self.values.len()];
        let mut values = self.values.clone();
        for row in 0..self.rows {
            for position in self.row_ptr[row]..self.row_ptr[row + 1] {
                let output_row = self.col_idx[position];
                let destination = next[output_row];
                col_idx[destination] = row;
                values[destination] = self.values[position];
                next[output_row] += 1;
            }
        }

        Self {
            rows: self.cols,
            cols: self.rows,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Preserve the sparsity pattern while replacing every value with `one`.
    pub fn structural(&self, one: T) -> Self
    where
        T: Default + PartialEq,
    {
        assert!(
            one != T::default(),
            "structural replacement must be nonzero"
        );
        Self {
            rows: self.rows,
            cols: self.cols,
            row_ptr: self.row_ptr.clone(),
            col_idx: self.col_idx.clone(),
            values: vec![one; self.values.len()],
        }
    }

    /// Number of stored entries in each row.
    pub fn row_nnz(&self) -> Vec<usize> {
        self.row_ptr
            .windows(2)
            .map(|range| range[1] - range[0])
            .collect()
    }

    /// Number of stored entries in each column.
    pub fn column_nnz(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.cols];
        for &column in &self.col_idx {
            counts[column] += 1;
        }
        counts
    }

    /// Copy selected source rows into a compact matrix in the requested order.
    ///
    /// Repeated row indices intentionally produce repeated output rows.
    pub fn select_rows(&self, rows: &[usize]) -> Result<Self, CsrBuildError> {
        let mut row_ptr = Vec::with_capacity(rows.len() + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        for &row in rows {
            if row >= self.rows {
                return Err(CsrBuildError::RowOutOfBounds {
                    row,
                    rows: self.rows,
                });
            }
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            col_idx.extend_from_slice(&self.col_idx[start..end]);
            values.extend_from_slice(&self.values[start..end]);
            row_ptr.push(values.len());
        }
        Ok(Self {
            rows: rows.len(),
            cols: self.cols,
            row_ptr,
            col_idx,
            values,
        })
    }
}

impl<T> CsrMatrix<T>
where
    T: CheckedAddScalar + Copy + Default + PartialEq,
{
    /// Add two canonical CSR matrices without silently overflowing.
    pub fn try_add(&self, other: &Self) -> Result<Self, SpGemmError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SpGemmError::DimensionMismatch {
                left: (self.rows, self.cols),
                right: (other.rows, other.cols),
            });
        }

        let zero = T::default();
        let mut row_ptr = Vec::with_capacity(self.rows + 1);
        let mut col_idx = Vec::with_capacity(self.nnz().saturating_add(other.nnz()));
        let mut values = Vec::with_capacity(col_idx.capacity());
        row_ptr.push(0);

        for row in 0..self.rows {
            let mut left = self.row(row).peekable();
            let mut right = other.row(row).peekable();
            while left.peek().is_some() || right.peek().is_some() {
                let (column, value) = match (left.peek(), right.peek()) {
                    (Some(&(left_column, _)), Some(&(right_column, _)))
                        if left_column == right_column =>
                    {
                        let (_, left_value) = left.next().expect("peeked left entry");
                        let (_, right_value) = right.next().expect("peeked right entry");
                        let sum = left_value.checked_add_value(right_value).ok_or(
                            SpGemmError::ArithmeticOverflow {
                                operation: crate::ArithmeticOperation::Add,
                                row,
                                inner: left_column,
                                column: left_column,
                            },
                        )?;
                        (left_column, sum)
                    }
                    (Some(&(left_column, _)), Some(&(right_column, _)))
                        if left_column < right_column =>
                    {
                        left.next().expect("peeked left entry")
                    }
                    (Some(_), Some(_)) => right.next().expect("peeked right entry"),
                    (Some(_), None) => left.next().expect("peeked left entry"),
                    (None, Some(_)) => right.next().expect("peeked right entry"),
                    (None, None) => unreachable!(),
                };
                if value != zero {
                    col_idx.push(column);
                    values.push(value);
                }
            }
            row_ptr.push(values.len());
        }

        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Checked sum of the stored values in each row.
    pub fn try_row_sums(&self) -> Result<Vec<T>, CsrBuildError> {
        let mut sums = vec![T::default(); self.rows];
        for (row, sum) in sums.iter_mut().enumerate() {
            for (column, value) in self.row(row) {
                *sum = sum
                    .checked_add_value(value)
                    .ok_or(CsrBuildError::ArithmeticOverflow { row, column })?;
            }
        }
        Ok(sums)
    }

    /// Checked sum of the stored values in each column.
    pub fn try_column_sums(&self) -> Result<Vec<T>, CsrBuildError> {
        let mut sums = vec![T::default(); self.cols];
        for row in 0..self.rows {
            for (column, value) in self.row(row) {
                sums[column] = sums[column]
                    .checked_add_value(value)
                    .ok_or(CsrBuildError::ArithmeticOverflow { row, column })?;
            }
        }
        Ok(sums)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_preserves_values_and_canonical_order() {
        let matrix = CsrMatrix::from_triplets(2, 3, &[(0, 2, 3_i64), (1, 0, 4), (1, 2, 5)]);
        let transposed = matrix.transpose();
        assert_eq!((transposed.rows, transposed.cols), (3, 2));
        assert_eq!(transposed.row_ptr, vec![0, 1, 1, 3]);
        assert_eq!(transposed.col_idx, vec![1, 0, 1]);
        assert_eq!(transposed.values, vec![4, 3, 5]);
        assert_eq!(transposed.transpose(), matrix);
    }

    #[test]
    fn checked_add_merges_patterns_and_removes_zeros() {
        let left = CsrMatrix::from_triplets(2, 3, &[(0, 0, 2_i32), (0, 2, 3), (1, 1, 4)]);
        let right = CsrMatrix::from_triplets(2, 3, &[(0, 0, -2_i32), (0, 1, 7), (1, 1, 1)]);
        let sum = left.try_add(&right).unwrap();
        assert_eq!(sum.to_dense().data, vec![0, 7, 3, 0, 5, 0]);
        assert_eq!(sum.row_nnz(), vec![2, 1]);
        assert_eq!(sum.column_nnz(), vec![0, 2, 1]);
        assert_eq!(sum.try_row_sums().unwrap(), vec![10, 5]);
        assert_eq!(sum.try_column_sums().unwrap(), vec![0, 12, 3]);
    }

    #[test]
    fn checked_add_reports_overflow() {
        let left = CsrMatrix::from_triplets(1, 1, &[(0, 0, i8::MAX)]);
        let right = CsrMatrix::from_triplets(1, 1, &[(0, 0, 1_i8)]);
        assert!(matches!(
            left.try_add(&right),
            Err(SpGemmError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn selected_rows_are_compact_and_may_repeat() {
        let matrix = CsrMatrix::from_triplets(3, 2, &[(0, 0, 1_i64), (1, 1, 2), (2, 0, 3)]);
        let selected = matrix.select_rows(&[2, 0, 2]).unwrap();
        assert_eq!((selected.rows, selected.cols), (3, 2));
        assert_eq!(selected.to_dense().data, vec![3, 0, 1, 0, 3, 0]);
        assert!(matches!(
            matrix.select_rows(&[3]),
            Err(CsrBuildError::RowOutOfBounds { row: 3, rows: 3 })
        ));
    }
}
