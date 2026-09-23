use crate::error::{ArithmeticOperation, SpGemmError};
use crate::matrix::{CheckedSpGemmScalar, CsrInput, CsrMatrix, DenseMatrix, SpGemmScalar};
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct SpGemmStats {
    pub candidate_products: u128,
}

/// Baseline row-wise hash-accumulator SpGEMM over any supported scalar.
/// This is intentionally simple: it is a correctness/baseline kernel, not a
/// replacement for SuiteSparse/Kokkos/cuSPARSE.
pub fn spgemm_hash<T, A, B>(a: &A, b: &B) -> (CsrMatrix<T>, SpGemmStats)
where
    T: SpGemmScalar,
    A: CsrInput<Scalar = T> + ?Sized,
    B: CsrInput<Scalar = T> + ?Sized,
{
    assert_eq!(a.cols(), b.rows(), "incompatible matrix dimensions");

    let mut triplets = Vec::new();
    let mut stats = SpGemmStats::default();

    for i in 0..a.rows() {
        let mut acc: HashMap<usize, T> = HashMap::new();
        for (k, av) in a.row(i) {
            for (j, bv) in b.row(k) {
                stats.candidate_products += 1;
                *acc.entry(j).or_default() += av * bv;
            }
        }

        let zero = T::default();
        let mut row: Vec<(usize, T)> = acc.into_iter().filter(|&(_, v)| v != zero).collect();
        row.sort_unstable_by_key(|&(j, _)| j);
        triplets.extend(row.into_iter().map(|(j, v)| (i, j, v)));
    }

    (
        CsrMatrix::from_triplets(a.rows(), b.cols(), &triplets),
        stats,
    )
}

/// Fallible direct multiplication for arbitrary canonical CSR inputs.
pub fn try_spgemm_hash<T, A, B>(a: &A, b: &B) -> Result<(CsrMatrix<T>, SpGemmStats), SpGemmError>
where
    T: SpGemmScalar,
    A: CsrInput<Scalar = T> + ?Sized,
    B: CsrInput<Scalar = T> + ?Sized,
{
    if a.cols() != b.rows() {
        return Err(SpGemmError::DimensionMismatch {
            left: (a.rows(), a.cols()),
            right: (b.rows(), b.cols()),
        });
    }
    Ok(spgemm_hash(a, b))
}

/// Overflow-detecting row-wise hash-accumulator SpGEMM.
///
/// This exact direct kernel returns the first scalar multiplication or
/// accumulator addition that cannot be represented by `T`. It intentionally
/// does not enter the `i64` sketch pipeline, whose recovery arithmetic has
/// different bounds and remains available through [`crate::try_auto_spgemm`].
/// Accumulation is checked after every candidate product in canonical inner
/// index order; it does not use a wider temporary accumulator, so intermediate
/// overflow is reported even when later cancellation could fit in `T`.
pub fn try_spgemm_hash_checked<T, A, B>(
    a: &A,
    b: &B,
) -> Result<(CsrMatrix<T>, SpGemmStats), SpGemmError>
where
    T: CheckedSpGemmScalar,
    A: CsrInput<Scalar = T> + ?Sized,
    B: CsrInput<Scalar = T> + ?Sized,
{
    if a.cols() != b.rows() {
        return Err(SpGemmError::DimensionMismatch {
            left: (a.rows(), a.cols()),
            right: (b.rows(), b.cols()),
        });
    }

    let zero = T::default();
    let mut row_ptr = Vec::with_capacity(a.rows().saturating_add(1));
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    let mut stats = SpGemmStats::default();
    row_ptr.push(0);

    for row in 0..a.rows() {
        let mut acc: HashMap<usize, T> = HashMap::new();
        for (inner, left) in a.row(row) {
            for (column, right) in b.row(inner) {
                stats.candidate_products = stats.candidate_products.saturating_add(1);
                let product =
                    left.checked_mul_value(right)
                        .ok_or(SpGemmError::ArithmeticOverflow {
                            operation: ArithmeticOperation::Multiply,
                            row,
                            inner,
                            column,
                        })?;
                let current = acc.get(&column).copied().unwrap_or(zero);
                let sum =
                    current
                        .checked_add_value(product)
                        .ok_or(SpGemmError::ArithmeticOverflow {
                            operation: ArithmeticOperation::Add,
                            row,
                            inner,
                            column,
                        })?;
                acc.insert(column, sum);
            }
        }

        let mut output_row: Vec<_> = acc
            .into_iter()
            .filter(|&(_, value)| value != zero)
            .collect();
        output_row.sort_unstable_by_key(|&(column, _)| column);
        for (column, value) in output_row {
            col_idx.push(column);
            values.push(value);
        }
        row_ptr.push(values.len());
    }

    Ok((
        CsrMatrix {
            rows: a.rows(),
            cols: b.cols(),
            row_ptr,
            col_idx,
            values,
        },
        stats,
    ))
}

/// Overflow-detecting exact sparse matrix multiplication.
///
/// This convenience entry point currently uses the row-wise hash accumulator
/// and never enters the unchecked `i64` sketch pipeline.
pub fn try_spgemm_checked<T, A, B>(a: &A, b: &B) -> Result<(CsrMatrix<T>, SpGemmStats), SpGemmError>
where
    T: CheckedSpGemmScalar,
    A: CsrInput<Scalar = T> + ?Sized,
    B: CsrInput<Scalar = T> + ?Sized,
{
    try_spgemm_hash_checked(a, b)
}

/// Straightforward dense rectangular GEMM. The inner loop skips zero entries in
/// the left factor so that hashed sketches that remain sparse do not pay the full
/// m*n*g cost.
pub fn dense_matmul<T>(a: &DenseMatrix<T>, b: &DenseMatrix<T>) -> DenseMatrix<T>
where
    T: SpGemmScalar,
{
    assert_eq!(a.cols, b.rows);
    let mut out: DenseMatrix<T> = DenseMatrix::zeros(a.rows, b.cols);

    for i in 0..a.rows {
        for k in 0..a.cols {
            let av = a[(i, k)];
            if av == T::default() {
                continue;
            }
            let bbase = k * b.cols;
            let obase = i * out.cols;
            for j in 0..b.cols {
                out.data[obase + j] += av * b.data[bbase + j];
            }
        }
    }
    out
}

/// Overflow-detecting dense matrix multiplication.
///
/// Accumulation is checked after every product in inner-index order. A
/// temporarily unrepresentable sum is therefore an error even if subsequent
/// terms would bring the mathematical result back into range.
pub fn try_dense_matmul_checked<T>(
    a: &DenseMatrix<T>,
    b: &DenseMatrix<T>,
) -> Result<DenseMatrix<T>, SpGemmError>
where
    T: CheckedSpGemmScalar,
{
    if a.cols != b.rows {
        return Err(SpGemmError::DimensionMismatch {
            left: (a.rows, a.cols),
            right: (b.rows, b.cols),
        });
    }

    let zero = T::default();
    let mut out: DenseMatrix<T> = DenseMatrix::zeros(a.rows, b.cols);
    for row in 0..a.rows {
        for inner in 0..a.cols {
            let left = a[(row, inner)];
            if left == zero {
                continue;
            }
            for column in 0..b.cols {
                let right = b[(inner, column)];
                let product =
                    left.checked_mul_value(right)
                        .ok_or(SpGemmError::ArithmeticOverflow {
                            operation: ArithmeticOperation::Multiply,
                            row,
                            inner,
                            column,
                        })?;
                out[(row, column)] = out[(row, column)].checked_add_value(product).ok_or(
                    SpGemmError::ArithmeticOverflow {
                        operation: ArithmeticOperation::Add,
                        row,
                        inner,
                        column,
                    },
                )?;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_spgemm() {
        let a = CsrMatrix::from_triplets(2, 2, &[(0, 0, 2), (0, 1, 3), (1, 1, 4)]);
        let b = CsrMatrix::from_triplets(2, 2, &[(0, 0, 5), (1, 0, 7), (1, 1, 11)]);
        let (c, stats) = spgemm_hash(&a, &b);
        assert_eq!(stats.candidate_products, 5);
        assert_eq!(c.to_dense().data, vec![31, 33, 28, 44]);
    }

    #[test]
    fn generic_i32_spgemm() {
        let a = CsrMatrix::<i32>::from_triplets(1, 2, &[(0, 0, 2), (0, 1, 3)]);
        let b = CsrMatrix::<i32>::from_triplets(2, 1, &[(0, 0, 5), (1, 0, 7)]);
        let (c, _) = try_spgemm_hash(&a, &b).unwrap();
        assert_eq!(c.values, vec![31]);
    }

    #[test]
    fn generic_f64_dense_and_sparse_kernels() {
        let a = CsrMatrix::<f64>::from_triplets(1, 2, &[(0, 0, 0.5), (0, 1, 2.0)]);
        let b = CsrMatrix::<f64>::from_triplets(2, 1, &[(0, 0, 4.0), (1, 0, 1.5)]);
        let (sparse, _) = try_spgemm_hash(&a, &b).unwrap();
        let dense = dense_matmul(&a.to_dense(), &b.to_dense());
        assert_eq!(sparse.values, vec![5.0]);
        assert_eq!(dense.data, vec![5.0]);
    }

    #[test]
    fn fallible_generic_api_reports_shape_mismatch() {
        let a = CsrMatrix::<u32>::zeros(1, 2);
        let b = CsrMatrix::<u32>::zeros(3, 1);
        assert!(matches!(
            try_spgemm_hash(&a, &b),
            Err(SpGemmError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn checked_sparse_kernel_reports_multiplication_overflow() {
        let a = CsrMatrix::<i8>::from_triplets(1, 1, &[(0, 0, 100)]);
        let b = CsrMatrix::<i8>::from_triplets(1, 1, &[(0, 0, 2)]);

        assert!(matches!(
            try_spgemm_hash_checked(&a, &b),
            Err(SpGemmError::ArithmeticOverflow {
                operation: ArithmeticOperation::Multiply,
                row: 0,
                inner: 0,
                column: 0,
            })
        ));
    }

    #[test]
    fn checked_sparse_kernel_reports_accumulator_overflow() {
        let a = CsrMatrix::<i8>::from_triplets(1, 2, &[(0, 0, 100), (0, 1, 100)]);
        let b = CsrMatrix::<i8>::from_triplets(2, 1, &[(0, 0, 1), (1, 0, 1)]);

        assert!(matches!(
            try_spgemm_hash_checked(&a, &b),
            Err(SpGemmError::ArithmeticOverflow {
                operation: ArithmeticOperation::Add,
                row: 0,
                inner: 1,
                column: 0,
            })
        ));
    }

    #[test]
    fn checked_sparse_and_dense_kernels_match_exact_result() {
        let a = CsrMatrix::<i32>::from_triplets(1, 2, &[(0, 0, 2), (0, 1, 3)]);
        let b = CsrMatrix::<i32>::from_triplets(2, 1, &[(0, 0, 5), (1, 0, 7)]);

        let (sparse, stats) = try_spgemm_hash_checked(&a, &b).unwrap();
        let dense = try_dense_matmul_checked(&a.to_dense(), &b.to_dense()).unwrap();
        assert_eq!(sparse.values, vec![31]);
        assert_eq!(dense.data, vec![31]);
        assert_eq!(stats.candidate_products, 2);
    }
}
