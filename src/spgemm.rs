use crate::error::SpGemmError;
use crate::matrix::{CsrInput, CsrMatrix, DenseMatrix, SpGemmScalar};
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

/// Straightforward dense rectangular GEMM. The inner loop skips zero entries in
/// the left factor so that hashed sketches that remain sparse do not pay the full
/// m*n*g cost.
pub fn dense_matmul<T>(a: &DenseMatrix<T>, b: &DenseMatrix<T>) -> DenseMatrix<T>
where
    T: SpGemmScalar,
{
    assert_eq!(a.cols, b.rows);
    let mut out = DenseMatrix::zeros(a.rows, b.cols);

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
}
