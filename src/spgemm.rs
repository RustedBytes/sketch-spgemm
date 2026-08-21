use crate::matrix::{CsrInput, CsrMatrix, DenseMatrix};
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct SpGemmStats {
    pub candidate_products: u128,
}

/// Baseline row-wise hash-accumulator SpGEMM over i64.
/// This is intentionally simple: it is a correctness/baseline kernel, not a
/// replacement for SuiteSparse/Kokkos/cuSPARSE.
pub fn spgemm_hash<A, B>(a: &A, b: &B) -> (CsrMatrix, SpGemmStats)
where
    A: CsrInput + ?Sized,
    B: CsrInput + ?Sized,
{
    assert_eq!(a.cols(), b.rows(), "incompatible matrix dimensions");

    let mut triplets = Vec::new();
    let mut stats = SpGemmStats::default();

    for i in 0..a.rows() {
        let mut acc: HashMap<usize, i64> = HashMap::new();
        for (k, av) in a.row(i) {
            for (j, bv) in b.row(k) {
                stats.candidate_products += 1;
                *acc.entry(j).or_insert(0) += av * bv;
            }
        }

        let mut row: Vec<(usize, i64)> = acc.into_iter().filter(|&(_, v)| v != 0).collect();
        row.sort_unstable_by_key(|&(j, _)| j);
        triplets.extend(row.into_iter().map(|(j, v)| (i, j, v)));
    }

    (
        CsrMatrix::from_triplets(a.rows(), b.cols(), &triplets),
        stats,
    )
}

/// Straightforward dense rectangular GEMM. The inner loop skips zero entries in
/// the left factor so that hashed sketches that remain sparse do not pay the full
/// m*n*g cost.
pub fn dense_matmul(a: &DenseMatrix, b: &DenseMatrix) -> DenseMatrix {
    assert_eq!(a.cols, b.rows);
    let mut out = DenseMatrix::zeros(a.rows, b.cols);

    for i in 0..a.rows {
        for k in 0..a.cols {
            let av = a[(i, k)];
            if av == 0 {
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
}
