use crate::matrix::DenseMatrix;
use std::fmt;

/// Rectangular multiplication policy for the compressed product (H*A)(B*G^T).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RectangularPolicy {
    Auto,
    Dense,
    SparseLeft,
    SparseRight,
    SparseSparse,
}

impl Default for RectangularPolicy {
    fn default() -> Self {
        Self::Auto
    }
}

impl std::str::FromStr for RectangularPolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "dense" => Ok(Self::Dense),
            "sparse-left" | "left" => Ok(Self::SparseLeft),
            "sparse-right" | "right" => Ok(Self::SparseRight),
            "sparse-sparse" | "sparse" => Ok(Self::SparseSparse),
            _ => Err(format!(
                "unknown rectangular policy {s:?}; expected auto|dense|sparse-left|sparse-right|sparse-sparse"
            )),
        }
    }
}

impl fmt::Display for RectangularPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Auto => "auto",
            Self::Dense => "dense",
            Self::SparseLeft => "sparse-left",
            Self::SparseRight => "sparse-right",
            Self::SparseSparse => "sparse-sparse",
        };
        f.write_str(s)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RectangularKernel {
    DenseBlocked,
    SparseLeft,
    SparseRight,
    SparseSparse,
}

impl fmt::Display for RectangularKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::DenseBlocked => "dense-blocked",
            Self::SparseLeft => "sparse-left",
            Self::SparseRight => "sparse-right",
            Self::SparseSparse => "sparse-sparse",
        };
        f.write_str(s)
    }
}

#[derive(Clone, Debug, Default)]
pub struct RectangularStats {
    pub kernel: Option<RectangularKernel>,
    pub a_nnz: usize,
    pub b_nnz: usize,
    pub a_density: f64,
    pub b_density: f64,
    /// m*n*g, i.e. multiply count of a truly dense GEMM.
    pub dense_ops: u128,
    /// Cost estimates used by auto dispatch. Sparse estimates include scans
    /// needed to create their compact row views.
    pub dense_estimated_cost: u128,
    pub sparse_left_estimated_cost: u128,
    pub sparse_right_estimated_cost: u128,
    pub sparse_sparse_estimated_cost: u128,
    /// Sum_k nnz(A[:,k]) * nnz(B[k,:]).
    pub sparse_candidate_products: u128,
    /// Scalar multiplies actually issued by the selected kernel.
    pub scalar_multiplications: u128,
}

#[derive(Clone, Debug)]
struct FactorCounts {
    a_nnz: usize,
    b_nnz: usize,
    a_col_nnz: Vec<usize>,
    b_row_nnz: Vec<usize>,
}

impl FactorCounts {
    fn build(a: &DenseMatrix, b: &DenseMatrix) -> Self {
        assert_eq!(a.cols, b.rows);
        let mut a_col_nnz = vec![0usize; a.cols];
        let mut a_nnz = 0usize;
        for i in 0..a.rows {
            let base = i * a.cols;
            for k in 0..a.cols {
                if a.data[base + k] != 0 {
                    a_col_nnz[k] += 1;
                    a_nnz += 1;
                }
            }
        }

        let mut b_row_nnz = vec![0usize; b.rows];
        let mut b_nnz = 0usize;
        for (k, row_count) in b_row_nnz.iter_mut().enumerate() {
            let base = k * b.cols;
            for j in 0..b.cols {
                if b.data[base + j] != 0 {
                    *row_count += 1;
                    b_nnz += 1;
                }
            }
        }

        Self {
            a_nnz,
            b_nnz,
            a_col_nnz,
            b_row_nnz,
        }
    }

    fn sparse_candidate_products(&self) -> u128 {
        self.a_col_nnz
            .iter()
            .zip(&self.b_row_nnz)
            .map(|(&x, &y)| (x as u128) * (y as u128))
            .sum()
    }
}

/// Adaptive dense-output rectangular multiplication.
///
/// The outer sparse-recovery decoder wants a dense measurement matrix W, but
/// its factors can be highly sparse. This routine keeps the output dense while
/// selecting how to traverse the two input factors.
pub fn adaptive_matmul(
    a: &DenseMatrix,
    b: &DenseMatrix,
    policy: RectangularPolicy,
) -> (DenseMatrix, RectangularStats) {
    assert_eq!(a.cols, b.rows, "incompatible matrix dimensions");

    let counts = FactorCounts::build(a, b);
    let m = a.rows as u128;
    let n = a.cols as u128;
    let g = b.cols as u128;
    let a_cells = m.saturating_mul(n);
    let b_cells = n.saturating_mul(g);

    let dense_ops = m.saturating_mul(n).saturating_mul(g);
    let sparse_left_ops = (counts.a_nnz as u128).saturating_mul(g);
    let sparse_right_ops = m.saturating_mul(counts.b_nnz as u128);
    let sparse_sparse_ops = counts.sparse_candidate_products();

    let dense_cost = dense_ops;
    let sparse_left_cost = a_cells.saturating_add(sparse_left_ops);
    let sparse_right_cost = b_cells.saturating_add(sparse_right_ops);
    let sparse_sparse_cost = a_cells
        .saturating_add(b_cells)
        .saturating_add(sparse_sparse_ops);

    let kernel = match policy {
        RectangularPolicy::Dense => RectangularKernel::DenseBlocked,
        RectangularPolicy::SparseLeft => RectangularKernel::SparseLeft,
        RectangularPolicy::SparseRight => RectangularKernel::SparseRight,
        RectangularPolicy::SparseSparse => RectangularKernel::SparseSparse,
        RectangularPolicy::Auto => [
            (dense_cost, RectangularKernel::DenseBlocked),
            (sparse_left_cost, RectangularKernel::SparseLeft),
            (sparse_right_cost, RectangularKernel::SparseRight),
            (sparse_sparse_cost, RectangularKernel::SparseSparse),
        ]
        .into_iter()
        .min_by_key(|(cost, _)| *cost)
        .map(|(_, kernel)| kernel)
        .unwrap(),
    };

    let (out, scalar_multiplications) = match kernel {
        RectangularKernel::DenseBlocked => dense_blocked(a, b),
        RectangularKernel::SparseLeft => {
            let a_rows = sparse_rows(a);
            sparse_left(a, b, &a_rows)
        }
        RectangularKernel::SparseRight => {
            let b_rows = sparse_rows(b);
            sparse_right(a, b, &b_rows)
        }
        RectangularKernel::SparseSparse => {
            let a_rows = sparse_rows(a);
            let b_rows = sparse_rows(b);
            sparse_sparse(a, b, &a_rows, &b_rows)
        }
    };

    let a_total = a.rows.saturating_mul(a.cols).max(1);
    let b_total = b.rows.saturating_mul(b.cols).max(1);
    let stats = RectangularStats {
        kernel: Some(kernel),
        a_nnz: counts.a_nnz,
        b_nnz: counts.b_nnz,
        a_density: counts.a_nnz as f64 / a_total as f64,
        b_density: counts.b_nnz as f64 / b_total as f64,
        dense_ops,
        dense_estimated_cost: dense_cost,
        sparse_left_estimated_cost: sparse_left_cost,
        sparse_right_estimated_cost: sparse_right_cost,
        sparse_sparse_estimated_cost: sparse_sparse_cost,
        sparse_candidate_products: sparse_sparse_ops,
        scalar_multiplications,
    };

    (out, stats)
}

fn sparse_rows(m: &DenseMatrix) -> Vec<Vec<(usize, i64)>> {
    let mut rows = Vec::with_capacity(m.rows);
    for i in 0..m.rows {
        let mut row = Vec::new();
        let base = i * m.cols;
        for j in 0..m.cols {
            let v = m.data[base + j];
            if v != 0 {
                row.push((j, v));
            }
        }
        rows.push(row);
    }
    rows
}

fn dense_blocked(a: &DenseMatrix, b: &DenseMatrix) -> (DenseMatrix, u128) {
    let mut out = DenseMatrix::zeros(a.rows, b.cols);
    const BI: usize = 24;
    const BK: usize = 32;
    const BJ: usize = 64;

    let mut ii = 0usize;
    while ii < a.rows {
        let i_end = (ii + BI).min(a.rows);
        let mut kk = 0usize;
        while kk < a.cols {
            let k_end = (kk + BK).min(a.cols);
            let mut jj = 0usize;
            while jj < b.cols {
                let j_end = (jj + BJ).min(b.cols);
                for i in ii..i_end {
                    let abase = i * a.cols;
                    let obase = i * out.cols;
                    for k in kk..k_end {
                        let av = a.data[abase + k];
                        let bbase = k * b.cols;
                        for j in jj..j_end {
                            out.data[obase + j] += av * b.data[bbase + j];
                        }
                    }
                }
                jj = j_end;
            }
            kk = k_end;
        }
        ii = i_end;
    }

    let ops = (a.rows as u128)
        .saturating_mul(a.cols as u128)
        .saturating_mul(b.cols as u128);
    (out, ops)
}

fn sparse_left(
    a: &DenseMatrix,
    b: &DenseMatrix,
    a_rows: &[Vec<(usize, i64)>],
) -> (DenseMatrix, u128) {
    let mut out = DenseMatrix::zeros(a.rows, b.cols);
    let mut ops = 0u128;
    for (i, row) in a_rows.iter().enumerate() {
        let obase = i * out.cols;
        for &(k, av) in row {
            let bbase = k * b.cols;
            for j in 0..b.cols {
                out.data[obase + j] += av * b.data[bbase + j];
                ops += 1;
            }
        }
    }
    (out, ops)
}

fn sparse_right(
    a: &DenseMatrix,
    b: &DenseMatrix,
    b_rows: &[Vec<(usize, i64)>],
) -> (DenseMatrix, u128) {
    let mut out = DenseMatrix::zeros(a.rows, b.cols);
    let mut ops = 0u128;
    for i in 0..a.rows {
        let abase = i * a.cols;
        let obase = i * out.cols;
        for (k, row) in b_rows.iter().enumerate() {
            let av = a.data[abase + k];
            for &(j, bv) in row {
                out.data[obase + j] += av * bv;
                ops += 1;
            }
        }
    }
    (out, ops)
}

fn sparse_sparse(
    a: &DenseMatrix,
    b: &DenseMatrix,
    a_rows: &[Vec<(usize, i64)>],
    b_rows: &[Vec<(usize, i64)>],
) -> (DenseMatrix, u128) {
    let mut out = DenseMatrix::zeros(a.rows, b.cols);
    let mut ops = 0u128;
    for (i, row) in a_rows.iter().enumerate() {
        let obase = i * out.cols;
        for &(k, av) in row {
            for &(j, bv) in &b_rows[k] {
                out.data[obase + j] += av * bv;
                ops += 1;
            }
        }
    }
    (out, ops)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> (DenseMatrix, DenseMatrix) {
        let mut a = DenseMatrix::zeros(3, 4);
        a[(0, 0)] = 2;
        a[(0, 3)] = 1;
        a[(1, 1)] = -3;
        a[(2, 2)] = 5;

        let mut b = DenseMatrix::zeros(4, 3);
        b[(0, 0)] = 7;
        b[(1, 1)] = 11;
        b[(2, 2)] = 13;
        b[(3, 0)] = 17;
        b[(3, 2)] = 19;
        (a, b)
    }

    #[test]
    fn all_kernels_are_exactly_equivalent() {
        let (a, b) = sample();
        let (reference, _) = adaptive_matmul(&a, &b, RectangularPolicy::Dense);
        for policy in [
            RectangularPolicy::SparseLeft,
            RectangularPolicy::SparseRight,
            RectangularPolicy::SparseSparse,
            RectangularPolicy::Auto,
        ] {
            let (actual, _) = adaptive_matmul(&a, &b, policy);
            assert_eq!(actual, reference, "policy={policy}");
        }
    }

    #[test]
    fn auto_prefers_sparse_sparse_for_very_sparse_factors() {
        let mut a = DenseMatrix::zeros(128, 128);
        let mut b = DenseMatrix::zeros(128, 128);
        for i in 0..8 {
            a[(i, i)] = 1;
            b[(i, i)] = 1;
        }
        let (_, stats) = adaptive_matmul(&a, &b, RectangularPolicy::Auto);
        assert_eq!(stats.kernel, Some(RectangularKernel::SparseSparse));
        assert_eq!(stats.sparse_candidate_products, 8);
        assert_eq!(stats.scalar_multiplications, 8);
    }

    #[test]
    fn auto_dispatches_all_density_extremes() {
        let mut dense_a = DenseMatrix::zeros(32, 32);
        let mut dense_b = DenseMatrix::zeros(32, 32);
        for i in 0..32 {
            for j in 0..32 {
                dense_a[(i, j)] = 1;
                dense_b[(i, j)] = 1;
            }
        }
        let (_, dense_stats) = adaptive_matmul(&dense_a, &dense_b, RectangularPolicy::Auto);
        assert_eq!(dense_stats.kernel, Some(RectangularKernel::DenseBlocked));

        let mut sparse_a = DenseMatrix::zeros(128, 128);
        let mut dense_b = DenseMatrix::zeros(128, 128);
        for i in 0..8 {
            sparse_a[(i, i)] = 1;
        }
        for i in 0..128 {
            for j in 0..128 {
                dense_b[(i, j)] = 1;
            }
        }
        let (_, left_stats) = adaptive_matmul(&sparse_a, &dense_b, RectangularPolicy::Auto);
        assert_eq!(left_stats.kernel, Some(RectangularKernel::SparseLeft));

        let mut dense_a = DenseMatrix::zeros(128, 128);
        let mut sparse_b = DenseMatrix::zeros(128, 128);
        for i in 0..128 {
            for j in 0..128 {
                dense_a[(i, j)] = 1;
            }
        }
        for i in 0..8 {
            sparse_b[(i, i)] = 1;
        }
        let (_, right_stats) = adaptive_matmul(&dense_a, &sparse_b, RectangularPolicy::Auto);
        assert_eq!(right_stats.kernel, Some(RectangularKernel::SparseRight));
    }

    #[test]
    fn forced_kernels_report_expected_multiplication_counts() {
        let (a, b) = sample();
        let (_, left) = adaptive_matmul(&a, &b, RectangularPolicy::SparseLeft);
        let (_, right) = adaptive_matmul(&a, &b, RectangularPolicy::SparseRight);
        let (_, ss) = adaptive_matmul(&a, &b, RectangularPolicy::SparseSparse);
        assert_eq!(left.scalar_multiplications, (left.a_nnz * b.cols) as u128);
        assert_eq!(right.scalar_multiplications, (a.rows * right.b_nnz) as u128);
        assert_eq!(ss.scalar_multiplications, ss.sparse_candidate_products);
    }
}
