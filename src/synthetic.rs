use crate::matrix::CsrMatrix;

#[derive(Clone, Debug)]
pub struct SyntheticProblem {
    pub a: CsrMatrix,
    pub b: CsrMatrix,
    pub expected_candidate_products: u128,
    pub canceled_columns: usize,
    /// Number of columns expected to contain at least one nonzero in C.
    pub active_output_columns: usize,
    /// Target nonzeros per active output column when the generator has a
    /// controlled sparse-output geometry. `None` for the legacy overlap case.
    pub target_nnz_per_active_column: Option<usize>,
    /// Effective dense mixing width used to create candidate amplification.
    pub amplification_width: usize,
    pub generator: &'static str,
}

/// Construct the original deliberately adversarial overlap workload.
///
/// All output rows share `hubs` inner coordinates. Consequently each surviving
/// output coordinate receives `hubs` scalar products, so F/nnz(C) is roughly
/// `hubs` (and larger when columns are algebraically canceled).
///
/// `cancel_fraction` in [0, 1] selects a prefix of output columns whose hub
/// contributions sum to zero. Exact cancellation requires an even hub count.
pub fn overlap_problem(
    rows: usize,
    inner: usize,
    cols: usize,
    hubs: usize,
    cancel_fraction: f64,
) -> SyntheticProblem {
    assert!(rows > 0 && inner > 0 && cols > 0);
    assert!((0.0..=1.0).contains(&cancel_fraction));
    let hubs = hubs.min(inner);
    assert!(hubs > 0);

    let canceled_columns = if hubs % 2 == 0 {
        ((cols as f64) * cancel_fraction).round() as usize
    } else {
        0
    }
    .min(cols);

    let mut at = Vec::with_capacity(rows * hubs);
    for i in 0..rows {
        for k in 0..hubs {
            at.push((i, k, 1));
        }
    }

    let mut bt = Vec::with_capacity(hubs * cols);
    for k in 0..hubs {
        for j in 0..cols {
            let value = if j < canceled_columns {
                if k < hubs / 2 { 1 } else { -1 }
            } else {
                1
            };
            bt.push((k, j, value));
        }
    }

    let a = CsrMatrix::from_triplets(rows, inner, &at);
    let b = CsrMatrix::from_triplets(inner, cols, &bt);
    let expected_candidate_products = rows as u128 * hubs as u128 * cols as u128;

    SyntheticProblem {
        a,
        b,
        expected_candidate_products,
        canceled_columns,
        active_output_columns: cols.saturating_sub(canceled_columns),
        target_nnz_per_active_column: Some(rows),
        amplification_width: hubs,
        generator: "overlap",
    }
}

/// Construct a high-amplification workload whose *output geometry* is directly
/// controllable.
///
/// The first `active_columns` columns of C contain exactly
/// `nnz_per_active_column` nonzeros. A Walsh-Hadamard mixing basis makes A dense
/// across `amplification_width` inner coordinates while B remains dense for the
/// common and especially useful case of odd output-column sparsity. Thus many
/// scalar products cancel even though C is sparse.
///
/// The construction uses the orthogonality identity
///
///     H_r dot H_s = amplification_width * [r == s].
///
/// For each active output column, B is the sum of the Hadamard codewords of the
/// selected output rows. Multiplication by A therefore leaves nonzeros only in
/// those selected rows. If `amplification_width > rows`, a codeword unused by A
/// is available; a fraction of the inactive columns can then contain a fully
/// dense B vector whose *entire* output column cancels to zero.
///
/// Requirements:
/// - `amplification_width` is a power of two,
/// - `rows <= amplification_width <= inner`.
pub fn sparse_output_problem(
    rows: usize,
    inner: usize,
    cols: usize,
    active_columns: usize,
    nnz_per_active_column: usize,
    cancel_fraction: f64,
    amplification_width: usize,
) -> SyntheticProblem {
    assert!(rows > 0 && inner > 0 && cols > 0);
    assert!((0.0..=1.0).contains(&cancel_fraction));
    assert!(amplification_width.is_power_of_two(), "amplification width must be a power of two");
    assert!(amplification_width <= inner, "amplification width exceeds inner dimension");
    assert!(rows <= amplification_width, "sparse-output generator requires rows <= amplification width");

    let active_columns = active_columns.min(cols);
    let nnz_per_active_column = nnz_per_active_column.min(rows);
    assert!(nnz_per_active_column > 0, "nnz per active output column must be positive");

    // A consists of the first `rows` Walsh-Hadamard codewords. Every entry in
    // the active width is ±1, so candidate amplification is large and regular.
    let mut at = Vec::with_capacity(rows.saturating_mul(amplification_width));
    for i in 0..rows {
        for k in 0..amplification_width {
            at.push((i, k, hadamard_sign(i, k)));
        }
    }

    let inactive_columns = cols - active_columns;
    let requested_canceled = ((inactive_columns as f64) * cancel_fraction).round() as usize;
    // We need one Hadamard codeword orthogonal to every row represented in A.
    // Such a row exists exactly when the mixing width is strictly larger.
    let canceled_columns = if amplification_width > rows {
        requested_canceled.min(inactive_columns)
    } else {
        0
    };

    let mut bt = Vec::new();
    for j in 0..active_columns {
        let selected = selected_rows(rows, nnz_per_active_column, j);
        for k in 0..amplification_width {
            let mut value = 0i64;
            for &r in &selected {
                value += hadamard_sign(r, k);
            }
            if value != 0 {
                bt.push((k, j, value));
            }
        }
    }

    // Fill some inactive columns with an unused orthogonal codeword. B is dense
    // in those columns and A*B is exactly zero, creating genuine algebraic
    // cancellation work rather than structural zeros.
    if canceled_columns > 0 {
        let orthogonal_row = rows;
        for j in active_columns..active_columns + canceled_columns {
            for k in 0..amplification_width {
                bt.push((k, j, hadamard_sign(orthogonal_row, k)));
            }
        }
    }

    let a = CsrMatrix::from_triplets(rows, inner, &at);
    let b = CsrMatrix::from_triplets(inner, cols, &bt);

    // A is nonzero in every active mixing coordinate of every row. Therefore
    // every nonzero in B participates in exactly `rows` candidate products.
    let expected_candidate_products = rows as u128 * b.nnz() as u128;

    SyntheticProblem {
        a,
        b,
        expected_candidate_products,
        canceled_columns,
        active_output_columns: active_columns,
        target_nnz_per_active_column: Some(nnz_per_active_column),
        amplification_width,
        generator: "sparse-output",
    }
}

fn selected_rows(rows: usize, count: usize, column: usize) -> Vec<usize> {
    debug_assert!(count <= rows);
    // A simple deterministic cyclic design. Consecutive columns are shifted so
    // recovery sees different support rather than one repeated sparse vector.
    let start = column.wrapping_mul(count.max(1)) % rows;
    (0..count).map(|offset| (start + offset) % rows).collect()
}

#[inline]
fn hadamard_sign(row: usize, col: usize) -> i64 {
    if ((row & col).count_ones() & 1) == 0 { 1 } else { -1 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spgemm::spgemm_hash;

    #[test]
    fn sparse_output_geometry_is_exact() {
        let p = sparse_output_problem(8, 16, 10, 4, 3, 1.0, 16);
        let (c, stats) = spgemm_hash(&p.a, &p.b);
        assert_eq!(stats.candidate_products, p.expected_candidate_products);
        assert_eq!(p.canceled_columns, 6);
        assert_eq!(c.nnz(), 4 * 3);
        for j in 0..4 {
            let col_nnz = (0..c.rows)
                .filter(|&i| c.row(i).any(|(cj, _)| cj == j))
                .count();
            assert_eq!(col_nnz, 3);
        }
        for j in 4..10 {
            assert!((0..c.rows).all(|i| c.row(i).all(|(cj, _)| cj != j)));
        }
    }

    #[test]
    fn sparse_output_without_unused_hadamard_row_uses_structural_zero_columns() {
        let p = sparse_output_problem(8, 8, 10, 4, 1, 1.0, 8);
        assert_eq!(p.canceled_columns, 0);
        let (c, _) = spgemm_hash(&p.a, &p.b);
        assert_eq!(c.nnz(), 4);
    }
}
