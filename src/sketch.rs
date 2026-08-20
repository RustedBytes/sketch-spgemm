use crate::matrix::{CsrMatrix, DenseMatrix};

/// Implicit binary, column-sparse sketch matrix.
///
/// For a logical sketch H with shape `bucket_count x domain`, each input index
/// contributes to `degree` distinct rows. H is never materialized.
#[derive(Clone, Debug)]
pub struct SketchMap {
    pub domain: usize,
    pub bucket_count: usize,
    pub degree: usize,
    pub seed: u64,
}

impl SketchMap {
    pub fn new(domain: usize, bucket_count: usize, degree: usize, seed: u64) -> Self {
        assert!(bucket_count > 0, "bucket_count must be positive");
        assert!(degree > 0, "degree must be positive");
        Self {
            domain,
            bucket_count,
            degree: degree.min(bucket_count),
            seed,
        }
    }

    /// Returns distinct bucket rows for one logical column of the sketch matrix.
    #[inline]
    pub fn buckets(&self, index: usize) -> Vec<usize> {
        assert!(index < self.domain);
        let mut out = Vec::with_capacity(self.degree);
        let mut attempt = 0u64;
        while out.len() < self.degree {
            let key = self.seed
                ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ attempt.wrapping_mul(0xD1B5_4A32_D192_ED03);
            let b = (splitmix64(key) % self.bucket_count as u64) as usize;
            if !out.contains(&b) {
                out.push(b);
            }
            attempt = attempt.wrapping_add(1);
        }
        out
    }
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Compute H*A without materializing H.
/// A has shape r x n, H has shape m x r, result is dense m x n.
pub fn left_sketch(a: &CsrMatrix, h: &SketchMap) -> DenseMatrix {
    assert_eq!(h.domain, a.rows);
    let mut out = DenseMatrix::zeros(h.bucket_count, a.cols);

    for i in 0..a.rows {
        let buckets = h.buckets(i);
        for (k, value) in a.row(i) {
            for &b in &buckets {
                out[(b, k)] += value;
            }
        }
    }
    out
}

/// Compute B*G^T without materializing G.
/// B has shape n x c, G has shape g x c, result is dense n x g.
pub fn right_sketch(b: &CsrMatrix, g: &SketchMap) -> DenseMatrix {
    assert_eq!(g.domain, b.cols);
    let mut out = DenseMatrix::zeros(b.rows, g.bucket_count);

    // B is row-major, so the same logical output-column map is revisited many
    // times. Materialize each tiny bucket list once instead of hashing and
    // allocating a Vec for every nonzero of B.
    let buckets_by_column: Vec<Vec<usize>> =
        (0..g.domain).map(|j| g.buckets(j)).collect();

    for k in 0..b.rows {
        for (j, value) in b.row(k) {
            for &bucket in &buckets_by_column[j] {
                out[(k, bucket)] += value;
            }
        }
    }
    out
}

/// Directly compute H*C*G^T from a sparse C. Used only as a correctness oracle
/// for the identity H(AB)G^T = (HA)(BG^T).
pub fn direct_two_sided_sketch(
    c: &CsrMatrix,
    h: &SketchMap,
    g: &SketchMap,
) -> DenseMatrix {
    assert_eq!(h.domain, c.rows);
    assert_eq!(g.domain, c.cols);
    let mut out = DenseMatrix::zeros(h.bucket_count, g.bucket_count);
    let h_rows: Vec<Vec<usize>> = (0..h.domain).map(|i| h.buckets(i)).collect();
    let g_rows: Vec<Vec<usize>> = (0..g.domain).map(|j| g.buckets(j)).collect();

    for i in 0..c.rows {
        let hb = &h_rows[i];
        for (j, value) in c.row(i) {
            let gb = &g_rows[j];
            for &x in hb {
                for &y in gb {
                    out[(x, y)] += value;
                }
            }
        }
    }
    out
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundParams {
    pub i: usize,
    pub t: usize,
    pub q: usize,
    pub p: usize,
    pub q_accumulated_after: usize,
}

/// The q/p/t schedule from Algorithm 1 of Graia (2026).
/// This does not implement the paper's expander recovery pair; it gives the exact
/// round capacities that a recovery backend must satisfy.
pub fn paper_schedule(r: usize, c: usize, k_bound: usize) -> Vec<RoundParams> {
    let k = k_bound.min(r.saturating_mul(c));
    if k == 0 || r == 0 || c == 0 {
        return Vec::new();
    }

    let target = r.min(k);
    let l = ceil_log2(target);
    let mut q_acc = 0usize;
    let mut rounds = Vec::with_capacity(l + 1);

    for i in 0..=l {
        let t = 1usize.checked_shl(i as u32).unwrap_or(usize::MAX);
        let q = r.min(t.saturating_add(q_acc));
        let p = if i == 0 {
            c.min(k)
        } else {
            let denom = 1usize.checked_shl((i - 1) as u32).unwrap_or(usize::MAX);
            c.min(ceil_div(k, denom))
        };
        q_acc = q_acc.saturating_add(q);
        rounds.push(RoundParams {
            i,
            t,
            q,
            p,
            q_accumulated_after: q_acc,
        });
    }
    rounds
}

fn ceil_log2(x: usize) -> usize {
    if x <= 1 {
        0
    } else {
        usize::BITS as usize - (x - 1).leading_zeros() as usize
    }
}

fn ceil_div(a: usize, b: usize) -> usize {
    if a == 0 {
        0
    } else {
        1 + (a - 1) / b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_are_distinct() {
        let s = SketchMap::new(100, 8, 4, 1);
        for i in 0..100 {
            let mut v = s.buckets(i);
            v.sort_unstable();
            v.dedup();
            assert_eq!(v.len(), 4);
        }
    }

    #[test]
    fn schedule_starts_like_paper() {
        let rounds = paper_schedule(100, 200, 1000);
        assert_eq!(rounds[0].t, 1);
        assert_eq!(rounds[0].q, 1);
        assert_eq!(rounds[0].p, 200);
        assert_eq!(rounds[1].t, 2);
        assert_eq!(rounds[1].q, 3);
    }
}
