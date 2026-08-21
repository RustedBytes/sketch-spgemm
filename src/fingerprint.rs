use crate::matrix::CsrMatrix;
use std::time::{SystemTime, UNIX_EPOCH};

/// 2^61-1, a Mersenne prime. Products fit comfortably in u128.
const MODULUS: u64 = (1u64 << 61) - 1;

#[derive(Clone, Copy, Debug)]
pub struct FingerprintConfig {
    /// Independent bilinear fingerprints. Three lanes give a negligible
    /// accidental-zero probability for non-adversarial/randomly seeded checks.
    pub lanes: usize,
    /// 0 requests a per-process/time-derived seed. Tests should use a fixed seed.
    pub seed: u64,
}

impl Default for FingerprintConfig {
    fn default() -> Self {
        Self { lanes: 3, seed: 0 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FingerprintStats {
    pub checks: usize,
    pub passes: usize,
    pub failures: usize,
    pub lanes: usize,
    pub seed: u64,
}

/// Precomputed bilinear fingerprints of A*B. Verification of a candidate D
/// costs O(lanes * nnz(D)) rather than another matrix multiplication.
#[derive(Clone, Debug)]
pub struct ResidualFingerprint {
    row_weights: Vec<Vec<u64>>,
    col_weights: Vec<Vec<u64>>,
    target: Vec<u64>,
    pub seed: u64,
}

impl ResidualFingerprint {
    pub fn new(a: &CsrMatrix, b: &CsrMatrix, config: FingerprintConfig) -> Self {
        assert_eq!(a.cols, b.rows);
        let lanes = config.lanes.max(1);
        let seed = if config.seed == 0 { runtime_seed() } else { config.seed };
        let mut row_weights = Vec::with_capacity(lanes);
        let mut col_weights = Vec::with_capacity(lanes);
        let mut target = Vec::with_capacity(lanes);

        for lane in 0..lanes {
            let lane_seed = splitmix64(seed ^ (lane as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let rows: Vec<u64> = (0..a.rows)
                .map(|i| nonzero_weight(lane_seed ^ (i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)))
                .collect();
            let cols: Vec<u64> = (0..b.cols)
                .map(|j| nonzero_weight(lane_seed ^ 0xA076_1D64_78BD_642F ^ (j as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB)))
                .collect();

            // left[k] = sum_i r_i A_ik
            let mut left = vec![0u64; a.cols];
            for i in 0..a.rows {
                let rw = rows[i];
                for (k, av) in a.row(i) {
                    left[k] = add_mod(left[k], mul_mod(rw, signed_mod(av)));
                }
            }
            // right[k] = sum_j B_kj s_j
            let mut right = vec![0u64; b.rows];
            for k in 0..b.rows {
                let mut acc = 0u64;
                for (j, bv) in b.row(k) {
                    acc = add_mod(acc, mul_mod(signed_mod(bv), cols[j]));
                }
                right[k] = acc;
            }
            let mut fp = 0u64;
            for k in 0..a.cols {
                fp = add_mod(fp, mul_mod(left[k], right[k]));
            }
            row_weights.push(rows);
            col_weights.push(cols);
            target.push(fp);
        }

        Self { row_weights, col_weights, target, seed }
    }

    pub fn lanes(&self) -> usize { self.target.len() }

    pub fn fingerprint(&self, d: &CsrMatrix) -> Vec<u64> {
        assert_eq!(d.rows, self.row_weights[0].len());
        assert_eq!(d.cols, self.col_weights[0].len());
        let mut out = vec![0u64; self.lanes()];
        for i in 0..d.rows {
            for (j, value) in d.row(i) {
                let vm = signed_mod(value);
                for lane in 0..self.lanes() {
                    let term = mul_mod(mul_mod(self.row_weights[lane][i], vm), self.col_weights[lane][j]);
                    out[lane] = add_mod(out[lane], term);
                }
            }
        }
        out
    }

    #[inline]
    pub fn verifies(&self, d: &CsrMatrix) -> bool {
        self.fingerprint(d) == self.target
    }

    pub fn target(&self) -> &[u64] { &self.target }
}

#[inline]
fn add_mod(a: u64, b: u64) -> u64 {
    let x = a as u128 + b as u128;
    (x % MODULUS as u128) as u64
}

#[inline]
fn mul_mod(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) % MODULUS as u128) as u64
}

#[inline]
fn signed_mod(v: i64) -> u64 {
    if v >= 0 {
        (v as u128 % MODULUS as u128) as u64
    } else {
        let mag = (-(v as i128)) as u128 % MODULUS as u128;
        if mag == 0 { 0 } else { MODULUS - mag as u64 }
    }
}

#[inline]
fn nonzero_weight(x: u64) -> u64 {
    1 + splitmix64(x) % (MODULUS - 1)
}

fn runtime_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xC0FF_EE00_D15C_A11E);
    splitmix64(nanos ^ (std::process::id() as u64).rotate_left(17))
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spgemm::spgemm_hash;

    #[test]
    fn fingerprint_accepts_exact_product_and_rejects_perturbation() {
        let a = CsrMatrix::from_triplets(3, 3, &[(0,0,2),(0,2,1),(1,1,-3),(2,0,5)]);
        let b = CsrMatrix::from_triplets(3, 3, &[(0,0,7),(1,2,11),(2,1,13)]);
        let (c, _) = spgemm_hash(&a, &b);
        let fp = ResidualFingerprint::new(&a, &b, FingerprintConfig { lanes: 3, seed: 42 });
        assert!(fp.verifies(&c));
        let mut t = Vec::new();
        for i in 0..c.rows { for (j,v) in c.row(i) { t.push((i,j,v)); } }
        t.push((2,2,1));
        let bad = CsrMatrix::from_triplets(c.rows, c.cols, &t);
        assert!(!fp.verifies(&bad));
    }
}
