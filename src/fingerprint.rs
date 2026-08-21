use crate::matrix::CsrInput;
use std::time::{SystemTime, UNIX_EPOCH};

/// 2^61-1, a Mersenne prime. A product of two residues is < 2^122, so two
/// Mersenne folds reduce it without integer division.
const MODULUS: u64 = (1u64 << 61) - 1;
const MODULUS_U128: u128 = MODULUS as u128;

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
///
/// v0.7.1 builds all lanes in one traversal of A and one traversal of B. v0.7
/// rescanned both sparse matrices once per lane. Modular multiplication also
/// uses the 2^61-1 Mersenne identity rather than generic `%` division.
#[derive(Clone, Debug)]
pub struct ResidualFingerprint {
    row_weights: Vec<Vec<u64>>,
    col_weights: Vec<Vec<u64>>,
    target: Vec<u64>,
    pub seed: u64,
}

impl ResidualFingerprint {
    pub fn new<A, B>(a: &A, b: &B, config: FingerprintConfig) -> Self
    where
        A: CsrInput + ?Sized,
        B: CsrInput + ?Sized,
    {
        assert_eq!(a.cols(), b.rows());
        let lanes = config.lanes.max(1);
        let seed = if config.seed == 0 {
            runtime_seed()
        } else {
            config.seed
        };

        let mut row_weights = vec![vec![0u64; a.rows()]; lanes];
        let mut col_weights = vec![vec![0u64; b.cols()]; lanes];
        for lane in 0..lanes {
            let lane_seed = splitmix64(seed ^ (lane as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            for i in 0..a.rows() {
                row_weights[lane][i] =
                    nonzero_weight(lane_seed ^ (i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
            }
            for j in 0..b.cols() {
                col_weights[lane][j] = nonzero_weight(
                    lane_seed
                        ^ 0xA076_1D64_78BD_642F
                        ^ (j as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB),
                );
            }
        }

        // left[lane][k] = sum_i r_i A_ik. Scan A once for all lanes.
        let mut left = vec![vec![0u64; a.cols()]; lanes];
        for i in 0..a.rows() {
            for (k, av) in a.row(i) {
                let avm = signed_mod(av);
                for lane in 0..lanes {
                    left[lane][k] = add_mod(left[lane][k], mul_mod(row_weights[lane][i], avm));
                }
            }
        }

        // right[lane][k] = sum_j B_kj s_j. Scan B once for all lanes.
        let mut right = vec![vec![0u64; b.rows()]; lanes];
        for k in 0..b.rows() {
            for (j, bv) in b.row(k) {
                let bvm = signed_mod(bv);
                for lane in 0..lanes {
                    right[lane][k] = add_mod(right[lane][k], mul_mod(bvm, col_weights[lane][j]));
                }
            }
        }

        let mut target = vec![0u64; lanes];
        for lane in 0..lanes {
            let mut fp = 0u64;
            for k in 0..a.cols() {
                fp = add_mod(fp, mul_mod(left[lane][k], right[lane][k]));
            }
            target[lane] = fp;
        }

        Self {
            row_weights,
            col_weights,
            target,
            seed,
        }
    }

    pub fn lanes(&self) -> usize {
        self.target.len()
    }

    pub fn fingerprint<D>(&self, d: &D) -> Vec<u64>
    where
        D: CsrInput + ?Sized,
    {
        assert_eq!(d.rows(), self.row_weights[0].len());
        assert_eq!(d.cols(), self.col_weights[0].len());
        let lanes = self.lanes();
        let mut out = vec![0u64; lanes];
        for i in 0..d.rows() {
            for (j, value) in d.row(i) {
                let vm = signed_mod(value);
                for lane in 0..lanes {
                    let term = mul_mod(
                        mul_mod(self.row_weights[lane][i], vm),
                        self.col_weights[lane][j],
                    );
                    out[lane] = add_mod(out[lane], term);
                }
            }
        }
        out
    }

    #[inline]
    pub fn verifies<D>(&self, d: &D) -> bool
    where
        D: CsrInput + ?Sized,
    {
        self.fingerprint(d) == self.target
    }

    pub fn target(&self) -> &[u64] {
        &self.target
    }
}

#[inline]
fn add_mod(a: u64, b: u64) -> u64 {
    // a,b < p and 2p < 2^62, so u64 addition cannot overflow.
    let x = a + b;
    if x >= MODULUS {
        x - MODULUS
    } else {
        x
    }
}

#[inline]
fn reduce_mersenne(mut x: u128) -> u64 {
    // x can be as large as ~2^126 for signed i64 conversion and <2^122 for
    // residue multiplication. Repeatedly fold high 61-bit limbs into low limbs.
    x = (x & MODULUS_U128) + (x >> 61);
    x = (x & MODULUS_U128) + (x >> 61);
    x = (x & MODULUS_U128) + (x >> 61);
    let mut r = x as u64;
    if r >= MODULUS {
        r -= MODULUS;
    }
    if r >= MODULUS {
        r -= MODULUS;
    }
    r
}

#[inline]
fn mul_mod(a: u64, b: u64) -> u64 {
    reduce_mersenne(a as u128 * b as u128)
}

#[inline]
fn signed_mod(v: i64) -> u64 {
    if v >= 0 {
        reduce_mersenne(v as u128)
    } else {
        let mag = (-(v as i128)) as u128;
        let r = reduce_mersenne(mag);
        if r == 0 {
            0
        } else {
            MODULUS - r
        }
    }
}

#[inline]
fn nonzero_weight(x: u64) -> u64 {
    // This setup path is tiny compared with sparse-matrix traversal. Keep the
    // simple modulus here; generated weights are not in the hot multiply loop.
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
    use crate::matrix::CsrMatrix;
    use crate::spgemm::spgemm_hash;

    #[test]
    fn fingerprint_accepts_exact_product_and_rejects_perturbation() {
        let a = CsrMatrix::from_triplets(3, 3, &[(0, 0, 2), (0, 2, 1), (1, 1, -3), (2, 0, 5)]);
        let b = CsrMatrix::from_triplets(3, 3, &[(0, 0, 7), (1, 2, 11), (2, 1, 13)]);
        let (c, _) = spgemm_hash(&a, &b);
        let fp = ResidualFingerprint::new(&a, &b, FingerprintConfig { lanes: 3, seed: 42 });
        assert!(fp.verifies(&c));
        let mut t = Vec::new();
        for i in 0..c.rows {
            for (j, v) in c.row(i) {
                t.push((i, j, v));
            }
        }
        t.push((2, 2, 1));
        let bad = CsrMatrix::from_triplets(c.rows, c.cols, &t);
        assert!(!fp.verifies(&bad));
    }

    #[test]
    fn fast_mersenne_reduction_matches_reference_modulus() {
        let values = [
            0u128,
            1,
            MODULUS as u128 - 1,
            MODULUS as u128,
            (MODULUS as u128) * (MODULUS as u128),
            ((1u128 << 122) - 1),
            (i64::MAX as u128) * (MODULUS as u128 - 7),
        ];
        for x in values {
            assert_eq!(reduce_mersenne(x), (x % MODULUS_U128) as u64, "x={x}");
        }
    }

    #[test]
    fn signed_mod_matches_reference_for_extremes() {
        for v in [i64::MIN, -9, -1, 0, 1, 9, i64::MAX] {
            let reference = if v >= 0 {
                (v as u128 % MODULUS_U128) as u64
            } else {
                let m = (-(v as i128)) as u128 % MODULUS_U128;
                if m == 0 {
                    0
                } else {
                    MODULUS - m as u64
                }
            };
            assert_eq!(signed_mod(v), reference, "v={v}");
        }
    }
}
