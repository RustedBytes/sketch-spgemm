use crate::matrix::CsrInput;
use std::time::{SystemTime, UNIX_EPOCH};

/// 2^61-1, a Mersenne prime. A product of two residues is < 2^122, so two
/// Mersenne folds reduce it without integer division.
const MODULUS: u64 = (1u64 << 61) - 1;
const MODULUS_U128: u128 = MODULUS as u128;
// Largest prime below 2^64. Two moduli distinguish every residual whose
// absolute value is smaller than their product.
const SECOND_MODULUS: u128 = (1u128 << 64) - 59;
const RESIDUAL_LIMIT: u128 = MODULUS_U128 * SECOND_MODULUS;

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
    second_target: Vec<u64>,
    product_bound: u128,
    pub seed: u64,
}

impl ResidualFingerprint {
    pub fn new<A, B>(a: &A, b: &B, config: FingerprintConfig) -> Self
    where
        A: CsrInput<Scalar = i64> + ?Sized,
        B: CsrInput<Scalar = i64> + ?Sized,
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

        // Bound each exact output entry without computing the product. Saturation
        // is conservative: an unbounded certificate is rejected, never trusted.
        let b_bounds: Vec<u128> = (0..b.rows())
            .map(|k| {
                b.row(k)
                    .map(|(_, v)| u128::from(v.unsigned_abs()))
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        let product_bound = (0..a.rows())
            .map(|i| {
                a.row(i).fold(0u128, |sum, (k, v)| {
                    sum.saturating_add(u128::from(v.unsigned_abs()).saturating_mul(b_bounds[k]))
                })
            })
            .max()
            .unwrap_or(0);
        let mut second_target = vec![0; lanes];
        for lane in 0..lanes {
            let mut left = vec![0; a.cols()];
            for (i, &row_weight) in row_weights[lane].iter().enumerate() {
                for (k, value) in a.row(i) {
                    left[k] = second_add(left[k], second_mul(row_weight, second_signed(value)));
                }
            }
            for (k, &weight) in left.iter().enumerate() {
                for (j, value) in b.row(k) {
                    second_target[lane] = second_add(
                        second_target[lane],
                        second_mul(
                            second_mul(weight, second_signed(value)),
                            col_weights[lane][j],
                        ),
                    );
                }
            }
        }

        Self {
            row_weights,
            col_weights,
            target,
            second_target,
            product_bound,
            seed,
        }
    }

    pub fn lanes(&self) -> usize {
        self.target.len()
    }

    pub fn fingerprint<D>(&self, d: &D) -> Vec<u64>
    where
        D: CsrInput<Scalar = i64> + ?Sized,
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

    /// Check both prime fields, rejecting certificates whose conservative
    /// residual bound exceeds their combined range. False negatives are possible
    /// for extremely large products; callers can then use an exact fallback.
    #[inline]
    pub fn verifies<D>(&self, d: &D) -> bool
    where
        D: CsrInput<Scalar = i64> + ?Sized,
    {
        if self.fingerprint(d) != self.target {
            return false;
        }
        let mut second = vec![0; self.lanes()];
        for i in 0..d.rows() {
            for (j, value) in d.row(i) {
                if self
                    .product_bound
                    .saturating_add(u128::from(value.unsigned_abs()))
                    >= RESIDUAL_LIMIT
                {
                    return false;
                }
                for (lane, output) in second.iter_mut().enumerate() {
                    *output = second_add(
                        *output,
                        second_mul(
                            second_mul(self.row_weights[lane][i], second_signed(value)),
                            self.col_weights[lane][j],
                        ),
                    );
                }
            }
        }
        // Also cover implicit zeros in the candidate.
        self.product_bound < RESIDUAL_LIMIT && second == self.second_target
    }

    pub fn target(&self) -> &[u64] {
        &self.target
    }
}

#[inline]
fn second_add(a: u64, b: u64) -> u64 {
    ((u128::from(a) + u128::from(b)) % SECOND_MODULUS) as u64
}

#[inline]
fn second_mul(a: u64, b: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) % SECOND_MODULUS) as u64
}

#[inline]
fn second_signed(value: i64) -> u64 {
    if value >= 0 {
        value as u64
    } else {
        (SECOND_MODULUS - u128::from(value.unsigned_abs())) as u64
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
    fn rejects_primary_modulus_aliases_and_accepts_large_exact_values() {
        let zero = CsrMatrix::<i64>::from_triplets(1, 1, &[]);
        for seed in [1, 42, 999] {
            let fp = ResidualFingerprint::new(&zero, &zero, FingerprintConfig { lanes: 3, seed });
            for value in [MODULUS as i64, -(MODULUS as i64), i64::MIN, i64::MAX] {
                assert!(!fp.verifies(&CsrMatrix::from_triplets(1, 1, &[(0, 0, value)])));
            }
        }
        let one = CsrMatrix::from_triplets(1, 1, &[(0, 0, 1)]);
        for value in [i64::MIN, i64::MAX, MODULUS as i64] {
            let a = CsrMatrix::from_triplets(1, 1, &[(0, 0, value)]);
            assert!(ResidualFingerprint::new(&a, &one, FingerprintConfig::default()).verifies(&a));
        }
    }

    #[test]
    fn rejects_certificates_outside_proven_residual_range() {
        let a = CsrMatrix::from_triplets(1, 2, &[(0, 0, i64::MAX), (0, 1, i64::MAX)]);
        let b = CsrMatrix::from_triplets(2, 1, &[(0, 0, i64::MAX), (1, 0, -i64::MAX)]);
        let zero = CsrMatrix::<i64>::from_triplets(1, 1, &[]);
        assert!(!ResidualFingerprint::new(&a, &b, FingerprintConfig::default()).verifies(&zero));
    }

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
