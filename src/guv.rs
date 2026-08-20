//! Strongly-explicit Guruswami–Umans–Vadhan / Parvaresh–Vardy expander.
//!
//! This module implements the graph construction stated as Algorithm 3 in
//! Bennett, Gajulapalli, Golovnev, and Warton (2025/2026), Appendix A.2.
//! The graph is never materialized: `neighbors(index)` constructs the q right
//! neighbors of one left vertex on demand.
//!
//! The construction uses q = 2^s, so the base field is represented as
//! GF(2)[z] / f(z).  The degree-n extension polynomial p over GF(q) is found
//! deterministically by exhaustive monic-polynomial search plus Rabin's
//! irreducibility test.  This is mathematically sufficient for the GUV graph;
//! it is deliberately simpler than Shoup's asymptotically fast irreducible-
//! polynomial construction used in the paper's preprocessing bound.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Clone, Debug)]
pub struct GuvConfig {
    /// The constant alpha from Theorem A.3. Smaller values improve the
    /// asymptotic K exponent but make the hidden constants dramatically larger.
    pub alpha: f64,
    /// Expansion error. Bennett et al.'s recovery proof uses epsilon = 1/12.
    pub epsilon: f64,
    /// Use I_N whenever it has no more rows than A \otimes_r B.
    pub identity_fallback: bool,
    /// Optional exact final residual pass for implementation validation.
    /// This is not required by the theorem when the GUV decoder is used.
    pub guaranteed_correction: bool,
}

impl Default for GuvConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            epsilon: 1.0 / 12.0,
            identity_fallback: true,
            guaranteed_correction: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GuvParameters {
    pub domain: usize,
    pub padded_domain: usize,
    pub capacity: usize,
    pub alpha: f64,
    pub epsilon: f64,
    /// n := ceil(log_2 N') where N' + 1 is a power of two.
    pub n: usize,
    /// h from Algorithm 3.
    pub h: u64,
    /// m from Algorithm 3.
    pub m: usize,
    /// q = 2^field_bits.
    pub field_bits: u32,
    pub q: usize,
    /// |R| = q^(m+1), i.e. number of expander rows before binary signatures.
    pub right_vertices: usize,
    /// |R| * log_2(N'+1), i.e. rows of H = A \otimes_r B.
    pub measurement_rows: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GuvError {
    InvalidParameters(&'static str),
    ArithmeticOverflow(&'static str),
    UnsupportedFieldDegree(u32),
    IrreducibleSearchExhausted,
}

impl fmt::Display for GuvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameters(s) => write!(f, "invalid GUV parameters: {s}"),
            Self::ArithmeticOverflow(s) => write!(f, "GUV parameter overflow: {s}"),
            Self::UnsupportedFieldDegree(s) => write!(
                f,
                "GF(2^{s}) is outside this prototype's exact field implementation (max 32)"
            ),
            Self::IrreducibleSearchExhausted => {
                write!(f, "deterministic irreducible-polynomial search exhausted")
            }
        }
    }
}

impl std::error::Error for GuvError {}

/// Implicit adjacency/signature matrix H = A \otimes_r B where A is the
/// explicit GUV unbalanced expander and B is the binary index matrix.
#[derive(Clone, Debug)]
pub struct GuvRecovery {
    pub domain: usize,
    pub padded_domain: usize,
    pub capacity: usize,
    pub bucket_count: usize,
    pub degree: usize,
    pub bits: usize,
    pub alpha: f64,
    pub epsilon: f64,
    pub h: u64,
    pub m: usize,
    field: BinaryExtensionField,
    extension_modulus: Vec<u64>,
    neighbor_cache: Arc<Vec<OnceLock<Arc<[usize]>>>>,
    row_cache: Arc<Vec<OnceLock<Arc<[usize]>>>>,
}

impl GuvParameters {
    /// Compute the literal parameters of Bennett et al. Algorithm 3.
    pub fn new(
        domain: usize,
        capacity: usize,
        alpha: f64,
        epsilon: f64,
    ) -> Result<Self, GuvError> {
        if domain == 0 {
            return Err(GuvError::InvalidParameters("domain must be positive"));
        }
        if capacity == 0 || capacity > domain {
            return Err(GuvError::InvalidParameters(
                "capacity must lie in 1..=domain",
            ));
        }
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(GuvError::InvalidParameters("alpha must be positive"));
        }
        if !epsilon.is_finite() || !(0.0 < epsilon && epsilon < 1.0) {
            return Err(GuvError::InvalidParameters("epsilon must lie in (0,1)"));
        }

        let padded_domain = padded_mersenne_domain(domain)?;
        let n = ceil_log2(padded_domain.saturating_add(1)).max(1);

        // Algorithm 3 has k = log K, so K=1 is degenerate (log K = 0).
        // The recovery layer handles this exact case with I_N instead.
        if capacity == 1 {
            return Err(GuvError::InvalidParameters(
                "capacity K=1 is handled by the identity recovery matrix",
            ));
        }
        let k_log = (capacity as f64).log2();
        let h_real = (2.0 * n as f64 * k_log / epsilon).powf(1.0 / alpha);
        if !h_real.is_finite() || h_real > u64::MAX as f64 {
            return Err(GuvError::ArithmeticOverflow("h"));
        }
        let h = h_real.ceil().max(2.0) as u64;
        let log_h = (h as f64).log2();
        let m = (k_log / log_h).ceil().max(1.0) as usize;

        let q_log = ((1.0 + alpha) * log_h).floor();
        if !q_log.is_finite() || q_log < 1.0 || q_log >= usize::BITS as f64 {
            return Err(GuvError::ArithmeticOverflow("q = 2^floor(log h^(1+alpha))"));
        }
        let field_bits = q_log as u32;
        let q = 1usize
            .checked_shl(field_bits)
            .ok_or(GuvError::ArithmeticOverflow("q"))?;
        let right_vertices = checked_pow_usize(q, m + 1)
            .ok_or(GuvError::ArithmeticOverflow("q^(m+1)"))?;
        let measurement_rows = right_vertices
            .checked_mul(n)
            .ok_or(GuvError::ArithmeticOverflow("|R| * log(N+1)"))?;

        Ok(Self {
            domain,
            padded_domain,
            capacity,
            alpha,
            epsilon,
            n,
            h,
            m,
            field_bits,
            q,
            right_vertices,
            measurement_rows,
        })
    }

    /// Saturating row estimate used to decide whether the theorem's identity
    /// fallback is cheaper before constructing any finite fields.
    pub fn estimated_rows(
        domain: usize,
        capacity: usize,
        alpha: f64,
        epsilon: f64,
    ) -> usize {
        match Self::new(domain, capacity, alpha, epsilon) {
            Ok(p) => p.measurement_rows,
            Err(GuvError::ArithmeticOverflow(_)) => usize::MAX,
            Err(_) => usize::MAX,
        }
    }
}

impl GuvRecovery {
    pub fn new(
        domain: usize,
        capacity: usize,
        alpha: f64,
        epsilon: f64,
    ) -> Result<Self, GuvError> {
        let params = GuvParameters::new(domain, capacity, alpha, epsilon)?;
        if params.field_bits > 32 {
            return Err(GuvError::UnsupportedFieldDegree(params.field_bits));
        }

        let field = BinaryExtensionField::new(params.field_bits)?;
        let extension_modulus = cached_irreducible_over_field(&field, params.n)?;

        Ok(Self {
            domain: params.domain,
            padded_domain: params.padded_domain,
            capacity: params.capacity,
            bucket_count: params.right_vertices,
            degree: params.q,
            bits: params.n,
            alpha: params.alpha,
            epsilon: params.epsilon,
            h: params.h,
            m: params.m,
            field,
            extension_modulus,
            neighbor_cache: Arc::new((0..params.domain).map(|_| OnceLock::new()).collect()),
            row_cache: Arc::new((0..params.domain).map(|_| OnceLock::new()).collect()),
        })
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.bucket_count
            .checked_mul(self.bits)
            .expect("GUV measurement row count overflow")
    }

    /// Strongly-explicit neighbor query for one left coordinate.
    ///
    /// The left vertex is interpreted as one degree < n polynomial t over
    /// GF(q).  Its q neighbors are
    ///   (y, t(y), t^h(y), t^(h^2)(y), ..., t^(h^(m-1))(y)).
    fn compute_neighbors(&self, index: usize) -> Vec<usize> {
        assert!(index < self.domain);

        let q = self.field.order() as usize;
        let mut t = ordinal_polynomial(index as u128, q, self.bits);
        trim_poly(&mut t);

        let mut powers = Vec::with_capacity(self.m);
        let mut current = t;
        for i in 0..self.m {
            if i > 0 {
                current = poly_pow_mod(
                    &current,
                    self.h,
                    &self.extension_modulus,
                    &self.field,
                );
            }
            powers.push(current.clone());
        }

        let mut out = Vec::with_capacity(self.degree);
        for y in 0..q {
            let mut encoded = y;
            let mut factor = q;
            for p in &powers {
                let e = poly_eval(p, y as u64, &self.field) as usize;
                encoded = encoded
                    .checked_add(e.checked_mul(factor).expect("GUV neighbor overflow"))
                    .expect("GUV neighbor overflow");
                factor = factor
                    .checked_mul(q)
                    .expect("GUV right-vertex encoding overflow");
            }
            debug_assert!(encoded < self.bucket_count);
            out.push(encoded);
        }
        out
    }

    /// Shared cached GUV neighbors. The finite-field polynomial expansion is
    /// evaluated at most once per logical coordinate for this recovery graph.
    pub fn neighbors_cached(&self, index: usize) -> Arc<[usize]> {
        assert!(index < self.domain);
        self.neighbor_cache[index]
            .get_or_init(|| Arc::<[usize]>::from(self.compute_neighbors(index)))
            .clone()
    }

    pub fn neighbors(&self, index: usize) -> Vec<usize> {
        self.neighbors_cached(index).as_ref().to_vec()
    }

    /// Shared cached rows of H = A \otimes_r B for one logical coordinate.
    pub fn rows_for_index_cached(&self, index: usize) -> Arc<[usize]> {
        assert!(index < self.domain);
        self.row_cache[index]
            .get_or_init(|| {
                let code = index + 1;
                let neighbors = self.neighbors_cached(index);
                let mut out = Vec::with_capacity(self.degree.saturating_mul(self.bits));
                for &bucket in neighbors.iter() {
                    let base = bucket * self.bits;
                    for bit in 0..self.bits {
                        if ((code >> bit) & 1) != 0 {
                            out.push(base + bit);
                        }
                    }
                }
                Arc::<[usize]>::from(out)
            })
            .clone()
    }

    #[inline]
    pub fn rows_for_index(&self, index: usize) -> Vec<usize> {
        self.rows_for_index_cached(index).as_ref().to_vec()
    }

    pub fn cached_neighbor_count(&self) -> usize {
        self.neighbor_cache
            .iter()
            .filter(|entry| entry.get().is_some())
            .count()
    }

    pub fn cached_row_count(&self) -> usize {
        self.row_cache
            .iter()
            .filter(|entry| entry.get().is_some())
            .count()
    }

}

#[derive(Clone, Debug)]
struct BinaryExtensionField {
    degree: u32,
    /// Monic irreducible polynomial over GF(2), including x^degree.
    modulus: u64,
    mask: u64,
}

impl BinaryExtensionField {
    fn new(degree: u32) -> Result<Self, GuvError> {
        if degree == 0 || degree > 32 {
            return Err(GuvError::UnsupportedFieldDegree(degree));
        }
        let modulus = binary_irreducible_modulus(degree)
            .ok_or(GuvError::UnsupportedFieldDegree(degree))?;
        let mask = if degree == 64 {
            u64::MAX
        } else {
            (1u64 << degree) - 1
        };
        Ok(Self {
            degree,
            modulus,
            mask,
        })
    }

    #[inline]
    fn order(&self) -> u64 {
        1u64 << self.degree
    }

    #[inline]
    fn add(&self, a: u64, b: u64) -> u64 {
        a ^ b
    }

    fn mul(&self, mut a: u64, mut b: u64) -> u64 {
        a &= self.mask;
        b &= self.mask;
        let top = 1u64 << self.degree;
        let mut out = 0u64;
        while b != 0 {
            if (b & 1) != 0 {
                out ^= a;
            }
            b >>= 1;
            a <<= 1;
            if (a & top) != 0 {
                a ^= self.modulus;
            }
        }
        out & self.mask
    }

    fn pow(&self, mut a: u64, mut e: u64) -> u64 {
        let mut out = 1u64;
        while e != 0 {
            if (e & 1) != 0 {
                out = self.mul(out, a);
            }
            e >>= 1;
            if e != 0 {
                a = self.mul(a, a);
            }
        }
        out
    }

    fn inv(&self, a: u64) -> u64 {
        assert!(a != 0, "zero has no multiplicative inverse");
        self.pow(a, self.order() - 2)
    }
}

fn cached_irreducible_over_field(
    field: &BinaryExtensionField,
    degree: usize,
) -> Result<Vec<u64>, GuvError> {
    static CACHE: OnceLock<Mutex<HashMap<(u32, usize), Arc<Vec<u64>>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (field.degree, degree);

    if let Some(value) = cache
        .lock()
        .expect("GUV irreducible-polynomial cache poisoned")
        .get(&key)
        .cloned()
    {
        return Ok((*value).clone());
    }

    // Search outside the mutex so independent constructions do not block one
    // another. A rare concurrent duplicate search is harmless.
    let computed = find_irreducible_over_field(field, degree)?;
    let mut guard = cache
        .lock()
        .expect("GUV irreducible-polynomial cache poisoned");
    let value = guard
        .entry(key)
        .or_insert_with(|| Arc::new(computed.clone()))
        .clone();
    Ok((*value).clone())
}

fn find_irreducible_over_field(
    field: &BinaryExtensionField,
    degree: usize,
) -> Result<Vec<u64>, GuvError> {
    if degree == 0 {
        return Err(GuvError::InvalidParameters(
            "extension degree must be positive",
        ));
    }
    if degree == 1 {
        return Ok(vec![1, 1]);
    }

    let q = field.order() as u128;
    let nonzero = q - 1;
    let mut ordinal = 0u128;

    loop {
        let mut x = ordinal;
        let mut candidate = vec![0u64; degree + 1];
        candidate[0] = (x % nonzero) as u64 + 1;
        x /= nonzero;
        for coeff in candidate.iter_mut().take(degree).skip(1) {
            *coeff = (x % q) as u64;
            x /= q;
        }
        candidate[degree] = 1;

        if polynomial_is_irreducible(&candidate, field) {
            return Ok(candidate);
        }

        ordinal = ordinal
            .checked_add(1)
            .ok_or(GuvError::IrreducibleSearchExhausted)?;
    }
}

/// Rabin irreducibility test over GF(q).
fn polynomial_is_irreducible(f: &[u64], field: &BinaryExtensionField) -> bool {
    let n = poly_degree(f);
    if n <= 0 {
        return false;
    }
    let n = n as usize;
    if n == 1 {
        return true;
    }

    let x_poly = vec![0u64, 1u64];
    let factors = distinct_prime_factors(n);
    let mut checkpoints: Vec<usize> = factors.iter().map(|&p| n / p).collect();
    checkpoints.sort_unstable();
    checkpoints.dedup();

    let mut cur = x_poly.clone();
    for i in 1..=n {
        // q = 2^s, hence raising to q is s repeated squarings.
        for _ in 0..field.degree {
            cur = poly_mul_mod(&cur, &cur, f, field);
        }

        if checkpoints.binary_search(&i).is_ok() {
            let diff = poly_add(&cur, &x_poly, field);
            let g = poly_gcd(f.to_vec(), diff, field);
            if poly_degree(&g) > 0 {
                return false;
            }
        }
    }

    poly_equal(&cur, &x_poly)
}

fn ordinal_polynomial(mut ordinal: u128, q: usize, max_coeffs: usize) -> Vec<u64> {
    let q = q as u128;
    let mut out = vec![0u64; max_coeffs];
    for c in &mut out {
        *c = (ordinal % q) as u64;
        ordinal /= q;
    }
    out
}

fn poly_eval(p: &[u64], x: u64, field: &BinaryExtensionField) -> u64 {
    let mut out = 0u64;
    for &c in p.iter().rev() {
        out = field.add(field.mul(out, x), c);
    }
    out
}

fn poly_pow_mod(
    base: &[u64],
    mut exponent: u64,
    modulus: &[u64],
    field: &BinaryExtensionField,
) -> Vec<u64> {
    let mut out = vec![1u64];
    let mut b = poly_mod(base.to_vec(), modulus, field);
    while exponent != 0 {
        if (exponent & 1) != 0 {
            out = poly_mul_mod(&out, &b, modulus, field);
        }
        exponent >>= 1;
        if exponent != 0 {
            b = poly_mul_mod(&b, &b, modulus, field);
        }
    }
    out
}

fn poly_mul_mod(
    a: &[u64],
    b: &[u64],
    modulus: &[u64],
    field: &BinaryExtensionField,
) -> Vec<u64> {
    if poly_is_zero(a) || poly_is_zero(b) {
        return vec![0];
    }
    let mut product = vec![0u64; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        if x == 0 {
            continue;
        }
        for (j, &y) in b.iter().enumerate() {
            if y == 0 {
                continue;
            }
            product[i + j] ^= field.mul(x, y);
        }
    }
    poly_mod(product, modulus, field)
}

fn poly_add(a: &[u64], b: &[u64], field: &BinaryExtensionField) -> Vec<u64> {
    let n = a.len().max(b.len());
    let mut out = vec![0u64; n];
    for i in 0..n {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        out[i] = field.add(x, y);
    }
    trim_poly(&mut out);
    out
}

fn poly_mod(
    mut a: Vec<u64>,
    modulus: &[u64],
    field: &BinaryExtensionField,
) -> Vec<u64> {
    trim_poly(&mut a);
    let md = poly_degree(modulus);
    assert!(md >= 0, "zero polynomial modulus");
    let md = md as usize;
    let lead_inv = field.inv(modulus[md]);

    while !poly_is_zero(&a) && poly_degree(&a) as usize >= md {
        let ad = poly_degree(&a) as usize;
        let shift = ad - md;
        let factor = field.mul(a[ad], lead_inv);
        if factor != 0 {
            for j in 0..=md {
                a[j + shift] ^= field.mul(factor, modulus[j]);
            }
        }
        trim_poly(&mut a);
    }
    a
}

fn poly_gcd(
    mut a: Vec<u64>,
    mut b: Vec<u64>,
    field: &BinaryExtensionField,
) -> Vec<u64> {
    trim_poly(&mut a);
    trim_poly(&mut b);
    while !poly_is_zero(&b) {
        let r = poly_mod(a, &b, field);
        a = b;
        b = r;
    }
    if poly_is_zero(&a) {
        return vec![0];
    }
    let d = poly_degree(&a) as usize;
    let inv = field.inv(a[d]);
    for c in &mut a {
        *c = field.mul(*c, inv);
    }
    trim_poly(&mut a);
    a
}

fn poly_equal(a: &[u64], b: &[u64]) -> bool {
    let mut aa = a.to_vec();
    let mut bb = b.to_vec();
    trim_poly(&mut aa);
    trim_poly(&mut bb);
    aa == bb
}

fn poly_degree(p: &[u64]) -> isize {
    p.iter()
        .rposition(|&x| x != 0)
        .map(|i| i as isize)
        .unwrap_or(-1)
}

fn poly_is_zero(p: &[u64]) -> bool {
    p.iter().all(|&x| x == 0)
}

fn trim_poly(p: &mut Vec<u64>) {
    while p.len() > 1 && p.last() == Some(&0) {
        p.pop();
    }
    if p.is_empty() {
        p.push(0);
    }
}

fn padded_mersenne_domain(domain: usize) -> Result<usize, GuvError> {
    let x = domain
        .checked_add(1)
        .ok_or(GuvError::ArithmeticOverflow("domain + 1"))?;
    let p2 = x
        .checked_next_power_of_two()
        .ok_or(GuvError::ArithmeticOverflow("next_power_of_two(domain+1)"))?;
    p2.checked_sub(1)
        .ok_or(GuvError::ArithmeticOverflow("padded domain"))
}

fn checked_pow_usize(mut base: usize, mut exp: usize) -> Option<usize> {
    let mut out = 1usize;
    while exp != 0 {
        if (exp & 1) != 0 {
            out = out.checked_mul(base)?;
        }
        exp >>= 1;
        if exp != 0 {
            base = base.checked_mul(base)?;
        }
    }
    Some(out)
}

fn ceil_log2(x: usize) -> usize {
    if x <= 1 {
        0
    } else {
        usize::BITS as usize - (x - 1).leading_zeros() as usize
    }
}

fn distinct_prime_factors(mut n: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut p = 2usize;
    while p * p <= n {
        if n % p == 0 {
            out.push(p);
            while n % p == 0 {
                n /= p;
            }
        }
        p += if p == 2 { 1 } else { 2 };
    }
    if n > 1 {
        out.push(n);
    }
    out
}

/// First lexicographically small irreducible polynomials over GF(2) for
/// degrees 1..=32.  Bit i is the coefficient of x^i.
fn binary_irreducible_modulus(degree: u32) -> Option<u64> {
    let p = match degree {
        1 => 0x3,
        2 => 0x7,
        3 => 0xb,
        4 => 0x13,
        5 => 0x25,
        6 => 0x43,
        7 => 0x83,
        8 => 0x11b,
        9 => 0x203,
        10 => 0x409,
        11 => 0x805,
        12 => 0x1009,
        13 => 0x201b,
        14 => 0x4021,
        15 => 0x8003,
        16 => 0x1002b,
        17 => 0x20009,
        18 => 0x40009,
        19 => 0x80027,
        20 => 0x100009,
        21 => 0x200005,
        22 => 0x400003,
        23 => 0x800021,
        24 => 0x100001b,
        25 => 0x2000009,
        26 => 0x400001b,
        27 => 0x8000027,
        28 => 0x10000003,
        29 => 0x20000005,
        30 => 0x40000003,
        31 => 0x80000009,
        32 => 0x10000008d,
        _ => return None,
    };
    Some(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_modulus_table_is_irreducible() {
        // Independent table sanity: construct each advertised field and verify
        // every nonzero element tested at small degrees has a multiplicative
        // inverse. For larger fields, the polynomial table itself is covered
        // by the construction tests without exhaustive element enumeration.
        for degree in 1..=12 {
            let f = BinaryExtensionField::new(degree).unwrap();
            let limit = f.order().min(4096);
            for a in 1..limit {
                assert_eq!(f.mul(a, f.inv(a)), 1, "degree={degree}, a={a}");
            }
        }
    }

    #[test]
    fn gf256_aes_polynomial_behaves_as_field() {
        let f = BinaryExtensionField::new(8).unwrap();
        assert_eq!(f.mul(0x57, 0x83), 0xc1); // standard AES example
        for a in 1..=255u64 {
            assert_eq!(f.mul(a, f.inv(a)), 1);
        }
    }

    #[test]
    fn finds_irreducible_extension_polynomial() {
        let f = BinaryExtensionField::new(4).unwrap();
        let p = find_irreducible_over_field(&f, 3).unwrap();
        assert_eq!(poly_degree(&p), 3);
        assert!(polynomial_is_irreducible(&p, &f));
    }

    #[test]
    fn cloned_guv_recovery_shares_neighbor_and_row_caches() {
        let g = GuvRecovery::new(7, 2, 4.0, 1.0 / 12.0).unwrap();
        let clone = g.clone();
        assert_eq!(g.cached_neighbor_count(), 0);
        assert_eq!(clone.cached_row_count(), 0);

        let rows = g.rows_for_index_cached(3);
        assert!(!rows.is_empty());
        assert_eq!(g.cached_neighbor_count(), 1);
        assert_eq!(clone.cached_neighbor_count(), 1);
        assert_eq!(clone.cached_row_count(), 1);

        let rows2 = clone.rows_for_index_cached(3);
        assert_eq!(rows.as_ref(), rows2.as_ref());
    }

    #[test]
    fn guv_neighbors_are_q_distinct_right_vertices() {
        // Tiny construction used only to exercise the exact finite-field path.
        // Its measurement matrix is much larger than I_7, so production code
        // with identity_fallback=true would correctly choose I_7 instead.
        let params = GuvParameters::new(7, 2, 4.0, 1.0 / 12.0).unwrap();
        assert_eq!(params.h, 3);
        assert_eq!(params.m, 1);
        assert_eq!(params.q, 128);
        assert_eq!(params.right_vertices, 16_384);
        assert_eq!(params.measurement_rows, 49_152);
        let g = GuvRecovery::new(7, 2, 4.0, 1.0 / 12.0).unwrap();
        assert_eq!(g.degree, 128);
        for index in 0..7 {
            let mut n = g.neighbors(index);
            assert_eq!(n.len(), g.degree);
            assert!(n.iter().all(|&x| x < g.bucket_count));
            n.sort_unstable();
            n.dedup();
            assert_eq!(n.len(), g.degree);
        }
    }

    #[test]
    fn literal_guv_constants_make_identity_cheaper_on_tiny_domains() {
        let rows = GuvParameters::estimated_rows(63, 4, 1.0, 1.0 / 12.0);
        assert!(rows > 63);
    }
}
