# SketchSpGEMM v0.6 — moment peeling + masked residual multiplication

Research prototype inspired by Graia, **Optimal Deterministic Fully Sparse Matrix Multiplication** (arXiv:2608.18496), and Bennett et al.'s deterministic sparse-recovery construction (arXiv:2508.10250).

v0.6 is driven by the v0.5 sparse-output benchmark where `C` had 128 active columns × 7 nonzeros/column, `rho ≈ 60,855`, but the binary-signature backend still lost to one exact multiplication. The run showed two concrete problems:

1. `q=8` binary signatures used 216 recovery rows and recovered only 19/128 columns.
2. Those 19 recovered columns did not reduce the next multiplication enough; fallback still processed essentially the whole output domain.

v0.6 attacks both directly.

## 1. New `moment` recovery backend

The binary-signature backend spends roughly `ceil(log2(N+1))` rows per bucket. At `N=256`, that is 9 rows/bucket.

`moment` instead stores three exact integer measurements per bucket:

```text
S0 = Σ x_i
S1 = Σ (i+1) x_i
S2 = Σ (i+1)^2 x_i
```

For an isolated bucket:

```text
code = S1 / S0
index = code - 1
S2 == code^2 * S0     # collision check
```

The same idea works for the outer direct-product measurement: `S0`, `S1`, and `S2` are vectors and all vector coordinates must agree on the same recovered index.

The decoder repeatedly peels validated singleton buckets and finally re-measures the recovered sparse vector exactly. If its measurement does not reproduce the original sketch, the decoder rejects it.

For the benchmark `N=256`, `q=8`, degree 3, oversampling 3:

```text
binary signature: 24 buckets × 9 rows = 216 rows
moment backend:    24 buckets × 3 rows = 72 rows
```

This backend is an **engineering backend over exact `i64` arithmetic**, not Graia/Bennett's arbitrary-ring theorem-level recovery pair.

## 2. Support-masked residual multiplication

When the outer recovery matrix is identity, the algorithm sees concrete residual output-column IDs. With:

```text
--masked-residual true
```

v0.6 records that observed support and later zeros every other column of `B` before forming `B*G^T`.

So after observing 128 live output columns out of 512, later work becomes conceptually:

```text
A * B[:, observed_residual_columns]
```

instead of multiplying all 512 output columns again.

For the practical `moment` backend, columns successfully inner-decoded are removed from the mask. The theorem-oriented library entry points leave this optimization disabled by default; the benchmark CLI enables it by default.

Because an early compressed `H` can theoretically hide a residual column, support masking is a **practical heuristic**, not a proof-level support certificate. The benchmark CLI knows the exact `nnz(C)` and can use `--exact-k-bound true` to trigger a full safety correction if the masked path finishes with fewer than `K` entries.

## 3. Observed-geometry q scheduler

The paper schedule remains available unchanged. The practical CLI additionally enables:

```text
--practical-scheduler true
```

After identity outer recovery exposes `m` live residual columns, v0.6 estimates:

```text
average remaining column sparsity ≈ K_remaining / m
```

and skips paper rounds whose `q` is below the next power of two of that estimate.

For the 128 × 7 sparse-output benchmark:

```text
K = 896
observed columns = 128
896 / 128 = 7
next practical q = 8
```

so the no-progress `q=3` round can be skipped.

## 4. v0.5 caches remain

v0.6 keeps the existing hierarchy:

```text
recovery matrix
    ↓
HA / BG^T
    ↓
HABG^T
    ↓
W = H(AB-D)G^T for a D version
```

Support-masked products are deliberately round-local because the mask changes as columns are recovered.

## Recommended v0.6 experiment

This is the first run to try:

```bash
cargo run --release -- \
  --synthetic sparse-output \
  --rows 256 \
  --inner 512 \
  --cols 512 \
  --active-cols 128 \
  --output-col-nnz 7 \
  --amplification-width 512 \
  --cancel 0.75 \
  --nested-backend moment \
  --recovery-degree 3 \
  --recovery-oversampling 3.0 \
  --identity-fallback true \
  --guaranteed-correction false \
  --practical-scheduler true \
  --masked-residual true \
  --exact-k-bound true \
  --rect-kernel auto \
  --repeats 5
```

The useful qualitative target is:

```text
round q=1: observe ~128 residual columns
scheduler: skip q=3, target q=8
round q=8: H around 72 rows, mask around 128 columns, recover most columns
next compressed round: mask contains only a handful of unresolved columns
terminate before a full 512-column exact multiplication
```

The decisive comparison is now:

```text
Nested total < baseline adaptive exact
```

not the much slower hash-map SpGEMM baseline.

## Recovery backends

- `moment` — 3-row/bucket algebraic moment peeling; practical v0.6 default.
- `signature` — deterministic SplitMix bucket graph + binary index signatures; practical comparison.
- `guv` — explicit deterministic Parvaresh–Vardy/GUV graph + Bennett decoder.
- `identity` — exact uncompressed recovery reference.

Graia permits `I_N` whenever a recovery construction would use at least as many rows as identity. `--identity-fallback true` implements that behavior.

## Practical vs theorem-oriented mode

Library calls `nested_spgemm()` and `nested_spgemm_with_policy()` keep the practical scheduler and support mask **off by default**.

The CLI calls `nested_spgemm_with_options()` with:

```text
practical_scheduler = true
masked_residual      = true
exact_k_bound        = true
```

unless overridden. This distinction is intentional: the moment backend and observed-support mask are engineering experiments, not additions to Graia's deterministic theorem.

## CLI additions

```text
--nested-backend guv|signature|moment|identity|off
--practical-scheduler BOOL
--masked-residual BOOL
--exact-k-bound BOOL
```

Existing controls remain:

```text
--recovery-degree N
--recovery-oversampling X
--identity-fallback BOOL
--guaranteed-correction BOOL
--rect-kernel auto|dense|sparse-left|sparse-right|sparse-sparse
```

## Build

```bash
cargo test
cargo run --release -- --help
```

## Source layout

```text
src/
├── guv.rs        explicit GUV finite-field construction
├── recovery.rs   signature/GUV/moment recovery, masking, scheduler, caches
├── rect.rs       adaptive rectangular multiplication
├── sketch.rs     cheap sketch probe and Graia q/p/t schedule
├── spgemm.rs     hash baseline and exact dense kernel
├── matrix.rs     CSR/dense primitives
├── synthetic.rs  overlap + Walsh-Hadamard sparse-output generators
├── lib.rs
└── main.rs       benchmark CLI
```

## Current research boundary

The `moment` backend relies on integer-weighted measurements and exact division, so it does not preserve the arbitrary-ring guarantee of the paper. Its singleton test is deliberately conservative and its final decoder re-measures every accepted vector, but adversarial signed collisions are still outside the theorem-level guarantee.

The purpose of v0.6 is to answer a narrower practical question: **can a compact recoverable sketch plus support-restricted follow-up work beat the already-fast exact rectangular baseline on high-amplification, genuinely sparse-output matrices?**
