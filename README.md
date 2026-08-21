# SketchSpGEMM v0.7 — adaptive execution + residual fingerprints

Research prototype inspired by Graia, **Optimal Deterministic Fully Sparse Matrix Multiplication** (arXiv:2608.18496), with practical moment/IBLT-style recovery derived from the nested sparse-recovery architecture.

v0.7 builds on the first successful v0.6 benchmark where the complete moment+mask+scheduler algorithm beat the adaptive exact baseline. The production-oriented question is now: **can the library decide when to use sketch recovery without knowing the true `nnz(A*B)`, and can it stop safely without an exact `K`?**

## What is new

### 1. `AutoSpGEMM`

New library API:

```rust
let (c, stats) = auto_spgemm(&a, &b, AutoSpGemmConfig::default());
```

It does not receive the true output or true `nnz(C)`. It:

1. computes the candidate-product count `F` in `O(nnz(A))` using B row degrees;
2. exactly multiplies a small set of evenly spaced A rows;
3. estimates output nnz, active output columns, average nnz per active column, output density, and `rho = F / estimated_nnz(C)`;
4. predicts the moment-recovery row count at the likely useful `q`;
5. chooses **exact** or **moment-sketch** execution;
6. if sketch execution cannot obtain a residual certificate, optionally falls back to exact multiplication.

The exact branch itself is adaptive: below `exact_dense_cell_limit` (16M total A+B+C dense cells by default) it uses the fast adaptive rectangular kernel; above that budget it stays in CSR and uses the hash SpGEMM path rather than forcing a dangerous dense materialization.

The default selector prefers sketching only when all of these look favorable:

```text
estimated rho                 >= 256
estimated avg nnz/active col  <= 64
estimated output density      <= 10%
estimated moment rows / rows  <  80%
```

These are engineering defaults, not theorem constants. They are configurable through `AutoSpGemmConfig`.

### 2. Independent residual fingerprint

v0.7 can certify that a reconstructed candidate `D` satisfies `AB-D = 0` without knowing exact `K`.

For each independent lane it chooses random field weights `r_i` and `s_j` modulo the Mersenne prime `2^61-1` and precomputes:

```text
phi(AB) = r^T A B s = (r^T A)(B s)
```

Verification then computes only:

```text
phi(D) = sum_(i,j in supp(D)) r_i * D_ij * s_j
```

and checks `phi(D) == phi(AB)`.

The expensive target fingerprint is still only linear in the input nnz per lane; checking a candidate is linear in `nnz(D)`. Three independently seeded lanes are the default. A zero seed requests a runtime-derived seed; tests use fixed seeds for reproducibility.

This is a **probabilistic residual certificate**, not Graia's deterministic recovery theorem. Exact identity recovery and exact correction remain available when deterministic verification is required.

CLI controls:

```text
--residual-fingerprint true|false
--fingerprint-lanes N
--fingerprint-seed N     # 0 = runtime-derived
```

### 3. Prepared rectangular factors

v0.6 repeatedly converted dense sketch factors into sparse row views inside the rectangular dispatcher. v0.7 introduces `PreparedFactor`:

```text
DenseMatrix
  + cached sparse rows
  + row nnz counts
  + column nnz counts
```

Recovery-factor caches now store prepared factors, so a reused `HA` or `BG^T` also reuses its sparse representation.

`adaptive_matmul_prepared()` therefore chooses among:

```text
dense-blocked
sparse-left
sparse-right
sparse-sparse
```

without charging another full factor scan. The prepared auto policy applies a small 9/8 irregularity penalty to sparse×sparse traversal so a nearly-dense `HA` with a sparse masked `BG^T` tends to select the more cache-friendly sparse-right path.

The old one-shot `adaptive_matmul()` remains unchanged for standalone/exact baseline comparisons and still includes sparse-view construction cost.

### 4. Loose K bound + tight scheduler hint

The nested algorithm still needs a finite `K` bound to construct its paper-style schedule. `AutoSpGEMM` now separates:

```text
K bound supplied to schedule    = sampled estimate × safety factor
scheduler K hint                = tighter sampled estimate
```

So the schedule can be conservative without forcing the practical q scheduler to jump too high.

If the estimate is wrong, the residual fingerprint detects an incomplete result. `AutoSpGEMM` then falls back to exact multiplication when `exact_fallback=true`.

## Recommended v0.7 benchmark

First verify the crate:

```bash
cargo test
```

Then run the existing sparse-output workload with **no exact-K stop condition** and enable the production selector:

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
  --exact-k-bound false \
  --residual-fingerprint true \
  --fingerprint-lanes 3 \
  --auto-select true \
  --auto-sample-rows 8 \
  --rect-kernel auto \
  --repeats 5
```

There are two useful outputs:

1. the explicit nested benchmark, which still receives `K=nnz(C)` as its schedule bound because the CLI is a research harness, but does **not** use it as an exact stop condition when `--exact-k-bound false`;
2. the `AutoSpGEMM v0.7` section, which receives **no true K at all** and uses only sampled estimates + residual fingerprinting.

A successful auto result should look qualitatively like:

```text
AutoSpGEMM v0.7 (does not use true K):
  choice: Sketch
  exact product check (benchmark oracle): PASS
  estimate: ... est-col-nnz around single digits ... q=8 ...
  sketch certificate: fingerprint=true ...
```

For the old overlap workload with dense surviving columns, the selector should instead report `choice: Exact` because estimated per-column output sparsity/output density are too high.

## Existing v0.6 practical path

The moment backend remains:

```text
3 rows / bucket:
S0 = sum x_i
S1 = sum (i+1)x_i
S2 = sum (i+1)^2 x_i
```

with peeling, exact remeasurement, support masking, and the observed-geometry q scheduler.

Support masking still activates only after identity outer recovery exposes concrete residual output-column IDs. Successfully decoded moment columns are removed from the mask, so later multiplication touches only the unresolved tail.

## Recovery backends

- `moment` — 3-row/bucket algebraic moment peeling; practical default.
- `signature` — deterministic SplitMix bucket graph + binary index signatures.
- `guv` — explicit Parvaresh–Vardy/GUV graph + Bennett decoder.
- `identity` — exact uncompressed recovery reference.

## Main APIs

```rust
// theorem/control-flow oriented
nested_spgemm(...)
nested_spgemm_with_policy(...)

// engineering controls
nested_spgemm_with_options(...)

// v0.7 production-oriented wrapper
auto_spgemm(...)
analyze_workload(...)

// rectangular kernels
adaptive_matmul(...)
adaptive_matmul_prepared(...)
```

## Source layout

```text
src/
├── auto.rs        workload sampling + exact/sketch selection
├── fingerprint.rs bilinear residual certificate over 2^61-1
├── guv.rs         explicit GUV finite-field construction
├── recovery.rs    nested recovery, moment/signature/GUV, masks, scheduler, caches
├── rect.rs        one-shot + prepared adaptive rectangular multiplication
├── sketch.rs      cheap probe and Graia q/p/t schedule
├── spgemm.rs      hash baseline and simple dense kernel
├── matrix.rs      CSR/dense primitives
├── synthetic.rs   overlap + Walsh-Hadamard sparse-output generators
├── lib.rs
└── main.rs        benchmark CLI
```

## Research boundary

The winning practical path is no longer a literal implementation of the deterministic theorem. It combines:

```text
outer support discovery
+ column masking
+ moment-hash inner recovery
+ practical q scheduling
+ randomized residual fingerprinting
+ exact fallback when certification fails
```

The GUV backend remains available for theorem-oriented experiments. The purpose of the v0.7 path is different: turn the paper's compressed-recovery architecture into a useful adaptive SpGEMM strategy on workloads where candidate-product amplification is extreme but the true output is column-sparse.
