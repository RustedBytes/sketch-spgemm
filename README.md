# SketchSpGEMM v0.5 — cached residual recovery + sparse-output workloads

Research prototype for Graia, **Optimal Deterministic Fully Sparse Matrix Multiplication** (arXiv:2608.18496), with the deterministic sparse-recovery construction of Bennett et al. (arXiv:2508.10250).

v0.5 is driven by the first real 512×512 benchmark runs. It keeps the GUV, signature-hash and identity backends from v0.4, but fixes the major engineering pathologies those runs exposed.

## What changed in v0.5

### 1. Four-level reuse cache

The nested loop now caches deterministic/repeated work as:

```text
recovery matrix
    ↓
HA / BG^T
    ↓
HABG^T
    ↓
W = H(AB-D)G^T for a specific D version
```

A `D` generation counter invalidates only residual measurements when the current approximation changes. If `(H,G)` repeats while `D` is unchanged, the whole `W` measurement is reused.

The practical `signature` backend is still reseeded every round, so genuine signature matrices are intentionally not cached. However, if a signature request falls back to `I_N`, the **actual resulting matrix** receives an identity cache key and is reused across later rounds.

Per-round output now reports:

```text
matrix-cache(H/G)
factor-cache(HA/BGT)
HABG-cache
W-cache
```

### 2. Early termination

Two exact identity cases can stop the schedule safely:

- `H = I`, `G = I`, and measured `W = 0`: the residual is exactly zero;
- `H = I_r`, `G = I_c`, and every nonzero outer residual column is successfully inner-decoded: identity measurements expose the entire residual, so another verification round is unnecessary.

This removes the redundant late rounds seen in the v0.4 512×512 tests.

### 3. Faster right sketches

`B*G^T` previously recomputed `rows_for_index(j)` for every nonzero of `B`. v0.5 precomputes the mapping once per logical column. Identity sketches are special-cased to direct CSR→dense conversion.

The same optimization is applied to the cheap kernel-only `SketchMap` probe.

### 4. Fairer exact baseline

The CLI now reports both:

```text
baseline hash SpGEMM
baseline adaptive exact
```

The adaptive exact baseline densifies A/B and runs the same rectangular engine used by nested recovery. This is important because the original hash baseline can be tens of times slower on regular, moderately dense synthetic matrices.

### 5. Sparse-output benchmark generator

The original `overlap` generator creates output columns that are either zero or dense. That is useful for cancellation stress, but it is close to a worst case for the inner recovery decoder.

v0.5 adds:

```text
--synthetic sparse-output
--active-cols N
--output-col-nnz N
--amplification-width N
```

The generator uses Walsh–Hadamard mixing. For active output columns, it constructs `B` so that enormous numbers of products cancel while exactly the requested output rows survive:

```text
C active columns       = --active-cols
nnz / active C column  = --output-col-nnz
mixing width            = --amplification-width
```

`amplification-width` must be a power of two and satisfy:

```text
rows <= amplification-width <= inner
```

Use `0` to select the largest power of two not exceeding `inner`.

When `amplification_width > rows`, an unused Hadamard codeword is available. A fraction of inactive columns selected by `--cancel` are then made dense in `B` while their complete output cancels to zero.

## Recommended v0.5 experiments

### Re-run the dense-column workload

```bash
cargo run --release -- \
  --rows 512 --inner 512 --cols 512 \
  --hubs 256 --cancel 0.75 \
  --nested-backend signature \
  --identity-fallback true \
  --guaranteed-correction false \
  --rect-kernel auto
```

Expected qualitative behavior versus v0.4:

- rounds that fall back to identity reuse the same matrix/factors/product;
- while `D=0`, repeated identity rounds reuse `W`;
- the run terminates on the full-domain identity recovery round instead of executing the final two rounds.

### Test the geometry Graia actually wants

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
  --nested-backend signature \
  --identity-fallback true \
  --guaranteed-correction false \
  --rect-kernel auto \
  --repeats 5
```

Using an odd `output-col-nnz` tends to keep the Hadamard-summed `B` columns dense, maximizing candidate amplification while `C` stays sparse.

## Recovery backends

- `guv` — explicit deterministic Parvaresh–Vardy/GUV graph + Bennett `A ⊗_r B` decoder.
- `signature` — deterministic SplitMix bucket graph + the same signature decoder; engineering comparison, not theorem-level expansion guarantee.
- `identity` — exact uncompressed recovery reference.

Graia permits `I_N` whenever the recovery construction uses at least as many rows as identity. `--identity-fallback true` implements that rule.

## Rectangular kernels

```text
auto
dense
sparse-left
sparse-right
sparse-sparse
```

`auto` compares estimated traversal cost and reports the selected kernel, factor densities, candidate products and scalar multiplications.

## Build

```bash
cargo test
cargo run --release -- --help
```

## CLI

```text
--rows N
--inner N
--cols N

--synthetic overlap|sparse-output
--hubs N
--cancel X
--active-cols N
--output-col-nnz N
--amplification-width N

--h-buckets N
--g-buckets N
--degree N
--repeats N

--nested-backend guv|signature|identity|off
--recovery-degree N
--recovery-oversampling X
--guv-alpha X
--guv-epsilon X
--identity-fallback BOOL
--guaranteed-correction BOOL

--rect-kernel auto|dense|sparse-left|sparse-right|sparse-sparse
```

## Source layout

```text
src/
├── guv.rs        explicit GUV finite-field construction and neighbor caches
├── recovery.rs   nested recovery, decoders, v0.5 transform/product/residual caches
├── rect.rs       adaptive rectangular multiplication
├── sketch.rs     cheap sketch probe and Graia q/p/t schedule
├── spgemm.rs     hash baseline and legacy exact dense kernel
├── matrix.rs     CSR/dense primitives
├── synthetic.rs  overlap + Walsh-Hadamard sparse-output generators
├── lib.rs
└── main.rs       benchmark CLI
```

## Current research boundary

The major unresolved question is still recovery efficiency. The proof-oriented GUV constants often trigger identity fallback at practical dimensions, while the hash-signature backend has no expander guarantee. v0.5 is designed to make that question measurable without repeatedly paying for invariant matrix products or benchmarking only the dense-column worst case.
