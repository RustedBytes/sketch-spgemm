# SketchSpGEMM

[![Crates.io](https://img.shields.io/crates/v/sketch-spgemm.svg)](https://crates.io/crates/sketch-spgemm)
[![Documentation](https://docs.rs/sketch-spgemm/badge.svg)](https://docs.rs/sketch-spgemm)
[![License: MIT](https://img.shields.io/crates/l/sketch-spgemm.svg)](#license)

Adaptive, sketch-based sparse matrix multiplication in Rust.

SketchSpGEMM is a research prototype for computing `C = A × B` when `A` and
`B` are sparse, the ordinary multiplication exposes many candidate scalar
products, and the final matrix `C` is much sparser than that intermediate work
suggests. It automatically chooses between exact multiplication and compressed
moment-sketch recovery, then checks a recovered result with independent
residual fingerprints.

The project is inspired by Graia,
[*Optimal Deterministic Fully Sparse Matrix Multiplication*](https://arxiv.org/abs/2608.18496),
while its practical path uses an IBLT-style moment recovery strategy rather
than being a literal implementation of the deterministic theorem.

> [!IMPORTANT]
> The matrix containers are generic, but the current multiplication, recovery,
> and fingerprint algorithms use exact `i64` arithmetic. This is not a general
> tensor library, GPU kernel, or LLM inference engine. See
> [Limitations](#limitations).

## Why SketchSpGEMM?

A conventional sparse matrix product may perform a large number of candidate
multiplications even when relatively few nonzeros survive in the output. Define

```text
F   = number of candidate scalar products
K   = nnz(C)
rho = F / K
```

When `rho` is large and the output is sparse, compressed measurements and
sparse recovery can avoid materializing much of the intermediate work.
SketchSpGEMM analyzes the input structure, samples output rows, predicts
recovery cost, and selects the appropriate execution path without being given
the true `K`.

## Applications

The algorithm is intended for exact sparse products with high candidate-product
amplification and a sparse final result. Candidate application areas include:

- **Graph analytics** — sparse two-hop relations, path-count products, and
  signed or weighted graph composition when the resulting relation remains
  sparse.
- **Sparse relational joins** — incidence-matrix formulations of joins or
  grouped aggregates where many input matches collapse into relatively few
  output pairs.
- **Incremental computation** — products such as `ΔA × B` or `A × ΔB` when a
  sparse update affects only a small part of the result.
- **Discrete scientific models** — composition of integer-valued incidence,
  connectivity, boundary, or other sparse combinatorial operators.
- **Sparse polynomial and combinatorial algebra** — exact integer products in
  which many generated terms combine into a small output support.
- **Sparse-recovery research** — experiments with sketch schedules, peeling
  decoders, residual certification, and output-sensitive SpGEMM.

These are workload shapes, not claims of production readiness. Measure the
actual `F`, output geometry, recovery cost, and fallback rate for a particular
dataset before choosing the sketch path.

### Good fit

- Exact integer arithmetic is acceptable.
- Both inputs are sparse.
- Candidate work is much larger than the number of output nonzeros.
- Surviving output columns contain relatively few nonzeros.
- Batch throughput matters more than single-operation latency.

### Poor fit

- The output is dense or nearly dense.
- The candidate-product amplification ratio is small.
- Floating-point, quantized, GPU, or distributed execution is required.
- The workload is a dense neural-network layer or a full model-inference task.
- A probabilistic certificate is unacceptable and exact correction is disabled.

For unsuitable inputs, `auto_spgemm` can select an exact kernel rather than
forcing sketch recovery.

## Quick start

Requirements:

- Rust stable toolchain with Cargo

Clone the repository, run the tests, and execute the default benchmark:

```bash
cargo test
cargo run --release
```

### Library example

Add the library to a Cargo project:

```bash
cargo add sketch-spgemm
```

```rust
use sketch_spgemm::{auto_spgemm, AutoSpGemmConfig, CsrMatrix};

fn main() {
    let a = CsrMatrix::from_triplets(
        2,
        2,
        &[(0, 0, 2), (0, 1, 3), (1, 1, 4)],
    );
    let b = CsrMatrix::from_triplets(
        2,
        2,
        &[(0, 0, 5), (1, 0, 7), (1, 1, 11)],
    );

    let (c, stats) = auto_spgemm(&a, &b, AutoSpGemmConfig::default());

    assert_eq!(c.to_dense().data, vec![31, 33, 28, 44]);
    println!("selected path: {:?}", stats.choice);
}
```

The main automatic API is:

```rust
let (c, stats) = auto_spgemm(&a, &b, AutoSpGemmConfig::default());
```

It does not receive the true product or `nnz(C)`.

## Matrix types

The storage containers share a common scalar default and metadata interface:

```rust
use sketch_spgemm::{CsrMatrix, DenseMatrix, Matrix, MatrixLike, Scalar};

fn metadata<M: MatrixLike>(matrix: &M) -> ((usize, usize), usize) {
    (matrix.shape(), matrix.nnz())
}

let mut dense = DenseMatrix::<i32>::zeros(2, 2);
dense[(0, 1)] = 7;

let csr: CsrMatrix<i32> = dense.to_csr();
let matrix: Matrix<i32> = csr.into();

assert_eq!(metadata(&matrix), ((2, 2), 1));

let exact_value: Scalar = 7i64;
assert_eq!(exact_value, 7);
```

- `CsrMatrix<T = Scalar>` stores compressed sparse rows.
- `DenseMatrix<T = Scalar>` stores contiguous row-major values.
- `Matrix<T = Scalar>` is an owning enum for interfaces that accept either
  representation.
- `MatrixLike` exposes shared `rows`, `cols`, `shape`, and `nnz`
  metadata.
- `Scalar` is the `i64` type used by the current algorithms.

Kernels continue to accept concrete dense or CSR types so representation
dispatch happens outside performance-sensitive inner loops.

## How automatic execution works

`AutoSpGEMM`:

1. Counts candidate products `F` from row degrees in `O(nnz(A))`.
2. Applies a structural amplification prefilter.
3. Exactly multiplies a staged sample of evenly spaced rows of `A`.
4. Estimates output nonzeros, active columns, output density, and `rho`.
5. Predicts the moment-recovery row count at a likely useful schedule point.
6. Chooses exact execution or moment-sketch recovery.
7. Certifies a recovered candidate with residual fingerprints.
8. Optionally falls back to exact multiplication if certification fails.

The default selector prefers sketching only when all of these conditions appear
favorable:

```text
estimated rho                 >= 256
estimated avg nnz/active col  <= 64
estimated output density      <= 10%
estimated moment rows / rows  <  80%
```

These are configurable engineering defaults, not theorem constants.

The exact branch is adaptive. Below `exact_dense_cell_limit`—16 million total
`A + B + C` dense cells by default—it uses the adaptive rectangular kernel.
Above that limit it remains in CSR and uses hash SpGEMM instead of forcing a
large dense allocation.

## Residual certification

SketchSpGEMM can test whether a reconstructed candidate `D` satisfies
`AB - D = 0` without knowing the exact `K`.

For each independent lane it selects random field weights `r_i` and `s_j`
modulo the Mersenne prime `2^61 - 1` and computes:

```text
phi(AB) = r^T A B s = (r^T A)(B s)
phi(D)  = sum_(i,j in supp(D)) r_i * D_ij * s_j
```

The candidate passes when `phi(D) == phi(AB)` in every lane. Target setup is
linear in the input nonzeros, and candidate checking is linear in `nnz(D)`.
Three independently seeded lanes are enabled by default.

This is a **probabilistic residual certificate**. Exact identity recovery and
exact correction remain available when deterministic verification is required.

Relevant CLI options:

```text
--residual-fingerprint true|false
--fingerprint-lanes N
--fingerprint-seed N     # 0 selects a runtime-derived seed
```

## Recovery backends

- `moment` — three-row-per-bucket algebraic moment peeling; the practical
  default.
- `signature` — deterministic SplitMix bucket graph with binary index
  signatures.
- `guv` — explicit Parvaresh–Vardy/GUV graph with a Bennett decoder.
- `identity` — exact uncompressed recovery reference.

The moment backend stores:

```text
S0 = sum x_i
S1 = sum (i + 1)x_i
S2 = sum (i + 1)^2 x_i
```

It combines peeling with exact remeasurement, support masking, and an
observed-geometry scheduler. Once outer recovery identifies residual output
columns, later multiplication can restrict work to that unresolved support.

## Rectangular kernels

`adaptive_matmul_prepared` selects among:

```text
dense-blocked
sparse-left
sparse-right
sparse-sparse
```

`PreparedFactor` caches sparse row views, row counts, and column counts so
reused recovery factors do not pay repeated conversion costs. The one-shot
`adaptive_matmul` API remains available for standalone and baseline use.

## Recommended sparse-output benchmark

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

A successful automatic run should report a sketch choice, a passing benchmark
oracle check, a successful fingerprint certificate, and an estimated
per-active-column output size near the configured single digits. The overlap
workload with dense surviving columns should instead select exact execution.

The benchmark prints separate timings for analysis, candidate counting, row
sampling, nested recovery, fingerprint setup and checks, and exact fallback.

## Public API

```text
auto_spgemm(...)                   automatic exact/sketch selection
analyze_workload(...)              workload estimator
nested_spgemm(...)                 theorem-oriented control flow
nested_spgemm_with_policy(...)     custom rectangular policy
nested_spgemm_with_options(...)    engineering controls
adaptive_matmul(...)               one-shot rectangular multiplication
adaptive_matmul_prepared(...)      cached-factor rectangular multiplication
spgemm_hash(...)                   exact CSR baseline
```

## What's new in v0.8.0

- Generic `CsrMatrix<T>` and `DenseMatrix<T>` containers with backward-
  compatible `i64` defaults.
- Shared `MatrixLike` metadata and a `Matrix<T>` boundary enum.
- Explicit `Scalar` alias for the exact arithmetic used by recovery.
- Lower-overhead structural analysis and staged row sampling.
- Dense scratch accumulation with touched-column tracking for sampled rows.
- Fused residual-fingerprint lanes over one sparse traversal of each input.
- Fast Mersenne reduction in place of generic 128-bit remainder operations.
- Separate conservative schedule bounds and tighter scheduler hints.
- Detailed timing fields in `AutoSpGemmStats`.

## Limitations

- The matrix containers are generic, but multiplication, recovery, and
  fingerprint APIs currently operate on exact signed 64-bit integers.
- Arithmetic overflow is not converted into a recoverable error.
- Execution is CPU-only and currently single-process.
- There are no CUDA, Metal, distributed, or production storage integrations.
- The practical moment/fingerprint path is probabilistic rather than the
  deterministic theorem from the motivating paper.
- Performance depends strongly on output geometry. Sparse inputs alone do not
  imply that sketch recovery will be beneficial.
- The CLI is a research and benchmark harness, not a production service.

## Source layout

```text
src/
├── auto.rs        workload sampling and exact/sketch selection
├── fingerprint.rs bilinear residual certificate over 2^61 - 1
├── guv.rs         explicit GUV finite-field construction
├── matrix.rs      CSR and dense integer matrix primitives
├── recovery.rs    recovery backends, masks, schedulers, and caches
├── rect.rs        adaptive rectangular kernels and prepared factors
├── sketch.rs      probes and the Graia q/p/t schedule
├── spgemm.rs      hash baseline and simple dense kernel
├── synthetic.rs   synthetic sparse-output workloads
├── lib.rs         public library exports
└── main.rs        benchmark CLI
```

## Project status

SketchSpGEMM is suitable for research, reproducible experiments, and evaluation
of high-amplification sparse products. It should be benchmarked and validated on
representative data before being embedded into a larger system.

Contributions are welcome, particularly around generic arithmetic, parallel
kernels, additional datasets, property testing, and reproducible benchmarks.

## License

Licensed under the [MIT License](LICENSE-MIT).
