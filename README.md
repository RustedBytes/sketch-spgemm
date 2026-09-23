# SketchSpGEMM

[![Crates.io](https://img.shields.io/crates/v/sketch-spgemm.svg)](https://crates.io/crates/sketch-spgemm)
[![PyPI version](https://img.shields.io/pypi/v/sketch-spgemm.svg)](https://pypi.org/project/sketch-spgemm/)
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
> Matrix containers and direct multiplication kernels are scalar-generic.
> Automatic sketch recovery and fingerprints use exact `i64` arithmetic. This
> is not a general tensor library, GPU kernel, or LLM inference engine. See
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

- Rust 1.91 or newer with Cargo

Clone the repository, run the tests, and execute the default benchmark:

```bash
cargo test
cargo bench --bench benchmark
```

### Python bindings

The Python package supports CPython 3.9 or newer. CSR values use NumPy `int32`,
`int64`, `uint64`, `float32`, or `float64` buffers, while column indices and row
pointers always use `int64`. Build and install it into the active virtual
environment with [Maturin](https://www.maturin.rs/):

```bash
python -m pip install maturin
maturin develop --release
```

```python
import numpy as np
from sketch_spgemm import CsrMatrix, auto_spgemm

a = CsrMatrix(
    np.array([2, 3, 4], dtype=np.int64),
    np.array([0, 1, 1], dtype=np.int64),
    np.array([0, 2, 3], dtype=np.int64),
    (2, 2),
)
b = CsrMatrix(
    np.array([5, 7, 11], dtype=np.int64),
    np.array([0, 0, 1], dtype=np.int64),
    np.array([0, 1, 3], dtype=np.int64),
    (2, 2),
)

product, stats = auto_spgemm(a, b)
np.testing.assert_array_equal(product.to_dense(), [[31, 33], [28, 44]])
print(stats.choice, stats.timing.total)
```

`data` must be a contiguous one-dimensional `int32`, `int64`, `uint64`,
`float32`, or `float64` array. `indices` and `indptr` remain contiguous
one-dimensional `int64` arrays. Column indices in each row must be strictly
increasing, duplicates and explicit zero values are rejected, and the
constructor copies all input data. `auto_spgemm` and `analyze_workload` require
`int64` matrices; `checked_spgemm` supports every listed dtype and requires both
operands to have the same dtype.

Python also exposes the checked exact kernel and fallible COO construction:

```python
from sketch_spgemm import CsrBuilder, CsrMatrix, checked_spgemm

safe_product, direct_stats = checked_spgemm(a, b)

matrix = CsrMatrix.from_triplets(
    np.array([4, -1, 7], dtype=np.int64),
    np.array([0, 0, 2], dtype=np.int64),
    np.array([1, 1, 0], dtype=np.int64),
    (3, 3),
)

builder = CsrBuilder(3, 3, capacity=3, dtype="int64")
builder.extend(
    np.array([4, -1], dtype=np.int64),
    np.array([0, 0], dtype=np.int64),
    np.array([1, 1], dtype=np.int64),
)
builder.push(2, 0, 7)
streamed_matrix = builder.finish()
```

`CsrBuilder` requires globally row-major sorted coordinates across every
`extend` and `push` call. Duplicate aggregation and checked multiplication
raise Python `OverflowError`; malformed coordinates raise `ValueError`.
`extend` is streaming rather than transactional: if a later coordinate fails,
the successfully processed prefix remains in the builder.

Direct multiplication and common transaction-graph transformations are
available for every supported dtype. `spgemm` follows the ordinary arithmetic
semantics of the selected dtype; use `checked_spgemm` when integer overflow or
non-finite floating-point results must be reported:

```python
import numpy as np

from sketch_spgemm import from_scipy, spgemm, to_scipy

two_hop, stats = spgemm(matrix, matrix, max_output_nnz=1_000_000)
incoming = matrix.transpose()
degrees = matrix.row_nnz()
outflow = matrix.row_sums()
binary = matrix.binarize()
combined = matrix.checked_add(matrix)
focused = matrix.select_rows(np.array([2, 0, 2], dtype=np.int64))

# SciPy remains optional; install with `pip install sketch-spgemm[scipy]`.
scipy_csr = to_scipy(matrix)
native = from_scipy(scipy_csr)
```

An exceeded `max_output_nnz` raises `MemoryError` before the completed row is
appended to the result. SciPy adapters copy into canonical owned storage.

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

For borrowed CSR implementations, use the fallible generic entry point:

```rust
use sketch_spgemm::{try_auto_spgemm, AutoSpGemmConfig, CsrInput};

fn multiply<A, B>(a: &A, b: &B)
where
    A: CsrInput<Scalar = i64>,
    B: CsrInput<Scalar = i64>,
{
    let (product, stats) =
        try_auto_spgemm(a, b, AutoSpGemmConfig::default()).unwrap();
    println!("{} nonzeros via {:?}", product.nnz(), stats.choice);
}
```

`CsrInput` has an associated scalar type and requires canonical CSR rows:
sorted unique columns and no explicit zeros. It lets external sparse containers
participate without first copying their complete input into `CsrMatrix`.
`try_auto_spgemm` requires `Scalar = i64`. The scalar-aware `try_spgemm` entry
point runs that automatic pipeline for `i64` and transparently uses the direct
kernel for other supported scalar types.
External containers exposing `usize` CSR buffers can use the checked,
zero-copy `CsrView` adapter instead of defining a dedicated wrapper.

## Ecosystem integrations

Integrations are optional and disabled by default, keeping the core crate's
dependency graph empty:

```bash
cargo add sketch-spgemm --features sprs
cargo add sketch-spgemm --features petgraph
```

The repository includes complete runnable examples:

```bash
cargo run --example sprs --features sprs
cargo run --example petgraph --features petgraph
```

It also includes end-to-end Criterion throughput comparisons:

```bash
cargo bench --bench sprs --features sprs
cargo bench --bench petgraph --features petgraph
```

Each benchmark first checks that the standard ecosystem result and the
`sketch-spgemm` result are identical. Criterion then performs warm-up and
statistical sampling, and reports timing and throughput estimates.
The `sprs` comparison uses native CSR multiplication. The `petgraph` comparison
uses direct weighted two-hop edge traversal. Both measurements are end to end,
including output construction and, for `sketch-spgemm`, workload selection and
adapter overhead. Pass Criterion options after `--`, for example
`--measurement-time 5` or `--sample-size 50`.

### `sprs`

The `sprs` feature provides a generic `SprsCsrView<'_, T, I, Iptr>` for borrowed
CSR operands. It can be passed directly to generic direct kernels without
copying input values, indices, or row pointers. The `sprs::auto_spgemm`
convenience function remains specialized to `i64` and returns an owning
`CsMatI<i64, I, Iptr>`. CSC is rejected because converting it would violate the
zero-copy CSR contract. Explicit stored zeros are ignored.

```rust
use sketch_spgemm::AutoSpGemmConfig;
use sketch_spgemm::interop::sprs::auto_spgemm;
use sprs::CsMat;

let a = CsMat::new((1, 2), vec![0, 2], vec![0, 1], vec![2_i64, 3]);
let b = CsMat::new((2, 1), vec![0, 1, 2], vec![0, 0], vec![5_i64, 7]);
let (c, _) = auto_spgemm(a.view(), b.view(), AutoSpGemmConfig::default())?;
assert_eq!(c.get(0, 0), Some(&31));
# Ok::<(), sketch_spgemm::SpGemmError>(())
```

This is a downstream adapter in `sketch-spgemm`; neither `sprs` nor its public
API needs to change.

### `petgraph`

The `petgraph` feature exposes `adjacency_csr` and `two_hop_path_counts` for any
graph view implementing `IntoNodeIdentifiers + IntoEdges + NodeIndexable`.
Edge weights are mapped to `i64`; parallel weights sum, undirected adjacency is
symmetric, and `NodeIndexable::node_bound` preserves vacant `StableGraph`
indices. The product remains `sketch_spgemm::CsrMatrix<i64>` for subsequent
library operations.

```rust
use petgraph::graph::DiGraph;
use sketch_spgemm::AutoSpGemmConfig;
use sketch_spgemm::interop::petgraph::two_hop_path_counts;

let mut graph = DiGraph::<(), i64>::new();
let a = graph.add_node(());
let b = graph.add_node(());
let c = graph.add_node(());
graph.add_edge(a, b, 2);
graph.add_edge(b, c, 7);

let (paths, _) = two_hop_path_counts(
    &graph,
    |weight| *weight,
    AutoSpGemmConfig::default(),
)?;
assert_eq!(paths.row(a.index()).collect::<Vec<_>>(), vec![(c.index(), 14)]);
# Ok::<(), sketch_spgemm::SpGemmError>(())
```

This integration works entirely through `petgraph` visitor traits, so no
upstream storage exposure or API adjustment is required.

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
- `CsrInput::Scalar` identifies the value type exposed by a borrowed CSR input.
- `SpGemmScalar` is the minimal scalar contract for direct sparse and dense
  multiplication.
- `Scalar` is the `i64` type used by automatic sketch recovery and fingerprints.

COO input can be canonicalized with `CsrMatrix::from_triplets`. CSC buffers can
be converted once with `CsrMatrix::from_csc`; the conversion is intentionally
allocating because all sparse kernels use row-oriented access.

Generic direct multiplication accepts any numeric type satisfying
`SpGemmScalar`, including integer and floating-point primitives:

```rust
use sketch_spgemm::{try_spgemm_hash, CsrMatrix};

let a = CsrMatrix::<i32>::from_triplets(1, 2, &[(0, 0, 2), (0, 1, 3)]);
let b = CsrMatrix::<i32>::from_triplets(2, 1, &[(0, 0, 5), (1, 0, 7)]);
let (c, _) = try_spgemm_hash(&a, &b)?;
assert_eq!(c.values, vec![31]);
# Ok::<(), sketch_spgemm::SpGemmError>(())
```

Here "direct" means that no probabilistic sketch is used. Unchecked
floating-point arithmetic retains the usual primitive rounding and NaN
behavior. Canonical sparse output omits values that compare equal to zero, so
neither `0.0` nor `-0.0` is stored as an explicit CSR entry. Checked
floating-point APIs additionally reject non-finite multiplication or
accumulation results.

For overflow-sensitive workloads, `try_spgemm_checked` and
`try_dense_matmul_checked` use exact direct kernels and return
`SpGemmError::ArithmeticOverflow` with the output coordinate and inner index:

```rust
use sketch_spgemm::{try_spgemm_checked, CsrMatrix};

let a = CsrMatrix::<i64>::try_from_triplets(1, 2, &[(0, 0, 2), (0, 1, 3)])?;
let b = CsrMatrix::<i64>::try_from_triplets(2, 1, &[(0, 0, 5), (1, 0, 7)])?;
let (c, _) = try_spgemm_checked(&a, &b)?;
assert_eq!(c.values, vec![31]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Each multiplication and intermediate accumulation is checked in canonical
inner-index order. The kernels intentionally do not widen the accumulator, so
temporary overflow is reported even if later cancellation would make the final
mathematical sum fit in the scalar type. Use a wider scalar type when that
behavior is required.

`CsrBuilder` constructs canonical CSR incrementally from row-major sorted
triplets without retaining the complete COO input. It combines duplicates with
checked addition, removes resulting zeros, preserves empty rows, and reports
out-of-range or out-of-order coordinates:

```rust
use sketch_spgemm::CsrBuilder;

let mut builder = CsrBuilder::<i64>::with_capacity(3, 3, 4);
builder.try_extend([
    (0, 1, 4),
    (0, 1, -1),
    (2, 0, 7),
])?;
let matrix = builder.finish();
assert_eq!(matrix.row_ptr, vec![0, 1, 1, 2]);
assert_eq!(matrix.values, vec![3, 7]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Use `CsrMatrix::try_from_triplets` when input coordinates are not sorted. That
path allocates per-row maps; `try_from_sorted_triplets` and `CsrBuilder` are the
streaming alternatives for already ordered data. `CsrBuilder::try_extend` is
non-transactional: entries preceding an error remain applied to the builder.

Canonical CSR matrices also provide `transpose`, `try_add`, `structural`,
`select_rows`, `row_nnz`, `column_nnz`, `try_row_sums`, and `try_column_sums`.
Direct products can enforce an output budget through `SpGemmOptions`. For
domain-specific path algebras, implement `Semiring<T>` and call
`try_spgemm_semiring`. Inputs can be losslessly converted to a wider checked
output type with
`try_spgemm_checked_with_accumulator`, such as `i64` inputs accumulated into an
`i128` result.

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

Relevant `AutoSpGemmConfig` fields:

```text
residual_fingerprint: bool
fingerprint_lanes: usize
fingerprint_seed: u64     // 0 selects a runtime-derived seed
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
cargo bench --bench benchmark -- 'sparse-product/.*/sparse-output'
```

The benchmark uses fixed deterministic overlap and sparse-output workloads and
checks every measured implementation against the exact CSR result before
sampling. Run the complete suite with `cargo bench --bench benchmark`, or tune
Criterion after `--`, for example with `--measurement-time 5 --sample-size 50`.

Criterion stores detailed reports under `target/criterion/` when its plotting
backend is available.

## Public API

```text
auto_spgemm(...)                   automatic exact/sketch selection
try_auto_spgemm(...)               fallible i64 automatic entry point
try_spgemm(...)                    scalar-aware automatic/direct dispatch
analyze_workload(...)              workload estimator
try_analyze_workload(...)          fallible generic workload estimator
nested_spgemm(...)                 theorem-oriented control flow
nested_spgemm_with_policy(...)     custom rectangular policy
nested_spgemm_with_options(...)    engineering controls
adaptive_matmul(...)               one-shot rectangular multiplication
adaptive_matmul_prepared(...)      cached-factor rectangular multiplication
spgemm_hash(...)                   direct CSR baseline
try_spgemm_hash(...)               fallible scalar-generic CSR baseline
try_spgemm_checked(...)            overflow-detecting exact CSR product
try_spgemm_hash_checked(...)       checked hash-accumulator kernel
try_spgemm_hash_checked_with_options(...) checked product with output budget
try_spgemm_hash_with_options(...)  direct product with output budget
try_spgemm_semiring(...)           configurable path algebra
try_spgemm_checked_with_accumulator(...) widened checked result
try_dense_matmul_checked(...)      checked dense product
CsrMatrix::try_from_triplets(...)  checked unsorted COO conversion
CsrBuilder                         checked streaming sorted COO conversion
CsrMatrix::transpose/try_add(...)  canonical graph transformations
CsrMatrix::try_row_sums/try_column_sums(...) checked reductions
CsrMatrix::select_rows(...)        checked row selection
```

## Changelog

Release-to-release changes are maintained in [changelog.md](changelog.md).

## Limitations

- Automatic sketch recovery and fingerprint APIs currently operate on exact
  signed 64-bit integers. Other scalar types use the direct kernels.
- Automatic sketch execution follows ordinary `i64` overflow semantics;
  callers requiring recoverable overflow must use the checked exact kernels.
- Execution is CPU-only and currently single-process.
- There are no CUDA, Metal, or distributed integrations.
- The practical moment/fingerprint path is probabilistic rather than the
  deterministic theorem from the motivating paper.
- Performance depends strongly on output geometry. Sparse inputs alone do not
  imply that sketch recovery will be beneficial.

## Source layout

```text
src/
├── auto.rs        workload sampling and exact/sketch selection
├── dispatch.rs    scalar-aware automatic/direct dispatch
├── error.rs       structured construction and arithmetic errors
├── fingerprint.rs bilinear residual certificate over 2^61 - 1
├── guv.rs         explicit GUV finite-field construction
├── interop/       optional sprs and petgraph adapters
├── matrix.rs      scalar-generic CSR and dense matrix containers
├── ops.rs         CSR transformations, reductions, and checked addition
├── recovery.rs    recovery backends, masks, schedulers, and caches
├── rect.rs        adaptive rectangular kernels and prepared factors
├── sketch.rs      probes and the Graia q/p/t schedule
├── spgemm.rs      direct, checked, semiring, and output-budget kernels
├── synthetic.rs   synthetic sparse-output workloads
└── lib.rs         public library exports
benches/
├── benchmark.rs   Criterion product and sketch-kernel benchmarks
├── petgraph.rs    Criterion petgraph comparison
└── sprs.rs        Criterion sprs comparison
examples/
├── petgraph.rs    petgraph integration example
└── sprs.rs        sprs integration example
```

## Project status

SketchSpGEMM is suitable for research, reproducible experiments, and evaluation
of high-amplification sparse products. It should be benchmarked and validated on
representative data before being embedded into a larger system.

Contributions are welcome, particularly around generic arithmetic, parallel
kernels, additional datasets, property testing, and reproducible benchmarks.

## License

Licensed under the [MIT License](LICENSE-MIT).
