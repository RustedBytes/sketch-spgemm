# Changelog

All notable changes to `sketch-spgemm` are documented in this file.

The project follows Semantic Versioning while it is in the `0.x` development
series. Breaking public API changes may therefore appear in minor releases.

## 0.11.0

### Added

- `CheckedAddScalar` and `CheckedSpGemmScalar` extension traits for custom
  overflow-detecting scalar arithmetic.
- Overflow-detecting sparse and dense exact kernels through
  `try_spgemm_checked`, `try_spgemm_hash_checked`, and
  `try_dense_matmul_checked`.
- Fallible `CsrMatrix::try_from_triplets` construction and a streaming
  row-major `CsrBuilder` with checked duplicate aggregation.
- Python `checked_spgemm`, `SpGemmStats`, `CsrBuilder`, and
  `CsrMatrix.from_triplets` bindings with NumPy input and mapped Python errors.

### Changed

- The Rust crate, Python package, bindings, and type information were advanced
  to `0.11.0`.

## 0.10.0

### Added

- Scalar-generic `CsrInput`, `spgemm_hash`, `try_spgemm_hash`, and
  `dense_matmul` APIs.
- Scalar-aware `try_spgemm` dispatch: `i64` uses automatic exact/sketch
  selection, while other supported primitive scalars use direct CSR
  multiplication.
- `SpGemmDispatchScalar` for extending high-level dispatch to custom scalar
  types.
- `SpGemmExecutionStats` for reporting whether automatic or direct execution
  was used.
- Checked zero-copy `CsrView` support for external CSR buffers.
- Explicit allocating CSC-to-CSR conversion through `CsrMatrix::from_csc`.
- Scalar-generic borrowed `sprs` views and `petgraph` adjacency conversion.

### Changed

- `CsrInput` now exposes its scalar as an associated type.
- Sketch recovery and residual fingerprints remain specialized to exact `i64`
  arithmetic; other scalars transparently use the direct high-level path.
- Custom benchmark timing was replaced with Criterion suites for the core,
  `sprs`, and `petgraph` workloads.
- The Rust crate, Python package, and bindings were advanced to `0.10.0`.

### Compatibility

- External `CsrInput` implementations must define `type Scalar`.
- Explicit `SprsCsrView` type annotations must account for the new scalar type
  parameter.

## 0.9.0

### Added

- Optional zero-copy `sprs` CSR input and native-output integration.
- Optional weighted `petgraph` adjacency and two-hop path-count integration.
- Public `CsrInput` abstraction and fallible borrowed-input automatic APIs.
- Structured dimension, storage, index-overflow, and output-construction
  errors.
- End-to-end `sprs` and `petgraph` comparison benchmarks.
- PyO3 and NumPy bindings with Python type information and tests.

### Changed

- The minimum supported Rust version was raised to 1.91.

## 0.8.0

### Added

- Generic `CsrMatrix<T>` and `DenseMatrix<T>` containers with backward-
  compatible `i64` defaults.
- Shared `MatrixLike` metadata and an owning `Matrix<T>` boundary enum.
- Explicit `Scalar` alias for recovery arithmetic.
- Library-first packaging, public API documentation, and runnable examples.

### Changed

- Licensing was simplified to MIT.

## 0.7.1

### Changed

- Added a structural-amplification prefilter and staged row sampling to reduce
  automatic-selection overhead.
- Replaced sampled-row hash accumulation with dense scratch storage and touched
  column tracking.
- Fused fingerprint lanes into one traversal of each sparse input.
- Replaced generic remainder operations in the hot fingerprint path with fast
  Mersenne reduction.
- Added detailed analysis, sampling, recovery, fingerprint, and fallback
  timings to `AutoSpGemmStats`.

## 0.7.0

### Added

- `AutoSpGEMM` workload analysis and automatic exact/sketch selection without
  requiring the true output sparsity.
- Independent bilinear residual fingerprints over the Mersenne prime
  `2^61 - 1`.
- Exact fallback when a recovered candidate fails certification.
- `PreparedFactor` caching for dense factors and their sparse row views.
- Adaptive dense, sparse-left, sparse-right, and sparse-sparse rectangular
  kernels.

## 0.6.1

### Added

- Three-moment sparse-recovery backend using `S0`, `S1`, and `S2` singleton
  decoding.
- Masked residual multiplication based on observed active output columns.
- Observed-geometry scheduling for practical recovery rounds.
- Optional exact correction when a conservative recovery bound is required.

### Changed

- Retained and expanded recovery-matrix and factor caches introduced in 0.5.0.

## 0.5.0

### Added

- Multi-level reuse caching for recovery matrices, sketched factors,
  measurements, and residual generations.
- Early termination after full identity recovery.
- Faster right-sketch construction through precomputed column mappings.
- Adaptive exact baseline alongside hash-based sparse multiplication.
- Walsh-Hadamard sparse-output synthetic workloads for cancellation-heavy
  benchmarks.
