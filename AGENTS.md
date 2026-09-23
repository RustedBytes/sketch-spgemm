# Repository guide for coding agents

These instructions apply to the entire repository. Preserve unrelated user
changes and do not commit, tag, publish, or push unless the user explicitly
requests it.

## Project overview

`sketch-spgemm` is a Rust-first sparse matrix multiplication library with a
PyO3/NumPy extension. The core crate provides canonical CSR containers, exact
direct kernels, adaptive sketch-based multiplication, optional ecosystem
interop, and Criterion benchmarks. The Python package exposes a deliberately
smaller API over the same core implementation.

Important paths:

- `src/matrix.rs`: generic dense/CSR storage, checked construction, and
  `CsrBuilder`.
- `src/spgemm.rs`: direct and checked sparse/dense multiplication.
- `src/auto.rs`, `src/recovery.rs`, `src/sketch.rs`, `src/fingerprint.rs`:
  `i64` automatic sketch selection and recovery.
- `src/interop/`: optional `sprs` and `petgraph` adapters.
- `python/src/lib.rs`: PyO3 runtime bindings and dtype dispatch.
- `python/sketch_spgemm/_sketch_spgemm.pyi`: public extension type contract.
- `python/tests/`: Python integration and wheel-level behavior tests.
- `benches/`: Criterion suites; do not introduce custom wall-clock harnesses.
- `.github/workflows/`: Python CI and tagged release publication.

## Architectural invariants

- Keep the core matrix and direct-kernel APIs scalar-generic. Sketch recovery,
  fingerprints, and automatic selection intentionally remain specialized to
  exact signed `i64` arithmetic.
- Canonical CSR rows have strictly increasing column indices, contain no
  explicit zeros, and use a monotonic `row_ptr` of length `rows + 1` whose last
  entry equals `nnz`.
- `CsrBuilder` input is globally nondecreasing in `(row, column)` order across
  all `try_push` and `try_extend` calls. Duplicate coordinates are combined and
  canceled zeros are removed.
- `CsrBuilder::try_extend` is streaming and non-transactional. On error, its
  successfully processed prefix remains applied; do not silently change this
  contract.
- Checked arithmetic validates every multiplication and intermediate addition
  in iteration order. It does not use a widened accumulator to allow temporary
  overflow followed by cancellation.
- Checked fallible entry points must return structured errors rather than panic
  or silently wrap. Preserve output coordinates and the inner index in
  arithmetic errors.
- Keep performance-sensitive dispatch outside inner loops. Avoid allocating or
  cloning complete matrices in hot kernel paths unless the API explicitly
  promises conversion.
- The Rust crate forbids unsafe code. Do not weaken `#![forbid(unsafe_code)]`.

## Rust workflow

Use Rust 1.91-compatible language and library features; `1.91` is the current
minimum supported Rust version (MSRV). After Rust changes, run the smallest
relevant tests while iterating and the following baseline before handoff:

```text
cargo fmt --all --check
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features
```

Clippy currently reports pre-existing warnings in older core modules. Do not
introduce new warnings in touched code, and do not perform unrelated cleanup
merely to make a focused change pass with `-D warnings`.

Additional checks by change type:

- MSRV or dependency changes: `cargo +1.91 check --workspace --all-features`.
- Feature/interop changes: test both `--features sprs` and
  `--features petgraph`, or use `--all-features`.
- Benchmark changes: `cargo bench --workspace --all-features --no-run` and run
  the affected Criterion benchmark when reporting performance.
- Packaging changes: `cargo package --allow-dirty` and inspect the packaged
  file list.
- Public Rust examples or rustdoc changes: ensure doctests pass as part of
  `cargo test`.

Do not edit `Cargo.lock` by hand. Let Cargo update it and review dependency
changes separately from source changes.

## Python bindings

The supported matrix value dtypes are exactly `int32`, `int64`, `uint64`,
`float32`, and `float64`. CSR indices, COO row/column indices, and row pointers
remain contiguous one-dimensional `int64` NumPy arrays.

- Preserve dtype through construction, `CsrBuilder`, `to_arrays`, `to_dense`,
  and multiplication results.
- Reject unsupported or mixed operand dtypes explicitly; do not silently cast.
- `checked_spgemm` supports every listed dtype.
- `auto_spgemm` and `analyze_workload` are `int64`-only because they use the
  sketch pipeline.
- Preserve exception categories: invalid shapes/order/storage are `ValueError`,
  unsupported or mixed dtypes are `TypeError`, checked arithmetic failures are
  `OverflowError`, output-budget failures are `MemoryError`, and
  consumed-builder lifecycle errors are `RuntimeError`.
- When adding or changing a Python symbol, update all of: the PyO3 module,
  `python/sketch_spgemm/__init__.py`, `__init__.pyi`,
  `_sketch_spgemm.pyi`, documentation, and tests as applicable.
- Keep Python compatibility at 3.9 or newer. Do not use source syntax that
  raises the minimum version unintentionally.

Format and validate Python sources and stubs with:

```text
ruff format python
ruff check python
pyright
```

Run Python tests against a freshly built extension, not a previously installed
wheel. A CI-equivalent local sequence is:

```text
python -m pip install "maturin>=1,<2" "numpy>=1.23" "pytest>=7" "pyright>=1.1.405" "ruff>=0.13"
wheel_dir="$(mktemp -d)"
maturin build --release --out "$wheel_dir"
python -m pip install --force-reinstall --no-deps "$wheel_dir"/*.whl
python -m pytest -q python/tests
```

Run Pyright from the same activated environment that contains NumPy so import
resolution matches CI.

## Testing expectations

- Add a regression test for every bug fix and boundary behavior.
- For arithmetic changes, cover multiplication overflow, accumulation
  overflow, zeros/cancellation, empty shapes, and dimension mismatch as
  relevant.
- For CSR construction, cover empty rows, duplicates, out-of-order input,
  bounds, explicit-zero removal, and builder lifecycle.
- For dtype work, test each supported dtype and verify both numerical output
  and returned NumPy dtype. Also test unsupported and mixed dtypes.
- Prefer deterministic seeds for randomized/property-style tests.
- Compare optimized or probabilistic paths against a simple exact reference;
  do not only compare two implementations that share the same accumulator.
- Do not weaken assertions, lint rules, type checking, or CI coverage to make a
  change pass.

## Documentation, versions, and releases

- Update `README.md` for user-visible behavior and `changelog.md` for notable
  release changes.
- Keep examples executable and aligned with the actual API.
- A version bump must stay synchronized in root `Cargo.toml`,
  `python/Cargo.toml`, `pyproject.toml`, `Cargo.lock`, and the Python module's
  `__version__` value.
- Release tags use `v*`; `.github/workflows/release.yml` validates versions,
  builds wheels/sdist, creates GitHub release assets, and publishes to PyPI.
- Never create a release tag or publish artifacts unless explicitly requested.

## Change hygiene

- Keep patches focused and preserve backward compatibility unless a breaking
  change is intentional and documented.
- Preserve optional-feature boundaries; default builds must not require
  `sprs`, `petgraph`, Python, or benchmark-only dependencies.
- Do not commit generated wheels, compiled extensions, `target/`, virtual
  environments, caches, or benchmark output.
- Before handoff, run `git diff --check`, summarize the validations actually
  run, and disclose any known pre-existing failures separately from new ones.
