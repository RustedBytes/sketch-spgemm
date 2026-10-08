# Criterion benchmarks

The existing `benchmark`, `sprs`, and `petgraph` targets cover multiplication,
sketch kernels, and optional integration comparisons. `core_operations` adds
24 cases across four public operations and six deterministic fixtures.

| Group | Timed operation | Throughput unit |
|---|---|---|
| `csr-builder` | Fresh builder, extend, finish, output destruction | Input triplets |
| `csr-transpose` | Transpose and output destruction | Input nonzeros |
| `fingerprint-setup` | Certificate construction and destruction | A+B input nonzeros |
| `fingerprint-verify` | Full successful certificate verification, including temporary allocation | Candidate nonzeros |

Each group has small (32), medium (256), and large (2048) square matrices.
`band` uses 16 consecutive wraparound columns per row. `hub` uses two columns
per ordinary row and up to 64 columns every 32nd row. Values are deterministic
signed integers of magnitude 1–7. Builder streams contain three sorted entries
per coordinate (`v`, `-v`, `v`) to exercise duplicates and cancellation.
Fingerprint fixtures use `B = 2I`, three lanes, and seed `20261008`.

Fixture construction and correctness assertions are outside timing. The product
oracle doubles each original entry without calling multiplication. Transpose
is checked against coordinate reversal; streamed construction is checked
against the original canonical matrix. Both accepting the product and rejecting
the original undoubled matrix are asserted before measurement. Fixture inputs
are reused, so these measure repeated warm workloads, not cold-storage latency.
All operations use `iter`; costs of newly allocated results and their destruction
are included. There is no preallocated-buffer or zero-allocation claim.
Throughput is an operation-specific normalization, not a count of internal
modular arithmetic instructions. The diagonal product deliberately isolates
certificate costs; use the multiplication suite for product geometry comparisons.

Compile and smoke-test:

```sh
cargo bench -p sketch-spgemm --bench core_operations --no-run
cargo test -p sketch-spgemm --bench core_operations
```

Measure all cases using Criterion's default 100 samples, 3-second warm-up,
5-second measurement, and 95% confidence intervals:

```sh
cargo bench -p sketch-spgemm --bench core_operations -- --save-baseline reference-UNIQUE
```

Limit a run with a group filter, e.g. `-- fingerprint-verify`. Smoke runs do not
supply performance evidence. Reports and raw samples stay under
`target/criterion/`; do not commit them.

For a revision comparison, apply this identical harness to both revisions,
record their SHAs and dirty patches, and use the same host, CPU affinity,
compiler, lockfile, features, allocator, release profile and RUSTFLAGS. Build
sequentially and keep a shared Criterion output directory. Choose a fresh
baseline name because `--save-baseline` can overwrite an existing baseline.
Run the candidate with `--baseline reference-UNIQUE`. Repeat material differences
in alternating order and report confidence intervals and practical magnitude.
A useful change threshold for this suite is 5% time reduction, subject to
repeatability; statistical significance alone is insufficient. These estimates
are not request p95/p99 latency. Measure allocation counts separately if needed.

The new target uses the existing Criterion 0.8.2 dependency (crate MSRV 1.86),
without changing the project's Rust 1.91 requirement or adding dependencies.
No automatic performance gate or scheduled benchmark workflow is added.

## Initial measurement

One complete run on base `a31bd0d` with this added harness, Rust 1.99.0 /
LLVM 23.1.1, AMD EPYC 9V74 VM on CPU 2, default system allocator, default
features, thin LTO and one codegen unit. RUSTFLAGS was empty. Criterion used
its defaults; `RAYON_NUM_THREADS=1` limited statistical analysis. No other
benchmark/build was started concurrently. These are host-specific initial
medians with 95% confidence intervals, not before/after speedup claims.

| Operation (2048×2048 band, 32768 nonzeros) | Median | 95% CI |
|---|---:|---:|
| `csr-builder` | 280.90 µs | 279.69–281.88 µs |
| `csr-transpose` | 69.25 µs | 68.89–69.68 µs |
| `fingerprint-setup` | 1291.58 µs | 1288.05–1293.88 µs |
| `fingerprint-verify` | 1852.62 µs | 1831.55–1876.51 µs |

Baseline: `initial-a31bd0d-core-20261008T1320`. All 24 cases completed. Samples/outliers are retained
in Criterion output; a single VM run cannot establish repeatability or tail
latency. No allocation counts were measured.
