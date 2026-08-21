//! Adaptive, sketch-based sparse matrix multiplication.
//!
//! `sketch-spgemm` computes exact integer products `C = A * B` for sparse
//! matrices. Its high-level [`auto_spgemm`] entry point estimates the workload,
//! chooses either an exact kernel or compressed moment-sketch recovery, and
//! verifies recovered candidates with independent residual fingerprints.
//!
//! The default feature set has no dependencies. Enable `sprs` for zero-copy
//! borrowed `sprs` CSR operands and native output, or `petgraph` for weighted
//! adjacency and two-hop path-count helpers. See the `interop` module when either
//! feature is enabled.
//!
//! The matrix containers are generic, but the multiplication and recovery
//! algorithms currently operate on [`Scalar`] (`i64`). Arithmetic overflow
//! follows Rust's normal integer-overflow behavior.
//!
//! # Example
//!
//! ```
//! use sketch_spgemm::{auto_spgemm, AutoSpGemmConfig, CsrMatrix};
//!
//! let a = CsrMatrix::from_triplets(
//!     2,
//!     2,
//!     &[(0, 0, 2), (0, 1, 3), (1, 1, 4)],
//! );
//! let b = CsrMatrix::from_triplets(
//!     2,
//!     2,
//!     &[(0, 0, 5), (1, 0, 7), (1, 1, 11)],
//! );
//!
//! let (c, stats) = auto_spgemm(&a, &b, AutoSpGemmConfig::default());
//!
//! assert_eq!(c.to_dense().data, vec![31, 33, 28, 44]);
//! println!("selected path: {:?}", stats.choice);
//! ```
//!
//! See the [project README](https://github.com/RustedBytes/sketch-spgemm#readme)
//! for algorithm details, workload guidance, and benchmark options.

#![forbid(unsafe_code)]

/// Automatic workload analysis and exact/sketch execution selection.
pub mod auto;
/// Errors returned by fallible multiplication and adapter APIs.
pub mod error;
/// Probabilistic residual fingerprints for recovered products.
pub mod fingerprint;
/// Explicit Guruswami–Umans–Vadhan recovery construction.
pub mod guv;
/// Generic dense, CSR, and representation-independent matrix types.
pub mod matrix;
/// Sparse-recovery backends and nested multiplication algorithms.
pub mod recovery;
/// Adaptive rectangular dense/sparse multiplication kernels.
pub mod rect;
/// Implicit sketch maps and the theoretical round schedule.
pub mod sketch;
/// Exact baseline sparse and dense multiplication kernels.
pub mod spgemm;
/// Deterministic synthetic workloads for experiments and benchmarks.
pub mod synthetic;

/// Optional interoperability with sparse-matrix and graph ecosystems.
#[cfg(any(feature = "sprs", feature = "petgraph"))]
pub mod interop;

pub use auto::{
    analyze_workload, auto_spgemm, candidate_product_count, try_analyze_workload, try_auto_spgemm,
    AutoChoice, AutoSpGemmConfig, AutoSpGemmStats, AutoTimingStats, ExactMethod, WorkloadEstimate,
};
pub use error::{MatrixOperand, SpGemmError};
pub use fingerprint::{FingerprintConfig, FingerprintStats, ResidualFingerprint};
pub use guv::{GuvConfig, GuvError, GuvParameters, GuvRecovery};
pub use matrix::{CsrInput, CsrMatrix, CsrRowIter, DenseMatrix, Matrix, MatrixLike, Scalar};
pub use recovery::{
    left_recovery_sketch, nested_spgemm, nested_spgemm_with_options, nested_spgemm_with_policy,
    right_recovery_sketch, right_recovery_sketch_masked, safe_decode_product, safe_decode_scalar,
    BinaryRecoveryMatrix, CorrectionPassStats, MomentConfig, MomentRecovery, NestedOptions,
    NestedRoundStats, NestedSpGemmStats, RecoveryBackend, SignatureConfig, SignatureRecovery,
};
pub use rect::{
    adaptive_matmul, adaptive_matmul_prepared, PreparedFactor, RectangularKernel,
    RectangularPolicy, RectangularStats,
};
pub use sketch::{
    direct_two_sided_sketch, left_sketch, paper_schedule, right_sketch, RoundParams, SketchMap,
};
pub use spgemm::{dense_matmul, spgemm_hash, SpGemmStats};
pub use synthetic::{overlap_problem, sparse_output_problem, SyntheticProblem};
