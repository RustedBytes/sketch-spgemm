pub mod auto;
pub mod fingerprint;
pub mod guv;
pub mod matrix;
pub mod recovery;
pub mod rect;
pub mod sketch;
pub mod spgemm;
pub mod synthetic;

pub use auto::{
    analyze_workload, auto_spgemm, candidate_product_count, AutoChoice, AutoSpGemmConfig,
    AutoSpGemmStats, ExactMethod, WorkloadEstimate,
};
pub use fingerprint::{FingerprintConfig, FingerprintStats, ResidualFingerprint};
pub use guv::{GuvConfig, GuvError, GuvParameters, GuvRecovery};
pub use matrix::{CsrMatrix, DenseMatrix};
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
