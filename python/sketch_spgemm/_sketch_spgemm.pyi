from typing import Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt

Int64Array = npt.NDArray[np.int64]
RectangularPolicy = Literal["auto", "dense", "sparse_left", "sparse_right", "sparse_sparse"]
RectangularKernel = Literal["dense_blocked", "sparse_left", "sparse_right", "sparse_sparse"]
AutoChoice = Literal["exact", "sketch", "exact_fallback"]
ExactMethod = Literal["adaptive_dense", "hash_sparse"]

__version__: str

class CsrMatrix:
    """Immutable canonical CSR matrix with signed 64-bit values."""
    def __init__(self, data: Int64Array, indices: Int64Array, indptr: Int64Array, shape: Tuple[int, int]) -> None: ...
    @property
    def rows(self) -> int: ...
    @property
    def cols(self) -> int: ...
    @property
    def shape(self) -> Tuple[int, int]: ...
    @property
    def nnz(self) -> int: ...
    def to_arrays(self) -> Tuple[Int64Array, Int64Array, Int64Array]: ...
    def to_dense(self) -> Int64Array: ...

class MomentConfig:
    def __init__(self, *, degree: int = 3, oversampling: float = 3.0, seed: int = 0x4D4F4D454E540001, identity_fallback: bool = True, guaranteed_correction: bool = True) -> None: ...
    @property
    def degree(self) -> int: ...
    @property
    def oversampling(self) -> float: ...
    @property
    def seed(self) -> int: ...
    @property
    def identity_fallback(self) -> bool: ...
    @property
    def guaranteed_correction(self) -> bool: ...

class FingerprintConfig:
    def __init__(self, *, lanes: int = 3, seed: int = 0) -> None: ...
    @property
    def lanes(self) -> int: ...
    @property
    def seed(self) -> int: ...

class AutoSpGemmConfig:
    def __init__(self, *, sample_rows: int = 8, initial_sample_rows: int = 2, min_structural_amplification: float = 64.0, k_safety_factor: float = 1.5, min_estimated_rho: float = 256.0, max_estimated_avg_column_nnz: float = 64.0, max_estimated_output_density: float = 0.10, max_moment_row_ratio: float = 0.80, rectangular_policy: RectangularPolicy = "auto", moment: Optional[MomentConfig] = None, fingerprint: Optional[FingerprintConfig] = None, exact_dense_cell_limit: int = 16_000_000, exact_fallback: bool = True) -> None: ...
    @property
    def sample_rows(self) -> int: ...
    @property
    def initial_sample_rows(self) -> int: ...
    @property
    def min_structural_amplification(self) -> float: ...
    @property
    def k_safety_factor(self) -> float: ...
    @property
    def min_estimated_rho(self) -> float: ...
    @property
    def max_estimated_avg_column_nnz(self) -> float: ...
    @property
    def max_estimated_output_density(self) -> float: ...
    @property
    def max_moment_row_ratio(self) -> float: ...
    @property
    def rectangular_policy(self) -> RectangularPolicy: ...
    @property
    def moment(self) -> MomentConfig: ...
    @property
    def fingerprint(self) -> FingerprintConfig: ...
    @property
    def exact_dense_cell_limit(self) -> int: ...
    @property
    def exact_fallback(self) -> bool: ...

class WorkloadEstimate:
    candidate_products: int
    structural_amplification: float
    sampled_rows: int
    sampled_output_nnz: int
    sampled_unique_columns: int
    estimated_output_nnz: int
    estimated_active_columns: int
    estimated_avg_nnz_per_active_column: float
    estimated_output_density: float
    estimated_rho: float
    target_q: int
    estimated_moment_rows: int
    choose_sketch: bool
    reason: str
    structural_prefilter_only: bool

class AutoTimingStats:
    total: float
    analysis_total: float
    candidate_count: float
    row_sampling: float
    nested: float
    fingerprint_setup: float
    fingerprint_checks: float
    exact: float

class FingerprintStats:
    checks: int
    passes: int
    failures: int
    lanes: int
    seed: int

class RoundParams:
    i: int
    t: int
    q: int
    p: int
    q_accumulated_after: int

class CorrectionPassStats:
    residual_columns: int
    residual_nnz: int
    elapsed: float

class RectangularStats:
    kernel: Optional[RectangularKernel]
    a_nnz: int
    b_nnz: int
    a_density: float
    b_density: float
    dense_ops: int
    dense_estimated_cost: int
    sparse_left_estimated_cost: int
    sparse_right_estimated_cost: int
    sparse_sparse_estimated_cost: int
    sparse_candidate_products: int
    scalar_multiplications: int

class NestedRoundStats:
    params: RoundParams
    h_rows: int
    g_rows: int
    h_kind: str
    g_kind: str
    w_nnz: int
    outer_recovered: int
    inner_updates: int
    d_nnz_after: int
    left_time: float
    right_time: float
    rectangular_time: float
    rectangular_kernel: RectangularKernel
    rectangular_scalar_multiplications: int
    rectangular_candidate_products: int
    ha_density: float
    bgt_density: float
    h_matrix_cache_hit: bool
    g_matrix_cache_hit: bool
    ha_cache_hit: bool
    bgt_cache_hit: bool
    product_cache_hit: bool
    residual_cache_hit: bool
    masked_residual: bool
    active_mask_columns: int
    scheduler_target_q: int
    residual_measure_time: float
    decode_time: float

class NestedSpGemmStats:
    rounds: list[NestedRoundStats]
    correction_pass: Optional[CorrectionPassStats]
    terminated_early: bool
    termination_reason: Optional[str]
    scheduler_skipped_rounds: int
    fingerprint: Optional[FingerprintStats]
    fingerprint_setup_time: float
    fingerprint_check_time: float
    fingerprint_verified: bool
    deterministic_verified: bool

class AutoSpGemmStats:
    choice: AutoChoice
    estimate: WorkloadEstimate
    k_bound_used: int
    nested: Optional[NestedSpGemmStats]
    exact_stats: Optional[RectangularStats]
    exact_method: Optional[ExactMethod]
    fallback_reason: Optional[str]
    timing: AutoTimingStats

def auto_spgemm(left: CsrMatrix, right: CsrMatrix, config: Optional[AutoSpGemmConfig] = None) -> Tuple[CsrMatrix, AutoSpGemmStats]: ...
def analyze_workload(left: CsrMatrix, right: CsrMatrix, config: Optional[AutoSpGemmConfig] = None) -> WorkloadEstimate: ...
