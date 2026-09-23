"""Adaptive sketch-based sparse matrix multiplication."""

import importlib
from typing import Any

import numpy as np

from ._sketch_spgemm import (
    AutoSpGemmConfig,
    AutoSpGemmStats,
    AutoTimingStats,
    CorrectionPassStats,
    CsrBuilder,
    CsrMatrix,
    FingerprintConfig,
    FingerprintStats,
    MomentConfig,
    NestedRoundStats,
    NestedSpGemmStats,
    RectangularStats,
    RoundParams,
    SpGemmStats,
    WorkloadEstimate,
    __version__,
    analyze_workload,
    auto_spgemm,
    checked_spgemm,
    spgemm,
)


def from_scipy(matrix: Any) -> CsrMatrix:
    """Copy a SciPy CSR matrix/array into canonical ``CsrMatrix`` storage."""
    if getattr(matrix, "format", None) != "csr":
        raise ValueError("from_scipy requires CSR storage")
    matrix = matrix.copy()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return CsrMatrix(
        np.ascontiguousarray(matrix.data),
        np.ascontiguousarray(matrix.indices, dtype=np.int64),
        np.ascontiguousarray(matrix.indptr, dtype=np.int64),
        tuple(matrix.shape),
    )


def to_scipy(matrix: CsrMatrix) -> Any:
    """Copy a ``CsrMatrix`` into a SciPy ``csr_matrix``."""
    try:
        sparse = importlib.import_module("scipy.sparse")
    except ImportError as error:
        raise ImportError("to_scipy requires scipy") from error
    data, indices, indptr = matrix.to_arrays()
    return sparse.csr_matrix((data, indices, indptr), shape=matrix.shape)


__all__ = [
    "AutoSpGemmConfig",
    "AutoSpGemmStats",
    "AutoTimingStats",
    "CorrectionPassStats",
    "CsrBuilder",
    "CsrMatrix",
    "FingerprintConfig",
    "FingerprintStats",
    "MomentConfig",
    "NestedRoundStats",
    "NestedSpGemmStats",
    "RectangularStats",
    "RoundParams",
    "SpGemmStats",
    "WorkloadEstimate",
    "__version__",
    "analyze_workload",
    "auto_spgemm",
    "checked_spgemm",
    "from_scipy",
    "spgemm",
    "to_scipy",
]
