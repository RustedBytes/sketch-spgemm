from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import sketch_spgemm as ssg


def csr(data, indices, indptr, shape):
    return ssg.CsrMatrix(
        np.asarray(data, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
        np.asarray(indptr, dtype=np.int64),
        shape,
    )


@pytest.fixture
def operands():
    left = csr([2, 3, 4], [0, 1, 1], [0, 2, 3], (2, 2))
    right = csr([5, 7, 11], [0, 0, 1], [0, 1, 3], (2, 2))
    return left, right


def test_import_and_exact_product(operands):
    left, right = operands
    product, stats = ssg.auto_spgemm(left, right)

    np.testing.assert_array_equal(product.to_dense(), [[31, 33], [28, 44]])
    assert product.shape == (2, 2)
    assert product.nnz == 4
    assert stats.choice == "exact"
    assert stats.exact_method in {"adaptive_dense", "hash_sparse"}
    assert stats.estimate.candidate_products == 5
    assert stats.timing.total >= stats.timing.analysis_total >= 0.0
    assert stats.exact_stats is not None


def test_version():
    assert ssg.__version__ == "0.11.0"


def test_checked_product_and_overflow(operands):
    left, right = operands
    product, stats = ssg.checked_spgemm(left, right)
    np.testing.assert_array_equal(product.to_dense(), [[31, 33], [28, 44]])
    assert stats.candidate_products == 5

    maximum = np.iinfo(np.int64).max
    overflowing_left = csr([maximum], [0], [0, 1], (1, 1))
    multiplier = csr([2], [0], [0, 1], (1, 1))
    with pytest.raises(OverflowError, match="multiplication"):
        ssg.checked_spgemm(overflowing_left, multiplier)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_supported_matrix_dtypes(dtype):
    data = np.asarray([2, 3, 4], dtype=dtype)
    left = ssg.CsrMatrix(
        data,
        np.asarray([0, 1, 1], dtype=np.int64),
        np.asarray([0, 2, 3], dtype=np.int64),
        (2, 2),
    )
    right = ssg.CsrMatrix.from_triplets(
        np.asarray([5, 7, 11], dtype=dtype),
        np.asarray([0, 1, 1], dtype=np.int64),
        np.asarray([0, 0, 1], dtype=np.int64),
        (2, 2),
    )

    product, stats = ssg.checked_spgemm(left, right)
    assert left.dtype == np.dtype(dtype).name
    assert product.dtype == np.dtype(dtype).name
    assert product.to_dense().dtype == dtype
    assert product.to_arrays()[0].dtype == dtype
    np.testing.assert_allclose(product.to_dense(), [[31, 33], [28, 44]])
    assert stats.candidate_products == 5


@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_streaming_builder_preserves_dtype(dtype):
    numpy_dtype = np.dtype(dtype)
    builder = ssg.CsrBuilder(2, 2, dtype=dtype)
    assert builder.dtype == dtype
    builder.extend(
        np.asarray([2, 3], dtype=numpy_dtype),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
    )
    builder.push(1, 1, 1.5 if dtype.startswith("float") else 1)
    matrix = builder.finish()
    assert builder.dtype == dtype
    assert matrix.dtype == dtype
    assert matrix.to_dense().dtype == numpy_dtype
    np.testing.assert_allclose(
        matrix.to_dense(), [[2, 0], [0, 4.5 if dtype.startswith("float") else 4]]
    )


def test_dtype_validation_and_algorithm_boundaries():
    indices = np.asarray([0], dtype=np.int64)
    indptr = np.asarray([0, 1], dtype=np.int64)
    with pytest.raises(TypeError, match="data dtype"):
        ssg.CsrMatrix(
            np.asarray([1], dtype=np.uint64),  # pyright: ignore[reportArgumentType]
            indices,
            indptr,
            (1, 1),
        )

    int_matrix = ssg.CsrMatrix(np.asarray([1], dtype=np.int32), indices, indptr, (1, 1))
    float_matrix = ssg.CsrMatrix(np.asarray([1], dtype=np.float32), indices, indptr, (1, 1))
    with pytest.raises(TypeError, match="same dtype"):
        ssg.checked_spgemm(int_matrix, float_matrix)
    with pytest.raises(TypeError, match="requires int64"):
        ssg.auto_spgemm(float_matrix, float_matrix)


def test_int32_checked_overflow():
    maximum = np.iinfo(np.int32).max
    left = ssg.CsrMatrix.from_triplets(
        np.asarray([maximum], dtype=np.int32),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        (1, 1),
    )
    right = ssg.CsrMatrix.from_triplets(
        np.asarray([2], dtype=np.int32),
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        (1, 1),
    )
    with pytest.raises(OverflowError, match="multiplication"):
        ssg.checked_spgemm(left, right)


def test_csr_from_unsorted_triplets():
    matrix = ssg.CsrMatrix.from_triplets(
        np.asarray([7, 4, -1, 5], dtype=np.int64),
        np.asarray([2, 0, 0, 2], dtype=np.int64),
        np.asarray([0, 1, 1, 2], dtype=np.int64),
        (3, 3),
    )
    np.testing.assert_array_equal(matrix.to_dense(), [[0, 3, 0], [0, 0, 0], [7, 0, 5]])


def test_streaming_csr_builder_accepts_chunks_and_closes_after_finish():
    builder = ssg.CsrBuilder(3, 3, capacity=3)
    builder.extend(
        np.asarray([4, -1], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
        np.asarray([1, 1], dtype=np.int64),
    )
    builder.push(2, 0, 7)
    assert builder.shape == (3, 3)

    matrix = builder.finish()
    np.testing.assert_array_equal(matrix.to_dense(), [[0, 3, 0], [0, 0, 0], [7, 0, 0]])
    with pytest.raises(RuntimeError, match="already been finished"):
        builder.push(2, 1, 1)
    empty = np.asarray([], dtype=np.int64)
    with pytest.raises(RuntimeError, match="already been finished"):
        builder.extend(empty, empty, empty)


def test_streaming_csr_builder_reports_order_and_overflow_errors():
    builder = ssg.CsrBuilder(2, 2)
    builder.push(1, 0, 1)
    with pytest.raises(ValueError, match="row-major order"):
        builder.push(0, 1, 1)

    builder = ssg.CsrBuilder(1, 1)
    builder.push(0, 0, np.iinfo(np.int64).max)
    with pytest.raises(OverflowError, match="combining triplets"):
        builder.push(0, 0, 1)


def test_numpy_outputs_are_independent_int64_copies(operands):
    left, _ = operands
    data, indices, indptr = left.to_arrays()
    assert data.dtype == indices.dtype == indptr.dtype == np.int64
    data[0] = 999
    indices[0] = 1
    indptr[-1] = 0
    np.testing.assert_array_equal(left.to_dense(), [[2, 3], [0, 4]])


def test_analyze_and_immutable_configuration(operands):
    left, right = operands
    config = ssg.AutoSpGemmConfig(
        rectangular_policy="sparse_sparse",
        moment=ssg.MomentConfig(degree=4, seed=11),
        fingerprint=ssg.FingerprintConfig(lanes=2, seed=13),
    )
    estimate = ssg.analyze_workload(left, right, config)
    assert estimate.candidate_products == 5
    assert config.rectangular_policy == "sparse_sparse"
    assert config.moment.degree == 4
    assert config.fingerprint.seed == 13
    with pytest.raises(AttributeError):
        config.sample_rows = 99  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize(
    ("data", "indices", "indptr", "shape", "message"),
    [
        ([1], [0, 0], [0, 1], (1, 1), "equal lengths"),
        ([1], [0], [1, 1], (1, 1), "start at zero"),
        ([1], [-1], [0, 1], (1, 1), "non-negative"),
        ([1], [1], [0, 1], (1, 1), "outside matrix width"),
        ([1, 2], [1, 0], [0, 2], (1, 2), "strictly increasing"),
        ([1, 2], [0, 0], [0, 2], (1, 2), "strictly increasing"),
        ([0], [0], [0, 1], (1, 1), "explicit zero"),
    ],
)
def test_rejects_noncanonical_csr(data, indices, indptr, shape, message):
    with pytest.raises(ValueError, match=message):
        csr(data, indices, indptr, shape)


def test_rejects_wrong_dtype_and_noncontiguous_arrays():
    with pytest.raises(TypeError):
        ssg.CsrMatrix(
            np.asarray([1], dtype=np.uint16),  # pyright: ignore[reportArgumentType]
            np.asarray([0], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
            (1, 1),
        )

    noncontiguous = np.asarray([1, 99, 2, 99], dtype=np.int64)[::2]
    with pytest.raises(ValueError, match="contiguous"):
        ssg.CsrMatrix(
            noncontiguous,
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([0, 2], dtype=np.int64),
            (1, 2),
        )


def test_dimension_mismatch_is_value_error():
    left = csr([1], [0], [0, 1], (1, 1))
    right = csr([1], [0], [0, 1, 1], (2, 1))
    with pytest.raises(ValueError, match="incompatible matrix dimensions"):
        ssg.auto_spgemm(left, right)


def test_empty_rectangular_product():
    left = csr([], [], [0, 0, 0], (2, 0))
    right = csr([], [], [0], (0, 3))
    product, _ = ssg.auto_spgemm(left, right)
    assert product.shape == (2, 3)
    np.testing.assert_array_equal(product.to_dense(), np.zeros((2, 3), dtype=np.int64))


def test_parallel_calls_share_immutable_inputs(operands):
    left, right = operands

    def multiply(_):
        return ssg.auto_spgemm(left, right)[0].to_dense()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(multiply, range(16)))
    for result in results:
        np.testing.assert_array_equal(result, [[31, 33], [28, 44]])


def test_forced_sketch_exposes_nested_stats():
    rows = 64
    entries_per_row = 8
    domain = rows * entries_per_row
    left = csr(
        [1] * domain,
        list(range(domain)),
        [row * entries_per_row for row in range(rows + 1)],
        (rows, domain),
    )
    right = csr(
        [1] * domain,
        list(range(domain)),
        list(range(domain + 1)),
        (domain, domain),
    )
    config = ssg.AutoSpGemmConfig(
        min_structural_amplification=0.0,
        min_estimated_rho=0.0,
        max_estimated_avg_column_nnz=1_000.0,
        max_estimated_output_density=1.0,
        max_moment_row_ratio=1.0,
        fingerprint=ssg.FingerprintConfig(seed=123),
    )

    product, stats = ssg.auto_spgemm(left, right, config)
    assert product.nnz == domain
    assert stats.choice == "sketch"
    assert stats.nested is not None
    assert stats.nested.fingerprint_verified
    assert len(stats.nested.rounds) >= 1
    assert stats.nested.rounds[0].params.q >= 1


def test_configuration_validation():
    with pytest.raises(ValueError, match="sample_rows"):
        ssg.AutoSpGemmConfig(sample_rows=0)
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        ssg.AutoSpGemmConfig(max_estimated_output_density=2.0)
    with pytest.raises(ValueError, match="rectangular_policy"):
        ssg.AutoSpGemmConfig(
            rectangular_policy="unknown"  # pyright: ignore[reportArgumentType]
        )
    with pytest.raises(ValueError, match="degree"):
        ssg.MomentConfig(degree=0)
    with pytest.raises(ValueError, match="lanes"):
        ssg.FingerprintConfig(lanes=0)
