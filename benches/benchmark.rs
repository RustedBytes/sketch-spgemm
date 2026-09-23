use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sketch_spgemm::{
    adaptive_matmul, auto_spgemm, left_sketch, overlap_problem, right_sketch,
    sparse_output_problem, spgemm_hash, AutoSpGemmConfig, RectangularPolicy, SketchMap,
    SyntheticProblem,
};
use std::hint::black_box;
use std::time::Duration;

fn workloads() -> [(&'static str, SyntheticProblem); 2] {
    [
        ("overlap", overlap_problem(64, 128, 128, 32, 0.5)),
        (
            "sparse-output",
            sparse_output_problem(64, 128, 128, 32, 5, 0.75, 128),
        ),
    ]
}

fn product_benchmarks(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("sparse-product");

    for (name, problem) in workloads() {
        let (expected, stats) = spgemm_hash(&problem.a, &problem.b);
        assert_eq!(
            stats.candidate_products,
            problem.expected_candidate_products
        );

        let dense_a = problem.a.to_dense();
        let dense_b = problem.b.to_dense();
        let (dense_product, _) = adaptive_matmul(&dense_a, &dense_b, RectangularPolicy::Auto);
        assert_eq!(dense_product, expected.to_dense());

        let auto_config = AutoSpGemmConfig::default();
        let (automatic_product, _) = auto_spgemm(&problem.a, &problem.b, auto_config.clone());
        assert_eq!(automatic_product, expected);

        group.throughput(Throughput::Elements(
            problem.expected_candidate_products.min(u64::MAX as u128) as u64,
        ));

        group.bench_with_input(BenchmarkId::new("hash-csr", name), &problem, |b, input| {
            b.iter(|| spgemm_hash(black_box(&input.a), black_box(&input.b)))
        });

        group.bench_with_input(
            BenchmarkId::new("adaptive-dense", name),
            &(&dense_a, &dense_b),
            |b, &(left, right)| {
                b.iter(|| {
                    adaptive_matmul(black_box(left), black_box(right), RectangularPolicy::Auto)
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("automatic", name), &problem, |b, input| {
            b.iter(|| {
                auto_spgemm(
                    black_box(&input.a),
                    black_box(&input.b),
                    auto_config.clone(),
                )
            })
        });
    }

    group.finish();
}

fn sketch_kernel_benchmarks(criterion: &mut Criterion) {
    let problem = sparse_output_problem(64, 128, 128, 32, 5, 0.75, 128);
    let left_map = SketchMap::new(problem.a.rows, 32, 3, 0xA11CE);
    let right_map = SketchMap::new(problem.b.cols, 32, 3, 0xB0B);
    let left = left_sketch(&problem.a, &left_map);
    let right = right_sketch(&problem.b, &right_map);

    let mut group = criterion.benchmark_group("sketch-kernels");
    group.throughput(Throughput::Elements(
        problem.expected_candidate_products.min(u64::MAX as u128) as u64,
    ));

    group.bench_function("left-sketch", |b| {
        b.iter(|| left_sketch(black_box(&problem.a), black_box(&left_map)))
    });
    group.bench_function("right-sketch", |b| {
        b.iter(|| right_sketch(black_box(&problem.b), black_box(&right_map)))
    });
    group.bench_function("compressed-matmul", |b| {
        b.iter(|| adaptive_matmul(black_box(&left), black_box(&right), RectangularPolicy::Auto))
    });

    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
        .sample_size(20)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = product_benchmarks, sketch_kernel_benchmarks
}
criterion_main!(benches);
