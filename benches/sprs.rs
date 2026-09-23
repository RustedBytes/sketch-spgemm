use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use sketch_spgemm::interop::sprs::auto_spgemm;
use sketch_spgemm::{sparse_output_problem, AutoSpGemmConfig};
use sprs::CsMat;
use std::hint::black_box;
use std::time::Duration;

fn sprs_comparison(criterion: &mut Criterion) {
    let problem = sparse_output_problem(128, 256, 256, 64, 7, 0.75, 256);
    let left = CsMat::new(
        (problem.a.rows, problem.a.cols),
        problem.a.row_ptr,
        problem.a.col_idx,
        problem.a.values,
    );
    let right = CsMat::new(
        (problem.b.rows, problem.b.cols),
        problem.b.row_ptr,
        problem.b.col_idx,
        problem.b.values,
    );
    let config = AutoSpGemmConfig::default();

    let standard_product = &left * &right;
    let (sketch_product, _) =
        auto_spgemm(left.view(), right.view(), config.clone()).expect("compatible CSR inputs");
    assert_eq!(sketch_product.to_dense(), standard_product.to_dense());

    let mut group = criterion.benchmark_group("sprs-comparison");
    group.throughput(Throughput::Elements(
        problem.expected_candidate_products.min(u64::MAX as u128) as u64,
    ));
    group.bench_function("sprs-native", |b| {
        b.iter(|| black_box(&left) * black_box(&right))
    });
    group.bench_function("sketch-spgemm", |b| {
        b.iter(|| {
            auto_spgemm(
                black_box(left.view()),
                black_box(right.view()),
                config.clone(),
            )
            .expect("compatible CSR inputs")
        })
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
    targets = sprs_comparison
}
criterion_main!(benches);
