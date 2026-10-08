use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sketch_spgemm::{CsrBuilder, CsrMatrix, FingerprintConfig, ResidualFingerprint};
use std::hint::black_box;

type Triplet = (usize, usize, i64);

struct Fixture {
    name: String,
    matrix: CsrMatrix,
    diagonal: CsrMatrix,
    product: CsrMatrix,
    transpose: CsrMatrix,
    stream: Vec<Triplet>,
}

fn fixtures() -> Vec<Fixture> {
    let mut fixtures = Vec::new();
    for (size, rows) in [("small", 32), ("medium", 256), ("large", 2048)] {
        for skewed in [false, true] {
            let mut triplets = Vec::new();
            for row in 0..rows {
                // Hub rows model an uneven graph degree distribution. Uniform
                // rows form a wraparound band. No random or external dataset.
                let degree = if skewed {
                    if row % 32 == 0 {
                        rows.min(64)
                    } else {
                        2
                    }
                } else {
                    rows.min(16)
                };
                for offset in 0..degree {
                    let column = (row + offset) % rows;
                    let value = 1 + ((row + column) % 7) as i64;
                    triplets.push((row, column, if column % 2 == 0 { value } else { -value }));
                }
            }
            triplets.sort_unstable_by_key(|&(row, column, _)| (row, column));
            let matrix = CsrMatrix::from_triplets(rows, rows, &triplets);
            let diagonal = CsrMatrix::from_triplets(
                rows,
                rows,
                &(0..rows).map(|i| (i, i, 2)).collect::<Vec<_>>(),
            );
            // Independent oracle: multiplying by 2I doubles each input entry.
            let product = CsrMatrix::from_triplets(
                rows,
                rows,
                &triplets
                    .iter()
                    .map(|&(i, j, v)| (i, j, 2 * v))
                    .collect::<Vec<_>>(),
            );
            let transpose = CsrMatrix::from_triplets(
                rows,
                rows,
                &triplets
                    .iter()
                    .map(|&(i, j, v)| (j, i, v))
                    .collect::<Vec<_>>(),
            );
            let stream = triplets
                .iter()
                .flat_map(|&(i, j, v)| [(i, j, v), (i, j, -v), (i, j, v)])
                .collect();
            fixtures.push(Fixture {
                name: format!("{size}-{}", if skewed { "hub" } else { "band" }),
                matrix,
                diagonal,
                product,
                transpose,
                stream,
            });
        }
    }
    fixtures
}

fn build(fixture: &Fixture) -> CsrMatrix {
    let mut builder = CsrBuilder::with_capacity(
        fixture.matrix.rows,
        fixture.matrix.cols,
        fixture.stream.len(),
    );
    builder.try_extend(fixture.stream.iter().copied()).unwrap();
    builder.finish()
}

fn core_operations(c: &mut Criterion) {
    let fixtures = fixtures();
    let config = FingerprintConfig {
        lanes: 3,
        seed: 20261008,
    };
    // Assertions and reference construction never enter the timed closures.
    for fixture in &fixtures {
        assert_eq!(build(fixture), fixture.matrix);
        assert_eq!(fixture.matrix.transpose(), fixture.transpose);
        let fingerprint = ResidualFingerprint::new(&fixture.matrix, &fixture.diagonal, config);
        assert!(fingerprint.verifies(&fixture.product));
        assert!(!fingerprint.verifies(&fixture.matrix));
    }

    let mut group = c.benchmark_group("csr-builder");
    for fixture in &fixtures {
        group.throughput(Throughput::Elements(fixture.stream.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("duplicates-cancellation", &fixture.name),
            fixture,
            |b, fixture| b.iter(|| build(black_box(fixture))),
        );
    }
    group.finish();

    let mut group = c.benchmark_group("csr-transpose");
    for fixture in &fixtures {
        group.throughput(Throughput::Elements(fixture.matrix.nnz() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(&fixture.name),
            fixture,
            |b, fixture| b.iter(|| black_box(&fixture.matrix).transpose()),
        );
    }
    group.finish();

    let mut group = c.benchmark_group("fingerprint-setup");
    for fixture in &fixtures {
        group.throughput(Throughput::Elements(
            (fixture.matrix.nnz() + fixture.diagonal.nnz()) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::from_parameter(&fixture.name),
            fixture,
            |b, fixture| {
                b.iter(|| {
                    ResidualFingerprint::new(
                        black_box(&fixture.matrix),
                        black_box(&fixture.diagonal),
                        black_box(config),
                    )
                })
            },
        );
    }
    group.finish();

    let mut group = c.benchmark_group("fingerprint-verify");
    for fixture in &fixtures {
        let fingerprint = ResidualFingerprint::new(&fixture.matrix, &fixture.diagonal, config);
        group.throughput(Throughput::Elements(fixture.product.nnz() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(&fixture.name),
            fixture,
            |b, fixture| {
                b.iter(|| black_box(black_box(&fingerprint).verifies(black_box(&fixture.product))))
            },
        );
    }
    group.finish();
}

criterion_group!(benches, core_operations);
criterion_main!(benches);
