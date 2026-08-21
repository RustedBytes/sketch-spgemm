mod support;

use sketch_spgemm::interop::sprs::auto_spgemm;
use sketch_spgemm::synthetic::sparse_output_problem;
use sketch_spgemm::AutoSpGemmConfig;
use sprs::CsMat;

fn main() {
    let duration = support::duration_from_args();
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
    let (sketch_product, initial_stats) =
        auto_spgemm(left.view(), right.view(), config.clone()).expect("compatible CSR inputs");
    assert_eq!(sketch_product.to_dense(), standard_product.to_dense());

    println!("sprs CSR multiplication benchmark");
    println!(
        "shape: {}x{} * {}x{}, input nnz: {} + {}, logical output nnz: {}",
        left.rows(),
        left.cols(),
        right.rows(),
        right.cols(),
        left.nnz(),
        right.nnz(),
        sketch_product.nnz()
    );
    println!(
        "sprs stored output entries: {} (includes explicit cancellation zeros)",
        standard_product.nnz()
    );
    println!("sketch-spgemm selected: {:?}", initial_stats.choice);
    println!(
        "measurement window per implementation: {:.2}s",
        duration.as_secs_f64()
    );

    let standard = support::measure(duration, || &left * &right);
    let sketch = support::measure(duration, || {
        auto_spgemm(left.view(), right.view(), config.clone())
            .expect("compatible CSR inputs")
            .0
    });

    support::print_comparison("sprs standard", &standard, &sketch);
}
