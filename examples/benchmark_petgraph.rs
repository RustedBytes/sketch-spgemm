mod support;

use petgraph::graph::DiGraph;
use petgraph::visit::{EdgeRef, NodeIndexable};
use sketch_spgemm::interop::petgraph::two_hop_path_counts;
use sketch_spgemm::synthetic::sparse_output_problem;
use sketch_spgemm::{AutoSpGemmConfig, CsrMatrix};
use std::collections::BTreeMap;

fn main() {
    let duration = support::duration_from_args();
    let problem = sparse_output_problem(128, 256, 256, 64, 7, 0.75, 256);
    let mut graph = DiGraph::<(), i64>::new();
    let nodes: Vec<_> = (0..problem.a.rows + problem.a.cols + problem.b.cols)
        .map(|_| graph.add_node(()))
        .collect();

    for row in 0..problem.a.rows {
        for (column, value) in problem.a.row(row) {
            graph.add_edge(nodes[row], nodes[problem.a.rows + column], value);
        }
    }
    for row in 0..problem.b.rows {
        for (column, value) in problem.b.row(row) {
            graph.add_edge(
                nodes[problem.a.rows + row],
                nodes[problem.a.rows + problem.a.cols + column],
                value,
            );
        }
    }

    let config = sketch_config();
    let standard_product = direct_two_hop_counts(&graph);
    let (sketch_product, initial_stats) =
        two_hop_path_counts(&graph, |weight| *weight, config.clone()).unwrap();
    assert_eq!(sketch_product, standard_product);

    println!("petgraph weighted two-hop benchmark");
    println!(
        "nodes: {}, edges: {}, output nnz: {}",
        graph.node_count(),
        graph.edge_count(),
        standard_product.nnz()
    );
    println!("sketch-spgemm selected: {:?}", initial_stats.choice);
    println!(
        "measurement window per implementation: {:.2}s",
        duration.as_secs_f64()
    );

    let standard = support::measure(duration, || direct_two_hop_counts(&graph));
    let sketch = support::measure(duration, || {
        two_hop_path_counts(&graph, |weight| *weight, config.clone())
            .unwrap()
            .0
    });

    support::print_comparison("petgraph traversal", &standard, &sketch);
}

fn direct_two_hop_counts(graph: &DiGraph<(), i64>) -> CsrMatrix {
    let node_bound = graph.node_bound();
    let mut rows: Vec<BTreeMap<usize, i64>> = (0..node_bound).map(|_| BTreeMap::new()).collect();
    for source in graph.node_indices() {
        for first in graph.edges(source) {
            for second in graph.edges(first.target()) {
                let value = first.weight().wrapping_mul(*second.weight());
                let entry = rows[source.index()]
                    .entry(second.target().index())
                    .or_default();
                *entry = entry.wrapping_add(value);
            }
        }
    }

    let mut row_ptr = Vec::with_capacity(node_bound + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for row in rows {
        for (column, value) in row {
            if value != 0 {
                col_idx.push(column);
                values.push(value);
            }
        }
        row_ptr.push(col_idx.len());
    }
    CsrMatrix {
        rows: node_bound,
        cols: node_bound,
        row_ptr,
        col_idx,
        values,
    }
}

fn sketch_config() -> AutoSpGemmConfig {
    AutoSpGemmConfig {
        sample_rows: 16,
        initial_sample_rows: 4,
        min_structural_amplification: 0.0,
        min_estimated_rho: 0.0,
        max_estimated_avg_column_nnz: f64::INFINITY,
        max_estimated_output_density: 1.0,
        max_moment_row_ratio: 2.0,
        ..AutoSpGemmConfig::default()
    }
}
