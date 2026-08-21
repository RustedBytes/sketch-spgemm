//! Weighted adjacency and two-hop path-count support for `petgraph` graphs.

use crate::auto::{AutoSpGemmConfig, AutoSpGemmStats};
use crate::error::SpGemmError;
use crate::matrix::{CsrMatrix, Scalar};
use petgraph::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeIndexable};

/// Converts a graph to an `i64` adjacency matrix in canonical CSR form.
///
/// Rows and columns use [`NodeIndexable::to_index`], and the shape uses
/// [`NodeIndexable::node_bound`]. This preserves vacant indices in graph types
/// such as `StableGraph`. Parallel edge weights are summed. For undirected
/// graphs, `IntoEdges` naturally emits the symmetric adjacency entries.
pub fn adjacency_csr<G, F>(graph: G, edge_weight: F) -> CsrMatrix
where
    G: IntoNodeIdentifiers + IntoEdges + NodeIndexable,
    F: Fn(&<G::EdgeRef as EdgeRef>::Weight) -> Scalar,
{
    let node_bound = graph.node_bound();
    let mut triplets = Vec::new();
    for source in graph.node_identifiers() {
        let row = graph.to_index(source);
        for edge in graph.edges(source) {
            let value = edge_weight(edge.weight());
            if value != 0 {
                triplets.push((row, graph.to_index(edge.target()), value));
            }
        }
    }
    CsrMatrix::from_triplets(node_bound, node_bound, &triplets)
}

/// Computes weighted two-hop path counts for a `petgraph` graph.
///
/// An edge's closure-provided weight participates multiplicatively along each
/// two-edge path, and contributions from parallel paths are added. The output
/// is this crate's canonical CSR matrix so it can be reused by all kernels.
pub fn two_hop_path_counts<G, F>(
    graph: G,
    edge_weight: F,
    config: AutoSpGemmConfig,
) -> Result<(CsrMatrix, AutoSpGemmStats), SpGemmError>
where
    G: IntoNodeIdentifiers + IntoEdges + NodeIndexable,
    F: Fn(&<G::EdgeRef as EdgeRef>::Weight) -> Scalar,
{
    let adjacency = adjacency_csr(graph, edge_weight);
    crate::try_auto_spgemm(&adjacency, &adjacency, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::Graph;
    use petgraph::stable_graph::StableGraph;
    use petgraph::{Directed, Undirected};

    #[test]
    fn directed_parallel_edges_sum_before_weighted_path_counting() {
        let mut graph = Graph::<(), i64, Directed>::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        graph.add_edge(a, b, 2);
        graph.add_edge(a, b, 3);
        graph.add_edge(b, c, 7);

        let adjacency = adjacency_csr(&graph, |weight| *weight);
        assert_eq!(adjacency.row(0).collect::<Vec<_>>(), vec![(1, 5)]);

        let (paths, _) =
            two_hop_path_counts(&graph, |weight| *weight, AutoSpGemmConfig::default()).unwrap();
        assert_eq!(paths.row(0).collect::<Vec<_>>(), vec![(2, 35)]);
    }

    #[test]
    fn undirected_adjacency_is_symmetric() {
        let mut graph = Graph::<(), i64, Undirected>::new_undirected();
        let a = graph.add_node(());
        let b = graph.add_node(());
        graph.add_edge(a, b, 4);

        let adjacency = adjacency_csr(&graph, |weight| *weight);
        assert_eq!(adjacency.row(0).collect::<Vec<_>>(), vec![(1, 4)]);
        assert_eq!(adjacency.row(1).collect::<Vec<_>>(), vec![(0, 4)]);
    }

    #[test]
    fn stable_graph_holes_remain_in_the_matrix_index_space() {
        let mut graph = StableGraph::<(), i64, Directed>::new();
        let a = graph.add_node(());
        let removed = graph.add_node(());
        let c = graph.add_node(());
        graph.remove_node(removed);
        graph.add_edge(a, c, 1);

        let adjacency = adjacency_csr(&graph, |weight| *weight);
        assert_eq!((adjacency.rows, adjacency.cols), (3, 3));
        assert_eq!(adjacency.row(0).collect::<Vec<_>>(), vec![(2, 1)]);
        assert!(adjacency.row(1).next().is_none());
    }
}
