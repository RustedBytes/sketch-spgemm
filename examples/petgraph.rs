use petgraph::graph::DiGraph;
use sketch_spgemm::interop::petgraph::{adjacency_csr, two_hop_path_counts};
use sketch_spgemm::{AutoSpGemmConfig, SpGemmError};

fn main() -> Result<(), SpGemmError> {
    let mut graph = DiGraph::<&str, i64>::new();
    let alice = graph.add_node("Alice");
    let bob = graph.add_node("Bob");
    let carol = graph.add_node("Carol");

    // Parallel Alice -> Bob edges sum to 5. The weighted two-hop count from
    // Alice to Carol is therefore (2 + 3) * 7 = 35.
    graph.add_edge(alice, bob, 2);
    graph.add_edge(alice, bob, 3);
    graph.add_edge(bob, carol, 7);

    let adjacency = adjacency_csr(&graph, |weight| *weight);
    let (paths, stats) =
        two_hop_path_counts(&graph, |weight| *weight, AutoSpGemmConfig::default())?;

    assert_eq!(
        adjacency.row(alice.index()).collect::<Vec<_>>(),
        vec![(bob.index(), 5)]
    );
    assert_eq!(
        paths.row(alice.index()).collect::<Vec<_>>(),
        vec![(carol.index(), 35)]
    );

    println!("selected path: {:?}", stats.choice);
    println!("weighted adjacency: {adjacency:?}");
    println!("weighted two-hop paths: {paths:?}");
    Ok(())
}
