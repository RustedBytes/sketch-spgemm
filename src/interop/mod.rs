//! Feature-gated adapters for ecosystem matrix and graph types.

/// Zero-copy input and native-output support for `sprs` matrices.
#[cfg(feature = "sprs")]
pub mod sprs;

/// Weighted adjacency and two-hop path-count support for `petgraph` graphs.
#[cfg(feature = "petgraph")]
pub mod petgraph;
