//! Hierarchical navigable small-world graphs.
//!
//! An approximate alternative to scoring a whole catalog: the graph links each item to a
//! spread of its nearest neighbours across a few levels, and a search walks it from the
//! top down, so a query touches a few hundred items instead of every one of them. The
//! reference implementation is Qdrant's, and the departures from it are documented where
//! they are made.
//!
//! What it costs is exactness. The result is the best items the walk *found*, which is
//! usually but not always the best items there are; `benchmarks/indexes.py` is where
//! that difference is measured per model rather than assumed.
//!
//! The graph is built once by [`build`] and searched by [`top_k`]. Between the two it is
//! nothing but the flat arrays of [`Graph`], which is what lets a fitted estimator pickle
//! without carrying any of this along.

pub(crate) mod build;
mod graph;
mod search;
mod visited;

pub use crate::vectors::{DenseItems, Items, Probe, Queries, Scatter, SparseItems};
pub use build::build;
pub use graph::{Graph, GraphView, Links, MAX_LEVEL, Params};
pub use search::top_k;
