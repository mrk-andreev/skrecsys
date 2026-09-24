//! The approximate path: building a navigable graph, and walking it instead of the
//! catalog.
//!
//! The search benchmarks sit next to `ranking`'s on purpose. `ranking::top_k_per_row`
//! and `recommend::top_k_from_similarity` both grow with the catalog; a graph walk is
//! meant not to, and the only way to see that is to run the same shapes at two catalog
//! sizes and watch which curves bend. What the walk costs in recall is not measurable
//! here -- `benchmarks/indexes.py` does that against real models -- so this file is
//! about time alone.

mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use skrecsys_kernels::hnsw::{self, DenseItems, GraphView, Params, Queries, SparseItems};
use skrecsys_kernels::ranking::Excluded;
use support::{Matrix, Pattern, Rng, criterion, install_pool};

/// What a query asks for, and how many items it has already seen.
const K: usize = 10;
const SEEN_PER_QUERY: usize = 20;
/// Defaults a user who passed `index="hnsw"` would get.
const M: usize = 16;
const EF_CONSTRUCTION: usize = 200;

/// Row-major unit-norm vectors: the shape a latent-factor model hands the index.
fn dense_vectors(n_items: usize, dim: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut vectors: Vec<f64> = (0..n_items * dim)
        .map(|_| rng.next_f64() * 2.0 - 1.0)
        .collect();
    for row in vectors.chunks_mut(dim) {
        let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        row.iter_mut().for_each(|v| *v /= norm);
    }
    vectors
}

fn params(seed: u64) -> Params {
    Params {
        m: M,
        ef_construction: EF_CONSTRUCTION,
        seed,
    }
}

fn build_dense(c: &mut Criterion) {
    install_pool();
    let mut group = c.benchmark_group("hnsw::build/dense");
    for n_items in [5_000usize, 20_000] {
        let vectors = dense_vectors(n_items, 32, 0xbeef);
        let items = DenseItems {
            vectors: &vectors,
            dim: 32,
        };
        group.throughput(Throughput::Elements(n_items as u64));
        group.bench_function(BenchmarkId::from_parameter(n_items), |b| {
            b.iter(|| hnsw::build(black_box(&items), &params(0xcafe)))
        });
    }
    group.finish();
}

fn build_sparse(c: &mut Criterion) {
    install_pool();
    let mut group = c.benchmark_group("hnsw::build/sparse");
    for n_items in [2_000usize, 8_000] {
        // One hundred neighbours a row, as a pruned item-item similarity carries.
        let matrix = Matrix::random(n_items, n_items, 100, 0xcafe);
        let items = SparseItems {
            matrix: matrix.csr(),
        };
        group.throughput(Throughput::Elements(n_items as u64));
        group.bench_function(BenchmarkId::from_parameter(n_items), |b| {
            b.iter(|| hnsw::build(black_box(&items), &params(0xcafe)))
        });
    }
    group.finish();
}

fn search_dense(c: &mut Criterion) {
    install_pool();
    let n_queries = 1_000;
    let mut group = c.benchmark_group("hnsw::top_k/dense");
    for n_items in [5_000usize, 50_000] {
        let vectors = dense_vectors(n_items, 32, 0xbeef);
        let items = DenseItems {
            vectors: &vectors,
            dim: 32,
        };
        let graph = hnsw::build(&items, &params(0xcafe));
        let view = GraphView::new(
            &graph.node_level,
            &graph.links_indptr,
            &graph.links_indices,
            graph.entry_point,
        )
        .expect("a built graph describes itself");
        let candidates: Vec<usize> = (0..n_items).collect();
        let position: Vec<i64> = (0..n_items).map(|i| i as i64).collect();
        let query_data = dense_vectors(n_queries, 32, 0xf00d);
        let queries = Queries::Dense {
            data: &query_data,
            dim: 32,
        };
        let seen = Pattern::random(n_queries, n_items, SEEN_PER_QUERY, 0xbeef);

        // Elements are queries, not query-item pairs: the whole point is that the work
        // a query does no longer follows the catalog.
        group.throughput(Throughput::Elements(n_queries as u64));
        for ef in [32usize, 128] {
            group.bench_function(
                BenchmarkId::from_parameter(format!("{n_items}/ef{ef}")),
                |b| {
                    b.iter(|| {
                        hnsw::top_k(
                            black_box(&items),
                            &view,
                            &queries,
                            &candidates,
                            &position,
                            &seen.excluded(),
                            K,
                            ef,
                        )
                        .map_err(|e| e.row)
                    })
                },
            );
        }
    }
    group.finish();
}

fn search_sparse(c: &mut Criterion) {
    install_pool();
    let n_queries = 1_000;
    let n_items = 8_000;
    let matrix = Matrix::random(n_items, n_items, 100, 0xcafe);
    let items = SparseItems {
        matrix: matrix.csr(),
    };
    let graph = hnsw::build(&items, &params(0xcafe));
    let view = GraphView::new(
        &graph.node_level,
        &graph.links_indptr,
        &graph.links_indices,
        graph.entry_point,
    )
    .expect("a built graph describes itself");
    let candidates: Vec<usize> = (0..n_items).collect();
    let position: Vec<i64> = (0..n_items).map(|i| i as i64).collect();
    let users = Matrix::random(n_queries, n_items, SEEN_PER_QUERY, 0xbeef);
    let queries = Queries::Scattered(users.csr());
    let seen = Pattern::random(n_queries, n_items, SEEN_PER_QUERY, 0xbeef);
    let excluded: Excluded<'_> = seen.excluded();

    let mut group = c.benchmark_group("hnsw::top_k/sparse");
    group.throughput(Throughput::Elements(n_queries as u64));
    for ef in [32usize, 128] {
        group.bench_function(BenchmarkId::from_parameter(format!("ef{ef}")), |b| {
            b.iter(|| {
                hnsw::top_k(
                    black_box(&items),
                    &view,
                    &queries,
                    &candidates,
                    &position,
                    &excluded,
                    K,
                    ef,
                )
                .map_err(|e| e.row)
            })
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = criterion();
    targets = build_dense, build_sparse, search_dense, search_sparse
}
criterion_main!(benches);
