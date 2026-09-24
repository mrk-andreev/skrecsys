//! The item-item similarity kernels, and the pruning pass that trims one after the fact.
//!
//! All four walk the same shape of input -- a users x items matrix, one row of the
//! product at a time -- so they share the fixture, and their numbers are directly
//! comparable with each other as well as with the previous run.

mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use skrecsys_kernels::{knn, prune, rp3beta};
use support::{Matrix, install_pool};

/// `(n_users, n_items, nnz per user)`. The catalogs stay small: the kernels are
/// quadratic in the item count, and a fixture that took a minute per sample would make
/// the benches too slow to run on every change.
const SHAPES: [(usize, usize, usize); 2] = [(5_000, 1_000, 20), (5_000, 2_000, 40)];

/// The neighbourhood size the estimators default to.
const K: usize = 100;

fn label(n_rows: usize, n_cols: usize, per_row: usize) -> String {
    format!("{n_rows}x{n_cols}/{per_row}")
}

fn item_knn(c: &mut Criterion) {
    install_pool();
    let mut group = c.benchmark_group("knn::all_pairs_top_k");
    group.sample_size(20);
    for (n_rows, n_cols, per_row) in SHAPES {
        let matrix = Matrix::random(n_rows, n_cols, per_row, 0xf00d);
        group.throughput(Throughput::Elements(matrix.nnz() as u64));
        group.bench_function(
            BenchmarkId::from_parameter(label(n_rows, n_cols, per_row)),
            |b| b.iter(|| knn::all_pairs_top_k(black_box(&matrix.csr()), K)),
        );
    }
    group.finish();
}

fn item_cosine(c: &mut Criterion) {
    install_pool();
    let mut group = c.benchmark_group("knn::cosine_top_k");
    group.sample_size(20);
    for (n_rows, n_cols, per_row) in SHAPES {
        let matrix = Matrix::random(n_rows, n_cols, per_row, 0xf00d);
        let norms = matrix.column_norms();
        group.throughput(Throughput::Elements(matrix.nnz() as u64));
        group.bench_function(
            BenchmarkId::from_parameter(label(n_rows, n_cols, per_row)),
            |b| b.iter(|| knn::cosine_top_k(black_box(&matrix.csr()), &norms, 10.0, K)),
        );
    }
    group.finish();
}

fn rp3beta_similarity(c: &mut Criterion) {
    install_pool();
    let mut group = c.benchmark_group("rp3beta::similarity");
    group.sample_size(20);
    for (n_rows, n_cols, per_row) in SHAPES {
        let matrix = Matrix::random(n_rows, n_cols, per_row, 0xf00d).row_normalized();
        let norms = matrix.column_norms();
        // Stand-ins for the walk and popularity damping, which only scale the result.
        let row_scale: Vec<f64> = norms.iter().map(|n| 1.0 / (1.0 + n)).collect();
        let col_scale: Vec<f64> = norms.iter().map(|n| 1.0 / (1.0 + n).sqrt()).collect();
        group.throughput(Throughput::Elements(matrix.nnz() as u64));
        group.bench_function(
            BenchmarkId::from_parameter(label(n_rows, n_cols, per_row)),
            |b| b.iter(|| rp3beta::similarity(black_box(&matrix.csr()), &row_scale, &col_scale, K)),
        );
    }
    group.finish();
}

/// Pruning an existing similarity matrix, both when every row already fits within `k`
/// and when every row has to be cut down -- the two branches of the kernel.
fn csr_top_k(c: &mut Criterion) {
    install_pool();
    let mut group = c.benchmark_group("prune::top_k_per_row");
    for (per_row, k) in [(20usize, 100usize), (400, 100)] {
        let matrix = Matrix::random(20_000, 2_000, per_row, 0xf00d);
        group.throughput(Throughput::Elements(matrix.nnz() as u64));
        group.bench_function(
            BenchmarkId::from_parameter(format!("{per_row}/k{k}")),
            |b| b.iter(|| prune::top_k_per_row(black_box(&matrix.csr()), k)),
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = support::criterion();
    targets = item_knn, item_cosine, rp3beta_similarity, csr_top_k
}
criterion_main!(benches);
