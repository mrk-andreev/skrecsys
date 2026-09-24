//! The predict path: selecting each query's best `k` items.
//!
//! Two kernels serve it. `ranking::top_k_per_row` ranks a dense score matrix the caller
//! already has, and `recommend::top_k_from_similarity` fuses the scoring into the
//! selection so the dense matrix is never built. They are benchmarked on the same
//! shapes on purpose: the gap between them is the reason the fused path exists, and a
//! regression that closes it is worth seeing.

mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use skrecsys_kernels::{ranking, recommend};
use support::{Matrix, Pattern, dense_scores, install_pool};

/// What a query asks for, and how many items it has already seen.
const K: usize = 10;
const SEEN_PER_QUERY: usize = 20;

fn dense_top_k(c: &mut Criterion) {
    install_pool();
    let mut group = c.benchmark_group("ranking::top_k_per_row");
    for (n_rows, n_cols) in [(1_000usize, 2_000usize), (1_000, 20_000)] {
        let scores = dense_scores(n_rows, n_cols, 0xbeef);
        let seen = Pattern::random(n_rows, n_cols, SEEN_PER_QUERY, 0xbeef);
        group.throughput(Throughput::Elements((n_rows * n_cols) as u64));
        group.bench_function(
            BenchmarkId::from_parameter(format!("{n_rows}x{n_cols}")),
            |b| {
                b.iter(|| {
                    ranking::top_k_per_row(black_box(&scores), &seen.excluded(), n_cols, K)
                        .map_err(|e| e.row)
                })
            },
        );
    }
    group.finish();
}

/// The fused path, over the whole catalog and over a restricted candidate set -- the
/// latter goes through the position mapping that a filtered catalog needs.
fn fused_top_k(c: &mut Criterion) {
    install_pool();
    let n_queries = 1_000;
    let n_items = 2_000;
    let users = Matrix::random(n_queries, n_items, SEEN_PER_QUERY, 0xbeef);
    let similarity = Matrix::random(n_items, n_items, 100, 0xcafe);
    let rows: Vec<usize> = (0..n_queries).collect();

    let mut group = c.benchmark_group("recommend::top_k_from_similarity");
    for share in [1usize, 2] {
        // Every item, then every second one.
        let candidates: Vec<usize> = (0..n_items).step_by(share).collect();
        let mut position = vec![-1i64; n_items];
        for (p, &j) in candidates.iter().enumerate() {
            position[j] = p as i64;
        }
        let seen = Pattern::random(n_queries, candidates.len(), SEEN_PER_QUERY, 0xbeef);
        group.throughput(Throughput::Elements((n_queries * candidates.len()) as u64));
        group.bench_function(
            BenchmarkId::from_parameter(format!("{n_queries}x{}", candidates.len())),
            |b| {
                b.iter(|| {
                    recommend::top_k_from_similarity(
                        black_box(&recommend::CsrRows::from(&users.csr())),
                        &rows,
                        &recommend::CsrRows::from(&similarity.csr()),
                        recommend::Candidates::Subset {
                            items: &candidates,
                            position: &position,
                        },
                        &recommend::Exclusions::<usize>::Positions(seen.excluded()),
                        K,
                    )
                    .map_err(|e| e.row)
                })
            },
        );
    }
    group.finish();
}

/// The latent-factor path at the batch sizes a service sees: one user, where the tiling
/// cannot help and the per-call overhead is the whole story, and batches large enough
/// for full tiles and threads.
fn factors_top_k(c: &mut Criterion) {
    install_pool();
    let n_items = 20_000;
    let n_users = 10_000;
    let mut group = c.benchmark_group("recommend::top_k_from_factors");
    for dim in [8usize, 64] {
        let users = dense_scores(n_users, dim, 0xbeef);
        let items = dense_scores(n_items, dim, 0xcafe);
        let bias = dense_scores(1, n_items, 0xf00d);
        let seen = Matrix::random(n_users, n_items, SEEN_PER_QUERY, 0xbeef);
        let factors = recommend::Factors {
            users: &users,
            items: &items,
            dim,
            item_bias: Some(&bias),
            user_offset: None,
        };
        for n_queries in [1usize, 100, 1_000] {
            let rows: Vec<usize> = (0..n_queries).collect();
            group.throughput(Throughput::Elements((n_queries * n_items) as u64));
            group.bench_function(
                BenchmarkId::from_parameter(format!("d{dim}/{n_queries}x{n_items}")),
                |b| {
                    b.iter(|| {
                        let seen = seen.csr();
                        recommend::top_k_from_factors(
                            black_box(&factors),
                            &rows,
                            recommend::Candidates::All(n_items),
                            &recommend::Exclusions::Items {
                                indptr: seen.indptr,
                                indices: seen.indices,
                            },
                            K,
                        )
                        .map_err(|e| e.row)
                    })
                },
            );
        }
    }
    group.finish();
}

/// EASE's path: an `axpy` per seen item over dense catalog-wide rows.
fn dense_rows_top_k(c: &mut Criterion) {
    install_pool();
    let n_items = 5_000;
    let users = Matrix::random(2_000, n_items, SEEN_PER_QUERY, 0xbeef);
    let weights = dense_scores(n_items, n_items, 0xcafe);
    let mut group = c.benchmark_group("recommend::top_k_from_dense_rows");
    for n_queries in [1usize, 100, 1_000] {
        let rows: Vec<usize> = (0..n_queries).collect();
        group.throughput(Throughput::Elements((n_queries * n_items) as u64));
        group.bench_function(
            BenchmarkId::from_parameter(format!("{n_queries}x{n_items}")),
            |b| {
                b.iter(|| {
                    let users = users.csr();
                    recommend::top_k_from_dense_rows(
                        black_box(&recommend::CsrRows::from(&users)),
                        &rows,
                        &weights,
                        recommend::Candidates::All(n_items),
                        &recommend::Exclusions::Items {
                            indptr: users.indptr,
                            indices: users.indices,
                        },
                        K,
                    )
                    .map_err(|e| e.row)
                })
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = support::criterion();
    targets = dense_top_k, fused_top_k, factors_top_k, dense_rows_top_k
}
criterion_main!(benches);
