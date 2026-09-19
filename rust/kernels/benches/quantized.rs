//! The other approximate path: scanning narrow codes instead of walking a graph.
//!
//! These sit next to `hnsw`'s benches and measure the opposite bet. A graph walk is
//! meant not to grow with the catalog; this scan is meant to grow with it *more slowly
//! per candidate* than an exact pass does. So the question here is cost per candidate,
//! not cost per query, and the throughput below is in candidates: a scan that is worth
//! running has to beat the exact dot product this crate already ships next door, at the
//! same catalog, on the same thread.
//!
//! Two things this file is for in particular. The first is the code width: the whole
//! reason `bits` is a parameter is that narrower codes should cost less, and a run where
//! four bits is slower than eight means the unpacking has stopped paying for the memory
//! it saves. The second is `oversample`, which sets the shortlist and should barely show
//! up at all -- the heap sees every candidate either way, and only the rerank grows.
//!
//! What the codes cost in *answers* is not measurable here; `benchmarks/indexes.py` does
//! that against real models. This file is about time alone.

mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use skrecsys_kernels::quantized::{self, Codes, Store, row_bytes};
use skrecsys_kernels::ranking::Excluded;
use skrecsys_kernels::vectors::{DenseItems, Items, Probe, Queries, SparseItems};
use support::{Matrix, Pattern, Rng, criterion};

/// What a query asks for, and how many items it has already seen.
const K: usize = 10;
const SEEN_PER_QUERY: usize = 20;

/// The width a latent-factor model hands the index.
const DIM: usize = 64;

/// Row-major vectors in `[-1, 1)`: the shape a latent-factor model hands the index.
fn dense_vectors(n_items: usize, dim: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n_items * dim)
        .map(|_| rng.next_f64() * 2.0 - 1.0)
        .collect()
}

/// Quantize `values` over `[lo, hi]` and pack them, as the Python side does.
fn encode(values: &[f64], lo: f64, hi: f64, bits: u32, dim: usize) -> (Vec<u8>, f64, f64) {
    let levels = f64::from((1u32 << bits) - 1);
    let scale = if hi > lo { (hi - lo) / levels } else { 0.0 };
    let codes: Vec<u8> = values
        .iter()
        .map(|&v| (((v.clamp(lo, hi) - lo) / scale).round()).clamp(0.0, levels) as u8)
        .collect();
    let per_byte = 8 / bits as usize;
    let stride = row_bytes(dim, bits);
    let mut packed = vec![0u8; codes.len().div_ceil(dim) * stride];
    for (row, chunk) in codes.chunks(dim).enumerate() {
        for (i, &code) in chunk.iter().enumerate() {
            packed[row * stride + i / per_byte] |= code << ((i % per_byte) as u32 * bits);
        }
    }
    (packed, scale, lo)
}

/// The scan at every code width, against the exact dot product it has to beat.
///
/// The `exact` row is the same catalog scored the way `index=None` scores it, through
/// `vectors::DenseItems` on one thread. It is the baseline that decides whether any of
/// the rows above it are worth having.
fn scan_dense(c: &mut Criterion) {
    let n_queries = 100;
    let mut group = c.benchmark_group("quantized::top_k/dense");
    for n_items in [20_000usize, 200_000] {
        let vectors = dense_vectors(n_items, DIM, 0xcafe);
        let items = DenseItems {
            vectors: &vectors,
            dim: DIM,
        };
        let candidates: Vec<usize> = (0..n_items).collect();
        let query_data = dense_vectors(n_queries, DIM, 0xf00d);
        let queries = Queries::Dense {
            data: &query_data,
            dim: DIM,
        };
        let seen = Pattern::random(n_queries, n_items, SEEN_PER_QUERY, 0xbeef);
        let excluded: Excluded<'_> = seen.excluded();

        // Elements are candidates, not queries: a scan is linear in the catalog by
        // construction, so the number worth comparing is what one candidate costs.
        group.throughput(Throughput::Elements((n_queries * n_items) as u64));
        for bits in [8u32, 4, 2, 1] {
            let (packed, scale, offset) = encode(&vectors, -1.0, 1.0, bits, DIM);
            let scales = vec![scale; DIM];
            let offsets = vec![offset; DIM];
            let store = Store::Dense {
                codes: Codes {
                    data: &packed,
                    bits,
                    stride: row_bytes(DIM, bits),
                },
                dim: DIM,
                scale: &scales,
                offset: &offsets,
            };
            group.bench_function(
                BenchmarkId::from_parameter(format!("{n_items}/bits{bits}")),
                |b| {
                    b.iter(|| {
                        quantized::top_k(
                            black_box(&store),
                            &items,
                            &queries,
                            &candidates,
                            &excluded,
                            K,
                            4,
                        )
                        .map_err(|e| e.row)
                    })
                },
            );
        }

        group.bench_function(
            BenchmarkId::from_parameter(format!("{n_items}/exact")),
            |b| {
                b.iter(|| {
                    let mut total = 0.0;
                    for q in 0..n_queries {
                        let query = &query_data[q * DIM..(q + 1) * DIM];
                        let probe = Probe::Dense(query);
                        for &item in &candidates {
                            total += black_box(&items).score(item, &probe);
                        }
                    }
                    total
                })
            },
        );
    }
    group.finish();
}

/// What widening the shortlist costs, which should be almost nothing.
///
/// Every candidate is scored and offered to the heap whatever `oversample` is; only the
/// rerank, which runs over `oversample * k` items rather than the catalog, grows. A run
/// where this curve bends means the heap has become the scan's cost rather than the dot
/// product, which would be worth knowing.
fn shortlist_width(c: &mut Criterion) {
    let (n_queries, n_items, bits) = (100usize, 200_000usize, 8u32);
    let vectors = dense_vectors(n_items, DIM, 0xcafe);
    let items = DenseItems {
        vectors: &vectors,
        dim: DIM,
    };
    let (packed, scale, offset) = encode(&vectors, -1.0, 1.0, bits, DIM);
    let (scales, offsets) = (vec![scale; DIM], vec![offset; DIM]);
    let candidates: Vec<usize> = (0..n_items).collect();
    let query_data = dense_vectors(n_queries, DIM, 0xf00d);
    let queries = Queries::Dense {
        data: &query_data,
        dim: DIM,
    };
    let seen = Pattern::random(n_queries, n_items, SEEN_PER_QUERY, 0xbeef);
    let excluded: Excluded<'_> = seen.excluded();

    let mut group = c.benchmark_group("quantized::top_k/oversample");
    group.throughput(Throughput::Elements((n_queries * n_items) as u64));
    for oversample in [1usize, 4, 16, 64] {
        let store = Store::Dense {
            codes: Codes {
                data: &packed,
                bits,
                stride: row_bytes(DIM, bits),
            },
            dim: DIM,
            scale: &scales,
            offset: &offsets,
        };
        group.bench_function(
            BenchmarkId::from_parameter(format!("ov{oversample}")),
            |b| {
                b.iter(|| {
                    quantized::top_k(
                        black_box(&store),
                        &items,
                        &queries,
                        &candidates,
                        &excluded,
                        K,
                        oversample,
                    )
                    .map_err(|e| e.row)
                })
            },
        );
    }
    group.finish();
}

/// The item-item shape: short sparse rows over a catalog-sized dimension.
///
/// The codes ride the matrix's own `indptr` and `indices`, so what is quantized is the
/// `data` array alone -- a third of a CSR, not the whole of it. That bounds how much
/// this shape can ever win, which is the point of measuring it apart from the dense one.
fn scan_sparse(c: &mut Criterion) {
    let (n_queries, n_items) = (200usize, 20_000usize);
    let matrix = Matrix::random(n_items, n_items, 100, 0xcafe);
    let items = SparseItems {
        matrix: matrix.csr(),
    };
    let candidates: Vec<usize> = (0..n_items).collect();
    let users = Matrix::random(n_queries, n_items, SEEN_PER_QUERY, 0xbeef);
    let queries = Queries::Scattered(users.csr());
    let seen = Pattern::random(n_queries, n_items, SEEN_PER_QUERY, 0xbeef);
    let excluded: Excluded<'_> = seen.excluded();

    let mut group = c.benchmark_group("quantized::top_k/sparse");
    group.throughput(Throughput::Elements((n_queries * n_items) as u64));
    for bits in [8u32, 4] {
        let (packed, scale, offset) = encode(&matrix.data, 0.0, 5.1, bits, matrix.data.len());
        let store = Store::Sparse {
            codes: Codes {
                data: &packed,
                bits,
                stride: 1,
            },
            matrix: matrix.csr(),
            scale,
            offset,
        };
        group.bench_function(BenchmarkId::from_parameter(format!("bits{bits}")), |b| {
            b.iter(|| {
                quantized::top_k(
                    black_box(&store),
                    &items,
                    &queries,
                    &candidates,
                    &excluded,
                    K,
                    4,
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
    targets = scan_dense, shortlist_width, scan_sparse
}
criterion_main!(benches);
