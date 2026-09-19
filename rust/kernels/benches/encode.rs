//! Identifier encoding and matrix assembly: what every `fit` call pays before a model
//! sees any data.
//!
//! `factorize` has two paths, chosen by how wide a span the ids cover, and both are
//! measured here: bucketing ids that are counted upwards from zero, and hashing ids
//! that are spread out. A change that moves the cutoff shows up as one of the two
//! getting suddenly faster or slower.

mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use skrecsys_kernels::encode;
use support::{Rng, install_pool};

/// How many interactions the fixtures stand in for.
const NNZ: usize = 1_000_000;
const N_USERS: usize = 50_000;
const N_ITEMS: usize = 5_000;

/// Ids drawn from `0..distinct`, which is the shape a re-encoded dataset has.
fn contiguous_ids(len: usize, distinct: usize, seed: u64) -> Vec<i64> {
    let mut rng = Rng::new(seed);
    (0..len).map(|_| rng.below(distinct) as i64).collect()
}

/// The same ids spread far enough apart that bucketing them would cost more memory than
/// the input, so `factorize` hashes them instead.
fn spread_ids(len: usize, distinct: usize, seed: u64) -> Vec<i64> {
    contiguous_ids(len, distinct, seed)
        .into_iter()
        .map(|v| v * 1_000_003 + 7)
        .collect()
}

fn factorize(c: &mut Criterion) {
    install_pool();
    let mut group = c.benchmark_group("encode::factorize");
    group.sample_size(30);
    group.throughput(Throughput::Elements(NNZ as u64));
    for (name, values) in [
        ("contiguous", contiguous_ids(NNZ, N_USERS, 0x1d)),
        ("spread", spread_ids(NNZ, N_USERS, 0x1d)),
    ] {
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| encode::factorize(black_box(&values)));
        });
    }
    group.finish();
}

fn coo_to_csr(c: &mut Criterion) {
    install_pool();
    let mut rng = Rng::new(0x2c);
    // Duplicate pairs are ordinary input -- the kernel sums them -- and at this density
    // a fair number of them occur by chance, as in a real interaction log.
    let rows: Vec<i64> = (0..NNZ).map(|_| rng.below(N_USERS) as i64).collect();
    let cols: Vec<i64> = (0..NNZ).map(|_| rng.below(N_ITEMS) as i64).collect();
    let data: Vec<f64> = (0..NNZ).map(|_| rng.next_f64()).collect();

    let mut group = c.benchmark_group("encode::coo_to_csr");
    group.sample_size(30);
    group.throughput(Throughput::Elements(NNZ as u64));
    group.bench_function(
        BenchmarkId::from_parameter(format!("{N_USERS}x{N_ITEMS}/{NNZ}")),
        |b| b.iter(|| encode::coo_to_csr(black_box(&rows), &cols, &data, N_USERS, N_ITEMS)),
    );
    group.finish();
}

criterion_group! {
    name = benches;
    config = support::criterion();
    targets = factorize, coo_to_csr
}
criterion_main!(benches);
