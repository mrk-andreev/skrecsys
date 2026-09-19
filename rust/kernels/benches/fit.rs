//! The training kernels: EASE's closed form, SLIM's per-item coordinate descent, the
//! factorization machine's ALS sweeps and BPR's sampled SGD.
//!
//! The four cost wildly different amounts per sample, so each group sets its own sample
//! count rather than letting criterion's default make the slow ones take minutes. The
//! two in-place fits are handed a fresh copy of the parameters for every iteration, so
//! that each measured call starts from the same state and does the same work.

mod support;

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use faer::Par;

use skrecsys_kernels::sparse::Csr;
use skrecsys_kernels::{bpr, ease, fm_als, slim};
use support::{Matrix, Rng, install_pool};

fn ease_gram(c: &mut Criterion) {
    install_pool();
    let mut group = c.benchmark_group("ease::gram");
    group.sample_size(30);
    for n_cols in [512usize, 1_024] {
        let matrix = Matrix::random(10_000, n_cols, 20, 0xea5e);
        group.throughput(Throughput::Elements(matrix.nnz() as u64));
        group.bench_function(BenchmarkId::from_parameter(n_cols), |b| {
            b.iter(|| ease::gram(black_box(&matrix.csr())));
        });
    }
    group.finish();
}

/// The Cholesky solve, which is cubic in the catalog size and is what dominates EASE on
/// anything but a tiny catalog.
fn ease_weights(c: &mut Criterion) {
    install_pool();
    let mut group = c.benchmark_group("ease::weights");
    group.sample_size(20);
    for n_cols in [512usize, 1_024] {
        let matrix = Matrix::random(10_000, n_cols, 20, 0xea5e);
        let gram = ease::gram(&matrix.csr());
        group.bench_function(BenchmarkId::from_parameter(n_cols), |b| {
            b.iter(|| {
                ease::weights(black_box(&gram), n_cols, 250.0, Par::rayon(0))
                    .map(|w| w.len())
                    .map_err(|_| "not positive definite")
            });
        });
    }
    group.finish();
}

fn slim_similarity(c: &mut Criterion) {
    install_pool();
    let mut group = c.benchmark_group("slim::similarity");
    group.sample_size(10);
    let enet = slim::ElasticNet {
        alpha: 1e-3,
        l1_ratio: 0.1,
        positive: true,
        // The estimator's default is 100; half of that keeps the bench under a second a
        // sample while still exercising the descent rather than only its first sweep.
        max_iter: 50,
        tol: 1e-4,
    };
    for n_cols in [256usize, 512] {
        let matrix = Matrix::random(5_000, n_cols, 20, 0x5115);
        group.throughput(Throughput::Elements(matrix.nnz() as u64));
        group.bench_function(BenchmarkId::from_parameter(n_cols), |b| {
            b.iter(|| slim::similarity(black_box(&matrix.csr()), &enet, 100).1);
        });
    }
    group.finish();
}

/// One-hot user and item features over `n_rows` interactions, which is the design matrix
/// a factorization machine is trained on.
struct Design {
    indptr: Vec<usize>,
    indices: Vec<i64>,
    data: Vec<f64>,
    y: Vec<f64>,
    n_rows: usize,
    n_features: usize,
}

impl Design {
    fn new(n_rows: usize, n_users: usize, n_items: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut indptr = Vec::with_capacity(n_rows + 1);
        let mut indices = Vec::with_capacity(n_rows * 2);
        let mut y = Vec::with_capacity(n_rows);
        indptr.push(0);
        for _ in 0..n_rows {
            let user = rng.below(n_users);
            let item = n_users + rng.below(n_items);
            indices.push(user as i64);
            indices.push(item as i64);
            indptr.push(indices.len());
            y.push(1.0 + rng.next_f64() * 4.0);
        }
        Self {
            data: vec![1.0; indices.len()],
            indptr,
            indices,
            y,
            n_rows,
            n_features: n_users + n_items,
        }
    }

    fn csr(&self) -> Csr<'_> {
        Csr {
            n_rows: self.n_rows,
            n_cols: self.n_features,
            indptr: &self.indptr,
            indices: &self.indices,
            data: &self.data,
        }
    }
}

fn fm_als_fit(c: &mut Criterion) {
    install_pool();
    let mut group = c.benchmark_group("fm_als::fit");
    group.sample_size(20);
    let (n_users, n_items) = (5_000usize, 1_000usize);
    let design = Design::new(200_000, n_users, n_items, 0xf3a1);
    // One group per feature block, as the estimator sets them up.
    let group_of: Vec<usize> = (0..design.n_features)
        .map(|j| usize::from(j >= n_users))
        .collect();
    let reg_w = vec![1.0, 1.0];
    let reg_v = vec![1.0, 1.0];

    for n_factors in [8usize, 32] {
        let mut rng = Rng::new(0xf3a2);
        let w0 = vec![0.0; design.n_features];
        let v0: Vec<f64> = (0..n_factors * design.n_features)
            .map(|_| (rng.next_f64() - 0.5) * 0.1)
            .collect();
        group.throughput(Throughput::Elements(design.n_rows as u64));
        group.bench_function(BenchmarkId::from_parameter(n_factors), |b| {
            b.iter_batched_ref(
                || (w0.clone(), v0.clone()),
                |(w, v)| {
                    let mut model = fm_als::Model {
                        w0: 0.0,
                        w,
                        v,
                        group: &group_of,
                        reg0: 1.0,
                        reg_w: &reg_w,
                        reg_v: &reg_v,
                    };
                    fm_als::fit(black_box(&design.csr()), &design.y, &mut model, 1)
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bpr_fit(c: &mut Criterion) {
    install_pool();
    let mut group = c.benchmark_group("bpr::fit");
    group.sample_size(20);
    let (n_users, n_items) = (10_000usize, 2_000usize);
    let matrix = Matrix::random(n_users, n_items, 20, 0xb9c0);

    for n_factors in [16usize, 64] {
        let mut rng = Rng::new(0xb9c1);
        let mut factors = |n: usize| -> Vec<f64> {
            (0..n * n_factors)
                .map(|_| (rng.next_f64() - 0.5) * 0.1)
                .collect()
        };
        let users0 = factors(n_users);
        let items0 = factors(n_items);
        let hyper = bpr::Hyper {
            n_factors,
            learning_rate: 0.01,
            regularization: 0.01,
            use_bias: true,
            n_iter: 1,
            seed: 0xb9c2,
        };
        group.throughput(Throughput::Elements(matrix.nnz() as u64));
        group.bench_function(BenchmarkId::from_parameter(n_factors), |b| {
            b.iter_batched_ref(
                || (users0.clone(), items0.clone(), vec![0.0; n_items]),
                |(users, items, bias)| {
                    let mut model = bpr::Model {
                        user_factors: users,
                        item_factors: items,
                        item_bias: bias,
                    };
                    bpr::fit(black_box(&matrix.structure()), None, &mut model, &hyper)
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = support::criterion();
    targets = ease_gram, ease_weights, slim_similarity, fm_als_fit, bpr_fit
}
criterion_main!(benches);
