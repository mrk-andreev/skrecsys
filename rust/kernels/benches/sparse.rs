//! The sparse scaffolding every kernel is built on: the transpose, the parallel row
//! builder and the accumulator a sparse product is summed into.
//!
//! These are benchmarked apart from the kernels that use them because a regression here
//! shows up in all of them at once, and only here can it be told apart from a change in
//! the arithmetic layered on top.

mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use skrecsys_kernels::sparse::{Accumulator, Csc, CsrOwned};
use support::{Matrix, install_pool};

/// The interaction shapes the kernels see: a wide catalog with short rows, and a
/// narrower one with the long rows a dense-ish dataset produces.
const SHAPES: [(usize, usize, usize); 3] = [
    (10_000, 2_000, 20),
    (10_000, 2_000, 100),
    (50_000, 5_000, 20),
];

fn label(n_rows: usize, n_cols: usize, per_row: usize) -> String {
    format!("{n_rows}x{n_cols}/{per_row}")
}

fn csc_from_csr(c: &mut Criterion) {
    install_pool();
    let mut group = c.benchmark_group("Csc::from_csr");
    for (n_rows, n_cols, per_row) in SHAPES {
        let matrix = Matrix::random(n_rows, n_cols, per_row, 0x51ab);
        group.throughput(Throughput::Elements(matrix.nnz() as u64));
        group.bench_function(
            BenchmarkId::from_parameter(label(n_rows, n_cols, per_row)),
            |b| {
                b.iter(|| Csc::from_csr(black_box(&matrix.csr())));
            },
        );
    }
    group.finish();
}

/// The parallel row builder alone: `fill` only copies a row, so what is left is the
/// blocking, the per-block buffers and the concatenation.
fn csr_owned_build(c: &mut Criterion) {
    install_pool();
    let mut group = c.benchmark_group("CsrOwned::build");
    for (n_rows, n_cols, per_row) in SHAPES {
        let matrix = Matrix::random(n_rows, n_cols, per_row, 0x51ab);
        group.throughput(Throughput::Elements(matrix.nnz() as u64));
        group.bench_function(
            BenchmarkId::from_parameter(label(n_rows, n_cols, per_row)),
            |b| {
                b.iter(|| {
                    let m = matrix.csr();
                    CsrOwned::build(m.n_rows, || (), |(), i, out| out.extend(m.row(i)))
                });
            },
        );
    }
    group.finish();
}

/// One row of a sparse product: scatter into the accumulator, read the touched entries
/// back and clear it, which is the inner loop of every similarity kernel.
fn accumulator(c: &mut Criterion) {
    let mut group = c.benchmark_group("Accumulator");
    for (n_cols, per_row) in [(2_000usize, 20usize), (20_000, 100)] {
        // One item's column of the transpose, times the rows it appears in.
        let matrix = Matrix::random(2_000, n_cols, per_row, 0x51ab);
        let m = matrix.csr();
        let rows: Vec<usize> = (0..64).map(|i| i * 17 % m.n_rows).collect();
        let touched: u64 = rows
            .iter()
            .map(|&i| (m.indptr[i + 1] - m.indptr[i]) as u64)
            .sum();
        group.throughput(Throughput::Elements(touched));
        group.bench_function(
            BenchmarkId::from_parameter(format!("{n_cols}/{per_row}")),
            |b| {
                let mut acc = Accumulator::new(n_cols);
                b.iter(|| {
                    for &i in &rows {
                        for (j, v) in m.row(i) {
                            acc.add(j, v);
                        }
                    }
                    let mut total = 0.0;
                    for &j in acc.touched() {
                        total += acc.get(j);
                    }
                    acc.reset();
                    black_box(total)
                });
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = support::criterion();
    targets = csc_from_csr, csr_owned_build, accumulator
}
criterion_main!(benches);
