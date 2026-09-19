//! Deterministic fixtures shared by the criterion benches.
//!
//! The benches exist to catch regressions between two builds of this crate, so every
//! input is generated from a fixed seed and the rayon pool is pinned to a fixed size:
//! a run whose thread count or matrix contents drift is a run whose numbers cannot be
//! compared with the previous one.
//!
//! Each bench binary compiles this module separately and uses a subset of it, hence the
//! blanket `dead_code` allowance.

#![allow(dead_code)]

use std::sync::Once;

use criterion::Criterion;

use skrecsys_kernels::sparse::Csr;

/// xorshift64, as used by the kernels' own test fixtures: reproducible, fast enough
/// that generating a fixture is not what a bench ends up measuring, and free of a
/// dependency whose version could change the inputs underneath us.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // A zero state is a fixed point of xorshift, so it is never a valid seed.
        Self(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A value in `[0, bound)`.
    pub fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// Owning counterpart of [`Csr`], so a bench can hand the kernels a borrowed matrix.
pub struct Matrix {
    pub n_rows: usize,
    pub n_cols: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<i64>,
    pub data: Vec<f64>,
}

impl Matrix {
    /// A matrix with about `per_row` stored values in every row, columns ascending and
    /// distinct, and values in `[0.1, 5.1)`.
    ///
    /// Distinct ascending columns are what the kernels validate at the Python boundary,
    /// so a fixture that lacked them would benchmark input no caller can pass.
    pub fn random(n_rows: usize, n_cols: usize, per_row: usize, seed: u64) -> Self {
        let per_row = per_row.min(n_cols);
        let mut rng = Rng::new(seed);
        let mut indptr = Vec::with_capacity(n_rows + 1);
        let mut indices = Vec::with_capacity(n_rows * per_row);
        let mut data = Vec::with_capacity(n_rows * per_row);
        let mut row = Vec::with_capacity(per_row);
        indptr.push(0);
        for _ in 0..n_rows {
            row.clear();
            row.extend((0..per_row).map(|_| rng.below(n_cols) as i64));
            row.sort_unstable();
            row.dedup();
            indices.extend_from_slice(&row);
            data.extend((0..row.len()).map(|_| 0.1 + rng.next_f64() * 5.0));
            indptr.push(indices.len());
        }
        Self {
            n_rows,
            n_cols,
            indptr,
            indices,
            data,
        }
    }

    pub fn csr(&self) -> Csr<'_> {
        Csr {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            indptr: &self.indptr,
            indices: &self.indices,
            data: &self.data,
        }
    }

    /// The same structure with no values attached, which is what [`skrecsys_kernels::bpr::fit`]
    /// takes.
    pub fn structure(&self) -> Csr<'_> {
        Csr {
            data: &[],
            ..self.csr()
        }
    }

    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// The Euclidean norm of every column, for the cosine kernel.
    pub fn column_norms(&self) -> Vec<f64> {
        let mut norms = vec![0.0; self.n_cols];
        for (&j, &v) in self.indices.iter().zip(&self.data) {
            norms[j as usize] += v * v;
        }
        for norm in &mut norms {
            *norm = norm.sqrt();
        }
        norms
    }

    /// A copy whose rows sum to one, which is the shape the RP3beta walk expects.
    pub fn row_normalized(&self) -> Self {
        let mut data = self.data.clone();
        for p in self.indptr.windows(2) {
            let row = &mut data[p[0]..p[1]];
            let total: f64 = row.iter().sum();
            if total > 0.0 {
                for v in row {
                    *v /= total;
                }
            }
        }
        Self {
            data,
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
        }
    }
}

/// CSR exclusion pattern over `n_rows x n_cols`, with `per_row` ascending columns each.
///
/// The selection kernels take the already-seen items in this layout; the columns have
/// to ascend within a row for their single cursor to keep up with the scan.
pub struct Pattern {
    pub indptr: Vec<usize>,
    pub indices: Vec<i64>,
}

impl Pattern {
    pub fn random(n_rows: usize, n_cols: usize, per_row: usize, seed: u64) -> Self {
        let matrix = Matrix::random(n_rows, n_cols, per_row, seed);
        Self {
            indptr: matrix.indptr,
            indices: matrix.indices,
        }
    }

    pub fn excluded(&self) -> skrecsys_kernels::ranking::Excluded<'_> {
        skrecsys_kernels::ranking::Excluded {
            indptr: &self.indptr,
            indices: &self.indices,
        }
    }

    /// An empty pattern over `n_rows`, for the benches that measure selection alone.
    pub fn empty(n_rows: usize) -> Self {
        Self {
            indptr: vec![0; n_rows + 1],
            indices: Vec::new(),
        }
    }
}

/// A dense row-major `n_rows x n_cols` score matrix.
pub fn dense_scores(n_rows: usize, n_cols: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    (0..n_rows * n_cols).map(|_| rng.next_f64()).collect()
}

static POOL: Once = Once::new();

/// Pin rayon's global pool, so two runs on the same machine use the same parallelism.
///
/// The kernels all run on the pool the caller installs -- Python builds one per call --
/// and a bench that let rayon size it by the machine's load would report the pool's
/// width as a performance change. Override with `SKRECSYS_BENCH_THREADS`.
pub fn install_pool() {
    POOL.call_once(|| {
        let threads = threads();
        // An error means a pool is already installed, which is just as deterministic.
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global();
    });
}

/// The pool width the benches run at: `SKRECSYS_BENCH_THREADS`, else four threads, or
/// fewer on a smaller machine.
pub fn threads() -> usize {
    std::env::var("SKRECSYS_BENCH_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&t| t > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map_or(1, |n| n.get())
                .min(4)
        })
}

/// Criterion configured to flag only the changes this suite can actually resolve.
///
/// Two runs of identical code drift by one to two percent here, and criterion's default
/// threshold calls anything past one percent a regression, so the default turns that
/// drift into a verdict. Three percent is above the measured noise floor and still well
/// under the size of change worth acting on; a smaller one needs more samples, a quieter
/// machine, or both, rather than a lower threshold.
pub fn criterion() -> Criterion {
    Criterion::default().noise_threshold(0.03)
}
