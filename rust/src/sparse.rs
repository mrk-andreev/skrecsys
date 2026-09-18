//! Borrowed sparse matrices shared by the kernels.

use rayon::prelude::*;

/// Borrowed compressed sparse row matrix of shape `(n_rows, n_cols)`.
///
/// The column indices stay in the `i64` layout numpy hands over: they are the largest
/// array in play, and copying them into `usize` cost a pass over `nnz` on every call.
/// They are validated against `n_cols` once at the boundary, so [`Csr::col`] is a plain
/// widening cast.
pub struct Csr<'a> {
    pub n_rows: usize,
    pub n_cols: usize,
    pub indptr: &'a [usize],
    pub indices: &'a [i64],
    pub data: &'a [f64],
}

impl Csr<'_> {
    /// The column of the `p`-th stored value.
    #[inline]
    pub fn col(&self, p: usize) -> usize {
        self.indices[p] as usize
    }

    /// The `(column, value)` pairs stored in row `i`.
    pub fn row(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let range = self.indptr[i]..self.indptr[i + 1];
        self.indices[range.clone()]
            .iter()
            .map(|&j| j as usize)
            .zip(self.data[range].iter().copied())
    }
}

/// Column-oriented copy of a [`Csr`] matrix (libFM's `data_t`, implicit's `users.T`), rows ascending.
pub struct Csc {
    pub indptr: Vec<usize>,
    pub rows: Vec<usize>,
    pub data: Vec<f64>,
}

impl Csc {
    pub fn from_csr(x: &Csr) -> Self {
        let mut indptr = vec![0; x.n_cols + 1];
        for &j in x.indices {
            indptr[j as usize + 1] += 1;
        }
        for j in 0..x.n_cols {
            indptr[j + 1] += indptr[j];
        }
        let mut next = indptr.clone();
        let mut rows = vec![0; x.indices.len()];
        let mut data = vec![0.0; x.indices.len()];
        for c in 0..x.n_rows {
            for k in x.indptr[c]..x.indptr[c + 1] {
                let j = x.col(k);
                rows[next[j]] = c;
                data[next[j]] = x.data[k];
                next[j] += 1;
            }
        }
        Self { indptr, rows, data }
    }

    pub fn column(&self, j: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let range = self.indptr[j]..self.indptr[j + 1];
        self.rows[range.clone()]
            .iter()
            .copied()
            .zip(self.data[range].iter().copied())
    }
}

/// One block of rows built by [`CsrOwned::build`]: their lengths and their entries.
type Fragment = (Vec<usize>, Vec<(usize, f64)>);

/// Owned compressed sparse row matrix produced by a kernel.
pub struct CsrOwned {
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
}

impl CsrOwned {
    /// Build the rows in parallel, each worker appending to buffers of its own.
    ///
    /// `fill` appends row `i`'s `(column, value)` pairs, which must ascend by column,
    /// to the buffer it is handed; that buffer is reused across the rows of a block, so
    /// a row costs no allocation of its own. Collecting a `Vec` per row instead means one
    /// allocation per row and a second pass to concatenate, which is what the similarity
    /// kernels used to pay for every item in the catalog.
    pub fn build<S, I, F>(n_rows: usize, init: I, fill: F) -> Self
    where
        S: Send,
        I: Fn() -> S + Sync + Send,
        F: Fn(&mut S, usize, &mut Vec<(usize, f64)>) + Sync + Send,
    {
        // Enough blocks to keep every worker fed, few enough that the buffers are few.
        let block = n_rows.div_ceil(rayon::current_num_threads() * 4).max(1);
        let fragments: Vec<Fragment> = (0..n_rows.div_ceil(block))
            .into_par_iter()
            .map_init(init, |state, b| {
                let stop = ((b + 1) * block).min(n_rows);
                let mut lengths = Vec::with_capacity(stop - b * block);
                let mut entries: Vec<(usize, f64)> = Vec::new();
                for row in b * block..stop {
                    let before = entries.len();
                    fill(state, row, &mut entries);
                    lengths.push(entries.len() - before);
                }
                (lengths, entries)
            })
            .collect();

        let nnz = fragments.iter().map(|(_, e)| e.len()).sum();
        let mut indptr = Vec::with_capacity(n_rows + 1);
        let mut indices = Vec::with_capacity(nnz);
        let mut data = Vec::with_capacity(nnz);
        indptr.push(0);
        for (lengths, entries) in &fragments {
            let mut at = 0;
            for &length in lengths {
                for &(j, value) in &entries[at..at + length] {
                    indices.push(j);
                    data.push(value);
                }
                at += length;
                indptr.push(indices.len());
            }
        }
        Self {
            indptr,
            indices,
            data,
        }
    }
}

/// Dense accumulator that remembers which indices were touched, so a sparse row of a
/// matrix product can be built in `O(nnz)` rather than `O(n_cols)` (implicit's
/// `SparseMatrixMultiplier`).
pub struct Accumulator {
    sums: Vec<f64>,
    seen: Vec<bool>,
    touched: Vec<usize>,
}

impl Accumulator {
    pub fn new(n: usize) -> Self {
        Self {
            sums: vec![0.0; n],
            seen: vec![false; n],
            touched: Vec::new(),
        }
    }

    pub fn add(&mut self, index: usize, value: f64) {
        self.sums[index] += value;
        if !self.seen[index] {
            self.seen[index] = true;
            self.touched.push(index);
        }
    }

    /// Indices touched since the last [`Accumulator::reset`], in first-touch order.
    pub fn touched(&self) -> &[usize] {
        &self.touched
    }

    /// Whether `index` was touched since the last [`Accumulator::reset`].
    pub fn is_touched(&self, index: usize) -> bool {
        self.seen[index]
    }

    pub fn get(&self, index: usize) -> f64 {
        self.sums[index]
    }

    /// Clear the accumulated row, ready for the next one.
    pub fn reset(&mut self) {
        for &index in &self.touched {
            self.sums[index] = 0.0;
            self.seen[index] = false;
        }
        self.touched.clear();
    }
}

#[cfg(test)]
pub mod testing {
    use super::Csr;

    /// Owning counterpart of [`Csr`], so tests can build fixtures from dense rows.
    pub struct Owned {
        n_rows: usize,
        n_cols: usize,
        indptr: Vec<usize>,
        indices: Vec<i64>,
        data: Vec<f64>,
    }

    impl Owned {
        pub fn from_dense(dense: &[Vec<f64>]) -> Self {
            let mut indptr = vec![0];
            let mut indices = Vec::new();
            let mut data = Vec::new();
            for row in dense {
                for (j, &v) in row.iter().enumerate() {
                    if v != 0.0 {
                        indices.push(j as i64);
                        data.push(v);
                    }
                }
                indptr.push(indices.len());
            }
            Self {
                n_rows: dense.len(),
                n_cols: dense[0].len(),
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
    }

    /// The `(column, value)` pairs stored in row `i`.
    pub fn row(matrix: &super::CsrOwned, i: usize) -> Vec<(usize, f64)> {
        (matrix.indptr[i]..matrix.indptr[i + 1])
            .map(|p| (matrix.indices[p], matrix.data[p]))
            .collect()
    }

    /// A sparse matrix of reproducible values, without depending on an RNG crate.
    pub fn pseudo_random_dense(n_users: usize, n_items: usize) -> Vec<Vec<f64>> {
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        (0..n_users)
            .map(|_| {
                (0..n_items)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        if state.is_multiple_of(3) {
                            (state % 1000) as f64 / 250.0 + 0.1
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect()
    }
}
