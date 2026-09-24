//! The vector stores an index scores against, and the query forms it scores with.
//!
//! Everything here maximizes a similarity rather than minimizing a distance. Every
//! recommender in the library scores an item by a dot product, so a distance would only
//! be a sign flip that a search has to undo, and [`Candidate`] already orders by
//! descending score with a deterministic tie-break.
//!
//! [`Items`] carries two comparisons, because an index needs both: `similarity` compares
//! one stored item with another, which is what HNSW construction links by, and `score`
//! compares an incoming query with a stored item, which is what every search does. A
//! query is prepared once into a [`Probe`] before it is scored against the many items a
//! traversal or a scan visits.
//!
//! The [`Probe`] variants exist because there is no single cheapest form. An item-item
//! model stores short sparse item vectors over a catalog-sized dimension, so a query is
//! worth scattering once into a dense buffer and gathering the item's stored entries
//! against it. EASE stores *dense* item vectors that are also catalog-sized, so the same
//! trick would cost a full pass per score; there the query stays sparse and the gather
//! runs over its handful of nonzeros instead. Latent-factor models are short and dense on
//! both sides and just want a dot product.
//!
//! [`quantized`] stores the same vectors a second time as narrow codes and scores those
//! in its own scan, then reranks the survivors through this module -- so the exact
//! scoring lives here once and both indexes share it.
//!
//! [`Candidate`]: crate::ranking::Candidate
//! [`quantized`]: crate::quantized

use multiversion::multiversion;

use crate::sparse::Csr;

/// One query, prepared for repeated scoring against stored items.
pub enum Probe<'a> {
    /// A short dense query vector, scored by a dot product over `dim` values.
    Dense(&'a [f64]),
    /// A query scattered into a zeroed buffer of `dim` values, scored by gathering the
    /// item's stored entries out of it. Worth the scatter only when items are sparse.
    Scattered(&'a [f64]),
    /// A query left sparse, scored by gathering the item's vector at the query's own
    /// columns. Worth it when the item vectors are long, dense or otherwise.
    Sparse { indices: &'a [i64], data: &'a [f64] },
}

/// The item vectors a graph is built over and searched against.
pub trait Items: Sync {
    /// How many items the graph has nodes for.
    fn len(&self) -> usize;

    /// Whether there is nothing to index.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of values a [`Probe::Scattered`] buffer must hold.
    fn dim(&self) -> usize;

    /// Similarity of two stored items, which is what construction links by.
    fn similarity(&self, a: usize, b: usize) -> f64;

    /// Similarity of a prepared query to a stored item. Higher is nearer.
    fn score(&self, item: usize, probe: &Probe<'_>) -> f64;
}

/// Row-major dense vectors, one row per item.
pub struct DenseItems<'a> {
    pub vectors: &'a [f64],
    pub dim: usize,
}

impl<'a> DenseItems<'a> {
    /// The vector of item `i`.
    #[inline]
    pub fn row(&self, i: usize) -> &'a [f64] {
        &self.vectors[i * self.dim..(i + 1) * self.dim]
    }
}

impl Items for DenseItems<'_> {
    fn len(&self) -> usize {
        self.vectors.len().checked_div(self.dim).unwrap_or(0)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    fn similarity(&self, a: usize, b: usize) -> f64 {
        dot(self.row(a), self.row(b))
    }

    #[inline]
    fn score(&self, item: usize, probe: &Probe<'_>) -> f64 {
        let vector = self.row(item);
        match probe {
            Probe::Dense(query) | Probe::Scattered(query) => dot(query, vector),
            // EASE: the query's nonzeros are few and the item vector is catalog-sized,
            // so the gather runs over the query rather than over the vector.
            Probe::Sparse { indices, data } => indices
                .iter()
                .zip(*data)
                .map(|(&j, value)| value * vector[j as usize])
                .sum(),
        }
    }
}

/// Sparse rows, one per item.
///
/// Row `j` is item `j`'s vector: the weights that make up item `j`'s score, which is the
/// transposed orientation of `similarity_` and exactly what `_neighbors_of_items` already
/// returns on the Python side.
pub struct SparseItems<'a> {
    pub matrix: Csr<'a>,
}

impl Items for SparseItems<'_> {
    fn len(&self) -> usize {
        self.matrix.n_rows
    }

    fn dim(&self) -> usize {
        self.matrix.n_cols
    }

    #[inline]
    fn similarity(&self, a: usize, b: usize) -> f64 {
        // Construction compares two stored rows, both short and both sorted, so the
        // merge is cheaper than scattering one of them into a catalog-sized buffer.
        sparse_dot(&self.matrix, a, &self.matrix, b)
    }

    #[inline]
    fn score(&self, item: usize, probe: &Probe<'_>) -> f64 {
        let range = self.matrix.indptr[item]..self.matrix.indptr[item + 1];
        match probe {
            Probe::Scattered(query) | Probe::Dense(query) => self.matrix.indices[range.clone()]
                .iter()
                .zip(&self.matrix.data[range])
                .map(|(&j, value)| value * query[j as usize])
                .sum(),
            Probe::Sparse { indices, data } => merge_dot(
                indices,
                data,
                &self.matrix.indices[range.clone()],
                &self.matrix.data[range],
            ),
        }
    }
}

/// `a * b + c`, fused into one instruction where the hardware has one.
///
/// Rust never contracts `a * b + c` on its own -- that would change the rounding, and it
/// has no fast-math -- so an unfused loop issues a multiply and a dependent add for
/// every term. `mul_add` asks for the fused form explicitly, which is one `fmla` on
/// aarch64 (always present there) and one `vfmadd` in an x86 clone built with `fma`.
/// Where the hardware has no FMA, `mul_add` would become a call into libm's software
/// emulation, so those clones keep the plain expression; `FMA` is a constant per clone
/// and the branch folds away.
#[inline(always)]
pub(crate) fn madd<const FMA: bool>(a: f64, b: f64, c: f64) -> f64 {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

/// Whether the target this is compiled for fuses a multiply-add in hardware.
///
/// Only the baseline: an x86 clone that `multiversion` builds with `fma` asks
/// `target_cfg_f!` instead, which sees the clone's features rather than the crate's.
pub(crate) const BASELINE_FMA: bool = cfg!(target_arch = "aarch64") || cfg!(target_feature = "fma");

/// Independent accumulators in [`dot`]'s main loop.
///
/// A dot product is a chain of dependent adds, so its speed is the number of chains in
/// flight, not the vector width. Apple's cores retire four 2-lane NEON FMAs a cycle at
/// a latency of about four, which wants sixteen or so `f64` accumulators to stay busy;
/// the four this used to keep filled two registers and left most of the FP pipes idle.
/// On x86 sixteen is four AVX2 registers, still well within the sixteen it has.
const LANES: usize = 16;

/// Dot product of two equally long vectors.
///
/// Cloned for the same targets as the SLIM inner loop: AVX-512 is left out because the
/// downclocking it causes costs more than the extra lanes win on the chips that have it.
/// aarch64 needs no clone at all -- NEON and FMA are in its baseline, and the default
/// `aarch64-apple-darwin` CPU is `apple-m1` -- so there the function is compiled once and
/// vectorized for NEON by the ordinary auto-vectorizer.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse2"))]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    const FMA: bool = BASELINE_FMA || multiversion::target::target_cfg_f!(target_feature = "fma");
    let mut acc = [0.0f64; LANES];
    let (left, left_rest) = a.as_chunks::<LANES>();
    let (right, right_rest) = b.as_chunks::<LANES>();
    for (x, y) in left.iter().zip(right) {
        for i in 0..LANES {
            acc[i] = madd::<FMA>(x[i], y[i], acc[i]);
        }
    }
    // A short vector -- a latent-factor model's eight or so factors -- never reaches the
    // main loop, so the remainder is folded four at a time rather than one.
    let mut tail = [0.0f64; 4];
    let (left_quads, left_rest) = left_rest.as_chunks::<4>();
    let (right_quads, right_rest) = right_rest.as_chunks::<4>();
    for (x, y) in left_quads.iter().zip(right_quads) {
        for i in 0..4 {
            tail[i] = madd::<FMA>(x[i], y[i], tail[i]);
        }
    }
    for i in 0..4 {
        acc[i] += tail[i];
    }
    let mut total = reduce(&acc);
    for (x, y) in left_rest.iter().zip(right_rest) {
        total = madd::<FMA>(*x, *y, total);
    }
    total
}

/// Pairwise sum of the accumulators, which is both shorter and better conditioned than
/// folding them left to right.
#[inline(always)]
pub(crate) fn reduce<const N: usize>(acc: &[f64; N]) -> f64 {
    let mut values = *acc;
    let mut width = N;
    while width > 1 {
        width /= 2;
        for i in 0..width {
            values[i] += values[i + width];
        }
    }
    values[0]
}

/// `y += a * x` over two equally long slices.
///
/// Nothing but contiguous loads and stores, the shape a compiler vectorizes well; the
/// clones exist because the default x86-64 baseline is only SSE2, and FMA is used where
/// [`madd`] says it is a single instruction. Unlike [`dot`] there is no chain to break:
/// every element is independent, so one pass is already as parallel as it gets.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse2"))]
pub(crate) fn axpy(y: &mut [f64], x: &[f64], a: f64) {
    const FMA: bool = BASELINE_FMA || multiversion::target::target_cfg_f!(target_feature = "fma");
    for (y, x) in y.iter_mut().zip(x) {
        *y = madd::<FMA>(a, *x, *y);
    }
}

/// Dot product of two sorted sparse rows of one matrix, by merge intersection.
#[inline]
fn sparse_dot(left: &Csr<'_>, a: usize, right: &Csr<'_>, b: usize) -> f64 {
    let (p, q) = (
        left.indptr[a]..left.indptr[a + 1],
        right.indptr[b]..right.indptr[b + 1],
    );
    merge_dot(
        &left.indices[p.clone()],
        &left.data[p],
        &right.indices[q.clone()],
        &right.data[q],
    )
}

/// Dot product of two sparse vectors given as sorted `(indices, data)` runs.
///
/// Both operands come from scipy CSR matrices with sorted indices, so the columns ascend
/// and the merge is one pass over the union rather than a lookup per entry.
#[inline]
fn merge_dot(a_indices: &[i64], a_data: &[f64], b_indices: &[i64], b_data: &[f64]) -> f64 {
    let (mut p, mut q) = (0usize, 0usize);
    let mut total = 0.0;
    while p < a_indices.len() && q < b_indices.len() {
        match a_indices[p].cmp(&b_indices[q]) {
            std::cmp::Ordering::Less => p += 1,
            std::cmp::Ordering::Greater => q += 1,
            std::cmp::Ordering::Equal => {
                total += a_data[p] * b_data[q];
                p += 1;
                q += 1;
            }
        }
    }
    total
}

/// The queries a batch search ranks, in the form each one is cheapest to score in.
pub enum Queries<'a> {
    /// Short dense query vectors, `dim` values each.
    Dense { data: &'a [f64], dim: usize },
    /// Sparse queries scored against long item vectors, kept sparse.
    Sparse(Csr<'a>),
    /// Sparse queries scored against short sparse item vectors, worth scattering once
    /// into a catalog-sized buffer before the traversal gathers against it.
    Scattered(Csr<'a>),
}

impl Queries<'_> {
    pub fn len(&self) -> usize {
        match self {
            Self::Dense { data, dim } => {
                if *dim == 0 {
                    0
                } else {
                    data.len() / dim
                }
            }
            Self::Sparse(csr) | Self::Scattered(csr) => csr.n_rows,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The buffer width a worker must reserve, or `None` when nothing is scattered.
    pub fn scatter_dim(&self) -> Option<usize> {
        match self {
            Self::Scattered(csr) => Some(csr.n_cols),
            _ => None,
        }
    }

    /// Prepare query `q` and hand it to `f` as a probe.
    pub(crate) fn with_probe<R>(
        &self,
        q: usize,
        scatter: &mut Option<Scatter>,
        f: impl FnOnce(&Probe<'_>) -> R,
    ) -> R {
        match self {
            Self::Dense { data, dim } => f(&Probe::Dense(&data[q * dim..(q + 1) * dim])),
            Self::Sparse(csr) => {
                let range = csr.indptr[q]..csr.indptr[q + 1];
                f(&Probe::Sparse {
                    indices: &csr.indices[range.clone()],
                    data: &csr.data[range],
                })
            }
            Self::Scattered(csr) => {
                let range = csr.indptr[q]..csr.indptr[q + 1];
                let scatter = scatter
                    .as_mut()
                    .expect("a scattered query set reserves a buffer");
                scatter.with(&csr.indices[range.clone()], &csr.data[range], f)
            }
        }
    }
}

/// A zeroed buffer a sparse query is scattered into, cleared by walking its own columns.
///
/// Clearing costs the query's nonzeros rather than the catalog, which is the same trick
/// [`Accumulator`] plays for the fused exact path.
///
/// [`Accumulator`]: crate::sparse::Accumulator
pub struct Scatter {
    values: Vec<f64>,
}

impl Scatter {
    pub fn new(dim: usize) -> Self {
        Self {
            values: vec![0.0; dim],
        }
    }

    /// Scatter one sparse row in, run `f` with it as a probe, then clear it again.
    pub fn with<R>(&mut self, indices: &[i64], data: &[f64], f: impl FnOnce(&Probe<'_>) -> R) -> R {
        for (&j, &value) in indices.iter().zip(data) {
            self.values[j as usize] = value;
        }
        let out = f(&Probe::Scattered(&self.values));
        for &j in indices {
            self.values[j as usize] = 0.0;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::testing::Owned;

    #[test]
    fn dense_dot_matches_the_naive_sum_over_every_length() {
        // The lane loop and its remainder meet at every length modulo four, so the
        // boundary is what the test walks rather than one convenient size.
        for dim in 1..=17 {
            let a: Vec<f64> = (0..dim).map(|i| i as f64 * 0.5 - 1.0).collect();
            let b: Vec<f64> = (0..dim).map(|i| 2.0 - i as f64 * 0.25).collect();
            let naive: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!((dot(&a, &b) - naive).abs() < 1e-12, "dim {dim}");
        }
    }

    #[test]
    fn dense_items_score_rows_the_way_a_matmul_would() {
        let vectors = [1.0, 2.0, 3.0, -1.0, 0.5, 4.0];
        let items = DenseItems {
            vectors: &vectors,
            dim: 3,
        };
        assert_eq!(items.len(), 2);
        assert!((items.similarity(0, 1) - (-1.0 + 1.0 + 12.0)).abs() < 1e-12);
        assert!((items.score(1, &Probe::Dense(&[1.0, 2.0, 3.0])) - 12.0).abs() < 1e-12);
    }

    #[test]
    fn every_probe_shape_scores_a_query_alike() {
        // The same query and the same items, expressed three ways: the variants are an
        // optimization, so they must not be a semantic choice.
        let dense = Owned::from_dense(&[vec![0.0, 2.0, 0.0, 3.0]]);
        let items = Owned::from_dense(&[
            vec![1.0, 0.0, 2.0, 0.0],
            vec![0.0, 3.0, 4.0, 5.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ]);
        let sparse_items = SparseItems {
            matrix: items.csr(),
        };
        let flat = [1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0, 0.0];
        let dense_items = DenseItems {
            vectors: &flat,
            dim: 4,
        };

        let query = [0.0, 2.0, 0.0, 3.0];
        let row = dense.csr();
        let (indices, data) = (row.indices, row.data);
        let mut scatter = Scatter::new(4);
        for item in 0..3 {
            let expected = dense_items.score(item, &Probe::Dense(&query));
            let sparse_probe = Probe::Sparse { indices, data };
            assert!((dense_items.score(item, &sparse_probe) - expected).abs() < 1e-12);
            assert!((sparse_items.score(item, &sparse_probe) - expected).abs() < 1e-12);
            let scattered = scatter.with(indices, data, |probe| sparse_items.score(item, probe));
            assert!((scattered - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn scatter_leaves_its_buffer_clean_for_the_next_query() {
        let mut scatter = Scatter::new(4);
        scatter.with(&[1, 3], &[2.0, 3.0], |_| ());
        assert_eq!(scatter.values, vec![0.0; 4]);
    }

    #[test]
    fn sparse_similarity_intersects_the_stored_columns() {
        let owned = Owned::from_dense(&[
            vec![1.0, 0.0, 2.0, 0.0],
            vec![0.0, 3.0, 4.0, 5.0],
            vec![0.0, 0.0, 0.0, 0.0],
        ]);
        let items = SparseItems {
            matrix: owned.csr(),
        };
        assert_eq!(items.dim(), 4);
        assert!((items.similarity(0, 1) - 8.0).abs() < 1e-12);
        assert_eq!(items.similarity(0, 2), 0.0);
        assert!((items.similarity(1, 1) - 50.0).abs() < 1e-12);
    }
}
