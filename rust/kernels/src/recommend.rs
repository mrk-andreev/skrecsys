//! Fused scoring and top-k selection: `recommend` without a dense score matrix.
//!
//! `recommend` used to materialize a dense `(n_queries, n_items)` score matrix, rank it
//! and throw all but `k` columns per row away. Here each query is scored and reduced to
//! its `k` best entries before the next one starts, so a score row never outlives its
//! query and never leaves the native side. Three scorers share that shape:
//!
//! * [`top_k_from_similarity`] -- a sparse item-item model (KNN, RP3beta, SLIM, BM25),
//!   accumulated with the SMMP accumulator over the entries the query reaches.
//! * [`top_k_from_dense_rows`] -- a dense item-item model (EASE), one `axpy` per item the
//!   user interacted with into a catalog-sized buffer.
//! * [`top_k_from_factors`] -- a latent-factor model (ALS, BPR, and most-popular as the
//!   zero-dimensional case), scanned a tile of queries at a time by a register-blocked
//!   micro-kernel.
//!
//! All three take the queries as *rows* of a user-side matrix the caller owns in full,
//! and their exclusions in either form in [`Exclusions`], so the Python side hands over
//! the fitted arrays as they are instead of slicing a copy per call.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use multiversion::multiversion;
use rayon::prelude::*;

use crate::ranking::{Candidate, Excluded, TooFewEligible};
use crate::sparse::{Accumulator, Csr};
use crate::vectors::{BASELINE_FMA, axpy, madd, reduce};

/// Batches up to this many queries run on the calling thread.
///
/// Waking a pool, splitting the batch and joining it costs tens of microseconds, and
/// each worker then builds its own catalog-sized scratch space. A single query's own
/// work is often a microsecond or two, so for a handful of queries all of that is pure
/// overhead -- it was most of the latency of a one-user `recommend`.
pub const SERIAL_QUERIES: usize = 8;

/// A query switches to a dense score row once its stored-weight work reaches the
/// catalog size divided by this. Measured on a Zipf-distributed catalog, where the
/// sparse accumulator's bookkeeping lost from about a quarter of the catalog on.
const DENSE_WORK_RATIO: usize = 4;

/// The items a query may be recommended, addressed by candidate position.
#[derive(Clone, Copy)]
pub enum Candidates<'a> {
    /// Every one of this many items, so position and item index coincide.
    All(usize),
    /// A subset: `items[p]` is the item at position `p`, ascending, and `position` is
    /// its inverse over the whole catalog, `-1` for an item that is not a candidate.
    Subset {
        items: &'a [usize],
        position: &'a [i64],
    },
}

impl Candidates<'_> {
    pub fn len(&self) -> usize {
        match self {
            Self::All(n) => *n,
            Self::Subset { items, .. } => items.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The item index at candidate position `p`.
    #[inline(always)]
    pub fn item(&self, p: usize) -> usize {
        match self {
            Self::All(_) => p,
            Self::Subset { items, .. } => items[p],
        }
    }

    /// The candidate position of item `j`, if it is one.
    #[inline(always)]
    pub fn position(&self, j: usize) -> Option<usize> {
        match self {
            Self::All(n) => (j < *n).then_some(j),
            Self::Subset { position, .. } => position.get(j).and_then(|&p| usize::try_from(p).ok()),
        }
    }
}

/// An index-pointer entry: the `usize` the crate's own matrices hold, or the `i64` numpy
/// hands over.
///
/// The recommend kernels read the whole fitted interaction matrix but touch only the
/// rows being asked for, so converting its index pointer up front would cost a pass
/// over every user on every request. Reading `i64` entries directly and converting each
/// on the way in keeps that cost at the rows touched, with no reinterpreting cast: a
/// negative entry becomes an out-of-range offset, which the slice bounds checks refuse.
pub trait Offset: Copy + Send + Sync {
    fn offset(self) -> usize;
}

impl Offset for usize {
    #[inline(always)]
    fn offset(self) -> usize {
        self
    }
}

impl Offset for i64 {
    #[inline(always)]
    fn offset(self) -> usize {
        usize::try_from(self).unwrap_or(usize::MAX)
    }
}

/// A CSR matrix a recommend call reads row by row: the fitted interactions on the user
/// side, or a sparse similarity on the item side.
///
/// A [`Csr`] would do, except that its index pointer is `usize`; this takes either
/// [`Offset`] type, so the extension can pass a fitted matrix without converting it.
#[derive(Clone, Copy)]
pub struct CsrRows<'a, P: Offset = usize> {
    pub n_cols: usize,
    pub indptr: &'a [P],
    pub indices: &'a [i64],
    pub data: &'a [f64],
}

impl<'a, P: Offset> CsrRows<'a, P> {
    /// The number of rows the index pointer describes.
    pub fn n_rows(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    #[inline]
    fn range(&self, row: usize) -> std::ops::Range<usize> {
        self.indptr[row].offset()..self.indptr[row + 1].offset()
    }

    /// The `(column, value)` pairs stored in row `row`.
    pub fn row(&self, row: usize) -> impl Iterator<Item = (usize, f64)> + 'a {
        let range = self.range(row);
        self.indices[range.clone()]
            .iter()
            .map(|&j| j as usize)
            .zip(self.data[range].iter().copied())
    }
}

impl<'a> From<&Csr<'a>> for CsrRows<'a, usize> {
    fn from(csr: &Csr<'a>) -> Self {
        Self {
            n_cols: csr.n_cols,
            indptr: csr.indptr,
            indices: csr.indices,
            data: csr.data,
        }
    }
}

/// What each query must not be recommended.
pub enum Exclusions<'a, P: Offset = usize> {
    /// Nothing is excluded.
    Nothing,
    /// Candidate positions per query, strictly increasing within each query.
    Positions(Excluded<'a>),
    /// Item indices per *user row*, as a CSR interaction matrix stores them: query `q`
    /// excludes the items of row `rows[q]`. This is `exclude_seen` without the caller
    /// slicing and remapping the rows first.
    Items { indptr: &'a [P], indices: &'a [i64] },
    /// `Items` and `Positions` together: query `q` skips the items of row `rows[q]` and
    /// the candidate positions of row `q` of `extra`. The fitted model's seen items plus
    /// what a caller adds on top of them, with neither copied into the other.
    ItemsAndPositions {
        indptr: &'a [P],
        indices: &'a [i64],
        extra: Excluded<'a>,
    },
}

impl<P: Offset> Exclusions<'_, P> {
    /// The candidate positions query `q`, which is user row `row`, must skip, ascending
    /// and distinct, written into `out`.
    fn collect(&self, q: usize, row: usize, candidates: &Candidates<'_>, out: &mut Vec<usize>) {
        out.clear();
        match self {
            Self::Nothing => {}
            Self::Positions(excluded) => out.extend(excluded.row(q).iter().map(|&p| p as usize)),
            Self::Items { indptr, indices } => {
                collect_items(indptr, indices, row, candidates, out);
            }
            Self::ItemsAndPositions {
                indptr,
                indices,
                extra,
            } => {
                collect_items(indptr, indices, row, candidates, out);
                out.extend(extra.row(q).iter().map(|&p| p as usize));
            }
        }
        // scipy keeps a canonical matrix sorted, but nothing forces a caller to hand one
        // over, and two sources interleave; the scans below rely on ascending positions.
        if out.windows(2).any(|w| w[0] >= w[1]) {
            out.sort_unstable();
            out.dedup();
        }
    }
}

/// Append the candidate positions of the items stored in row `row` of a CSR matrix.
fn collect_items<P: Offset>(
    indptr: &[P],
    indices: &[i64],
    row: usize,
    candidates: &Candidates<'_>,
    out: &mut Vec<usize>,
) {
    let items = &indices[indptr[row].offset()..indptr[row + 1].offset()];
    out.extend(
        items
            .iter()
            .filter_map(|&j| candidates.position(j as usize)),
    );
}

/// Fail with the first query that has fewer than `k` eligible candidates.
///
/// Checked up front and in query order, so the error names the same query however the
/// batch is later split across threads; the kernels can then assume every query fills.
fn check_eligible<P: Offset>(
    rows: &[usize],
    candidates: &Candidates<'_>,
    exclusions: &Exclusions<'_, P>,
    k: usize,
) -> Result<(), TooFewEligible> {
    let n_candidates = candidates.len();
    let mut skip = Vec::new();
    for (q, &row) in rows.iter().enumerate() {
        exclusions.collect(q, row, candidates, &mut skip);
        let found = n_candidates - skip.len().min(n_candidates);
        if found < k {
            return Err(TooFewEligible { row: q, found });
        }
    }
    Ok(())
}

/// The `k` best candidates offered so far, worst on top.
struct Best {
    heap: BinaryHeap<Candidate>,
    k: usize,
    /// The worst kept score once `k` are kept, else negative infinity.
    ///
    /// Nearly every candidate a scan offers loses to the `k` it already has, so the
    /// common case is one float compare against this rather than a heap peek and a
    /// two-key comparison. Only a score at or above it -- where the tie-break on the
    /// index can matter -- goes on to the full comparison.
    floor: f64,
}

impl Best {
    fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k + 1),
            k,
            floor: f64::NEG_INFINITY,
        }
    }

    #[inline(always)]
    fn offer(&mut self, score: f64, index: usize) {
        if score < self.floor {
            return;
        }
        let candidate = Candidate { score, index };
        if self.heap.len() < self.k {
            self.heap.push(candidate);
            if self.heap.len() == self.k {
                self.floor = self.heap.peek().map_or(f64::NEG_INFINITY, |w| w.score);
            }
            return;
        }
        let Some(mut worst) = self.heap.peek_mut() else {
            return;
        };
        if candidate.worst_first(&worst) == Ordering::Less {
            // Replacing the top in place sifts once, where a pop and a push sift twice.
            *worst = candidate;
            drop(worst);
            self.floor = self.heap.peek().map_or(f64::NEG_INFINITY, |w| w.score);
        }
    }

    fn len(&self) -> usize {
        self.heap.len()
    }

    /// Move the kept candidates into `out`, best first, and start over empty.
    fn drain_sorted(&mut self, out: &mut Vec<Candidate>) {
        out.clear();
        out.extend(self.heap.drain());
        self.floor = f64::NEG_INFINITY;
        out.sort_unstable_by(|a, b| a.worst_first(b));
    }

    /// Write the kept candidates best first, adding `offset` to every score.
    fn write(
        &mut self,
        scratch: &mut Vec<Candidate>,
        offset: f64,
        order: &mut [usize],
        scores: &mut [f64],
    ) {
        self.drain_sorted(scratch);
        for ((slot, score), candidate) in
            order.iter_mut().zip(scores.iter_mut()).zip(scratch.iter())
        {
            *slot = candidate.index;
            *score = candidate.score + offset;
        }
    }
}

/// Run `work` over the batch `per` queries at a time, in parallel unless it is small.
///
/// `work` gets its per-worker state, the first query of its chunk and the chunk's slice
/// of both outputs, `per * k` values each (fewer for the last chunk).
fn run_chunks<S: Send>(
    n_queries: usize,
    per: usize,
    k: usize,
    order: &mut [usize],
    scores: &mut [f64],
    init: impl Fn() -> S + Sync + Send,
    work: impl Fn(&mut S, usize, &mut [usize], &mut [f64]) + Sync + Send,
) {
    let width = per * k;
    if n_queries <= SERIAL_QUERIES.max(per) || rayon::current_num_threads() == 1 {
        let mut state = init();
        for (chunk, (o, s)) in order
            .chunks_mut(width)
            .zip(scores.chunks_mut(width))
            .enumerate()
        {
            work(&mut state, chunk * per, o, s);
        }
        return;
    }
    order
        .par_chunks_mut(width)
        .zip(scores.par_chunks_mut(width))
        .enumerate()
        .for_each_init(init, |state, (chunk, (o, s))| {
            work(state, chunk * per, o, s)
        });
}

/// For each query, the `k` best candidates of `users[rows[q]] * similarity`, best first,
/// with ties going to the lower candidate position.
///
/// `users` is the whole user-side matrix; row `rows[q]` is query `q`. The similarity has
/// one row per item the user side can hold, spreading that item's weight over the items
/// it contributes to.
///
/// Returns the candidate positions and their scores, both `(rows.len(), k)` row-major.
pub fn top_k_from_similarity<P: Offset, S: Offset, Q: Offset>(
    users: &CsrRows<'_, P>,
    rows: &[usize],
    similarity: &CsrRows<'_, S>,
    candidates: Candidates<'_>,
    exclusions: &Exclusions<'_, Q>,
    k: usize,
) -> Result<(Vec<usize>, Vec<f64>), TooFewEligible> {
    check_eligible(rows, &candidates, exclusions, k)?;
    let n_candidates = candidates.len();
    let mut order = vec![0usize; rows.len() * k];
    let mut scores = vec![0.0f64; rows.len() * k];
    if k == 0 {
        return Ok((order, scores));
    }

    struct State {
        acc: Accumulator,
        dense: Vec<f64>,
        dropped: Vec<bool>,
        skip: Vec<usize>,
        best: Best,
        scored: Vec<Candidate>,
        zeros: Vec<Candidate>,
    }
    run_chunks(
        rows.len(),
        1,
        k,
        &mut order,
        &mut scores,
        || State {
            acc: Accumulator::new(similarity.n_cols),
            dense: Vec::new(),
            dropped: vec![false; n_candidates],
            skip: Vec::new(),
            best: Best::new(k),
            scored: Vec::with_capacity(k),
            zeros: Vec::with_capacity(k),
        },
        |st, q, out_order, out_scores| {
            let row = rows[q];
            exclusions.collect(q, row, &candidates, &mut st.skip);

            // A query whose items have long weight rows -- popular items, under a skewed
            // catalog -- touches most of the catalog anyway, and then the accumulator's
            // per-add bookkeeping costs more than a plain dense row: unconditional adds,
            // then one ordered pass over the candidates. That pass ranks an untouched
            // candidate at exactly zero with ties to the lower position, which is what
            // the sparse path's zero fallback reproduces, so the result is the same.
            let work: usize = users
                .row(row)
                .map(|(item, _)| similarity.range(item).len())
                .sum();
            if work >= similarity.n_cols / DENSE_WORK_RATIO {
                st.dense.resize(similarity.n_cols, 0.0);
                for (item, weight) in users.row(row) {
                    for (j, w) in similarity.row(item) {
                        st.dense[j] += weight * w;
                    }
                }
                let mut cursor = 0usize;
                for p in 0..n_candidates {
                    if cursor < st.skip.len() && st.skip[cursor] == p {
                        cursor += 1;
                        continue;
                    }
                    st.best.offer(st.dense[candidates.item(p)], p);
                }
                st.best.write(&mut st.scored, 0.0, out_order, out_scores);
                st.dense.fill(0.0);
                return;
            }

            for &p in &st.skip {
                st.dropped[p] = true;
            }
            for (item, weight) in users.row(row) {
                for (j, w) in similarity.row(item) {
                    st.acc.add(j, weight * w);
                }
            }
            for &j in st.acc.touched() {
                if let Some(p) = candidates.position(j)
                    && !st.dropped[p]
                {
                    st.best.offer(st.acc.get(j), p);
                }
            }
            let reached = st.best.len();
            st.best.drain_sorted(&mut st.scored);

            // A candidate the product never reached scores exactly zero, which the dense
            // path ranked alongside the rest. That can only matter when fewer than `k`
            // were reached, or when the worst score kept is not positive.
            st.zeros.clear();
            if reached < k || st.scored[reached - 1].score <= 0.0 {
                for p in 0..n_candidates {
                    if st.zeros.len() == k {
                        break;
                    }
                    if !st.dropped[p] && !st.acc.is_touched(candidates.item(p)) {
                        st.zeros.push(Candidate {
                            score: 0.0,
                            index: p,
                        });
                    }
                }
            }

            st.acc.reset();
            for &p in &st.skip {
                st.dropped[p] = false;
            }
            merge(out_order, out_scores, &st.scored, &st.zeros);
        },
    );
    Ok((order, scores))
}

/// For each query, the `k` best candidates of `users[rows[q]] * weights`, where
/// `weights` is a dense row-major `(n_items, n_items)` matrix: EASE's `B`.
///
/// Row `i` of `weights` is what an interaction with item `i` adds to every item's score,
/// so a query is one `axpy` per item it holds into a catalog-sized buffer. That reads
/// each touched row once, contiguously, which is the whole cost; the sparse-times-dense
/// product scipy ran here instead is single-threaded and went through a dense result
/// matrix that was only ranked and thrown away.
pub fn top_k_from_dense_rows<P: Offset, Q: Offset>(
    users: &CsrRows<'_, P>,
    rows: &[usize],
    weights: &[f64],
    candidates: Candidates<'_>,
    exclusions: &Exclusions<'_, Q>,
    k: usize,
) -> Result<(Vec<usize>, Vec<f64>), TooFewEligible> {
    check_eligible(rows, &candidates, exclusions, k)?;
    let n_items = users.n_cols;
    let n_candidates = candidates.len();
    let mut order = vec![0usize; rows.len() * k];
    let mut scores = vec![0.0f64; rows.len() * k];
    if k == 0 {
        return Ok((order, scores));
    }

    run_chunks(
        rows.len(),
        1,
        k,
        &mut order,
        &mut scores,
        || {
            (
                vec![0.0f64; n_items],
                Vec::new(),
                Best::new(k),
                Vec::with_capacity(k),
            )
        },
        |(buffer, skip, best, scratch), q, out_order, out_scores| {
            let row = rows[q];
            exclusions.collect(q, row, &candidates, skip);
            buffer.fill(0.0);
            for (item, weight) in users.row(row) {
                axpy(
                    buffer,
                    &weights[item * n_items..(item + 1) * n_items],
                    weight,
                );
            }
            let mut cursor = 0usize;
            for p in 0..n_candidates {
                if cursor < skip.len() && skip[cursor] == p {
                    cursor += 1;
                    continue;
                }
                best.offer(buffer[candidates.item(p)], p);
            }
            best.write(scratch, 0.0, out_order, out_scores);
        },
    );
    Ok((order, scores))
}

/// A latent-factor model: `score(u, j) = item_bias[j] + <users[u], items[j]> +
/// user_offset[u]`.
///
/// `users` and `items` are row-major with `dim` values per row. The user offset cannot
/// change a ranking and is only added to the reported scores. A zero `dim` with a bias
/// is a popularity ranking, which is how most-popular runs through here too.
pub struct Factors<'a> {
    pub users: &'a [f64],
    pub items: &'a [f64],
    pub dim: usize,
    pub item_bias: Option<&'a [f64]>,
    pub user_offset: Option<&'a [f64]>,
}

/// Queries scored together by the tiled micro-kernel.
///
/// Eight `f64` lanes are four NEON or two AVX2 registers per item, and the kernel
/// interleaves [`ITEM_BLOCK`] items so that the chains in flight -- eight registers --
/// cover the FMA latency on both.
const QUERY_TILE: usize = 8;
/// Items scored together by the tiled micro-kernel.
const ITEM_BLOCK: usize = 2;
/// Items scored together when a single query is scanned on its own.
const SINGLE_BLOCK: usize = 4;
/// Factor values per item held in flight when a single query is scanned on its own.
const SINGLE_LANES: usize = 8;
/// A partial tile at least this full is scanned padded rather than query by query.
const MIN_TILED: usize = 4;

/// For each query, the `k` best candidates of a latent-factor model, best first, with
/// ties going to the lower candidate position.
///
/// The batch is scanned a tile of [`QUERY_TILE`] queries at a time, which is the loop
/// order a matmul micro-kernel uses: each item vector is loaded once per tile and
/// multiplied into `QUERY_TILE` accumulators held in registers, instead of once per
/// query. The dense `(n_queries, n_items)` product -- and the three bias-broadcast
/// temporaries of the same size numpy built around it -- is never formed.
pub fn top_k_from_factors<Q: Offset>(
    factors: &Factors<'_>,
    rows: &[usize],
    candidates: Candidates<'_>,
    exclusions: &Exclusions<'_, Q>,
    k: usize,
) -> Result<(Vec<usize>, Vec<f64>), TooFewEligible> {
    check_eligible(rows, &candidates, exclusions, k)?;
    let n_queries = rows.len();
    let dim = factors.dim;
    let mut order = vec![0usize; n_queries * k];
    let mut scores = vec![0.0f64; n_queries * k];
    if k == 0 {
        return Ok((order, scores));
    }

    if factors_split_items(n_queries, candidates.len(), dim) && rayon::current_num_threads() > 1 {
        rank_split_items(
            factors,
            rows,
            candidates,
            exclusions,
            k,
            &mut order,
            &mut scores,
        );
        return Ok((order, scores));
    }

    struct Tile {
        /// Query `t`'s value for dimension `d` at `d * QUERY_TILE + t`.
        transposed: Vec<f64>,
        skips: Vec<Vec<usize>>,
        bests: Vec<Best>,
        scratch: Vec<Candidate>,
    }
    run_chunks(
        n_queries,
        QUERY_TILE,
        k,
        &mut order,
        &mut scores,
        || Tile {
            transposed: vec![0.0; dim * QUERY_TILE],
            skips: (0..QUERY_TILE).map(|_| Vec::new()).collect(),
            bests: (0..QUERY_TILE).map(|_| Best::new(k)).collect(),
            scratch: Vec::with_capacity(k),
        },
        |tile, first, out_order, out_scores| {
            let size = QUERY_TILE.min(n_queries - first);
            for t in 0..size {
                exclusions.collect(first + t, rows[first + t], &candidates, &mut tile.skips[t]);
            }
            if size >= MIN_TILED {
                tile.transposed.fill(0.0);
                for t in 0..size {
                    let user = rows[first + t];
                    let query = &factors.users[user * dim..(user + 1) * dim];
                    for (d, &value) in query.iter().enumerate() {
                        tile.transposed[d * QUERY_TILE + t] = value;
                    }
                }
                scan_tiled(
                    factors,
                    &candidates,
                    &tile.transposed,
                    &tile.skips,
                    &mut tile.bests[..size],
                );
            } else {
                for t in 0..size {
                    let user = rows[first + t];
                    let query = &factors.users[user * dim..(user + 1) * dim];
                    scan_single(
                        factors,
                        &candidates,
                        query,
                        &tile.skips[t],
                        &mut tile.bests[t],
                        0..candidates.len(),
                    );
                }
            }
            for t in 0..size {
                let offset = factors.user_offset.map_or(0.0, |o| o[rows[first + t]]);
                tile.bests[t].write(
                    &mut tile.scratch,
                    offset,
                    &mut out_order[t * k..(t + 1) * k],
                    &mut out_scores[t * k..(t + 1) * k],
                );
            }
        },
    );
    Ok((order, scores))
}

/// Scanned-value count (candidates times factors) from which a small batch splits the
/// catalog across threads instead of scanning it on one.
///
/// Around a millisecond of single-threaded scan; below it the fork and join cost more
/// than they save, above it a one-user request on a large catalog would otherwise run
/// on a single core however many the pool has.
const ITEM_SPLIT_WORK: usize = 1 << 21;
/// Candidates per task when a query's catalog is split across threads.
const ITEM_SPLIT_CHUNK: usize = 1 << 14;

/// Whether a latent-factor batch this shape splits its catalog across threads.
fn factors_split_items(n_queries: usize, n_candidates: usize, dim: usize) -> bool {
    n_queries <= SERIAL_QUERIES && n_candidates.saturating_mul(dim.max(1)) >= ITEM_SPLIT_WORK
}

/// Whether [`top_k_from_factors`] can use a thread pool for a batch this shape, so a
/// caller can skip waking one when it cannot.
pub fn factors_want_threads(n_queries: usize, n_candidates: usize, dim: usize) -> bool {
    n_queries > SERIAL_QUERIES || factors_split_items(n_queries, n_candidates, dim)
}

/// Rank a small batch against a large catalog by splitting the catalog, not the batch.
///
/// Each query's candidates are cut into ranges scanned in parallel, each keeping its own
/// `k` best, and the per-range winners are merged. Ranking is by score then position
/// throughout, so the merge picks exactly what one scan over everything would have.
fn rank_split_items<Q: Offset>(
    factors: &Factors<'_>,
    rows: &[usize],
    candidates: Candidates<'_>,
    exclusions: &Exclusions<'_, Q>,
    k: usize,
    order: &mut [usize],
    scores: &mut [f64],
) {
    let dim = factors.dim;
    let n_candidates = candidates.len();
    let mut skip = Vec::new();
    for (q, &user) in rows.iter().enumerate() {
        exclusions.collect(q, user, &candidates, &mut skip);
        let query = &factors.users[user * dim..(user + 1) * dim];
        let mut merged: Vec<Candidate> = (0..n_candidates)
            .step_by(ITEM_SPLIT_CHUNK)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|start| {
                let mut best = Best::new(k);
                let range = start..(start + ITEM_SPLIT_CHUNK).min(n_candidates);
                scan_single(factors, &candidates, query, &skip, &mut best, range);
                best.heap.into_vec()
            })
            .flatten()
            .collect();
        merged.sort_unstable_by(|a, b| a.worst_first(b));
        let offset = factors.user_offset.map_or(0.0, |o| o[user]);
        for ((slot, score), candidate) in order[q * k..(q + 1) * k]
            .iter_mut()
            .zip(&mut scores[q * k..(q + 1) * k])
            .zip(&merged)
        {
            *slot = candidate.index;
            *score = candidate.score + offset;
        }
    }
}

/// Scores of `I` items against `T` queries, `acc[i][t]`, starting from `start[i]`.
///
/// `transposed` holds the queries as `transposed[d * T + t]`, so for each dimension the
/// innermost loop is `T` contiguous values times one broadcast item value: vertical
/// FMAs with no horizontal reduction anywhere. The `I` items are interleaved so the
/// dependent chains of one hide the latency of the others.
#[inline(always)]
fn micro_kernel<const FMA: bool, const T: usize, const I: usize>(
    transposed: &[f64],
    items: [&[f64]; I],
    start: [f64; I],
) -> [[f64; T]; I] {
    let mut acc = [[0.0f64; T]; I];
    for i in 0..I {
        acc[i] = [start[i]; T];
    }
    for (d, lane) in transposed.as_chunks::<T>().0.iter().enumerate() {
        for i in 0..I {
            let value = items[i][d];
            for t in 0..T {
                acc[i][t] = madd::<FMA>(lane[t], value, acc[i][t]);
            }
        }
    }
    acc
}

/// Items `p..p + N` as vectors and starting scores, the block padded by repeating the
/// last candidate so the micro-kernel never branches on a short block.
#[inline(always)]
fn item_block<'a, const N: usize>(
    factors: &Factors<'a>,
    candidates: &Candidates<'_>,
    p: usize,
    n_candidates: usize,
) -> ([&'a [f64]; N], [f64; N]) {
    let dim = factors.dim;
    let js: [usize; N] = core::array::from_fn(|i| candidates.item((p + i).min(n_candidates - 1)));
    (
        js.map(|j| &factors.items[j * dim..(j + 1) * dim]),
        js.map(|j| factors.item_bias.map_or(0.0, |b| b[j])),
    )
}

/// Offer candidate `p`'s score to one query, unless that query excludes it.
#[inline(always)]
fn offer_unless_skipped(best: &mut Best, skip: &[usize], cursor: &mut usize, p: usize, score: f64) {
    // The excluded positions ascend and so does `p`, so one cursor keeps up.
    if *cursor < skip.len() && skip[*cursor] == p {
        *cursor += 1;
        return;
    }
    best.offer(score, p);
}

/// Scan every candidate against a full (or zero-padded) tile of queries.
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse2"))]
fn scan_tiled(
    factors: &Factors<'_>,
    candidates: &Candidates<'_>,
    transposed: &[f64],
    skips: &[Vec<usize>],
    bests: &mut [Best],
) {
    const FMA: bool = BASELINE_FMA || multiversion::target::target_cfg_f!(target_feature = "fma");
    let n_candidates = candidates.len();
    let mut cursors = [0usize; QUERY_TILE];
    for p in (0..n_candidates).step_by(ITEM_BLOCK) {
        let (items, start) = item_block::<ITEM_BLOCK>(factors, candidates, p, n_candidates);
        let acc = micro_kernel::<FMA, QUERY_TILE, ITEM_BLOCK>(transposed, items, start);
        for (i, scores) in acc.iter().enumerate().take(n_candidates - p) {
            for (t, best) in bests.iter_mut().enumerate() {
                offer_unless_skipped(best, &skips[t], &mut cursors[t], p + i, scores[t]);
            }
        }
    }
}

/// Scores of `I` items against one query, `start[i] + <query, items[i]>`.
///
/// With one query there is nothing to broadcast across, so the vectors run along the
/// factor dimension instead: [`SINGLE_LANES`] values of each of the `I` items at a
/// time, vertical FMAs into `I * SINGLE_LANES` accumulators, and one horizontal sum
/// per item at the end. Interleaving the items is what keeps enough chains in flight.
#[inline(always)]
fn single_kernel<const FMA: bool, const I: usize>(
    query: &[f64],
    items: [&[f64]; I],
    start: [f64; I],
) -> [f64; I] {
    let mut acc = [[0.0f64; SINGLE_LANES]; I];
    let (query_chunks, query_rest) = query.as_chunks::<SINGLE_LANES>();
    let rows = items.map(|row| row.as_chunks::<SINGLE_LANES>());
    for (c, q) in query_chunks.iter().enumerate() {
        for i in 0..I {
            let v = &rows[i].0[c];
            for l in 0..SINGLE_LANES {
                acc[i][l] = madd::<FMA>(q[l], v[l], acc[i][l]);
            }
        }
    }
    let mut out = start;
    for i in 0..I {
        let mut total = reduce(&acc[i]);
        for (q, v) in query_rest.iter().zip(rows[i].1) {
            total = madd::<FMA>(*q, *v, total);
        }
        out[i] += total;
    }
    out
}

/// Scan every candidate against one query, several items at a time.
///
/// A partial tile would waste most of its lanes on padding, so a batch of one -- the
/// latency case -- takes this path, vectorized along the factors by [`single_kernel`].
#[multiversion(targets("x86_64+avx2+fma", "x86_64+avx", "x86_64+sse2"))]
fn scan_single(
    factors: &Factors<'_>,
    candidates: &Candidates<'_>,
    query: &[f64],
    skip: &[usize],
    best: &mut Best,
    range: std::ops::Range<usize>,
) {
    const FMA: bool = BASELINE_FMA || multiversion::target::target_cfg_f!(target_feature = "fma");
    let end = range.end;
    let mut cursor = skip.partition_point(|&p| p < range.start);
    for p in range.step_by(SINGLE_BLOCK) {
        let (items, start) = item_block::<SINGLE_BLOCK>(factors, candidates, p, end);
        let scores = single_kernel::<FMA, SINGLE_BLOCK>(query, items, start);
        for (i, &score) in scores.iter().enumerate().take(end - p) {
            offer_unless_skipped(best, skip, &mut cursor, p + i, score);
        }
    }
}

/// Fill one row of the output from the two ranked runs, best first.
fn merge(order: &mut [usize], scores: &mut [f64], scored: &[Candidate], zeros: &[Candidate]) {
    let (mut next_scored, mut next_zero) = (0usize, 0usize);
    for slot in 0..order.len() {
        let take_scored = match (scored.get(next_scored), zeros.get(next_zero)) {
            (Some(a), Some(b)) => a.worst_first(b) == Ordering::Less,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => return,
        };
        let pick = if take_scored {
            next_scored += 1;
            scored[next_scored - 1]
        } else {
            next_zero += 1;
            zeros[next_zero - 1]
        };
        order[slot] = pick.index;
        scores[slot] = pick.score;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::testing::{Owned, pseudo_random_dense};

    struct Case {
        users: Owned,
        rows: Vec<usize>,
        similarity: Owned,
        candidates: Vec<usize>,
        position: Vec<i64>,
        indptr: Vec<usize>,
        indices: Vec<i64>,
    }

    impl Case {
        fn new(users: &[Vec<f64>], similarity: &[Vec<f64>], candidates: Vec<usize>) -> Self {
            let n_items = similarity[0].len();
            let mut position = vec![-1i64; n_items];
            for (p, &j) in candidates.iter().enumerate() {
                position[j] = p as i64;
            }
            Self {
                users: Owned::from_dense(users),
                rows: (0..users.len()).collect(),
                similarity: Owned::from_dense(similarity),
                candidates,
                position,
                indptr: vec![0; users.len() + 1],
                indices: Vec::new(),
            }
        }

        fn exclude(mut self, rows: &[Vec<usize>]) -> Self {
            self.indptr = vec![0];
            self.indices = Vec::new();
            for row in rows {
                self.indices.extend(row.iter().map(|&c| c as i64));
                self.indptr.push(self.indices.len());
            }
            self
        }

        fn candidates(&self) -> Candidates<'_> {
            Candidates::Subset {
                items: &self.candidates,
                position: &self.position,
            }
        }

        fn exclusions(&self) -> Exclusions<'_> {
            Exclusions::Positions(Excluded {
                indptr: &self.indptr,
                indices: &self.indices,
            })
        }

        fn try_run(&self, k: usize) -> Result<(Vec<usize>, Vec<f64>), TooFewEligible> {
            top_k_from_similarity(
                &CsrRows::from(&self.users.csr()),
                &self.rows,
                &CsrRows::from(&self.similarity.csr()),
                self.candidates(),
                &self.exclusions(),
                k,
            )
        }

        fn run(&self, k: usize) -> (Vec<usize>, Vec<f64>) {
            self.try_run(k)
                .unwrap_or_else(|_| panic!("too few eligible"))
        }
    }

    /// The dense product this kernel replaces, restricted to the candidates.
    fn dense_scores(
        users: &[Vec<f64>],
        similarity: &[Vec<f64>],
        candidates: &[usize],
    ) -> Vec<Vec<f64>> {
        users
            .iter()
            .map(|u| {
                candidates
                    .iter()
                    .map(|&j| (0..u.len()).map(|i| u[i] * similarity[i][j]).sum())
                    .collect()
            })
            .collect()
    }

    fn rank(scores: &[f64], excluded: &[usize], k: usize) -> (Vec<usize>, Vec<f64>) {
        let mut all: Vec<Candidate> = scores
            .iter()
            .enumerate()
            .filter(|(p, _)| !excluded.contains(p))
            .map(|(index, &score)| Candidate { score, index })
            .collect();
        all.sort_by(|a, b| a.worst_first(b));
        all.truncate(k);
        (
            all.iter().map(|c| c.index).collect(),
            all.iter().map(|c| c.score).collect(),
        )
    }

    /// Every row of `got` against the brute-force ranking of `dense`, scores to `tol`.
    fn assert_ranked(
        got: &(Vec<usize>, Vec<f64>),
        dense: &[Vec<f64>],
        excluded: impl Fn(usize) -> Vec<usize>,
        k: usize,
        tol: f64,
    ) {
        let (order, scores) = got;
        for (row, row_scores) in dense.iter().enumerate() {
            let (want_order, want_scores) = rank(row_scores, &excluded(row), k);
            assert_eq!(&order[row * k..(row + 1) * k], &want_order[..], "row {row}");
            for (got, want) in scores[row * k..(row + 1) * k].iter().zip(&want_scores) {
                assert!((got - want).abs() < tol, "row {row}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn matches_the_dense_product_and_ranking() {
        let users = pseudo_random_dense(40, 25);
        let similarity = pseudo_random_dense(25, 25);
        let candidates: Vec<usize> = (0..25).collect();
        let case = Case::new(&users, &similarity, candidates.clone());
        let k = 6;
        let dense = dense_scores(&users, &similarity, &candidates);
        assert_ranked(&case.run(k), &dense, |_| Vec::new(), k, 1e-12);
    }

    #[test]
    fn falls_back_to_untouched_candidates() {
        // User 0 reaches item 1 only, so the other candidates score zero and rank by
        // position; a negative weight must rank below those zeros.
        let users = vec![vec![1.0, 0.0, 0.0]];
        let similarity = vec![vec![0.0, -2.0, 0.0], vec![0.0; 3], vec![0.0; 3]];
        let case = Case::new(&users, &similarity, vec![0, 1, 2]);
        let (order, scores) = case.run(3);
        assert_eq!(order, vec![0, 2, 1]);
        assert_eq!(scores, vec![0.0, 0.0, -2.0]);
    }

    #[test]
    fn honours_exclusions_and_candidate_subsets() {
        let users = vec![vec![1.0, 1.0, 0.0, 0.0]];
        let similarity = vec![
            vec![0.0, 1.0, 5.0, 2.0],
            vec![0.0, 0.0, 3.0, 9.0],
            vec![0.0; 4],
            vec![0.0; 4],
        ];
        // Candidates are items 2 and 3, whose positions are 0 and 1; exclude item 3.
        let case = Case::new(&users, &similarity, vec![2, 3]).exclude(&[vec![1]]);
        let (order, scores) = case.run(1);
        assert_eq!((order, scores), (vec![0], vec![8.0]));
    }

    #[test]
    fn reports_a_query_with_too_few_candidates() {
        let users = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let similarity = vec![vec![0.0, 1.0], vec![0.0, 0.0]];
        let case = Case::new(&users, &similarity, vec![0, 1]).exclude(&[vec![], vec![0]]);
        let error = case.try_run(2).expect_err("should fail");
        assert_eq!((error.row, error.found), (1, 1));
    }

    #[test]
    fn the_first_short_query_is_reported_whatever_the_threads() {
        let users = pseudo_random_dense(200, 3);
        let similarity = pseudo_random_dense(3, 3);
        let mut excluded = vec![Vec::new(); 200];
        excluded[57] = vec![0, 2];
        excluded[180] = vec![1];
        let case = Case::new(&users, &similarity, vec![0, 1, 2]).exclude(&excluded);
        let error = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("pool")
            .install(|| case.try_run(2))
            .expect_err("should fail");
        assert_eq!((error.row, error.found), (57, 1));
    }

    /// The same batch through a picked subset of user rows, with the exclusions read
    /// straight out of the interaction matrix rather than handed over as positions.
    #[test]
    fn rows_and_seen_items_match_a_sliced_batch() {
        let users = pseudo_random_dense(30, 20);
        let similarity = pseudo_random_dense(20, 20);
        let full = Owned::from_dense(&users);
        let full = full.csr();
        let rows = vec![7usize, 3, 29, 3, 0];
        let candidates: Vec<usize> = (0..20).step_by(2).collect();
        let mut position = vec![-1i64; 20];
        for (p, &j) in candidates.iter().enumerate() {
            position[j] = p as i64;
        }
        let k = 3;
        let got = top_k_from_similarity(
            &CsrRows::from(&full),
            &rows,
            &CsrRows::from(&Owned::from_dense(&similarity).csr()),
            Candidates::Subset {
                items: &candidates,
                position: &position,
            },
            &Exclusions::Items {
                indptr: full.indptr,
                indices: full.indices,
            },
            k,
        )
        .expect("enough");
        let picked: Vec<Vec<f64>> = rows.iter().map(|&r| users[r].clone()).collect();
        let dense = dense_scores(&picked, &similarity, &candidates);
        let seen = |q: usize| -> Vec<usize> {
            candidates
                .iter()
                .enumerate()
                .filter(|&(_, &j)| picked[q][j] != 0.0)
                .map(|(p, _)| p)
                .collect()
        };
        assert_ranked(&got, &dense, seen, k, 1e-12);
    }

    /// Seen items and caller-supplied positions exclude together, overlaps and all,
    /// exactly as one merged list of positions would.
    #[test]
    fn seen_items_and_extra_positions_exclude_together() {
        let users = pseudo_random_dense(12, 16);
        let similarity = pseudo_random_dense(16, 16);
        let full = Owned::from_dense(&users);
        let full = full.csr();
        let rows = vec![4usize, 0, 11, 4];
        let candidates: Vec<usize> = (0..16).collect();
        let position: Vec<i64> = (0..16).collect();
        // Per query: positions the caller adds, some of them already seen.
        let added: Vec<Vec<usize>> = vec![vec![1, 5], vec![], vec![0, 15], vec![5]];
        let mut extra_indptr = vec![0usize];
        let mut extra_indices = Vec::new();
        for row in &added {
            extra_indices.extend(row.iter().map(|&p| p as i64));
            extra_indptr.push(extra_indices.len());
        }
        let k = 2;
        let got = top_k_from_similarity(
            &CsrRows::from(&full),
            &rows,
            &CsrRows::from(&Owned::from_dense(&similarity).csr()),
            Candidates::Subset {
                items: &candidates,
                position: &position,
            },
            &Exclusions::ItemsAndPositions {
                indptr: full.indptr,
                indices: full.indices,
                extra: Excluded {
                    indptr: &extra_indptr,
                    indices: &extra_indices,
                },
            },
            k,
        )
        .expect("enough");
        let picked: Vec<Vec<f64>> = rows.iter().map(|&r| users[r].clone()).collect();
        let dense = dense_scores(&picked, &similarity, &candidates);
        let excluded = |q: usize| -> Vec<usize> {
            let mut all: Vec<usize> = (0..16).filter(|&j| picked[q][j] != 0.0).collect();
            all.extend(&added[q]);
            all.sort_unstable();
            all.dedup();
            all
        };
        assert_ranked(&got, &dense, excluded, k, 1e-12);
    }

    /// Queries that touch a sliver of a large catalog stay on the sparse accumulator,
    /// whose zero fallback has to rank the untouched candidates -- including above a
    /// negative score -- exactly as the dense product would.
    #[test]
    fn a_sparse_query_ranks_like_the_dense_product() {
        let n = 400;
        let users: Vec<Vec<f64>> = (0..30)
            .map(|u| {
                (0..n)
                    .map(|j| {
                        if j == u * 7 % n || j == u * 13 % n {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();
        let similarity: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| match (i + j) % 97 {
                        0 => 2.0 + (j % 5) as f64,
                        1 => -1.5,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect();
        let candidates: Vec<usize> = (0..n).step_by(3).collect();
        let case = Case::new(&users, &similarity, candidates.clone());
        let k = 12;
        let dense = dense_scores(&users, &similarity, &candidates);
        assert_ranked(&case.run(k), &dense, |_| Vec::new(), k, 1e-12);
    }

    #[test]
    fn thread_count_does_not_change_the_result() {
        let users = pseudo_random_dense(120, 40);
        let similarity = pseudo_random_dense(40, 40);
        let case = Case::new(&users, &similarity, (0..40).collect());
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| case.run(9))
        };
        assert_eq!(run(1), run(4));
    }

    #[test]
    fn dense_rows_match_the_dense_product() {
        let users = pseudo_random_dense(50, 30);
        // Negative weights too: EASE's `B` has plenty, and zeros must rank among them.
        let weights: Vec<Vec<f64>> = pseudo_random_dense(30, 30)
            .into_iter()
            .enumerate()
            .map(|(i, row)| {
                row.into_iter()
                    .map(|v| if i % 2 == 0 { v } else { -v })
                    .collect()
            })
            .collect();
        let flat: Vec<f64> = weights.concat();
        let full = Owned::from_dense(&users);
        let full = full.csr();
        let rows: Vec<usize> = (0..50).rev().collect();
        let k = 7;
        for threads in [1, 4] {
            let got = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| {
                    top_k_from_dense_rows(
                        &CsrRows::from(&full),
                        &rows,
                        &flat,
                        Candidates::All(30),
                        &Exclusions::Items {
                            indptr: full.indptr,
                            indices: full.indices,
                        },
                        k,
                    )
                })
                .expect("enough");
            let picked: Vec<Vec<f64>> = rows.iter().map(|&r| users[r].clone()).collect();
            let dense = dense_scores(&picked, &weights, &(0..30).collect::<Vec<_>>());
            let seen =
                |q: usize| -> Vec<usize> { (0..30).filter(|&j| picked[q][j] != 0.0).collect() };
            assert_ranked(&got, &dense, seen, k, 1e-9);
        }
    }

    fn factor_scores(
        users: &[f64],
        items: &[f64],
        dim: usize,
        bias: &[f64],
        offset: &[f64],
        rows: &[usize],
        candidates: &[usize],
    ) -> Vec<Vec<f64>> {
        rows.iter()
            .map(|&u| {
                candidates
                    .iter()
                    .map(|&j| {
                        let dot: f64 = (0..dim)
                            .map(|d| users[u * dim + d] * items[j * dim + d])
                            .sum();
                        bias[j] + dot + offset[u]
                    })
                    .collect()
            })
            .collect()
    }

    /// Every batch size the tiling distinguishes -- one query, a partial tile scanned
    /// query by query, a padded partial tile, full tiles plus a remainder -- and every
    /// dimension the micro-kernel's blocks can leave a remainder of.
    #[test]
    fn factors_match_the_dense_product_for_every_tiling() {
        let n_users = 45;
        let n_items = 37;
        for dim in [0usize, 1, 3, 8, 13] {
            let users: Vec<f64> = pseudo_random_dense(n_users, dim.max(1)).concat()
                [..n_users * dim]
                .iter()
                .enumerate()
                .map(|(i, v)| if i % 3 == 0 { -v } else { *v })
                .collect();
            let items: Vec<f64> =
                pseudo_random_dense(n_items, dim.max(1)).concat()[..n_items * dim].to_vec();
            let bias: Vec<f64> = (0..n_items).map(|j| (j % 5) as f64 * 0.25).collect();
            let offset: Vec<f64> = (0..n_users).map(|u| u as f64 * 0.5).collect();
            let factors = Factors {
                users: &users,
                items: &items,
                dim,
                item_bias: Some(&bias),
                user_offset: Some(&offset),
            };
            let candidates: Vec<usize> = (0..n_items).filter(|j| j % 4 != 1).collect();
            let mut position = vec![-1i64; n_items];
            for (p, &j) in candidates.iter().enumerate() {
                position[j] = p as i64;
            }
            // A seen-matrix with a few items per user, some outside the candidates.
            let seen_rows: Vec<Vec<(usize, f64)>> = (0..n_users)
                .map(|u| {
                    (0..n_items)
                        .filter(|j| (u + j) % 9 == 0)
                        .map(|j| (j, 1.0))
                        .collect()
                })
                .collect();
            let seen = Owned::from_rows(seen_rows.clone(), n_items);
            let seen = seen.csr();
            for n_queries in [1usize, 3, 5, 8, 21, 45] {
                let rows: Vec<usize> = (0..n_queries).map(|q| (q * 7) % n_users).collect();
                let k = 5;
                let got = top_k_from_factors(
                    &factors,
                    &rows,
                    Candidates::Subset {
                        items: &candidates,
                        position: &position,
                    },
                    &Exclusions::Items {
                        indptr: seen.indptr,
                        indices: seen.indices,
                    },
                    k,
                )
                .expect("enough");
                let dense = factor_scores(&users, &items, dim, &bias, &offset, &rows, &candidates);
                let excluded = |q: usize| -> Vec<usize> {
                    seen_rows[rows[q]]
                        .iter()
                        .filter_map(|&(j, _)| usize::try_from(position[j]).ok())
                        .collect()
                };
                assert_ranked(&got, &dense, excluded, k, 1e-9);
            }
        }
    }

    /// A single query against a catalog large enough to be split across threads must
    /// rank exactly as the unsplit scan does, exclusions and ties included.
    #[test]
    fn a_split_catalog_ranks_like_one_scan() {
        let dim = 4;
        let n_items = ITEM_SPLIT_WORK / dim + 3 * ITEM_SPLIT_CHUNK / 2;
        // Coarse values, so plenty of exact ties straddle the range boundaries.
        let items: Vec<f64> = (0..n_items * dim)
            .map(|i| ((i * 7919) % 13) as f64)
            .collect();
        let users = vec![1.0, 0.5, 0.25, 2.0, -1.0, 0.0, 1.0, 1.0];
        let bias: Vec<f64> = (0..n_items).map(|j| (j % 3) as f64).collect();
        let factors = Factors {
            users: &users,
            items: &items,
            dim,
            item_bias: Some(&bias),
            user_offset: None,
        };
        let seen_rows: Vec<Vec<(usize, f64)>> = vec![
            (0..n_items)
                .step_by(ITEM_SPLIT_CHUNK / 3)
                .map(|j| (j, 1.0))
                .collect(),
            vec![(5, 1.0)],
        ];
        let seen = Owned::from_rows(seen_rows, n_items);
        let seen = seen.csr();
        let exclusions = Exclusions::Items {
            indptr: seen.indptr,
            indices: seen.indices,
        };
        let rows = [0usize, 1];
        let k = 25;
        assert!(factors_split_items(rows.len(), n_items, dim));
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| {
                    top_k_from_factors(&factors, &rows, Candidates::All(n_items), &exclusions, k)
                })
                .expect("enough")
        };
        // One thread takes the unsplit path; several take the split one.
        assert_eq!(run(1), run(4));
    }

    #[test]
    fn factors_without_bias_or_offset_rank_by_the_dot_product() {
        let users = vec![1.0, 0.0, 0.0, 1.0];
        let items = vec![0.5, 0.0, 0.0, 2.0, 1.0, 1.0];
        let factors = Factors {
            users: &users,
            items: &items,
            dim: 2,
            item_bias: None,
            user_offset: None,
        };
        let (order, scores) = top_k_from_factors(
            &factors,
            &[0, 1],
            Candidates::All(3),
            &Exclusions::<usize>::Nothing,
            2,
        )
        .expect("enough");
        assert_eq!(order, vec![2, 0, 1, 2]);
        assert_eq!(scores, vec![1.0, 0.5, 2.0, 1.0]);
    }
}
