//! A flat scan over quantized codes, with an exact rerank of what it shortlists.
//!
//! The other approximate index in this crate, [`crate::hnsw`], makes a query cheap by
//! visiting fewer items. This one visits every candidate and makes each visit cheap
//! instead: the item vectors are stored a second time as `bits`-wide codes, so a pass
//! over the catalog moves four to eight times less memory than a pass over the `f64`
//! vectors would. What the codes buy is a shortlist, not an answer -- the best
//! `oversample * k` of them are then rescored with the original vectors through
//! [`crate::vectors`], and the `k` returned are the best of *those*, with the exact
//! scores. A quantization error only costs an answer when it pushes a true top-`k` item
//! out of the shortlist entirely.
//!
//! That is a different trade from the graph's, and a strictly safer one. A scan cannot
//! fail to reach a candidate, so it needs no navigability precondition and no exhaustive
//! fallback, and its build is one pass of arithmetic rather than a graph construction.
//! It is also still linear in the catalog, which is exactly what the graph is not.
//!
//! # Scoring without decoding
//!
//! Quantization here is affine, `v ~= code * scale + offset`, so a score decomposes into
//! terms that do not need the code decoded one value at a time:
//!
//! ```text
//! <q, dec(c)>  =  sum_d q_d (c_d*s_d + o_d)  =  sum_d (q_d*s_d)*c_d  +  sum_d q_d*o_d
//! ```
//!
//! For a dense store the scale and the offset are per-dimension, so the second term is a
//! constant the query pays once and the inner loop is an `f64`-by-`u8` dot product. For a
//! sparse store they are scalars over the stored values, and the second term depends on
//! which columns the *item* stores, so the loop carries two accumulators instead and
//! finishes with `scale * sum(q*c) + offset * sum(q)`.
//!
//! Either way the arithmetic is an identity, not an approximation: all the error is in
//! the codes, which is where it can be reasoned about.
//!
//! # Loop order
//!
//! The dense shape is scanned a tile of queries at a time rather than a query at a time,
//! which is worth more than any of the arithmetic above: it amortizes the `u8`-to-`f64`
//! widening across the tile and reads the codes once per tile instead of once per query.
//! It took that path from 22.9 to 5.5 nanoseconds a candidate on a 200,000-item catalog.
//! [`scan_tiled`] says why it is written as a matmul micro-kernel, and which two shapes
//! keep the per-query path instead.
//!
//! # Threads
//!
//! There are none. The scan is sequential over queries by construction -- no `rayon`
//! import, no thread count to pass -- because it exists to be the predictable option
//! next to the graph, and because one workspace reused down the batch is cheaper than
//! one per worker when each query's work is already a tight loop over contiguous bytes.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::ranking::{Candidate, Excluded, TooFewEligible};
use crate::sparse::Csr;
use crate::vectors::{Items, Probe, Queries, Scatter};

/// The code widths a store may use.
///
/// Powers of two only, so a code never straddles a byte boundary and unpacking one is a
/// shift and a mask. The widths in between would need a two-byte read and a pair of
/// shifts per value, for a resolution nobody asks for by name.
pub const SUPPORTED_BITS: [u32; 4] = [1, 2, 4, 8];

/// Whether `bits` is a width this module can pack and unpack.
pub fn supported_bits(bits: u32) -> bool {
    SUPPORTED_BITS.contains(&bits)
}

/// Bytes one row of `len` codes of `bits` each occupies.
///
/// Rows are byte-aligned rather than packed end to end: it wastes under a byte per item
/// and it makes a row's address a multiplication instead of a bit offset, which the
/// dense scan does once per candidate.
pub fn row_bytes(len: usize, bits: u32) -> usize {
    let per_byte = 8 / bits as usize;
    len.div_ceil(per_byte)
}

/// Codes packed `8 / bits` to a byte, low-order code first, one aligned run per row.
#[derive(Clone, Copy)]
pub struct Codes<'a> {
    pub data: &'a [u8],
    pub bits: u32,
    /// Bytes from one row's first code to the next row's, as [`row_bytes`] gives it.
    pub stride: usize,
}

impl<'a> Codes<'a> {
    /// The packed bytes of row `row`.
    #[inline]
    fn row(&self, row: usize) -> &'a [u8] {
        &self.data[row * self.stride..(row + 1) * self.stride]
    }

    /// Whether `data` is long enough to hold `n_rows` rows at this stride.
    fn covers(&self, n_rows: usize) -> bool {
        self.data.len() >= n_rows * self.stride
    }
}

/// The `i`-th code of one packed row, at a width only known at run time.
///
/// The scan does not use this -- it matches the width once and runs a loop generic over
/// a `const BITS` instead -- but the packing has to be readable from a plain `bits` for
/// the round-trip tests to mean anything.
#[cfg(test)]
#[inline]
fn unpack(row: &[u8], bits: u32, i: usize) -> u8 {
    match bits {
        8 => code_at::<8>(row, i),
        4 => code_at::<4>(row, i),
        2 => code_at::<2>(row, i),
        _ => code_at::<1>(row, i),
    }
}

/// The quantized item vectors a scan runs over.
///
/// The two variants mirror [`crate::vectors::DenseItems`] and
/// [`crate::vectors::SparseItems`]: row `j` is item `j`, and the codes are in the same
/// order the exact vectors are, so a shortlisted item reranks by index with no mapping.
pub enum Store<'a> {
    /// `dim` codes an item, with a scale and an offset per dimension.
    Dense {
        codes: Codes<'a>,
        dim: usize,
        scale: &'a [f64],
        offset: &'a [f64],
    },
    /// One code per stored value, sharing the sparse rows' own `indptr`/`indices`, with
    /// one scale and one offset over the whole matrix.
    ///
    /// A catalog-sized column dimension has no per-column statistics worth keeping: the
    /// scale and offset arrays would outweigh the codes they compress. The implicit
    /// zeros are never visited, so the scheme does not need zero to be representable.
    Sparse {
        codes: Codes<'a>,
        matrix: Csr<'a>,
        scale: f64,
        offset: f64,
    },
}

impl Store<'_> {
    /// How many items the store holds codes for.
    pub fn len(&self) -> usize {
        match self {
            Self::Dense { codes, dim, .. } => {
                if *dim == 0 {
                    0
                } else {
                    codes.data.len() / codes.stride.max(1)
                }
            }
            Self::Sparse { matrix, .. } => matrix.n_rows,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the arrays describe a store of `n_items` over `dim` columns.
    ///
    /// A fitted index pickles as these arrays, so they can come back altered or
    /// truncated; every way of being wrong has to be a rejection rather than a panic.
    pub fn is_well_formed(&self, n_items: usize, dim: usize) -> bool {
        match self {
            Self::Dense {
                codes,
                dim: store_dim,
                scale,
                offset,
            } => {
                supported_bits(codes.bits)
                    && *store_dim == dim
                    && scale.len() == dim
                    && offset.len() == dim
                    && codes.stride == row_bytes(dim, codes.bits)
                    && codes.covers(n_items)
            }
            Self::Sparse { codes, matrix, .. } => {
                supported_bits(codes.bits)
                    && matrix.n_rows == n_items
                    && matrix.n_cols == dim
                    && codes.stride == 1
                    && codes.data.len() >= row_bytes(matrix.indptr[matrix.n_rows], codes.bits)
            }
        }
    }
}

/// A query prepared for the coarse pass: the query already folded into the scale.
///
/// Every variant carries the values `q_d * s_d` rather than `q_d`, so the scan multiplies
/// by a code and nothing else. `bias` is the part of the score that does not vary with
/// the codes -- a constant for a dense store, and the multiplier of the item's own
/// `sum(q)` for a sparse one.
enum Coarse<'a> {
    /// Dense store, dense or scattered query: `dim` pre-scaled values and a constant.
    Dense { scaled: &'a [f64], bias: f64 },
    /// Dense store, sparse query: the query's own columns, pre-scaled, and a constant.
    Sparse {
        indices: &'a [i64],
        scaled: &'a [f64],
        bias: f64,
    },
    /// Sparse store: the query scattered over the catalog dimension, unscaled, with the
    /// store's single scale and offset applied once per item instead.
    Scattered {
        values: &'a [f64],
        scale: f64,
        offset: f64,
    },
}

impl Coarse<'_> {
    /// The approximate score of `item`, by the identity in the module docs.
    ///
    /// Generic over the width rather than matching on it: the caller settles `BITS` once
    /// per query and the whole candidate loop is monomorphic from there, so the shift,
    /// the mask and the codes-per-byte are constants the compiler folds away and this
    /// inlines into the scan. Matching per candidate instead costs an indirect call in
    /// the hot loop and leaves the dot product unvectorized -- measurably so: it made
    /// four-bit codes slower than eight-bit ones, which is the opposite of the point.
    #[inline]
    fn score<const BITS: u32>(&self, store: &Store<'_>, item: usize) -> f64 {
        match (self, store) {
            (Self::Dense { scaled, bias }, Store::Dense { codes, .. }) => {
                bias + dense_dot::<BITS>(scaled, codes.row(item))
            }
            (
                Self::Sparse {
                    indices,
                    scaled,
                    bias,
                },
                Store::Dense { codes, .. },
            ) => bias + gather_dot::<BITS>(indices, scaled, codes.row(item)),
            (
                Self::Scattered {
                    values,
                    scale,
                    offset,
                },
                Store::Sparse { codes, matrix, .. },
            ) => {
                let range = matrix.indptr[item]..matrix.indptr[item + 1];
                let columns = &matrix.indices[range.clone()];
                let (coded, plain) =
                    scattered_dot::<BITS>(values, columns, codes.data, range.start);
                scale * coded + offset * plain
            }
            // The wrapper pairs a store with the query form that fits it, and `top_k`
            // builds both from the same `Queries`, so no other pairing is reachable.
            _ => unreachable!("a coarse query always matches the store it was built for"),
        }
    }
}

/// Codes converted per pass of the dense dot product.
///
/// Wide enough that the widening chain a code takes to become an `f64` -- and on every
/// target that is a chain, not one instruction -- runs on whole vectors, and wide enough
/// to hold eight one-bit codes so a byte is never split across passes.
const BLOCK: usize = 8;

/// Dot product of a pre-scaled query with a row of `BITS`-wide codes.
///
/// Written as "convert a block, then multiply the block" rather than as one fused loop,
/// which is the difference between vectorized and not. A code has to widen from `u8` to
/// `f64` before it can be multiplied, and asking for that one value at a time leaves the
/// compiler emitting the whole chain scalar; gathering a fixed-size block first gives it
/// a shape it can turn into vector widening plus a vector multiply-add.
///
/// `BLOCK` accumulators, kept live across the row, so the loop never serializes on the
/// latency of a single add -- the same reason [`crate::vectors`] keeps four.
#[inline]
fn dense_dot<const BITS: u32>(scaled: &[f64], row: &[u8]) -> f64 {
    let mut acc = [0.0f64; BLOCK];
    let (blocks, rest) = scaled.as_chunks::<BLOCK>();
    for (block, chunk) in blocks.iter().enumerate() {
        let codes = block_codes::<BITS>(row, block * BLOCK);
        for i in 0..BLOCK {
            acc[i] += chunk[i] * codes[i];
        }
    }
    let mut total =
        ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    let tail = scaled.len() - rest.len();
    for (i, value) in rest.iter().enumerate() {
        total += value * f64::from(code_at::<BITS>(row, tail + i));
    }
    total
}

/// The `BLOCK` codes starting at `start`, already widened to `f64`.
///
/// At eight bits this is a straight load and widen. Below it the codes come out of one
/// or two bytes by shift and mask, and because `BITS` is a constant the indices and the
/// shifts are too, so the whole thing folds into a handful of vector operations.
#[inline(always)]
fn block_codes<const BITS: u32>(row: &[u8], start: usize) -> [f64; BLOCK] {
    if BITS == 8 {
        let bytes = &row[start..start + BLOCK];
        return core::array::from_fn(|i| f64::from(bytes[i]));
    }
    let per_byte = 8 / BITS as usize;
    let mask = (1u8 << BITS) - 1;
    let first = start / per_byte;
    let bytes = &row[first..first + BLOCK / per_byte];
    core::array::from_fn(|i| {
        let byte = bytes[i / per_byte];
        f64::from((byte >> ((i % per_byte) as u32 * BITS)) & mask)
    })
}

/// Dot product of a sparse query with the codes of an item at the query's own columns.
///
/// EASE's shape: the item row is catalog-wide, so the gather runs over the query's
/// handful of nonzeros and each one is a random access into the packed row.
#[inline]
fn gather_dot<const BITS: u32>(indices: &[i64], scaled: &[f64], row: &[u8]) -> f64 {
    let mut total = 0.0;
    for (&column, value) in indices.iter().zip(scaled) {
        total += value * f64::from(code_at::<BITS>(row, column as usize));
    }
    total
}

/// The two sums a sparse store's scan carries, over one item's stored columns.
///
/// The offset multiplies the query values at the columns *this item* stores, which is
/// not a per-query constant, so it is accumulated alongside rather than folded in.
#[inline]
fn scattered_dot<const BITS: u32>(
    values: &[f64],
    columns: &[i64],
    codes: &[u8],
    start: usize,
) -> (f64, f64) {
    let mut coded = 0.0;
    let mut plain = 0.0;
    for (offset, &column) in columns.iter().enumerate() {
        let q = values[column as usize];
        coded += q * f64::from(code_at::<BITS>(codes, start + offset));
        plain += q;
    }
    (coded, plain)
}

/// The `i`-th code of a packed run, at a width known at compile time.
#[inline(always)]
fn code_at<const BITS: u32>(row: &[u8], i: usize) -> u8 {
    if BITS == 8 {
        return row[i];
    }
    let per_byte = 8 / BITS as usize;
    (row[i / per_byte] >> ((i % per_byte) as u32 * BITS)) & ((1u8 << BITS) - 1)
}

/// Scratch a whole batch shares, so a query costs no allocation of its own.
struct Workspace {
    /// The pre-scaled query, for the two query forms that need one.
    scaled: Vec<f64>,
    /// The scattered query, for a sparse store.
    scatter: Option<Scatter>,
    /// The shortlist the coarse pass fills, worst on top.
    shortlist: BinaryHeap<Candidate>,
    /// The shortlist reranked exactly, before it is cut to `k`.
    reranked: Vec<Candidate>,
}

impl Workspace {
    fn new(width: usize, pool: usize, scatter_dim: Option<usize>) -> Self {
        Self {
            scaled: vec![0.0; width],
            scatter: scatter_dim.map(Scatter::new),
            shortlist: BinaryHeap::with_capacity(pool + 1),
            reranked: Vec::with_capacity(pool),
        }
    }
}

/// The `k` best candidate positions of each query, and their exact scores.
///
/// `candidates` holds the item index of each candidate position, ascending, and excluded
/// columns are candidate positions, as in [`crate::ranking`]. Note what is missing next
/// to [`crate::hnsw::top_k`]: the inverse table mapping an item back to its candidate
/// position. A graph walk arrives at items and has to look up where each one sits in the
/// answer; a scan enumerates the positions itself, so it already knows.
///
/// `exact` must be the same vectors `store` holds codes for, indexed the same way: it is
/// what the shortlist is rescored against, and what makes the returned scores exact.
/// `oversample` sets the shortlist at `oversample * k`; at `1` the codes choose the answer
/// outright and the rerank only corrects the scores.
///
/// Unlike the graph search there is no fallback path, because there is nothing to fall
/// back from: a scan sees every eligible candidate, so a query short of `k` is genuinely
/// short of `k` and returns [`TooFewEligible`].
#[allow(clippy::too_many_arguments)]
pub fn top_k(
    store: &Store<'_>,
    exact: &impl Items,
    queries: &Queries<'_>,
    candidates: &[usize],
    excluded: &Excluded<'_>,
    k: usize,
    oversample: usize,
) -> Result<(Vec<usize>, Vec<f64>), TooFewEligible> {
    let n_queries = queries.len();
    let mut selected = vec![0usize; n_queries * k];
    let mut scores = vec![0.0f64; n_queries * k];
    if k == 0 || n_queries == 0 {
        return Ok((selected, scores));
    }

    // Shortlisting more than there are candidates is the exact path with extra steps, so
    // the pool is capped there and a big `oversample` degrades into exactness.
    let pool = oversample
        .saturating_mul(k)
        .clamp(k, candidates.len().max(k));
    // The latent-factor shape -- dense codes, dense queries -- is the one that pays for
    // being blocked, and the one whose competitor is a blocked matmul. See `scan_tiled`.
    if let (
        Store::Dense {
            codes,
            dim,
            scale,
            offset,
        },
        Queries::Dense { data, .. },
    ) = (store, queries)
        && *dim > 0
    {
        let tiled = Tiled {
            codes: *codes,
            dim: *dim,
            scale,
            offset,
            data,
        };
        match codes.bits {
            8 => scan_tiled::<8>(
                &tiled,
                exact,
                candidates,
                excluded,
                k,
                pool,
                &mut selected,
                &mut scores,
            )?,
            4 => scan_tiled::<4>(
                &tiled,
                exact,
                candidates,
                excluded,
                k,
                pool,
                &mut selected,
                &mut scores,
            )?,
            2 => scan_tiled::<2>(
                &tiled,
                exact,
                candidates,
                excluded,
                k,
                pool,
                &mut selected,
                &mut scores,
            )?,
            _ => scan_tiled::<1>(
                &tiled,
                exact,
                candidates,
                excluded,
                k,
                pool,
                &mut selected,
                &mut scores,
            )?,
        }
        return Ok((selected, scores));
    }

    let mut work = Workspace::new(
        coarse_width(store, queries),
        pool,
        scatter_width(store, queries),
    );

    for row in 0..n_queries {
        let (out_index, out_score) = (
            &mut selected[row * k..(row + 1) * k],
            &mut scores[row * k..(row + 1) * k],
        );
        rank_one(
            store, exact, queries, row, candidates, excluded, k, pool, &mut work, out_index,
            out_score,
        )?;
    }
    Ok((selected, scores))
}

/// Queries scored per pass over the candidates.
///
/// The whole point of the tile, so it is worth saying what sets it. Each candidate's
/// codes are widened to `f64` once per *tile* rather than once per query, so a tile of
/// `T` divides that cost by `T`; the codes are also read once per tile instead of once
/// per query, which is what lets a catalog that does not fit in cache behave as if it
/// did. Against that, the tile holds `T` pre-scaled queries and `T` shortlists live at
/// once. Eight is where the widening has stopped being the visible cost on the shapes
/// measured while the working set is still a few kilobytes.
const QUERY_TILE: usize = 16;

/// The dense store and dense queries a tiled scan runs over, gathered into one place
/// so the scan's own signature stays readable.
struct Tiled<'a> {
    codes: Codes<'a>,
    dim: usize,
    scale: &'a [f64],
    offset: &'a [f64],
    data: &'a [f64],
}

/// Scan the candidates once per tile of queries rather than once per query.
///
/// This is the structural answer to losing against a matmul. The per-query scan walks the
/// catalog for every query, so it widens each `u8` code to an `f64` once per query and
/// re-reads the whole code array every time -- which is a GEMV, and a GEMV cannot keep up
/// with a GEMM that blocks its queries against its items and reuses everything it loads.
///
/// So this is written as a matmul micro-kernel rather than as a loop of dot products. A
/// candidate's code is widened once and immediately multiplied into `QUERY_TILE`
/// accumulators that stay in registers for the whole row, which amortizes the widening
/// over the tile and reads the codes once per tile instead of once per query. Two things
/// were measured and both matter: decoding a row into a scratch buffer and then calling a
/// dot product per query is *slower* than the per-query scan, because it adds a store and
/// reload that the fused version never pays and puts an out-of-line call in the hot loop.
/// The query vectors are therefore stored transposed -- `scaled[d * QUERY_TILE + t]` -- so
/// the innermost loop walks one contiguous run and vectorizes.
///
/// Only this shape is tiled, and deliberately. A sparse store would need one
/// catalog-sized scatter buffer per query in the tile, which is megabytes on a six-figure
/// catalog; and EASE's catalog-wide dense rows would have to be decoded in full to serve a
/// query that only ever gathers a handful of columns, which is more work, not less. Both
/// keep the per-query path, where their costs already sit somewhere else.
#[allow(clippy::too_many_arguments)]
fn scan_tiled<const BITS: u32>(
    tiled: &Tiled<'_>,
    exact: &impl Items,
    candidates: &[usize],
    excluded: &Excluded<'_>,
    k: usize,
    pool: usize,
    selected: &mut [usize],
    scores: &mut [f64],
) -> Result<(), TooFewEligible> {
    let (codes, dim) = (tiled.codes, tiled.dim);
    let n_queries = tiled.data.len() / dim;
    // Transposed: query `t`'s value for dimension `d` sits at `d * QUERY_TILE + t`, so the
    // innermost loop walks one contiguous run of `QUERY_TILE` values.
    let mut scaled = vec![0.0f64; dim * QUERY_TILE];
    let mut bias = [0.0f64; QUERY_TILE];
    let mut cursor = [0usize; QUERY_TILE];
    let mut shortlists: Vec<BinaryHeap<Candidate>> = (0..QUERY_TILE)
        .map(|_| BinaryHeap::with_capacity(pool + 1))
        .collect();
    let mut reranked: Vec<Candidate> = Vec::with_capacity(pool);

    for start in (0..n_queries).step_by(QUERY_TILE) {
        let tile = QUERY_TILE.min(n_queries - start);
        // A short last tile keeps computing all `QUERY_TILE` accumulators -- the lanes it
        // does not need are zeroed here, which is cheaper than branching in the hot loop.
        scaled.fill(0.0);
        for (t, shortlist) in shortlists.iter_mut().enumerate().take(tile) {
            // Fold the scale into the query once, exactly as the per-query path does: the
            // identity is the same, only the loop order around it has changed.
            let query = &tiled.data[(start + t) * dim..(start + t + 1) * dim];
            let mut constant = 0.0;
            for (d, &q) in query.iter().enumerate() {
                scaled[d * QUERY_TILE + t] = q * tiled.scale[d];
                constant += q * tiled.offset[d];
            }
            bias[t] = constant;
            cursor[t] = 0;
            shortlist.clear();
        }

        for (slot, &item) in candidates.iter().enumerate() {
            let acc = tile_row::<BITS>(codes.row(item), &scaled, dim);
            for (t, shortlist) in shortlists.iter_mut().enumerate().take(tile) {
                // The excluded positions ascend, so one cursor per query keeps up. The
                // score is already computed by the time this runs: with the tile fused,
                // dropping a candidate for one query would save nothing but the heap.
                let skip = excluded.row(start + t);
                if cursor[t] < skip.len() && skip[cursor[t]] == slot as i64 {
                    cursor[t] += 1;
                    continue;
                }
                let candidate = Candidate {
                    score: bias[t] + acc[t],
                    index: slot,
                };
                if shortlist.len() < pool {
                    shortlist.push(candidate);
                } else if let Some(worst) = shortlist.peek()
                    && candidate.worst_first(worst) == Ordering::Less
                {
                    shortlist.pop();
                    shortlist.push(candidate);
                }
            }
        }

        for (t, shortlist) in shortlists.iter_mut().enumerate().take(tile) {
            let row = start + t;
            if shortlist.len() < k {
                return Err(TooFewEligible {
                    row,
                    found: shortlist.len(),
                });
            }
            let probe = Probe::Dense(&tiled.data[row * dim..(row + 1) * dim]);
            reranked.clear();
            for candidate in shortlist.drain() {
                reranked.push(Candidate {
                    score: exact.score(candidates[candidate.index], &probe),
                    index: candidate.index,
                });
            }
            reranked.sort_unstable_by(|a, b| a.worst_first(b));
            let out_index = &mut selected[row * k..(row + 1) * k];
            let out_score = &mut scores[row * k..(row + 1) * k];
            for ((slot, score), candidate) in out_index.iter_mut().zip(out_score).zip(&reranked) {
                *slot = candidate.index;
                *score = candidate.score;
            }
        }
    }
    Ok(())
}

/// One candidate's codes against a whole tile of pre-scaled queries.
///
/// The micro-kernel: `QUERY_TILE` accumulators held in registers across the row, each code
/// widened once and multiplied into all of them. `scaled` is transposed, so the innermost
/// loop is a contiguous run that compiles to a handful of vector multiply-adds.
#[inline]
fn tile_row<const BITS: u32>(row: &[u8], scaled: &[f64], dim: usize) -> [f64; QUERY_TILE] {
    let mut acc = [0.0f64; QUERY_TILE];
    let blocks = dim / BLOCK;
    for block in 0..blocks {
        let codes = block_codes::<BITS>(row, block * BLOCK);
        for (i, &code) in codes.iter().enumerate() {
            let lane = &scaled[(block * BLOCK + i) * QUERY_TILE..][..QUERY_TILE];
            for t in 0..QUERY_TILE {
                acc[t] += lane[t] * code;
            }
        }
    }
    for d in blocks * BLOCK..dim {
        let code = f64::from(code_at::<BITS>(row, d));
        let lane = &scaled[d * QUERY_TILE..][..QUERY_TILE];
        for t in 0..QUERY_TILE {
            acc[t] += lane[t] * code;
        }
    }
    acc
}

/// Values the pre-scaled query buffer must hold, which depends on the query form.
fn coarse_width(store: &Store<'_>, queries: &Queries<'_>) -> usize {
    match (store, queries) {
        (Store::Dense { dim, .. }, Queries::Dense { .. }) => *dim,
        // Both sparse query forms take the same branch of `with_coarse`, which pre-scales
        // the query's own nonzeros rather than a dimension's worth.
        (Store::Dense { .. }, Queries::Sparse(csr) | Queries::Scattered(csr)) => widest_row(csr),
        // A sparse store scatters instead of pre-scaling: its scale is one number.
        (Store::Sparse { .. }, _) => 0,
    }
}

/// The buffer a sparse store's scattered query needs, or `None` when nothing scatters.
fn scatter_width(store: &Store<'_>, queries: &Queries<'_>) -> Option<usize> {
    match (store, queries) {
        (Store::Sparse { matrix, .. }, _) => Some(matrix.n_cols),
        _ => queries.scatter_dim(),
    }
}

/// The most stored values any one row of `csr` holds.
fn widest_row(csr: &Csr<'_>) -> usize {
    (0..csr.n_rows)
        .map(|r| csr.indptr[r + 1] - csr.indptr[r])
        .max()
        .unwrap_or(0)
}

/// Rank one query: coarse scan, exact rerank, cut to `k`.
#[allow(clippy::too_many_arguments)]
fn rank_one(
    store: &Store<'_>,
    exact: &impl Items,
    queries: &Queries<'_>,
    row: usize,
    candidates: &[usize],
    excluded: &Excluded<'_>,
    k: usize,
    pool: usize,
    work: &mut Workspace,
    out_index: &mut [usize],
    out_score: &mut [f64],
) -> Result<(), TooFewEligible> {
    let Workspace {
        scaled,
        scatter,
        shortlist,
        reranked,
    } = work;
    shortlist.clear();
    reranked.clear();

    let skip = excluded.row(row);
    let bits = match store {
        Store::Dense { codes, .. } | Store::Sparse { codes, .. } => codes.bits,
    };
    with_coarse(store, queries, row, scaled, scatter, |coarse| {
        // The one place the width is a value rather than a constant. Everything below it
        // is monomorphic, which is what lets the dot product inline and vectorize.
        match bits {
            8 => shortlist_by_codes::<8>(coarse, store, candidates, skip, pool, shortlist),
            4 => shortlist_by_codes::<4>(coarse, store, candidates, skip, pool, shortlist),
            2 => shortlist_by_codes::<2>(coarse, store, candidates, skip, pool, shortlist),
            _ => shortlist_by_codes::<1>(coarse, store, candidates, skip, pool, shortlist),
        }
    });

    if shortlist.len() < k {
        return Err(TooFewEligible {
            row,
            found: shortlist.len(),
        });
    }

    // The rerank is what makes the scores exact, so it goes through the same scoring the
    // exact path uses rather than dequantizing the codes.
    queries.with_probe(row, scatter, |probe| {
        for candidate in shortlist.drain() {
            reranked.push(Candidate {
                score: exact.score(candidates[candidate.index], probe),
                index: candidate.index,
            });
        }
    });
    reranked.sort_unstable_by(|a, b| a.worst_first(b));

    for ((slot, score), candidate) in out_index.iter_mut().zip(out_score).zip(&*reranked) {
        *slot = candidate.index;
        *score = candidate.score;
    }
    Ok(())
}

/// Fill `shortlist` with the best `pool` candidates by their coarse scores.
///
/// The hot loop of the whole index: one pass over the candidates, skipping the excluded
/// ones by the same ascending cursor [`crate::ranking::top_k_per_row`] uses, keeping a
/// `pool`-sized heap of the best seen so far. Ties go to the lower candidate position,
/// so the shortlist a query gets is a function of the codes and nothing else.
fn shortlist_by_codes<const BITS: u32>(
    coarse: &Coarse<'_>,
    store: &Store<'_>,
    candidates: &[usize],
    skip: &[i64],
    pool: usize,
    shortlist: &mut BinaryHeap<Candidate>,
) {
    let mut next_skipped = 0usize;
    for (slot, &item) in candidates.iter().enumerate() {
        // The excluded positions ascend, so one cursor keeps up with the scan.
        if next_skipped < skip.len() && skip[next_skipped] == slot as i64 {
            next_skipped += 1;
            continue;
        }
        let candidate = Candidate {
            score: coarse.score::<BITS>(store, item),
            index: slot,
        };
        if shortlist.len() < pool {
            shortlist.push(candidate);
        } else if let Some(worst) = shortlist.peek()
            && candidate.worst_first(worst) == Ordering::Less
        {
            shortlist.pop();
            shortlist.push(candidate);
        }
    }
}

/// Prepare query `row` for the coarse pass and hand it to `f`.
///
/// The preparation is the whole reason the scan is cheap: the query is folded into the
/// scale once here, so the inner loop over the catalog multiplies by a code alone.
fn with_coarse<R>(
    store: &Store<'_>,
    queries: &Queries<'_>,
    row: usize,
    scaled: &mut [f64],
    scatter: &mut Option<Scatter>,
    f: impl FnOnce(&Coarse<'_>) -> R,
) -> R {
    match store {
        Store::Dense { scale, offset, .. } => match queries {
            Queries::Dense { data, dim } => {
                let query = &data[row * dim..(row + 1) * dim];
                let mut bias = 0.0;
                for (d, &q) in query.iter().enumerate() {
                    scaled[d] = q * scale[d];
                    bias += q * offset[d];
                }
                f(&Coarse::Dense { scaled, bias })
            }
            Queries::Sparse(csr) | Queries::Scattered(csr) => {
                // A dense store with catalog-sized rows is EASE's shape: the query's few
                // nonzeros are what the gather runs over, scattered or not.
                let range = csr.indptr[row]..csr.indptr[row + 1];
                let (indices, data) = (&csr.indices[range.clone()], &csr.data[range]);
                let mut bias = 0.0;
                for (p, (&column, &q)) in indices.iter().zip(data).enumerate() {
                    scaled[p] = q * scale[column as usize];
                    bias += q * offset[column as usize];
                }
                f(&Coarse::Sparse {
                    indices,
                    scaled: &scaled[..indices.len()],
                    bias,
                })
            }
        },
        Store::Sparse { scale, offset, .. } => {
            // Short sparse item rows over a catalog-sized dimension: scatter the query
            // once and let every candidate gather its own stored columns out of it.
            let csr = match queries {
                Queries::Sparse(csr) | Queries::Scattered(csr) => csr,
                Queries::Dense { .. } => {
                    unreachable!("a sparse store takes sparse queries")
                }
            };
            let range = csr.indptr[row]..csr.indptr[row + 1];
            let buffer = scatter
                .as_mut()
                .expect("a sparse store reserves a scatter buffer");
            buffer.with(&csr.indices[range.clone()], &csr.data[range], |probe| {
                let Probe::Scattered(values) = probe else {
                    unreachable!("`Scatter::with` hands over a scattered probe")
                };
                f(&Coarse::Scattered {
                    values,
                    scale: *scale,
                    offset: *offset,
                })
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::testing::Owned;
    use crate::vectors::{DenseItems, SparseItems};

    /// Quantize `values` to `bits` over `[lo, hi]`, as the Python side does.
    fn encode(values: &[f64], lo: f64, hi: f64, bits: u32) -> (Vec<u8>, f64, f64) {
        let levels = f64::from((1u32 << bits) - 1);
        let scale = if hi > lo { (hi - lo) / levels } else { 0.0 };
        let codes = values
            .iter()
            .map(|&v| {
                if scale == 0.0 {
                    0
                } else {
                    (((v.clamp(lo, hi) - lo) / scale).round()).clamp(0.0, levels) as u8
                }
            })
            .collect();
        (codes, scale, lo)
    }

    /// Pack one row of codes the way [`Codes`] reads them back.
    fn pack_row(codes: &[u8], bits: u32) -> Vec<u8> {
        let mut out = vec![0u8; row_bytes(codes.len(), bits)];
        let per_byte = 8 / bits as usize;
        for (i, &code) in codes.iter().enumerate() {
            out[i / per_byte] |= code << ((i % per_byte) as u32 * bits);
        }
        out
    }

    fn pack(codes: &[u8], bits: u32, dim: usize) -> Vec<u8> {
        codes
            .chunks(dim)
            .flat_map(|row| pack_row(row, bits))
            .collect()
    }

    #[test]
    fn packing_round_trips_at_every_supported_width() {
        for bits in SUPPORTED_BITS {
            let max = (1u16 << bits) - 1;
            let values: Vec<u8> = (0..37).map(|i| (i % max as usize) as u8).collect();
            let packed = pack_row(&values, bits);
            assert_eq!(packed.len(), row_bytes(values.len(), bits), "bits {bits}");
            for (i, &expected) in values.iter().enumerate() {
                assert_eq!(unpack(&packed, bits, i), expected, "bits {bits}, i {i}");
            }
        }
    }

    /// A dense catalog with enough spread that quantization actually loses something.
    fn dense_fixture(n_items: usize, dim: usize) -> Vec<f64> {
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        (0..n_items * dim)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 2001) as f64 / 1000.0 - 1.0
            })
            .collect()
    }

    fn all_candidates(n: usize) -> Vec<usize> {
        (0..n).collect()
    }

    fn no_exclusions(n_rows: usize) -> (Vec<usize>, Vec<i64>) {
        (vec![0; n_rows + 1], Vec::new())
    }

    /// Rank every candidate exactly, which is what the index must agree with.
    fn brute_force(items: &DenseItems<'_>, query: &[f64], k: usize) -> Vec<usize> {
        let mut all: Vec<Candidate> = (0..items.len())
            .map(|i| Candidate {
                score: items.score(i, &Probe::Dense(query)),
                index: i,
            })
            .collect();
        all.sort_unstable_by(|a, b| a.worst_first(b));
        all.into_iter().take(k).map(|c| c.index).collect()
    }

    #[test]
    fn eight_bits_and_a_wide_shortlist_reproduce_the_exact_ranking() {
        // The property the rerank is for: with the full code range and a shortlist that
        // cannot miss, the answer and the scores are the exact path's, not near it.
        let (n_items, dim, k) = (300, 16, 10);
        let vectors = dense_fixture(n_items, dim);
        let items = DenseItems {
            vectors: &vectors,
            dim,
        };
        let lo = vectors.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = vectors.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (codes, scale, offset) = encode(&vectors, lo, hi, 8);
        let packed = pack(&codes, 8, dim);
        let store = Store::Dense {
            codes: Codes {
                data: &packed,
                bits: 8,
                stride: row_bytes(dim, 8),
            },
            dim,
            scale: &vec![scale; dim],
            offset: &vec![offset; dim],
        };
        assert!(store.is_well_formed(n_items, dim));

        let queries = dense_fixture(5, dim);
        let candidates = all_candidates(n_items);
        let (indptr, indices) = no_exclusions(5);
        let (order, scores) = top_k(
            &store,
            &items,
            &Queries::Dense {
                data: &queries,
                dim,
            },
            &candidates,
            &Excluded {
                indptr: &indptr,
                indices: &indices,
            },
            k,
            n_items,
        )
        .expect("every query has candidates");

        for row in 0..5 {
            let query = &queries[row * dim..(row + 1) * dim];
            let expected = brute_force(&items, query, k);
            assert_eq!(&order[row * k..(row + 1) * k], &expected[..], "row {row}");
            for (slot, &item) in expected.iter().enumerate() {
                let exact = items.score(item, &Probe::Dense(query));
                assert!((scores[row * k + slot] - exact).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn a_batch_that_straddles_the_tile_boundary_still_matches_the_exact_ranking() {
        // The tiled scan processes `QUERY_TILE` queries at a time, so the query count
        // that matters is one that is neither a multiple of it nor smaller than it: the
        // last tile is then short, and its unused accumulator lanes have to stay out of
        // the answer. Every other test here uses fewer queries than one tile holds.
        let (n_items, dim, k) = (300, 16, 10);
        let n_queries = 2 * QUERY_TILE + 3;
        let vectors = dense_fixture(n_items, dim);
        let items = DenseItems {
            vectors: &vectors,
            dim,
        };
        let (codes, scale, offset) = encode(&vectors, -1.0, 1.0, 8);
        let packed = pack(&codes, 8, dim);
        let store = Store::Dense {
            codes: Codes {
                data: &packed,
                bits: 8,
                stride: row_bytes(dim, 8),
            },
            dim,
            scale: &vec![scale; dim],
            offset: &vec![offset; dim],
        };
        let queries = dense_fixture(n_queries, dim);
        let candidates = all_candidates(n_items);
        let (indptr, indices) = no_exclusions(n_queries);
        let (order, scores) = top_k(
            &store,
            &items,
            &Queries::Dense {
                data: &queries,
                dim,
            },
            &candidates,
            &Excluded {
                indptr: &indptr,
                indices: &indices,
            },
            k,
            n_items,
        )
        .expect("every query has candidates");

        for row in 0..n_queries {
            let query = &queries[row * dim..(row + 1) * dim];
            let expected = brute_force(&items, query, k);
            assert_eq!(&order[row * k..(row + 1) * k], &expected[..], "row {row}");
            for (slot, &item) in expected.iter().enumerate() {
                let exact = items.score(item, &Probe::Dense(query));
                assert!((scores[row * k + slot] - exact).abs() < 1e-12, "row {row}");
            }
        }
    }

    #[test]
    fn the_tiled_and_per_query_paths_agree_on_the_same_data() {
        // Only the dense-query shape is tiled. The sparse-query shape over the same dense
        // store takes the per-query path, and the two must not disagree about anything --
        // they are the same identity with a different loop order around it.
        let (n_items, dim, k) = (200, 8, 5);
        let vectors = dense_fixture(n_items, dim);
        let items = DenseItems {
            vectors: &vectors,
            dim,
        };
        let (codes, scale, offset) = encode(&vectors, -1.0, 1.0, 4);
        let packed = pack(&codes, 4, dim);
        let n_queries = QUERY_TILE + 5;
        let dense_queries = dense_fixture(n_queries, dim);
        // The same queries, expressed sparsely: every dimension stored, so nothing is lost.
        let rows: Vec<Vec<f64>> = dense_queries.chunks(dim).map(<[f64]>::to_vec).collect();
        let owned = Owned::from_dense(&rows);
        let candidates = all_candidates(n_items);
        let (indptr, indices) = no_exclusions(n_queries);
        let excluded = Excluded {
            indptr: &indptr,
            indices: &indices,
        };
        let scales = vec![scale; dim];
        let offsets = vec![offset; dim];
        let store = || Store::Dense {
            codes: Codes {
                data: &packed,
                bits: 4,
                stride: row_bytes(dim, 4),
            },
            dim,
            scale: &scales,
            offset: &offsets,
        };
        let tiled = top_k(
            &store(),
            &items,
            &Queries::Dense {
                data: &dense_queries,
                dim,
            },
            &candidates,
            &excluded,
            k,
            4,
        )
        .expect("every query has candidates");
        let per_query = top_k(
            &store(),
            &items,
            &Queries::Sparse(owned.csr()),
            &candidates,
            &excluded,
            k,
            4,
        )
        .expect("every query has candidates");
        assert_eq!(tiled.0, per_query.0);
        for (a, b) in tiled.1.iter().zip(&per_query.1) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn a_narrow_code_still_returns_exact_scores() {
        // The coarse pass may shortlist the wrong items at one bit, but whatever it
        // returns is scored by the exact vectors, never by the codes.
        let (n_items, dim, k) = (200, 8, 5);
        let vectors = dense_fixture(n_items, dim);
        let items = DenseItems {
            vectors: &vectors,
            dim,
        };
        let (codes, scale, offset) = encode(&vectors, -1.0, 1.0, 1);
        let packed = pack(&codes, 1, dim);
        let store = Store::Dense {
            codes: Codes {
                data: &packed,
                bits: 1,
                stride: row_bytes(dim, 1),
            },
            dim,
            scale: &vec![scale; dim],
            offset: &vec![offset; dim],
        };
        let queries = dense_fixture(3, dim);
        let candidates = all_candidates(n_items);
        let (indptr, indices) = no_exclusions(3);
        let (order, scores) = top_k(
            &store,
            &items,
            &Queries::Dense {
                data: &queries,
                dim,
            },
            &candidates,
            &Excluded {
                indptr: &indptr,
                indices: &indices,
            },
            k,
            2,
        )
        .expect("every query has candidates");

        for row in 0..3 {
            let query = &queries[row * dim..(row + 1) * dim];
            for slot in 0..k {
                let item = order[row * k + slot];
                let exact = items.score(item, &Probe::Dense(query));
                assert!((scores[row * k + slot] - exact).abs() < 1e-12);
            }
            // Still ranked, whatever was shortlisted.
            let row_scores = &scores[row * k..(row + 1) * k];
            assert!(row_scores.windows(2).all(|w| w[0] >= w[1]));
        }
    }

    #[test]
    fn a_wider_shortlist_never_ranks_worse() {
        let (n_items, dim, k) = (400, 12, 10);
        let vectors = dense_fixture(n_items, dim);
        let items = DenseItems {
            vectors: &vectors,
            dim,
        };
        let (codes, scale, offset) = encode(&vectors, -1.0, 1.0, 2);
        let packed = pack(&codes, 2, dim);
        let queries = dense_fixture(6, dim);
        let candidates = all_candidates(n_items);
        let (indptr, indices) = no_exclusions(6);

        let mut previous = 0usize;
        for oversample in [1, 2, 4, 16, 64] {
            let store = Store::Dense {
                codes: Codes {
                    data: &packed,
                    bits: 2,
                    stride: row_bytes(dim, 2),
                },
                dim,
                scale: &vec![scale; dim],
                offset: &vec![offset; dim],
            };
            let (order, _) = top_k(
                &store,
                &items,
                &Queries::Dense {
                    data: &queries,
                    dim,
                },
                &candidates,
                &Excluded {
                    indptr: &indptr,
                    indices: &indices,
                },
                k,
                oversample,
            )
            .expect("every query has candidates");
            let hits: usize = (0..6)
                .map(|row| {
                    let expected = brute_force(&items, &queries[row * dim..(row + 1) * dim], k);
                    order[row * k..(row + 1) * k]
                        .iter()
                        .filter(|i| expected.contains(i))
                        .count()
                })
                .sum();
            assert!(hits >= previous, "oversample {oversample} lost ground");
            previous = hits;
        }
        assert_eq!(
            previous, 60,
            "a shortlist of the whole catalog must be exact"
        );
    }

    #[test]
    fn excluded_positions_are_skipped_and_candidates_are_positions() {
        let dim = 2;
        let vectors = vec![1.0, 0.0, 0.9, 0.0, 0.8, 0.0, 0.7, 0.0];
        let items = DenseItems {
            vectors: &vectors,
            dim,
        };
        let (codes, scale, offset) = encode(&vectors, 0.0, 1.0, 8);
        let packed = pack(&codes, 8, dim);
        let store = Store::Dense {
            codes: Codes {
                data: &packed,
                bits: 8,
                stride: row_bytes(dim, 8),
            },
            dim,
            scale: &vec![scale; dim],
            offset: &vec![offset; dim],
        };
        // Candidates 1 and 3 of the catalog, so position 0 is item 1 and 1 is item 3.
        let candidates = vec![1usize, 3];
        let queries = vec![1.0, 0.0];
        let (indptr, indices) = (vec![0usize, 1], vec![0i64]);
        let (order, scores) = top_k(
            &store,
            &items,
            &Queries::Dense {
                data: &queries,
                dim,
            },
            &candidates,
            &Excluded {
                indptr: &indptr,
                indices: &indices,
            },
            1,
            4,
        )
        .expect("one candidate survives");
        assert_eq!(order, vec![1]);
        assert!((scores[0] - 0.7).abs() < 1e-12);
    }

    #[test]
    fn a_query_short_of_k_names_its_row() {
        let dim = 2;
        let vectors = vec![1.0, 0.0, 0.5, 0.0];
        let items = DenseItems {
            vectors: &vectors,
            dim,
        };
        let (codes, scale, offset) = encode(&vectors, 0.0, 1.0, 8);
        let packed = pack(&codes, 8, dim);
        let store = Store::Dense {
            codes: Codes {
                data: &packed,
                bits: 8,
                stride: row_bytes(dim, 8),
            },
            dim,
            scale: &vec![scale; dim],
            offset: &vec![offset; dim],
        };
        let candidates = all_candidates(2);
        let queries = vec![1.0, 0.0, 1.0, 0.0];
        let (indptr, indices) = (vec![0usize, 0, 1], vec![0i64]);
        let error = top_k(
            &store,
            &items,
            &Queries::Dense {
                data: &queries,
                dim,
            },
            &candidates,
            &Excluded {
                indptr: &indptr,
                indices: &indices,
            },
            2,
            4,
        )
        .expect_err("row 1 has one eligible candidate");
        assert_eq!((error.row, error.found), (1, 1));
    }

    #[test]
    fn a_sparse_store_scores_the_way_the_exact_rows_do() {
        let owned = Owned::from_dense(&[
            vec![1.0, 0.0, 2.0, 0.0],
            vec![0.0, 3.0, 4.0, 5.0],
            vec![0.5, 0.0, 0.0, 1.5],
            vec![0.0, 0.0, 2.5, 0.0],
        ]);
        let items = SparseItems {
            matrix: owned.csr(),
        };
        let (codes, scale, offset) = encode(owned.csr().data, 0.0, 5.0, 8);
        let store = Store::Sparse {
            codes: Codes {
                data: &codes,
                bits: 8,
                stride: 1,
            },
            matrix: owned.csr(),
            scale,
            offset,
        };
        assert!(store.is_well_formed(4, 4));

        let query_owned = Owned::from_dense(&[vec![0.0, 2.0, 1.0, 3.0], vec![1.0, 0.0, 0.0, 0.0]]);
        let candidates = all_candidates(4);
        let (indptr, indices) = no_exclusions(2);
        let (order, scores) = top_k(
            &store,
            &items,
            &Queries::Scattered(query_owned.csr()),
            &candidates,
            &Excluded {
                indptr: &indptr,
                indices: &indices,
            },
            2,
            4,
        )
        .expect("every query has candidates");

        let query_csr = query_owned.csr();
        for row in 0..2 {
            let range = query_csr.indptr[row]..query_csr.indptr[row + 1];
            let probe = Probe::Sparse {
                indices: &query_csr.indices[range.clone()],
                data: &query_csr.data[range],
            };
            for slot in 0..2 {
                let exact = items.score(order[row * 2 + slot], &probe);
                assert!((scores[row * 2 + slot] - exact).abs() < 1e-12, "row {row}");
            }
        }
    }

    #[test]
    fn a_truncated_or_mislabelled_store_is_not_well_formed() {
        let packed = vec![0u8; 8];
        let scale = vec![1.0; 4];
        let short = Store::Dense {
            codes: Codes {
                data: &packed,
                bits: 8,
                stride: 4,
            },
            dim: 4,
            scale: &scale,
            offset: &scale,
        };
        assert!(short.is_well_formed(2, 4));
        assert!(
            !short.is_well_formed(3, 4),
            "three rows do not fit in eight bytes"
        );
        assert!(
            !short.is_well_formed(2, 5),
            "dim must match the scale arrays"
        );

        let odd_bits = Store::Dense {
            codes: Codes {
                data: &packed,
                bits: 3,
                stride: 4,
            },
            dim: 4,
            scale: &scale,
            offset: &scale,
        };
        assert!(
            !odd_bits.is_well_formed(2, 4),
            "three-bit codes are not supported"
        );
    }

    #[test]
    fn nothing_to_rank_is_not_an_error() {
        let packed = vec![0u8; 4];
        let scale = vec![1.0; 2];
        let store = Store::Dense {
            codes: Codes {
                data: &packed,
                bits: 8,
                stride: 2,
            },
            dim: 2,
            scale: &scale,
            offset: &scale,
        };
        let vectors = vec![0.0; 4];
        let items = DenseItems {
            vectors: &vectors,
            dim: 2,
        };
        let candidates = all_candidates(2);
        let (indptr, indices) = no_exclusions(0);
        let (order, scores) = top_k(
            &store,
            &items,
            &Queries::Dense { data: &[], dim: 2 },
            &candidates,
            &Excluded {
                indptr: &indptr,
                indices: &indices,
            },
            3,
            4,
        )
        .expect("no queries is no work");
        assert!(order.is_empty() && scores.is_empty());
    }
}
