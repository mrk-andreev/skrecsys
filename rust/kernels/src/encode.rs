//! Identifier encoding: distinct values, and the CSR matrix their codes describe.
//!
//! `np.unique(values, return_inverse=True)` argsorts the whole column, which is
//! `O(n log n)` over every interaction. Identifiers repeat heavily -- that is what makes
//! them identifiers -- so it is cheaper to find the distinct values in one pass and sort
//! only those. Two passes fit the two shapes ids come in: a dense range of integers is
//! bucketed, and anything more spread out goes through an open-addressed table.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::sparse::CsrOwned;

/// Sorted distinct values, and the position of every input value among them.
pub fn factorize(values: &[i64]) -> (Vec<i64>, Vec<i64>) {
    if values.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let (min, max) = values
        .iter()
        .fold((i64::MAX, i64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let span = (max as i128 - min as i128 + 1) as u128;
    // Bucketing costs 4 bytes per value of the span, so it has to stay proportional to
    // the input; ids that are counted upwards from zero, which is the usual shape, are
    // far below this.
    let budget = (values.len() as u128 * 2).min(u32::MAX as u128 - 1);
    if span <= budget {
        bucketed(values, min, span as usize)
    } else {
        hashed(values)
    }
}

/// Mark which offsets of the span occur, then number them in place, ascending.
fn bucketed(values: &[i64], min: i64, span: usize) -> (Vec<i64>, Vec<i64>) {
    let mut code = vec![0u32; span];
    for &v in values {
        code[(v - min) as usize] = 1;
    }
    let mut uniques = Vec::new();
    for (offset, slot) in code.iter_mut().enumerate() {
        if *slot == 1 {
            uniques.push(min + offset as i64);
            // Codes are stored one-based, so that zero still means absent.
            *slot = uniques.len() as u32;
        }
    }
    let codes = values
        .iter()
        .map(|&v| code[(v - min) as usize] as i64 - 1)
        .collect();
    (uniques, codes)
}

/// Number the values in first-seen order, then sort the distinct ones and renumber.
fn hashed(values: &[i64]) -> (Vec<i64>, Vec<i64>) {
    let mut table = Table::new();
    let seen: Vec<usize> = values.iter().map(|&v| table.insert(v)).collect();

    let mut order: Vec<usize> = (0..table.keys.len()).collect();
    order.sort_unstable_by_key(|&code| table.keys[code]);
    let mut rank = vec![0usize; order.len()];
    for (sorted, &code) in order.iter().enumerate() {
        rank[code] = sorted;
    }
    (
        order.iter().map(|&code| table.keys[code]).collect(),
        seen.iter().map(|&code| rank[code] as i64).collect(),
    )
}

const EMPTY: usize = usize::MAX;

/// Open-addressed map from a value to the order it was first seen in.
///
/// It starts small and doubles, so it stays proportional to the number of distinct ids
/// rather than to the number of rows.
struct Table {
    keys: Vec<i64>,
    slots: Vec<usize>,
    mask: usize,
}

impl Table {
    fn new() -> Self {
        Self {
            keys: Vec::new(),
            slots: vec![EMPTY; 1024],
            mask: 1023,
        }
    }

    /// Fibonacci hashing: the high bits of the product are the well-mixed ones.
    fn probe(slots: &[usize], mask: usize, keys: &[i64], key: i64) -> usize {
        let mut at = ((key as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32) as usize & mask;
        while slots[at] != EMPTY && keys[slots[at]] != key {
            at = (at + 1) & mask;
        }
        at
    }

    fn insert(&mut self, key: i64) -> usize {
        let at = Self::probe(&self.slots, self.mask, &self.keys, key);
        if self.slots[at] != EMPTY {
            return self.slots[at];
        }
        let code = self.keys.len();
        self.keys.push(key);
        self.slots[at] = code;
        if self.keys.len() * 4 > self.slots.len() * 3 {
            self.grow();
        }
        code
    }

    fn grow(&mut self) {
        let mut slots = vec![EMPTY; self.slots.len() * 2];
        let mask = slots.len() - 1;
        for (code, &key) in self.keys.iter().enumerate() {
            let at = Self::probe(&slots, mask, &self.keys, key);
            slots[at] = code;
        }
        self.slots = slots;
        self.mask = mask;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What `np.unique(values, return_inverse=True)` returns.
    fn reference(values: &[i64]) -> (Vec<i64>, Vec<i64>) {
        let mut uniques: Vec<i64> = values.to_vec();
        uniques.sort_unstable();
        uniques.dedup();
        let codes = values
            .iter()
            .map(|v| uniques.binary_search(v).expect("present") as i64)
            .collect();
        (uniques, codes)
    }

    fn check(values: &[i64]) {
        assert_eq!(factorize(values), reference(values), "{values:?}");
    }

    #[test]
    fn matches_a_sorted_unique() {
        check(&[]);
        check(&[7]);
        check(&[2, 2, 2]);
        check(&[3, 1, 2, 1, 3, 0]);
        check(&[-5, 5, 0, -5, 3]);
    }

    #[test]
    fn both_paths_agree() {
        // The same values, once dense enough to bucket and once spread out.
        let dense: Vec<i64> = (0..5_000).map(|i| (i * 7) % 300).collect();
        check(&dense);
        let spread: Vec<i64> = dense.iter().map(|v| v * 1_000_000_007).collect();
        check(&spread);
        assert_eq!(factorize(&spread).1, factorize(&dense).1, "same grouping");
    }

    #[test]
    fn survives_the_extremes_of_the_range() {
        check(&[i64::MIN, i64::MAX, 0, i64::MIN]);
    }

    #[test]
    fn grows_the_table_past_its_first_size() {
        let spread: Vec<i64> = (0..5_000).map(|i| i * 1_000_000_007).collect();
        check(&spread);
    }
}

/// Smallest slice of the input worth handing to a worker of its own.
const MIN_BLOCK: usize = 1 << 16;

/// Where each block's entries go in one counting-sort pass, and where each bucket starts.
///
/// `bounds` splits the pass into blocks of consecutive entries. Every block counts into a
/// row of the table of its own, which is what makes a striped counter cheap to bump: no
/// two workers share a cell, so nothing is contended and nothing has to be atomic.
/// Combining happens once, and turns the counts into disjoint write positions -- after
/// which the scatter needs no coordination between workers at all.
fn block_offsets(
    bounds: &[usize],
    n_buckets: usize,
    key: impl Fn(usize) -> usize + Sync,
) -> (Vec<usize>, Vec<usize>) {
    let mut table = vec![0usize; (bounds.len() - 1) * n_buckets];
    table
        .par_chunks_mut(n_buckets)
        .enumerate()
        .for_each(|(b, counts)| {
            for k in bounds[b]..bounds[b + 1] {
                counts[key(k)] += 1;
            }
        });

    // Combine the stripes: the total per bucket, then where the bucket starts, then each
    // block's slice of it. Both passes walk the table forwards, and the only scattered
    // access is into the per-bucket array, which is small enough to stay in cache.
    let mut start = vec![0usize; n_buckets + 1];
    for counts in table.chunks(n_buckets) {
        for (bucket, &count) in counts.iter().enumerate() {
            start[bucket + 1] += count;
        }
    }
    for bucket in 0..n_buckets {
        start[bucket + 1] += start[bucket];
    }
    let mut next = start.clone();
    for counts in table.chunks_mut(n_buckets) {
        for (bucket, count) in counts.iter_mut().enumerate() {
            let at = next[bucket];
            next[bucket] += *count;
            *count = at;
        }
    }
    (table, start)
}

/// Entries several workers scatter into at once, each to positions of its own.
///
/// The positions are disjoint by construction, so these stores need no ordering against
/// one another and `Relaxed` compiles to an ordinary store. They are atomic only because
/// that is what makes one buffer writable from several threads without `unsafe`.
struct Scattered {
    keys: Vec<AtomicUsize>,
    values: Vec<AtomicU64>,
}

impl Scattered {
    fn new(len: usize) -> Self {
        Self {
            keys: (0..len).map(|_| AtomicUsize::new(0)).collect(),
            values: (0..len).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    fn put(&self, at: usize, key: usize, value: f64) {
        self.keys[at].store(key, Ordering::Relaxed);
        self.values[at].store(value.to_bits(), Ordering::Relaxed);
    }

    fn key(&self, at: usize) -> usize {
        self.keys[at].load(Ordering::Relaxed)
    }

    fn value(&self, at: usize) -> f64 {
        f64::from_bits(self.values[at].load(Ordering::Relaxed))
    }
}

/// Split `0..len` into `n` blocks of consecutive entries, `n` of them at most.
fn even_blocks(len: usize, n: usize) -> Vec<usize> {
    let block = len.div_ceil(n.max(1)).max(MIN_BLOCK);
    let mut bounds: Vec<usize> = (0..len.div_ceil(block)).map(|b| b * block).collect();
    bounds.push(len);
    bounds
}

/// Split the buckets of `start` into `n` groups holding about as many entries each.
///
/// The second counting sort walks its input in bucket order rather than in index order,
/// so its blocks have to be whole buckets; they are cut by entry count, because that is
/// what the work is proportional to.
fn bucket_blocks(start: &[usize], n: usize) -> Vec<usize> {
    let total = *start.last().expect("start");
    let mut buckets = vec![0usize];
    for g in 1..n.max(1) {
        let at = start.partition_point(|&s| s < total * g / n.max(1));
        if at > *buckets.last().expect("bucket") {
            buckets.push(at);
        }
    }
    buckets.push(start.len() - 1);
    buckets
}

/// Build a CSR matrix from coordinate triples, summing duplicates.
///
/// Two counting sorts, by column and then by row, leave each row's columns ascending
/// without comparing anything -- scipy reaches the same canonical form by sorting each
/// row -- and one merge pass adds up the entries a pair occurs in more than once.
/// Explicit zeros are kept, so a weight of zero still marks the pair as observed.
///
/// Both sorts count into striped per-block tables and then scatter in parallel, which is
/// what the passes cost: they are scattered writes over the whole matrix, and one thread
/// spends most of its time waiting for memory.
pub fn coo_to_csr(
    rows: &[i64],
    cols: &[i64],
    data: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> CsrOwned {
    let nnz = data.len();
    if nnz == 0 || n_rows == 0 || n_cols == 0 {
        return CsrOwned {
            indptr: vec![0; n_rows + 1],
            indices: Vec::new(),
            data: Vec::new(),
        };
    }
    let workers = rayon::current_num_threads().max(1);

    // By column, in blocks of the input.
    let bounds = even_blocks(nnz, workers);
    let (mut cursor, column_start) = block_offsets(&bounds, n_cols, |k| cols[k] as usize);
    let by_column = Scattered::new(nnz);
    cursor
        .par_chunks_mut(n_cols)
        .enumerate()
        .for_each(|(b, next)| {
            for k in bounds[b]..bounds[b + 1] {
                let at = &mut next[cols[k] as usize];
                by_column.put(*at, rows[k] as usize, data[k]);
                *at += 1;
            }
        });

    // By row, in blocks of whole columns, so that visiting the entries in column order
    // keeps this pass stable and every row comes out sorted by column.
    let groups = bucket_blocks(&column_start, workers);
    let bounds: Vec<usize> = groups.iter().map(|&c| column_start[c]).collect();
    let (mut cursor, row_start) = block_offsets(&bounds, n_rows, |k| by_column.key(k));
    let by_row = Scattered::new(nnz);
    cursor
        .par_chunks_mut(n_rows)
        .enumerate()
        .for_each(|(b, next)| {
            for column in groups[b]..groups[b + 1] {
                for p in column_start[column]..column_start[column + 1] {
                    let at = &mut next[by_column.key(p)];
                    by_row.put(*at, column, by_column.value(p));
                    *at += 1;
                }
            }
        });

    // Merging what is left is the one pass that cannot be split by entry, because a row
    // only knows where its output starts once every row before it has been merged. So it
    // is counted first, in parallel, and then written in parallel into the slices that
    // the counts hand out.
    let groups = bucket_blocks(&row_start, workers);
    let mut kept = vec![0usize; n_rows];
    slice_by(&mut kept, &groups)
        .into_par_iter()
        .enumerate()
        .for_each(|(g, counts)| {
            for (r, count) in counts.iter_mut().enumerate() {
                *count = distinct(&by_row, &row_start, groups[g] + r);
            }
        });

    let mut indptr = Vec::with_capacity(n_rows + 1);
    indptr.push(0);
    for &count in &kept {
        indptr.push(indptr[indptr.len() - 1] + count);
    }
    let total = indptr[n_rows];
    let mut indices = vec![0usize; total];
    let mut values = vec![0.0f64; total];
    let cuts: Vec<usize> = groups.iter().map(|&r| indptr[r]).collect();
    slice_by(&mut indices, &cuts)
        .into_par_iter()
        .zip(slice_by(&mut values, &cuts))
        .enumerate()
        .for_each(|(g, (out_indices, out_values))| {
            let mut at = 0;
            for r in groups[g]..groups[g + 1] {
                for p in row_start[r]..row_start[r + 1] {
                    let column = by_row.key(p);
                    if at > 0 && p > row_start[r] && out_indices[at - 1] == column {
                        out_values[at - 1] += by_row.value(p);
                    } else {
                        out_indices[at] = column;
                        out_values[at] = by_row.value(p);
                        at += 1;
                    }
                }
            }
        });
    CsrOwned {
        indptr,
        indices,
        data: values,
    }
}

/// How many entries row `r` keeps once its repeated columns are merged.
fn distinct(by_row: &Scattered, row_start: &[usize], r: usize) -> usize {
    let mut count = 0;
    let mut previous = usize::MAX;
    for p in row_start[r]..row_start[r + 1] {
        let column = by_row.key(p);
        if column != previous {
            count += 1;
            previous = column;
        }
    }
    count
}

/// Cut `values` into the consecutive pieces `cuts` describes, one per block.
fn slice_by<'a, T>(values: &'a mut [T], cuts: &[usize]) -> Vec<&'a mut [T]> {
    let mut rest = values;
    let mut pieces = Vec::with_capacity(cuts.len() - 1);
    for window in cuts.windows(2) {
        let (head, tail) = rest.split_at_mut(window[1] - window[0]);
        pieces.push(head);
        rest = tail;
    }
    pieces
}

#[cfg(test)]
mod csr_tests {
    use super::*;

    /// The dense matrix a set of triples adds up to.
    fn dense(rows: &[i64], cols: &[i64], data: &[f64], n_rows: usize, n_cols: usize) -> Vec<f64> {
        let mut out = vec![0.0; n_rows * n_cols];
        for k in 0..data.len() {
            out[rows[k] as usize * n_cols + cols[k] as usize] += data[k];
        }
        out
    }

    fn check(rows: &[i64], cols: &[i64], data: &[f64], n_rows: usize, n_cols: usize) {
        let m = coo_to_csr(rows, cols, data, n_rows, n_cols);
        let want = dense(rows, cols, data, n_rows, n_cols);
        assert_eq!(m.indptr.len(), n_rows + 1);
        assert_eq!(*m.indptr.last().expect("indptr"), m.indices.len());
        for r in 0..n_rows {
            let row = m.indptr[r]..m.indptr[r + 1];
            let columns: Vec<usize> = m.indices[row.clone()].to_vec();
            assert!(
                columns.windows(2).all(|c| c[0] < c[1]),
                "row {r} not canonical"
            );
            for p in row {
                let value = want[r * n_cols + m.indices[p]];
                assert!((m.data[p] - value).abs() < 1e-12, "({r}, {})", m.indices[p]);
            }
        }
        // Everything nonzero in the dense form is stored.
        let stored: usize = (0..n_rows).map(|r| m.indptr[r + 1] - m.indptr[r]).sum();
        let distinct: std::collections::HashSet<(i64, i64)> =
            rows.iter().copied().zip(cols.iter().copied()).collect();
        assert_eq!(stored, distinct.len());
    }

    #[test]
    fn sums_duplicates_and_sorts_each_row() {
        check(
            &[0, 1, 0, 1, 0],
            &[2, 0, 2, 1, 0],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            2,
            3,
        );
    }

    #[test]
    fn keeps_explicit_zeros_and_empty_rows() {
        let m = coo_to_csr(&[2], &[1], &[0.0], 4, 3);
        assert_eq!(m.indptr, vec![0, 0, 0, 1, 1]);
        assert_eq!((m.indices, m.data), (vec![1], vec![0.0]));
    }

    #[test]
    fn handles_an_empty_matrix() {
        let m = coo_to_csr(&[], &[], &[], 3, 2);
        assert_eq!(m.indptr, vec![0, 0, 0, 0]);
        assert!(m.indices.is_empty());
    }

    #[test]
    fn matches_the_dense_sum_on_a_larger_case() {
        let (n_rows, n_cols) = (37, 23);
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for _ in 0..4_000 {
            rows.push((next() % n_rows as u64) as i64);
            cols.push((next() % n_cols as u64) as i64);
            data.push((next() % 100) as f64 / 10.0);
        }
        check(&rows, &cols, &data, n_rows, n_cols);
    }
}
