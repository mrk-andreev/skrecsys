//! Item-item nearest neighbours: the rows of `W^T W` pruned to their top `k` entries.
//!
//! A port of `all_pairs_knn` from implicit (benfred/implicit 0.7.3,
//! `_nearest_neighbours.pyx` and `nearest_neighbours.h`). Each row is accumulated with
//! the SMMP sparse-product accumulator and pruned with the same top-k rule, so the
//! kept neighbours, including the choice among tied scores, match implicit.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::sparse::{Accumulator, Csc, Csr, CsrOwned};

/// Return the top `k` entries of every row of `w^T w`, where `w` has users as rows
/// and items as columns. Rows are computed in parallel on the current rayon pool.
pub fn all_pairs_top_k(w: &Csr, k: usize) -> CsrOwned {
    let items = Csc::from_csr(w);
    let rows: Vec<Vec<(usize, f64)>> = (0..w.n_cols)
        .into_par_iter()
        .map_init(
            || Accumulator::new(w.n_cols),
            |acc, i| {
                for (u, w1) in items.column(i) {
                    for p in w.indptr[u]..w.indptr[u + 1] {
                        acc.add(w.indices[p], w.data[p] * w1);
                    }
                }
                drain_top_k(acc, k)
            },
        )
        .collect();
    CsrOwned::from_rows(rows)
}

/// Select the top `k` accumulated entries, sorted by index, and reset the accumulator.
///
/// implicit walks a linked list whose head is the most recently touched index, so
/// candidates arrive in reverse first-touch order. A candidate enters when fewer than
/// `k` are kept or its score beats the smallest kept score; the evicted entry is the
/// smallest `(score, index)` pair.
fn drain_top_k(acc: &mut Accumulator, k: usize) -> Vec<(usize, f64)> {
    let mut heap: BinaryHeap<Reverse<Entry>> = BinaryHeap::with_capacity(k + 1);
    for &index in acc.touched().iter().rev() {
        let score = acc.get(index);
        let admit = match heap.peek() {
            _ if heap.len() < k => true,
            Some(Reverse(min)) => score > min.score,
            None => false,
        };
        if admit {
            if heap.len() >= k {
                heap.pop();
            }
            heap.push(Reverse(Entry { score, index }));
        }
    }
    acc.reset();

    let mut kept: Vec<(usize, f64)> = heap
        .into_iter()
        .map(|Reverse(e)| (e.index, e.score))
        .collect();
    kept.sort_unstable_by_key(|&(index, _)| index);
    kept
}

/// `std::pair<double, int>` ordering: by score, then by index.
#[derive(Clone, Copy)]
struct Entry {
    score: f64,
    index: usize,
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then(self.index.cmp(&other.index))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Entry {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::testing::{Owned, pseudo_random_dense, row};

    #[test]
    fn matches_dense_product_without_ties() {
        let dense = pseudo_random_dense(30, 12);
        let w = Owned::from_dense(&dense);
        let k = 4;
        let result = all_pairs_top_k(&w.csr(), k);
        for i in 0..12 {
            let mut scores: Vec<(usize, f64)> = (0..12)
                .map(|j| (j, dense.iter().map(|r| r[i] * r[j]).sum::<f64>()))
                .filter(|&(_, s)| s != 0.0)
                .collect();
            scores.sort_by(|a, b| b.1.total_cmp(&a.1));
            scores.truncate(k);
            scores.sort_by_key(|&(j, _)| j);
            let got = row(&result, i);
            assert_eq!(got.len(), scores.len(), "row {i}");
            for ((gj, gs), (ej, es)) in got.iter().zip(&scores) {
                assert_eq!(gj, ej, "row {i}");
                assert!((gs - es).abs() < 1e-9, "row {i}: {gs} vs {es}");
            }
        }
    }

    #[test]
    fn ties_keep_what_implicit_keeps() {
        // Item 0 co-occurs equally with items 0, 1 and 2, touched in that order.
        // Candidates arrive as 2, 1, 0; later equal scores do not displace item 2.
        let w = Owned::from_dense(&[vec![1.0, 1.0, 1.0]]);
        let result = all_pairs_top_k(&w.csr(), 1);
        assert_eq!(row(&result, 0), vec![(2, 1.0)]);
        let result = all_pairs_top_k(&w.csr(), 2);
        assert_eq!(row(&result, 0), vec![(1, 1.0), (2, 1.0)]);
    }

    #[test]
    fn thread_count_does_not_change_result() {
        let dense = pseudo_random_dense(200, 80);
        let w = Owned::from_dense(&dense);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| all_pairs_top_k(&w.csr(), 7))
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a.indptr, b.indptr);
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.data, b.data);
    }
}
