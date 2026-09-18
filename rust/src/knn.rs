//! Item-item nearest neighbours: the rows of `W^T W` pruned to their top `k` entries.
//!
//! A port of `all_pairs_knn` from implicit (benfred/implicit 0.7.3,
//! `_nearest_neighbours.pyx` and `nearest_neighbours.h`). Each row is accumulated with
//! the SMMP sparse-product accumulator and pruned with the same top-k rule, so the
//! kept neighbours, including the choice among tied scores, match implicit.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::sparse::{Accumulator, Csc, Csr, CsrOwned};

/// Return the top `k` entries of every row of `w^T w`, where `w` has users as rows
/// and items as columns. Rows are computed in parallel on the current rayon pool.
pub fn all_pairs_top_k(w: &Csr, k: usize) -> CsrOwned {
    let items = Csc::from_csr(w);
    CsrOwned::build(
        w.n_cols,
        || (Accumulator::new(w.n_cols), BinaryHeap::with_capacity(k + 1)),
        |(acc, heap), i, out| {
            for (u, w1) in items.column(i) {
                for p in w.indptr[u]..w.indptr[u + 1] {
                    acc.add(w.col(p), w.data[p] * w1);
                }
            }
            drain_top_k(acc, heap, k, out);
        },
    )
}

/// Top `k` cosine neighbours of every item, the diagonal excluded.
///
/// `w` has users as rows and items as columns and `norms[j]` is the Euclidean norm of
/// its column `j`, so entry `(i, j)` is `<w_i, w_j> / (norms[i] * norms[j] + shrink)`.
///
/// The shrinkage denominator does not factor into a per-row and a per-column scale, so
/// unlike [`crate::rp3beta::similarity`] this cannot be expressed as a scaled walk and
/// applies the divisor per pair. Pairs whose co-occurrence is exactly zero are dropped
/// before the division, which is also what keeps an item nobody interacted with -- whose
/// norm is zero -- from dividing by zero when `shrink` is zero.
///
/// Ties are broken by ascending item index, and rows are computed in parallel on the
/// current rayon pool.
pub fn cosine_top_k(w: &Csr, norms: &[f64], shrink: f64, k: usize) -> CsrOwned {
    let n = w.n_cols;
    let items = Csc::from_csr(w);
    CsrOwned::build(
        n,
        || Accumulator::new(n),
        |acc, i, out| {
            for (u, w1) in items.column(i) {
                for p in w.indptr[u]..w.indptr[u + 1] {
                    acc.add(w.col(p), w.data[p] * w1);
                }
            }
            // The candidates are gathered straight into the output buffer and cut down
            // in place, so a row of them never gets an allocation of its own.
            let before = out.len();
            out.extend(
                acc.touched()
                    .iter()
                    .filter(|&&j| j != i && acc.get(j) != 0.0)
                    .map(|&j| (j, acc.get(j) / (norms[i] * norms[j] + shrink))),
            );
            acc.reset();
            keep_best(out, before, k);
        },
    )
}

/// Cut `out[from..]` down to its `k` best entries, sorted by column.
///
/// Ties go to the lower column index, and partitioning alone leaves the `k` best first
/// in no particular order, so the kept entries are sorted afterwards.
pub(crate) fn keep_best(out: &mut Vec<(usize, f64)>, from: usize, k: usize) {
    if out.len() - from > k {
        out[from..].select_nth_unstable_by(k, |a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out.truncate(from + k);
    }
    out[from..].sort_unstable_by_key(|&(j, _)| j);
}

/// Select the top `k` accumulated entries, sorted by index, and reset the accumulator.
///
/// implicit walks a linked list whose head is the most recently touched index, so
/// candidates arrive in reverse first-touch order. A candidate enters when fewer than
/// `k` are kept or its score beats the smallest kept score; the evicted entry is the
/// smallest `(score, index)` pair.
fn drain_top_k(
    acc: &mut Accumulator,
    heap: &mut BinaryHeap<Reverse<Entry>>,
    k: usize,
    out: &mut Vec<(usize, f64)>,
) {
    heap.clear();
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

    let before = out.len();
    out.extend(heap.drain().map(|Reverse(e)| (e.index, e.score)));
    out[before..].sort_unstable_by_key(|&(index, _)| index);
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

    /// The dense cosine matrix the Python implementation built before this kernel: the
    /// co-occurrence divided by the shrunk product of the column norms, diagonal dropped.
    fn dense_cosine(dense: &[Vec<f64>], shrink: f64) -> Vec<Vec<f64>> {
        let n = dense[0].len();
        let norms: Vec<f64> = (0..n)
            .map(|j| dense.iter().map(|r| r[j] * r[j]).sum::<f64>().sqrt())
            .collect();
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let cooc: f64 = dense.iter().map(|r| r[i] * r[j]).sum();
                        if i == j || cooc == 0.0 {
                            0.0
                        } else {
                            cooc / (norms[i] * norms[j] + shrink)
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn norms_of(dense: &[Vec<f64>]) -> Vec<f64> {
        (0..dense[0].len())
            .map(|j| dense.iter().map(|r| r[j] * r[j]).sum::<f64>().sqrt())
            .collect()
    }

    #[test]
    fn cosine_matches_the_dense_definition() {
        let dense = pseudo_random_dense(40, 10);
        let norms = norms_of(&dense);
        for shrink in [0.0, 5.0] {
            let want = dense_cosine(&dense, shrink);
            let got = cosine_top_k(&Owned::from_dense(&dense).csr(), &norms, shrink, 10);
            for (i, want_row) in want.iter().enumerate() {
                for (j, value) in row(&got, i) {
                    assert!(i != j, "the diagonal must be dropped");
                    assert!(
                        (value - want_row[j]).abs() < 1e-9,
                        "({i}, {j}) shrink={shrink}"
                    );
                }
            }
        }
    }

    #[test]
    fn cosine_breaks_ties_by_ascending_index() {
        // Items 1 and 2 are indistinguishable from item 0's point of view, so the lower
        // index wins, which is what a stable argsort on descending score used to give.
        let w = Owned::from_dense(&[vec![1.0, 1.0, 1.0]]);
        let norms = vec![1.0, 1.0, 1.0];
        let got = cosine_top_k(&w.csr(), &norms, 0.0, 1);
        assert_eq!(row(&got, 0), vec![(1, 1.0)]);
    }

    #[test]
    fn cosine_leaves_an_unseen_item_empty() {
        // Item 2 has no interactions, so its norm is zero; it must not divide by zero.
        let dense = vec![vec![1.0, 1.0, 0.0], vec![1.0, 0.0, 0.0]];
        let norms = norms_of(&dense);
        assert_eq!(norms[2], 0.0);
        let got = cosine_top_k(&Owned::from_dense(&dense).csr(), &norms, 0.0, 5);
        assert_eq!(row(&got, 2), vec![]);
        for (_, value) in row(&got, 0) {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn cosine_thread_count_does_not_change_the_result() {
        let dense = pseudo_random_dense(200, 80);
        let w = Owned::from_dense(&dense);
        let norms = norms_of(&dense);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| cosine_top_k(&w.csr(), &norms, 1.5, 7))
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a.indptr, b.indptr);
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.data, b.data);
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
