//! Prune a sparse matrix to the `k` largest entries of each row.
//!
//! Unlike the kernels that build a similarity matrix and prune as they accumulate, this
//! one takes a matrix that already exists, which is what a model needs when it has to
//! rescale its similarities before pruning them a second time.
//!
//! Selection reproduces `numpy.argsort(-row, kind="stable")[:k]`: the largest values
//! first, ties resolved by the position an entry occupies in its row. Explicit zeros are
//! ordinary candidates rather than something to drop, so a pruned matrix keeps whatever
//! stored structure survives the cut.

use crate::sparse::{Csr, CsrOwned};

/// The `k` largest entries of every row of `m`, each row sorted by column index.
///
/// Rows are independent and run in parallel on the current rayon pool.
pub fn top_k_per_row(m: &Csr, k: usize) -> CsrOwned {
    CsrOwned::build(
        m.n_rows,
        // Scratch for the rows that have to be cut down, reused across a block of them.
        Vec::<(usize, usize, f64)>::new,
        |candidates, i, out| {
            let (start, end) = (m.indptr[i], m.indptr[i + 1]);
            let before = out.len();
            if end - start <= k {
                out.extend((start..end).map(|p| (m.col(p), m.data[p])));
            } else {
                // Carry the position so that ties resolve the way a stable sort left
                // them, whatever order the column indices happen to arrive in.
                candidates.clear();
                candidates.extend((start..end).map(|p| (p, m.col(p), m.data[p])));
                // Partition alone puts the `k` best first, in no particular order.
                candidates
                    .select_nth_unstable_by(k, |a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
                out.extend(candidates[..k].iter().map(|&(_, j, v)| (j, v)));
            }
            out[before..].sort_unstable_by_key(|&(j, _)| j);
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::testing::{Owned, pseudo_random_dense, row};

    /// What `keep_top_k_per_row` did in Python, straight from its definition.
    fn reference(dense: &[Vec<f64>], k: usize) -> Vec<Vec<(usize, f64)>> {
        dense
            .iter()
            .map(|r| {
                let mut stored: Vec<(usize, f64)> = r
                    .iter()
                    .enumerate()
                    .filter(|&(_, &v)| v != 0.0)
                    .map(|(j, &v)| (j, v))
                    .collect();
                if stored.len() > k {
                    // Stable by descending value, then by position, then take k.
                    stored = {
                        let mut indexed: Vec<(usize, (usize, f64))> =
                            stored.into_iter().enumerate().collect();
                        indexed.sort_by(|a, b| b.1.1.total_cmp(&a.1.1).then(a.0.cmp(&b.0)));
                        indexed.truncate(k);
                        let mut kept: Vec<(usize, f64)> =
                            indexed.into_iter().map(|(_, e)| e).collect();
                        kept.sort_by_key(|&(j, _)| j);
                        kept
                    };
                }
                stored
            })
            .collect()
    }

    #[test]
    fn matches_the_python_selection() {
        let dense = pseudo_random_dense(30, 20);
        let m = Owned::from_dense(&dense);
        for k in [1usize, 3, 7, 50] {
            let got = top_k_per_row(&m.csr(), k);
            let want = reference(&dense, k);
            for (i, want_row) in want.iter().enumerate() {
                assert_eq!(&row(&got, i), want_row, "row {i}, k={k}");
            }
        }
    }

    #[test]
    fn keeps_a_short_row_whole() {
        let m = Owned::from_dense(&[vec![3.0, 0.0, 1.0]]);
        assert_eq!(
            row(&top_k_per_row(&m.csr(), 5), 0),
            vec![(0, 3.0), (2, 1.0)]
        );
    }

    #[test]
    fn ties_keep_the_earlier_position() {
        // All three tie, so the first k positions win, which for a sorted row is the
        // lowest column indices.
        let m = Owned::from_dense(&[vec![1.0, 1.0, 1.0]]);
        assert_eq!(
            row(&top_k_per_row(&m.csr(), 2), 0),
            vec![(0, 1.0), (1, 1.0)]
        );
    }

    #[test]
    fn an_explicit_zero_is_an_ordinary_candidate() {
        // A stored zero outranks a negative neighbour and must survive the cut.
        let indptr = vec![0usize, 2];
        let indices = vec![0i64, 1];
        let data = vec![-1.0, 0.0];
        let m = Csr {
            n_rows: 1,
            n_cols: 2,
            indptr: &indptr,
            indices: &indices,
            data: &data,
        };
        assert_eq!(row(&top_k_per_row(&m, 1), 0), vec![(1, 0.0)]);
    }

    #[test]
    fn thread_count_does_not_change_the_result() {
        let dense = pseudo_random_dense(300, 60);
        let m = Owned::from_dense(&dense);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| top_k_per_row(&m.csr(), 9))
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a.indptr, b.indptr);
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.data, b.data);
    }
}
