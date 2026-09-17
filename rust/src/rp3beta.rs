//! RP3beta: item-item similarity from a three-step random walk on the user-item graph,
//! damped by item popularity.
//!
//! The walk `item -> user -> item` gives `W = Piu Pui`, where `Pui` is the row-normalized
//! interaction matrix and `Piu` the row-normalized binary transpose, both raised to the
//! power `alpha`. RP3beta then divides column `j` by `popularity[j]^beta`
//! (B. Paudel et al., "Updatable, Accurate, Diverse, and Scalable Recommendations for
//! Interactive Applications", TiiS 2017; F. Christoffel et al., "Blockbusters and
//! Wallflowers", RecSys 2015).
//!
//! Every entry of row `i` of `Piu^alpha` is `popularity[i]^-alpha`, because a binary row
//! normalizes to a constant. The walk therefore needs only `Pui^alpha` and two scaling
//! vectors, and this kernel takes them already computed:
//!
//! `W[i][j] = row_scale[i] * col_scale[j] * sum over the users u of item i of pui[u][j]`
//!
//! Rows are accumulated sparsely and pruned to their `k` largest entries, which is the
//! dominant cost of the model. The reference implementation (M. Ferrari Dacrema et al.,
//! `GraphBased/RP3betaRecommender.py`) materializes blocks of 200 dense rows instead.

use rayon::prelude::*;

use crate::sparse::{Accumulator, Csc, Csr, CsrOwned};

/// Top `k` entries of every row of the scaled walk matrix, diagonal excluded.
///
/// `pui` has users as rows and items as columns; `row_scale` and `col_scale` are indexed
/// by item. Exact zeros are dropped, ties are broken by ascending item index, and rows
/// are computed in parallel on the current rayon pool.
pub fn similarity(pui: &Csr, row_scale: &[f64], col_scale: &[f64], k: usize) -> CsrOwned {
    let n = pui.n_cols;
    let items = Csc::from_csr(pui);
    let rows: Vec<Vec<(usize, f64)>> = (0..n)
        .into_par_iter()
        .map_init(
            || Accumulator::new(n),
            |acc, i| {
                for (u, _) in items.column(i) {
                    for p in pui.indptr[u]..pui.indptr[u + 1] {
                        acc.add(pui.indices[p], pui.data[p]);
                    }
                }
                let mut candidates: Vec<(usize, f64)> = acc
                    .touched()
                    .iter()
                    .filter(|&&j| j != i)
                    .map(|&j| (j, acc.get(j) * row_scale[i] * col_scale[j]))
                    .filter(|&(_, value)| value != 0.0)
                    .collect();
                acc.reset();

                if candidates.len() > k {
                    // Partition alone puts the `k` best first, in no particular order.
                    candidates
                        .select_nth_unstable_by(k, |a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                    candidates.truncate(k);
                }
                candidates.sort_unstable_by_key(|&(j, _)| j);
                candidates
            },
        )
        .collect();
    CsrOwned::from_rows(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::testing::{Owned, pseudo_random_dense, row};

    /// `W` before pruning, straight from the definition.
    fn dense_walk(dense: &[Vec<f64>], row_scale: &[f64], col_scale: &[f64]) -> Vec<Vec<f64>> {
        let n = dense[0].len();
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let walk: f64 = dense
                            .iter()
                            .filter(|user| user[i] != 0.0)
                            .map(|user| user[j])
                            .sum();
                        walk * row_scale[i] * col_scale[j]
                    })
                    .collect()
            })
            .collect()
    }

    fn scales(dense: &[Vec<f64>], alpha: f64, beta: f64) -> (Vec<f64>, Vec<f64>) {
        let n = dense[0].len();
        let popularity: Vec<f64> = (0..n)
            .map(|j| dense.iter().filter(|user| user[j] != 0.0).count() as f64)
            .collect();
        (
            popularity.iter().map(|&p| p.powf(-alpha)).collect(),
            popularity.iter().map(|&p| p.powf(-beta)).collect(),
        )
    }

    #[test]
    fn matches_the_dense_walk() {
        let dense = pseudo_random_dense(40, 10);
        let (row_scale, col_scale) = scales(&dense, 1.0, 0.6);
        let want = dense_walk(&dense, &row_scale, &col_scale);
        let got = similarity(&Owned::from_dense(&dense).csr(), &row_scale, &col_scale, 10);
        for (i, want_row) in want.iter().enumerate() {
            for (j, value) in row(&got, i) {
                assert!(i != j, "the diagonal must be dropped");
                assert!((value - want_row[j]).abs() < 1e-9, "({i}, {j})");
            }
        }
    }

    #[test]
    fn keeps_the_k_largest_entries_of_each_row() {
        let dense = pseudo_random_dense(60, 12);
        let (row_scale, col_scale) = scales(&dense, 1.0, 0.6);
        let want = dense_walk(&dense, &row_scale, &col_scale);
        let k = 3;
        let got = similarity(&Owned::from_dense(&dense).csr(), &row_scale, &col_scale, k);
        for (i, want_row) in want.iter().enumerate() {
            let kept = row(&got, i);
            assert!(kept.len() <= k, "row {i} kept {}", kept.len());
            let smallest_kept = kept.iter().map(|&(_, v)| v).fold(f64::MAX, f64::min);
            let dropped = (0..12)
                .filter(|&j| j != i && !kept.iter().any(|&(c, _)| c == j))
                .map(|j| want_row[j]);
            for value in dropped {
                assert!(value <= smallest_kept, "row {i} dropped {value}");
            }
        }
    }

    #[test]
    fn a_larger_beta_penalizes_popular_items_more() {
        let dense = pseudo_random_dense(80, 12);
        let (row_scale, weak) = scales(&dense, 1.0, 0.0);
        let (_, strong) = scales(&dense, 1.0, 1.5);
        let w = Owned::from_dense(&dense);
        let a = similarity(&w.csr(), &row_scale, &weak, 12);
        let b = similarity(&w.csr(), &row_scale, &strong, 12);
        // Column scaling is positive, so it cannot change which entries are nonzero.
        assert_eq!(a.indices, b.indices);
        let popularity: Vec<f64> = (0..12)
            .map(|j| dense.iter().filter(|user| user[j] != 0.0).count() as f64)
            .collect();
        let most_popular = (0..12).max_by(|&x, &y| popularity[x].total_cmp(&popularity[y]));
        let share = |m: &CsrOwned| {
            let target = most_popular.expect("non-empty");
            let hit: f64 = m
                .indices
                .iter()
                .zip(&m.data)
                .filter(|&(&j, _)| j == target)
                .map(|(_, v)| v)
                .sum();
            hit / m.data.iter().sum::<f64>()
        };
        assert!(share(&b) < share(&a));
    }

    #[test]
    fn thread_count_does_not_change_the_result() {
        let dense = pseudo_random_dense(200, 60);
        let w = Owned::from_dense(&dense);
        let (row_scale, col_scale) = scales(&dense, 0.7, 0.6);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| similarity(&w.csr(), &row_scale, &col_scale, 7))
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a.indptr, b.indptr);
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn an_item_without_neighbours_yields_an_empty_row() {
        // Item 2 shares no user with any other item.
        let dense = vec![vec![1.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let w = Owned::from_dense(&dense);
        let got = similarity(&w.csr(), &[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0], 5);
        assert_eq!(row(&got, 2), vec![]);
        assert_eq!(row(&got, 0), vec![(1, 1.0)]);
    }
}
