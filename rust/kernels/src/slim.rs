//! SLIM with elastic-net weights: one sparse linear regression per item, fitted by
//! coordinate descent (X. Ning and G. Karypis, "SLIM: Sparse Linear Methods for Top-N
//! Recommender Systems", ICDM 2011; M. Levy and K. Jack, "Efficient Top-N
//! Recommendation by Linear Regression", LSRS 2013).
//!
//! Item `j` is regressed on every other item, `min_w 1/(2 m) ||r_j - R_-j w||^2 +
//! alpha l1_ratio ||w||_1 + alpha (1 - l1_ratio) / 2 ||w||^2` over `m` users, with `w`
//! usually constrained to be non-negative. Column `j` of the weight matrix is that
//! solution, and the diagonal is zero by construction.
//!
//! The reference implementation (M. Ferrari Dacrema et al.,
//! `SLIM_ElasticNet/SLIMElasticNetRecommender.py`) hands each column to scikit-learn's
//! `ElasticNet` with the whole interaction matrix as the design matrix, which re-reads
//! `R` once per item. Every solve only ever touches `R` through `R^T R`, so this kernel
//! forms the item Gram matrix once, shared with EASE, and runs the same coordinate
//! descent scikit-learn runs when `precompute=True`: for target `j` the subproblem is
//! `min_w 1/2 w^T G w - G_j^T w + l1 ||w||_1 + l2 / 2 ||w||^2` with row and column `j`
//! of `G` left out. Zeroing the target column, as the reference does, leaves `G[j][j]`
//! as its only trace, and that is `||r_j||^2`, which the stopping rule needs.
//!
//! Columns are solved in parallel and pruned to their `k` largest weights. Memory is
//! quadratic in the number of items, like EASE.

use crate::ease::gram;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::sparse::{Csr, CsrOwned};
use crate::vectors::axpy;

/// The per-item elastic net, in scikit-learn's parameterization.
pub struct ElasticNet {
    pub alpha: f64,
    pub l1_ratio: f64,
    pub positive: bool,
    pub max_iter: usize,
    pub tol: f64,
}

/// Per-item elastic-net weights for a users x items CSR matrix `r`, pruned to `k`.
///
/// Row `j` of the result holds the `k` largest weights of the regression that predicts
/// item `j`, that is column `j` of the SLIM weight matrix; entry `j` itself is never
/// stored. Rows are solved in parallel on the current rayon pool, sorted by item index,
/// and exact zeros are dropped. Returns the number of columns that hit `max_iter`
/// without converging alongside the weights.
pub fn similarity(r: &Csr, enet: &ElasticNet, k: usize) -> (CsrOwned, usize) {
    let g = gram(r);
    similarity_from_gram(&g, r.n_cols, r.n_rows as f64, None, None, enet, k)
}

/// [`similarity`] from an item Gram matrix that is already in hand.
///
/// `g` is the dense row-major `n x n` [`gram`] and `m` the number of users, which the
/// penalties scale by. `columns` restricts the solve to those targets, in that order,
/// and `warm` gives each of them a starting point: row `t` of `warm` holds the
/// coefficients to start target `columns[t]` from. Both `None` reproduces
/// [`similarity`] exactly.
///
/// An incremental fit uses all three: it keeps the Gram matrix and updates it by the
/// rank the batch has, re-solves only the columns the batch can reach, and starts each
/// of those from the solution it already had. Coordinate descent stops at a duality
/// gap rather than at the exact optimum, so a warm start lands at a different point
/// inside the same tolerance -- and the columns that are not re-solved keep an optimum
/// that has moved. This is an approximation of a full fit and does not claim otherwise.
pub fn similarity_from_gram(
    g: &[f64],
    n: usize,
    m: f64,
    columns: Option<&[usize]>,
    warm: Option<&Csr>,
    enet: &ElasticNet,
    k: usize,
) -> (CsrOwned, usize) {
    // scikit-learn folds the `1 / (2 m)` of the least-squares term into the penalties.
    let l1 = enet.alpha * enet.l1_ratio * m;
    let l2 = enet.alpha * (1.0 - enet.l1_ratio) * m;

    let unconverged = AtomicUsize::new(0);
    let init = || Solver::new(n);
    let solve = |solver: &mut Solver, at: usize, j: usize, out: &mut Vec<(usize, f64)>| {
        let start = warm.map_or(0..0, |w| w.indptr[at]..w.indptr[at + 1]);
        let warm_row = warm.map(|w| (&w.indices[start.clone()], &w.data[start]));
        if !solver.solve(g, n, j, l1, l2, enet, warm_row) {
            unconverged.fetch_add(1, Ordering::Relaxed);
        }
        top_k(&solver.w, j, k, out);
    };
    let weights = match columns {
        Some(columns) => CsrOwned::build_rows(columns, init, |solver, j, out| {
            let at = columns.binary_search(&j).expect("a row that was asked for");
            solve(solver, at, j, out);
        }),
        None => CsrOwned::build(n, init, |solver, j, out| solve(solver, j, j, out)),
    };
    (weights, unconverged.into_inner())
}

/// Reusable buffers for one column's coordinate descent.
struct Solver {
    /// Coefficients of the current column.
    w: Vec<f64>,
    /// `G w`, kept up to date so that a coordinate update costs one row of `G`.
    h: Vec<f64>,
}

impl Solver {
    fn new(n: usize) -> Self {
        Self {
            w: vec![0.0; n],
            h: vec![0.0; n],
        }
    }

    /// Fit column `target`, leaving the coefficients in `self.w`; `true` if it converged.
    ///
    /// Cyclic coordinate descent with the soft-thresholding update and the duality-gap
    /// stopping rule of scikit-learn's `enet_coordinate_descent_gram`. The reference
    /// leaves `selection='random'` on, which visits the same coordinates in a random
    /// order; both converge to the same optimum of a strictly convex problem.
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &mut self,
        g: &[f64],
        n: usize,
        target: usize,
        l1: f64,
        l2: f64,
        enet: &ElasticNet,
        warm: Option<(&[i64], &[f64])>,
    ) -> bool {
        self.w.fill(0.0);
        self.h.fill(0.0);
        if let Some((indices, values)) = warm {
            // A coordinate the descent will not visit must not keep a warm value, or it
            // would survive into the output without ever having been solved for.
            for (&ii, &value) in indices.iter().zip(values) {
                let ii = ii as usize;
                if ii != target && g[ii * n + ii] != 0.0 {
                    self.w[ii] = value;
                }
            }
            for &ii in indices {
                let ii = ii as usize;
                if self.w[ii] != 0.0 {
                    axpy(&mut self.h, &g[ii * n..(ii + 1) * n], self.w[ii]);
                }
            }
        }
        // `G` is symmetric, so its column `target` is the contiguous row `target`.
        let q = &g[target * n..(target + 1) * n];
        let y_norm2 = q[target];
        if y_norm2 == 0.0 {
            // An item nobody interacted with: the target is the zero vector.
            return true;
        }

        for iteration in 0..enet.max_iter {
            let (mut w_max, mut d_w_max) = (0.0_f64, 0.0_f64);
            for ii in 0..n {
                let g_ii = g[ii * n + ii];
                if ii == target || g_ii == 0.0 {
                    continue;
                }
                let row = &g[ii * n..(ii + 1) * n];
                let w_ii = self.w[ii];
                if w_ii != 0.0 {
                    axpy(&mut self.h, row, -w_ii);
                }
                let rho = q[ii] - self.h[ii];
                self.w[ii] = if enet.positive && rho <= 0.0 {
                    0.0
                } else {
                    rho.signum() * (rho.abs() - l1).max(0.0) / (g_ii + l2)
                };
                if self.w[ii] != 0.0 {
                    axpy(&mut self.h, row, self.w[ii]);
                }
                d_w_max = d_w_max.max((self.w[ii] - w_ii).abs());
                w_max = w_max.max(self.w[ii].abs());
            }
            let last = iteration + 1 == enet.max_iter;
            if w_max == 0.0 || d_w_max / w_max < enet.tol || last {
                let gap = self.duality_gap(q, target, l1, l2, y_norm2, enet.positive);
                if gap < enet.tol * y_norm2 {
                    return true;
                }
                if last {
                    return false;
                }
            }
        }
        true
    }

    /// The elastic-net duality gap at the current `w`, as scikit-learn computes it.
    ///
    /// The dual point is the residual rescaled to be feasible, `const = l1 /
    /// max(|G w - q|)` when that exceeds one, and the gap is the difference between the
    /// primal objective and the resulting dual objective. Index `target` is skipped
    /// throughout: the reference zeroes that column of the design matrix, which zeroes
    /// its entry of `q` and of `G w` together.
    fn duality_gap(
        &self,
        q: &[f64],
        target: usize,
        l1: f64,
        l2: f64,
        y_norm2: f64,
        positive: bool,
    ) -> f64 {
        let (mut q_dot_w, mut h_dot_w, mut w_norm1, mut w_norm2) = (0.0, 0.0, 0.0, 0.0);
        let mut dual_norm = 0.0_f64;
        for (i, (&q_i, (&h_i, &w_i))) in q.iter().zip(self.h.iter().zip(&self.w)).enumerate() {
            if i == target {
                continue;
            }
            q_dot_w += q_i * w_i;
            h_dot_w += h_i * w_i;
            w_norm1 += w_i.abs();
            w_norm2 += w_i * w_i;
            let xta = q_i - h_i - l2 * w_i;
            dual_norm = dual_norm.max(if positive { xta } else { xta.abs() });
        }

        let r_norm2 = y_norm2 + h_dot_w - 2.0 * q_dot_w;
        let (scale, mut gap) = if dual_norm > l1 {
            let scale = l1 / dual_norm;
            (scale, 0.5 * (r_norm2 + r_norm2 * scale * scale))
        } else {
            (1.0, r_norm2)
        };
        gap += l1 * w_norm1 - scale * y_norm2 + scale * q_dot_w;
        gap + 0.5 * l2 * (1.0 + scale * scale) * w_norm2
    }
}

/// The `k` largest weights of a fitted column, sorted by item index.
///
/// Ranks by descending weight and breaks ties by ascending item index; the reference
/// ranks by descending weight too, with an unstable sort.
fn top_k(w: &[f64], target: usize, k: usize, out: &mut Vec<(usize, f64)>) {
    let before = out.len();
    out.extend(
        w.iter()
            .enumerate()
            .filter(|&(i, &value)| i != target && value != 0.0)
            .map(|(i, &value)| (i, value)),
    );
    if out.len() - before > k {
        crate::knn::keep_best(out, before, k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Warm starting must not change where a tight solve lands: the problem is strictly
    /// convex, so both starting points converge on the same optimum.
    #[test]
    fn a_warm_start_reaches_the_same_optimum() {
        use crate::sparse::testing::{Owned, pseudo_random_dense, row};
        let dense = pseudo_random_dense(60, 14);
        let r = Owned::from_dense(&dense);
        let enet = ElasticNet {
            alpha: 0.05,
            l1_ratio: 0.5,
            positive: true,
            max_iter: 2000,
            tol: 1e-12,
        };
        let (cold, _) = similarity(&r.csr(), &enet, 14);
        let g = gram(&r.csr());
        let wanted = [2usize, 5, 9];
        let warm = Owned::from_rows(
            wanted.iter().map(|&j| row(&cold, j)).collect::<Vec<_>>(),
            14,
        );
        let (warmed, _) =
            similarity_from_gram(&g, 14, 60.0, Some(&wanted), Some(&warm.csr()), &enet, 14);
        for (at, &j) in wanted.iter().enumerate() {
            let (got, want) = (row(&warmed, at), row(&cold, j));
            assert_eq!(got.len(), want.len(), "column {j}");
            for ((gj, gv), (wj, wv)) in got.iter().zip(&want) {
                assert_eq!(gj, wj, "column {j}");
                assert!((gv - wv).abs() < 1e-8, "column {j}: {gv} vs {wv}");
            }
        }
    }

    /// Asking for every column with no warm start is the unrestricted kernel, which is
    /// the guarantee that lets `fit` and `partial_fit` share one solver.
    #[test]
    fn every_column_cold_is_the_unrestricted_fit() {
        use crate::sparse::testing::{Owned, pseudo_random_dense};
        let dense = pseudo_random_dense(40, 11);
        let r = Owned::from_dense(&dense);
        let enet = ElasticNet {
            alpha: 0.05,
            l1_ratio: 0.5,
            positive: true,
            max_iter: 100,
            tol: 1e-6,
        };
        let all: Vec<usize> = (0..11).collect();
        let g = gram(&r.csr());
        let (full, a) = similarity(&r.csr(), &enet, 11);
        let (part, b) = similarity_from_gram(&g, 11, 40.0, Some(&all), None, &enet, 11);
        assert_eq!(
            (full.indptr, full.indices, full.data, a),
            (part.indptr, part.indices, part.data, b)
        );
    }

    use crate::sparse::testing::{Owned, pseudo_random_dense, row};

    fn enet(alpha: f64, l1_ratio: f64, positive: bool) -> ElasticNet {
        ElasticNet {
            alpha,
            l1_ratio,
            positive,
            max_iter: 1000,
            tol: 1e-10,
        }
    }

    /// `(G w - q)_i`, the gradient of the least-squares part of the subproblem.
    fn gradient(g: &[f64], n: usize, target: usize, w: &[f64], i: usize) -> f64 {
        let dot: f64 = (0..n)
            .filter(|&c| c != target)
            .map(|c| g[i * n + c] * w[c])
            .sum();
        dot - g[target * n + i]
    }

    /// Assert the first-order optimality conditions of the elastic net at `w`.
    ///
    /// The subproblem is strictly convex, so they pin the solution without needing a
    /// second implementation to compare against: a nonzero coefficient sits where the
    /// gradient balances the penalties, and a zero one where the `l1` term dominates.
    fn assert_optimal(
        g: &[f64],
        n: usize,
        target: usize,
        w: &[f64],
        l1: f64,
        l2: f64,
        positive: bool,
    ) {
        for i in (0..n).filter(|&i| i != target && g[i * n + i] != 0.0) {
            let grad = gradient(g, n, target, w, i) + l2 * w[i];
            if w[i] != 0.0 {
                assert!(w[i] > 0.0 || !positive, "coefficient {i} is negative");
                let residual = grad + l1 * w[i].signum();
                assert!(residual.abs() < 1e-6, "coefficient {i}: {residual}");
            } else if positive {
                assert!(grad + l1 > -1e-6, "zero coefficient {i}: {grad}");
            } else {
                assert!(grad.abs() < l1 + 1e-6, "zero coefficient {i}: {grad}");
            }
        }
    }

    fn fit(dense: &[Vec<f64>], enet: &ElasticNet, k: usize) -> (CsrOwned, Vec<f64>, usize) {
        let owned = Owned::from_dense(dense);
        let (weights, unconverged) = similarity(&owned.csr(), enet, k);
        (weights, gram(&owned.csr()), unconverged)
    }

    #[test]
    fn every_column_solves_its_elastic_net() {
        let dense = pseudo_random_dense(60, 14);
        let (n, m) = (14, 60.0);
        let params = enet(0.05, 0.3, true);
        let (weights, g, unconverged) = fit(&dense, &params, n);
        assert_eq!(unconverged, 0);
        let (l1, l2) = (params.alpha * params.l1_ratio * m, params.alpha * 0.7 * m);

        for target in 0..n {
            let mut w = vec![0.0; n];
            for (i, value) in row(&weights, target) {
                assert_ne!(i, target, "the diagonal must stay zero");
                w[i] = value;
            }
            assert_optimal(&g, n, target, &w, l1, l2, true);
        }
    }

    #[test]
    fn signed_weights_solve_the_unconstrained_problem() {
        let dense = pseudo_random_dense(50, 12);
        let (n, m) = (12, 50.0);
        let params = enet(0.02, 0.5, false);
        let (weights, g, _) = fit(&dense, &params, n);
        let (l1, l2) = (params.alpha * params.l1_ratio * m, params.alpha * 0.5 * m);

        let mut negatives = 0;
        for target in 0..n {
            let mut w = vec![0.0; n];
            for (i, value) in row(&weights, target) {
                w[i] = value;
                negatives += usize::from(value < 0.0);
            }
            assert_optimal(&g, n, target, &w, l1, l2, false);
        }
        assert!(
            negatives > 0,
            "the unconstrained fit found no negative weight"
        );
    }

    #[test]
    fn a_larger_alpha_shrinks_the_weights() {
        let dense = pseudo_random_dense(80, 15);
        let sum = |alpha| {
            let (weights, _, _) = fit(&dense, &enet(alpha, 0.3, true), 15);
            (weights.data.iter().sum::<f64>(), weights.data.len())
        };
        let (weak, weak_nnz) = sum(0.005);
        let (strong, strong_nnz) = sum(0.5);
        assert!(strong < weak, "{strong} vs {weak}");
        assert!(strong_nnz < weak_nnz, "{strong_nnz} vs {weak_nnz}");
    }

    #[test]
    fn keeps_the_k_largest_weights_of_each_column() {
        let dense = pseudo_random_dense(70, 16);
        let params = enet(0.005, 0.2, true);
        let (full, _, _) = fit(&dense, &params, 16);
        let k = 3;
        let (pruned, _, _) = fit(&dense, &params, k);
        for target in 0..16 {
            let kept = row(&pruned, target);
            assert!(kept.len() <= k, "column {target} kept {}", kept.len());
            let smallest = kept.iter().map(|&(_, v)| v).fold(f64::MAX, f64::min);
            for (i, value) in row(&full, target) {
                if !kept.iter().any(|&(c, _)| c == i) {
                    assert!(value <= smallest, "column {target} dropped {value}");
                }
            }
        }
    }

    #[test]
    fn reports_columns_that_hit_the_iteration_limit() {
        let dense = pseudo_random_dense(60, 14);
        let params = ElasticNet {
            alpha: 1e-4,
            l1_ratio: 0.1,
            positive: true,
            max_iter: 1,
            tol: 1e-12,
        };
        let (_, _, unconverged) = fit(&dense, &params, 14);
        assert!(unconverged > 0);
    }

    #[test]
    fn an_item_without_interactions_has_no_weights() {
        // Item 2 is in the catalog but nobody interacted with it.
        let dense = vec![vec![1.0, 1.0, 0.0], vec![1.0, 2.0, 0.0]];
        let (weights, _, unconverged) = fit(&dense, &enet(0.001, 0.5, true), 3);
        assert_eq!(unconverged, 0);
        assert_eq!(row(&weights, 2), vec![]);
        // Nor can it be a neighbour of anything else.
        assert!(!weights.indices.contains(&2));
    }

    #[test]
    fn thread_count_does_not_change_the_result() {
        let dense = pseudo_random_dense(200, 40);
        let owned = Owned::from_dense(&dense);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| similarity(&owned.csr(), &enet(0.01, 0.3, true), 7).0)
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a.indptr, b.indptr);
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.data, b.data);
    }
}
