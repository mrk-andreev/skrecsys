//! EASE: the closed-form linear item-item model of H. Steck, "Embarrassingly Shallow
//! Autoencoders for Sparse Data", WWW 2019.
//!
//! With `G = R^T R` the item Gram matrix and `A = G + l2_reg * I`, the minimizer of the
//! squared reconstruction error under a zero-diagonal constraint is
//! `B = I - A^-1 diag(1 / diag(A^-1))`, that is `B[i][j] = -P[i][j] / P[j][j]` with
//! `P = A^-1`, and `B[j][j] = 0`.
//!
//! `A` is symmetric positive definite whenever `l2_reg > 0`, so it is factored by
//! Cholesky rather than inverted generally: fewer operations and better conditioning for
//! the same answer. The reference implementations
//! (`microsoft/UniRec`, `unirec/model/cf/ease.py` and `MTSWebServices/RecTools`,
//! `rectools/models/ease.py`) call `numpy.linalg.inv` instead, so the two agree only up
//! to rounding.

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::llt;
use faer::{Mat, Par};
use rayon::prelude::*;

use crate::sparse::{Csc, Csr};

/// `A = G + l2_reg * I` was not positive definite, so it has no Cholesky factor.
pub struct NotPositiveDefinite;

/// Dense row-major `w^T w`, of shape `n_cols x n_cols`.
///
/// Rows are independent and the output is dense, so each thread writes one row of the
/// result directly; no sparse accumulator is needed.
pub fn gram(w: &Csr) -> Vec<f64> {
    let n = w.n_cols;
    let items = Csc::from_csr(w);
    let mut g = vec![0.0; n * n];
    g.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for (u, w1) in items.column(i) {
            for p in w.indptr[u]..w.indptr[u + 1] {
                row[w.col(p)] += w.data[p] * w1;
            }
        }
    });
    g
}

/// Item-item weights `B` from a dense row-major Gram matrix, as described above.
///
/// `gram` is `n x n` and symmetric; it is left untouched. The returned matrix is dense
/// row-major with a zero diagonal.
pub fn weights(
    gram: &[f64],
    n: usize,
    l2_reg: f64,
    par: Par,
) -> Result<Vec<f64>, NotPositiveDefinite> {
    if n == 0 {
        return Ok(Vec::new());
    }
    // Symmetric, so row-major and faer's column-major layout agree.
    let mut a = Mat::from_fn(n, n, |i, j| {
        gram[i * n + j] + if i == j { l2_reg } else { 0.0 }
    });

    let params = Default::default();
    let mut buffer = MemBuffer::new(StackReq::any_of(&[
        llt::factor::cholesky_in_place_scratch::<f64>(n, par, params),
        llt::inverse::inverse_scratch::<f64>(n, par),
    ]));
    let stack = MemStack::new(&mut buffer);
    llt::factor::cholesky_in_place(a.as_mut(), Default::default(), par, stack, params)
        .map_err(|_| NotPositiveDefinite)?;

    let mut p = Mat::<f64>::zeros(n, n);
    llt::inverse::inverse(p.as_mut(), a.as_ref(), par, stack);

    // `inverse` fills the lower triangle only; the inverse of a symmetric matrix is
    // symmetric, so the rest is a mirror image.
    let mut b = vec![0.0; n * n];
    b.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for (j, value) in row.iter_mut().enumerate() {
            if i != j {
                let p_ij = if i > j { p[(i, j)] } else { p[(j, i)] };
                *value = -p_ij / p[(j, j)];
            }
        }
    });
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::testing::{Owned, pseudo_random_dense};

    /// Gauss-Jordan inverse, the general-purpose route the reference implementations take.
    fn naive_inverse(a: &[f64], n: usize) -> Vec<f64> {
        let mut m: Vec<f64> = (0..n)
            .flat_map(|i| {
                let row = a[i * n..(i + 1) * n].to_vec();
                let eye = (0..n).map(move |j| if i == j { 1.0 } else { 0.0 });
                row.into_iter().chain(eye)
            })
            .collect();
        let width = 2 * n;
        for col in 0..n {
            let pivot = (col..n)
                .max_by(|&x, &y| {
                    m[x * width + col]
                        .abs()
                        .total_cmp(&m[y * width + col].abs())
                })
                .expect("non-empty");
            for j in 0..width {
                m.swap(col * width + j, pivot * width + j);
            }
            let d = m[col * width + col];
            for j in 0..width {
                m[col * width + j] /= d;
            }
            for row in 0..n {
                if row != col {
                    let factor = m[row * width + col];
                    for j in 0..width {
                        m[row * width + j] -= factor * m[col * width + j];
                    }
                }
            }
        }
        (0..n)
            .flat_map(|i| m[i * width + n..(i + 1) * width].to_vec())
            .collect()
    }

    #[test]
    fn gram_matches_a_dense_product() {
        let dense = pseudo_random_dense(25, 9);
        let w = Owned::from_dense(&dense);
        let got = gram(&w.csr());
        for i in 0..9 {
            for j in 0..9 {
                let want: f64 = dense.iter().map(|r| r[i] * r[j]).sum();
                assert!((got[i * 9 + j] - want).abs() < 1e-9, "({i}, {j})");
            }
        }
    }

    #[test]
    fn matches_an_explicit_inverse() {
        let dense = pseudo_random_dense(40, 11);
        let g = gram(&Owned::from_dense(&dense).csr());
        let n = 11;
        let l2_reg = 3.0;
        let got = weights(&g, n, l2_reg, Par::Seq).unwrap_or_else(|_| panic!("not definite"));

        let a: Vec<f64> = (0..n * n)
            .map(|k| g[k] + if k % (n + 1) == 0 { l2_reg } else { 0.0 })
            .collect();
        let p = naive_inverse(&a, n);
        for i in 0..n {
            for j in 0..n {
                let want = if i == j {
                    0.0
                } else {
                    -p[i * n + j] / p[j * n + j]
                };
                assert!((got[i * n + j] - want).abs() < 1e-9, "({i}, {j})");
            }
        }
    }

    #[test]
    fn satisfies_the_constrained_optimum() {
        // At the optimum `A B = A - diag(gamma)` with `gamma > 0` and a zero diagonal on
        // `B`, which pins the solution without ever forming an inverse.
        let dense = pseudo_random_dense(60, 14);
        let g = gram(&Owned::from_dense(&dense).csr());
        let (n, l2_reg) = (14, 7.5);
        let b = weights(&g, n, l2_reg, Par::Seq).unwrap_or_else(|_| panic!("not definite"));
        let a: Vec<f64> = (0..n * n)
            .map(|k| g[k] + if k % (n + 1) == 0 { l2_reg } else { 0.0 })
            .collect();

        for j in 0..n {
            assert_eq!(b[j * n + j], 0.0, "diagonal {j}");
        }
        for i in 0..n {
            for j in 0..n {
                let ab: f64 = (0..n).map(|k| a[i * n + k] * b[k * n + j]).sum();
                let residual = a[i * n + j] - ab;
                if i == j {
                    assert!(residual > 0.0, "gamma {j} = {residual}");
                } else {
                    assert!(residual.abs() < 1e-6, "({i}, {j}): {residual}");
                }
            }
        }
    }

    #[test]
    fn thread_count_does_not_change_the_result() {
        let dense = pseudo_random_dense(200, 60);
        let g = gram(&Owned::from_dense(&dense).csr());
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool")
                .install(|| weights(&g, 60, 12.0, Par::rayon(0)).unwrap_or_else(|_| unreachable!()))
        };
        let (a, b) = (run(1), run(4));
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-9, "{x} vs {y}");
        }
    }

    #[test]
    fn rejects_a_singular_gram_matrix() {
        // Two identical items make the Gram matrix rank deficient without regularization.
        let w = Owned::from_dense(&[vec![1.0, 1.0], vec![2.0, 2.0]]);
        let g = gram(&w.csr());
        assert!(weights(&g, 2, 0.0, Par::Seq).is_err());
    }
}
