//! Factorization machine fitted by alternating least squares.
//!
//! A port of libFM's ALS learner (`fm_learn_mcmc_simultaneous` with sampling
//! disabled): S. Rendle, Z. Gantner, C. Freudenthaler, and L. Schmidt-Thieme, "Fast
//! Context-aware Recommendations with Factorization Machines", SIGIR 2011.
//!
//! Every parameter is set in turn to the minimizer of the regularized squared loss
//! with all other parameters held fixed. The residuals `e[c] = y_hat(c) - y[c]` and,
//! per factor, the inner products `q[c] = sum_j v[f][j] * x[c][j]` are cached and
//! updated incrementally, so one update costs O(nnz of its column).

use crate::sparse::{Csc, Csr};

/// Model parameters and regularization for [`fit`].
///
/// `v` holds `n_factors` rows of `n_features` values; `group[j]` indexes `reg_w` and
/// `reg_v` for feature `j`.
pub struct Model<'a> {
    pub w0: f64,
    pub w: &'a mut [f64],
    pub v: &'a mut [f64],
    pub group: &'a [usize],
    pub reg0: f64,
    pub reg_w: &'a [f64],
    pub reg_v: &'a [f64],
}

/// Run `n_iter` ALS sweeps in place and return the training RMSE after each sweep.
pub fn fit(x: &Csr, y: &[f64], model: &mut Model, n_iter: usize) -> Vec<f64> {
    let n_features = x.n_cols;
    let n_factors = model.v.len().checked_div(n_features).unwrap_or(0);
    let csc = Csc::from_csr(x);

    let mut e: Vec<f64> = (0..x.n_rows)
        .map(|c| predict_row(x, model, n_factors, c) - y[c])
        .collect();
    let mut q = vec![0.0; x.n_rows];
    let mut history = Vec::with_capacity(n_iter);

    for _ in 0..n_iter {
        update_w0(&mut e, model);
        for j in 0..n_features {
            update_w(&csc, &mut e, model, j);
        }
        for f in 0..n_factors {
            let v_f = &mut model.v[f * n_features..(f + 1) * n_features];
            for (c, q_c) in q.iter_mut().enumerate() {
                *q_c = (x.indptr[c]..x.indptr[c + 1])
                    .map(|k| v_f[x.indices[k]] * x.data[k])
                    .sum();
            }
            for j in 0..n_features {
                update_v(&csc, &mut e, &mut q, v_f, j, model.reg_v[model.group[j]]);
            }
        }
        history.push(rmse(&e));
    }
    history
}

fn predict_row(x: &Csr, model: &Model, n_factors: usize, c: usize) -> f64 {
    let n_features = x.n_cols;
    let row = x.indptr[c]..x.indptr[c + 1];
    let mut pred = model.w0;
    for k in row.clone() {
        pred += model.w[x.indices[k]] * x.data[k];
    }
    for f in 0..n_factors {
        let v_f = &model.v[f * n_features..(f + 1) * n_features];
        let mut sum = 0.0;
        let mut sum_sqr = 0.0;
        for k in row.clone() {
            let d = v_f[x.indices[k]] * x.data[k];
            sum += d;
            sum_sqr += d * d;
        }
        pred += 0.5 * (sum * sum - sum_sqr);
    }
    pred
}

fn update_w0(e: &mut [f64], model: &mut Model) {
    let denominator = model.reg0 + e.len() as f64;
    if denominator == 0.0 {
        return;
    }
    let mean: f64 = e.iter().map(|&e_c| e_c - model.w0).sum();
    let new = -mean / denominator;
    let delta = new - model.w0;
    for e_c in e.iter_mut() {
        *e_c += delta;
    }
    model.w0 = new;
}

fn update_w(csc: &Csc, e: &mut [f64], model: &mut Model, j: usize) {
    let old = model.w[j];
    let mut mean = 0.0;
    let mut sigma_sqr = 0.0;
    for (c, x) in csc.column(j) {
        mean += x * (e[c] - old * x);
        sigma_sqr += x * x;
    }
    let denominator = sigma_sqr + model.reg_w[model.group[j]];
    // libFM resets a parameter with an unbounded posterior variance to 0.
    let new = if denominator == 0.0 {
        0.0
    } else {
        -mean / denominator
    };
    let delta = new - old;
    for (c, x) in csc.column(j) {
        e[c] += delta * x;
    }
    model.w[j] = new;
}

fn update_v(csc: &Csc, e: &mut [f64], q: &mut [f64], v_f: &mut [f64], j: usize, reg: f64) {
    let old = v_f[j];
    let mut mean = 0.0;
    let mut sigma_sqr = 0.0;
    for (c, x) in csc.column(j) {
        let h = x * q[c] - x * x * old;
        mean += h * e[c];
        sigma_sqr += h * h;
    }
    mean -= old * sigma_sqr;
    let denominator = sigma_sqr + reg;
    // libFM resets a parameter with an unbounded posterior variance to 0.
    let new = if denominator == 0.0 {
        0.0
    } else {
        -mean / denominator
    };
    let delta = new - old;
    for (c, x) in csc.column(j) {
        let h = x * q[c] - x * x * old;
        e[c] += h * delta;
        q[c] += x * delta;
    }
    v_f[j] = new;
}

fn rmse(e: &[f64]) -> f64 {
    if e.is_empty() {
        return 0.0;
    }
    (e.iter().map(|e_c| e_c * e_c).sum::<f64>() / e.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    type Design = (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>, Vec<usize>);

    /// One-hot user-item design: users are features `0..3`, items `3..6`.
    fn one_hot() -> Design {
        let pairs = [
            (0, 0, 5.0),
            (0, 1, 3.0),
            (1, 1, 4.0),
            (1, 2, 1.0),
            (2, 0, 4.0),
            (2, 2, 2.0),
        ];
        let mut indptr = vec![0];
        let mut indices = Vec::new();
        for &(u, i, _) in &pairs {
            indices.extend([u, 3 + i]);
            indptr.push(indices.len());
        }
        let data = vec![1.0; indices.len()];
        let y = pairs.iter().map(|p| p.2).collect();
        (indptr, indices, data, y, vec![0, 0, 0, 1, 1, 1])
    }

    fn run(n_factors: usize, n_iter: usize, reg: f64) -> (f64, Vec<f64>, Vec<f64>) {
        let (indptr, indices, data, y, group) = one_hot();
        let x = Csr {
            n_rows: y.len(),
            n_cols: 6,
            indptr: &indptr,
            indices: &indices,
            data: &data,
        };
        let mut w = vec![0.0; 6];
        let mut v: Vec<f64> = (0..n_factors * 6)
            .map(|k| ((k * 7 % 11) as f64 - 5.0) / 50.0)
            .collect();
        let mut model = Model {
            w0: 0.0,
            w: &mut w,
            v: &mut v,
            group: &group,
            reg0: reg,
            reg_w: &[reg, reg],
            reg_v: &[reg, reg],
        };
        let history = fit(&x, &y, &mut model, n_iter);
        let w0 = model.w0;
        (w0, w, history)
    }

    #[test]
    fn biases_only_converges_to_ridge_solution() {
        let reg = 0.5;
        let (w0, w, _) = run(0, 2000, reg);
        let (indptr, indices, _, y, _) = one_hot();
        // Gradient of sum_c (y_hat - y)^2 + reg * ||theta||^2 vanishes at the optimum.
        let resid: Vec<f64> = (0..y.len())
            .map(|c| {
                w0 + indices[indptr[c]..indptr[c + 1]]
                    .iter()
                    .map(|&j| w[j])
                    .sum::<f64>()
                    - y[c]
            })
            .collect();
        assert!((resid.iter().sum::<f64>() + reg * w0).abs() < 1e-9);
        for (j, &w_j) in w.iter().enumerate() {
            let grad: f64 = (0..y.len())
                .filter(|&c| indices[indptr[c]..indptr[c + 1]].contains(&j))
                .map(|c| resid[c])
                .sum::<f64>()
                + reg * w_j;
            assert!(grad.abs() < 1e-9, "feature {j}: gradient {grad}");
        }
    }

    #[test]
    fn rmse_never_increases_without_regularization() {
        // Each update minimizes the loss, which is the squared RMSE when reg is 0.
        let (_, _, history) = run(3, 50, 0.0);
        assert_eq!(history.len(), 50);
        for pair in history.windows(2) {
            assert!(pair[1] <= pair[0] + 1e-12, "{pair:?}");
        }
    }
}
