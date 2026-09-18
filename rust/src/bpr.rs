//! Bayesian personalized ranking for implicit feedback, fitted by stochastic gradient
//! ascent over sampled triplets (S. Rendle, C. Freudenthaler, Z. Gantner, and L.
//! Schmidt-Thieme, "BPR: Bayesian Personalized Ranking from Implicit Feedback", UAI
//! 2009).
//!
//! A triplet is a user `u`, an item `i` they interacted with, and an item `j` they did
//! not. The model scores an item as `b_i + <p_u, q_i>` and maximizes the log-likelihood
//! of ranking `i` above `j`,
//!
//! `log sigma(x_uij) - lambda (||p_u||^2 + ||q_i||^2 + ||q_j||^2)`, where
//! `x_uij = b_i - b_j + <p_u, q_i - q_j>`,
//!
//! by taking one gradient step per sampled triplet. This is a port of Cornac's `BPR`
//! (`cornac/models/bpr/recom_bpr.pyx`), which follows implicit's: one epoch draws as
//! many triplets as there are interactions, the positive is a uniformly drawn stored
//! entry, the negative a uniformly drawn item, and a draw whose negative turns out to be
//! one of the user's own items is skipped rather than resampled. Interaction values are
//! ignored; only which entries are stored matters.
//!
//! Threads update the factors without locking, so two samples that touch the same row
//! race (Hogwild!, F. Niu et al., NIPS 2011), exactly as in the reference. Which
//! triplets are drawn is not affected: the draw is a pure function of the seed, the
//! epoch and the sample number, so it does not depend on how the work is scheduled.

use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::sparse::Csr;

/// Hyper-parameters of the fit, in the reference's parameterization.
pub struct Hyper {
    pub n_factors: usize,
    pub learning_rate: f64,
    pub regularization: f64,
    pub use_bias: bool,
    pub n_iter: usize,
    pub seed: u64,
}

/// Model parameters, updated in place. `user_factors` holds `n_users` rows of
/// `n_factors` values, `item_factors` and `item_bias` one entry per item.
pub struct Model<'a> {
    pub user_factors: &'a mut [f64],
    pub item_factors: &'a mut [f64],
    pub item_bias: &'a mut [f64],
}

/// Run `n_iter` epochs of SGD in place on the current rayon pool.
///
/// `r` has users as rows and items as columns and must have sorted column indices.
/// Returns, per epoch, the share of the triplets it used that the model already ranked
/// the right way round, which is the reference's `correct` statistic and estimates the
/// training AUC.
pub fn fit(r: &Csr, model: &mut Model, hyper: &Hyper) -> Vec<f64> {
    let n_samples = r.indices.len();
    let user_of = user_of_each_entry(r);
    let users = Shared::new(model.user_factors);
    let items = Shared::new(model.item_factors);
    let biases = Shared::new(model.item_bias);

    let history = (0..hyper.n_iter)
        .map(|epoch| {
            let (correct, skipped) = (0..n_samples)
                .into_par_iter()
                .with_min_len(256)
                .map(|s| step(r, &user_of, &users, &items, &biases, hyper, epoch, s))
                .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
            let used = n_samples - skipped;
            if used == 0 {
                0.0
            } else {
                correct as f64 / used as f64
            }
        })
        .collect();

    users.write_back(model.user_factors);
    items.write_back(model.item_factors);
    biases.write_back(model.item_bias);
    history
}

/// One triplet: draw it, and unless it is skipped, take a gradient step.
///
/// Returns `(correct, skipped)`, both 0 or 1.
#[allow(clippy::too_many_arguments)]
fn step(
    r: &Csr,
    user_of: &[usize],
    users: &Shared,
    items: &Shared,
    biases: &Shared,
    hyper: &Hyper,
    epoch: usize,
    s: usize,
) -> (usize, usize) {
    let n_samples = r.indices.len();
    let (first, second) = draw(hyper.seed, (epoch * n_samples + s) as u64);
    let entry = (first % n_samples as u64) as usize;
    let u = user_of[entry];
    let i = r.col(entry);
    let j = (second % r.n_cols as u64) as usize;
    // The reference skips a negative the user has interacted with rather than redraw it.
    if r.indices[r.indptr[u]..r.indptr[u + 1]]
        .binary_search(&(j as i64))
        .is_ok()
    {
        return (0, 1);
    }

    let k = hyper.n_factors;
    let (u0, i0, j0) = (u * k, i * k, j * k);
    let mut score = if hyper.use_bias {
        biases.get(i) - biases.get(j)
    } else {
        0.0
    };
    for f in 0..k {
        score += users.get(u0 + f) * (items.get(i0 + f) - items.get(j0 + f));
    }
    // sigma(-x_uij): the gradient weight, largest where the pair is ranked worst.
    let z = 1.0 / (1.0 + score.exp());

    let (lr, reg) = (hyper.learning_rate, hyper.regularization);
    for f in 0..k {
        let (p_f, i_f, j_f) = (users.get(u0 + f), items.get(i0 + f), items.get(j0 + f));
        users.set(u0 + f, p_f + lr * (z * (i_f - j_f) - reg * p_f));
        items.set(i0 + f, i_f + lr * (z * p_f - reg * i_f));
        items.set(j0 + f, j_f + lr * (-z * p_f - reg * j_f));
    }
    if hyper.use_bias {
        let (b_i, b_j) = (biases.get(i), biases.get(j));
        biases.set(i, b_i + lr * (z - reg * b_i));
        biases.set(j, b_j + lr * (-z - reg * b_j));
    }
    (usize::from(z < 0.5), 0)
}

/// The user each stored entry belongs to, that is the `row` of the COO form.
fn user_of_each_entry(r: &Csr) -> Vec<usize> {
    let mut user_of = vec![0; r.indices.len()];
    for u in 0..r.n_rows {
        user_of[r.indptr[u]..r.indptr[u + 1]].fill(u);
    }
    user_of
}

/// splitmix64's finalizer: a bijection of `u64` with good avalanche.
fn mix(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The two uniform draws of sample number `counter`: the stored entry that provides the
/// user and the positive item, and the candidate negative item.
///
/// Counter-based rather than sequential, so a sample's triplet is the same whichever
/// thread happens to run it.
fn draw(seed: u64, counter: u64) -> (u64, u64) {
    let counter = counter.wrapping_mul(2);
    (
        mix(seed ^ mix(counter)),
        mix(seed ^ mix(counter.wrapping_add(1))),
    )
}

/// Parameters shared by the SGD threads, read and written without locking.
///
/// Relaxed atomics give the reference's Hogwild! semantics without undefined behaviour:
/// an update may be lost when two threads touch the same row, but no value is ever torn.
struct Shared(Vec<AtomicU64>);

impl Shared {
    fn new(values: &[f64]) -> Self {
        Self(
            values
                .iter()
                .map(|&v| AtomicU64::new(v.to_bits()))
                .collect(),
        )
    }

    fn get(&self, index: usize) -> f64 {
        f64::from_bits(self.0[index].load(Ordering::Relaxed))
    }

    fn set(&self, index: usize, value: f64) {
        self.0[index].store(value.to_bits(), Ordering::Relaxed);
    }

    fn write_back(&self, out: &mut [f64]) {
        for (value, slot) in out.iter_mut().zip(&self.0) {
            *value = f64::from_bits(slot.load(Ordering::Relaxed));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::testing::{Owned, pseudo_random_dense};

    struct Fitted {
        user_factors: Vec<f64>,
        item_factors: Vec<f64>,
        item_bias: Vec<f64>,
        history: Vec<f64>,
    }

    fn hyper(n_factors: usize, n_iter: usize) -> Hyper {
        Hyper {
            n_factors,
            learning_rate: 0.05,
            regularization: 0.01,
            use_bias: true,
            n_iter,
            seed: 7,
        }
    }

    /// Start from the reference's initialization, `(uniform - 0.5) / n_factors`, with a
    /// reproducible stand-in for the uniform draws.
    fn initial(n: usize, k: usize, offset: u64) -> Vec<f64> {
        (0..n * k)
            .map(|t| {
                let u = (mix(offset + t as u64) >> 11) as f64 / (1u64 << 53) as f64;
                (u - 0.5) / k as f64
            })
            .collect()
    }

    fn fit_dense(dense: &[Vec<f64>], hyper: &Hyper) -> Fitted {
        let owned = Owned::from_dense(dense);
        let r = owned.csr();
        let (n_users, n_items) = (r.n_rows, r.n_cols);
        let mut user_factors = initial(n_users, hyper.n_factors, 1);
        let mut item_factors = initial(n_items, hyper.n_factors, 99);
        let mut item_bias = vec![0.0; n_items];
        let history = fit(
            &r,
            &mut Model {
                user_factors: &mut user_factors,
                item_factors: &mut item_factors,
                item_bias: &mut item_bias,
            },
            hyper,
        );
        Fitted {
            user_factors,
            item_factors,
            item_bias,
            history,
        }
    }

    fn score(fitted: &Fitted, k: usize, u: usize, i: usize) -> f64 {
        fitted.item_bias[i]
            + (0..k)
                .map(|f| fitted.user_factors[u * k + f] * fitted.item_factors[i * k + f])
                .sum::<f64>()
    }

    /// Share of (observed, unobserved) item pairs the model ranks the right way round.
    fn auc(dense: &[Vec<f64>], fitted: &Fitted, k: usize) -> f64 {
        let (mut right, mut total) = (0.0, 0.0);
        for (u, row) in dense.iter().enumerate() {
            for (i, &r_ui) in row.iter().enumerate() {
                for (j, &r_uj) in row.iter().enumerate() {
                    if r_ui != 0.0 && r_uj == 0.0 {
                        total += 1.0;
                        if score(fitted, k, u, i) > score(fitted, k, u, j) {
                            right += 1.0;
                        }
                    }
                }
            }
        }
        right / total
    }

    #[test]
    fn training_ranks_observed_items_above_unobserved_ones() {
        let dense = pseudo_random_dense(60, 25);
        let hyper = hyper(16, 200);
        let fitted = fit_dense(&dense, &hyper);
        assert!(auc(&dense, &fitted, hyper.n_factors) > 0.9);
    }

    #[test]
    fn the_share_of_correctly_ranked_triplets_rises() {
        let dense = pseudo_random_dense(60, 25);
        let hyper = hyper(16, 200);
        let fitted = fit_dense(&dense, &hyper);
        assert_eq!(fitted.history.len(), hyper.n_iter);
        assert!(fitted.history[0] < 0.6, "{:?}", fitted.history[0]);
        assert!(
            fitted.history[hyper.n_iter - 1] > 0.9,
            "{:?}",
            fitted.history
        );
    }

    #[test]
    fn one_sample_takes_the_gradient_step_of_the_reference() {
        // One user and two items, so the only triplet that is not skipped is (0, 0, 1).
        let dense = vec![vec![1.0, 0.0]];
        let hyper = Hyper {
            n_factors: 2,
            learning_rate: 0.1,
            regularization: 0.05,
            use_bias: true,
            n_iter: 1,
            // Chosen so that the single draw picks item 1, the unobserved one, as the
            // negative; any other seed would make the sample a skip.
            seed: 2,
        };
        let before_user = initial(1, 2, 1);
        let before_item = initial(2, 2, 99);
        let fitted = fit_dense(&dense, &hyper);
        assert_eq!(draw(hyper.seed, 0).1 % 2, 1);
        let (lr, reg) = (hyper.learning_rate, hyper.regularization);
        let score: f64 = (0..2)
            .map(|f| before_user[f] * (before_item[f] - before_item[2 + f]))
            .sum();
        let z = 1.0 / (1.0 + score.exp());
        for f in 0..2 {
            let (p, q_i, q_j) = (before_user[f], before_item[f], before_item[2 + f]);
            let expect = |value: f64, gradient: f64| value + lr * (gradient - reg * value);
            assert!((fitted.user_factors[f] - expect(p, z * (q_i - q_j))).abs() < 1e-15);
            assert!((fitted.item_factors[f] - expect(q_i, z * p)).abs() < 1e-15);
            assert!((fitted.item_factors[2 + f] - expect(q_j, -z * p)).abs() < 1e-15);
        }
        assert!((fitted.item_bias[0] - lr * z).abs() < 1e-15);
        assert!((fitted.item_bias[1] + lr * z).abs() < 1e-15);
    }

    #[test]
    fn a_seeded_fit_is_reproducible() {
        let dense = pseudo_random_dense(40, 20);
        let first = fit_dense(&dense, &hyper(8, 30));
        let second = fit_dense(&dense, &hyper(8, 30));
        assert_eq!(first.user_factors, second.user_factors);
        assert_eq!(first.item_factors, second.item_factors);
        assert_eq!(first.item_bias, second.item_bias);
    }

    #[test]
    fn another_seed_draws_another_triplet_stream() {
        let dense = pseudo_random_dense(40, 20);
        let other = Hyper {
            seed: 8,
            ..hyper(8, 30)
        };
        let first = fit_dense(&dense, &hyper(8, 30));
        let second = fit_dense(&dense, &other);
        assert_ne!(first.user_factors, second.user_factors);
    }

    #[test]
    fn without_epochs_the_parameters_do_not_move() {
        let dense = pseudo_random_dense(40, 20);
        let fitted = fit_dense(&dense, &hyper(8, 0));
        assert_eq!(fitted.user_factors, initial(40, 8, 1));
        assert!(fitted.history.is_empty());
    }

    #[test]
    fn the_item_biases_stay_zero_without_them() {
        let dense = pseudo_random_dense(40, 20);
        let hyper = Hyper {
            use_bias: false,
            ..hyper(8, 20)
        };
        let fitted = fit_dense(&dense, &hyper);
        assert!(fitted.item_bias.iter().all(|&b| b == 0.0));
    }

    #[test]
    fn stronger_regularization_shrinks_the_factors() {
        let dense = pseudo_random_dense(60, 25);
        let norm = |values: &[f64]| values.iter().map(|v| v * v).sum::<f64>();
        let weak = fit_dense(&dense, &hyper(16, 100));
        let strong = fit_dense(
            &dense,
            &Hyper {
                regularization: 0.5,
                ..hyper(16, 100)
            },
        );
        assert!(norm(&strong.user_factors) < norm(&weak.user_factors));
        assert!(norm(&strong.item_factors) < norm(&weak.item_factors));
    }

    #[test]
    fn an_item_nobody_skipped_is_still_a_candidate_negative() {
        // Every user interacted with item 0, so it is only ever drawn as a skipped
        // negative; the fit must still terminate and leave its factors trained as a
        // positive.
        let dense = vec![vec![1.0, 1.0, 0.0], vec![1.0, 0.0, 1.0]];
        let fitted = fit_dense(&dense, &hyper(4, 50));
        assert!(fitted.item_bias[0] != 0.0);
    }
}
