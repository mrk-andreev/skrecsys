//! Hyperparameter sampling for `skrecsys.tune`: random search and the Tree-structured
//! Parzen Estimator.
//!
//! The sampler is univariate and stateless, as Optuna's default `TPESampler` is: every
//! call proposes one value of one parameter from the history of that parameter alone,
//! so the Python side keeps the trials and the search can be define-by-run.
//!
//! TPE [1] splits the finished trials into the best `gamma(n)` ("below") and the rest
//! ("above"), fits a Parzen estimator `l(x)` to the values of the first group and `g(x)`
//! to the second, draws candidates from `l` and keeps the one maximizing `l(x) / g(x)`,
//! which is proportional to the expected improvement. The details -- `gamma`, the
//! weights, the prior component and the bandwidth clipping -- follow Optuna's defaults.
//!
//! Values cross this module in the parameter's own space. Internally a log-scaled
//! parameter is searched in `ln x`, an integer in a range widened by half a step on each
//! side so both ends are as likely as any other value, and a categorical as the index of
//! its choice.
//!
//! [1] J. Bergstra, R. Bardenet, Y. Bengio and B. Kégl, "Algorithms for
//! Hyper-Parameter Optimization", `NeurIPS` 2011.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use crate::vectors::{BASELINE_FMA, madd};

/// The range of one hyperparameter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Distribution {
    /// A real number in `[low, high]`; `log` searches `ln x`, which needs `low > 0`.
    Float { low: f64, high: f64, log: bool },
    /// An integer in `[low, high]`, carried as `f64`; `log` needs `low >= 1`.
    Int { low: f64, high: f64, log: bool },
    /// One of `n_choices` choices, by index.
    Categorical { n_choices: usize },
}

/// Settings of [`suggest`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TpeConfig {
    /// Trials sampled at random before TPE takes over, so both groups have members.
    pub n_startup_trials: usize,
    /// Candidates drawn from `l(x)` per suggestion.
    pub n_ei_candidates: usize,
    /// Weight of the prior component of each Parzen estimator.
    pub prior_weight: f64,
}

impl Default for TpeConfig {
    fn default() -> Self {
        Self {
            n_startup_trials: 10,
            n_ei_candidates: 24,
            prior_weight: 1.0,
        }
    }
}

/// xoshiro256** seeded through splitmix64: small, fast and reproducible across platforms.
pub struct Rng([u64; 4]);

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let z = (state ^ (state >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        Self([next(), next(), next(), next()])
    }

    pub const fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)`, from the top 53 bits.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `[low, high)`.
    pub fn uniform_in(&mut self, low: f64, high: f64) -> f64 {
        madd::<BASELINE_FMA>(high - low, self.uniform(), low)
    }

    /// Standard normal, by Box-Muller.
    pub fn normal(&mut self) -> f64 {
        let u = 1.0 - self.uniform(); // (0, 1], so the logarithm is finite
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
    }

    /// A draw from `weights`, which must sum to a positive number.
    pub fn choice(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut target = self.uniform() * total;
        for (index, &w) in weights.iter().enumerate() {
            if target < w {
                return index;
            }
            target -= w;
        }
        weights.len() - 1
    }
}

/// A value drawn uniformly from `dist`, in the parameter's own space.
pub fn sample_random(dist: &Distribution, rng: &mut Rng) -> f64 {
    if let Distribution::Categorical { n_choices } = *dist {
        ((rng.uniform() * n_choices as f64) as usize).min(n_choices - 1) as f64
    } else {
        let (low, high) = internal_bounds(dist);
        to_external(dist, rng.uniform_in(low, high))
    }
}

/// Propose a value of one parameter from its history.
///
/// `observed[t]` is the value trial `t` used and `scores[t]` what it scored, in trial
/// order; higher is better. With fewer than `n_startup_trials` observations the value is
/// drawn at random.
///
/// # Panics
///
/// Panics if `observed` and `scores` have different lengths.
pub fn suggest(
    dist: &Distribution,
    observed: &[f64],
    scores: &[f64],
    config: &TpeConfig,
    seed: u64,
) -> f64 {
    assert_eq!(observed.len(), scores.len());
    let mut rng = Rng::new(seed);
    let n = observed.len();
    if n < config.n_startup_trials.max(1) || config.n_ei_candidates == 0 {
        return sample_random(dist, &mut rng);
    }

    let (below, above) = split(scores);
    let values = |group: &[usize]| -> Vec<f64> {
        group
            .iter()
            .map(|&t| to_internal(dist, observed[t]))
            .collect()
    };
    let weights = |group: &[usize]| -> Vec<f64> { recency_weights(group.len()) };
    let (below_values, above_values) = (values(&below), values(&above));
    let (below_weights, above_weights) = (weights(&below), weights(&above));

    if let Distribution::Categorical { n_choices } = *dist {
        let l = categorical(
            &below_values,
            &below_weights,
            n_choices,
            config.prior_weight,
        );
        let g = categorical(
            &above_values,
            &above_weights,
            n_choices,
            config.prior_weight,
        );
        let mut best = (f64::NEG_INFINITY, 0);
        for _ in 0..config.n_ei_candidates {
            let c = rng.choice(&l);
            let gain = l[c].ln() - g[c].ln();
            if gain > best.0 {
                best = (gain, c);
            }
        }
        best.1 as f64
    } else {
        let (low, high) = internal_bounds(dist);
        let l = Parzen::new(
            &below_values,
            &below_weights,
            low,
            high,
            config.prior_weight,
        );
        let g = Parzen::new(
            &above_values,
            &above_weights,
            low,
            high,
            config.prior_weight,
        );
        let mut best = (f64::NEG_INFINITY, f64::midpoint(low, high));
        for _ in 0..config.n_ei_candidates {
            let x = l.sample(&mut rng);
            let gain = l.log_pdf(x) - g.log_pdf(x);
            if gain > best.0 {
                best = (gain, x);
            }
        }
        to_external(dist, best.1)
    }
}

/// Trial positions of the best `gamma(n)` scores and of the rest, each in trial order.
///
/// `gamma(n) = min(ceil(n / 10), 25)`, Optuna's default. Ties go to the earlier trial.
fn split(scores: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let n = scores.len();
    let n_below = n.div_ceil(10).min(25);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let mut below = order[..n_below].to_vec();
    let mut above = order[n_below..].to_vec();
    below.sort_unstable();
    above.sort_unstable();
    (below, above)
}

/// Optuna's default weights over observations in trial order: flat for the last 25 and
/// ramping down linearly before that, so old trials count less.
fn recency_weights(n: usize) -> Vec<f64> {
    const FLAT: usize = 25;
    if n < FLAT {
        return vec![1.0; n];
    }
    let ramp = n - FLAT;
    let mut weights: Vec<f64> = (0..ramp)
        .map(|i| {
            if ramp == 1 {
                1.0 / n as f64
            } else {
                1.0 / n as f64 + (1.0 - 1.0 / n as f64) * i as f64 / (ramp - 1) as f64
            }
        })
        .collect();
    weights.extend(std::iter::repeat_n(1.0, FLAT));
    weights
}

/// Choice probabilities: observation weights plus the prior spread evenly.
fn categorical(values: &[f64], weights: &[f64], n_choices: usize, prior_weight: f64) -> Vec<f64> {
    let mut p = vec![prior_weight / n_choices as f64; n_choices];
    for (&v, &w) in values.iter().zip(weights) {
        p[(v as usize).min(n_choices - 1)] += w;
    }
    let total: f64 = p.iter().sum();
    for x in &mut p {
        *x /= total;
    }
    p
}

/// A mixture of Gaussians truncated to `[low, high]`, one per observation plus a wide
/// prior centred on the range.
struct Parzen {
    mus: Vec<f64>,
    sigmas: Vec<f64>,
    weights: Vec<f64>,
    log_weights: Vec<f64>,
    /// `ln` of the mass each component has inside the bounds.
    log_mass: Vec<f64>,
    low: f64,
    high: f64,
}

impl Parzen {
    fn new(values: &[f64], weights: &[f64], low: f64, high: f64, prior_weight: f64) -> Self {
        let width = high - low;
        let mut mus = values.to_vec();
        let mut component_weights = weights.to_vec();
        mus.push(f64::midpoint(low, high));
        component_weights.push(prior_weight);

        // Bandwidth: the larger gap to a sorted neighbour, clipped to
        // [width / min(100, 1 + n), width] -- Optuna's "magic clip". The prior keeps
        // the whole width.
        let n = mus.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| mus[a].total_cmp(&mus[b]));
        let min_sigma = width / (1.0 + n as f64).min(100.0);
        let mut sigmas = vec![width; n];
        for (rank, &i) in order.iter().enumerate() {
            let left = if rank == 0 {
                mus[i] - low
            } else {
                mus[i] - mus[order[rank - 1]]
            };
            let right = if rank + 1 == n {
                high - mus[i]
            } else {
                mus[order[rank + 1]] - mus[i]
            };
            sigmas[i] = left.max(right).clamp(min_sigma, width);
        }
        sigmas[n - 1] = width;

        let total: f64 = component_weights.iter().sum();
        for w in &mut component_weights {
            *w /= total;
        }
        let log_weights = component_weights.iter().map(|w| w.ln()).collect();
        let log_mass = mus
            .iter()
            .zip(&sigmas)
            .map(|(&mu, &sigma)| normal_mass((low - mu) / sigma, (high - mu) / sigma).ln())
            .collect();
        Self {
            mus,
            sigmas,
            weights: component_weights,
            log_weights,
            log_mass,
            low,
            high,
        }
    }

    fn sample(&self, rng: &mut Rng) -> f64 {
        let c = rng.choice(&self.weights);
        let (mu, sigma) = (self.mus[c], self.sigmas[c]);
        // Every component is centred inside the bounds with a bandwidth no wider than
        // them, so at least a third of its mass is inside and rejection ends quickly.
        for _ in 0..64 {
            let x = madd::<BASELINE_FMA>(sigma, rng.normal(), mu);
            if (self.low..=self.high).contains(&x) {
                return x;
            }
        }
        rng.uniform_in(self.low, self.high)
    }

    fn log_pdf(&self, x: f64) -> f64 {
        let terms: Vec<f64> = (0..self.mus.len())
            .map(|c| {
                let z = (x - self.mus[c]) / self.sigmas[c];
                madd::<BASELINE_FMA>(
                    0.5,
                    -(2.0 * PI).ln(),
                    madd::<BASELINE_FMA>(0.5 * z, -z, self.log_weights[c]) - self.sigmas[c].ln(),
                ) - self.log_mass[c]
            })
            .collect();
        log_sum_exp(&terms)
    }
}

fn log_sum_exp(terms: &[f64]) -> f64 {
    let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return max;
    }
    max + terms.iter().map(|t| (t - max).exp()).sum::<f64>().ln()
}

/// `P(a <= Z <= b)` for a standard normal `Z`, accurate in either tail.
fn normal_mass(a: f64, b: f64) -> f64 {
    let upper = |z: f64| 0.5 * erfc(z * FRAC_1_SQRT_2); // P(Z > z)
    let mass = if a > 0.0 {
        upper(a) - upper(b)
    } else {
        upper(-b) - upper(-a)
    };
    mass.max(f64::MIN_POSITIVE)
}

/// The complementary error function, with a relative error below 1.2e-7 everywhere
/// (Numerical Recipes' `erfcc`).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / madd::<BASELINE_FMA>(0.5, z, 1.0);
    let poly = madd::<BASELINE_FMA>(-z, z, -1.265_512_23)
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let r = t * poly.exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

fn internal_bounds(dist: &Distribution) -> (f64, f64) {
    match *dist {
        Distribution::Float { low, high, log } => {
            if log {
                (low.ln(), high.ln())
            } else {
                (low, high)
            }
        }
        Distribution::Int { low, high, log } => {
            let (low, high) = (low - 0.5, high + 0.5);
            if log {
                (low.ln(), high.ln())
            } else {
                (low, high)
            }
        }
        Distribution::Categorical { n_choices } => (0.0, n_choices as f64 - 1.0),
    }
}

fn to_internal(dist: &Distribution, x: f64) -> f64 {
    let (low, high) = internal_bounds(dist);
    let v = match *dist {
        Distribution::Float { log: true, .. } | Distribution::Int { log: true, .. } => x.ln(),
        _ => x,
    };
    v.clamp(low, high)
}

fn to_external(dist: &Distribution, v: f64) -> f64 {
    match *dist {
        Distribution::Float { low, high, log } => {
            let x = if log { v.exp() } else { v };
            x.clamp(low, high)
        }
        Distribution::Int { low, high, log } => {
            let x = if log { v.exp() } else { v };
            x.round().clamp(low, high)
        }
        Distribution::Categorical { n_choices } => v.round().clamp(0.0, n_choices as f64 - 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The best score after `n_trials` of maximizing `objective` over `dist`.
    fn search(
        dist: &Distribution,
        objective: impl Fn(f64) -> f64,
        n_trials: usize,
        config: &TpeConfig,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>) {
        let (mut observed, mut scores) = (Vec::new(), Vec::new());
        for t in 0..n_trials {
            let x = suggest(
                dist,
                &observed,
                &scores,
                config,
                seed.wrapping_mul(1000) + t as u64,
            );
            observed.push(x);
            scores.push(objective(x));
        }
        (observed, scores)
    }

    fn best(scores: &[f64]) -> f64 {
        scores.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    const RANDOM: TpeConfig = TpeConfig {
        n_startup_trials: usize::MAX,
        n_ei_candidates: 24,
        prior_weight: 1.0,
    };

    #[test]
    fn samples_stay_in_bounds() {
        let dists = [
            Distribution::Float {
                low: -2.0,
                high: 3.0,
                log: false,
            },
            Distribution::Float {
                low: 1e-4,
                high: 10.0,
                log: true,
            },
            Distribution::Int {
                low: 1.0,
                high: 7.0,
                log: false,
            },
            Distribution::Int {
                low: 5.0,
                high: 1000.0,
                log: true,
            },
            Distribution::Categorical { n_choices: 4 },
        ];
        for dist in &dists {
            let (observed, _) = search(dist, |x| -x, 60, &TpeConfig::default(), 3);
            let (low, high) = match *dist {
                Distribution::Float { low, high, .. } | Distribution::Int { low, high, .. } => {
                    (low, high)
                }
                Distribution::Categorical { n_choices } => (0.0, n_choices as f64 - 1.0),
            };
            for &x in &observed {
                assert!((low..=high).contains(&x), "{x} outside {dist:?}");
                if !matches!(dist, Distribution::Float { .. }) {
                    assert_eq!(x.to_bits(), x.round().to_bits());
                }
            }
        }
    }

    #[test]
    fn integer_ends_are_reachable() {
        let dist = Distribution::Int {
            low: 0.0,
            high: 2.0,
            log: false,
        };
        let mut rng = Rng::new(0);
        let mut seen = [false; 3];
        for _ in 0..200 {
            seen[sample_random(&dist, &mut rng) as usize] = true;
        }
        assert_eq!(seen, [true; 3]);
    }

    #[test]
    fn same_seed_same_suggestion() {
        let dist = Distribution::Float {
            low: 0.0,
            high: 1.0,
            log: false,
        };
        let observed: Vec<f64> = (0..20).map(|i| f64::from(i) / 20.0).collect();
        let scores: Vec<f64> = observed.iter().map(|x| -(x - 0.3f64).powi(2)).collect();
        let config = TpeConfig::default();
        let a = suggest(&dist, &observed, &scores, &config, 42);
        assert_eq!(
            a.to_bits(),
            suggest(&dist, &observed, &scores, &config, 42).to_bits()
        );
        assert_ne!(
            a.to_bits(),
            suggest(&dist, &observed, &scores, &config, 43).to_bits()
        );
    }

    #[test]
    fn tpe_beats_random_on_a_quadratic() {
        let dist = Distribution::Float {
            low: -10.0,
            high: 10.0,
            log: false,
        };
        let objective = |x: f64| -(x - 3.7).powi(2);
        let (mut tpe, mut random) = (0.0, 0.0);
        for seed in 0..20 {
            tpe += best(&search(&dist, objective, 50, &TpeConfig::default(), seed).1);
            random += best(&search(&dist, objective, 50, &RANDOM, seed).1);
        }
        assert!(tpe > random, "tpe {tpe} vs random {random}");
        assert!(tpe / 20.0 > -0.01, "mean best {}", tpe / 20.0);
    }

    #[test]
    fn tpe_concentrates_on_the_best_choice() {
        let dist = Distribution::Categorical { n_choices: 10 };
        let objective = |x: f64| {
            if x.to_bits() == 7.0_f64.to_bits() {
                1.0
            } else {
                x / 100.0
            }
        };
        let hits = |observed: &[f64]| {
            observed[30..]
                .iter()
                .filter(|&&x| x.to_bits() == 7.0_f64.to_bits())
                .count()
        };
        let (mut tpe, mut random) = (0, 0);
        for seed in 0..10 {
            tpe += hits(&search(&dist, objective, 60, &TpeConfig::default(), seed).0);
            random += hits(&search(&dist, objective, 60, &RANDOM, seed).0);
        }
        assert!(tpe > 3 * random, "tpe {tpe} vs random {random}");
    }

    #[test]
    fn erfc_matches_known_values() {
        for (x, expected) in [
            (0.0, 1.0),
            (1.0, 0.157_299_207),
            (-1.0, 1.842_700_793),
            (3.0, 2.209_049_7e-5),
        ] {
            assert!((erfc(x) - expected).abs() / expected < 1e-6, "erfc({x})");
        }
    }

    #[test]
    fn recency_weights_ramp_before_the_last_25() {
        assert_eq!(recency_weights(3), vec![1.0; 3]);
        let w = recency_weights(30);
        assert_eq!(w.len(), 30);
        assert!(w[..5].windows(2).all(|p| p[0] < p[1]));
        assert!(w[5..].iter().all(|&x| x.to_bits() == 1.0_f64.to_bits()));
    }
}
