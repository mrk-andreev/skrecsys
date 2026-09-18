import numpy as np
import pytest
from sklearn.utils import check_random_state

from skrecsys.recommendation import BayesianPersonalizedRanking

MASK = (1 << 64) - 1


def _counts(seed=0, n_users=40, n_items=15, density=0.3):
    rng = np.random.default_rng(seed)
    mask = rng.random((n_users, n_items)) < density
    mask[np.arange(n_users), np.arange(n_users) % n_items] = True
    users, items = np.nonzero(mask)
    return np.column_stack([users, items]), rng.integers(1, 6, len(users)).astype(float)


def _mix(z):
    """splitmix64's finalizer, as ``rust/src/bpr.rs`` defines it."""
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def _draw(seed, counter):
    """The two uniform draws of sample number ``counter``."""
    counter = (counter * 2) & MASK
    return _mix(seed ^ _mix(counter)), _mix(seed ^ _mix(counter + 1))


def _reference_fit(est, interactions):
    """BPR as written in ``cornac/models/bpr/recom_bpr.pyx``, one triplet at a time.

    Initialization, triplet stream and update order all follow the kernel, so a
    single-threaded fit must reproduce this elementwise.
    """
    rng = check_random_state(est.random_state)
    n_users, n_items = interactions.shape
    k, lr, reg = est.n_factors, est.learning_rate, est.regularization
    user_factors = (rng.uniform(size=(n_users, k)) - 0.5) / k
    item_factors = (rng.uniform(size=(n_items, k)) - 0.5) / k
    item_bias = np.zeros(n_items)
    seed = int(rng.randint(2**31))

    indptr, indices = interactions.indptr, interactions.indices
    user_of = np.repeat(np.arange(n_users), np.diff(indptr))
    n_samples = len(indices)
    correct = []
    for epoch in range(est.max_iter):
        right, used = 0, 0
        for sample in range(n_samples):
            first, second = _draw(seed, epoch * n_samples + sample)
            entry = first % n_samples
            user, item_i = user_of[entry], indices[entry]
            item_j = second % n_items
            if item_j in indices[indptr[user] : indptr[user + 1]]:
                continue
            used += 1
            score = item_bias[item_i] - item_bias[item_j] if est.use_bias else 0.0
            for f in range(k):
                score += user_factors[user, f] * (item_factors[item_i, f] - item_factors[item_j, f])
            z = 1.0 / (1.0 + np.exp(score))
            right += z < 0.5

            p_u = user_factors[user].copy()
            q_i, q_j = item_factors[item_i].copy(), item_factors[item_j].copy()
            user_factors[user] = p_u + lr * (z * (q_i - q_j) - reg * p_u)
            item_factors[item_i] = q_i + lr * (z * p_u - reg * q_i)
            item_factors[item_j] = q_j + lr * (-z * p_u - reg * q_j)
            if est.use_bias:
                b_i, b_j = item_bias[item_i], item_bias[item_j]
                item_bias[item_i] = b_i + lr * (z - reg * b_i)
                item_bias[item_j] = b_j + lr * (-z - reg * b_j)
        correct.append(right / used if used else 0.0)
    return user_factors, item_factors, item_bias, np.asarray(correct)


def _train_auc(est):
    """Share of (observed, unobserved) item pairs the model ranks the right way round."""
    scores = est._score_users(np.arange(est.n_users_), np.arange(est.n_items_))
    observed = est.interactions_.toarray() > 0
    right, total = 0, 0
    for row, seen in zip(scores, observed, strict=True):
        positive, negative = row[seen], row[~seen]
        right += int((positive[:, None] > negative[None, :]).sum())
        total += positive.size * negative.size
    return right / total if total else 0.0


@pytest.mark.parametrize("use_bias", [True, False])
def test_factors_match_the_reference_implementation(use_bias):
    X, y = _counts()
    est = BayesianPersonalizedRanking(
        n_factors=4, max_iter=5, use_bias=use_bias, random_state=0, n_jobs=1
    ).fit(X, y)
    factors, items, bias, curve = _reference_fit(est, est.interactions_)
    np.testing.assert_allclose(est.user_factors_, factors, rtol=1e-9)
    np.testing.assert_allclose(est.item_factors_, items, rtol=1e-9)
    np.testing.assert_allclose(est.item_bias_, bias, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(est.auc_curve_, curve, rtol=1e-9)


def test_training_ranks_observed_items_above_unobserved_ones():
    X, y = _counts()
    est = BayesianPersonalizedRanking(n_factors=16, max_iter=200, random_state=0).fit(X, y)
    assert _train_auc(est) > 0.9


def test_the_auc_curve_reports_one_score_per_epoch_and_improves():
    X, y = _counts()
    est = BayesianPersonalizedRanking(n_factors=16, max_iter=200, random_state=0).fit(X, y)
    assert est.auc_curve_.shape == (200,)
    assert est.auc_curve_[0] < est.auc_curve_[-1]
    assert est.auc_curve_[-1] > 0.9


def test_the_same_random_state_gives_the_same_fit():
    X, y = _counts()
    first = BayesianPersonalizedRanking(n_factors=8, max_iter=20, random_state=3).fit(X, y)
    again = BayesianPersonalizedRanking(n_factors=8, max_iter=20, random_state=3).fit(X, y)
    other = BayesianPersonalizedRanking(n_factors=8, max_iter=20, random_state=4).fit(X, y)
    np.testing.assert_array_equal(first.user_factors_, again.user_factors_)
    np.testing.assert_array_equal(first.item_factors_, again.item_factors_)
    assert not np.allclose(first.user_factors_, other.user_factors_)


def test_fitting_on_several_threads_reaches_the_same_quality():
    X, y = _counts()
    sequential = BayesianPersonalizedRanking(
        n_factors=16, max_iter=200, random_state=0, n_jobs=1
    ).fit(X, y)
    parallel = BayesianPersonalizedRanking(
        n_factors=16, max_iter=200, random_state=0, n_jobs=-1
    ).fit(X, y)
    assert abs(_train_auc(sequential) - _train_auc(parallel)) < 0.05


def test_interaction_values_are_ignored():
    X, y = _counts()
    weighted = BayesianPersonalizedRanking(n_factors=8, max_iter=20, random_state=0).fit(X, y)
    binary = BayesianPersonalizedRanking(n_factors=8, max_iter=20, random_state=0).fit(X)
    np.testing.assert_array_equal(weighted.user_factors_, binary.user_factors_)


def test_without_bias_the_item_biases_stay_zero():
    X, y = _counts()
    est = BayesianPersonalizedRanking(max_iter=20, use_bias=False, random_state=0).fit(X, y)
    assert not est.item_bias_.any()


def test_without_epochs_the_fit_is_the_initialization():
    X, y = _counts()
    est = BayesianPersonalizedRanking(n_factors=4, max_iter=0, random_state=0).fit(X, y)
    rng = check_random_state(0)
    expected = (rng.uniform(size=(est.n_users_, 4)) - 0.5) / 4
    np.testing.assert_array_equal(est.user_factors_, expected)
    assert est.auc_curve_.shape == (0,)


def test_stronger_regularization_shrinks_the_factors():
    X, y = _counts()
    weak = BayesianPersonalizedRanking(
        n_factors=16, max_iter=100, regularization=0.001, random_state=0
    ).fit(X, y)
    strong = BayesianPersonalizedRanking(
        n_factors=16, max_iter=100, regularization=0.5, random_state=0
    ).fit(X, y)
    assert np.linalg.norm(strong.user_factors_) < np.linalg.norm(weak.user_factors_)
    assert np.linalg.norm(strong.item_factors_) < np.linalg.norm(weak.item_factors_)


def test_scores_are_the_item_bias_plus_the_factor_product():
    X, y = _counts()
    est = BayesianPersonalizedRanking(n_factors=8, max_iter=20, random_state=0).fit(X, y)
    expected = est.item_bias_ + est.user_factors_ @ est.item_factors_.T
    np.testing.assert_allclose(est.predict(X), expected[X[:, 0], X[:, 1]], rtol=1e-9)


def test_a_single_item_cannot_be_compared_with_anything():
    est = BayesianPersonalizedRanking(n_factors=4, max_iter=10, random_state=0)
    est.fit([["u1", "a"], ["u2", "a"]])
    # Every triplet is skipped, so nothing is ever updated.
    assert not est.item_bias_.any()
    assert not est.auc_curve_.any()


@pytest.mark.parametrize("n_factors", [-1, 1.5, "8", None])
def test_invalid_n_factors_raises(n_factors):
    with pytest.raises(ValueError, match="n_factors"):
        BayesianPersonalizedRanking(n_factors=n_factors).fit(*_counts())


@pytest.mark.parametrize("max_iter", [-1, 1.5, "10", None])
def test_invalid_max_iter_raises(max_iter):
    with pytest.raises(ValueError, match="max_iter"):
        BayesianPersonalizedRanking(max_iter=max_iter).fit(*_counts())


@pytest.mark.parametrize("learning_rate", [-0.1, "0.05", None, np.nan])
def test_invalid_learning_rate_raises(learning_rate):
    with pytest.raises(ValueError, match="learning_rate"):
        BayesianPersonalizedRanking(learning_rate=learning_rate).fit(*_counts())


@pytest.mark.parametrize("regularization", [-0.1, "0.01", None, np.nan])
def test_invalid_regularization_raises(regularization):
    with pytest.raises(ValueError, match="regularization"):
        BayesianPersonalizedRanking(regularization=regularization).fit(*_counts())


@pytest.mark.parametrize("use_bias", [1, "yes", None])
def test_invalid_use_bias_raises(use_bias):
    with pytest.raises(ValueError, match="use_bias"):
        BayesianPersonalizedRanking(use_bias=use_bias).fit(*_counts())


@pytest.mark.parametrize("n_jobs", [0, -2, 1.5, "all"])
def test_invalid_n_jobs_raises(n_jobs):
    with pytest.raises(ValueError, match="n_jobs"):
        BayesianPersonalizedRanking(n_jobs=n_jobs).fit(*_counts())
