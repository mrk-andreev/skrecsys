import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet

from skrecsys.recommendation import SLIMElasticNet

# Tight enough that coordinate descent runs to the optimum, so the comparisons below
# measure the model rather than where two solvers happened to stop.
MAX_ITER, TOL = 10_000, 1e-10


def converged(**params):
    """An estimator whose columns are solved to the optimum, not to ``max_iter``."""
    return SLIMElasticNet(max_iter=MAX_ITER, tol=TOL, **params)


def _counts(seed=0, n_users=80, n_items=20, density=0.3):
    rng = np.random.default_rng(seed)
    mask = rng.random((n_users, n_items)) < density
    mask[np.arange(n_users), np.arange(n_users) % n_items] = True
    users, items = np.nonzero(mask)
    return np.column_stack([users, items]), rng.integers(1, 6, len(users)).astype(float)


def _reference_weights(interactions, alpha, l1_ratio, *, positive=True):
    """SLIM as written in ``SLIM_ElasticNet/SLIMElasticNetRecommender.py``, unpruned.

    One scikit-learn ``ElasticNet`` per item, fitted on the interaction matrix with the
    target column zeroed out, which is the reference's design matrix.
    """
    ratings = np.asarray(interactions.todense(), dtype=np.float64)
    n_items = ratings.shape[1]
    weights = np.zeros((n_items, n_items))
    for item in range(n_items):
        design = ratings.copy()
        design[:, item] = 0.0
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            positive=positive,
            fit_intercept=False,
            copy_X=True,
            precompute=True,
            selection="cyclic",
            max_iter=MAX_ITER,
            tol=TOL,
        )
        weights[:, item] = model.fit(design, ratings[:, item]).coef_
    return weights


def test_weights_match_the_reference_implementation():
    X, y = _counts()
    est = converged(alpha=0.1, l1_ratio=0.5, n_neighbors=20).fit(X, y)
    expected = _reference_weights(est.interactions_, 0.1, 0.5)
    assert est.n_unconverged_ == 0
    np.testing.assert_allclose(est.similarity_.toarray(), expected, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize(
    ("alpha", "l1_ratio", "positive"), [(1.0, 0.1, True), (0.01, 0.9, True), (0.05, 0.5, False)]
)
def test_weights_match_the_reference_across_parameters(alpha, l1_ratio, positive):
    X, y = _counts(seed=3)
    est = converged(alpha=alpha, l1_ratio=l1_ratio, n_neighbors=20, positive=positive).fit(X, y)
    expected = _reference_weights(est.interactions_, alpha, l1_ratio, positive=positive)
    np.testing.assert_allclose(est.similarity_.toarray(), expected, rtol=1e-6, atol=1e-9)


def test_the_diagonal_is_zero():
    X, y = _counts()
    est = SLIMElasticNet(alpha=0.01).fit(X, y)
    assert est.similarity_.diagonal().sum() == 0.0


def test_weights_are_non_negative_by_default():
    X, y = _counts()
    assert (converged(alpha=0.01).fit(X, y).similarity_.data >= 0).all()
    signed = converged(alpha=0.01, positive=False).fit(X, y).similarity_
    assert (signed.data < 0).any()


def test_n_neighbors_bounds_the_column_length():
    X, y = _counts()
    est = SLIMElasticNet(alpha=0.001, n_neighbors=3).fit(X, y)
    assert np.diff(est.similarity_.tocsc().indptr).max() <= 3


def test_pruning_keeps_the_largest_weights_of_each_column():
    X, y = _counts()
    full = converged(alpha=0.01, n_neighbors=20).fit(X, y).similarity_.toarray()
    pruned = converged(alpha=0.01, n_neighbors=3).fit(X, y).similarity_.toarray()
    for column in range(full.shape[1]):
        kept = pruned[:, column] != 0.0
        assert kept.sum() <= 3
        np.testing.assert_allclose(pruned[kept, column], full[kept, column], rtol=1e-9)
        if kept.sum() == 3:
            assert full[~kept, column].max() <= full[kept, column].min()


def test_a_larger_alpha_shrinks_and_sparsifies_the_weights():
    X, y = _counts()

    def fitted(alpha):
        # Mostly lasso, so that the penalty that grows with alpha is the sparsifying one.
        return converged(alpha=alpha, l1_ratio=0.9, n_neighbors=20).fit(X, y).similarity_

    weak, strong = fitted(0.005), fitted(0.5)
    assert strong.nnz < weak.nnz
    assert strong.sum() < weak.sum()


def test_a_single_item_has_no_neighbours():
    est = SLIMElasticNet(alpha=0.01).fit([["u1", "a"], ["u2", "a"]])
    assert est.similarity_.nnz == 0


def test_scores_weight_the_items_of_the_user():
    X, y = _counts()
    est = SLIMElasticNet(alpha=0.01, n_neighbors=5).fit(X, y)
    expected = est.interactions_.toarray() @ est.similarity_.toarray()
    np.testing.assert_allclose(est.predict(X), expected[X[:, 0], X[:, 1]], rtol=1e-9)


def test_interaction_values_are_not_binarized():
    X, y = _counts()
    weighted = converged(alpha=0.01).fit(X, y).similarity_.toarray()
    binary = converged(alpha=0.01).fit(X).similarity_.toarray()
    assert not np.allclose(weighted, binary)


def test_thread_count_does_not_change_the_fit():
    X, y = _counts()
    sequential = converged(alpha=0.01, n_jobs=1).fit(X, y).similarity_
    parallel = converged(alpha=0.01, n_jobs=-1).fit(X, y).similarity_
    np.testing.assert_array_equal(sequential.indptr, parallel.indptr)
    np.testing.assert_array_equal(sequential.indices, parallel.indices)
    np.testing.assert_allclose(sequential.data, parallel.data, rtol=0, atol=1e-12)


def test_a_too_small_iteration_budget_warns_and_still_fits():
    X, y = _counts()
    est = SLIMElasticNet(alpha=1e-4, max_iter=1, tol=1e-12)
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        est.fit(X, y)
    assert est.n_unconverged_ > 0
    assert est.similarity_.nnz > 0


def test_a_converged_fit_does_not_warn():
    X, y = _counts()
    est = converged(alpha=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        est.fit(X, y)
    assert est.n_unconverged_ == 0


@pytest.mark.parametrize("alpha", [0.0, -1.0, np.inf, np.nan, "1"])
def test_invalid_alpha_raises(alpha):
    with pytest.raises(ValueError, match="alpha"):
        SLIMElasticNet(alpha=alpha).fit(*_counts())


@pytest.mark.parametrize("l1_ratio", [-0.1, 1.5, np.nan, "0.5"])
def test_invalid_l1_ratio_raises(l1_ratio):
    with pytest.raises(ValueError, match="l1_ratio"):
        SLIMElasticNet(l1_ratio=l1_ratio).fit(*_counts())


@pytest.mark.parametrize("n_neighbors", [0, -1, 1.5, "all"])
def test_invalid_n_neighbors_raises(n_neighbors):
    with pytest.raises(ValueError, match="n_neighbors"):
        SLIMElasticNet(n_neighbors=n_neighbors).fit(*_counts())


@pytest.mark.parametrize("positive", [1, "yes", None])
def test_invalid_positive_raises(positive):
    with pytest.raises(ValueError, match="positive"):
        SLIMElasticNet(positive=positive).fit(*_counts())


@pytest.mark.parametrize("max_iter", [0, -1, 1.5, "many"])
def test_invalid_max_iter_raises(max_iter):
    with pytest.raises(ValueError, match="max_iter"):
        SLIMElasticNet(max_iter=max_iter).fit(*_counts())


@pytest.mark.parametrize("tol", [0.0, -1e-4, np.inf, np.nan, "1e-4"])
def test_invalid_tol_raises(tol):
    with pytest.raises(ValueError, match="tol"):
        SLIMElasticNet(tol=tol).fit(*_counts())


@pytest.mark.parametrize("n_jobs", [0, -2, 1.5, "all"])
def test_invalid_n_jobs_raises(n_jobs):
    with pytest.raises(ValueError, match="n_jobs"):
        SLIMElasticNet(n_jobs=n_jobs).fit(*_counts())
