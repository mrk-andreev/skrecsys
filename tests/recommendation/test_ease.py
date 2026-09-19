import numpy as np
import pytest
import scipy.sparse as sp

from skrecsys.recommendation import EASE


def _counts(seed=0, n_users=60, n_items=15, density=0.3):
    rng = np.random.default_rng(seed)
    mask = rng.random((n_users, n_items)) < density
    mask[np.arange(n_users), np.arange(n_users) % n_items] = True
    users, items = np.nonzero(mask)
    return np.column_stack([users, items]), rng.integers(1, 6, len(users)).astype(float)


def _reference_weights(interactions, l2_reg):
    """EASE as written in UniRec's ``ease.py`` and RecTools' ``ease.py``."""
    gram = np.asarray((sp.csr_matrix(interactions).T @ sp.csr_matrix(interactions)).todense())
    diagonal = np.diag_indices(gram.shape[0])
    gram[diagonal] += l2_reg
    inverse = np.linalg.inv(gram)
    weights = inverse / (-np.diag(inverse))
    weights[diagonal] = 0.0
    return weights


def test_weights_match_the_reference_implementation():
    X, y = _counts()
    est = EASE(l2_reg=7.0).fit(X, y)
    expected = _reference_weights(est.interactions_, 7.0)
    # Cholesky and a general inverse differ in rounding, not in the answer.
    np.testing.assert_allclose(est.similarity_, expected, rtol=1e-7, atol=1e-10)


def test_the_diagonal_is_exactly_zero():
    X, y = _counts()
    est = EASE(l2_reg=7.0).fit(X, y)
    assert np.array_equal(np.diag(est.similarity_), np.zeros(est.n_items_))


def test_scores_weight_the_items_of_the_user():
    X, y = _counts()
    est = EASE(l2_reg=7.0).fit(X, y)
    ratings = est.interactions_.toarray()
    expected = ratings @ est.similarity_
    np.testing.assert_allclose(est.predict(X), expected[X[:, 0], X[:, 1]], rtol=1e-9)


def test_interaction_values_are_not_binarized():
    X, y = _counts()
    weighted = EASE(l2_reg=7.0).fit(X, y).similarity_
    binary = EASE(l2_reg=7.0).fit(X).similarity_
    assert not np.allclose(weighted, binary)


def test_thread_count_does_not_change_the_fit():
    X, y = _counts()
    sequential = EASE(l2_reg=7.0, n_jobs=1).fit(X, y).similarity_
    parallel = EASE(l2_reg=7.0, n_jobs=-1).fit(X, y).similarity_
    np.testing.assert_allclose(sequential, parallel, rtol=0, atol=1e-9)


def test_stronger_regularization_shrinks_the_weights():
    X, y = _counts()
    weak = np.abs(EASE(l2_reg=1.0).fit(X, y).similarity_).sum()
    strong = np.abs(EASE(l2_reg=1000.0).fit(X, y).similarity_).sum()
    assert strong < weak


@pytest.mark.parametrize("l2_reg", [0.0, -1.0, np.inf, np.nan, "500"])
def test_invalid_l2_reg_raises(l2_reg):
    with pytest.raises(ValueError, match="l2_reg"):
        EASE(l2_reg=l2_reg).fit(*_counts())


@pytest.mark.parametrize("n_jobs", [0, -2, 1.5, "all"])
def test_invalid_n_jobs_raises(n_jobs):
    with pytest.raises(ValueError, match="n_jobs"):
        EASE(n_jobs=n_jobs).fit(*_counts())


def test_partial_fit_updates_the_inverse_gram_exactly():
    """Woodbury must land on the inverse a fresh factorization would have produced.

    The batch touches three users out of two hundred, so the update runs at rank six
    against a sixty-item catalog -- the regime the identity exists for, and the one the
    common-check fixtures are far too small to reach.
    """
    rng = np.random.default_rng(0)
    n_users, n_items, l2_reg = 200, 60, 5.0
    pairs = {(int(u), int(i)) for u, i in rng.integers(0, [n_users, n_items], size=(1500, 2))}
    pairs |= {(u, 0) for u in range(n_users)}
    X = np.array(sorted(pairs))
    recent = X[:, 0] >= n_users - 3

    est = EASE(l2_reg=l2_reg).partial_fit(X[~recent])
    est.partial_fit(X[recent])
    full = EASE(l2_reg=l2_reg).fit(X)

    gram = (full.interactions_.T @ full.interactions_).toarray()
    expected = np.linalg.inv(gram + l2_reg * np.eye(n_items))
    np.testing.assert_allclose(est.inverse_gram_, expected, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(est.similarity_, full.similarity_, rtol=1e-7, atol=1e-10)


def test_many_batches_do_not_drift():
    """A long chain of rank-limited updates must not wander off the factorized answer."""
    rng = np.random.default_rng(1)
    n_users, n_items, l2_reg = 200, 60, 5.0
    pairs = {(int(u), int(i)) for u, i in rng.integers(0, [n_users, n_items], size=(1500, 2))}
    pairs |= {(u, 0) for u in range(n_users)}
    X = np.array(sorted(pairs))

    chunks = np.array_split(X, 12)
    est = EASE(l2_reg=l2_reg).partial_fit(chunks[0])
    for chunk in chunks[1:]:
        est.partial_fit(chunk)
    full = EASE(l2_reg=l2_reg).fit(X)
    np.testing.assert_allclose(est.similarity_, full.similarity_, rtol=1e-6, atol=1e-9)


def test_fit_keeps_the_memory_it_always_did():
    """The inverse is state only an incremental fit has any use for."""
    X = np.array([["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "c"]], dtype=object)
    est = EASE(l2_reg=1.0).fit(X)
    assert not hasattr(est, "inverse_gram_")
    est.partial_fit(np.array([["u3", "a"]], dtype=object))
    assert est.inverse_gram_.shape == (est.n_items_, est.n_items_)
    est.fit(X)
    assert not hasattr(est, "inverse_gram_"), "a refit left a stale inverse behind."
