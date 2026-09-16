import numpy as np
import pytest
import scipy.sparse as sp

from skrecsys.recommendation import RP3Beta


def _counts(seed=0, n_users=70, n_items=18, density=0.35):
    rng = np.random.default_rng(seed)
    mask = rng.random((n_users, n_items)) < density
    mask[np.arange(n_users), np.arange(n_users) % n_items] = True
    users, items = np.nonzero(mask)
    return np.column_stack([users, items]), rng.integers(1, 6, len(users)).astype(float)


def _reference_similarity(interactions, alpha, beta, k, *, normalize_similarity=True):
    """RP3beta as written in ``GraphBased/RP3betaRecommender.py``, without the blocking.

    Follows the reference step by step: row-normalize the interactions, row-normalize
    the binary transpose, raise both to ``alpha``, walk, damp the destination item by
    its popularity, prune each row, normalize the rows, then prune each column.
    """
    urm = np.asarray(sp.csr_array(interactions).todense(), dtype=np.float64)
    n_items = urm.shape[1]

    pui = urm / np.abs(urm).sum(axis=1, keepdims=True)
    binary = (urm != 0.0).astype(np.float64)
    popularity = binary.sum(axis=0)
    piu = binary.T / popularity[:, None]
    if alpha != 1.0:
        pui, piu = pui**alpha, piu**alpha

    walk = piu @ pui
    walk *= popularity[None, :] ** -beta
    np.fill_diagonal(walk, 0.0)

    weights = np.zeros_like(walk)
    for i in range(n_items):
        best = np.argsort(-walk[i], kind="stable")[:k]
        best = best[walk[i][best] != 0.0]
        weights[i, best] = walk[i][best]
    if normalize_similarity:
        sums = np.abs(weights).sum(axis=1, keepdims=True)
        weights = np.divide(weights, sums, out=np.zeros_like(weights), where=sums > 0)

    pruned = np.zeros_like(weights)
    for j in range(n_items):
        column = weights[:, j]
        best = np.argsort(-column, kind="stable")[:k]
        best = best[column[best] != 0.0]
        pruned[best, j] = column[best]
    return pruned


def test_similarity_matches_the_reference_implementation():
    X, y = _counts()
    est = RP3Beta(n_neighbors=5, alpha=0.9, beta=0.6).fit(X, y)
    expected = _reference_similarity(est.interactions_, 0.9, 0.6, 5)
    np.testing.assert_allclose(est.similarity_.toarray(), expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(
    ("alpha", "beta", "k", "normalize"), [(1.0, 0.6, 100, True), (0.5, 1.2, 4, False)]
)
def test_similarity_matches_the_reference_across_parameters(alpha, beta, k, normalize):
    X, y = _counts(seed=2)
    est = RP3Beta(n_neighbors=k, alpha=alpha, beta=beta, normalize_similarity=normalize).fit(X, y)
    expected = _reference_similarity(
        est.interactions_, alpha, beta, k, normalize_similarity=normalize
    )
    np.testing.assert_allclose(est.similarity_.toarray(), expected, rtol=1e-9, atol=1e-12)


def test_the_diagonal_is_zero():
    X, y = _counts()
    est = RP3Beta().fit(X, y)
    assert est.similarity_.diagonal().sum() == 0.0


def test_n_neighbors_bounds_the_row_length():
    X, y = _counts()
    est = RP3Beta(n_neighbors=3).fit(X, y)
    assert np.diff(est.similarity_.indptr).max() <= 3


def test_rows_are_normalized_when_requested():
    X, y = _counts()
    # With every neighbour kept the column-wise pass prunes nothing, so the rows of a
    # normalized similarity matrix still sum to exactly one.
    keep_all = {"n_neighbors": X[:, 1].max() + 1}
    normalized = RP3Beta(**keep_all, normalize_similarity=True).fit(X, y).similarity_
    np.testing.assert_allclose(normalized.toarray().sum(axis=1), 1.0, rtol=1e-12)
    plain = RP3Beta(**keep_all, normalize_similarity=False).fit(X, y).similarity_
    assert not np.allclose(plain.toarray().sum(axis=1), 1.0)

    # Pruning aside, normalizing only rescales each row, so the rows agree in direction.
    for row in range(normalized.shape[0]):
        a, b = normalized.toarray()[row], plain.toarray()[row]
        np.testing.assert_allclose(a * b.sum(), b, rtol=1e-9, atol=1e-12)


def test_a_single_item_has_no_neighbours():
    est = RP3Beta().fit([["u1", "a"], ["u2", "a"]])
    assert est.similarity_.nnz == 0


def test_scores_weight_the_items_of_the_user():
    X, y = _counts()
    est = RP3Beta(n_neighbors=5).fit(X, y)
    expected = est.interactions_.toarray() @ est.similarity_.toarray()
    np.testing.assert_allclose(est.predict(X), expected[X[:, 0], X[:, 1]], rtol=1e-9)


def test_a_larger_beta_shifts_weight_away_from_popular_items():
    X, y = _counts()
    popularity = np.bincount(X[:, 1])
    head = np.argsort(-popularity)[: len(popularity) // 4]

    def head_share(beta):
        similarity = RP3Beta(n_neighbors=10, beta=beta).fit(X, y).similarity_.toarray()
        return similarity[:, head].sum() / similarity.sum()

    assert head_share(1.5) < head_share(0.0)


def test_interaction_values_are_not_binarized():
    X, y = _counts()
    weighted = RP3Beta(n_neighbors=5).fit(X, y).similarity_.toarray()
    binary = RP3Beta(n_neighbors=5).fit(X).similarity_.toarray()
    assert not np.allclose(weighted, binary)


def test_thread_count_does_not_change_the_fit():
    X, y = _counts()
    sequential = RP3Beta(n_neighbors=5, n_jobs=1).fit(X, y).similarity_
    parallel = RP3Beta(n_neighbors=5, n_jobs=-1).fit(X, y).similarity_
    np.testing.assert_array_equal(sequential.indptr, parallel.indptr)
    np.testing.assert_array_equal(sequential.indices, parallel.indices)
    np.testing.assert_allclose(sequential.data, parallel.data, rtol=0, atol=1e-12)


@pytest.mark.parametrize("n_neighbors", [0, -1, 1.5, "all"])
def test_invalid_n_neighbors_raises(n_neighbors):
    with pytest.raises(ValueError, match="n_neighbors"):
        RP3Beta(n_neighbors=n_neighbors).fit(*_counts())


@pytest.mark.parametrize("name", ["alpha", "beta"])
@pytest.mark.parametrize("value", [-1.0, np.inf, np.nan, "1"])
def test_invalid_alpha_and_beta_raise(name, value):
    with pytest.raises(ValueError, match=name):
        RP3Beta(**{name: value}).fit(*_counts())


@pytest.mark.parametrize("normalize_similarity", [1, "yes", None])
def test_invalid_normalize_similarity_raises(normalize_similarity):
    with pytest.raises(ValueError, match="normalize_similarity"):
        RP3Beta(normalize_similarity=normalize_similarity).fit(*_counts())


@pytest.mark.parametrize("n_jobs", [0, -2, 1.5, "all"])
def test_invalid_n_jobs_raises(n_jobs):
    with pytest.raises(ValueError, match="n_jobs"):
        RP3Beta(n_jobs=n_jobs).fit(*_counts())
