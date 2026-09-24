import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone

from skrecsys.indexing import HNSW
from skrecsys.recommendation import (
    EASE,
    AlternatingLeastSquares,
    BayesianPersonalizedRanking,
    BM25Recommender,
    ItemKNNRecommender,
    MostPopularRecommender,
    RP3Beta,
    SLIMElasticNet,
)
from skrecsys.recommendation._base import BaseRecommender, kernel_indices
from tests.estimator_checks import (
    _both_interactions,
    _interactions,
    _next_interactions,
    check_pandas_input_matches_numpy,
    yield_incremental_recommender_checks,
    yield_recommender_checks,
)

IMPLEMENTED = [
    MostPopularRecommender(),
    MostPopularRecommender(weighting="sum"),
    ItemKNNRecommender(),
    ItemKNNRecommender(n_neighbors=1, shrink=10.0),
    ItemKNNRecommender(n_neighbors=None),
    AlternatingLeastSquares(n_factors=2, n_iter=5, random_state=0),
    BayesianPersonalizedRanking(n_factors=4, max_iter=10, random_state=0),
    BayesianPersonalizedRanking(n_factors=2, max_iter=5, use_bias=False, random_state=0, n_jobs=1),
    BM25Recommender(),
    BM25Recommender(n_neighbors=2, n_jobs=1),
    EASE(),
    EASE(l2_reg=1.0, n_jobs=1),
    RP3Beta(),
    RP3Beta(n_neighbors=2, alpha=0.7, beta=0.0, normalize_similarity=False, n_jobs=1),
    SLIMElasticNet(alpha=0.01),
    SLIMElasticNet(alpha=0.05, l1_ratio=0.5, n_neighbors=2, positive=False, n_jobs=1),
]

#: The same contract, carrying an approximate index.
#:
#: These satisfy every check below that is about the *contract* -- k eligible items,
#: deterministic, ties by fitted item order, the right errors -- because on a catalog
#: this small `min_index_size` keeps them on the exact path, which is the point of that
#: parameter and is pinned by `test_index_falls_back_below_min_index_size`. They are
#: deliberately absent from the exact-equality tests further down, which an approximate
#: index would fail by construction on a catalog where the graph *was* used; what those
#: tests cover for an index instead lives in `tests/indexing/`.
INDEXED = [
    ItemKNNRecommender(index="hnsw"),
    BM25Recommender(index=HNSW(m=4, ef_search=8)),
    RP3Beta(index="hnsw"),
    SLIMElasticNet(alpha=0.01, index="hnsw"),
    EASE(index="hnsw"),
    AlternatingLeastSquares(n_factors=2, n_iter=5, random_state=0, index="hnsw"),
    BayesianPersonalizedRanking(n_factors=4, max_iter=10, random_state=0, index="hnsw"),
]


@pytest.mark.parametrize("estimator", IMPLEMENTED + INDEXED, ids=repr)
@pytest.mark.parametrize("check", list(yield_recommender_checks()), ids=lambda c: c.__name__)
def test_common_checks(estimator, check):
    check(type(estimator).__name__, estimator)


@pytest.mark.parametrize("estimator", IMPLEMENTED, ids=repr)
def test_pandas_input_matches_numpy(estimator):
    check_pandas_input_matches_numpy(type(estimator).__name__, estimator)


#: The estimators that have an incremental update, with the parameters that make one
#: cheap. Everything here is seeded, because the incremental checks compare a chain of
#: `partial_fit` calls against a single `fit`.
INCREMENTAL = [
    MostPopularRecommender(),
    MostPopularRecommender(weighting="sum"),
    AlternatingLeastSquares(n_factors=2, n_iter=5, random_state=0),
    AlternatingLeastSquares(n_factors=2, n_iter=5, n_iter_partial=2, random_state=0),
    BayesianPersonalizedRanking(n_factors=4, max_iter=10, random_state=0),
    BayesianPersonalizedRanking(
        n_factors=2, max_iter=5, max_iter_partial=2, use_bias=False, random_state=0, n_jobs=1
    ),
    EASE(),
    EASE(l2_reg=1.0, n_jobs=1),
    ItemKNNRecommender(),
    ItemKNNRecommender(n_neighbors=1, shrink=10.0),
    ItemKNNRecommender(n_neighbors=None),
    BM25Recommender(),
    BM25Recommender(n_neighbors=2, n_jobs=1),
    BM25Recommender(b=0.0),
    RP3Beta(),
    RP3Beta(n_neighbors=2, alpha=0.7, beta=0.0, normalize_similarity=False, n_jobs=1),
    SLIMElasticNet(alpha=0.01),
    SLIMElasticNet(alpha=0.05, l1_ratio=0.5, n_neighbors=2, positive=False, n_jobs=1),
]

#: Those whose incremental update is the same answer a single `fit` reaches, not merely
#: a defensible one. Warm-started descent -- ALS and BPR -- cannot be here by
#: construction: it resumes from parameters a fresh fit would have thrown away.
INCREMENTAL_EXACT = [
    MostPopularRecommender(),
    MostPopularRecommender(weighting="sum"),
    EASE(),
    EASE(l2_reg=1.0, n_jobs=1),
    ItemKNNRecommender(),
    ItemKNNRecommender(n_neighbors=None),
    BM25Recommender(),
    BM25Recommender(b=0.0),
    RP3Beta(),
    RP3Beta(n_neighbors=2, alpha=0.7, beta=0.0, normalize_similarity=False, n_jobs=1),
]

#: An index on top, to pin that it is rebuilt rather than left describing the old
#: catalog. On these fixtures the graph is never walked, so the answers stay exact.
INCREMENTAL_INDEXED = [
    EASE(index="hnsw"),
    SLIMElasticNet(alpha=0.01, index="hnsw"),
    ItemKNNRecommender(index="hnsw"),
    AlternatingLeastSquares(n_factors=2, n_iter=5, random_state=0, index="hnsw"),
]


@pytest.mark.parametrize("estimator", INCREMENTAL + INCREMENTAL_INDEXED, ids=repr)
@pytest.mark.parametrize(
    "check", list(yield_incremental_recommender_checks()), ids=lambda c: c.__name__
)
def test_incremental_checks(estimator, check):
    check(type(estimator).__name__, estimator)


@pytest.mark.parametrize("estimator", INCREMENTAL_EXACT, ids=repr)
def test_partial_fit_matches_fit(estimator):
    """Where the update is exact, the batching must be invisible in the answers.

    Not bit-for-bit: EASE reaches its weights by updating an inverse rather than by
    factorizing, which is the same number by a different route.
    """
    incremental = clone(estimator).partial_fit(_interactions()).partial_fit(_next_interactions())
    batch = clone(estimator).fit(_both_interactions())
    if hasattr(batch, "similarity_"):
        difference = abs(incremental.similarity_ - batch.similarity_)
        largest = difference.max() if sp.issparse(difference) and difference.nnz else difference
        assert np.max(largest) < 1e-9, f"{estimator!r} fitted different weights."
    queries = incremental.user_ids_
    items, scores = incremental.recommend(queries, n_recommendations=3)
    expected_items, expected_scores = batch.recommend(queries, n_recommendations=3)
    np.testing.assert_array_equal(items, expected_items)
    np.testing.assert_allclose(scores, expected_scores, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("estimator", INCREMENTAL, ids=repr)
def test_partial_fit_does_not_refit(estimator, monkeypatch):
    """The point of the feature, encoded: a later batch must not reach `_fit`.

    An incremental update that quietly called `_fit` on the accumulated matrix would
    pass every other check in this file -- the answers would be right, and right for the
    wrong reason -- while costing exactly what it was added to avoid.
    """
    est = clone(estimator).partial_fit(_interactions())
    calls = []
    monkeypatch.setattr(type(est), "_fit", lambda self, interactions: calls.append(interactions))
    est.partial_fit(_next_interactions())
    assert not calls, f"{type(est).__name__}.partial_fit refitted from scratch."


def test_partial_fit_rejects_incompatible_identifier_types():
    """Merging the vocabularies must not silently rewrite the fitted identifiers."""
    est = MostPopularRecommender().partial_fit(np.array([[1, 2], [1, 3], [2, 2]]))
    with pytest.raises(ValueError, match="dtype"):
        est.partial_fit(np.array([[1.5, 2.5]]))


def test_partial_fit_rebuilds_the_index_over_the_grown_catalog():
    """An index left describing the old catalog would silently stop returning new items."""
    X, _ = _sparse_interactions(n_users=60, n_items=40, density=0.3)
    est = AlternatingLeastSquares(
        n_factors=2, n_iter=3, random_state=0, index=HNSW(m=4, min_index_size=8)
    )
    half = np.unique(X[:, 1])[:20]
    est.partial_fit(X[np.isin(X[:, 1], half)])
    assert isinstance(est.index_, HNSW)
    assert est.index_.space_.n_items == est.n_items_ == len(half)
    est.partial_fit(X)
    assert isinstance(est.index_, HNSW)
    assert est.index_.space_.n_items == est.n_items_ == 40


@pytest.mark.parametrize("estimator", INDEXED, ids=repr)
def test_index_falls_back_below_min_index_size(estimator):
    """An index on a catalog too small to be worth walking changes nothing at all.

    This is why every check above passes for an indexed estimator: on these fixtures
    the graph is built and then never used. It also documents the rule, so that raising
    `min_index_size` past a real catalog is a visible decision rather than a surprise.
    """
    X = _interactions()
    users = np.unique(X[:, 0])
    exact = clone(estimator).set_params(index=None)
    approx_items, approx_scores = clone(estimator).fit(X).recommend(users, n_recommendations=2)
    exact_items, exact_scores = exact.fit(X).recommend(users, n_recommendations=2)
    np.testing.assert_array_equal(approx_items, exact_items)
    np.testing.assert_allclose(approx_scores, exact_scores)


@pytest.mark.parametrize("estimator", IMPLEMENTED, ids=repr)
def test_predict_matches_full_catalog_scoring(estimator):
    """`_score_pairs` must agree with scoring the whole catalog and reading cells back."""
    users = ["u0", "u0", "u1", "u1", "u1", "u2", "u2", "u3", "u3", "u3"]
    items = ["i0", "i1", "i1", "i2", "i3", "i0", "i4", "i2", "i4", "i5"]
    X = np.column_stack([users, items]).astype(object)
    est = clone(estimator).fit(X)

    pairs = np.array([[u, i] for u in sorted(set(users)) for i in sorted(set(items))], dtype=object)
    user_idx = np.searchsorted(est.user_ids_, pairs[:, 0])
    item_idx = np.searchsorted(est.item_ids_, pairs[:, 1])
    expected = BaseRecommender._score_pairs(est, user_idx, item_idx)

    # `predict` may post-process (ALS clips to the observed target range), so the hook
    # itself is what has to stay equivalent to scoring the catalog.
    scores = est._score_pairs(user_idx, item_idx)
    assert scores.shape == expected.shape
    assert scores.dtype == expected.dtype
    np.testing.assert_allclose(scores, expected, rtol=0, atol=1e-12)


def _sparse_interactions(seed=0, n_users=40, n_items=15, density=0.12):
    rng = np.random.default_rng(seed)
    pairs = {
        (int(u), int(i))
        for u, i in rng.integers(0, [n_users, n_items], size=(int(n_users * n_items * density), 2))
    }
    pairs |= {(u, 0) for u in range(n_users)}  # nobody is left without an interaction
    X = np.array([[f"u{u}", f"i{i:02d}"] for u, i in sorted(pairs)], dtype=object)
    return X, rng


def _dense_reference(est, users, item_indices, k, *, exclude_seen):
    """Rank the dense score matrix, which is what `recommend` used to do."""
    user_idx = np.searchsorted(est.user_ids_, users)
    scores = est._score_users(user_idx, item_indices)
    eligible = np.ones(scores.shape, dtype=bool)
    if exclude_seen:
        seen = est.interactions_[user_idx][:, item_indices].toarray() != 0
        eligible &= ~seen
    order = np.argsort(np.where(eligible, -scores, np.inf), axis=1, kind="stable")[:, :k]
    return est.item_ids_[item_indices[order]], np.take_along_axis(scores, order, axis=1)


@pytest.mark.parametrize("estimator", IMPLEMENTED, ids=repr)
@pytest.mark.parametrize("exclude_seen", [True, False])
def test_recommend_matches_dense_scoring(estimator, exclude_seen):
    """Ranking in blocks, fused or not, must reproduce ranking the whole dense matrix."""
    X, _ = _sparse_interactions()
    est = clone(estimator).fit(X)
    users = est.user_ids_
    k = 5

    items, scores = est.recommend(users, n_recommendations=k, exclude_seen=exclude_seen)
    want_items, want_scores = _dense_reference(
        est, users, np.arange(est.n_items_), k, exclude_seen=exclude_seen
    )
    np.testing.assert_array_equal(items, want_items)
    np.testing.assert_allclose(scores, want_scores, rtol=0, atol=1e-12)


@pytest.mark.parametrize("estimator", IMPLEMENTED, ids=repr)
def test_recommend_with_candidates_matches_dense_scoring(estimator):
    X, rng = _sparse_interactions()
    est = clone(estimator).fit(X)
    candidates = np.sort(rng.choice(est.item_ids_, 8, replace=False))
    item_indices = np.searchsorted(est.item_ids_, candidates)
    users = est.user_ids_[:10]
    k = 2

    items, scores = est.recommend(users, n_recommendations=k, candidates=candidates)
    want_items, want_scores = _dense_reference(est, users, item_indices, k, exclude_seen=True)
    np.testing.assert_array_equal(items, want_items)
    np.testing.assert_allclose(scores, want_scores, rtol=0, atol=1e-12)


@pytest.mark.parametrize("size", [1, 3, 7])
def test_blocking_does_not_change_the_ranking(monkeypatch, size):
    """Every block size must produce the ranking of a single pass over all queries."""
    X, _ = _sparse_interactions()
    est = SLIMElasticNet(alpha=0.01).fit(X)
    expected = est.recommend(est.user_ids_, n_recommendations=4)

    monkeypatch.setattr(type(est), "_rank_chunk_size", lambda self, n_candidates, k: size)
    got = est.recommend(est.user_ids_, n_recommendations=4)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_array_equal(got[1], expected[1])


SPARSELY_CONNECTED = [
    ItemKNNRecommender(n_neighbors=1),
    SLIMElasticNet(alpha=0.05, l1_ratio=0.5, n_neighbors=1, positive=False),
    RP3Beta(n_neighbors=1),
    BM25Recommender(n_neighbors=1),
]


@pytest.mark.parametrize("estimator", SPARSELY_CONNECTED, ids=repr)
def test_recommend_falls_back_to_unreached_items(estimator):
    """Items the product never reaches score zero and rank among the rest, as before.

    The fused kernel only ever sees the items a query's neighbours reach, so this is the
    case where it has to reconstruct what the dense matrix held in its untouched cells.
    """
    X = np.array(
        [
            [f"u{u}", f"i{i:02d}"]
            for u, i in [
                (0, 0),
                (1, 0),
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 3),
                (5, 4),
                (5, 5),
                (6, 6),
                (6, 7),
            ]
        ],
        dtype=object,
    )
    est = clone(estimator).fit(X)
    k = 3

    items, scores = est.recommend(est.user_ids_, n_recommendations=k)
    want_items, want_scores = _dense_reference(
        est, est.user_ids_, np.arange(est.n_items_), k, exclude_seen=True
    )
    assert (scores == 0.0).any(), "the fallback to unreached items was not exercised"
    np.testing.assert_array_equal(items, want_items)
    np.testing.assert_allclose(scores, want_scores, rtol=0, atol=1e-12)


def test_kernel_indices_copies_only_when_it_has_to():
    already = np.arange(4, dtype=np.int64)
    assert kernel_indices(already) is already
    assert kernel_indices(np.arange(4, dtype=np.int32)).dtype == np.int64
    view = np.arange(8, dtype=np.int64)[::2]
    assert kernel_indices(view).flags.c_contiguous


def _downcast(matrix):
    """The same matrix with int32 index arrays, as scipy stores small ones."""
    return sp.csr_array(
        (matrix.data, matrix.indices.astype(np.int32), matrix.indptr.astype(np.int32)),
        shape=matrix.shape,
    )


@pytest.mark.parametrize("estimator", IMPLEMENTED, ids=repr)
def test_narrow_index_dtypes_reach_the_kernels(estimator):
    """The kernels borrow int64 indices, so a narrower CSR must still be upcast."""
    X, _ = _sparse_interactions()
    est = clone(estimator).fit(X)
    expected = est.recommend(est.user_ids_, n_recommendations=3)

    est.interactions_ = _downcast(est.interactions_)
    if hasattr(est, "similarity_") and sp.issparse(est.similarity_):
        est.similarity_ = _downcast(est.similarity_)
    got = est.recommend(est.user_ids_, n_recommendations=3)

    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_allclose(got[1], expected[1], rtol=0, atol=1e-12)
