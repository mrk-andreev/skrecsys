"""The HNSW index as a recommender uses it: recall, contracts, and pickling."""

import pickle

import numpy as np
import pytest

from skrecsys.indexing import (
    HNSW,
)
from skrecsys.recommendation import (
    EASE,
    AlternatingLeastSquares,
    BayesianPersonalizedRanking,
    ItemKNNRecommender,
    RP3Beta,
)

#: Small enough to fit a test, large enough that the index is used rather than skipped.
SMALL_INDEX = {"min_index_size": 200, "ef_search": 64}

#: Recall floors, per model, against the same model's exact answer.
#:
#: They differ because what is being indexed differs. A latent-factor model compresses
#: the catalog into a few dozen dimensions, which is the geometry a graph is good at.
#: An item-item model's vectors are short sparse similarity columns over a catalog-wide
#: dimension, where two items often share no support at all and the graph has far less
#: to navigate by -- `benchmarks/indexes.py` is where that shows up as a number worth
#: acting on. These are floors, set below what is measured, not targets.
RECALL_FLOOR = [
    (AlternatingLeastSquares(n_factors=16, n_iter=8, random_state=0), 0.90, True),
    (BayesianPersonalizedRanking(n_factors=16, max_iter=20, random_state=0), 0.90, False),
    (ItemKNNRecommender(n_neighbors=50), 0.75, False),
    (RP3Beta(n_neighbors=50), 0.75, False),
    (EASE(l2_reg=50.0), 0.60, False),
]

#: How far below a floor a measurement may land before it means something.
#:
#: `_index_build_threads` builds on one thread only for an estimator that has a
#: `random_state` to be reproducible for; the rest build on every core, and concurrent
#: insertions each see a different partial graph, so the same fixture gives a different
#: graph on every run. That is a deliberate trade -- a single-threaded build of this
#: catalog takes some fifty times longer -- and it makes recall a distribution rather
#: than a number. Measured over forty builds of the fixture below with every core
#: contended, RP3Beta, the narrowest of the five, ran from 0.743 to 0.808 around a
#: median of 0.778, so an unlucky schedule alone can carry it a little under its floor.
#: This slack covers that and nothing more: a regression in the index shows up as a
#: recall far below the floor, not as a thousandth.
RECALL_TOLERANCE = 0.02


def _fit_pair(estimator, X, y, **index_kwargs):
    """The same estimator, exactly and approximately."""
    params = {k: v for k, v in estimator.get_params().items() if k != "index"}
    exact = estimator.__class__(**params).fit(X, y)
    approx = estimator.__class__(**params, index=HNSW(**SMALL_INDEX, **index_kwargs)).fit(X, y)
    return exact, approx


def _recall(approx_items, exact_items):
    k = exact_items.shape[1]
    pairs = zip(approx_items, exact_items, strict=True)
    return np.mean([len(set(a) & set(b)) for a, b in pairs]) / k


@pytest.mark.parametrize(
    ("estimator", "floor", "needs_y"), RECALL_FLOOR, ids=lambda v: repr(v)[:40]
)
def test_recall_against_the_exact_path(estimator, floor, needs_y, interactions):
    rng = np.random.default_rng(1)
    y = rng.integers(1, 6, len(interactions)).astype(float) if needs_y else None
    exact, approx = _fit_pair(estimator, interactions, y)
    users = np.unique(interactions[:, 0])[:150]

    exact_items, _ = exact.recommend(users, n_recommendations=10)
    approx_items, _ = approx.recommend(users, n_recommendations=10)
    recall = _recall(approx_items, exact_items)
    assert recall >= floor - RECALL_TOLERANCE, (
        f"{estimator!r} found {recall:.4f} of the exact top 10, against a floor of "
        f"{floor} and {RECALL_TOLERANCE} of slack for the graph build."
    )


def test_the_index_returns_the_scores_the_model_really_gives(interactions):
    # Approximate about *which* items, never about what they score: a returned score
    # that did not match would make every metric downstream quietly wrong.
    _, approx = _fit_pair(
        AlternatingLeastSquares(n_factors=16, n_iter=8, random_state=0),
        interactions,
        np.ones(len(interactions)),
    )
    users = np.unique(interactions[:, 0])[:50]
    items, scores = approx.recommend(users, n_recommendations=10)
    pairs = np.column_stack([np.repeat(users, 10), items.ravel()])
    np.testing.assert_allclose(scores.ravel(), approx.predict(pairs), atol=1e-10)


def test_the_index_is_used_and_is_faster_to_rank(interactions):
    exact, approx = _fit_pair(ItemKNNRecommender(n_neighbors=50), interactions, None)
    assert approx.index_ is not None
    assert exact.index_ is None
    assert approx.index_.nbytes > 0


def test_results_do_not_change_between_calls(interactions):
    _, approx = _fit_pair(ItemKNNRecommender(n_neighbors=50), interactions, None)
    users = np.unique(interactions[:, 0])[:80]
    first = approx.recommend(users, n_recommendations=10)
    second = approx.recommend(users, n_recommendations=10)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_a_seeded_fit_builds_the_same_graph_twice(interactions):
    # Concurrent insertions see different partial graphs, so a seeded estimator builds
    # on one thread. Without that this test is what would flake.
    def build():
        return BayesianPersonalizedRanking(
            n_factors=8, max_iter=5, random_state=0, index=HNSW(**SMALL_INDEX)
        ).fit(interactions)

    first, second = build(), build()
    np.testing.assert_array_equal(first.index_.links_indices_, second.index_.links_indices_)
    np.testing.assert_array_equal(first.index_.node_level_, second.index_.node_level_)
    assert first.index_.entry_point_ == second.index_.entry_point_


def test_excluded_items_are_still_excluded(interactions):
    _, approx = _fit_pair(ItemKNNRecommender(n_neighbors=50), interactions, None)
    users = np.unique(interactions[:, 0])[:60]
    items, _ = approx.recommend(users, n_recommendations=10, exclude_seen=True)
    for user, row in zip(users, items, strict=True):
        seen = set(interactions[interactions[:, 0] == user, 1])
        assert not (set(row) & seen)
        assert len(set(row)) == 10


def test_exclude_interactions_is_honoured_by_the_index(interactions):
    _, approx = _fit_pair(ItemKNNRecommender(n_neighbors=50), interactions, None)
    users = np.unique(interactions[:, 0])[:60]
    before, _ = approx.recommend(users, n_recommendations=10)
    # Each user's three best items, as if they had just been shown them.
    pairs = np.column_stack([np.repeat(users, 3), before[:, :3].ravel()])
    items, _ = approx.recommend(users, n_recommendations=10, exclude_interactions=pairs)
    for user, row, shown in zip(users, items, before[:, :3], strict=True):
        seen = set(interactions[interactions[:, 0] == user, 1])
        assert not (set(row) & (seen | set(shown)))
        assert len(set(row)) == 10


def test_too_few_eligible_items_still_raises(interactions):
    _, approx = _fit_pair(ItemKNNRecommender(n_neighbors=50), interactions, None)
    users = np.unique(interactions[:, 0])[:5]
    with pytest.raises(ValueError, match="eligible"):
        approx.recommend(users, n_recommendations=approx.n_items_)


def test_candidates_below_the_floor_take_the_exact_path(interactions):
    # `min_index_size` is what keeps a short candidate list -- sampled negatives, a
    # filtered catalog -- on the path that is both exact and, at that size, faster.
    exact, approx = _fit_pair(ItemKNNRecommender(n_neighbors=50), interactions, None)
    users = np.unique(interactions[:, 0])[:40]
    candidates = np.unique(interactions[:, 1])[:150]
    expected = exact.recommend(users, n_recommendations=10, candidates=candidates)
    got = approx.recommend(users, n_recommendations=10, candidates=candidates)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_allclose(got[1], expected[1])


def test_a_small_catalog_takes_the_exact_path():
    # The default floor is far above a toy catalog, which is why the common estimator
    # checks pass unchanged for an indexed estimator: they never reach the graph.
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.integers(0, 40, 800), rng.integers(0, 60, 800)])
    exact = ItemKNNRecommender().fit(X)
    approx = ItemKNNRecommender(index="hnsw").fit(X)
    users = np.unique(X[:, 0])[:20]
    expected = exact.recommend(users, n_recommendations=5)
    got = approx.recommend(users, n_recommendations=5)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_allclose(got[1], expected[1])


def test_a_fitted_index_pickles_and_still_searches(interactions):
    _, approx = _fit_pair(
        AlternatingLeastSquares(n_factors=16, n_iter=8, random_state=0),
        interactions,
        np.ones(len(interactions)),
    )
    users = np.unique(interactions[:, 0])[:40]
    before = approx.recommend(users, n_recommendations=10)
    restored = pickle.loads(pickle.dumps(approx))
    after = restored.recommend(users, n_recommendations=10)
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_allclose(before[1], after[1])


def test_a_corrupted_graph_raises_rather_than_crashing(interactions):
    # A fitted index is numpy arrays, and numpy arrays come back from pickles that may
    # not be the ones we wrote. Every way of being wrong has to be a `ValueError`.
    _, approx = _fit_pair(
        AlternatingLeastSquares(n_factors=8, n_iter=5, random_state=0),
        interactions,
        np.ones(len(interactions)),
    )
    users = np.unique(interactions[:, 0])[:10]
    approx.index_.links_indices_ = approx.index_.links_indices_[:-1]
    with pytest.raises(ValueError, match="do not describe a graph"):
        approx.recommend(users, n_recommendations=10)


def test_raising_ef_search_does_not_lower_recall(interactions):
    # The one dial that can be turned on a fitted model, and the direction it turns.
    exact = ItemKNNRecommender(n_neighbors=50).fit(interactions)
    users = np.unique(interactions[:, 0])[:100]
    exact_items, _ = exact.recommend(users, n_recommendations=10)

    measured = []
    for ef in (16, 64, 256):
        approx = ItemKNNRecommender(
            n_neighbors=50, index=HNSW(min_index_size=200, ef_search=ef)
        ).fit(interactions)
        items, _ = approx.recommend(users, n_recommendations=10)
        measured.append(_recall(items, exact_items))
    assert measured[1] >= measured[0] - RECALL_TOLERANCE
    assert measured[2] >= measured[1] - RECALL_TOLERANCE


def _catalog(n_items, n_users=300, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [rng.integers(0, n_users, n_items * 12), rng.integers(0, n_items, n_items * 12)]
    )


def test_an_explicit_index_is_built_even_for_a_small_catalog():
    estimator = ItemKNNRecommender(index="hnsw").fit(_catalog(500))
    assert isinstance(estimator.index_, HNSW)


def test_a_seeded_model_searches_on_every_core():
    """The single-thread rule is about reproducing a graph, not about walking one.

    A search is a pure function of a fixed graph, so its thread count cannot change the
    answer -- and holding a seeded estimator to one thread here costs most of what the
    index was built for.
    """
    estimator = AlternatingLeastSquares(n_factors=8, random_state=0)
    assert estimator._index_build_threads() == 1, "a seeded build stays reproducible"
    assert estimator._index_query_threads() == 0, "0 means every core"


def test_an_explicit_n_jobs_still_narrows_both():
    estimator = ItemKNNRecommender(n_jobs=2)
    assert estimator._index_build_threads() == 2
    assert estimator._index_query_threads() == 2


def test_an_unseeded_model_builds_and_searches_on_every_core():
    estimator = ItemKNNRecommender()
    assert estimator._index_build_threads() == 0
    assert estimator._index_query_threads() == 0
