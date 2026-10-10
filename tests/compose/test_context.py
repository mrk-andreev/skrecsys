"""Query context: columns after the system ones in ``fit``, a matrix of queries in ``recommend``."""

import pickle
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from skrecsys.base import FeaturesMixin
from skrecsys.compose import (
    Cascade,
    ConcatFeatures,
    GeneratorScores,
    JoinDynamicFeatures,
    KnownUser,
    PointwiseRanker,
    ReciprocalRankFusion,
    Switch,
)
from skrecsys.inspection import explain, trace
from skrecsys.metrics import evaluate_recommender, hit_rate_at_k
from skrecsys.model_selection import ColdStartSplit, WarmStartKFold
from skrecsys.recommendation import ItemKNNRecommender, MostPopularRecommender
from skrecsys.typing import override

N_SHELF_ITEMS = 12


def shelf_interactions(n_users=200, seed=0):
    """``[user, item, shelf]`` rows: shelf ``s`` holds the items of parity ``s``.

    Every user takes three items from each shelf, in a random order, so which items a
    request wants depends on its shelf alone -- a generator retrieving by user cannot
    tell, a ranker reading the context can.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(n_users):
        taken = [
            [user, item, shelf]
            for shelf in (0, 1)
            for item in rng.choice(np.arange(shelf, N_SHELF_ITEMS, 2), size=3, replace=False)
        ]
        rows.extend(taken[i] for i in rng.permutation(len(taken)))
    return np.array(rows, dtype=np.int64)


def on_shelf(keys):
    """Whether the item of each ``[item, shelf]`` key is on that shelf."""
    keys = np.asarray(keys, dtype=float)
    return (keys[:, 0] % 2 == keys[:, 1]).astype(float)[:, None]


def as_floats(keys):
    """The context rows themselves, as features."""
    return np.asarray(keys, dtype=float)


def shelf_cascade(features=None, **params):
    defaults: dict[str, Any] = {
        "generator": MostPopularRecommender(),
        "features": features
        or ConcatFeatures([GeneratorScores(), JoinDynamicFeatures("item-context", on_shelf)]),
        "ranker": PointwiseRanker(LogisticRegression()),
        "n_retrieved": N_SHELF_ITEMS,
        "split": 0.3,
    }
    return Cascade(**{**defaults, **params})


class RecordContext(FeaturesMixin, BaseEstimator):
    """A feature component reporting the context it is given; its feature is constant.

    ``record`` is a function, which a clone shares, rather than a list, which it copies.
    """

    def __init__(self, record=None):
        self.record = record

    def fit(self, X=None, y=None):
        del y
        if self.record is not None:
            self.record("fit", None if X is None else np.asarray(X).shape[1])
        return self

    def __sklearn_is_fitted__(self):
        return True

    @override
    def transform(self, pairs, *, scores=None, context=None):
        del scores
        if self.record is not None:
            self.record("transform", np.asarray(pairs), context)
        return np.zeros((len(pairs), 1))


def _recording_cascade(calls, **params):
    def record(*call):
        calls.append(call)

    features = ConcatFeatures([GeneratorScores(), RecordContext(record)])
    return shelf_cascade(features, **params)


# -- Cascade -----------------------------------------------------------------------------


def test_the_ranker_learns_from_the_context():
    X = shelf_interactions()
    rec = shelf_cascade().fit(X)
    assert rec.n_context_ == 1
    for shelf in (0, 1):
        items, _ = rec.recommend([[0, shelf]], n_recommendations=3, exclude_seen=False)
        assert set((items.astype(int) % 2).ravel().tolist()) == {shelf}


def test_the_ranker_is_trained_with_the_context_of_the_first_held_out_row():
    X = shelf_interactions(n_users=20)
    calls = []
    rec = _recording_cascade(calls).fit(X)
    # The features are fitted on the system columns only: the context travels beside.
    assert ("fit", 2) in calls
    _, pairs, context = next(c for c in calls if c[0] == "transform")
    held_start = {}
    for user in range(20):
        rows = X[X[:, 0] == user]
        n_held = int(np.floor(0.3 * len(rows)))
        held_start[user] = rows[len(rows) - n_held, 2]
    want = np.array([held_start[u] for u in pairs[:, 0]])
    np.testing.assert_array_equal(context.ravel(), want)
    assert rec.n_context_ == 1


def test_a_query_without_context_gets_nan():
    calls = []
    rec = _recording_cascade(calls).fit(shelf_interactions(n_users=20))
    calls.clear()
    rec.recommend([0, 1], n_recommendations=2)
    _, pairs, context = calls[-1]
    assert context.shape == (len(pairs), 1)
    assert np.isnan(context.astype(float)).all()


def test_nan_context_reaches_a_ranker_that_does_not_accept_it():
    # Expected: a callback passing context through hands NaN to the ranker, and
    # LogisticRegression rejects it. One that handles NaN, as on_shelf does, serves.
    X = shelf_interactions(n_users=40)
    passthrough = ConcatFeatures([GeneratorScores(), JoinDynamicFeatures("context", as_floats)])
    rec = shelf_cascade(passthrough).fit(X)
    with pytest.raises(ValueError, match="NaN"):
        rec.recommend([0], n_recommendations=2)
    rec.recommend([[0, 1]], n_recommendations=2)
    shelf_cascade().fit(X).recommend([0], n_recommendations=2)


def test_a_time_column_without_time_is_context():
    # Expected: without time=True, [user, item, time] rows carry one context column.
    X = shelf_interactions(n_users=20)
    calls = []
    rec = _recording_cascade(calls).fit(X)
    assert rec.n_context_ == 1
    assert not hasattr(rec, "time_dtype_")
    assert ("fit", 2) in calls


def test_each_candidate_carries_the_context_of_its_query():
    calls = []
    rec = _recording_cascade(calls).fit(shelf_interactions(n_users=20))
    calls.clear()
    rec.recommend([[0, 1], [1, 0], [0, 0]], n_recommendations=2)
    _, pairs, context = calls[-1]
    # The three queries' candidates are contiguous, and consecutive queries differ in user.
    starts = np.flatnonzero(np.diff(pairs[:, 0].astype(int), prepend=-1))
    assert pairs[starts, 0].tolist() == [0, 1, 0]
    run = np.cumsum(np.isin(np.arange(len(pairs)), starts)) - 1
    np.testing.assert_array_equal(context.ravel(), np.array([1, 0, 0])[run])


def test_the_context_width_must_match_the_fit():
    rec = shelf_cascade().fit(shelf_interactions(n_users=40))
    with pytest.raises(ValueError, match="fitted with 1"):
        rec.recommend([[0, 1, 2]], n_recommendations=1)


def test_a_cascade_fitted_without_context_ignores_one():
    X = shelf_interactions(n_users=40)
    rec = shelf_cascade(GeneratorScores()).fit(X[:, :2])
    assert rec.n_context_ == 0
    np.testing.assert_array_equal(
        rec.recommend([[0, 1]], n_recommendations=2)[0],
        rec.recommend([0], n_recommendations=2)[0],
    )


def test_predict_reads_the_context_and_agrees_with_recommend():
    X = shelf_interactions()
    rec = shelf_cascade().fit(X)
    items, scores = rec.recommend([[3, 1]], n_recommendations=3)
    pairs = np.column_stack([np.full(3, 3), items[0], np.ones(3, dtype=np.int64)])
    np.testing.assert_allclose(rec.predict(pairs), scores[0])
    on, off = rec.predict([[3, 1, 1], [3, 1, 0]])
    assert on > off


def test_time_and_context_together():
    X = shelf_interactions(n_users=40)
    times = np.arange(len(X))
    timed = np.column_stack([X[:, :2], times, X[:, 2]])
    calls = []
    rec = _recording_cascade(calls, time=True).fit(timed)
    _, pairs, context = next(c for c in calls if c[0] == "transform")
    assert pairs.shape[1] == 3  # user, item, time: the layout is unchanged
    assert context.shape == (len(pairs), 1)
    calls.clear()
    rec.recommend([[0, 1]], n_recommendations=2, as_of=5)
    _, pairs, context = calls[-1]
    assert (pairs[:, 2] == 5).all()
    assert (context == 1).all()
    assert rec.predict([[0, 1, 5, 1]]).shape == (1,)


def test_exclude_interactions_may_carry_context():
    X = shelf_interactions(n_users=40)
    rec = shelf_cascade().fit(X)
    items, _ = rec.recommend([[0, 0]], n_recommendations=3, exclude_seen=False)
    excluded = [[0, items[0, 0], 0]]
    again, _ = rec.recommend(
        [[0, 0]], n_recommendations=3, exclude_seen=False, exclude_interactions=excluded
    )
    assert items[0, 0] not in again[0]


def test_pickles_with_context():
    rec = shelf_cascade().fit(shelf_interactions(n_users=60))
    restored = pickle.loads(pickle.dumps(rec))
    queries = [[0, 0], [0, 1]]
    np.testing.assert_array_equal(
        restored.recommend(queries, n_recommendations=2)[0],
        rec.recommend(queries, n_recommendations=2)[0],
    )


def test_string_context_reaches_a_dynamic_join():
    X = shelf_interactions(n_users=100).astype(object)
    X[:, 2] = np.where(X[:, 2] == 1, "odd", "even")

    def named_shelf(keys):
        return np.array([[(item % 2 == 1) == (shelf == "odd")] for item, shelf in keys], float)

    features = ConcatFeatures([GeneratorScores(), JoinDynamicFeatures("item-context", named_shelf)])
    rec = shelf_cascade(features).fit(X)
    items, _ = rec.recommend(
        np.array([[0, "odd"]], dtype=object), n_recommendations=3, exclude_seen=False
    )
    assert set((items.astype(int) % 2).ravel().tolist()) == {1}


# -- other composites ----------------------------------------------------------------------


def test_switch_passes_the_context_to_its_branches():
    X = shelf_interactions()
    switch = Switch(KnownUser(), shelf_cascade(), MostPopularRecommender()).fit(X)
    for shelf in (0, 1):
        items, _ = switch.recommend(
            [[0, shelf], [10_000, shelf]], n_recommendations=3, exclude_seen=False
        )
        assert set((items[0].astype(int) % 2).tolist()) == {shelf}
    assert switch.predict([[0, 1, 1], [0, 1, 0]]).shape == (2,)


def test_a_timed_switch_keeps_the_context_for_an_untimed_branch():
    X = shelf_interactions()
    timed = np.column_stack([X[:, :2], np.arange(len(X)), X[:, 2]])
    switch = Switch(KnownUser(), shelf_cascade(), MostPopularRecommender(), time=True).fit(timed)
    assert getattr(switch.on_true_, "n_context_", None) == 1
    items, _ = switch.recommend([[0, 1]], n_recommendations=3, exclude_seen=False)
    assert set((items.astype(int) % 2).ravel().tolist()) == {1}


def test_fusion_accepts_context_and_retrieves_by_user():
    X = shelf_interactions(n_users=40)
    fusion = ReciprocalRankFusion([ItemKNNRecommender(), MostPopularRecommender()]).fit(X)
    np.testing.assert_array_equal(
        fusion.recommend([[0, 1]], n_recommendations=2)[0],
        fusion.recommend([0], n_recommendations=2)[0],
    )


# -- features ------------------------------------------------------------------------------


def test_concat_features_forwards_the_context_to_the_parts_that_read_it():
    both = ConcatFeatures([GeneratorScores(), JoinDynamicFeatures("context", as_floats)]).fit()
    out = both.transform([["u", "a"]], scores=[0.5], context=np.array([[7]]))
    assert out.tolist() == [[0.5, 7.0]]


def test_a_dynamic_join_checks_the_shape_of_the_context():
    join = JoinDynamicFeatures("context", as_floats).fit()
    with pytest.raises(ValueError, match="shape"):
        join.transform([["u", "a"]], context=[[1], [2]])


def test_a_dynamic_join_keyed_by_context_alone_gets_every_context_column():
    seen = []

    def width(keys):
        seen.append(np.asarray(keys))
        return np.asarray(keys, dtype=float).sum(axis=1)

    join = JoinDynamicFeatures("context", width).fit()
    out = join.transform([["u", "a"], ["v", "b"], ["w", "c"]], context=[[1, 2], [3, 4], [1, 2]])
    assert out.ravel().tolist() == [3.0, 7.0, 3.0]
    assert seen[0].shape == (2, 2)  # distinct context rows
    with pytest.raises(ValueError, match="reads the query context"):
        join.transform([["u", "a"]])


# -- evaluation, inspection, splitting -----------------------------------------------------


def test_evaluation_asks_each_user_with_their_first_held_out_context():
    rec = shelf_cascade().fit(shelf_interactions())
    held = np.array([[0, 1, 1], [0, 3, 1], [1, 2, 0], [1, 5, 1]])
    scores = evaluate_recommender(rec, held, metrics=[hit_rate_at_k], k=3, exclude_seen=False)
    assert scores["hit_rate@3"] == 1.0
    calls = []

    class Spy(MostPopularRecommender):
        @override
        def recommend(self, X, **kwargs):
            calls.append(np.asarray(X))
            return super().recommend(np.asarray(X)[:, 0].astype(int), **kwargs)

    spy = Spy().fit(shelf_interactions(n_users=5))
    evaluate_recommender(spy, held, metrics=[hit_rate_at_k], k=1, exclude_seen=False)
    assert calls[0].tolist() == [[0, 1], [1, 0]]


def test_trace_and_explain_take_queries_with_context():
    rec = shelf_cascade().fit(shelf_interactions(n_users=60))
    with trace() as traced:
        rec.recommend([[0, 1]], n_recommendations=2, exclude_seen=False)
    assert traced.query(0).final is not None
    explanations = explain(rec, [[0, 1]], n_recommendations=2, exclude_seen=False)
    items = np.array([e.item for e in explanations], dtype=int)
    assert (items % 2).tolist() == [1, 1]


@pytest.mark.parametrize(
    "splitter", [WarmStartKFold(n_splits=2), ColdStartSplit(0.3, random_state=0)]
)
def test_splitters_accept_context_columns(splitter):
    X = shelf_interactions(n_users=20)
    with_context = [(a.tolist(), b.tolist()) for a, b in splitter.split(X)]
    without = [(a.tolist(), b.tolist()) for a, b in splitter.split(X[:, :2])]
    assert with_context == without
