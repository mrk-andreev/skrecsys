import pickle
from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit, cross_val_score

from skrecsys.base import RankerMixin
from skrecsys.compose import (
    Cascade,
    ConcatFeatures,
    GeneratorScores,
    JoinDynamicFeatures,
    JoinStaticFeatures,
    KnownUser,
    PointwiseRanker,
    Switch,
    _cascade,
)
from skrecsys.compose._candidates import retrieve_union
from skrecsys.exceptions import InsufficientDataError
from skrecsys.metrics import evaluate_recommender, make_recommender_scorer, ndcg_at_k
from skrecsys.model_selection import ColdStartSplit, LatestInteractionsSplit, WarmStartKFold
from skrecsys.recommendation import ItemKNNRecommender, MostPopularRecommender
from skrecsys.typing import override
from skrecsys.utils.validation import check_interactions
from tests.compose._data import N_USERS, TRENDING, trending_interactions, trending_table


def _learning_cascade(**params):
    defaults: dict[str, Any] = {
        "generator": MostPopularRecommender(),
        "features": ConcatFeatures(
            [JoinStaticFeatures("item", trending_table()), GeneratorScores()]
        ),
        "ranker": PointwiseRanker(HistGradientBoostingClassifier(max_iter=20)),
        "n_retrieved": 30,
    }
    return Cascade(**{**defaults, **params})


def test_ranker_learns_what_the_generator_cannot_see():
    """The latest interaction is always trending; popularity alone does not know that."""
    X = trending_interactions()
    cascade = _learning_cascade().fit(X)
    users = np.arange(N_USERS)
    ranked, _ = cascade.recommend(users, n_recommendations=3)
    generated, _ = cascade.generator_.recommend(users, n_recommendations=3)
    ranked_share = np.isin(ranked, TRENDING).mean()
    assert ranked_share > 0.9
    assert ranked_share > np.isin(generated, TRENDING).mean() + 0.3


_GENERATOR_FITS = []


class _RecordingPopular(MostPopularRecommender):
    """Records the rows of every fit, to see what the ranker's generator was shown."""

    @override
    def fit(self, X, y=None):
        _GENERATOR_FITS.append(np.asarray(X).copy())
        return super().fit(X, y)


class _RecordingRanker(RankerMixin, BaseEstimator):
    def fit(self, X, y, *, groups):
        self.X_, self.y_, self.groups_ = X, y, groups
        return self

    def predict(self, X, *, groups):
        return X[:, 0]


def test_an_increasing_ranker_on_generator_scores_keeps_the_generator_order():
    X = trending_interactions()
    cascade = Cascade(
        ItemKNNRecommender(), GeneratorScores(), _RecordingRanker(), n_retrieved=20
    ).fit(X)
    users = np.arange(10)
    items, scores = cascade.recommend(users, n_recommendations=5)
    expected, expected_scores = cascade.generator_.recommend(users, n_recommendations=5)
    np.testing.assert_array_equal(items, expected)
    np.testing.assert_array_equal(scores, expected_scores)


def test_the_ranker_learns_only_from_held_out_interactions():
    X = trending_interactions()
    _GENERATOR_FITS.clear()
    cascade = Cascade(
        _RecordingPopular(), GeneratorScores(), _RecordingRanker(), n_retrieved=30, split=0.2
    ).fit(X)
    first, final = _GENERATOR_FITS
    # The last of every user's five rows is held out from the first fit, and only it.
    assert len(first) == len(X) - N_USERS
    np.testing.assert_array_equal(final, X)
    held = {tuple(row) for row in X.reshape(N_USERS, 5, 2)[:, -1]}
    assert not held & {tuple(row) for row in first}

    ranker = cascade.ranker_
    assert isinstance(ranker, _RecordingRanker)
    assert ranker.groups_.sum() == len(ranker.y_)
    # One relevant candidate per group: the held-out item, when it was a candidate.
    group_of_row = np.repeat(np.arange(len(ranker.groups_)), ranker.groups_)
    np.testing.assert_array_equal(np.bincount(group_of_row, weights=ranker.y_), 1.0)
    assert cascade.n_ranker_groups_ == len(ranker.groups_)


def test_split_takes_the_last_rows_of_each_user():
    X = np.array([["a", 1], ["b", 1], ["a", 2], ["a", 3], ["b", 2]], dtype=object)
    cascade = Cascade(MostPopularRecommender(), GeneratorScores(), _RecordingRanker(), split=0.5)
    train, held = cascade._split_rows(X, None, X[:, 0])
    assert held.tolist() == [3, 4]
    assert train.tolist() == [0, 1, 2]


def test_split_accepts_a_splitter():
    X = trending_interactions()
    cascade = _learning_cascade(split=ShuffleSplit(n_splits=1, test_size=0.2, random_state=0))
    items, _ = cascade.fit(X).recommend([0, 1], n_recommendations=3)
    assert items.shape == (2, 3)


def test_numeric_identifiers():
    X = trending_interactions()
    cascade = _learning_cascade().fit(X)
    assert cascade.item_ids_.dtype.kind == "i"
    items, _ = cascade.recommend(np.array([3]), n_recommendations=2)
    assert items.dtype.kind == "i"
    assert not set(items[0]) & set(X[X[:, 0] == 3, 1])


def test_predict_agrees_with_recommend():
    X = trending_interactions()
    cascade = _learning_cascade().fit(X)
    items, scores = cascade.recommend(np.array([5, 6]), n_recommendations=4)
    pairs = np.column_stack([np.repeat([5, 6], 4), items.ravel()])
    np.testing.assert_allclose(cascade.predict(pairs), scores.ravel())


def test_cold_users_are_served_when_the_generator_can():
    X = trending_interactions()
    cascade = _learning_cascade().fit(X)
    items, _ = cascade.recommend(np.array([10_000]), n_recommendations=3)
    assert np.isin(items, TRENDING).all()


#: Rows 0-4 are user 0, held out whole; rows 5-9 are user 1, whose last row is held out.
_HELD_OUT_WHOLE = np.array([0, 1, 2, 3, 4, 9])


def _held_out_whole(X):
    return _FixedSplit([(np.setdiff1d(np.arange(len(X)), _HELD_OUT_WHOLE), _HELD_OUT_WHOLE)])


def test_the_ranker_trains_only_on_users_with_training_rows():
    """A splitter holding out a user's whole history leaves most generators blind to them."""
    X = trending_interactions()
    cascade = Cascade(
        ItemKNNRecommender(),
        GeneratorScores(),
        _RecordingRanker(),
        n_retrieved=30,
        split=_held_out_whole(X),
    ).fit(X)
    assert cascade.n_ranker_groups_ == 1  # user 1 only


def test_the_ranker_trains_on_cold_users_when_the_generator_serves_them():
    """A popularity generator serves a user it never saw, so that user teaches the ranker."""
    X = trending_interactions()
    ranker = _RecordingRanker()
    cascade = Cascade(
        MostPopularRecommender(),
        GeneratorScores(),
        ranker,
        n_retrieved=30,
        split=_held_out_whole(X),
    ).fit(X)
    assert cascade.n_ranker_groups_ == 2  # user 0, cold, and user 1


def test_a_cold_start_split_trains_a_cold_start_ranker():
    """Every group the ranker learns from is a user held out whole, as a cold user is."""
    X = trending_interactions()
    cascade = _learning_cascade(split=ColdStartSplit(0.3, 0.0, random_state=0)).fit(X)
    assert cascade.n_ranker_groups_ == int(0.3 * N_USERS)
    items, _ = cascade.recommend(np.array([10_000]), n_recommendations=3)
    assert items.shape == (1, 3)


class _FixedSplit:
    def __init__(self, folds):
        self.folds = folds

    def split(self, X, y=None):
        return iter(self.folds)


def test_blocks_do_not_change_the_answer(monkeypatch):
    X = trending_interactions()
    cascade = _learning_cascade().fit(X)
    users = np.arange(N_USERS)
    whole = cascade.recommend(users, n_recommendations=3)
    monkeypatch.setattr(_cascade, "_PAIRS_PER_BLOCK", 70)  # two queries per block
    blocked = cascade.recommend(users, n_recommendations=3)
    np.testing.assert_array_equal(whole[0], blocked[0])
    np.testing.assert_array_equal(whole[1], blocked[1])


def test_short_histories_get_fewer_candidates_not_an_error():
    """A query with fewer eligible items than n_retrieved gets them all."""
    X = trending_interactions()
    cascade = _learning_cascade(n_retrieved=1000).fit(X)
    items, _ = cascade.recommend(np.array([0]), n_recommendations=25)
    assert items.shape == (1, 25)
    with pytest.raises(ValueError, match="query 0 has only 25 eligible items"):
        cascade.recommend(np.array([0]), n_recommendations=26)


def test_n_recommendations_cannot_exceed_n_retrieved():
    cascade = _learning_cascade(n_retrieved=20).fit(trending_interactions())
    with pytest.raises(ValueError, match="exceeds n_retrieved=20"):
        cascade.recommend([0], n_recommendations=21)


def test_pickles_and_nests():
    X = trending_interactions()
    switch = Switch(KnownUser(), _learning_cascade(), MostPopularRecommender()).fit(X)
    restored = pickle.loads(pickle.dumps(switch))
    queries = np.array([1, 2, 99_999])
    np.testing.assert_array_equal(
        restored.recommend(queries, n_recommendations=3)[0],
        switch.recommend(queries, n_recommendations=3)[0],
    )
    assert switch.get_params()["on_true__ranker__estimator__max_iter"] == 20


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"generator": KnownUser()}, TypeError, "generator must be a recommender"),
        ({"features": MostPopularRecommender()}, TypeError, "features must be a feature"),
        ({"ranker": LogisticRegression()}, TypeError, "ranker must be a ranker"),
        ({"n_retrieved": 0}, ValueError, "n_retrieved"),
        ({"split": 1.0}, ValueError, "split"),
    ],
)
def test_fit_validates(params, error, match):
    with pytest.raises(error, match=match):
        clone(_learning_cascade()).set_params(**params).fit(trending_interactions())


def test_fit_explains_when_nothing_is_held_out():
    X = np.array([[0, 1], [1, 2], [2, 3]])
    with pytest.raises(InsufficientDataError, match="to hold out"):
        _learning_cascade(split=0.5).fit(X)


def test_fit_explains_when_no_candidate_is_relevant():
    X = trending_interactions()
    with pytest.raises(InsufficientDataError, match="increase n_retrieved"):
        Cascade(MostPopularRecommender(), GeneratorScores(), _RecordingRanker(), n_retrieved=1).fit(
            X[~np.isin(X[:, 1], TRENDING) | (np.arange(len(X)) % 5 == 4)]
        )


def test_too_little_data_is_told_apart_from_a_wrong_argument():
    """Every shortage of data raises the one error a caller may fall back on."""
    X = trending_interactions()
    knn = Cascade(ItemKNNRecommender(), GeneratorScores(), _RecordingRanker(), split=0.5)
    # every held-out row belongs to a user the generator was not fitted on
    with pytest.raises(InsufficientDataError, match="a user with training rows"):
        clone(knn).set_params(split=ColdStartSplit(cold_users=0.5, test_size=0.0)).fit(X)
    # two items, each held out for half the users: the only candidate a user has left
    # is the one item they have not kept, which is the held-out one
    pairs = np.array([[u, i] for u in range(6) for i in ((0, 1) if u % 2 else (1, 0))])
    with pytest.raises(InsufficientDataError, match="Every generated candidate"):
        clone(knn).fit(pairs)
    with pytest.raises(InsufficientDataError, match="0 row"):
        clone(knn).fit(np.empty((0, 2), dtype=np.int64))
    # still a ValueError, as it always was, while a wrong argument is not this one
    assert issubclass(InsufficientDataError, ValueError)
    with pytest.raises(ValueError, match="n_retrieved") as wrong:
        clone(knn).set_params(n_retrieved=0).fit(X)
    assert not isinstance(wrong.value, InsufficientDataError)


def test_latest_interactions_split_caps_the_users_the_ranker_learns_from():
    """The ranker's groups are the held-out users, so capping those bounds its fit."""
    X = trending_interactions()
    every = _learning_cascade(split=LatestInteractionsSplit(0.2)).fit(X)
    same = _learning_cascade(split=0.2).fit(X)
    assert every.n_ranker_groups_ == same.n_ranker_groups_
    users = np.arange(N_USERS)
    np.testing.assert_array_equal(
        every.recommend(users, n_recommendations=5)[0],
        same.recommend(users, n_recommendations=5)[0],
    )
    capped = _learning_cascade(split=LatestInteractionsSplit(0.2, max_users=10)).fit(X)
    assert capped.n_ranker_groups_ <= 10 < every.n_ranker_groups_
    # everyone else keeps all of their rows for the generator
    assert capped.generator_.interactions_.nnz == every.generator_.interactions_.nnz


def test_integer_users_and_string_items_keep_their_types():
    """Candidate pairs mixing the two must not coerce users into strings."""
    X = trending_interactions()
    mixed = np.empty(X.shape, dtype=object)
    mixed[:, 0] = X[:, 0]
    mixed[:, 1] = np.array([f"item-{i}" for i in X[:, 1]], dtype=object)
    table = trending_table().astype(object)
    table[:, 0] = [f"item-{int(i)}" for i in table[:, 0]]
    cascade = _learning_cascade(
        features=ConcatFeatures([JoinStaticFeatures("item", table), GeneratorScores()])
    ).fit(mixed)
    items, _ = cascade.recommend(np.array([0, 1]), n_recommendations=3)
    trending = {f"item-{i}" for i in TRENDING}
    assert {item for row in items for item in row} <= trending


def test_a_cascade_can_generate_for_another():
    """Nesting asks the inner cascade how many items each query can get."""
    X = trending_interactions()
    # The outer holdout leaves the inner cascade four rows per user, so it holds out 1 in 4.
    inner = _learning_cascade(split=0.25)
    outer = Cascade(inner, GeneratorScores(), _RecordingRanker(), n_retrieved=25)
    outer.fit(X)
    counts = outer.generator_._count_eligible(np.array([0]))
    assert counts.tolist() == [25]  # 30 items, 5 seen; within the inner n_retrieved=30
    items, _ = outer.recommend(np.array([0, 1]), n_recommendations=3)
    expected, _ = outer.generator_.recommend(np.array([0, 1]), n_recommendations=3)
    np.testing.assert_array_equal(items, expected)


def _two_generators(**params):
    return _learning_cascade(generator=[ItemKNNRecommender(), MostPopularRecommender()], **params)


def _fitted_generators(X):
    return [ItemKNNRecommender().fit(X), MostPopularRecommender().fit(X)]


def test_merged_candidates_are_distinct_and_interleaved():
    X = trending_interactions()
    knn, popular = _fitted_generators(X)
    queries = np.arange(10)
    pairs, scores, groups, kept = retrieve_union(
        [knn, popular], queries, n_retrieved=6, min_retrieved=0
    )
    assert kept.tolist() == queries.tolist()
    assert (groups == 6).all()
    assert len({(u, i) for u, i in pairs.tolist()}) == len(pairs)
    starts = np.cumsum(groups) - groups
    first_knn = knn.recommend(queries, n_recommendations=1)[0][:, 0]
    first_popular = popular.recommend(queries, n_recommendations=1)[0][:, 0]
    assert (pairs[starts, 1] == first_knn).all()
    for start, top in zip(starts, first_popular, strict=True):
        assert top in pairs[start : start + 2, 1]
    assert scores.shape == (len(pairs), 2)


def test_every_generator_scores_every_candidate():
    """A candidate one generator did not retrieve still gets its predict score."""
    X = trending_interactions()
    knn, popular = _fitted_generators(X)
    pairs, scores, _, _ = retrieve_union(
        [knn, popular], np.array([0, 1]), n_retrieved=10, min_retrieved=0
    )
    np.testing.assert_allclose(scores[:, 0], knn.predict(pairs))
    np.testing.assert_allclose(scores[:, 1], popular.predict(pairs))


def test_a_cold_user_gets_the_candidates_of_generators_that_serve_it():
    X = trending_interactions()
    knn, popular = _fitted_generators(X)
    pairs, scores, groups, _ = retrieve_union(
        [knn, popular], np.array([10_000]), n_retrieved=5, min_retrieved=0
    )
    expected = popular.recommend(np.array([10_000]), n_recommendations=5)[0][0]
    assert pairs[:, 1].tolist() == expected.tolist()
    assert groups.tolist() == [5]
    assert np.isnan(scores[:, 0]).all()
    assert not np.isnan(scores[:, 1]).any()


def test_a_query_no_generator_can_fill_raises():
    X = trending_interactions()
    with pytest.raises(ValueError, match="query 3 has only 0 eligible items"):
        retrieve_union(
            [ItemKNNRecommender().fit(X)],
            np.array([0, 1, 2, 10_000]),
            n_retrieved=5,
            min_retrieved=1,
        )


def test_several_generators_learn_and_serve():
    X = trending_interactions()
    cascade = _two_generators().fit(X)
    assert [name for name, _ in cascade.generators_] == [
        "itemknnrecommender",
        "mostpopularrecommender",
    ]
    assert not hasattr(cascade, "generator_")
    items, _ = cascade.recommend(np.arange(N_USERS), n_recommendations=3)
    assert np.isin(items, TRENDING).mean() > 0.9
    cold, _ = cascade.recommend(np.array([10_000]), n_recommendations=3)
    assert cold.shape == (1, 3)


def test_several_generators_predict_agrees_with_recommend():
    X = trending_interactions()
    cascade = _two_generators().fit(X)
    items, scores = cascade.recommend(np.array([5, 6]), n_recommendations=4)
    pairs = np.column_stack([np.repeat([5, 6], 4), items.ravel()])
    np.testing.assert_allclose(cascade.predict(pairs), scores.ravel())


def test_several_generators_train_on_cold_users_when_one_serves_them():
    X = trending_interactions()
    cascade = Cascade(
        [ItemKNNRecommender(), MostPopularRecommender()],
        GeneratorScores(n_generators=2),
        _RecordingRanker(),
        n_retrieved=30,
        split=_held_out_whole(X),
    ).fit(X)
    assert cascade.n_ranker_groups_ == 2  # user 0, cold, and user 1


def test_several_generators_count_what_the_merge_holds():
    X = trending_interactions()
    cascade = _two_generators(n_retrieved=1000).fit(X)
    queries = np.array([0, 1, 10_000])
    counts = cascade._count_eligible(queries)
    assert counts.tolist() == [25, 25, 30]  # 30 items, 5 seen; a cold user has seen none
    with pytest.raises(ValueError, match="query 0 has only 25 eligible items"):
        cascade.recommend(queries, n_recommendations=26)


def test_several_generators_blocks_do_not_change_the_answer(monkeypatch):
    X = trending_interactions()
    cascade = _two_generators().fit(X)
    users = np.arange(N_USERS)
    whole = cascade.recommend(users, n_recommendations=3)
    monkeypatch.setattr(_cascade, "_PAIRS_PER_BLOCK", 120)  # two queries per block
    blocked = cascade.recommend(users, n_recommendations=3)
    np.testing.assert_array_equal(whole[0], blocked[0])
    np.testing.assert_array_equal(whole[1], blocked[1])


def test_several_generators_expose_nested_params():
    cascade = _two_generators()
    params = cascade.get_params()
    assert params["generator__itemknnrecommender__n_neighbors"] == (
        ItemKNNRecommender().n_neighbors
    )
    cascade.set_params(generator__itemknnrecommender__n_neighbors=7)
    assert cascade.generator[0].n_neighbors == 7
    cascade.set_params(generator__mostpopularrecommender=ItemKNNRecommender(n_neighbors=3))
    assert isinstance(cascade.generator[1], ItemKNNRecommender)
    # Bare generators are named by class, so two ItemKNNs are now numbered.
    assert clone(cascade).get_params()["generator__itemknnrecommender-2__n_neighbors"] == 3
    named = _learning_cascade(generator=[("a", ItemKNNRecommender()), ("b", ItemKNNRecommender())])
    named.set_params(generator__b__n_neighbors=2)
    assert named.generator[1][1].n_neighbors == 2
    assert (
        "generator__n_neighbors" in _learning_cascade(generator=ItemKNNRecommender()).get_params()
    )


def test_several_generators_predict_raises_for_what_no_generator_knows():
    cascade = _two_generators().fit(trending_interactions())
    with pytest.raises(ValueError, match="Unknown item identifiers"):
        cascade.predict([[0, 999]])
    scores = cascade.predict([[10_000, 1]])  # a cold user, whom popularity serves
    assert scores.shape == (1,)


def test_several_generators_pickle():
    X = trending_interactions()
    cascade = _two_generators().fit(X)
    restored = pickle.loads(pickle.dumps(cascade))
    queries = np.array([1, 2, 99_999])
    np.testing.assert_array_equal(
        restored.recommend(queries, n_recommendations=3)[0],
        cascade.recommend(queries, n_recommendations=3)[0],
    )


@pytest.mark.parametrize(
    ("generator", "error", "match"),
    [
        ([], ValueError, "at least one recommender"),
        ([ItemKNNRecommender(), KnownUser()], TypeError, "generator must be a recommender"),
    ],
)
def test_several_generators_validate(generator, error, match):
    with pytest.raises(error, match=match):
        _learning_cascade(generator=generator).fit(trending_interactions())


def test_generator_scores_take_one_column_per_generator():
    scores = GeneratorScores(n_generators=2).fit()
    out = scores.transform([["u", "a"], ["u", "b"]], scores=[[1.0, np.nan], [2.0, 3.0]])
    np.testing.assert_array_equal(out, [[1.0, np.nan], [2.0, 3.0]])
    assert scores.get_feature_names_out().tolist() == ["generator_score_0", "generator_score_1"]
    with pytest.raises(ValueError, match="expected n_generators=2"):
        scores.transform([["u", "a"]], scores=[1.0])
    with pytest.raises(ValueError, match="scores must have shape"):
        GeneratorScores().fit().transform([["u", "a"]], scores=[[[1.0]]])


def _timed(X):
    """``X`` with a clock: the row index, so each user's rows are in time order."""
    return np.column_stack([X, np.arange(len(X))])


def times_of_keys(keys):
    """Module level, so that a cascade holding it pickles: the key's time as a feature."""
    times = keys if keys.ndim == 1 else keys[:, -1]
    return np.nan_to_num(times.astype(float), nan=-1.0)


def _timed_cascade(callback, kind="user-item-time", **params):
    features = ConcatFeatures([JoinDynamicFeatures(kind, callback), GeneratorScores()])
    defaults: dict[str, Any] = {"n_retrieved": 10, "split": 0.2, "time": True}
    return Cascade(MostPopularRecommender(), features, _RecordingRanker(), **{**defaults, **params})


def _recording(calls, feature=times_of_keys):
    def callback(keys):
        calls.append(keys.copy())
        return feature(keys)

    return callback


def test_timed_cascade_ranks_each_held_out_user_as_of_their_first_held_out_time():
    X = _timed(trending_interactions())
    calls = []
    cascade = _timed_cascade(_recording(calls)).fit(X)
    training_keys = calls[0]
    # split=0.2 of five rows holds out each user's latest: the ranker's candidates for a
    # user are featurized as of that row's time, before which it had not happened.
    latest = {int(u): int(t) for u, _, t in X}
    np.testing.assert_array_equal(
        training_keys[:, 2], [latest[int(u)] for u in training_keys[:, 0]]
    )
    # Users none of whose candidates is relevant teach nothing and are dropped first.
    assert 0 < len(set(training_keys[:, 0].tolist())) <= len(latest)
    # The ranker was trained on exactly those times.
    ranker = cascade.ranker_
    assert set(ranker.X_[:, 0].tolist()) <= set(latest.values())


def test_timed_cascade_holds_out_the_latest_rows_by_time_not_row_order():
    X = np.array([["a", 1, 30], ["b", 1, 5], ["a", 2, 10], ["a", 3, 20], ["b", 2, 1]], dtype=object)
    cascade = _timed_cascade(times_of_keys, split=0.5)
    users, _, _, times = check_interactions(X, time=True)
    train, held = cascade._split_rows(X, None, users, times)
    assert held.tolist() == [0, 1]
    assert train.tolist() == [2, 3, 4]


def test_timed_cascade_recommends_as_of():
    X = _timed(trending_interactions())
    calls = []
    cascade = _timed_cascade(_recording(calls)).fit(X)
    users = np.arange(3)

    cascade.recommend(users, n_recommendations=2)
    assert np.isnan(calls[-1][:, 2].astype(float)).all()
    cascade.recommend(users, n_recommendations=2, as_of=7)
    assert set(calls[-1][:, 2].tolist()) == {7}
    cascade.recommend(users, n_recommendations=2, as_of=[1, np.nan, 3])
    by_user = {int(u): t for u, _, t in calls[-1]}
    assert by_user[0] == 1
    assert np.isnan(by_user[1])
    assert by_user[2] == 3


def test_timed_cascade_scores_with_the_as_of_feature():
    X = _timed(trending_interactions())
    cascade = _timed_cascade(times_of_keys).fit(X)
    # The recording ranker scores by the first feature, the time the pair is ranked as of.
    _, scores = cascade.recommend([0, 1], n_recommendations=2, as_of=[5, 9])
    np.testing.assert_array_equal(scores, [[5.0, 5.0], [9.0, 9.0]])
    np.testing.assert_array_equal(cascade.predict([[0, 0, 4], [1, 0, np.nan]]), [4.0, -1.0])


def test_timed_cascade_takes_exclusions_with_or_without_time():
    X = _timed(trending_interactions())
    cascade = _timed_cascade(times_of_keys).fit(X)
    first = cascade.recommend([0], n_recommendations=1)[0][0, 0]
    untimed, timed = [[0, first]], [[0, first, 123]]
    for excluded in (untimed, timed):
        items = cascade.recommend([0], n_recommendations=1, exclude_interactions=excluded)[0]
        assert items[0, 0] != first
    np.testing.assert_array_equal(
        cascade._count_eligible([0], exclude_interactions=timed),
        cascade._count_eligible([0], exclude_interactions=untimed),
    )


def test_timed_cascade_with_datetimes_and_string_ids():
    X = trending_interactions()
    timed = np.empty((len(X), 3), dtype=object)
    timed[:, 0] = [f"u{u}" for u in X[:, 0]]
    timed[:, 1] = [f"i{i}" for i in X[:, 1]]
    clock = np.datetime64("2024-01-01T00:00") + np.arange(len(X)).astype("timedelta64[m]")
    timed[:, 2] = clock.astype("datetime64[us]").astype(object)
    calls = []
    cascade = _timed_cascade(_recording(calls, lambda keys: np.zeros(len(keys)))).fit(timed)
    assert cascade.time_dtype_ == np.dtype("datetime64[ns]")
    cascade.recommend(["u0"], n_recommendations=1, as_of=np.datetime64("2024-02-01"))
    assert {t for _, _, t in calls[-1]} == {np.datetime64("2024-02-01", "ns")}
    with pytest.raises(TypeError, match="datetimes"):
        cascade.recommend(["u0"], n_recommendations=1, as_of=5)


def test_timed_cascade_pickles_and_clones():
    X = _timed(trending_interactions())
    cascade = _timed_cascade(times_of_keys).fit(X)
    restored = pickle.loads(pickle.dumps(cascade))
    np.testing.assert_array_equal(
        restored.recommend([0, 1], n_recommendations=2, as_of=3)[0],
        cascade.recommend([0, 1], n_recommendations=2, as_of=3)[0],
    )
    assert clone(cascade).time is True


def test_time_must_be_asked_for():
    X = _timed(trending_interactions())
    untimed = _timed_cascade(times_of_keys, kind="user", time=False)
    fitted = untimed.fit(X[:, :2])
    with pytest.raises(ValueError, match="as_of needs"):
        fitted.recommend([0], n_recommendations=1, as_of=3)
    with pytest.raises(ValueError, match="time must be a bool"):
        _timed_cascade(times_of_keys, time=1).fit(X)
    with pytest.raises(ValueError, match="3 columns"):
        _timed_cascade(times_of_keys).fit(X[:, :2])


def test_timed_cascade_is_evaluated_as_of_each_users_first_held_out_time():
    X = _timed(trending_interactions())
    calls = []
    cascade = _timed_cascade(_recording(calls)).fit(X)
    held = np.array([[0, 3, 50], [0, 4, 40], [1, 3, 70]])
    evaluate_recommender(cascade, held, metrics=[ndcg_at_k], k=2)
    by_user = {int(u): int(t) for u, _, t in calls[-1]}
    assert by_user == {0: 40, 1: 70}


def test_timed_cascade_cross_validates():
    X = _timed(trending_interactions())
    scores = cross_val_score(
        _timed_cascade(times_of_keys),
        X,
        scoring=make_recommender_scorer(ndcg_at_k, k=2),
        cv=WarmStartKFold(2, shuffle=True, random_state=0),
    )
    assert np.isfinite(scores).all()


# Business rules: postprocess. Callbacks are module-level so that a cascade holding one
# pickles.


def _group_of_row(groups):
    return np.repeat(np.arange(len(groups)), groups)


def _keep(pairs, scores, groups, keep):
    sizes = np.bincount(_group_of_row(groups)[keep], minlength=len(groups))
    return pairs[keep], scores[keep], sizes


def drop_trending(pairs, scores, groups):
    return _keep(pairs, scores, groups, ~np.isin(pairs[:, 1], TRENDING))


def promote_item_0(pairs, scores, groups):
    """Item 0 first wherever it is a candidate, the rest in the ranker's order."""
    order = np.lexsort((pairs[:, 1] != 0, _group_of_row(groups)))
    return pairs[order], scores[order], groups


def keep_two(pairs, scores, groups):
    starts = np.concatenate([[0], np.cumsum(groups)[:-1]])
    rank = np.arange(len(pairs)) - np.repeat(starts, groups)
    return _keep(pairs, scores, groups, rank < 2)


def sponsor_first(pairs, scores, groups):
    """Put a sponsored item, never a candidate, in the first slot of every list."""
    queries = pairs[np.concatenate([[0], np.cumsum(groups)[:-1]]), 0]
    sponsored = np.column_stack([queries, np.full(len(queries), 999)])
    rows = np.concatenate([sponsored, pairs[:, :2]])
    new_scores = np.concatenate([np.full(len(queries), np.inf), scores])
    order = np.argsort(
        np.concatenate([np.arange(len(queries)), _group_of_row(groups)]), kind="stable"
    )
    return rows[order], new_scores[order], groups + 1


def test_postprocess_changes_what_is_recommended():
    X = trending_interactions()
    users = np.arange(N_USERS)
    plain = _learning_cascade().fit(X)
    dropped = _learning_cascade(postprocess=drop_trending).fit(X)
    items, _ = dropped.recommend(users, n_recommendations=3)
    assert not np.isin(items, TRENDING).any()
    # The ranker learned what it learns without the rules; the rules only serve.
    np.testing.assert_array_equal(dropped.predict(X[:50]), plain.predict(X[:50]))
    promoted, _ = (
        _learning_cascade(postprocess=promote_item_0)
        .fit(X)
        .recommend(users, n_recommendations=3, exclude_seen=False)
    )
    assert (promoted[:, 0] == 0).all()


def test_postprocess_backfills_from_the_ranked_list():
    X = trending_interactions()
    cascade = _learning_cascade(postprocess=drop_trending).fit(X)
    plain = clone(cascade).set_params(postprocess=None).fit(X)
    users = np.arange(N_USERS)
    ranked, _ = plain.recommend(users, n_recommendations=10)
    served, served_scores = cascade.recommend(users, n_recommendations=3)
    for row, kept, kept_scores in zip(ranked, served, served_scores, strict=True):
        assert kept.tolist() == [i for i in row if i not in TRENDING][:3]
        assert (np.diff(kept_scores) <= 0).all()


def test_postprocess_sees_every_candidate_best_first():
    calls = []

    def record(pairs, scores, groups):
        calls.append((pairs.copy(), scores.copy(), groups.copy()))
        return pairs, scores, groups

    X = trending_interactions()
    cascade = _learning_cascade(postprocess=record).fit(X)
    cascade.recommend([0, 1], n_recommendations=2)
    pairs, scores, groups = calls[-1]
    assert groups.tolist() == cascade._count_eligible([0, 1]).tolist()
    assert pairs[:, 0].tolist() == [0] * groups[0] + [1] * groups[1]
    for group in np.split(scores, np.cumsum(groups)[:-1]):
        assert (np.diff(group) <= 0).all()


def test_postprocess_may_serve_items_that_were_not_candidates():
    X = trending_interactions()
    cascade = _learning_cascade(postprocess=sponsor_first).fit(X)
    items, scores = cascade.recommend([0, 1], n_recommendations=2)
    assert items[:, 0].tolist() == [999, 999]
    assert np.isinf(scores[:, 0]).all()


def test_postprocess_serves_new_string_items_whole():
    def sponsor(pairs, scores, groups):
        pairs = pairs.astype(object)
        pairs[np.concatenate([[0], np.cumsum(groups)[:-1]]), 1] = "sponsored-item"
        return pairs, scores, groups

    X = trending_interactions().astype(str)
    items, _ = (
        _learning_cascade(features=GeneratorScores(), postprocess=sponsor)
        .fit(X)
        .recommend(["0"], n_recommendations=2)
    )
    assert items[0, 0] == "sponsored-item"


def test_postprocess_that_leaves_too_few_raises():
    cascade = _learning_cascade(postprocess=keep_two).fit(trending_interactions())
    assert cascade.recommend([0], n_recommendations=2)[0].shape == (1, 2)
    with pytest.raises(ValueError, match="left query 0 with 2 items, fewer than n_rec"):
        cascade.recommend([0], n_recommendations=3)


def _swap_queries(pairs, scores, groups):
    pairs = pairs.copy()
    pairs[:, 0] = pairs[::-1, 0]
    return pairs, scores, groups


@pytest.mark.parametrize(
    ("postprocess", "error", "match"),
    [
        (lambda p, s, g: (p, s), TypeError, "tuple"),
        (lambda p, s, g: (p, s, g[:1]), ValueError, "one integer group size per query"),
        (lambda p, s, g: (p[1:], s[1:], g), ValueError, "must agree"),
        (lambda p, s, g: (p, np.full(len(s), np.nan), g), ValueError, "NaN"),
        (_swap_queries, ValueError, "in its own group"),
    ],
)
def test_postprocess_output_is_checked(postprocess, error, match):
    cascade = _learning_cascade(postprocess=postprocess).fit(trending_interactions())
    with pytest.raises(error, match=match):
        cascade.recommend([0, 1], n_recommendations=2)


def test_postprocess_must_be_callable():
    with pytest.raises(TypeError, match="postprocess must be callable"):
        _learning_cascade(postprocess="drop").fit(trending_interactions())


def test_postprocess_blocks_do_not_change_the_answer(monkeypatch):
    X = trending_interactions()
    cascade = _learning_cascade(postprocess=drop_trending).fit(X)
    users = np.arange(N_USERS)
    whole = cascade.recommend(users, n_recommendations=3)
    monkeypatch.setattr(_cascade, "_PAIRS_PER_BLOCK", 70)
    blocked = cascade.recommend(users, n_recommendations=3)
    np.testing.assert_array_equal(whole[0], blocked[0])
    np.testing.assert_array_equal(whole[1], blocked[1])


def test_postprocess_pickles_and_clones():
    X = trending_interactions()
    cascade = _learning_cascade(postprocess=drop_trending).fit(X)
    restored = pickle.loads(pickle.dumps(cascade))
    np.testing.assert_array_equal(
        restored.recommend([0, 1], n_recommendations=3)[0],
        cascade.recommend([0, 1], n_recommendations=3)[0],
    )
    assert clone(cascade).postprocess is drop_trending


def test_timed_postprocess_sees_the_time():
    calls = []

    def record(pairs, scores, groups):
        calls.append(pairs.copy())
        return pairs, scores, groups

    X = _timed(trending_interactions())
    cascade = _timed_cascade(times_of_keys, postprocess=record).fit(X)
    cascade.recommend([0, 1], n_recommendations=2, as_of=[5, 9])
    assert calls[-1].shape[1] == 3
    assert {int(u): int(t) for u, _, t in calls[-1]} == {0: 5, 1: 9}


def test_timed_postprocess_keeps_the_item_dtype():
    X = _timed(trending_interactions())
    plain = _timed_cascade(times_of_keys).fit(X)
    ruled = clone(plain).set_params(postprocess=drop_trending).fit(X)
    for as_of in (None, [5, 9]):
        expected, _ = plain.recommend([0, 1], n_recommendations=2, as_of=as_of)
        items, _ = ruled.recommend([0, 1], n_recommendations=2, as_of=as_of)
        assert items.dtype == expected.dtype
