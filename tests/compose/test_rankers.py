import pickle
from collections.abc import Callable

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC

from skrecsys._typing import override
from skrecsys.base import FeaturesMixin, RankerMixin, is_ranker
from skrecsys.compose import (
    AugmentedRanker,
    BlendRanker,
    Cascade,
    ConcatFeatures,
    GeneratorScores,
    GroupRanker,
    InteractionCounts,
    JoinDynamicFeatures,
    JoinStaticFeatures,
    PointwiseRanker,
    ReciprocalRankRanker,
)
from skrecsys.compose._rankers import normalize_per_group
from skrecsys.recommendation import MostPopularRecommender
from tests.compose._data import N_USERS, TRENDING, trending_interactions, trending_table

X = np.array([[0.0], [1.0], [0.2], [0.9], [0.1], [0.7]])
Y = np.array([0, 1, 0, 1, 0, 1])
GROUPS = np.array([2, 2, 2])


@pytest.mark.parametrize(
    "estimator",
    [
        LogisticRegression(),
        LinearSVC(),
        LinearRegression(),
        HistGradientBoostingClassifier(min_samples_leaf=1),
    ],
    ids=lambda e: type(e).__name__,
)
def test_pointwise_ranker_orders_by_relevance(estimator):
    ranker = PointwiseRanker(estimator).fit(X, Y, groups=GROUPS)
    scores = ranker.predict(X, groups=GROUPS)
    assert scores.shape == (6,)
    assert scores.dtype == np.float64
    assert np.all(scores[Y == 1].min() > scores[Y == 0].max())


def test_pointwise_ranker_takes_missing_features():
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    ranker = PointwiseRanker(HistGradientBoostingClassifier()).fit(X_nan, Y, groups=GROUPS)
    assert np.isfinite(ranker.predict(X_nan, groups=GROUPS)).all()


class _RecordingRanker(BaseEstimator):
    """Stands in for LGBMRanker: records the group sizes it is given."""

    def fit(self, X, y, group=None):
        self.group_ = group
        return self

    def predict(self, X):
        return X[:, 0]


def test_group_ranker_forwards_group_sizes():
    ranker = GroupRanker(_RecordingRanker()).fit(X, Y, groups=GROUPS)
    assert isinstance(ranker.estimator_, _RecordingRanker)
    np.testing.assert_array_equal(ranker.estimator_.group_, [2, 2, 2])
    np.testing.assert_array_equal(ranker.predict(X, groups=GROUPS), X[:, 0])


@pytest.mark.parametrize(
    ("groups", "match"),
    [([2, 2], "summing"), ([6.0], "integer"), ([0, 6], "positive"), ([[6]], "one-dimensional")],
)
def test_groups_are_validated(groups, match):
    with pytest.raises(ValueError, match=match):
        PointwiseRanker(LogisticRegression()).fit(X, Y, groups=groups)


def test_tags():
    assert is_ranker(PointwiseRanker(LogisticRegression()))
    assert is_ranker(GroupRanker(_RecordingRanker()))
    assert not hasattr(PointwiseRanker(LogisticRegression()), "score")


# --- BlendRanker --------------------------------------------------------------------------


#: Every fit of a recording ranker, as (query ids of its rows, its group sizes). Module
#: level because a blend clones its rankers, and a clone would copy a list parameter.
_FITS: list[tuple[np.ndarray, np.ndarray]] = []


class _Recording(RankerMixin, BaseEstimator):
    """Scores by the first feature, and optionally logs the rows of every fit to ``_FITS``."""

    def __init__(self, *, record=False, sign=1.0):
        self.record = record
        self.sign = sign

    def fit(self, X, y, *, groups):
        if self.record:
            _FITS.append((np.asarray(X)[:, 1].copy(), np.asarray(groups).copy()))
        return self

    def predict(self, X, *, groups):
        return self.sign * np.asarray(X)[:, 0]


def _blend_data(n_groups=6, size=4):
    """``n_groups`` queries; feature 0 is relevance plus noise, feature 1 the query id."""
    rng = np.random.default_rng(0)
    y = np.tile([0, 0, 0, 1], n_groups).astype(float)
    X = np.column_stack([y + rng.normal(0, 0.3, len(y)), np.repeat(np.arange(n_groups), size)])
    return X, y, np.full(n_groups, size)


def test_rank_normalization_is_the_percentile_within_each_group():
    scores = np.array([[10.0], [30.0], [20.0], [5.0], [5.0], [7.0]])
    out = normalize_per_group(scores, np.array([3, 1, 2]), "rank")
    np.testing.assert_allclose(out.ravel(), [0.0, 1.0, 0.5, 0.5, 0.0, 1.0])


def test_rank_normalization_gives_ties_their_average():
    out = normalize_per_group(np.array([[1.0], [1.0], [2.0]]), np.array([3]), "rank")
    np.testing.assert_allclose(out.ravel(), [0.25, 0.25, 1.0])


def test_zscore_normalization_standardizes_each_group_and_zeroes_a_constant_one():
    scores = np.array([[1.0], [3.0], [4.0], [4.0]])
    out = normalize_per_group(scores, np.array([2, 2]), "zscore")
    np.testing.assert_allclose(out.ravel(), [-1.0, 1.0, 0.0, 0.0])


def test_normalization_works_column_by_column():
    scores = np.array([[1.0, 5.0], [2.0, 3.0]])
    out = normalize_per_group(scores, np.array([2]), "rank")
    np.testing.assert_allclose(out, [[0.0, 1.0], [1.0, 0.0]])


def test_an_average_blend_is_the_weighted_mean_of_normalized_scores():
    X, y, groups = _blend_data()
    blend = BlendRanker(
        [("up", _Recording()), ("down", _Recording(sign=-1.0))], blender=None, weights=[3, 1]
    ).fit(X, y, groups=groups)
    up = normalize_per_group(X[:, :1], groups, "rank")
    np.testing.assert_allclose(blend.predict(X, groups=groups), (3 * up + (1 - up)).ravel() / 4)
    assert blend.blender_ is None


def test_stacking_scores_every_group_with_rankers_that_never_saw_it():
    X, y, groups = _blend_data(n_groups=6)
    _FITS.clear()
    BlendRanker([_Recording(record=True)], cv=3, random_state=0).fit(X, y, groups=groups)
    assert len(_FITS) == 4
    folds, refit = _FITS[:3], _FITS[3]
    seen = [set(np.unique(rows).tolist()) for rows, _ in folds]
    # Each fold fit leaves out two whole groups, and together they leave out every group.
    assert all(len(ids) == 4 for ids in seen)
    assert set.union(*({*range(6)} - ids for ids in seen)) == set(range(6))
    for rows, sizes in folds:
        assert sizes.tolist() == [4] * 4
        assert len(rows) == 16
    # The last fit is the refit on every row, for predict.
    assert sorted(np.unique(refit[0]).tolist()) == list(range(6))


def test_stacking_learns_to_trust_the_ranker_that_is_right():
    X, y, groups = _blend_data(n_groups=30)
    blend = BlendRanker(
        [("right", _Recording()), ("wrong", _Recording(sign=-1.0))], random_state=0
    ).fit(X, y, groups=groups)
    scores = blend.predict(X, groups=groups).reshape(-1, 4)
    assert (scores.argmax(axis=1) == 3).mean() > 0.9
    assert isinstance(blend.blender_, PointwiseRanker)
    logistic = blend.blender_.estimator_
    assert isinstance(logistic, LogisticRegression)
    coef = logistic.coef_.ravel()
    assert coef[0] > 0 > coef[1]


def test_passthrough_hands_the_blender_the_features_too():
    X, y, groups = _blend_data(n_groups=9)
    blend = BlendRanker([_Recording()], passthrough=True, random_state=0).fit(X, y, groups=groups)
    assert isinstance(blend.blender_, PointwiseRanker)
    logistic = blend.blender_.estimator_
    assert isinstance(logistic, LogisticRegression)
    assert logistic.coef_.shape[1] == 1 + X.shape[1]


def test_a_blend_is_seeded():
    X, y, groups = _blend_data(n_groups=9)

    def scores(seed):
        blend = BlendRanker(
            [PointwiseRanker(LogisticRegression()), PointwiseRanker(LinearSVC())],
            random_state=seed,
        ).fit(X, y, groups=groups)
        return blend.predict(X, groups=groups)

    np.testing.assert_array_equal(scores(0), scores(0))


def test_blend_names_rankers_and_exposes_nested_params():
    blend = BlendRanker([PointwiseRanker(LogisticRegression()), PointwiseRanker(LinearSVC())])
    params = blend.get_params()
    assert "pointwiseranker-1__estimator__C" in params
    blend.set_params(**{"pointwiseranker-2__estimator__C": 5.0, "cv": 4})
    assert blend.get_params()["pointwiseranker-2__estimator__C"] == 5.0
    assert blend.cv == 4
    copy = clone(blend)
    assert copy.get_params()["pointwiseranker-2__estimator__C"] == 5.0


def test_blend_can_replace_a_ranker_by_name():
    blend = BlendRanker([("a", PointwiseRanker(LogisticRegression()))])
    other = PointwiseRanker(LinearSVC())
    blend.set_params(a=other)
    assert blend.rankers == [("a", other)]


def test_a_fitted_blend_pickles():
    X, y, groups = _blend_data(n_groups=9)
    blend = BlendRanker([PointwiseRanker(LogisticRegression())], random_state=0).fit(
        X, y, groups=groups
    )
    restored = pickle.loads(pickle.dumps(blend))
    np.testing.assert_array_equal(
        restored.predict(X, groups=groups), blend.predict(X, groups=groups)
    )


def test_a_blend_ranks_a_cascade():
    X = trending_interactions()
    cascade = Cascade(
        MostPopularRecommender(),
        ConcatFeatures([JoinStaticFeatures("item", trending_table()), GeneratorScores()]),
        BlendRanker(
            [
                PointwiseRanker(HistGradientBoostingClassifier(max_iter=20)),
                PointwiseRanker(LogisticRegression()),
            ],
            random_state=0,
        ),
        n_retrieved=30,
    ).fit(X)
    items, _ = cascade.recommend(np.arange(N_USERS), n_recommendations=3)
    assert np.isin(items, TRENDING).mean() > 0.9


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"rankers": []}, ValueError, "at least one ranker"),
        ({"rankers": [LogisticRegression()]}, TypeError, "not a ranker"),
        ({"normalize": "minmax"}, ValueError, "normalize"),
        ({"blender": "ridge"}, ValueError, "blender must be"),
        ({"blender": LogisticRegression()}, TypeError, "blender"),
        ({"cv": 1}, ValueError, "cv"),
        ({"cv": 10}, ValueError, "at least as many groups"),
        ({"blender": None, "weights": [1, 2]}, ValueError, "one entry per ranker"),
    ],
)
def test_blend_validates(params, error, match):
    X, y, groups = _blend_data(n_groups=6)
    blend = BlendRanker([PointwiseRanker(LogisticRegression())])
    # Through set_params, which checks a replaced ranker list itself and leaves the rest
    # to fit -- either way the mistake is refused before anything is fitted.
    with pytest.raises(error, match=match):
        blend.set_params(**params).fit(X, y, groups=groups)


def test_blend_checks_the_feature_count():
    X, y, groups = _blend_data(n_groups=6)
    blend = BlendRanker([PointwiseRanker(LogisticRegression())], blender=None).fit(
        X, y, groups=groups
    )
    with pytest.raises(ValueError, match="features"):
        blend.predict(X[:, :1], groups=groups)


def test_blend_is_a_ranker():
    assert is_ranker(BlendRanker([PointwiseRanker(LogisticRegression())]))


# --- AugmentedRanker ----------------------------------------------------------------------


def _trending_flags(items):
    """Module level, so that a cascade holding it pickles: 1 for a trending item."""
    return np.isin(items, TRENDING).astype(float)


def _augmented(ranker=None):
    """A ranker seeing the trending flag of each item, which the shared features lack."""
    return AugmentedRanker(
        ranker or PointwiseRanker(HistGradientBoostingClassifier(max_iter=20)),
        JoinDynamicFeatures("item", _trending_flags, n_features=1),
    )


def _generator_scores_cascade(ranker, **params):
    """Popularity candidates, whose only shared feature cannot tell the trend apart."""
    return Cascade(MostPopularRecommender(), GeneratorScores(), ranker, n_retrieved=30, **params)


def test_injected_features_reach_only_their_ranker():
    cascade = _generator_scores_cascade(
        BlendRanker(
            [("plain", PointwiseRanker(LogisticRegression())), ("augmented", _augmented())],
            random_state=0,
        )
    ).fit(trending_interactions())
    rankers = dict(cascade.ranker_.rankers_)
    assert rankers["plain"].estimator_.n_features_in_ == 1
    augmented = rankers["augmented"]
    assert (augmented.n_features_in_, augmented.n_injected_features_) == (1, 1)
    assert augmented.ranker_.estimator_.n_features_in_ == 2
    items, _ = cascade.recommend(np.arange(N_USERS), n_recommendations=3)
    assert np.isin(items, TRENDING).mean() > 0.9


def test_a_blend_without_the_injected_features_misses_the_trend():
    cascade = _generator_scores_cascade(
        BlendRanker([PointwiseRanker(LogisticRegression())], random_state=0)
    ).fit(trending_interactions())
    items, _ = cascade.recommend(np.arange(N_USERS), n_recommendations=3)
    assert np.isin(items, TRENDING).mean() < 0.5


@pytest.mark.parametrize(
    "ranker",
    [
        _augmented(),
        ReciprocalRankRanker([_augmented()]),
        BlendRanker([_augmented(), _augmented()], blender=None),
        BlendRanker([_augmented()], blender=_augmented(PointwiseRanker(LogisticRegression()))),
        _augmented(BlendRanker([_augmented()], blender=None)),
    ],
    ids=["alone", "fused", "averaged", "augmented-blender", "nested"],
)
def test_an_augmented_ranker_ranks_a_cascade_wherever_it_sits(ranker):
    cascade = _generator_scores_cascade(ranker).fit(trending_interactions())
    items, _ = cascade.recommend(np.arange(N_USERS), n_recommendations=3)
    assert np.isin(items, TRENDING).mean() > 0.9
    restored = pickle.loads(pickle.dumps(cascade))
    np.testing.assert_array_equal(
        restored.recommend(np.arange(5), n_recommendations=3)[1],
        cascade.recommend(np.arange(5), n_recommendations=3)[1],
    )


class _FitLog(FeaturesMixin, BaseEstimator):
    """A column of zeros, telling ``fitted`` how many interactions it is fitted on and
    ``joined`` how many pairs it joins onto -- callables, which a clone keeps."""

    def __init__(self, fitted: Callable[[int], None], joined: Callable[[int], None]):
        self.fitted = fitted
        self.joined = joined

    def fit(self, X, y=None):
        self.fitted(len(X))
        return self

    @override
    def transform(self, pairs, *, scores=None):
        self.joined(len(pairs))
        return np.zeros((len(pairs), 1))


def test_injected_features_are_fitted_once_for_training_and_once_for_serving():
    fitted, joined = [], []
    X = trending_interactions()
    blend = BlendRanker(
        [
            AugmentedRanker(
                PointwiseRanker(LogisticRegression()), _FitLog(fitted.append, joined.append)
            )
        ],
        cv=3,
        random_state=0,
    )
    _generator_scores_cascade(blend, split=0.2).fit(X)
    # Once on the interactions the ranker is not labelled from, shared by every fold,
    # then once on all of them for serving.
    assert fitted == [len(X) - N_USERS, len(X)]
    # Every fold fits on the other folds' rows and scores its own; then the refit on all.
    assert len(joined) == 2 * 3 + 1
    fold_fits, held_out, n_rows = joined[0:6:2], joined[1:6:2], joined[-1]
    assert sum(held_out) == n_rows
    assert [fit + held for fit, held in zip(fold_fits, held_out, strict=True)] == [n_rows] * 3


def test_injected_features_see_the_time_of_each_pair():
    X = trending_interactions()
    X = np.column_stack([X, np.arange(len(X))])
    calls = []

    def times(keys):
        calls.append(keys.copy())
        return np.nan_to_num(keys[:, 1:].astype(float), nan=-1.0)

    ranker = AugmentedRanker(
        PointwiseRanker(LogisticRegression()), JoinDynamicFeatures("item-time", times)
    )
    cascade = _generator_scores_cascade(ranker, split=0.2, time=True).fit(X)
    assert calls[0].shape[1] == 2
    assert not np.isnan(calls[0][:, 1].astype(float)).any()
    cascade.recommend([0], n_recommendations=1)
    assert np.isnan(calls[-1][:, 1].astype(float)).all()


def test_injected_features_serve_from_every_interaction():
    X = trending_interactions()
    ranker = AugmentedRanker(PointwiseRanker(LogisticRegression()), InteractionCounts("item"))
    cascade = _generator_scores_cascade(ranker).fit(X)
    assert cascade.ranker_.features_.counts_.sum() == len(X)


def test_an_augmented_ranker_needs_a_cascade():
    X, y, groups = _blend_data()
    ranker = AugmentedRanker(PointwiseRanker(LogisticRegression()), GeneratorScores())
    with pytest.raises(ValueError, match="only a Cascade supplies"):
        ranker.fit(X, y, groups=groups)
    with pytest.raises(ValueError, match="only a Cascade supplies"):
        BlendRanker([ranker], blender=None).fit(X, y, groups=groups)


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"ranker": LogisticRegression()}, TypeError, "ranker"),
        ({"features": PointwiseRanker(LogisticRegression())}, TypeError, "feature component"),
    ],
)
def test_an_augmented_ranker_validates(params, error, match):
    with pytest.raises(error, match=match):
        _generator_scores_cascade(_augmented().set_params(**params)).fit(trending_interactions())


def test_an_augmented_ranker_exposes_nested_params():
    blend = BlendRanker([("aug", _augmented())])
    params = blend.get_params()
    assert params["aug__features__kind"] == "item"
    assert params["aug__ranker__estimator__max_iter"] == 20
    blend.set_params(aug__features__n_features=None)
    assert clone(blend).rankers[0][1].features.n_features is None
    assert is_ranker(_augmented())
