import pickle

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC

from skrecsys.base import RankerMixin, is_ranker
from skrecsys.compose import (
    BlendRanker,
    Cascade,
    ConcatFeatures,
    GeneratorScores,
    GroupRanker,
    JoinStaticFeatures,
    PointwiseRanker,
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
