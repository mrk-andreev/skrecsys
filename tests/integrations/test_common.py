"""The contract every integration ranker keeps, run for each installed extra."""

import pickle

import numpy as np
import pytest
from sklearn.base import clone

from skrecsys.base import is_ranker
from skrecsys.compose import (
    Cascade,
    ConcatFeatures,
    GeneratorScores,
    JoinStaticFeatures,
)
from skrecsys.recommendation import MostPopularRecommender
from tests.compose._data import (
    N_USERS,
    TRENDING,
    trending_interactions,
    trending_table,
)
from tests.integrations._rankers import RANKERS, make_ranker

X = np.array([[0.0], [1.0], [0.2], [0.9], [0.1], [np.nan]])
Y = np.array([0, 1, 0, 1, 0, 1])
GROUPS = np.array([2, 2, 2])

pytestmark = pytest.mark.parametrize("spec", RANKERS, ids=lambda spec: spec[0])


def test_is_a_ranker_without_a_score(spec):
    ranker = make_ranker(*spec)
    assert is_ranker(ranker)
    assert not hasattr(ranker, "score")


def test_fits_and_scores_every_row(spec):
    ranker = make_ranker(*spec, random_state=0)
    scores = ranker.fit(X, Y, groups=GROUPS).predict(X, groups=GROUPS)
    assert scores.shape == (6,)
    assert np.isfinite(scores).all()
    assert scores[1] > scores[0]


def test_seeded_fits_agree(spec):
    first = make_ranker(*spec, random_state=3).fit(X, Y, groups=GROUPS)
    second = clone(first).fit(X, Y, groups=GROUPS)
    np.testing.assert_array_equal(first.predict(X), second.predict(X))


def test_an_unseeded_fit_works(spec):
    assert make_ranker(*spec).fit(X, Y, groups=GROUPS).predict(X).shape == (6,)


def test_checks_the_feature_count(spec):
    ranker = make_ranker(*spec, random_state=0).fit(X, Y, groups=GROUPS)
    with pytest.raises(ValueError, match="fitted with 1"):
        ranker.predict(np.zeros((2, 2)))


def test_groups_must_partition_the_rows(spec):
    with pytest.raises(ValueError, match="summing"):
        make_ranker(*spec).fit(X, Y, groups=[2, 2])


def _cascade(spec):
    return Cascade(
        MostPopularRecommender(),
        ConcatFeatures([JoinStaticFeatures("item", trending_table()), GeneratorScores()]),
        make_ranker(*spec, random_state=0),
        n_retrieved=30,
    )


def test_ranks_a_cascade(spec):
    cascade = _cascade(spec).fit(trending_interactions())
    items, _ = cascade.recommend(np.arange(N_USERS), n_recommendations=3)
    assert np.isin(items, TRENDING).mean() > 0.9


def test_a_fitted_cascade_pickles(spec):
    cascade = _cascade(spec).fit(trending_interactions())
    restored = pickle.loads(pickle.dumps(cascade))
    users = np.arange(5)
    np.testing.assert_array_equal(
        restored.recommend(users, n_recommendations=3)[1],
        cascade.recommend(users, n_recommendations=3)[1],
    )
    assert cascade.get_params()["ranker__random_state"] == 0
