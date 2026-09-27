"""A blend of every installed integration ranker, the way the reranking benchmark uses one."""

import pickle

import numpy as np
import pytest

from skrecsys.compose import (
    AugmentedRanker,
    BlendRanker,
    Cascade,
    ConcatFeatures,
    GeneratorScores,
    JoinStaticFeatures,
)
from skrecsys.recommendation import MostPopularRecommender
from tests.compose._data import N_USERS, TRENDING, trending_interactions, trending_table
from tests.integrations._rankers import RANKERS, make_ranker


def _boosters():
    """Every integration ranker whose extra is installed; skips when fewer than two are."""
    found = []
    for spec in RANKERS:
        try:
            found.append((spec[0], make_ranker(*spec, random_state=0)))
        except pytest.skip.Exception:
            continue
    if len(found) < 2:
        pytest.skip("a blend needs at least two integration extras installed")
    return found


@pytest.mark.parametrize("blender", ["logistic", None])
def test_a_blend_of_boosters_ranks_a_cascade_and_pickles(blender):
    cascade = Cascade(
        MostPopularRecommender(),
        ConcatFeatures([JoinStaticFeatures("item", trending_table()), GeneratorScores()]),
        BlendRanker(_boosters(), blender=blender, random_state=0),
        n_retrieved=30,
    ).fit(trending_interactions())
    items, scores = cascade.recommend(np.arange(N_USERS), n_recommendations=3)
    assert np.isin(items, TRENDING).mean() > 0.9
    restored = pickle.loads(pickle.dumps(cascade))
    np.testing.assert_array_equal(
        restored.recommend(np.arange(5), n_recommendations=3)[1], scores[:5]
    )


def _trending_share(augmented):
    """Share of trending items served by a blend whose first booster alone may see the table.

    Only that booster can tell trending items apart; the others see generator scores alone.
    """
    (name, first), *rest = _boosters()
    if augmented:
        first = AugmentedRanker(first, JoinStaticFeatures("item", trending_table()))
    cascade = Cascade(
        MostPopularRecommender(),
        GeneratorScores(),
        BlendRanker([(name, first), *rest], random_state=0),
        n_retrieved=30,
    ).fit(trending_interactions())
    items, _ = cascade.recommend(np.arange(N_USERS), n_recommendations=3)
    blend = cascade.ranker_
    assert isinstance(blend, BlendRanker)
    return np.isin(items, TRENDING).mean(), dict(blend.rankers_)[name]


def test_a_booster_with_features_of_its_own_ranks_a_cascade():
    # Not an absolute bar: the rank-normalized noise of the other boosters dilutes the one
    # that knows, by how much depends on which extras are installed.
    share, first = _trending_share(augmented=True)
    baseline, _ = _trending_share(augmented=False)
    assert first.n_injected_features_ == 1
    assert share > baseline + 0.4
