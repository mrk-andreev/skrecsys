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
from skrecsys.tune import AutoTune, Categorical, search_space
from tests.compose._data import (
    N_USERS,
    TRENDING,
    trending_interactions,
    trending_table,
)
from tests.integrations._rankers import RANKERS, library_params, make_ranker

X = np.array([[0.0], [1.0], [0.2], [0.9], [0.1], [np.nan]])
Y = np.array([0, 1, 0, 1, 0, 1])
GROUPS = np.array([2, 2, 2])

pytestmark = pytest.mark.parametrize("spec", RANKERS, ids=lambda spec: spec[0])


def test_is_a_ranker_without_a_score(spec):
    ranker = make_ranker(*spec)
    assert is_ranker(ranker)
    assert not hasattr(ranker, "score")


def test_contributions_add_up_to_the_score(spec):
    ranker = make_ranker(*spec, random_state=0).fit(X, Y, groups=GROUPS)
    contributions = ranker._contributions(X)
    assert contributions.shape == (len(X), X.shape[1] + 1)
    np.testing.assert_allclose(
        contributions.sum(axis=1), ranker.predict(X, groups=GROUPS), rtol=1e-5, atol=1e-6
    )


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


#: A library parameter the ranker does not expose, forwarded through extra_params.
EXTRA = {
    "catboost": {"bootstrap_type": "Bernoulli"},
    "xgboost": {"max_bin": 64},
    "lightgbm": {"max_bin": 63},
}


def test_declares_a_search_space_holding_its_defaults(spec):
    ranker = make_ranker(*spec)
    space = search_space(ranker)
    assert space
    defaults = type(ranker)().get_params()
    for name, dist in space.items():
        # None leaves the choice to the library, which no range can hold.
        assert defaults[name] is None or dist.contains(defaults[name]), name


def test_every_tunable_parameter_reaches_the_model(spec):
    ranker = make_ranker(*spec, random_state=0)
    values = {
        name: dist.choices[-1] if isinstance(dist, Categorical) else dist.low
        for name, dist in search_space(ranker).items()
    }
    ranker.set_params(**values).fit(X, Y, groups=GROUPS)
    passed = library_params(ranker)
    for name, value in values.items():
        # XGBoost stores its parameters as float32.
        assert passed[name] == (value if isinstance(value, str) else pytest.approx(value)), name


def test_extra_params_reach_the_model(spec):
    extra = EXTRA[spec[0]]
    ranker = make_ranker(*spec, random_state=0, extra_params=extra).fit(X, Y, groups=GROUPS)
    passed = library_params(ranker)
    assert {name: passed[name] for name in extra} == extra
    assert ranker.extra_params == extra


@pytest.mark.parametrize("name", ["learning_rate", "eta", "random_state", "verbose"])
def test_extra_params_cannot_override_the_rankers_own(spec, name):
    ranker = make_ranker(*spec, extra_params={name: 0.1})
    with pytest.raises(ValueError, match="sets itself"):
        ranker.fit(X, Y, groups=GROUPS)


def test_extra_params_must_be_a_mapping(spec):
    with pytest.raises(TypeError, match="extra_params must be a mapping"):
        make_ranker(*spec, extra_params=[("max_bin", 8)]).fit(X, Y, groups=GROUPS)


def test_autotune_tunes_the_ranker_of_a_cascade(spec):
    # The fixture's parameters keep the ranker small enough for the tiny data; hold them.
    frozen = [f"ranker__{name}" for name in spec[2]]
    tuned = AutoTune(_cascade(spec), freeze=frozen, n_trials=3, random_state=0)
    tuned.fit(trending_interactions())
    assert tuned.best_params_
    assert all(name.startswith("ranker__") for name in tuned.best_params_)
    assert not set(frozen) & set(tuned.best_params_)
