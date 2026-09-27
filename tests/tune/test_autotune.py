import pickle

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator

from skrecsys._typing import override
from skrecsys.base import RankerMixin, uses_time
from skrecsys.compose import Cascade, GeneratorScores
from skrecsys.metrics import Recall, make_recommender_scorer, ndcg_at_k, recall_at_k
from skrecsys.recommendation import BM25Recommender, ItemKNNRecommender, MostPopularRecommender
from skrecsys.tune import AutoTune, Float, Int
from tests.estimator_checks import check_numeric_ids, yield_recommender_checks


class _WarmHoldOut(BaseCrossValidator):
    """One fold holding out every interaction whose user and item occur earlier.

    The shared fixture is too small for three warm-start folds.
    """

    @override
    def _iter_test_masks(self, X=None, y=None, groups=None):
        X = np.asarray(X, dtype=object)
        test = np.zeros(len(X), dtype=bool)
        for column in (0, 1):
            _, first = np.unique(X[:, column].astype(str), return_index=True)
            test[first] = True
        yield ~test

    @override
    def get_n_splits(self, X=None, y=None, groups=None):
        return 1


def _interactions(n_users=40, n_items=25, per_user=6, seed=0):
    """Users in two taste groups, each drawing mostly from its half of the catalog."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        half = (u % 2) * (n_items // 2)
        items = rng.choice(n_items // 2, size=per_user, replace=False) + half
        rows += [[f"u{u}", f"i{i}"] for i in items]
    return np.array(rows, dtype=object)


#: The shared fixture's six items cannot fill the default scorer's ten slots.
TUNED = AutoTune(
    ItemKNNRecommender(),
    scoring=make_recommender_scorer(ndcg_at_k, k=1),
    cv=_WarmHoldOut(),
    n_trials=3,
    random_state=0,
)


@pytest.mark.parametrize("check", list(yield_recommender_checks()), ids=lambda c: c.__name__)
def test_common_checks(check):
    if check is check_numeric_ids:
        pytest.skip("five interactions leave nothing to hold out for tuning")
    check("AutoTune", TUNED)


def test_fit_tunes_and_refits_the_best():
    X = _interactions()
    tuned = AutoTune(BM25Recommender(), n_trials=12, random_state=0).fit(X)
    assert len(tuned.study_.trials) == 12
    assert set(tuned.best_params_) == {"n_neighbors", "k1", "b"}
    assert all(tuned.best_score_ >= t.value for t in tuned.study_.trials if t.value is not None)
    assert tuned.best_estimator_.get_params() | tuned.best_params_ == (
        tuned.best_estimator_.get_params()
    )
    items, _ = tuned.recommend(["u0", "u1"], n_recommendations=3)
    expected, _ = tuned.best_estimator_.recommend(["u0", "u1"], n_recommendations=3)
    np.testing.assert_array_equal(items, expected)
    assert tuned.n_users_ == 40


def test_the_first_trial_is_the_configured_estimator():
    tuned = AutoTune(BM25Recommender(k1=0.5), n_trials=4, random_state=0).fit(_interactions())
    assert tuned.study_.trials[0].params == {"n_neighbors": 20, "k1": 0.5, "b": 0.75}
    baseline = tuned.study_.trials[0].value
    assert baseline is not None
    assert tuned.best_score_ >= baseline


def test_the_estimator_itself_is_left_unfitted():
    estimator = BM25Recommender()
    AutoTune(estimator, n_trials=2, random_state=0).fit(_interactions())
    assert not hasattr(estimator, "similarity_")


def test_the_same_random_state_repeats_the_search():
    def run():
        return AutoTune(BM25Recommender(), n_trials=6, random_state=3).fit(_interactions())

    assert run().best_params_ == run().best_params_


def test_search_space_overrides_and_restricts_nothing_else():
    space = {"n_neighbors": Int(2, 4), "shrink": Float(0.0, 1.0)}
    tuned = AutoTune(ItemKNNRecommender(), search_space=space, n_trials=5, random_state=0).fit(
        _interactions()
    )
    for trial in tuned.study_.trials[1:]:
        assert space["n_neighbors"].contains(trial.params["n_neighbors"])
        assert space["shrink"].contains(trial.params["shrink"])


def test_frozen_parameters_keep_the_instance_value():
    tuned = AutoTune(
        BM25Recommender(k1=0.5, n_neighbors=7),
        freeze=["k1", "n_neighbors"],
        n_trials=6,
        random_state=0,
    ).fit(_interactions())
    assert set(tuned.best_params_) == {"b"}
    params = tuned.best_estimator_.get_params()
    assert (params["k1"], params["n_neighbors"]) == (0.5, 7)
    assert all(set(t.params) == {"b"} for t in tuned.study_.trials)


def test_a_parameter_added_by_search_space_can_be_frozen_too():
    tuned = AutoTune(
        ItemKNNRecommender(),
        search_space={"n_neighbors": Int(2, 4)},
        freeze=["n_neighbors"],
        n_trials=3,
        random_state=0,
    ).fit(_interactions())
    assert set(tuned.best_params_) == {"shrink"}


def _tune_with(scoring):
    return AutoTune(
        BM25Recommender(),
        scoring=scoring,
        cv=2,
        n_trials=3,
        sampler="random",
        random_state=0,
    ).fit(_interactions())


def test_custom_scoring_and_integer_cv():
    tuned = _tune_with(make_recommender_scorer(recall_at_k, k=5))
    assert 0.0 <= tuned.best_score_ <= 1.0


@pytest.mark.parametrize("scoring", ["recall@5", Recall(5)], ids=str)
def test_a_named_metric_scores_as_its_scorer(scoring):
    expected = _tune_with(make_recommender_scorer(recall_at_k, k=5))
    tuned = _tune_with(scoring)
    assert tuned.best_score_ == expected.best_score_
    assert tuned.best_params_ == expected.best_params_


def test_the_default_scoring_is_ndcg_at_10():
    expected = _tune_with(make_recommender_scorer(ndcg_at_k, k=10))
    assert _tune_with(None).best_score_ == expected.best_score_


def test_an_unannotated_estimator_needs_a_search_space():
    with pytest.raises(ValueError, match="declares no tunable parameters"):
        AutoTune(MostPopularRecommender(), n_trials=2).fit(_interactions())


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"estimator": "bm25"}, TypeError, "must be a recommender"),
        ({"n_trials": 0}, ValueError, "n_trials"),
        ({"search_space": {"nope": Int(1, 2)}}, ValueError, "not parameters"),
        ({"search_space": {"k1": (0, 1)}}, TypeError, "Float, Int or Categorical"),
        ({"search_space": [("k1", Float(0, 1))]}, TypeError, "mapping"),
        ({"freeze": "k1"}, TypeError, "sequence of parameter names"),
        ({"freeze": ["n_jobs"]}, ValueError, "not tunable"),
        ({"freeze": ["k1", "b", "n_neighbors"]}, ValueError, "nothing is left"),
        ({"scoring": "auc@10"}, ValueError, "Unknown metric"),
        ({"scoring": 10}, TypeError, "scoring must be"),
    ],
)
def test_invalid_parameters_raise_at_fit(params, error, match):
    tuned = AutoTune(BM25Recommender(), n_trials=2).set_params(**params)
    with pytest.raises(error, match=match):
        tuned.fit(_interactions())


def test_a_fitted_autotune_pickles():
    tuned = AutoTune(BM25Recommender(), n_trials=3, random_state=0).fit(_interactions())
    loaded = pickle.loads(pickle.dumps(tuned))
    assert loaded.best_params_ == tuned.best_params_
    assert [t.value for t in loaded.study_.trials] == [t.value for t in tuned.study_.trials]
    assert clone(tuned).get_params()["n_trials"] == 3


class _ByScore(RankerMixin, BaseEstimator):
    """Ranks candidates by their first feature, the generator score."""

    def fit(self, X, y, *, groups):
        return self

    def predict(self, X, *, groups):
        return X[:, 0]


def test_a_timed_estimator_is_tuned_on_timed_rows_and_recommends_as_of():
    X = _interactions()
    timed = np.empty((len(X), 3), dtype=object)
    timed[:, :2] = X
    timed[:, 2] = np.arange(len(X))
    cascade = Cascade(BM25Recommender(), GeneratorScores(), _ByScore(), n_retrieved=10, time=True)
    tuned = AutoTune(cascade, n_trials=2, random_state=0).fit(timed)
    assert uses_time(tuned)
    assert not uses_time(AutoTune(BM25Recommender()))
    best = tuned.best_estimator_
    assert uses_time(best)
    user = tuned.user_ids_[:1]
    np.testing.assert_array_equal(
        tuned.recommend(user, n_recommendations=2, as_of=5)[0],
        best.recommend(user, n_recommendations=2, as_of=5)[0],
    )


def test_as_of_needs_a_timed_estimator():
    tuned = AutoTune(BM25Recommender(), n_trials=1, random_state=0).fit(_interactions())
    with pytest.raises(ValueError, match="as_of needs"):
        tuned.recommend(tuned.user_ids_[:1], n_recommendations=1, as_of=5)
