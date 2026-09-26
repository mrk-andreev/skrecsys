import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from skrecsys.compose import (
    Cascade,
    ConcatFeatures,
    GeneratorScores,
    JoinDynamicFeatures,
    KnownUser,
    MinInteractions,
    PointwiseRanker,
    Switch,
)
from skrecsys.metrics import make_recommender_scorer, ndcg_at_k
from skrecsys.model_selection import WarmStartKFold
from skrecsys.recommendation import (
    AlternatingLeastSquares,
    ItemKNNRecommender,
    MostPopularRecommender,
)
from tests.compose._data import trending_interactions
from tests.estimator_checks import _interactions

ALS = AlternatingLeastSquares(n_factors=2, n_iter=5, random_state=0)


def test_rows_come_back_in_query_order_each_from_its_branch():
    X = _interactions()
    switch = Switch(KnownUser(), ALS, MostPopularRecommender()).fit(X)
    queries = np.array(["new-1", "u2", "new-2", "u0"], dtype=object)
    items, scores = switch.recommend(queries, n_recommendations=2)

    warm_items, warm_scores = switch.on_true_.recommend(queries[[1, 3]], n_recommendations=2)
    cold_items, cold_scores = switch.on_false_.recommend(queries[[0, 2]], n_recommendations=2)
    np.testing.assert_array_equal(items[[1, 3]], warm_items)
    np.testing.assert_array_equal(items[[0, 2]], cold_items)
    np.testing.assert_array_equal(scores[[1, 3]], warm_scores)
    np.testing.assert_array_equal(scores[[0, 2]], cold_scores)


def test_fit_clones_and_leaves_parameters_unfitted():
    switch = Switch(KnownUser(), ItemKNNRecommender(), MostPopularRecommender())
    switch.fit(_interactions())
    assert not hasattr(switch.on_true, "item_ids_")
    assert not hasattr(switch.condition, "user_ids_")


def test_predict_routes_pairs_by_user():
    X = _interactions()
    switch = Switch(MinInteractions(3), ItemKNNRecommender(), MostPopularRecommender()).fit(X)
    pairs = np.array([["u1", "i0"], ["u0", "i3"]], dtype=object)  # u1 has 3, u0 has 2
    expected = [
        ItemKNNRecommender().fit(X).predict(pairs[:1])[0],
        MostPopularRecommender().fit(X).predict(pairs[1:])[0],
    ]
    np.testing.assert_array_equal(switch.predict(pairs), expected)


def test_idle_branch_is_not_called():
    """All queries known: the cold branch must not even be asked (it would be fine, but
    a branch that cannot take an empty query list must not be handed one)."""
    X = _interactions()
    switch = Switch(KnownUser(), ItemKNNRecommender(), MostPopularRecommender()).fit(X)
    switch.on_false_ = None  # ty: ignore[invalid-assignment] - would raise if touched
    items, _ = switch.recommend(np.array(["u0"], dtype=object), n_recommendations=1)
    assert items.shape == (1, 1)


def test_an_unfit_route_raises_from_the_branch():
    """Unknown users routed to a branch that cannot serve them fail there, loudly."""
    switch = Switch(~KnownUser(), ItemKNNRecommender(), MostPopularRecommender())
    switch.fit(_interactions())
    with pytest.raises(ValueError, match="Unknown user"):
        switch.recommend(["new"], n_recommendations=1)


def test_unknown_candidates_raise():
    switch = Switch(KnownUser(), ItemKNNRecommender(), MostPopularRecommender())
    switch.fit(_interactions())
    with pytest.raises(ValueError, match="Unknown item"):
        switch.recommend(["u0"], n_recommendations=1, candidates=["nope"])


def test_count_eligible_routes():
    X = _interactions()
    switch = Switch(KnownUser(), ItemKNNRecommender(), MostPopularRecommender()).fit(X)
    counts = switch._count_eligible(np.array(["u0", "new"], dtype=object))
    assert counts.tolist() == [4, 6]  # u0 has seen 2 of 6 items; a new user none


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"condition": ALS}, "condition must be a condition"),
        ({"on_true": KnownUser()}, "on_true must be a recommender"),
        ({"on_false": "popular"}, "on_false must be a recommender"),
    ],
)
def test_fit_checks_roles(params, match):
    switch = Switch(KnownUser(), ItemKNNRecommender(), MostPopularRecommender())
    with pytest.raises(TypeError, match=match):
        switch.set_params(**params).fit(_interactions())


def test_grid_search_reaches_nested_parameters():
    X = trending_interactions()
    search = GridSearchCV(
        Switch(MinInteractions(1), ALS, MostPopularRecommender()),
        {"on_true__n_factors": [2, 4], "condition__n_interactions": [1, 100]},
        scoring=make_recommender_scorer(ndcg_at_k, k=5),
        cv=WarmStartKFold(n_splits=2),
    ).fit(X)
    assert set(search.best_params_) == {"on_true__n_factors", "condition__n_interactions"}
    assert np.isfinite(search.cv_results_["mean_test_score"]).all()


def _timed_cascade(calls):
    def callback(keys):
        calls.append(keys.copy())
        return np.nan_to_num(keys[:, -1].astype(float), nan=-1.0)

    return Cascade(
        MostPopularRecommender(),
        ConcatFeatures([JoinDynamicFeatures("user-item-time", callback), GeneratorScores()]),
        PointwiseRanker(LogisticRegression()),
        n_retrieved=10,
        time=True,
    )


def _timed_interactions():
    X = trending_interactions()
    return np.column_stack([X, np.arange(len(X))])


def test_timed_switch_gives_the_time_only_to_a_branch_that_uses_it():
    X = _timed_interactions()
    calls = []
    switch = Switch(
        MinInteractions(5), _timed_cascade(calls), MostPopularRecommender(), time=True
    ).fit(X)
    assert switch.time_dtype_ == np.int64

    queries = np.array([0, 1000, 1])
    items, _ = switch.recommend(queries, n_recommendations=2, as_of=[5, 6, 7])
    assert items.shape == (3, 2)
    by_user = {int(u): t for u, _, t in calls[-1]}
    assert by_user == {0: 5, 1: 7}

    switch.recommend(queries, n_recommendations=2)
    assert np.isnan(calls[-1][:, 2].astype(float)).all()
    scores = switch.predict([[0, 0, 3], [1, 0, 3]])
    assert scores.shape == (2,)
    assert switch._count_eligible(queries, exclude_interactions=[[0, 0, 1]]).shape == (3,)


def test_switch_without_time_refuses_as_of_and_timed_x():
    X = _timed_interactions()
    with pytest.raises(ValueError, match="time=True"):
        Switch(KnownUser(), MostPopularRecommender(), MostPopularRecommender()).fit(X)
    switch = Switch(KnownUser(), MostPopularRecommender(), MostPopularRecommender()).fit(X[:, :2])
    with pytest.raises(ValueError, match="as_of needs"):
        switch.recommend([0], n_recommendations=1, as_of=3)
