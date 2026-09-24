import functools
from unittest import mock

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_validate

from skrecsys.metrics import (
    average_precision_at_k,
    catalog_coverage_at_k,
    evaluate_recommender,
    hit_rate_at_k,
    item_popularity,
    make_recommender_scorer,
    mean_popularity_at_k,
    ndcg_at_k,
    novelty_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    user_coverage_at_k,
)
from skrecsys.model_selection import WarmStartKFold
from skrecsys.recommendation import ItemKNNRecommender, MostPopularRecommender

TRAIN = np.array(
    [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "c"], ["u3", "b"], ["u3", "c"], ["u4", "a"]],
    dtype=object,
)


def test_scorer_groups_held_out_items_by_user():
    rec = MostPopularRecommender().fit(TRAIN)
    # Popularity: a=3, b=2, c=2 -> u1 gets [c], u4 gets [b].
    test = np.array([["u1", "c"], ["u4", "c"], ["u4", "b"]], dtype=object)
    scorer = make_recommender_scorer(precision_at_k, k=1)
    assert scorer(rec, test) == pytest.approx((1 + 1) / 2)


def test_scorer_ignores_non_positive_relevance():
    rec = MostPopularRecommender().fit(TRAIN)
    test = np.array([["u1", "c"], ["u4", "b"]], dtype=object)
    scorer = make_recommender_scorer(hit_rate_at_k, k=1)
    assert scorer(rec, test, [1.0, 0.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="No held-out"):
        scorer(rec, test, [0.0, 0.0])


def test_scorer_requires_recommender():
    scorer = make_recommender_scorer(ndcg_at_k, k=1)
    with pytest.raises(TypeError, match="not a recommender"):
        scorer(LinearRegression(), TRAIN)  # ty: ignore[invalid-argument-type]
    assert "ndcg_at_k" in repr(scorer)


def _dataset(seed=0, n_users=30, n_items=15, n_interactions=200):
    rng = np.random.default_rng(seed)
    pairs = {
        (f"u{rng.integers(n_users)}", f"i{rng.integers(n_items)}") for _ in range(n_interactions)
    }
    return np.array(sorted(pairs), dtype=object)


def test_cross_validate_and_grid_search():
    X = _dataset()
    cv = WarmStartKFold(n_splits=3, shuffle=True, random_state=0)
    scorer = make_recommender_scorer(ndcg_at_k, k=3)
    result = cross_validate(ItemKNNRecommender(), X, cv=cv, scoring=scorer, error_score="raise")
    assert result["test_score"].shape == (3,)
    assert np.all((result["test_score"] >= 0) & (result["test_score"] <= 1))

    search = GridSearchCV(
        ItemKNNRecommender(), {"n_neighbors": [2, 10]}, cv=cv, scoring=scorer, error_score="raise"
    ).fit(X)
    assert search.best_params_["n_neighbors"] in {2, 10}


def test_scorer_drives_beyond_accuracy_metrics():
    rec = MostPopularRecommender().fit(TRAIN)
    test = np.array([["u1", "c"], ["u4", "b"]], dtype=object)

    # catalog_coverage_at_k has no per-query form, so the scorer must not pass average.
    coverage = make_recommender_scorer(catalog_coverage_at_k, k=1, catalog=rec.item_ids_)
    assert coverage(rec, test) == pytest.approx(2 / 3)
    assert make_recommender_scorer(user_coverage_at_k, k=1)(rec, test) == pytest.approx(1.0)

    popularity = item_popularity(TRAIN)
    scorer = make_recommender_scorer(mean_popularity_at_k, k=1, item_popularity=popularity)
    # u1 has seen a and b, so it is offered c (popularity 2); u4 is offered b (2).
    assert scorer(rec, test) == pytest.approx(2.0)
    novelty = make_recommender_scorer(novelty_at_k, k=1, item_popularity=popularity)
    assert novelty(rec, test) > 0.0


RANKING_METRICS = [
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    average_precision_at_k,
    reciprocal_rank_at_k,
]


def _fitted_split():
    X = _dataset()
    train, test = next(WarmStartKFold(n_splits=3, shuffle=True, random_state=0).split(X))
    return ItemKNNRecommender().fit(X[train]), X[test]


def test_evaluate_matches_one_scorer_per_metric_and_cutoff():
    rec, test = _fitted_split()
    coverage = functools.partial(catalog_coverage_at_k, catalog=rec.item_ids_)
    metrics = {**{f.__name__: f for f in RANKING_METRICS}, "coverage": coverage}
    scores = evaluate_recommender(rec, test, metrics=metrics, k=[5, 1, 3])
    assert list(scores) == [f"{name}@{k}" for name in metrics for k in (1, 3, 5)]
    for name, metric in metrics.items():
        for k in (1, 3, 5):
            expected = make_recommender_scorer(metric, k=k)(rec, test)
            assert scores[f"{name}@{k}"] == pytest.approx(expected)


def test_evaluate_ranks_once_for_the_largest_cutoff():
    rec, test = _fitted_split()
    with mock.patch.object(rec, "recommend", wraps=rec.recommend) as recommend:
        evaluate_recommender(rec, test, metrics=RANKING_METRICS, k=[1, 3, 5])
    recommend.assert_called_once()
    assert recommend.call_args.kwargs["n_recommendations"] == 5


def test_evaluate_names_metrics():
    rec, test = _fitted_split()
    assert list(evaluate_recommender(rec, test, metrics=ndcg_at_k, k=2)) == ["ndcg@2"]
    scores = evaluate_recommender(rec, test, metrics=[ndcg_at_k, hit_rate_at_k], k=[1, 2])
    assert list(scores) == ["ndcg@1", "ndcg@2", "hit_rate@1", "hit_rate@2"]
    popularity = item_popularity(TRAIN)
    novelty = functools.partial(novelty_at_k, item_popularity=popularity)
    assert list(evaluate_recommender(rec, test, metrics=[novelty], k=1)) == ["novelty@1"]
    with pytest.raises(ValueError, match="Two metrics are named 'ndcg'"):
        evaluate_recommender(rec, test, metrics=[ndcg_at_k, ndcg_at_k])


@pytest.mark.parametrize("k", [0, [], [1, 0], 1.5, True, [2, "3"]])
def test_evaluate_rejects_bad_cutoffs(k):
    rec, test = _fitted_split()
    with pytest.raises(ValueError, match="k must"):
        evaluate_recommender(rec, test, metrics=ndcg_at_k, k=k)


def test_evaluate_rejects_empty_metrics_and_non_recommenders():
    rec, test = _fitted_split()
    with pytest.raises(ValueError, match="at least one metric"):
        evaluate_recommender(rec, test, metrics=[])
    with pytest.raises(TypeError, match="not a recommender"):
        evaluate_recommender(LinearRegression(), test, metrics=ndcg_at_k)  # ty: ignore[invalid-argument-type]


def test_scorer_returns_a_float_for_one_metric_at_one_cutoff():
    rec, test = _fitted_split()
    assert isinstance(make_recommender_scorer(ndcg_at_k, k=3)(rec, test), float)
    assert isinstance(make_recommender_scorer(ndcg_at_k, k=[3])(rec, test), dict)
    assert isinstance(make_recommender_scorer([ndcg_at_k], k=3)(rec, test), dict)


def test_scorer_keyword_arguments_bind_a_single_metric():
    popularity = item_popularity(TRAIN)
    rec = MostPopularRecommender().fit(TRAIN)
    test = np.array([["u1", "c"], ["u4", "b"]], dtype=object)
    scorer = make_recommender_scorer(mean_popularity_at_k, k=[1], item_popularity=popularity)
    assert scorer(rec, test) == {"mean_popularity@1": pytest.approx(2.0)}
    with pytest.raises(TypeError, match=r"functools\.partial"):
        make_recommender_scorer([novelty_at_k], item_popularity=popularity)


def test_multi_metric_scorer_in_cross_validate_and_grid_search():
    X = _dataset()
    cv = WarmStartKFold(n_splits=3, shuffle=True, random_state=0)
    scorer = make_recommender_scorer([ndcg_at_k, recall_at_k], k=[1, 3])
    assert repr(scorer) == "make_recommender_scorer([ndcg, recall], k=[1, 3], exclude_seen=True)"
    result = cross_validate(ItemKNNRecommender(), X, cv=cv, scoring=scorer, error_score="raise")
    for key in ("ndcg@1", "ndcg@3", "recall@1", "recall@3"):
        assert result[f"test_{key}"].shape == (3,)

    search = GridSearchCV(
        ItemKNNRecommender(),
        {"n_neighbors": [2, 10]},
        cv=cv,
        scoring=scorer,
        refit="ndcg@3",
        error_score="raise",
    ).fit(X)
    assert "mean_test_recall@1" in search.cv_results_
    assert search.best_params_["n_neighbors"] in {2, 10}
