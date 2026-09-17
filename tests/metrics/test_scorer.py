import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_validate

from skrecsys.metrics import (
    catalog_coverage_at_k,
    hit_rate_at_k,
    item_popularity,
    make_recommender_scorer,
    mean_popularity_at_k,
    ndcg_at_k,
    novelty_at_k,
    precision_at_k,
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
        scorer(LinearRegression(), TRAIN)
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
