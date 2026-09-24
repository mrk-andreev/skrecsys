"""Quality benchmarks: data => split => recommender => metrics => threshold.

Thresholds sit about 5% below the scores measured when they were set, so a failure
signals a real regression. Raise them when a model improves.
"""

import numpy as np
import pytest
from sklearn.base import clone

from skrecsys.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    make_recommender_scorer,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank_at_k,
)
from skrecsys.recommendation import (
    EASE,
    AlternatingLeastSquares,
    BayesianPersonalizedRanking,
    BM25Recommender,
    ItemKNNRecommender,
    MostPopularRecommender,
    RP3Beta,
    SLIMElasticNet,
)

try:
    from skrecsys import nn
except ImportError:  # the `nn` extra is optional; see src/skrecsys/nn/__init__.py
    nn = None

pytestmark = pytest.mark.benchmark

K = 10
METRICS = {
    "ndcg": ndcg_at_k,
    "precision": precision_at_k,
    "hit_rate": hit_rate_at_k,
    "map": average_precision_at_k,
    "mrr": reciprocal_rank_at_k,
}

# Minimum scores at K on the ``ua`` split, fitted on ratings.
BENCHMARKS = {
    "MostPopularRecommender": (
        MostPopularRecommender(),
        {"ndcg": 0.126, "precision": 0.115, "hit_rate": 0.69, "map": 0.051, "mrr": 0.30},
    ),
    "ItemKNNRecommender": (
        ItemKNNRecommender(),
        {"ndcg": 0.24, "precision": 0.209, "hit_rate": 0.87, "map": 0.12, "mrr": 0.53},
    ),
    "BM25Recommender": (
        BM25Recommender(),
        {"ndcg": 0.252, "precision": 0.217, "hit_rate": 0.847, "map": 0.130, "mrr": 0.55},
    ),
    "EASE": (
        EASE(),
        {"ndcg": 0.262, "precision": 0.226, "hit_rate": 0.858, "map": 0.137, "mrr": 0.559},
    ),
    "RP3Beta": (
        RP3Beta(),
        {"ndcg": 0.262, "precision": 0.228, "hit_rate": 0.875, "map": 0.134, "mrr": 0.561},
    ),
    "BayesianPersonalizedRanking": (
        BayesianPersonalizedRanking(random_state=0),
        {"ndcg": 0.266, "precision": 0.234, "hit_rate": 0.876, "map": 0.138, "mrr": 0.549},
    ),
    "SLIMElasticNet": (
        SLIMElasticNet(),
        {"ndcg": 0.287, "precision": 0.243, "hit_rate": 0.876, "map": 0.154, "mrr": 0.609},
    ),
    # A rating predictor, not a top-N ranker: its ranking scores sit below popularity.
    "AlternatingLeastSquares": (
        AlternatingLeastSquares(random_state=0),
        {"ndcg": 0.040, "precision": 0.038, "hit_rate": 0.30, "map": 0.0136, "mrr": 0.106},
    ),
}


def _neural_benchmarks():
    """The `nn` extra's models, or nothing when torch is not installed.

    Both override the paper's learning rate: it is tuned for datasets orders of magnitude
    larger than this one, and at its default a fit here barely leaves the initialization.
    """
    if nn is None:
        return {}
    return {
        "SimpleX": (
            nn.SimpleX(learning_rate=1e-2, random_state=0),
            {"ndcg": 0.234, "precision": 0.203, "hit_rate": 0.797, "map": 0.120, "mrr": 0.507},
        ),
        # The contrastive term is worth little on a catalog this small: 0.2931 NDCG at
        # this weight against 0.2952 with the term switched off entirely, while the
        # paper's 0.2 costs 0.08. It is kept non-zero so the row is the model it names.
        "XSimGCL": (
            nn.XSimGCL(learning_rate=1e-2, contrastive_weight=0.01, random_state=0),
            {"ndcg": 0.278, "precision": 0.243, "hit_rate": 0.886, "map": 0.146, "mrr": 0.575},
        ),
    }


BENCHMARKS |= _neural_benchmarks()

# Maximum test RMSE on the ``ua`` split for rating predictors.
RMSE_BENCHMARKS = {
    "AlternatingLeastSquares": (AlternatingLeastSquares(random_state=0), 0.98),
}


def _fit_and_score(estimator, dataset):
    train, test = dataset.train_indices, dataset.test_indices
    fitted = clone(estimator).fit(dataset.data[train], dataset.target[train])
    return {
        name: make_recommender_scorer(metric, k=K)(fitted, dataset.data[test])
        for name, metric in METRICS.items()
    }


@pytest.mark.parametrize("name", BENCHMARKS)
def test_movielens_100k_quality(name, movielens_100k_ua):
    estimator, thresholds = BENCHMARKS[name]
    scores = _fit_and_score(estimator, movielens_100k_ua)
    below = {
        metric: f"{scores[metric]:.4f} < {minimum}"
        for metric, minimum in thresholds.items()
        if scores[metric] < minimum
    }
    assert not below, f"{name} regressed at k={K}: {below}; all scores: {scores}"


def test_personalized_beats_popularity(movielens_100k_ua):
    popular = _fit_and_score(MostPopularRecommender(), movielens_100k_ua)
    knn = _fit_and_score(ItemKNNRecommender(), movielens_100k_ua)
    assert knn["ndcg"] > popular["ndcg"]


@pytest.mark.parametrize("name", RMSE_BENCHMARKS)
def test_movielens_100k_rmse(name, movielens_100k_ua):
    estimator, maximum = RMSE_BENCHMARKS[name]
    dataset = movielens_100k_ua
    train, test = dataset.train_indices, dataset.test_indices
    fitted = clone(estimator).fit(dataset.data[train], dataset.target[train])
    X_test, y_test = dataset.data[test], dataset.target[test]
    warm = np.isin(X_test[:, 0], fitted.user_ids_) & np.isin(X_test[:, 1], fitted.item_ids_)
    rmse = np.sqrt(np.mean((fitted.predict(X_test[warm]) - y_test[warm]) ** 2))
    assert rmse <= maximum, f"{name} regressed: RMSE {rmse:.4f} > {maximum}"
