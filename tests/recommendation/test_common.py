import pytest

from skrecsys.recommendation import (
    EASE,
    AlternatingLeastSquares,
    BM25Recommender,
    ItemKNNRecommender,
    MostPopularRecommender,
    RP3Beta,
)
from skrecsys.utils.estimator_checks import yield_recommender_checks

IMPLEMENTED = [
    MostPopularRecommender(),
    MostPopularRecommender(weighting="sum"),
    ItemKNNRecommender(),
    ItemKNNRecommender(n_neighbors=1, shrink=10.0),
    ItemKNNRecommender(n_neighbors=None),
    AlternatingLeastSquares(n_factors=2, n_iter=5, random_state=0),
    BM25Recommender(),
    BM25Recommender(n_neighbors=2, n_jobs=1),
    EASE(),
    EASE(l2_reg=1.0, n_jobs=1),
    RP3Beta(),
    RP3Beta(n_neighbors=2, alpha=0.7, beta=0.0, normalize_similarity=False, n_jobs=1),
]


@pytest.mark.parametrize("estimator", IMPLEMENTED, ids=repr)
@pytest.mark.parametrize("check", list(yield_recommender_checks()), ids=lambda c: c.__name__)
def test_common_checks(estimator, check):
    check(type(estimator).__name__, estimator)
