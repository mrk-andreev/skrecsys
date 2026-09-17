"""Top-k ranking metrics and scorers for recommenders."""

from skrecsys.metrics._beyond_accuracy import (
    catalog_coverage_at_k,
    item_popularity,
    mean_popularity_at_k,
    novelty_at_k,
    user_coverage_at_k,
)
from skrecsys.metrics._ranking import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from skrecsys.metrics._scorer import make_recommender_scorer

__all__ = [
    "average_precision_at_k",
    "catalog_coverage_at_k",
    "hit_rate_at_k",
    "item_popularity",
    "make_recommender_scorer",
    "mean_popularity_at_k",
    "ndcg_at_k",
    "novelty_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
    "user_coverage_at_k",
]
