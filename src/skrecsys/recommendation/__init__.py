"""Classical collaborative-filtering recommenders."""

from skrecsys.recommendation._als import AlternatingLeastSquares
from skrecsys.recommendation._bm25 import BM25Recommender
from skrecsys.recommendation._ease import EASE
from skrecsys.recommendation._knn import ItemKNNRecommender
from skrecsys.recommendation._popular import MostPopularRecommender
from skrecsys.recommendation._rp3beta import RP3Beta

__all__ = [
    "EASE",
    "AlternatingLeastSquares",
    "BM25Recommender",
    "ItemKNNRecommender",
    "MostPopularRecommender",
    "RP3Beta",
]
