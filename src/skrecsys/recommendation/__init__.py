"""Classical collaborative-filtering recommenders."""

from skrecsys.recommendation._als import AlternatingLeastSquares
from skrecsys.recommendation._bm25 import BM25Recommender
from skrecsys.recommendation._bpr import BayesianPersonalizedRanking
from skrecsys.recommendation._ease import EASE
from skrecsys.recommendation._knn import ItemKNNRecommender
from skrecsys.recommendation._popular import MostPopularRecommender
from skrecsys.recommendation._rp3beta import RP3Beta
from skrecsys.recommendation._slim import SLIMElasticNet

__all__ = [
    "EASE",
    "AlternatingLeastSquares",
    "BM25Recommender",
    "BayesianPersonalizedRanking",
    "ItemKNNRecommender",
    "MostPopularRecommender",
    "RP3Beta",
    "SLIMElasticNet",
]
