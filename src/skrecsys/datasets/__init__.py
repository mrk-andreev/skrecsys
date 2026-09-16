"""Loaders for benchmark recommender datasets."""

from skrecsys.datasets._base import clear_data_home, get_data_home
from skrecsys.datasets._movielens import fetch_movielens_100k

__all__ = ["clear_data_home", "fetch_movielens_100k", "get_data_home"]
