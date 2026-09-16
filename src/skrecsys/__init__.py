"""Recommender systems in the scikit-learn style."""

from importlib.metadata import version

from skrecsys.base import RecommenderMixin, is_recommender

__version__ = version("skrecsys")

__all__ = ["RecommenderMixin", "__version__", "is_recommender"]
