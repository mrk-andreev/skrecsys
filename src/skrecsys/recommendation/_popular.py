"""Popularity baseline."""

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from skrecsys._typing import override
from skrecsys.recommendation._base import BaseRecommender


class MostPopularRecommender(BaseRecommender):
    """Recommend the most popular items to every user.

    A deterministic, non-personalized baseline for implicit and explicit feedback.

    Parameters
    ----------
    weighting : {"count", "sum"}, default="count"
        ``"count"`` scores an item by the number of distinct users who interacted with
        it. ``"sum"`` scores it by the sum of interaction values.

    Attributes
    ----------
    item_popularity_ : ndarray of shape (n_items_,)
        Popularity score of each fitted item.

    Examples
    --------
    >>> from skrecsys.recommendation import MostPopularRecommender
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "b"], ["u3", "c"]]
    >>> rec = MostPopularRecommender().fit(X)
    >>> items, scores = rec.recommend(["u3"], n_recommendations=2)
    >>> items.tolist()
    [['b', 'a']]
    """

    def __init__(self, weighting: str = "count") -> None:
        self.weighting = weighting

    @override
    def _fit(self, interactions: sp.csr_array) -> None:
        if self.weighting == "count":
            popularity = np.diff(interactions.tocsc().indptr)
        elif self.weighting == "sum":
            popularity = interactions.sum(axis=0)
        else:
            raise ValueError(f"weighting must be 'count' or 'sum', got {self.weighting!r}.")
        self.item_popularity_ = np.asarray(popularity, dtype=np.float64).ravel()

    @override
    def _score_users(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        return np.tile(self.item_popularity_[item_indices], (len(user_indices), 1))
