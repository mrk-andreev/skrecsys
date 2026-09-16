"""Item-based k-nearest-neighbor collaborative filtering."""

import numbers

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from skrecsys._typing import override
from skrecsys.recommendation._base import BaseRecommender, keep_top_k_per_row


class ItemKNNRecommender(BaseRecommender):
    """Item-based neighborhood recommender with cosine similarity.

    The score of item ``j`` for user ``u`` is ``sum_i r_ui * s(j, i)`` over the
    ``n_neighbors`` items ``i`` most similar to ``j`` [1]_.

    Parameters
    ----------
    n_neighbors : int or None, default=50
        Number of most similar items kept per item. ``None`` keeps all.

    shrink : float, default=0.0
        Shrinkage added to the cosine denominator, damping similarities supported by
        few co-occurrences.

    Attributes
    ----------
    similarity_ : scipy.sparse.csr_array of shape (n_items_, n_items_)
        Pruned item-item similarities; row ``j`` holds the neighbors of item ``j``.
        The diagonal is zero.

    References
    ----------
    .. [1] B. Sarwar et al., "Item-based collaborative filtering recommendation
       algorithms", WWW 2001. https://doi.org/10.1145/371920.372071

    Examples
    --------
    >>> from skrecsys.recommendation import ItemKNNRecommender
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "b"], ["u3", "a"]]
    >>> rec = ItemKNNRecommender(n_neighbors=10).fit(X)
    >>> rec.recommend(["u3"], n_recommendations=1)[0].tolist()
    [['b']]
    """

    def __init__(self, n_neighbors: int | None = 50, shrink: float = 0.0) -> None:
        self.n_neighbors = n_neighbors
        self.shrink = shrink

    @override
    def _fit(self, interactions: sp.csr_array) -> None:
        if self.n_neighbors is not None and (
            not isinstance(self.n_neighbors, numbers.Integral) or self.n_neighbors < 1
        ):
            raise ValueError(f"n_neighbors must be None or >= 1, got {self.n_neighbors!r}.")
        if self.shrink < 0:
            raise ValueError(f"shrink must be >= 0, got {self.shrink!r}.")

        R = interactions.tocsc()
        norms = np.sqrt(np.asarray(R.multiply(R).sum(axis=0), dtype=np.float64).ravel())
        cooc = (R.T @ R).tocoo()
        off_diagonal = (cooc.row != cooc.col) & (cooc.data != 0)
        rows, cols = cooc.row[off_diagonal], cooc.col[off_diagonal]
        data = cooc.data[off_diagonal] / (norms[rows] * norms[cols] + self.shrink)
        similarity = sp.csr_array((data, (rows, cols)), shape=cooc.shape)
        similarity.sort_indices()
        self.similarity_ = keep_top_k_per_row(similarity, self.n_neighbors)

    @override
    def _score_users(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        ratings = self.interactions_[user_indices]
        scores = ratings @ self.similarity_[item_indices].T
        return np.asarray(scores.toarray(), dtype=np.float64)
