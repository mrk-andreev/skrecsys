"""Item-based k-nearest-neighbor collaborative filtering."""

import numbers

import numpy as np
import scipy.sparse as sp

from skrecsys import _core
from skrecsys._typing import override
from skrecsys.recommendation._base import SimilarityRecommender, kernel_csr


class ItemKNNRecommender(SimilarityRecommender):
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

    n_jobs : int or None, default=None
        Threads used by the native kernel. ``None`` and ``-1`` use every core.

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

    _neighbors_by_row = True

    def __init__(
        self,
        n_neighbors: int | None = 50,
        shrink: float = 0.0,
        n_jobs: int | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.shrink = shrink
        self.n_jobs = n_jobs

    @override
    def _fit(self, interactions: sp.csr_array) -> None:
        n_threads = self._check_params()
        n_items = interactions.shape[1]
        norms = np.sqrt(
            np.asarray(interactions.multiply(interactions).sum(axis=0), dtype=np.float64).ravel()
        )
        # The kernel prunes as it accumulates, so `None` asks it for every column rather
        # than for a second pass over a fully materialized similarity matrix.
        k = n_items if self.n_neighbors is None else int(self.n_neighbors)
        indptr, indices, data = _core.item_cosine_top_k(
            *kernel_csr(interactions),
            n_items,
            norms,
            float(self.shrink),
            k,
            n_threads,
        )
        self.similarity_ = sp.csr_array(
            (data, indices, indptr), shape=(n_items, n_items), dtype=np.float64
        )

    @override
    def _check_params(self) -> int:
        """Validate parameters and return the thread count for the kernel (0 = all)."""
        if self.n_neighbors is not None and (
            not isinstance(self.n_neighbors, numbers.Integral) or self.n_neighbors < 1
        ):
            raise ValueError(f"n_neighbors must be None or >= 1, got {self.n_neighbors!r}.")
        if not isinstance(self.shrink, numbers.Real) or self.shrink < 0:
            raise ValueError(f"shrink must be >= 0, got {self.shrink!r}.")
        if self.n_jobs is None or self.n_jobs == -1:
            return 0
        if not isinstance(self.n_jobs, numbers.Integral) or self.n_jobs < 1:
            raise ValueError(f"n_jobs must be None, -1 or an integer >= 1, got {self.n_jobs!r}.")
        return int(self.n_jobs)
