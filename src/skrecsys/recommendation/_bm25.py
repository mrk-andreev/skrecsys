"""Item-based nearest neighbours on BM25-weighted interactions, ported from implicit."""

import numbers

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from skrecsys import _core
from skrecsys._typing import override
from skrecsys.recommendation._base import BaseRecommender


class BM25Recommender(BaseRecommender):
    """Item-item nearest-neighbour recommender with BM25 weighting.

    Items are treated as documents and users as terms. Each interaction is weighted
    by BM25 [1]_,

    ``w_ui = r_ui * (k1 + 1) / (k1 * L_i + r_ui) * idf_u``,

    with ``idf_u = log(n_items) - log(1 + n_u)``, where ``n_u`` is the number of items
    of user ``u``, and ``L_i = 1 - b + b * |i| / avg|i|``, where ``|i|`` is the total
    interaction value of item ``i``. Item similarities are the rows of ``W^T W``, each
    pruned to its ``n_neighbors`` largest entries; the item itself is kept as a
    neighbour. The score of item ``j`` for user ``u`` is ``sum_i r_ui * s(i, j)`` over
    the items ``i`` of the user.

    This is a port of ``implicit.nearest_neighbours.BM25Recommender`` [2]_, and the
    similarity matrix matches it, including which neighbours are kept on tied scores.
    Recommendations differ in three details: ``exclude_seen`` removes seen items
    instead of scoring them 0, items with no similarity to the user's items score 0
    instead of being omitted, and ties rank by fitted item order.

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of most similar items kept per item, the item itself included.
    k1 : float, default=1.2
        BM25 saturation of interaction values.
    b : float, default=0.75
        BM25 length normalization, from 0 (none) to 1 (full).
    n_jobs : int or None, default=None
        Number of threads for computing similarities. ``None`` or ``-1`` uses all cores.

    Attributes
    ----------
    similarity_ : scipy.sparse.csr_array of shape (n_items_, n_items_)
        Pruned item-item similarities; row ``i`` holds the neighbours of item ``i``.

    References
    ----------
    .. [1] S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25
       and Beyond", Foundations and Trends in Information Retrieval 2009.
       https://doi.org/10.1561/1500000019
    .. [2] B. Frederickson, "implicit: Fast Python Collaborative Filtering for Implicit
       Datasets". https://github.com/benfred/implicit

    Examples
    --------
    >>> from skrecsys.recommendation import BM25Recommender
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "b"], ["u3", "a"]]
    >>> rec = BM25Recommender(n_neighbors=10).fit(X)
    >>> rec.recommend(["u3"], n_recommendations=1)[0].tolist()
    [['b']]
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        k1: float = 1.2,
        b: float = 0.75,
        n_jobs: int | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.k1 = k1
        self.b = b
        self.n_jobs = n_jobs

    @override
    def _fit(self, interactions: sp.csr_array) -> None:
        n_threads = self._check_params()
        n_users, n_items = interactions.shape
        coo = interactions.tocoo()

        idf = np.log(n_items) - np.log1p(np.bincount(coo.row, minlength=n_users))
        item_sums = np.bincount(coo.col, weights=coo.data, minlength=n_items)
        length_norm = (1.0 - self.b) + self.b * item_sums / item_sums.mean()
        weights = sp.csr_array(
            (
                coo.data
                * (self.k1 + 1.0)
                / (self.k1 * length_norm[coo.col] + coo.data)
                * idf[coo.row],
                (coo.row, coo.col),
            ),
            shape=interactions.shape,
        )
        weights.sort_indices()

        indptr, indices, data = _core.item_knn_top_k(
            weights.indptr.astype(np.int64),
            weights.indices.astype(np.int64),
            weights.data.astype(np.float64),
            n_items,
            int(self.n_neighbors),
            n_threads,
        )
        self.similarity_ = sp.csr_array((data, indices, indptr), shape=(n_items, n_items))

    @override
    def _score_users(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        scores = sp.csr_array(self.interactions_[user_indices] @ self.similarity_)
        return np.asarray(scores[:, item_indices].toarray(), dtype=np.float64)

    def _check_params(self) -> int:
        """Validate parameters and return the thread count for the kernel (0 = all)."""
        if not isinstance(self.n_neighbors, numbers.Integral) or self.n_neighbors < 1:
            raise ValueError(f"n_neighbors must be an integer >= 1, got {self.n_neighbors!r}.")
        if not isinstance(self.k1, numbers.Real) or not self.k1 >= 0:
            raise ValueError(f"k1 must be a real number >= 0, got {self.k1!r}.")
        if not isinstance(self.b, numbers.Real) or not 0 <= self.b <= 1:
            raise ValueError(f"b must be a real number in [0, 1], got {self.b!r}.")
        if self.n_jobs is None or self.n_jobs == -1:
            return 0
        if not isinstance(self.n_jobs, numbers.Integral) or self.n_jobs < 1:
            raise ValueError(f"n_jobs must be None, -1 or an integer >= 1, got {self.n_jobs!r}.")
        return int(self.n_jobs)
