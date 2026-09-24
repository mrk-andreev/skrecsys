"""Item-based nearest neighbours on BM25-weighted interactions, ported from implicit."""

import numbers
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from skrecsys import _core
from skrecsys._typing import override
from skrecsys.indexing import SparseSpace
from skrecsys.recommendation._base import SimilarityRecommender, kernel_csr, kernel_indices
from skrecsys.recommendation._incremental import (
    IncrementalRecommenderMixin,
    affected_item_rows,
    replace_rows,
)


class BM25Recommender(IncrementalRecommenderMixin, SimilarityRecommender):
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
    index : None, str or VectorIndex, default=None
        Approximate index used by ``recommend``. ``None`` scores every candidate
        exactly; ``"hnsw"`` or a configured :class:`~skrecsys.indexing.HNSW` walks a
        graph instead. See :mod:`skrecsys.indexing`, and note that the exact path here
        is already an inverted-index scan that never touches an item nobody reached --
        so whether a graph beats it is a question for ``benchmarks/indexes.py``.

    Attributes
    ----------
    similarity_ : scipy.sparse.csr_array of shape (n_items_, n_items_)
        Pruned item-item similarities; row ``i`` holds the neighbours of item ``i``.

    Notes
    -----
    ``partial_fit`` is exact -- the similarities are what ``fit`` on every batch
    concatenated would have produced -- but it is usually not cheap, and the reason is in
    the weighting rather than in the implementation. ``idf_u`` carries ``log(n_items)``
    and ``L_i`` divides by the mean item length, so a batch that adds an item, or that
    moves that mean at all, changes *every* stored weight and therefore every
    similarity. The rows actually recomputed are narrowed to what the batch can reach
    only when neither happens, which means ``b=0`` and no new items. Otherwise the whole
    catalog is recomputed, and no amount of bookkeeping can avoid it while the answer
    stays exact.

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
        index: Any = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.k1 = k1
        self.b = b
        self.n_jobs = n_jobs
        self.index = index

    _incremental_state_ = (("similarity_", "item_item"),)

    @override
    def _fit(self, interactions: sp.csr_array) -> None:
        n_threads = self._check_params()
        self.similarity_ = self._neighbors(interactions, None, n_threads)

    @override
    def _partial_fit(
        self,
        interactions: sp.csr_array,
        *,
        delta: sp.csr_array,
        new_user_indices: NDArray[np.intp],
        new_item_indices: NDArray[np.intp],
        touched_user_indices: NDArray[np.intp],
        touched_item_indices: NDArray[np.intp],
    ) -> None:
        """Recompute what the batch can have changed, which is usually everything.

        Two things widen the set well past the batch. A user the batch names has a new
        ``idf``, so *all* of that user's weights move, not only the ones in the batch;
        and the corpus statistics are catalog-wide, so a new item or a shifted mean item
        length moves every weight there is. What is left after that is the ordinary
        two-hop rule, and it only applies when ``b=0`` and the catalog did not grow.
        """
        del delta, new_user_indices, touched_item_indices
        n_threads = self._check_params()
        rescaled = len(new_item_indices) > 0 or float(self.b) != 0.0
        if rescaled:
            self.similarity_ = self._neighbors(interactions, None, n_threads)
            return
        moved = np.unique(sp.csr_array(interactions[touched_user_indices]).indices)
        rows = affected_item_rows(interactions, moved.astype(np.intp))
        block = self._neighbors(interactions, rows, n_threads)
        self.similarity_ = replace_rows(self.similarity_, rows, block)

    def _neighbors(
        self, interactions: sp.csr_array, rows: NDArray[np.intp] | None, n_threads: int
    ) -> sp.csr_array:
        """The BM25 neighbours of ``rows``, or of every item when it is None."""
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
            *kernel_csr(weights),
            n_items,
            int(self.n_neighbors),
            n_threads,
            None if rows is None else kernel_indices(rows),
        )
        n_rows = n_items if rows is None else len(rows)
        return sp.csr_array((data, indices, indptr), shape=(n_rows, n_items))

    @override
    def _index_space(self) -> SparseSpace:
        # Row `j` of `_neighbors_of_items` holds exactly the weights that make up item
        # `j`'s score, which is the item vector, and it is already built and cached.
        return SparseSpace(sp.csr_array(self._neighbors_of_items()))

    @override
    def _index_queries(self, user_indices: NDArray[np.intp]) -> sp.csr_array:
        return sp.csr_array(self.interactions_[user_indices])

    @override
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
