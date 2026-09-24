"""Item-based k-nearest-neighbor collaborative filtering."""

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


class ItemKNNRecommender(IncrementalRecommenderMixin, SimilarityRecommender):
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
    index : None, str or VectorIndex, default=None
        Approximate index used by ``recommend``. ``None`` scores every candidate
        exactly; ``"hnsw"`` or a configured :class:`~skrecsys.indexing.HNSW` walks a
        graph instead. See :mod:`skrecsys.indexing`, and note that the exact path here
        is already an inverted-index scan that never touches an item nobody reached --
        so whether a graph beats it is a question for ``benchmarks/indexes.py``.

    Attributes
    ----------
    similarity_ : scipy.sparse.csr_array of shape (n_items_, n_items_)
        Pruned item-item similarities; row ``j`` holds the neighbors of item ``j``.
        The diagonal is zero.

    Notes
    -----
    ``partial_fit`` is exact: the neighbours are what ``fit`` on every batch concatenated
    would have produced. A batch can only change the rows of the items it reaches in two
    hops -- a user of a touched item, then everything that user also took -- and those
    rows are recomputed in full rather than patched, so an entry that was pruned away
    earlier can still come back. The saving is the rows it skips; a batch that touches a
    popular item reaches the whole catalog and costs a full recomputation.

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
        index: Any = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.shrink = shrink
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
        """Recompute the neighbour rows the batch can have changed, and only those."""
        del delta, new_user_indices, new_item_indices, touched_user_indices
        n_threads = self._check_params()
        rows = affected_item_rows(interactions, touched_item_indices)
        block = self._neighbors(interactions, rows, n_threads)
        self.similarity_ = replace_rows(self.similarity_, rows, block)

    def _neighbors(
        self, interactions: sp.csr_array, rows: NDArray[np.intp] | None, n_threads: int
    ) -> sp.csr_array:
        """The cosine neighbours of ``rows``, or of every item when it is None.

        The column norms are recomputed in a single pass over the stored interactions.
        Carrying them incrementally would need the values the batch replaced rather than
        the ones it added, and would save a pass that is already the cheapest thing here.
        """
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
            None if rows is None else kernel_indices(rows),
        )
        n_rows = n_items if rows is None else len(rows)
        return sp.csr_array((data, indices, indptr), shape=(n_rows, n_items), dtype=np.float64)

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
