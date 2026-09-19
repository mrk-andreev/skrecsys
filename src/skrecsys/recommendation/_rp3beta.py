"""RP3beta: random-walk item-item similarity, ported from Ferrari Dacrema's framework."""

import math
import numbers
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from skrecsys import _core
from skrecsys._typing import override
from skrecsys.indexing import SparseSpace
from skrecsys.recommendation._base import (
    SimilarityRecommender,
    keep_top_k_per_row,
    kernel_data,
    kernel_indices,
)
from skrecsys.recommendation._incremental import (
    IncrementalRecommenderMixin,
    affected_item_rows,
    remap_sparse,
    replace_rows,
)


class RP3Beta(IncrementalRecommenderMixin, SimilarityRecommender):
    """Item-item recommender from a popularity-damped random walk on the user-item graph.

    A three-step walk ``item -> user -> item`` gives the transition probabilities
    ``W = Piu Pui``, where ``Pui`` is the interaction matrix with rows normalized to
    sum to one and ``Piu`` the same for the binary transpose, both raised to the power
    ``alpha``. RP3beta [1]_ then damps popular destinations by dividing column ``j`` of
    ``W`` by ``popularity[j] ** beta``, which is what separates it from P3alpha; ``beta
    = 0`` recovers P3alpha [2]_. Each row is pruned to its ``n_neighbors`` largest
    entries, the diagonal is dropped, and the score of item ``j`` for user ``u`` is
    ``sum_i r_ui * s(i, j)`` over the items ``i`` of the user.

    This is a port of ``RP3betaRecommender`` from the evaluation framework of Ferrari
    Dacrema et al. [3]_, including its second, column-wise pruning pass after the rows
    are normalized. Two parameters of the reference are not exposed: ``min_rating``,
    which is left to the caller, and ``implicit``, which the reference only applies
    when ``min_rating > 0`` and which ``fit(X)`` without ``y`` already achieves. The
    reference computes in float32 and breaks ties by an unstable sort, so its weights
    agree with ours only to float32 precision and may differ on exactly tied entries,
    where we keep the lower item index.

    Parameters
    ----------
    n_neighbors : int, default=100
        Number of largest entries kept per row, the reference's ``topK``.
    alpha : float, default=1.0
        Power applied to both transition matrices. Values below 1 flatten the walk,
        spreading probability towards less obvious items.
    beta : float, default=0.6
        Popularity damping of the destination item. 0 disables it, larger values push
        recommendations further into the long tail.
    normalize_similarity : bool, default=True
        Normalize each row of the similarity matrix to sum to one before the
        column-wise pruning pass.
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
    walk_ : scipy.sparse.csr_array of shape (n_items_, n_items_)
        The walk matrix after the row-wise pruning and normalization but before the
        column-wise pruning pass. Present only once ``partial_fit`` has been used.
    similarity_ : scipy.sparse.csr_array of shape (n_items_, n_items_)
        Pruned item-item transition weights; row ``i`` holds the neighbours of item
        ``i``. The diagonal is zero.

    Notes
    -----
    ``partial_fit`` is exact: the similarities are what ``fit`` on every batch
    concatenated would have produced. It keeps ``walk_``, the matrix before the
    column-wise pruning pass, because that pass is global -- a change in one row can
    evict an entry from another -- and it cannot be redone from the pruned result. That
    doubles the model's memory once incremental fitting begins. The rows recomputed are
    those the batch reaches in two hops, widened by the fact that a new interaction
    rescales its user's whole row of ``Pui``.

    References
    ----------
    .. [1] F. Christoffel et al., "Blockbusters and Wallflowers: Accurate, Diverse and
       Scalable Recommendations with Random Walks", RecSys 2015.
       https://doi.org/10.1145/2792838.2800180
    .. [2] C. Cooper et al., "Random Walks in Recommender Systems: Exact Computation and
       Simulations", WWW 2014. https://doi.org/10.1145/2567948.2579244
    .. [3] M. Ferrari Dacrema et al., "Are We Really Making Much Progress? A Worrying
       Analysis of Recent Neural Recommendation Approaches", RecSys 2019.
       https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation
    .. [4] V. W. Anelli et al., "Challenging the Myth of Graph Collaborative Filtering:
       a Reasoned and Reproducibility-driven Analysis", RecSys 2023, which tunes RP3beta
       as a classic baseline against graph neural recommenders.
       https://github.com/sisinflab/Graph-RSs-Reproducibility

    Examples
    --------
    >>> from skrecsys.recommendation import RP3Beta
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "b"], ["u3", "a"]]
    >>> rec = RP3Beta().fit(X)
    >>> rec.recommend(["u3"], n_recommendations=1)[0].tolist()
    [['b']]
    """

    def __init__(
        self,
        n_neighbors: int = 100,
        alpha: float = 1.0,
        beta: float = 0.6,
        *,
        normalize_similarity: bool = True,
        n_jobs: int | None = None,
        index: Any = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.beta = beta
        self.normalize_similarity = normalize_similarity
        self.n_jobs = n_jobs
        self.index = index

    _incremental_state_ = (("similarity_", "item_item"),)

    @override
    def _fit(self, interactions: sp.csr_array) -> None:
        n_threads = self._check_params()
        walk = self._walk(interactions, None, n_threads)
        self.similarity_ = self._prune_columns(walk, n_threads)
        # A plain `fit` keeps the memory it always did; the walk matrix is state only an
        # incremental fit splices into, and the first `partial_fit` rebuilds it.
        self.__dict__.pop("walk_", None)

    @override
    def _remap(
        self,
        *,
        user_perm: NDArray[np.intp],
        item_perm: NDArray[np.intp],
        n_users: int,
        n_items: int,
    ) -> None:
        super()._remap(user_perm=user_perm, item_perm=item_perm, n_users=n_users, n_items=n_items)
        if "walk_" in self.__dict__:
            self.walk_ = remap_sparse(self.walk_, item_perm, item_perm, (n_items, n_items))

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
        """Recompute the walk rows the batch can have changed, then re-prune the columns.

        Two widenings on top of the plain two-hop rule. A user the batch names has their
        whole row of ``Pui`` rescaled, because the row is normalized to sum to one, so
        every item of that user moves and not only the ones in the batch. And the second
        pruning pass is *column*-wise: a change in one row can evict an entry from an
        unrelated one, so it is redone over the whole matrix -- which costs the stored
        entries of the walk rather than the catalog, and is the reason ``walk_`` is kept
        at all. The result is exactly what ``fit`` on every batch concatenated gives.
        """
        del delta, new_user_indices, new_item_indices, touched_item_indices
        n_threads = self._check_params()
        if "walk_" not in self.__dict__:
            # The model was `fit`, not `partial_fit`: the pre-pruning matrix every later
            # call splices into does not exist yet, and one full pass recovers it.
            walk = self._walk(interactions, None, n_threads)
        else:
            moved = np.unique(sp.csr_array(interactions[touched_user_indices]).indices)
            rows = affected_item_rows(interactions, moved.astype(np.intp))
            block = self._walk(interactions, rows, n_threads)
            walk = replace_rows(self.walk_, rows, block)
        self.walk_ = walk
        self.similarity_ = self._prune_columns(walk, n_threads)

    def _prune_columns(self, walk: sp.csr_array, n_threads: int) -> sp.csr_array:
        """The reference's second pruning pass, column-wise, over the whole walk matrix."""
        pruned = keep_top_k_per_row(sp.csr_array(walk.T), int(self.n_neighbors), n_threads)
        return sp.csr_array(pruned.T)

    def _walk(
        self, interactions: sp.csr_array, rows: NDArray[np.intp] | None, n_threads: int
    ) -> sp.csr_array:
        """Row-pruned, row-normalized walk rows: ``rows``, or every item when None."""
        n_users, n_items = interactions.shape
        observed = sp.csr_array(interactions, copy=True)
        observed.eliminate_zeros()
        entry_rows = np.repeat(np.arange(n_users), np.diff(observed.indptr))

        # Pui, the user-to-item step: each row normalized to sum to one, then ``alpha``.
        user_sums = np.bincount(entry_rows, weights=np.abs(observed.data), minlength=n_users)
        pui = observed.data * _reciprocal(user_sums)[entry_rows]
        if self.alpha != 1.0:
            pui = pui**self.alpha

        # Every entry of row i of the item-to-user step is popularity[i] ** -alpha, so
        # the walk needs only the two scaling vectors below alongside Pui.
        popularity = np.bincount(observed.indices, minlength=n_items).astype(np.float64)
        seen = popularity > 0
        row_scale, col_scale = np.zeros(n_items), np.zeros(n_items)
        row_scale[seen] = popularity[seen] ** -self.alpha
        col_scale[seen] = popularity[seen] ** -self.beta

        indptr, indices, data = _core.rp3beta_similarity(
            kernel_indices(observed.indptr),
            kernel_indices(observed.indices),
            kernel_data(pui),
            n_items,
            row_scale,
            col_scale,
            int(self.n_neighbors),
            n_threads,
            None if rows is None else kernel_indices(rows),
        )
        n_rows = n_items if rows is None else len(rows)
        walk = sp.csr_array((data, indices, indptr), shape=(n_rows, n_items), dtype=np.float64)

        if self.normalize_similarity:
            row_sums = np.bincount(
                np.repeat(np.arange(n_rows), np.diff(walk.indptr)),
                weights=np.abs(walk.data),
                minlength=n_rows,
            )
            walk.data *= np.repeat(_reciprocal(row_sums), np.diff(walk.indptr))
        return walk

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
        for name in ("alpha", "beta"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Real) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be a finite real number >= 0, got {value!r}.")
        if not isinstance(self.normalize_similarity, bool):
            raise ValueError(
                f"normalize_similarity must be a bool, got {self.normalize_similarity!r}."
            )
        if self.n_jobs is None or self.n_jobs == -1:
            return 0
        if not isinstance(self.n_jobs, numbers.Integral) or self.n_jobs < 1:
            raise ValueError(f"n_jobs must be None, -1 or an integer >= 1, got {self.n_jobs!r}.")
        return int(self.n_jobs)


def _reciprocal(values: NDArray[np.floating]) -> NDArray[np.float64]:
    """``1 / values``, leaving zeros in place of the entries that would divide by zero."""
    # An empty sum comes back as an integer array, which cannot hold the reciprocal.
    values = np.asarray(values, dtype=np.float64)
    return np.divide(1.0, values, out=np.zeros_like(values), where=values > 0)
