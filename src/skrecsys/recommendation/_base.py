"""Shared implementation for user-item collaborative-filtering recommenders."""

import numbers
from typing import Any, Self

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import _check_feature_names, check_is_fitted

from skrecsys import _core
from skrecsys._typing import override
from skrecsys.base import RecommenderMixin, seen_among
from skrecsys.indexing import VectorIndexMixin
from skrecsys.utils.validation import (
    check_ids,
    check_interactions,
    encode_ids,
    factorize,
)


class BaseRecommender(VectorIndexMixin, RecommenderMixin, BaseEstimator):
    """Base class for recommenders fitted from user-item interactions.

    Subclasses implement ``_fit(interactions)`` and
    ``_score_users(user_indices, item_indices)``.

    Fitted attributes
    -----------------
    user_ids_ : ndarray of shape (n_users_,)
        Sorted user identifiers seen during ``fit``.
    item_ids_ : ndarray of shape (n_items_,)
        Sorted item identifiers seen during ``fit``.
    n_users_ : int
    n_items_ : int
    interactions_ : scipy.sparse.csr_array of shape (n_users_, n_items_)
        Interaction values; duplicate user-item pairs are summed.
    """

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the recommender from user-item interactions.

        Parameters
        ----------
        X : array-like of shape (n_interactions, 2)
            ``X[:, 0]`` contains user identifiers, ``X[:, 1]`` item identifiers.

        y : array-like of shape (n_interactions,), default=None
            Rating, relevance, interaction weight, or confidence. If None, every
            observed interaction has weight 1.

        Returns
        -------
        self : object
        """
        users, items, weights = check_interactions(X, y)
        _check_feature_names(self, X, reset=True)
        self.user_ids_, user_codes = factorize(users)
        self.item_ids_, item_codes = factorize(items)
        self.n_users_ = len(self.user_ids_)
        self.n_items_ = len(self.item_ids_)
        # The kernel returns the canonical form directly -- one entry per pair, columns
        # ascending -- which scipy reaches by building a COO and sorting each row.
        indptr, indices, data = _core.coo_to_csr(
            kernel_indices(user_codes),
            kernel_indices(item_codes),
            kernel_data(weights),
            self.n_users_,
            self.n_items_,
            self._build_threads(),
        )
        self.interactions_ = sp.csr_array(
            (data, indices, indptr), shape=(self.n_users_, self.n_items_)
        )
        self._fit(self.interactions_)
        self._fit_index()
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.floating]:
        """Score user-item pairs.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            User-item pairs; both identifiers must have been seen during ``fit``.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        users, items, _ = check_interactions(X)
        _check_feature_names(self, X, reset=False)
        user_idx = encode_ids(users, self.user_ids_, name="user")
        item_idx = encode_ids(items, self.item_ids_, name="item")
        return self._score_pairs(user_idx, item_idx)

    @override
    def _score_queries(
        self, X: ArrayLike, item_indices: NDArray[np.intp], *, exclude_seen: bool
    ) -> tuple[NDArray[np.floating], sp.csr_array]:
        user_idx = encode_ids(check_ids(X), self.user_ids_, name="user")
        scores = self._score_users(user_idx, item_indices)
        shape = (len(user_idx), len(item_indices))
        if not exclude_seen:
            return scores, sp.csr_array(shape, dtype=bool)
        return scores, seen_among(self.interactions_, user_idx, item_indices)

    def _build_threads(self) -> int:
        """Threads for the kernels that run before the estimator sees its matrix.

        ``n_jobs`` is the estimator's own, and each one validates it in ``_fit``; here
        anything unusable simply means every core, and the real complaint follows.
        """
        n_jobs = getattr(self, "n_jobs", None)
        if isinstance(n_jobs, numbers.Integral) and not isinstance(n_jobs, bool) and n_jobs >= 1:
            return int(n_jobs)
        return 0

    def _rank_by_factors(
        self,
        queries: NDArray[Any],
        item_indices: NDArray[np.intp],
        k: int,
        *,
        exclude_seen: bool,
        excluded: sp.csr_array | None = None,
        first_query: int,
        user_factors: NDArray[np.floating],
        item_factors: NDArray[np.floating],
        item_bias: NDArray[np.floating] | None = None,
        user_offset: NDArray[np.floating] | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Rank a model scoring ``item_bias[j] + <p_u, q_j> + user_offset[u]``.

        The exact path of the latent-factor models, and of most-popular as the model
        with no factors. The kernel scores a tile of queries against each item vector
        and keeps only the ``k`` best per query, so neither the dense score matrix nor
        the bias-broadcast temporaries around it are built; the fitted arrays go over
        whole, with the rows being asked for.
        """
        user_idx = encode_ids(check_ids(queries), self.user_ids_, name="user")
        seen = self.interactions_ if exclude_seen else None
        return _core.recommend_from_factors(
            kernel_matrix(user_factors),
            kernel_matrix(item_factors),
            None if item_bias is None else kernel_data(item_bias),
            None if user_offset is None else kernel_data(user_offset),
            kernel_indices(user_idx),
            None if seen is None else kernel_indices(seen.indptr),
            None if seen is None else kernel_indices(seen.indices),
            kernel_candidates(item_indices, self.n_items_),
            k,
            self._build_threads(),
            first_query,
            **kernel_excluded(excluded),
        )

    def _fit(self, interactions: sp.csr_array) -> None:
        raise NotImplementedError

    def _score_users(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        """Return dense scores of shape (len(user_indices), len(item_indices))."""
        raise NotImplementedError

    def _score_pairs(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        """Return scores of shape (n_pairs,) for the aligned pairs.

        The fallback scores the whole catalog for every distinct user and reads back the
        requested cells, which costs a dense ``(n_distinct_users, n_items_)`` matrix.
        Subclasses that can evaluate a pair on its own override this.
        """
        unique_users, row = np.unique(user_indices, return_inverse=True)
        scores = self._score_users(unique_users, np.arange(self.n_items_))
        return scores[row, item_indices]


class SimilarityRecommender(BaseRecommender):
    """Recommender scoring a user by the item weights of everything they interacted with.

    Subclasses fit ``similarity_`` and set ``_neighbors_by_row``: ``True`` when row ``j``
    holds the neighbours that make up the score of item ``j``, as item-KNN stores them,
    and ``False`` when column ``j`` does.

    ``recommend`` goes through a kernel that scores a query and reduces it to its ``k``
    best entries before moving to the next, so it never builds the dense
    ``(n_queries, n_items_)`` matrix that ``_score_users`` returns.
    """

    _neighbors_by_row: bool = False
    similarity_: sp.csr_array

    def _check_params(self) -> int:
        """Validate parameters and return the thread count for the kernel (0 = all)."""
        raise NotImplementedError

    def _neighbors_of_items(self) -> sp.csr_array:
        """Row ``j`` holds the weights making up the score of item ``j``."""
        return self.similarity_ if self._neighbors_by_row else self._transposed_similarity()

    def _weights_of_items(self) -> sp.csr_array:
        """Row ``i`` holds what an interaction with item ``i`` contributes to."""
        return self._transposed_similarity() if self._neighbors_by_row else self.similarity_

    def _transposed_similarity(self) -> sp.csr_array:
        """``similarity_.T`` as CSR, kept against the matrix it was transposed from.

        Both orientations are needed -- one to score a pair, the other to score a query --
        and the transpose costs `O(nnz)`, which dwarfs scoring a single query. Memoizing
        against the fitted matrix itself means a refit invalidates it.
        """
        cached = getattr(self, "_similarity_t", None)
        if cached is None or cached[0] is not self.similarity_:
            cached = (self.similarity_, sp.csr_array(self.similarity_.T))
            self._similarity_t = cached
        return cached[1]

    @override
    def _score_users(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        scores = sp.csr_array(self.interactions_[user_indices] @ self._weights_of_items())
        return np.asarray(scores[:, item_indices].toarray(), dtype=np.float64)

    @override
    def _score_pairs(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        return score_pairs_from_similarity(
            self.interactions_,
            self._neighbors_of_items(),
            user_indices,
            item_indices,
            by_row=True,
        )

    @override
    def _rank_chunk_size(self, n_candidates: int, k: int) -> int:
        # The kernel keeps one accumulator per thread, so a block only holds its results.
        del n_candidates, k
        return 65_536

    @override
    def _rank_queries_exact(
        self,
        queries: NDArray[Any],
        item_indices: NDArray[np.intp],
        k: int,
        *,
        exclude_seen: bool,
        excluded: sp.csr_array | None = None,
        first_query: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.floating]]:
        user_idx = encode_ids(check_ids(queries), self.user_ids_, name="user")
        # The whole interaction matrix goes over with the rows being asked for: the
        # kernel reads each query's row -- to score it, and for its exclusions -- in
        # place, where slicing a copy per call cost more than the scoring.
        return _core.recommend_from_similarity(
            *kernel_csr(self.interactions_),
            kernel_indices(user_idx),
            *kernel_csr(self._weights_of_items()),
            kernel_candidates(item_indices, self.n_items_),
            exclude_seen,
            k,
            self._check_params(),
            first_query,
            **kernel_excluded(excluded),
        )


def score_pairs_from_similarity(
    interactions: sp.csr_array,
    similarity: Any,
    user_indices: NDArray[np.intp],
    item_indices: NDArray[np.intp],
    *,
    by_row: bool,
) -> NDArray[np.float64]:
    """Score the aligned pairs as the interactions of ``u`` weighted by those of ``j``.

    ``by_row`` says where the weights of item ``j`` live: in row ``j`` of ``similarity``,
    as the item-KNN model stores them, or in its column ``j``. The cost is the number of
    stored interactions of the queried users rather than the size of the catalog.
    """
    neighbors = similarity if by_row else similarity.T
    rows = sp.csr_array(interactions[user_indices])
    if sp.issparse(neighbors):
        block = sp.csr_array(neighbors)[item_indices]
        return np.asarray(rows.multiply(block).sum(axis=1), dtype=np.float64).ravel()
    # A dense similarity is gathered entry by entry: only the stored interactions of a
    # row can contribute, so a column of `neighbors` is never materialized in full.
    counts = np.diff(rows.indptr)
    weights = np.asarray(neighbors)[np.repeat(item_indices, counts), rows.indices]
    owners = np.repeat(np.arange(len(item_indices)), counts)
    return np.bincount(owners, weights=rows.data * weights, minlength=len(item_indices))


def kernel_indices(values: Any) -> NDArray[np.int64]:
    """An index array in the layout the kernels borrow, copied only when it must be.

    The kernels read `int64` indices straight out of the buffer numpy hands over, so an
    array that already has that dtype and layout -- which is what scipy stores here --
    passes through untouched. ``astype`` copied `nnz` values on every call instead, which
    dominated a small ``recommend`` once the dense score matrix was gone.
    """
    return np.ascontiguousarray(values, dtype=np.int64)


def kernel_data(values: Any) -> NDArray[np.float64]:
    """A value array in the layout the kernels borrow, copied only when it must be."""
    return np.ascontiguousarray(values, dtype=np.float64)


def kernel_matrix(values: Any) -> NDArray[np.float64]:
    """A dense matrix in the C-contiguous `float64` layout the kernels borrow."""
    return np.ascontiguousarray(values, dtype=np.float64)


def kernel_candidates(item_indices: NDArray[np.intp], n_items: int) -> NDArray[np.int64] | None:
    """The candidate item indices for a recommend kernel, or ``None`` for every item.

    ``recommend`` passes the candidates sorted and distinct, so as many as there are
    fitted items is all of them; the kernel then skips the catalog-sized position map a
    subset needs.
    """
    return None if len(item_indices) == n_items else kernel_indices(item_indices)


def kernel_excluded(excluded: sp.csr_array | None) -> dict[str, NDArray[np.int64]]:
    """The keyword arguments handing a recommend kernel per-query exclusions, if any.

    ``excluded`` holds candidate positions, one row per query of the block; the kernel
    skips them on top of the seen items it reads from the interaction matrix itself.
    """
    if excluded is None or not excluded.nnz:
        return {}
    excluded.sort_indices()
    return {
        "excluded_indptr": kernel_indices(excluded.indptr),
        "excluded_indices": kernel_indices(excluded.indices),
    }


def kernel_csr(matrix: sp.csr_array) -> tuple[Any, Any, Any]:
    """A sparse matrix as the ``(indptr, indices, data)`` triple the kernels take."""
    return (
        kernel_indices(matrix.indptr),
        kernel_indices(matrix.indices),
        kernel_data(matrix.data),
    )


def keep_top_k_per_row(matrix: sp.csr_array, k: int | None, n_threads: int = 0) -> sp.csr_array:
    """Keep the k largest entries of each row; ties keep the lower column index.

    ``n_threads`` is passed to the kernel, where 0 means every core.
    """
    if k is None:
        return matrix
    indptr, indices, data = _core.csr_top_k_per_row(
        *kernel_csr(matrix),
        matrix.shape[1],
        int(k),
        n_threads,
    )
    return sp.csr_array((data, indices, indptr), shape=matrix.shape, dtype=np.float64)
