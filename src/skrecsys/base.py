"""Base classes and helpers for recommender estimators."""

from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import get_tags
from sklearn.utils.validation import check_is_fitted

from skrecsys import _core
from skrecsys.utils.validation import (
    check_ids,
    check_interactions,
    encode_ids,
    factorize,
    lookup_ids,
)

__all__ = [
    "RecommenderMixin",
    "excluded_among",
    "is_recommender",
    "seen_among",
    "supports_partial_fit",
]


class RecommenderMixin:
    """Mixin class for all recommenders.

    Defines the ``recommend`` operation and sets the ``estimator_type`` tag to
    ``"recommender"``. The semantic type of a query is left to the estimator: subclasses
    implement ``_score_queries`` to score candidate items for each query in ``X``.

    The mixin deliberately provides no default ``score``: rating error, ranking quality
    and retrieval quality are different objectives. Use
    :func:`skrecsys.metrics.make_recommender_scorer` for model selection.

    Estimators using this mixin must expose a fitted ``item_ids_`` attribute.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()  # ty: ignore[unresolved-attribute]
        tags.estimator_type = "recommender"
        tags.target_tags.required = False
        tags.input_tags.string = True
        return tags

    def _score_queries(
        self, X: ArrayLike, item_indices: NDArray[np.intp], *, exclude_seen: bool
    ) -> tuple[NDArray[np.floating], sp.csr_array]:
        """Score ``item_indices`` for each query.

        Returns
        -------
        scores : ndarray of shape (n_queries, len(item_indices))
        excluded : scipy.sparse.csr_array of shape (n_queries, len(item_indices))
            Stored positions must not be recommended to the query. Sparse rather than a
            dense mask because the exclusions are the queries' own interactions, which
            are a vanishing fraction of the catalog.
        """
        raise NotImplementedError

    def recommend(
        self,
        X: ArrayLike,
        *,
        n_recommendations: int = 10,
        candidates: ArrayLike | None = None,
        exclude_seen: bool = True,
        exclude_interactions: ArrayLike | None = None,
    ) -> tuple[NDArray[Any], NDArray[np.floating]]:
        """Return recommended item identifiers and their scores.

        Parameters
        ----------
        X : array-like of shape (n_queries,)
            Queries. For the estimators in :mod:`skrecsys.recommendation` these are
            user identifiers seen during ``fit``.

        n_recommendations : int, default=10
            Number of items to return per query.

        candidates : array-like of shape (n_candidates,), default=None
            Item identifiers eligible for recommendation, shared by all queries.
            ``None`` means all fitted items.

        exclude_seen : bool, default=True
            Whether to remove items observed for the query during ``fit``.

        exclude_interactions : array-like of shape (n_interactions, 2), default=None
            Further user-item pairs that must not be recommended, laid out like the
            ``X`` of ``fit``: ``[:, 0]`` names a query, ``[:, 1]`` an item. Each pair
            removes its item from every query in ``X`` equal to its user, on top of
            ``exclude_seen`` and independently of it. Pairs whose user is not among the
            queries, or whose item the model does not know, are ignored, so the events a
            user produced since the model was fitted can be passed as they are -- the
            same rows a later ``partial_fit`` would take.

        Returns
        -------
        items : ndarray of shape (n_queries, n_recommendations)
            Recommended item identifiers ranked by descending score. Ties are resolved
            by fitted item order.

        scores : ndarray of shape (n_queries, n_recommendations)
            Scores of the recommended items.

        Raises
        ------
        ValueError
            If any query has fewer than ``n_recommendations`` eligible items.
        """
        check_is_fitted(self)
        item_ids: NDArray[Any] = self.item_ids_  # ty: ignore[unresolved-attribute]
        if isinstance(n_recommendations, bool) or not isinstance(
            n_recommendations, int | np.integer
        ):
            raise TypeError(
                f"n_recommendations must be an integer, got {type(n_recommendations).__name__}."
            )
        if n_recommendations < 1:
            raise ValueError(f"n_recommendations must be >= 1, got {n_recommendations}.")

        if candidates is None:
            item_indices = np.arange(len(item_ids))
        else:
            candidate_ids = check_ids(candidates, name="candidates")
            item_indices = np.unique(encode_ids(candidate_ids, item_ids, name="item"))

        # Queries are ranked in blocks: whatever a block scores is reduced to k columns
        # before the next one starts, so peak memory follows the block, not the query.
        queries = check_ids(X)
        excluded = (
            None
            if exclude_interactions is None
            else excluded_among(queries, exclude_interactions, item_ids, item_indices)
        )
        size = self._rank_chunk_size(len(item_indices), n_recommendations)
        items = np.empty((len(queries), n_recommendations), dtype=item_ids.dtype)
        top_scores = np.empty((len(queries), n_recommendations), dtype=np.float64)
        for start in range(0, len(queries), size):
            stop = min(start + size, len(queries))
            order, scores = self._rank_queries(
                queries[start:stop],
                item_indices,
                n_recommendations,
                exclude_seen=exclude_seen,
                excluded=None if excluded is None else excluded[start:stop],
                first_query=start,
            )
            items[start:stop] = item_ids[item_indices[order]]
            top_scores[start:stop] = scores
        return items, top_scores

    def _rank_chunk_size(self, n_candidates: int, k: int) -> int:
        """Queries ranked per block, holding the dense score matrix near 64 MB.

        ``k`` is unused here and is passed because an estimator that ranks without a
        dense matrix sizes its blocks by what it *does* build, which can depend on how
        many items each query asks for.
        """
        del k
        return max(1, min(8_000_000 // max(n_candidates, 1), 8192))

    def _rank_queries(
        self,
        queries: NDArray[Any],
        item_indices: NDArray[np.intp],
        k: int,
        *,
        exclude_seen: bool,
        excluded: sp.csr_array | None = None,
        first_query: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.floating]]:
        """Return the ``k`` best candidate positions of each query, and their scores.

        The one place ``recommend`` reaches for a ranking, and therefore the place an
        approximate index gets to intervene: :class:`skrecsys.indexing.VectorIndexMixin`
        overrides this and falls back to ``_rank_queries_exact``.
        """
        return self._rank_queries_exact(
            queries,
            item_indices,
            k,
            exclude_seen=exclude_seen,
            excluded=excluded,
            first_query=first_query,
        )

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
        """Rank by scoring every candidate, exactly.

        ``excluded`` holds, per query, further candidate positions to skip beyond what
        ``exclude_seen`` removes. ``first_query`` is the position of ``queries[0]`` among
        all the queries, so that an error names the query the caller asked about.
        Estimators that can rank without a dense score matrix override this.
        """
        scores, seen = self._score_queries(queries, item_indices, exclude_seen=exclude_seen)
        excluded = union_of_exclusions(seen, excluded)
        check_enough_eligible(excluded, len(item_indices), k, first_query)

        # Selecting k of n beats sorting all n. The kernel ranks by descending score and
        # breaks ties by column index, which is fitted item order because item_indices is
        # sorted.
        order = _core.top_k_per_row(
            np.ascontiguousarray(scores, dtype=np.float64),
            np.ascontiguousarray(excluded.indptr, dtype=np.int64),
            np.ascontiguousarray(excluded.indices, dtype=np.int64),
            k,
        )
        return order, np.take_along_axis(scores, order, axis=1)


def check_enough_eligible(
    excluded: sp.csr_array, n_candidates: int, k: int, first_query: int
) -> None:
    """Raise if a query has fewer than ``k`` candidates left once its exclusions go.

    The kernels check this too, but only they know the row within the block; the count
    is exact here because the exclusions are candidate positions, each stored once.
    """
    available = n_candidates - np.diff(excluded.indptr)
    short = np.flatnonzero(available < k)
    if short.size:
        row = int(short[0])
        raise ValueError(
            f"Cannot recommend {k} items: query {row + first_query} has only "
            f"{available[row]} eligible items."
        )


def is_recommender(estimator: object) -> bool:
    """Return True if the given estimator is a recommender."""
    return get_tags(estimator).estimator_type == "recommender"


def supports_partial_fit(estimator: object) -> bool:
    """Return True if the estimator can be fitted one batch at a time.

    Probing for the method is how scikit-learn itself decides -- ``learning_curve`` and
    the common checks do exactly this -- because there is no tag for incremental
    learning and no ``PartialFitMixin`` to check against. It is also why
    :class:`skrecsys.recommendation.IncrementalRecommenderMixin` is mixed into the
    estimators that have an incremental update rather than being folded into
    :class:`~skrecsys.recommendation.BaseRecommender`: the probe has to stay truthful.
    """
    return hasattr(estimator, "partial_fit")


def seen_among(
    interactions: sp.csr_array,
    user_indices: NDArray[np.intp],
    item_indices: NDArray[np.intp],
) -> sp.csr_array:
    """Stored interactions of the given users, in candidate coordinates.

    ``item_indices`` must be sorted, which is what ``recommend`` passes, so the column
    remapping preserves the ascending order of the CSR indices. Only the interactions
    themselves are touched: nothing here is proportional to the catalog size except a
    single lookup table, and only when the candidates are a subset.
    """
    rows = sp.csr_array(interactions[user_indices])
    shape = (len(user_indices), len(item_indices))
    if len(item_indices) == interactions.shape[1]:
        # Every fitted item is a candidate, so the item index is its own position.
        indptr, columns = rows.indptr, rows.indices
    else:
        position = np.full(interactions.shape[1], -1, dtype=np.int64)
        position[item_indices] = np.arange(len(item_indices))
        columns = position[rows.indices]
        keep = columns >= 0
        owners = np.repeat(np.arange(shape[0]), np.diff(rows.indptr))[keep]
        columns = columns[keep]
        indptr = np.concatenate([[0], np.cumsum(np.bincount(owners, minlength=shape[0]))])
    return sp.csr_array((np.ones(len(columns), dtype=bool), columns, indptr), shape=shape)


def union_of_exclusions(first: sp.csr_array, second: sp.csr_array | None) -> sp.csr_array:
    """Both per-query exclusion masks as one, sorted, each position stored once."""
    if second is not None and second.nnz:
        first = sp.csr_array(first.astype(bool) + second.astype(bool))
    first.sort_indices()
    return first


def excluded_among(
    queries: NDArray[Any],
    exclude_interactions: ArrayLike,
    item_ids: NDArray[Any],
    item_indices: NDArray[np.intp],
) -> sp.csr_array:
    """The pairs of ``exclude_interactions`` that apply, in candidate coordinates.

    Row ``q`` holds the candidate positions that some pair ``(queries[q], item)`` names.
    Pairs are matched against the queries themselves rather than against every fitted
    user, so the matrix built here has a row per *distinct query*, however many users
    the model has; a pair for another user or an unknown item is dropped, not an error.
    """
    users, items, _ = check_interactions(exclude_interactions)
    distinct, query_rows = factorize(queries)
    user_pos, user_known = lookup_ids(users, distinct, name="user")
    item_pos, item_known = lookup_ids(items, item_ids, name="item")
    keep = user_known & item_known
    pairs = sp.csr_array(
        (np.ones(int(keep.sum()), dtype=bool), (user_pos[keep], item_pos[keep])),
        shape=(len(distinct), len(item_ids)),
    )
    pairs.sum_duplicates()
    return seen_among(pairs, query_rows, item_indices)
