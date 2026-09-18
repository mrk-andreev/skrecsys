"""Base classes and helpers for recommender estimators."""

from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import get_tags
from sklearn.utils.validation import check_is_fitted

from skrecsys import _core
from skrecsys.utils.validation import check_ids, encode_ids

__all__ = ["RecommenderMixin", "is_recommender"]


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
        size = self._rank_chunk_size(len(item_indices))
        items = np.empty((len(queries), n_recommendations), dtype=item_ids.dtype)
        top_scores = np.empty((len(queries), n_recommendations), dtype=np.float64)
        for start in range(0, len(queries), size):
            stop = min(start + size, len(queries))
            order, scores = self._rank_queries(
                queries[start:stop],
                item_indices,
                n_recommendations,
                exclude_seen=exclude_seen,
                first_query=start,
            )
            items[start:stop] = item_ids[item_indices[order]]
            top_scores[start:stop] = scores
        return items, top_scores

    def _rank_chunk_size(self, n_candidates: int) -> int:
        """Queries ranked per block, holding the dense score matrix near 64 MB."""
        return max(1, min(8_000_000 // max(n_candidates, 1), 8192))

    def _rank_queries(
        self,
        queries: NDArray[Any],
        item_indices: NDArray[np.intp],
        k: int,
        *,
        exclude_seen: bool,
        first_query: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.floating]]:
        """Return the ``k`` best candidate positions of each query, and their scores.

        ``first_query`` is the position of ``queries[0]`` among all the queries, so that
        an error names the query the caller asked about. Estimators that can rank without
        a dense score matrix override this.
        """
        scores, excluded = self._score_queries(queries, item_indices, exclude_seen=exclude_seen)
        excluded.sort_indices()
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
