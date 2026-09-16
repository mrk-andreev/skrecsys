"""Base classes and helpers for recommender estimators."""

from typing import Any

import numpy as np
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
    ) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
        """Score ``item_indices`` for each query.

        Returns
        -------
        scores : ndarray of shape (n_queries, len(item_indices))
        eligible : ndarray of bool of shape (n_queries, len(item_indices))
            False for items that must not be recommended to the query.
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

        scores, eligible = self._score_queries(X, item_indices, exclude_seen=exclude_seen)

        # Selecting k of n beats sorting all n. The kernel ranks by descending score and
        # breaks ties by column index, which is fitted item order because item_indices is
        # sorted; it raises if a query has fewer than n_recommendations eligible items.
        order = _core.top_k_per_row(
            np.ascontiguousarray(scores, dtype=np.float64),
            np.ascontiguousarray(eligible),
            n_recommendations,
        )
        top_scores = np.take_along_axis(scores, order, axis=1)
        return item_ids[item_indices[order]], top_scores


def is_recommender(estimator: object) -> bool:
    """Return True if the given estimator is a recommender."""
    return get_tags(estimator).estimator_type == "recommender"
