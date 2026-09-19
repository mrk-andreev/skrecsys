"""Popularity baseline."""

from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from skrecsys._typing import override
from skrecsys.recommendation._base import BaseRecommender
from skrecsys.recommendation._incremental import IncrementalRecommenderMixin


class MostPopularRecommender(IncrementalRecommenderMixin, BaseRecommender):
    """Recommend the most popular items to every user.

    A deterministic, non-personalized baseline for implicit and explicit feedback.

    Parameters
    ----------
    weighting : {"count", "sum"}, default="count"
        ``"count"`` scores an item by the number of distinct users who interacted with
        it. ``"sum"`` scores it by the sum of interaction values.

    Attributes
    ----------
    item_popularity_ : ndarray of shape (n_items_,)
        Popularity score of each fitted item.

    Notes
    -----
    ``partial_fit`` adds the batch to the counts and is exact: the scores are what
    ``fit`` on every batch concatenated would have produced.

    Examples
    --------
    >>> from skrecsys.recommendation import MostPopularRecommender
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "b"], ["u3", "c"]]
    >>> rec = MostPopularRecommender().fit(X)
    >>> items, scores = rec.recommend(["u3"], n_recommendations=2)
    >>> items.tolist()
    [['b', 'a']]
    """

    def __init__(self, weighting: str = "count") -> None:
        self.weighting = weighting

    _incremental_state_ = (("item_popularity_", "item"),)

    def _check_params(self) -> None:
        """Validate parameters."""
        if self.weighting not in ("count", "sum"):
            raise ValueError(f"weighting must be 'count' or 'sum', got {self.weighting!r}.")

    @override
    def _fit(self, interactions: sp.csr_array) -> None:
        self._check_params()
        if self.weighting == "count":
            popularity = np.diff(interactions.tocsc().indptr)
        else:
            popularity = interactions.sum(axis=0)
        self.item_popularity_ = np.asarray(popularity, dtype=np.float64).ravel()

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
        """Add the batch to the popularity counts, exactly.

        Both weightings are what ``fit`` on every batch concatenated would have
        produced. ``"sum"`` is additive outright; ``"count"`` is not, because a pair the
        batch repeats was already counted, so the columns the batch touched are counted
        again from the accumulated matrix rather than incremented.
        """
        del new_user_indices, new_item_indices, touched_user_indices
        if self.weighting == "sum":
            self.item_popularity_ += np.asarray(delta.sum(axis=0), dtype=np.float64).ravel()
            return
        touched = np.isin(interactions.indices, touched_item_indices)
        counts = np.bincount(interactions.indices[touched], minlength=self.n_items_)
        self.item_popularity_[touched_item_indices] = counts[touched_item_indices]

    @override
    def _score_users(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        return np.tile(self.item_popularity_[item_indices], (len(user_indices), 1))

    @override
    def _score_pairs(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        return self.item_popularity_[item_indices]

    @override
    def _rank_chunk_size(self, n_candidates: int, k: int) -> int:
        # The kernel builds no dense score matrix, so a block only holds its results.
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
        # A latent-factor model with no factors: the popularity is the item bias, and
        # the zero-width factor matrices cost nothing to allocate or scan.
        return self._rank_by_factors(
            queries,
            item_indices,
            k,
            exclude_seen=exclude_seen,
            excluded=excluded,
            first_query=first_query,
            user_factors=np.empty((self.n_users_, 0)),
            item_factors=np.empty((self.n_items_, 0)),
            item_bias=self.item_popularity_,
        )
