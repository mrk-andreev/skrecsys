"""Shared implementation for user-item collaborative-filtering recommenders."""

from itertools import pairwise
from typing import Any, Self

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import _check_feature_names, check_is_fitted

from skrecsys._typing import override
from skrecsys.base import RecommenderMixin
from skrecsys.utils.validation import check_ids, check_interactions, encode_ids


class BaseRecommender(RecommenderMixin, BaseEstimator):
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
        self.user_ids_, user_codes = np.unique(users, return_inverse=True)
        self.item_ids_, item_codes = np.unique(items, return_inverse=True)
        self.n_users_ = len(self.user_ids_)
        self.n_items_ = len(self.item_ids_)
        self.interactions_ = sp.csr_array(
            (weights, (user_codes, item_codes)), shape=(self.n_users_, self.n_items_)
        )
        self.interactions_.sum_duplicates()
        self._fit(self.interactions_)
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
        unique_users, row = np.unique(user_idx, return_inverse=True)
        scores = self._score_users(unique_users, np.arange(self.n_items_))
        return scores[row, item_idx]

    @override
    def _score_queries(
        self, X: ArrayLike, item_indices: NDArray[np.intp], *, exclude_seen: bool
    ) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
        user_idx = encode_ids(check_ids(X), self.user_ids_, name="user")
        scores = self._score_users(user_idx, item_indices)
        eligible = np.ones(scores.shape, dtype=bool)
        if exclude_seen:
            eligible &= ~_structure(self.interactions_[user_idx][:, item_indices])
        return scores, eligible

    def _fit(self, interactions: sp.csr_array) -> None:
        raise NotImplementedError

    def _score_users(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        """Return dense scores of shape (len(user_indices), len(item_indices))."""
        raise NotImplementedError


def _structure(matrix: Any) -> NDArray[np.bool_]:
    """Dense boolean mask of stored entries, including explicit zeros."""
    matrix = sp.csr_array(matrix, copy=True)
    matrix.data = np.ones_like(matrix.data)
    return matrix.toarray() > 0


def keep_top_k_per_row(matrix: sp.csr_array, k: int | None) -> sp.csr_array:
    """Keep the k largest entries of each row; ties keep the lower column index."""
    if k is None:
        return matrix
    indptr, indices, data = matrix.indptr, matrix.indices, matrix.data
    keep = np.zeros(len(data), dtype=bool)
    for start, end in pairwise(indptr):
        if end - start <= k:
            keep[start:end] = True
        else:
            top = np.argsort(-data[start:end], kind="stable")[:k]
            keep[start + top] = True
    row_of = np.repeat(np.arange(matrix.shape[0]), np.diff(indptr))
    return sp.csr_array((data[keep], (row_of[keep], indices[keep])), shape=matrix.shape)
