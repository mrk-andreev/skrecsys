"""Conditions: fitted predicates over queries that route them between recommenders."""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from skrecsys.base import ConditionMixin
from skrecsys.typing import override
from skrecsys.utils._param_validation import check_int
from skrecsys.utils.validation import (
    check_ids,
    check_interactions,
    check_queries,
    factorize,
    lookup_ids,
)


class KnownUser(ConditionMixin, BaseEstimator):
    """Hold for the queries that name a user seen during ``fit``.

    Attributes
    ----------
    user_ids_ : ndarray of shape (n_users,)
        Sorted user identifiers seen during ``fit``.

    Examples
    --------
    >>> from skrecsys.compose import KnownUser
    >>> KnownUser().fit([["u1", "a"], ["u2", "b"]]).evaluate(["u1", "u3"]).tolist()
    [True, False]
    """

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Record the users of the interactions ``X``."""
        users, _, _ = check_interactions(X, y)
        self.user_ids_, _ = factorize(users)
        return self

    @override
    def evaluate(self, X: ArrayLike) -> NDArray[np.bool_]:
        check_is_fitted(self)
        _, known = lookup_ids(check_queries(X)[0], self.user_ids_, name="user")
        return known


class MinInteractions(ConditionMixin, BaseEstimator):
    """Hold for the queries whose user has at least ``n_interactions`` interactions.

    Interactions are the rows of the ``X`` of ``fit``, so a repeated user-item pair
    counts each time. A user never seen during ``fit`` has none.

    Parameters
    ----------
    n_interactions : int, default=5
        Minimum number of interactions for the condition to hold.

    Attributes
    ----------
    user_ids_ : ndarray of shape (n_users,)
        Sorted user identifiers seen during ``fit``.
    counts_ : ndarray of shape (n_users,)
        Number of interactions of each user.
    """

    def __init__(self, n_interactions: int = 5) -> None:
        self.n_interactions = n_interactions

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Count the interactions of each user in ``X``."""
        check_int(self.n_interactions, "n_interactions", min_value=0)
        users, _, _ = check_interactions(X, y)
        self.user_ids_, codes = factorize(users)
        self.counts_ = np.bincount(codes, minlength=len(self.user_ids_))
        return self

    @override
    def evaluate(self, X: ArrayLike) -> NDArray[np.bool_]:
        check_is_fitted(self)
        positions, known = lookup_ids(check_queries(X)[0], self.user_ids_, name="user")
        counts = np.where(known, self.counts_[positions], 0)
        return counts >= self.n_interactions


class QueryIn(ConditionMixin, BaseEstimator):
    """Hold for the queries listed in ``ids``, such as the users of an experiment cohort.

    Parameters
    ----------
    ids : array-like of shape (n_ids,)
        Query identifiers for which the condition holds. They need not have been seen
        during ``fit``.

    Attributes
    ----------
    ids_ : ndarray of shape (n_distinct_ids,)
        The distinct identifiers, sorted.
    """

    def __init__(self, ids: ArrayLike = ()) -> None:
        self.ids = ids

    def fit(self, X: ArrayLike | None = None, y: ArrayLike | None = None) -> Self:
        """Validate ``ids``; the interactions are not used."""
        del X, y
        ids = np.asarray(self.ids)
        self.ids_ = np.unique(check_ids(ids, name="ids")) if ids.size else ids.ravel()
        return self

    @override
    def evaluate(self, X: ArrayLike) -> NDArray[np.bool_]:
        check_is_fitted(self)
        return lookup_ids(check_queries(X)[0], self.ids_, name="query")[1]
