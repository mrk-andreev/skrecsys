"""Route each query to one of two recommenders."""

from collections.abc import Iterator
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import _check_feature_names, check_array, check_is_fitted

from skrecsys._tracing import active_tracer, span, traced_recommend
from skrecsys._typing import Condition, FittedRecommender, Recommender, clone_as, override
from skrecsys.base import (
    RecommenderMixin,
    check_n_recommendations,
    evaluate_condition,
    fit_clone,
    is_condition,
    is_recommender,
    predict_pairs,
    uses_time,
)
from skrecsys.utils._param_validation import check_bool, check_component
from skrecsys.utils.validation import check_as_of, check_ids, check_interactions, drop_time

#: Columns of timed interactions: user, item, time.
_N_TIMED_COLUMNS = 3


class Switch(RecommenderMixin, BaseEstimator):
    """Serve each query with ``on_true`` where ``condition`` holds, else with ``on_false``.

    The usual use is cold start: a personalized model for the users it knows and a
    popularity baseline for everyone else. The condition and both branches are fitted on
    the same interactions.

    Scores come from whichever branch served the query, so they are comparable within a
    row but not across rows served by different branches.

    Parameters
    ----------
    condition : condition
        Decides per query, for example :class:`~skrecsys.compose.KnownUser`. Cloned and
        fitted as ``condition_``.
    on_true, on_false : recommender
        The branches, cloned and fitted as ``on_true_`` and ``on_false_``. Either may be a
        composite itself.
    time : bool, default=False
        Whether interactions carry their time, as a third column of ``X``. A branch
        constructed with ``time=True`` too -- a :class:`~skrecsys.compose.Cascade`, say --
        is fitted on it and ranks as of ``as_of``; the condition and any other branch see
        ``[user, item]`` alone.

    Attributes
    ----------
    user_ids_, item_ids_ : ndarray
        Those of ``on_true_``: both branches are fitted on the same interactions.
    n_users_, n_items_ : int
    time_dtype_ : numpy.dtype
        Only when ``time`` is true: the dtype of the fitted times.

    Examples
    --------
    >>> from skrecsys.compose import KnownUser, Switch
    >>> from skrecsys.recommendation import ItemKNNRecommender, MostPopularRecommender
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "b"], ["u2", "c"], ["u3", "c"]]
    >>> rec = Switch(KnownUser(), ItemKNNRecommender(), MostPopularRecommender()).fit(X)
    >>> rec.recommend(["u3", "new-user"], n_recommendations=1)[0].tolist()
    [['b'], ['b']]
    """

    def __init__(
        self,
        condition: Condition,
        on_true: Recommender,
        on_false: Recommender,
        *,
        time: bool = False,
    ) -> None:
        self.condition = condition
        self.on_true = on_true
        self.on_false = on_false
        self.time = time

    @override
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the condition and both branches on the interactions ``X``.

        Parameters
        ----------
        X : array-like of shape (n_interactions, 2) or (n_interactions, 3)
            ``X[:, 0]`` contains user identifiers, ``X[:, 1]`` item identifiers and, when
            ``time`` is true, ``X[:, 2]`` the time of each interaction.
        y : array-like of shape (n_interactions,), default=None
            Interaction values; ``None`` gives every interaction weight 1.

        Returns
        -------
        self : object
        """
        check_component(self.condition, "condition", is_condition, "a condition")
        for name in ("on_true", "on_false"):
            check_component(getattr(self, name), name, is_recommender, "a recommender")
        check_bool(self.time, "time")
        if self.time:
            self.time_dtype_ = check_interactions(X, y, time=True)[3].dtype
        else:
            check_interactions(X, y)
            if hasattr(self, "time_dtype_"):
                del self.time_dtype_
        _check_feature_names(self, X, reset=True)
        # The branches see plain arrays: the feature names are checked here, once, and
        # `predict` hands each branch a slice that could not carry them anyway.
        X = check_array(X, dtype=None, ensure_all_finite=False)
        X_ids = drop_time(X) if self.time else X
        self.condition_ = clone_as(self.condition).fit(X_ids, y)
        self.on_true_ = fit_clone(self.on_true, X if uses_time(self.on_true) else X_ids, y)
        self.on_false_ = fit_clone(self.on_false, X if uses_time(self.on_false) else X_ids, y)
        self.user_ids_ = self.on_true_.user_ids_
        self.item_ids_ = self.on_true_.item_ids_
        self.n_users_ = len(self.user_ids_)
        self.n_items_ = len(self.item_ids_)
        return self

    def _for_branch(self, branch: object, interactions: ArrayLike | None) -> ArrayLike | None:
        """Interactions as ``branch`` takes them: with their time only if it uses time."""
        if not self.time or interactions is None or uses_time(branch):
            return interactions
        arr = check_array(interactions, dtype=None, ensure_all_finite=False)
        return drop_time(arr) if arr.shape[1] == _N_TIMED_COLUMNS else arr

    def _routes(
        self, queries: NDArray[np.generic], *, report: bool = False
    ) -> Iterator[tuple[str, FittedRecommender, NDArray[np.bool_]]]:
        """Each branch, named, with the mask of the queries it serves; idle ones skipped.

        With ``report``, a tracer listening to ``recommend`` is told the route taken.
        """
        mask = evaluate_condition(self.condition_, queries)
        if report and (tracer := active_tracer()) is not None:
            tracer.route(self.condition_, queries, mask)
        for name, branch, rows in (
            ("on_true", self.on_true_, mask),
            ("on_false", self.on_false_, ~mask),
        ):
            if rows.any():
                yield name, branch, rows

    @override
    @traced_recommend
    def recommend(
        self,
        X: ArrayLike,
        *,
        n_recommendations: int = 10,
        candidates: ArrayLike | None = None,
        exclude_seen: bool = True,
        exclude_interactions: ArrayLike | None = None,
        as_of: ArrayLike | None = None,
    ) -> tuple[NDArray[np.generic], NDArray[np.floating]]:
        """Return recommended item identifiers and their scores, each from its branch.

        Parameters are those of :meth:`skrecsys.base.RecommenderMixin.recommend`, and
        are passed to the branches as they are, and:

        as_of : scalar or array-like of shape (n_queries,), default=None
            Only with ``time``: the time the queries are ranked as of, passed to a branch
            that uses time; see :meth:`skrecsys.compose.Cascade.recommend`.
        """
        check_is_fitted(self)
        check_n_recommendations(n_recommendations)
        if as_of is not None and not self.time:
            raise ValueError("as_of needs a Switch constructed with time=True.")
        queries = check_ids(X)
        query_times = check_as_of(as_of, len(queries), self.time_dtype_) if self.time else None
        items = np.empty((len(queries), n_recommendations), dtype=self.item_ids_.dtype)
        scores = np.empty((len(queries), n_recommendations), dtype=np.float64)
        for name, branch, rows in self._routes(queries, report=True):
            excluded = self._for_branch(branch, exclude_interactions)
            with span(name):
                if query_times is not None and uses_time(branch):
                    items[rows], scores[rows] = branch.recommend(
                        queries[rows],
                        n_recommendations=n_recommendations,
                        candidates=candidates,
                        exclude_seen=exclude_seen,
                        exclude_interactions=excluded,
                        as_of=query_times[rows],
                    )
                else:
                    items[rows], scores[rows] = branch.recommend(
                        queries[rows],
                        n_recommendations=n_recommendations,
                        candidates=candidates,
                        exclude_seen=exclude_seen,
                        exclude_interactions=excluded,
                    )
        return items, scores

    @override
    def _count_eligible(
        self,
        X: ArrayLike,
        *,
        candidates: ArrayLike | None = None,
        exclude_seen: bool = True,
        exclude_interactions: ArrayLike | None = None,
    ) -> NDArray[np.int64]:
        check_is_fitted(self)
        queries = check_ids(X)
        counts = np.zeros(len(queries), dtype=np.int64)
        for _, branch, rows in self._routes(queries):
            counts[rows] = branch._count_eligible(
                queries[rows],
                candidates=candidates,
                exclude_seen=exclude_seen,
                exclude_interactions=self._for_branch(branch, exclude_interactions),
            )
        return counts

    def predict(self, X: ArrayLike) -> NDArray[np.floating]:
        """Score user-item pairs, each with the branch its user is routed to.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2) or (n_samples, 3)
            User-item pairs; with ``time``, a third column holds the time each is scored
            as of, which only a branch that uses time reads.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        pairs = check_array(X, dtype=None, ensure_all_finite=False)
        if self.time and pairs.shape[1] != _N_TIMED_COLUMNS:
            raise ValueError(
                "X must have exactly 3 columns (user identifiers, item identifiers, times), "
                f"got {pairs.shape[1]}."
            )
        users = check_interactions(drop_time(pairs) if self.time else pairs)[0]
        _check_feature_names(self, X, reset=False)
        scores = np.empty(len(users), dtype=np.float64)
        for _, branch, rows in self._routes(users):
            routed = pairs[rows] if uses_time(branch) or not self.time else drop_time(pairs[rows])
            scores[rows] = predict_pairs(branch, routed)
        return scores
