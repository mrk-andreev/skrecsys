"""Scorers adapting top-k metrics to scikit-learn model selection."""

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from skrecsys._typing import override
from skrecsys.base import is_recommender
from skrecsys.utils.validation import check_interactions

__all__ = ["make_recommender_scorer"]


class _RecommenderScorer:
    """Callable ``scorer(estimator, X, y=None)`` evaluating held-out interactions."""

    def __init__(
        self,
        metric: Callable[..., Any],
        k: int,
        candidates: ArrayLike | None,
        *,
        exclude_seen: bool,
        kwargs: dict[str, Any],
    ) -> None:
        self._metric = metric
        self._k = k
        self._candidates = candidates
        self._exclude_seen = exclude_seen
        self._kwargs = kwargs

    def __call__(self, estimator: Any, X: ArrayLike, y: ArrayLike | None = None) -> float:
        if not is_recommender(estimator):
            raise TypeError(f"{type(estimator).__name__} is not a recommender.")
        users, items, weights = check_interactions(X, y)
        relevant = weights > 0
        users, items = users[relevant], items[relevant]
        if len(users) == 0:
            raise ValueError("No held-out interactions with positive relevance to score.")

        query_users, codes = np.unique(users, return_inverse=True)
        order = np.argsort(codes, kind="stable")
        boundaries = np.cumsum(np.bincount(codes, minlength=len(query_users)))[:-1]
        y_true = [set(group.tolist()) for group in np.split(items[order], boundaries)]

        y_pred, _ = estimator.recommend(
            query_users,
            n_recommendations=self._k,
            candidates=self._candidates,
            exclude_seen=self._exclude_seen,
        )
        kwargs = dict(self._kwargs)
        # Corpus-level metrics such as catalog_coverage_at_k have no per-query form and
        # therefore no ``average`` parameter.
        if "average" in inspect.signature(self._metric).parameters:
            kwargs.setdefault("average", "macro")
        return float(self._metric(y_true, y_pred, k=self._k, **kwargs))

    @override
    def __repr__(self) -> str:
        name = getattr(self._metric, "__name__", repr(self._metric))
        return f"make_recommender_scorer({name}, k={self._k}, exclude_seen={self._exclude_seen})"


def make_recommender_scorer(
    metric: Callable[..., Any],
    *,
    k: int = 10,
    candidates: ArrayLike | None = None,
    exclude_seen: bool = True,
    **kwargs: Any,
) -> _RecommenderScorer:
    """Make a scorer from a top-k ranking metric.

    The scorer groups held-out interactions ``(X, y)`` by user, calls
    ``estimator.recommend`` for those users and evaluates the ranked items against each
    user's held-out items. Interactions with ``y <= 0`` are not relevant; if ``y`` is
    None every held-out interaction is relevant.

    Parameters
    ----------
    metric : callable
        A metric such as :func:`skrecsys.metrics.ndcg_at_k`. Beyond-accuracy metrics
        work too; pass their extra arguments, for instance
        ``make_recommender_scorer(novelty_at_k, item_popularity=item_popularity(X))``.
    k : int, default=10
        Number of recommendations requested and the metric cutoff.
    candidates : array-like, default=None
        Shared candidate item set. ``None`` ranks the full fitted catalog. Sampled
        negative evaluation must pass an explicit candidate set.
    exclude_seen : bool, default=True
        Passed to ``recommend``; removes training interactions from the ranking.
    **kwargs
        Additional keyword arguments passed to ``metric``.

    Returns
    -------
    scorer : callable
        ``scorer(estimator, X, y=None) -> float``, usable as ``scoring`` in
        :func:`sklearn.model_selection.cross_validate` and
        :class:`sklearn.model_selection.GridSearchCV`.
    """
    return _RecommenderScorer(metric, k, candidates, exclude_seen=exclude_seen, kwargs=kwargs)
