"""Scorers adapting top-k metrics to scikit-learn model selection."""

import functools
import inspect
from collections.abc import Hashable, Mapping, Sequence
from typing import overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from skrecsys._typing import RankingMetric, Recommender, override
from skrecsys.base import first_row_of, first_time_of, is_recommender, uses_time
from skrecsys.utils.validation import (
    check_interactions,
    check_rows,
    interaction_context,
    stack_columns,
)

__all__ = ["evaluate_recommender", "make_recommender_scorer"]

#: One metric, a list of them, or a mapping from result name to metric.
Metrics = RankingMetric | Sequence[RankingMetric] | Mapping[str, RankingMetric]


def _held_out_by_user(
    X: ArrayLike, y: ArrayLike | None, *, time: bool
) -> tuple[NDArray[np.generic], list[set[Hashable]], NDArray[np.generic] | None]:
    """Group the held-out interactions with positive relevance into one item set per user.

    Returns the queries to ask ``recommend`` -- the distinct users, or with context
    columns in ``X`` a matrix ``[user, context...]`` holding the context of each user's
    first held-out interaction --, the relevant items of each, and with ``time`` the time
    each user is ranked as of: when their held-out interactions begin, over all of them,
    relevant or not.
    """
    X_arr = check_rows(X)
    if time:
        users, items, weights, times = check_interactions(X_arr, y, time=True)
    else:
        users, items, weights = check_interactions(X_arr, y)
        times = None
    relevant = weights > 0
    if not relevant.any():
        raise ValueError("No held-out interactions with positive relevance to score.")

    query_users, codes = np.unique(users[relevant], return_inverse=True)
    order = np.argsort(codes, kind="stable")
    boundaries = np.cumsum(np.bincount(codes, minlength=len(query_users)))[:-1]
    y_true = [set(group.tolist()) for group in np.split(items[relevant][order], boundaries)]
    as_of = None if times is None else first_time_of(users, times, query_users)
    context = interaction_context(X_arr, time=time)
    if context is None:
        return query_users, y_true, as_of
    first = context[first_row_of(users, times, query_users)]
    return stack_columns([query_users, *first.T]), y_true, as_of


def _metric_name(metric: RankingMetric) -> str:
    """``ndcg_at_k`` for :func:`ndcg_at_k`, looking through ``functools.partial``."""
    while isinstance(metric, functools.partial):
        metric = metric.func
    return str(getattr(metric, "__name__", type(metric).__name__))


def _named_metrics(metrics: Metrics) -> dict[str, RankingMetric]:
    if isinstance(metrics, Mapping):
        named = dict(metrics)
    else:
        listed = list(metrics) if isinstance(metrics, Sequence) else [metrics]
        named = {}
        for metric in listed:
            name = _metric_name(metric).removesuffix("_at_k")
            if name in named:
                raise ValueError(f"Two metrics are named {name!r}; pass a dict to name them apart.")
            named[name] = metric
    if not named:
        raise ValueError("metrics must name at least one metric.")
    return named


def _cutoffs(k: int | Sequence[int]) -> list[int]:
    cutoffs = list(k) if isinstance(k, Sequence) and not isinstance(k, str) else [k]
    if not cutoffs:
        raise ValueError("k must name at least one cutoff.")
    for cutoff in cutoffs:
        if isinstance(cutoff, bool) or not isinstance(cutoff, int | np.integer) or cutoff < 1:
            raise ValueError(f"k must be a positive integer or a sequence of them, got {k!r}.")
    return sorted({int(cutoff) for cutoff in cutoffs})


def _score(metric: RankingMetric, y_true: list[set[Hashable]], y_pred: ArrayLike, k: int) -> float:
    # Corpus-level metrics such as catalog_coverage_at_k have no per-query form and
    # therefore no ``average`` parameter.
    kwargs = {"average": "macro"} if "average" in inspect.signature(metric).parameters else {}
    return float(metric(y_true, y_pred, k=k, **kwargs))


def evaluate_recommender(
    estimator: Recommender,
    X: ArrayLike,
    y: ArrayLike | None = None,
    *,
    metrics: Metrics,
    k: int | Sequence[int] = 10,
    candidates: ArrayLike | None = None,
    exclude_seen: bool = True,
    exclude_interactions: ArrayLike | None = None,
) -> dict[str, float]:
    """Evaluate several top-k metrics at several cutoffs from one ranking.

    Held-out interactions ``(X, y)`` are grouped by user as in
    :func:`make_recommender_scorer`. ``estimator.recommend`` is called once, for
    ``max(k)`` items per user, and every metric is evaluated on the first ``k`` of them for
    every cutoff, so the grid costs one ranking rather than one per metric and cutoff.

    Parameters
    ----------
    estimator : recommender
        A fitted recommender.
    X : array-like of shape (n_interactions, n_system + n_context)
        Held-out ``(user, item)`` pairs, laid out like the ``X`` of the estimator's
        ``fit``: ``(user, item, time)`` rows for a recommender constructed with
        ``time=True``, which then ranks each user as of their earliest held-out time, as
        :class:`~skrecsys.compose.Cascade` does when it trains its ranker. Further
        columns are the query context: each user is asked once, with the context of
        their first held-out interaction -- again as a Cascade trains its ranker. To
        score every request with its own context, call ``recommend`` per request.
    y : array-like of shape (n_interactions,), default=None
        Relevance of each pair; pairs with ``y <= 0`` are not relevant. ``None`` makes
        every pair relevant.
    metrics : callable, list of callables or dict of str to callable
        Metrics such as :func:`skrecsys.metrics.ndcg_at_k`. A list names each after its
        function without the ``_at_k`` suffix; a dict names them by its keys. A metric
        taking extra arguments is bound first, for instance
        ``functools.partial(novelty_at_k, item_popularity=item_popularity(X_train))``.
    k : int or sequence of int, default=10
        Cutoffs.
    candidates : array-like, default=None
        Shared candidate item set, passed to ``recommend``.
    exclude_seen : bool, default=True
        Passed to ``recommend``; removes training interactions from the ranking.
    exclude_interactions : array-like of shape (n_pairs, 2), default=None
        Passed to ``recommend``.

    Returns
    -------
    scores : dict of str to float
        One entry per metric and cutoff, keyed ``"<name>@<k>"``, for instance
        ``"ndcg@10"``, metrics in the given order and cutoffs ascending.

    Examples
    --------
    >>> from skrecsys.metrics import evaluate_recommender, hit_rate_at_k, ndcg_at_k
    >>> from skrecsys.recommendation import MostPopularRecommender
    >>> rec = MostPopularRecommender().fit([["u1", "a"], ["u2", "a"], ["u2", "b"]])
    >>> evaluate_recommender(rec, [["u1", "b"]], metrics=[ndcg_at_k, hit_rate_at_k], k=[1])
    {'ndcg@1': 1.0, 'hit_rate@1': 1.0}
    """
    if not is_recommender(estimator):
        raise TypeError(f"{type(estimator).__name__} is not a recommender.")
    named = _named_metrics(metrics)
    cutoffs = _cutoffs(k)
    timed = uses_time(estimator)
    queries, y_true, as_of = _held_out_by_user(X, y, time=timed)
    if timed:
        y_pred, _ = estimator.recommend(
            queries,
            n_recommendations=cutoffs[-1],
            candidates=candidates,
            exclude_seen=exclude_seen,
            exclude_interactions=exclude_interactions,
            as_of=as_of,
        )
    else:
        y_pred, _ = estimator.recommend(
            queries,
            n_recommendations=cutoffs[-1],
            candidates=candidates,
            exclude_seen=exclude_seen,
            exclude_interactions=exclude_interactions,
        )
    return {
        f"{name}@{cutoff}": _score(metric, y_true, y_pred, cutoff)
        for name, metric in named.items()
        for cutoff in cutoffs
    }


class _RecommenderScorer:
    """Callable ``scorer(estimator, X, y=None) -> float`` for one metric at one cutoff."""

    def __init__(
        self, metric: RankingMetric, k: int, candidates: ArrayLike | None, *, exclude_seen: bool
    ) -> None:
        self._metric = metric
        self._k = k
        self._candidates = candidates
        self._exclude_seen = exclude_seen

    def __call__(self, estimator: Recommender, X: ArrayLike, y: ArrayLike | None = None) -> float:
        scores = evaluate_recommender(
            estimator,
            X,
            y,
            metrics={"score": self._metric},
            k=self._k,
            candidates=self._candidates,
            exclude_seen=self._exclude_seen,
        )
        return scores[f"score@{self._k}"]

    @override
    def __repr__(self) -> str:
        name = _metric_name(self._metric)
        return f"make_recommender_scorer({name}, k={self._k}, exclude_seen={self._exclude_seen})"


class _MultiRecommenderScorer:
    """Callable ``scorer(estimator, X, y=None) -> dict``: scikit-learn's multi-metric form."""

    def __init__(
        self,
        metrics: Metrics,
        k: int | Sequence[int],
        candidates: ArrayLike | None,
        *,
        exclude_seen: bool,
    ) -> None:
        self._metrics = _named_metrics(metrics)
        self._k = _cutoffs(k)
        self._candidates = candidates
        self._exclude_seen = exclude_seen

    def __call__(
        self, estimator: Recommender, X: ArrayLike, y: ArrayLike | None = None
    ) -> dict[str, float]:
        return evaluate_recommender(
            estimator,
            X,
            y,
            metrics=self._metrics,
            k=self._k,
            candidates=self._candidates,
            exclude_seen=self._exclude_seen,
        )

    @override
    def __repr__(self) -> str:
        names = f"[{', '.join(self._metrics)}]"
        return f"make_recommender_scorer({names}, k={self._k}, exclude_seen={self._exclude_seen})"


@overload
def make_recommender_scorer(
    metric: RankingMetric,
    *,
    k: int = 10,
    candidates: ArrayLike | None = None,
    exclude_seen: bool = True,
    **kwargs: object,
) -> _RecommenderScorer: ...


@overload
def make_recommender_scorer(
    metric: Metrics,
    *,
    k: int | Sequence[int] = 10,
    candidates: ArrayLike | None = None,
    exclude_seen: bool = True,
    **kwargs: object,
) -> _MultiRecommenderScorer: ...


def make_recommender_scorer(
    metric: Metrics,
    *,
    k: int | Sequence[int] = 10,
    candidates: ArrayLike | None = None,
    exclude_seen: bool = True,
    **kwargs: object,
) -> _RecommenderScorer | _MultiRecommenderScorer:
    """Make a scorer from a top-k ranking metric, or from several at several cutoffs.

    The scorer groups held-out interactions ``(X, y)`` by user, calls
    ``estimator.recommend`` for those users and evaluates the ranked items against each
    user's held-out items. Interactions with ``y <= 0`` are not relevant; if ``y`` is
    None every held-out interaction is relevant.

    Parameters
    ----------
    metric : callable, list of callables or dict of str to callable
        A metric such as :func:`skrecsys.metrics.ndcg_at_k`. Beyond-accuracy metrics
        work too; pass their extra arguments, for instance
        ``make_recommender_scorer(novelty_at_k, item_popularity=item_popularity(X))``.
        Several metrics make a multi-metric scorer, named as in
        :func:`evaluate_recommender`.
    k : int or sequence of int, default=10
        Number of recommendations requested and the metric cutoff. Several cutoffs make
        a multi-metric scorer that ranks once, for the largest.
    candidates : array-like, default=None
        Shared candidate item set. ``None`` ranks the full fitted catalog. Sampled
        negative evaluation must pass an explicit candidate set.
    exclude_seen : bool, default=True
        Passed to ``recommend``; removes training interactions from the ranking.
    **kwargs
        Additional keyword arguments passed to ``metric``. With several metrics, bind
        each metric's arguments with ``functools.partial`` instead.

    Returns
    -------
    scorer : callable
        ``scorer(estimator, X, y=None)``, usable as ``scoring`` in
        :func:`sklearn.model_selection.cross_validate` and
        :class:`sklearn.model_selection.GridSearchCV`. It returns a float for one metric
        at one cutoff, otherwise a dict keyed like ``"ndcg@10"``: scikit-learn's
        multi-metric form, for which ``GridSearchCV`` needs ``refit="ndcg@10"`` or
        ``refit=False``.
    """
    if isinstance(metric, Sequence | Mapping):
        if kwargs:
            raise TypeError(
                "Keyword arguments apply to a single metric; bind them to each of several "
                "metrics with functools.partial."
            )
        return _MultiRecommenderScorer(metric, k, candidates, exclude_seen=exclude_seen)
    if kwargs:
        metric = functools.partial(metric, **kwargs)
    if not isinstance(k, int | np.integer):
        return _MultiRecommenderScorer(metric, k, candidates, exclude_seen=exclude_seen)
    return _RecommenderScorer(metric, _cutoffs(k)[0], candidates, exclude_seen=exclude_seen)
