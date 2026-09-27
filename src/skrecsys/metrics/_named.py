"""Ranking metrics named by configuration, ``NDCG(10)``, or by string, ``"ndcg@10"``."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from numpy.typing import ArrayLike

from skrecsys._typing import RankingMetric, Recommender, override
from skrecsys.metrics._ranking import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from skrecsys.metrics._scorer import make_recommender_scorer
from skrecsys.utils._param_validation import check_int

__all__ = ["MAP", "MRR", "NDCG", "HitRate", "Precision", "Recall", "get_scorer"]

#: What :func:`get_scorer` returns: ``scorer(estimator, X, y) -> float``, higher is better.
Scorer = Callable[[Recommender, ArrayLike, ArrayLike | None], float]


@dataclass(frozen=True)
class _TopKMetric:
    """A top-k ranking metric at one cutoff, callable as a scorer.

    Calling it is calling ``make_recommender_scorer(metric, k=k)``, so it works as
    ``scoring`` wherever such a scorer does, scikit-learn's model selection included.
    """

    k: int = 10

    #: The name :func:`~skrecsys.metrics.evaluate_recommender` gives the metric.
    name: ClassVar[str]
    _metric: ClassVar[RankingMetric]

    def __init_subclass__(cls, *, name: str, metric: RankingMetric) -> None:
        super().__init_subclass__()
        cls.name = name
        cls._metric = metric

    def __post_init__(self) -> None:
        check_int(self.k, "k", min_value=1)

    def __call__(self, estimator: Recommender, X: ArrayLike, y: ArrayLike | None = None) -> float:
        return make_recommender_scorer(type(self)._metric, k=self.k)(estimator, X, y)

    @override
    def __str__(self) -> str:
        return f"{self.name}@{self.k}"


@dataclass(frozen=True)
class NDCG(_TopKMetric, name="ndcg", metric=ndcg_at_k):
    """:func:`~skrecsys.metrics.ndcg_at_k` at cutoff ``k``, as a scorer."""


@dataclass(frozen=True)
class Recall(_TopKMetric, name="recall", metric=recall_at_k):
    """:func:`~skrecsys.metrics.recall_at_k` at cutoff ``k``, as a scorer."""


@dataclass(frozen=True)
class Precision(_TopKMetric, name="precision", metric=precision_at_k):
    """:func:`~skrecsys.metrics.precision_at_k` at cutoff ``k``, as a scorer."""


@dataclass(frozen=True)
class MAP(_TopKMetric, name="average_precision", metric=average_precision_at_k):
    """:func:`~skrecsys.metrics.average_precision_at_k` at cutoff ``k``, as a scorer."""


@dataclass(frozen=True)
class MRR(_TopKMetric, name="reciprocal_rank", metric=reciprocal_rank_at_k):
    """:func:`~skrecsys.metrics.reciprocal_rank_at_k` at cutoff ``k``, as a scorer."""


@dataclass(frozen=True)
class HitRate(_TopKMetric, name="hit_rate", metric=hit_rate_at_k):
    """:func:`~skrecsys.metrics.hit_rate_at_k` at cutoff ``k``, as a scorer."""


_BY_NAME: dict[str, type[_TopKMetric]] = {
    **{cls.name: cls for cls in (NDCG, Recall, Precision, MAP, MRR, HitRate)},
    "map": MAP,
    "mrr": MRR,
    "hr": HitRate,
}

_SPEC = re.compile(r"(?P<name>[a-z_]+)(?:@(?P<k>\S+))?")


def get_scorer(scoring: str | Scorer | None) -> Scorer:
    """Resolve a metric name, a metric such as ``NDCG(10)`` or a scorer into a scorer.

    Parameters
    ----------
    scoring : str, callable or None
        ``None`` is ``NDCG(10)``. A string ``"<name>@<k>"``, such as ``"recall@20"``,
        names a metric and its cutoff; ``"<name>"`` alone means ``k=10``. The names are
        those :func:`~skrecsys.metrics.evaluate_recommender` gives (``ndcg``, ``recall``,
        ``precision``, ``average_precision``, ``reciprocal_rank``, ``hit_rate``) and the
        aliases ``map``, ``mrr`` and ``hr``, in any case. A callable, such as a metric
        or what :func:`~skrecsys.metrics.make_recommender_scorer` returns, is returned
        as it is.

    Returns
    -------
    scorer : callable
        ``scorer(estimator, X, y) -> float``, higher being better.

    Examples
    --------
    >>> from skrecsys.metrics import NDCG, get_scorer
    >>> get_scorer("recall@20")
    Recall(k=20)
    >>> get_scorer("MAP"), get_scorer(None) == NDCG(10)
    (MAP(k=10), True)
    """
    if scoring is None:
        return NDCG(10)
    if isinstance(scoring, str):
        match = _SPEC.fullmatch(scoring.strip().lower())
        cls = _BY_NAME.get(match["name"]) if match else None
        if match is None or cls is None:
            raise ValueError(
                f"Unknown metric {scoring!r}; expected '<name>' or '<name>@<k>' with a name "
                f"among {sorted(_BY_NAME)}."
            )
        if match["k"] is None:
            return cls()
        if not match["k"].isdigit():
            raise ValueError(f"k must be an integer >= 1, got {match['k']!r} in {scoring!r}.")
        return cls(int(match["k"]))
    if callable(scoring):
        return scoring
    raise TypeError(
        f"scoring must be None, a metric name, a metric such as NDCG(10) or a scorer, "
        f"got {type(scoring).__name__}."
    )
