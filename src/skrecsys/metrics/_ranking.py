"""Top-k ranking metrics with binary relevance.

All metrics share the signature
``metric(y_true, y_pred, *, k=None, average="macro", sample_weight=None)`` where
``y_true`` is a sequence of relevant item collections, one per query, and ``y_pred`` is
an array of shape (n_queries, n_ranked) of item identifiers ranked best first.
"""

from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "average_precision_at_k",
    "hit_rate_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
]


#: ``y_pred`` is laid out as (n_queries, n_ranked).
_RANKED_NDIM = 2


def _check_two_dimensional(y_pred: NDArray[Any]) -> None:
    if y_pred.ndim != _RANKED_NDIM:
        raise ValueError(f"y_pred must be two-dimensional, got shape {y_pred.shape}.")


def _check_ranking(
    y_true: Sequence[Collection[Any]], y_pred: ArrayLike, k: int | None
) -> tuple[NDArray[np.bool_], NDArray[np.intp], int]:
    """Return the (n_queries, k) hit matrix, relevant counts and effective k."""
    ranked = np.asarray(y_pred)
    _check_two_dimensional(ranked)
    if len(y_true) != ranked.shape[0]:
        raise ValueError(
            f"y_true and y_pred have inconsistent numbers of queries: "
            f"{len(y_true)} != {ranked.shape[0]}."
        )
    n_ranked = int(ranked.shape[1])
    cutoff = n_ranked if k is None else k
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, int | np.integer)
        or not 1 <= cutoff <= n_ranked
    ):
        raise ValueError(f"k must be an integer in [1, {n_ranked}], got {k!r}.")
    ranked = ranked[:, :cutoff]

    relevant_sets = [set(relevant) for relevant in y_true]
    n_relevant = np.array([len(relevant) for relevant in relevant_sets], dtype=np.intp)
    if np.any(n_relevant == 0):
        raise ValueError("Every query in y_true must have at least one relevant item.")
    hits = np.zeros(ranked.shape, dtype=bool)
    for q, (row, relevant) in enumerate(zip(ranked, relevant_sets, strict=True)):
        if len(set(row.tolist())) != len(row):
            raise ValueError(f"y_pred contains duplicate items for query {q}.")
        hits[q] = [item in relevant for item in row.tolist()]
    return hits, n_relevant, int(cutoff)


def _average(
    values: NDArray[np.float64], average: str | None, sample_weight: ArrayLike | None
) -> float | NDArray[np.float64]:
    if average is None:
        return values
    if average != "macro":
        raise ValueError(f"average must be None or 'macro', got {average!r}.")
    weights = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != values.shape:
            raise ValueError(f"sample_weight must have shape {values.shape}, got {weights.shape}.")
    return float(np.average(values, weights=weights))


def _discounts(k: int) -> NDArray[np.float64]:
    return 1.0 / np.log2(np.arange(2, k + 2))


def precision_at_k(
    y_true: Sequence[Collection[Any]],
    y_pred: ArrayLike,
    *,
    k: int | None = None,
    average: str | None = "macro",
    sample_weight: ArrayLike | None = None,
) -> float | NDArray[np.float64]:
    """Fraction of the top-k items that are relevant.

    Parameters
    ----------
    y_true : sequence of collections
        Relevant item identifiers for each query; each must be non-empty.
    y_pred : array-like of shape (n_queries, n_ranked)
        Ranked item identifiers, best first, without duplicates per row.
    k : int, default=None
        Cutoff. ``None`` uses ``n_ranked``.
    average : {"macro"} or None, default="macro"
        ``None`` returns per-query values; ``"macro"`` their (weighted) mean.
    sample_weight : array-like of shape (n_queries,), default=None
        Query weights for ``average="macro"``.

    Returns
    -------
    score : float or ndarray of shape (n_queries,)
    """
    hits, _, k = _check_ranking(y_true, y_pred, k)
    return _average(hits.sum(axis=1) / k, average, sample_weight)


def recall_at_k(
    y_true: Sequence[Collection[Any]],
    y_pred: ArrayLike,
    *,
    k: int | None = None,
    average: str | None = "macro",
    sample_weight: ArrayLike | None = None,
) -> float | NDArray[np.float64]:
    """Fraction of relevant items that appear in the top k.

    See :func:`precision_at_k` for parameters.
    """
    hits, n_relevant, _ = _check_ranking(y_true, y_pred, k)
    return _average(hits.sum(axis=1) / n_relevant, average, sample_weight)


def ndcg_at_k(
    y_true: Sequence[Collection[Any]],
    y_pred: ArrayLike,
    *,
    k: int | None = None,
    average: str | None = "macro",
    sample_weight: ArrayLike | None = None,
) -> float | NDArray[np.float64]:
    """Normalized discounted cumulative gain at k with binary relevance.

    ``DCG = sum_r hit_r / log2(r + 1)``, normalized by the DCG of an ideal ranking
    placing ``min(n_relevant, k)`` relevant items first.

    See :func:`precision_at_k` for parameters.
    """
    hits, n_relevant, k = _check_ranking(y_true, y_pred, k)
    discounts = _discounts(k)
    dcg = hits @ discounts
    ideal = np.cumsum(discounts)[np.minimum(n_relevant, k) - 1]
    return _average(dcg / ideal, average, sample_weight)


def average_precision_at_k(
    y_true: Sequence[Collection[Any]],
    y_pred: ArrayLike,
    *,
    k: int | None = None,
    average: str | None = "macro",
    sample_weight: ArrayLike | None = None,
) -> float | NDArray[np.float64]:
    """Average precision at k.

    ``AP@k = sum_r precision@r * hit_r / min(n_relevant, k)``; its macro average is
    MAP@k.

    See :func:`precision_at_k` for parameters.
    """
    hits, n_relevant, k = _check_ranking(y_true, y_pred, k)
    precision_at_rank = np.cumsum(hits, axis=1) / np.arange(1, k + 1)
    values = (precision_at_rank * hits).sum(axis=1) / np.minimum(n_relevant, k)
    return _average(values, average, sample_weight)


def reciprocal_rank_at_k(
    y_true: Sequence[Collection[Any]],
    y_pred: ArrayLike,
    *,
    k: int | None = None,
    average: str | None = "macro",
    sample_weight: ArrayLike | None = None,
) -> float | NDArray[np.float64]:
    """Reciprocal rank of the first relevant item within the top k, else 0.

    Its macro average is MRR@k. See :func:`precision_at_k` for parameters.
    """
    hits, _, _ = _check_ranking(y_true, y_pred, k)
    any_hit = hits.any(axis=1)
    first_rank = hits.argmax(axis=1) + 1
    values = np.where(any_hit, 1.0 / first_rank, 0.0)
    return _average(values, average, sample_weight)


def hit_rate_at_k(
    y_true: Sequence[Collection[Any]],
    y_pred: ArrayLike,
    *,
    k: int | None = None,
    average: str | None = "macro",
    sample_weight: ArrayLike | None = None,
) -> float | NDArray[np.float64]:
    """1 if any relevant item appears in the top k, else 0.

    See :func:`precision_at_k` for parameters.
    """
    hits, _, _ = _check_ranking(y_true, y_pred, k)
    return _average(hits.any(axis=1).astype(np.float64), average, sample_weight)
