"""Beyond-accuracy top-k metrics: coverage, popularity bias and novelty.

These metrics judge a ranking without ground truth: they describe *what* a model
recommends rather than how often it is right. They share the signature of the ranking
metrics in :mod:`skrecsys.metrics._ranking` so that any of them can be handed to
:func:`skrecsys.metrics.make_recommender_scorer`, but ``y_true`` is ignored.

``y_pred`` may be padded or ragged here: a query that received fewer than ``k`` items is
the subject of :func:`user_coverage_at_k`, so rows may be shorter than ``k`` or contain
``fill_value`` (``None`` and NaN are always treated as missing).
"""

import math
from collections.abc import Collection, Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from skrecsys.metrics._ranking import _average, _check_two_dimensional
from skrecsys.utils.validation import check_interactions

RankedItems: TypeAlias = NDArray[Any] | Sequence[Sequence[Any]]
"""Ranked items per query: a 2-D array, or rows that may be short or padded."""

__all__ = [
    "catalog_coverage_at_k",
    "item_popularity",
    "mean_popularity_at_k",
    "novelty_at_k",
    "user_coverage_at_k",
]


def _is_missing(value: Any, fill_value: Any) -> bool:
    if value is None:
        return True
    if fill_value is not None and value == fill_value:
        return True
    return isinstance(value, float | np.floating) and math.isnan(float(value))


def _ranked_rows(
    y_pred: RankedItems, k: int | None, fill_value: Any
) -> tuple[list[list[Any]], int]:
    """Return the top-k valid items of each query and the effective cutoff.

    Unlike the ranking metrics, rows may be short or padded, so the returned lists have
    varying lengths.
    """
    if isinstance(y_pred, np.ndarray):
        _check_two_dimensional(y_pred)
    # Rows are not stacked into one array: that would coerce a NaN pad next to string
    # identifiers into the string "nan", and ragged rows into an object array.
    rows = [list(row.tolist()) if isinstance(row, np.ndarray) else list(row) for row in y_pred]
    widest = max((len(row) for row in rows), default=0)
    cutoff = widest if k is None else k
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, int | np.integer)
        or cutoff < 1
        or (k is not None and cutoff > widest)
    ):
        raise ValueError(f"k must be an integer in [1, {widest}], got {k!r}.")
    cutoff = int(cutoff)

    trimmed = []
    for q, row in enumerate(rows):
        top = [item for item in row[:cutoff] if not _is_missing(item, fill_value)]
        if len(set(top)) != len(top):
            raise ValueError(f"y_pred contains duplicate items for query {q}.")
        trimmed.append(top)
    return trimmed, cutoff


def _popularity_vector(
    rows: Sequence[Sequence[Any]], popularity: Mapping[Any, float], default: float
) -> NDArray[np.float64]:
    """Mean popularity of each query's items; NaN for a query with no items."""
    values = np.full(len(rows), np.nan, dtype=np.float64)
    for q, row in enumerate(rows):
        if row:
            values[q] = float(np.mean([popularity.get(item, default) for item in row]))
    return values


def item_popularity(
    X: ArrayLike,
    y: ArrayLike | None = None,
    *,
    weighting: str = "count",
    normalize: bool = False,
) -> dict[Any, float]:
    """Popularity of every item in a training set of interactions.

    Parameters
    ----------
    X : array-like of shape (n_interactions, 2)
        Training interactions, ``(user, item)`` per row.
    y : array-like of shape (n_interactions,), default=None
        Interaction values, used by ``weighting="sum"``.
    weighting : {"count", "sum"}, default="count"
        ``"count"`` counts interactions per item; ``"sum"`` sums their values.
    normalize : bool, default=False
        Divide every value by the total, giving the empirical distribution ``p(i)``.

    Returns
    -------
    popularity : dict
        Mapping from item identifier to popularity. Items absent from ``X`` are absent
        from the mapping; pass a catalog to the metrics to give them a default.

    Examples
    --------
    >>> from skrecsys.metrics import item_popularity
    >>> item_popularity([["u1", "a"], ["u2", "a"], ["u2", "b"]])
    {'a': 2.0, 'b': 1.0}
    """
    _, items, weights = check_interactions(X, y)
    if weighting == "count":
        values = np.ones(len(items), dtype=np.float64)
    elif weighting == "sum":
        values = weights
    else:
        raise ValueError(f"weighting must be 'count' or 'sum', got {weighting!r}.")

    unique, codes = np.unique(items, return_inverse=True)
    totals = np.bincount(codes, weights=values, minlength=len(unique))
    if normalize:
        grand_total = totals.sum()
        if grand_total <= 0:
            raise ValueError("Cannot normalize popularity: the total weight is not positive.")
        totals = totals / grand_total
    return dict(zip(unique.tolist(), totals.tolist(), strict=True))


def catalog_coverage_at_k(
    y_true: Sequence[Collection[Any]] | None,
    y_pred: RankedItems,
    *,
    k: int | None = None,
    catalog: ArrayLike | None = None,
    n_catalog_items: int | None = None,
    fill_value: Any = None,
) -> float:
    """Fraction of the catalog that appears in at least one top-k list.

    A low value flags a model that funnels every user towards the same few items.

    Parameters
    ----------
    y_true : ignored
        Present so that every top-k metric shares one signature.
    y_pred : array-like of shape (n_queries, n_ranked)
        Ranked item identifiers, best first. Rows may be padded with ``fill_value``.
    k : int, default=None
        Cutoff. ``None`` uses the width of ``y_pred``.
    catalog : array-like, default=None
        Identifiers of every item that could have been recommended. Recommending an
        item outside it is an error.
    n_catalog_items : int, default=None
        Catalog size, when the identifiers themselves are not at hand. Exactly one of
        ``catalog`` and ``n_catalog_items`` must be given.
    fill_value : object, default=None
        Padding marker in ``y_pred``; ``None`` and NaN always count as padding.

    Returns
    -------
    coverage : float
        In ``[0, 1]``. This metric is defined over the whole query set, so unlike the
        ranking metrics it has no per-query form.
    """
    del y_true
    if (catalog is None) == (n_catalog_items is None):
        raise ValueError("Give exactly one of catalog and n_catalog_items.")
    rows, _ = _ranked_rows(y_pred, k, fill_value)
    recommended = {item for row in rows for item in row}

    if catalog is not None:
        known = set(np.asarray(catalog).ravel().tolist())
        size = len(known)
        unknown = recommended - known
        if unknown:
            raise ValueError(f"y_pred contains items outside the catalog: {sorted(unknown)[:5]}.")
    else:
        size = int(n_catalog_items)  # ty: ignore[invalid-argument-type]
        if size < 1:
            raise ValueError(f"n_catalog_items must be >= 1, got {n_catalog_items}.")
        if len(recommended) > size:
            raise ValueError(
                f"y_pred contains {len(recommended)} distinct items, more than the "
                f"catalog size {size}."
            )
    return len(recommended) / size


def user_coverage_at_k(
    y_true: Sequence[Collection[Any]] | None,
    y_pred: RankedItems,
    *,
    k: int | None = None,
    average: str | None = "macro",
    sample_weight: ArrayLike | None = None,
    fill_value: Any = None,
) -> float | NDArray[np.float64]:
    """1 if the query received ``k`` valid recommendations, else 0.

    An operational metric: models that cannot serve cold or sparse users return short
    or padded lists for them. With lists that are always full it is 1 by construction.

    Parameters
    ----------
    y_true : ignored
        Present so that every top-k metric shares one signature.
    y_pred : array-like of shape (n_queries, n_ranked)
        Ranked item identifiers, best first. Rows may be short or padded.
    k : int, default=None
        Number of recommendations a query must receive. ``None`` uses the width of
        ``y_pred``.
    average : {"macro"} or None, default="macro"
        ``None`` returns per-query values; ``"macro"`` their (weighted) mean.
    sample_weight : array-like of shape (n_queries,), default=None
        Query weights for ``average="macro"``.
    fill_value : object, default=None
        Padding marker in ``y_pred``; ``None`` and NaN always count as padding.

    Returns
    -------
    score : float or ndarray of shape (n_queries,)
    """
    del y_true
    rows, cutoff = _ranked_rows(y_pred, k, fill_value)
    values = np.array([float(len(row) >= cutoff) for row in rows], dtype=np.float64)
    return _average(values, average, sample_weight)


def mean_popularity_at_k(
    y_true: Sequence[Collection[Any]] | None,
    y_pred: RankedItems,
    *,
    k: int | None = None,
    item_popularity: Mapping[Any, float],
    average: str | None = "macro",
    sample_weight: ArrayLike | None = None,
    default: float = 0.0,
    fill_value: Any = None,
) -> float | NDArray[np.float64]:
    """Mean training popularity of the recommended items.

    Measures popularity bias, on the scale of the values passed in: counts stay counts,
    and a normalized mapping gives the mean share of training interactions.

    Parameters
    ----------
    y_true : ignored
        Present so that every top-k metric shares one signature.
    y_pred : array-like of shape (n_queries, n_ranked)
        Ranked item identifiers, best first. Rows may be padded with ``fill_value``.
    k : int, default=None
        Cutoff. ``None`` uses the width of ``y_pred``.
    item_popularity : mapping
        Item identifier to training popularity, as built by :func:`item_popularity`.
    average : {"macro"} or None, default="macro"
        ``None`` returns per-query values; ``"macro"`` their (weighted) mean.
    sample_weight : array-like of shape (n_queries,), default=None
        Query weights for ``average="macro"``.
    default : float, default=0.0
        Popularity of items missing from ``item_popularity``.
    fill_value : object, default=None
        Padding marker in ``y_pred``; ``None`` and NaN always count as padding.

    Returns
    -------
    score : float or ndarray of shape (n_queries,)
        A query with no valid recommendation contributes NaN.
    """
    del y_true
    rows, _ = _ranked_rows(y_pred, k, fill_value)
    return _average(_popularity_vector(rows, item_popularity, default), average, sample_weight)


def novelty_at_k(
    y_true: Sequence[Collection[Any]] | None,
    y_pred: RankedItems,
    *,
    k: int | None = None,
    item_popularity: Mapping[Any, float],
    average: str | None = "macro",
    sample_weight: ArrayLike | None = None,
    smoothing: float = 1.0,
    fill_value: Any = None,
) -> float | NDArray[np.float64]:
    """Mean self-information ``-log2 p(i)`` of the recommended items.

    High novelty means the top-k reaches past the obvious head of the catalog. The
    distribution ``p`` is ``item_popularity`` renormalized over its own support, so the
    mapping should cover the whole catalog, including items nobody interacted with.

    Parameters
    ----------
    y_true : ignored
        Present so that every top-k metric shares one signature.
    y_pred : array-like of shape (n_queries, n_ranked)
        Ranked item identifiers, best first. Rows may be padded with ``fill_value``.
    k : int, default=None
        Cutoff. ``None`` uses the width of ``y_pred``.
    item_popularity : mapping
        Item identifier to training popularity or count, as built by
        :func:`item_popularity`.
    average : {"macro"} or None, default="macro"
        ``None`` returns per-query values; ``"macro"`` their (weighted) mean.
    sample_weight : array-like of shape (n_queries,), default=None
        Query weights for ``average="macro"``.
    smoothing : float, default=1.0
        Added to every popularity before normalizing, which keeps ``-log2 p(i)`` finite
        for items that were never interacted with. ``0.0`` gives the textbook
        definition and rejects such items.
    fill_value : object, default=None
        Padding marker in ``y_pred``; ``None`` and NaN always count as padding.

    Returns
    -------
    score : float or ndarray of shape (n_queries,)
        In bits. A query with no valid recommendation contributes NaN.
    """
    del y_true
    if smoothing < 0:
        raise ValueError(f"smoothing must be >= 0, got {smoothing}.")
    if not item_popularity:
        raise ValueError("item_popularity must not be empty.")
    counts = np.asarray(list(item_popularity.values()), dtype=np.float64)
    if np.any(counts < 0):
        raise ValueError("item_popularity must not contain negative values.")
    total = float(counts.sum()) + smoothing * len(item_popularity)
    if total <= 0:
        raise ValueError("item_popularity is all zeros; pass smoothing > 0.")

    rows, _ = _ranked_rows(y_pred, k, fill_value)
    information: dict[Any, float] = {}
    for row in rows:
        for item in row:
            if item in information:
                continue
            probability = (float(item_popularity.get(item, 0.0)) + smoothing) / total
            if probability <= 0:
                raise ValueError(
                    f"Item {item!r} has zero popularity, so its novelty is infinite; "
                    "pass smoothing > 0 or drop it from the recommendations."
                )
            information[item] = -math.log2(probability)
    return _average(_popularity_vector(rows, information, 0.0), average, sample_weight)
