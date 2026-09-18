"""Input validation for user-item interaction data."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import check_array, check_consistent_length

from skrecsys import _core

__all__ = ["check_ids", "check_interactions", "encode_ids", "factorize"]

_N_COLUMNS = 2
_INT64_BYTES = 8


def check_interactions(
    X: ArrayLike, y: ArrayLike | None = None
) -> tuple[NDArray[Any], NDArray[Any], NDArray[np.float64]]:
    """Validate user-item interactions.

    Parameters
    ----------
    X : array-like of shape (n_interactions, 2)
        ``X[:, 0]`` contains user identifiers, ``X[:, 1]`` item identifiers.

    y : array-like of shape (n_interactions,), default=None
        Interaction values. ``None`` gives every interaction weight 1.

    Returns
    -------
    users : ndarray of shape (n_interactions,)
    items : ndarray of shape (n_interactions,)
    y : ndarray of float64 of shape (n_interactions,)
    """
    X_arr = check_array(X, dtype=None, ensure_all_finite=False)
    if X_arr.shape[1] != _N_COLUMNS:
        raise ValueError(
            "X must have exactly 2 columns (user identifiers, item identifiers), "
            f"got {X_arr.shape[1]}."
        )
    users, items = X_arr[:, 0], X_arr[:, 1]
    for name, ids in (("user", users), ("item", items)):
        if ids.dtype.kind in "fc" and not np.all(np.isfinite(ids)):
            raise ValueError(f"X contains non-finite {name} identifiers.")

    if y is None:
        weights = np.ones(len(X_arr), dtype=np.float64)
    else:
        weights = check_array(y, ensure_2d=False, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError(f"y must be one-dimensional, got shape {weights.shape}.")
        check_consistent_length(X_arr, weights)
    return users, items, weights


def check_ids(ids: ArrayLike, *, name: str = "X") -> NDArray[Any]:
    """Validate a one-dimensional array of identifiers."""
    arr = check_array(ids, ensure_2d=False, dtype=None, ensure_all_finite=False)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array of identifiers.")
    return arr


def factorize(values: NDArray[Any]) -> tuple[NDArray[Any], NDArray[np.intp]]:
    """Sorted distinct identifiers, and the position of each value among them.

    What ``np.unique(values, return_inverse=True)`` returns, but that argsorts every
    interaction. Identifiers repeat heavily, so the distinct ones are found first and
    only those are sorted: a native pass for integers, a dictionary for the Python
    objects that ``np.unique`` would have to sort with ``<`` one pair at a time.
    """
    kind = values.dtype.kind
    if kind == "i" or (kind == "u" and values.dtype.itemsize < _INT64_BYTES):
        uniques, codes = _core.factorize(np.ascontiguousarray(values, dtype=np.int64))
        return uniques.astype(values.dtype, copy=False), codes.astype(np.intp, copy=False)
    if kind == "O":
        try:
            return _factorize_objects(values)
        except TypeError:
            # Unhashable or mutually incomparable identifiers; numpy raises its own way.
            pass
    return np.unique(values, return_inverse=True)


def _factorize_objects(values: NDArray[Any]) -> tuple[NDArray[Any], NDArray[np.intp]]:
    """Factorize an object array by hashing, then sort only the distinct values."""
    seen: dict[Any, int] = {}
    for value in values:
        if value not in seen:
            seen[value] = len(seen)
    order = sorted(seen)
    rank = np.empty(len(order), dtype=np.intp)
    for position, value in enumerate(order):
        rank[seen[value]] = position
    codes = np.fromiter((seen[value] for value in values), dtype=np.intp, count=len(values))
    uniques = np.empty(len(order), dtype=object)
    uniques[:] = order
    return uniques, rank[codes]


def encode_ids(ids: NDArray[Any], fitted_ids: NDArray[Any], *, name: str) -> NDArray[np.intp]:
    """Map identifiers to positions in the sorted array ``fitted_ids``.

    Raises
    ------
    ValueError
        If any identifier was not seen during ``fit``.
    """
    try:
        positions = np.searchsorted(fitted_ids, ids)
    except TypeError as exc:
        raise ValueError(f"Unknown {name} identifiers: incompatible types.") from exc
    positions = np.minimum(positions, len(fitted_ids) - 1)
    unknown = fitted_ids[positions] != ids if len(fitted_ids) else np.ones(len(ids), bool)
    if np.any(unknown):
        sample = list(np.asarray(ids)[unknown][:5])
        raise ValueError(f"Unknown {name} identifiers: {sample}.")
    return positions.astype(np.intp)
