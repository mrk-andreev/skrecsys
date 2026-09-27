"""Input validation for user-item interaction data.

Interactions are ``[user, item]`` rows, or ``[user, item, time]`` rows for a recommender
constructed with ``time=True``. A time is a number (such as Unix seconds) or a
``datetime64``; a column of ``datetime`` or ``pandas.Timestamp`` objects, which is what a
DataFrame's datetime column becomes, is read as ``datetime64[ns]``.
"""

import datetime
import numbers
from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import check_array, check_consistent_length

from skrecsys import _core
from skrecsys._typing import SortableId

__all__ = [
    "check_as_of",
    "check_ids",
    "check_interactions",
    "check_optionally_timed",
    "check_times",
    "drop_time",
    "encode_ids",
    "factorize",
    "lookup_ids",
    "stack_pairs",
]

_N_COLUMNS = 2
_N_TIMED_COLUMNS = 3
_INT64_BYTES = 8

#: The unit an object column of datetimes is read in: what pandas stores.
_DATETIME = np.dtype("datetime64[ns]")


@overload
def check_interactions(
    X: ArrayLike, y: ArrayLike | None = None, *, time: Literal[False] = False
) -> tuple[NDArray[np.generic], NDArray[np.generic], NDArray[np.float64]]: ...


@overload
def check_interactions(
    X: ArrayLike, y: ArrayLike | None = None, *, time: Literal[True]
) -> tuple[NDArray[np.generic], NDArray[np.generic], NDArray[np.float64], NDArray[np.generic]]: ...


def check_interactions(
    X: ArrayLike, y: ArrayLike | None = None, *, time: bool = False
) -> (
    tuple[NDArray[np.generic], NDArray[np.generic], NDArray[np.float64]]
    | tuple[NDArray[np.generic], NDArray[np.generic], NDArray[np.float64], NDArray[np.generic]]
):
    """Validate user-item interactions.

    Parameters
    ----------
    X : array-like of shape (n_interactions, 2) or (n_interactions, 3)
        ``X[:, 0]`` contains user identifiers, ``X[:, 1]`` item identifiers and, when
        ``time`` is true, ``X[:, 2]`` the time of each interaction.

    y : array-like of shape (n_interactions,), default=None
        Interaction values. ``None`` gives every interaction weight 1.

    time : bool, default=False
        Whether ``X`` carries a time column. Only a recommender constructed with
        ``time=True`` reads one; everywhere else a third column is an error, so that a
        stray column -- a rating, say -- is never mistaken for time.

    Returns
    -------
    users : ndarray of shape (n_interactions,)
    items : ndarray of shape (n_interactions,)
    y : ndarray of float64 of shape (n_interactions,)
    times : ndarray of shape (n_interactions,)
        Only when ``time`` is true: numbers or ``datetime64``, as :func:`check_times`
        returns them.
    """
    X_arr = check_array(X, dtype=None, ensure_all_finite=False)
    if time:
        if X_arr.shape[1] != _N_TIMED_COLUMNS:
            raise ValueError(
                "X must have exactly 3 columns (user identifiers, item identifiers, times), "
                f"got {X_arr.shape[1]}."
            )
    elif X_arr.shape[1] != _N_COLUMNS:
        hint = (
            " A time column is read only by a recommender constructed with time=True."
            if X_arr.shape[1] == _N_TIMED_COLUMNS
            else ""
        )
        raise ValueError(
            "X must have exactly 2 columns (user identifiers, item identifiers), "
            f"got {X_arr.shape[1]}.{hint}"
        )
    ids = drop_time(X_arr)
    users, items = ids[:, 0], ids[:, 1]
    for name, column in (("user", users), ("item", items)):
        if column.dtype.kind in "fc" and not np.all(np.isfinite(column)):
            raise ValueError(f"X contains non-finite {name} identifiers.")

    if y is None:
        weights = np.ones(len(X_arr), dtype=np.float64)
    else:
        weights = check_array(y, ensure_2d=False, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError(f"y must be one-dimensional, got shape {weights.shape}.")
        check_consistent_length(X_arr, weights)
    if time:
        return users, items, weights, check_times(X_arr[:, 2], name="X[:, 2]")
    return users, items, weights


def check_optionally_timed(
    X: ArrayLike, y: ArrayLike | None = None
) -> tuple[
    NDArray[np.generic], NDArray[np.generic], NDArray[np.float64], NDArray[np.generic] | None
]:
    """:func:`check_interactions` for code that serves recommenders with and without time.

    A splitter or a scorer does not choose whether time is used; the recommender it is
    handed does. So a third column is read as time here, and ``times`` is ``None`` for
    two-column interactions.
    """
    X_arr = check_array(X, dtype=None, ensure_all_finite=False)
    if X_arr.shape[1] == _N_TIMED_COLUMNS:
        return check_interactions(X_arr, y, time=True)
    users, items, weights = check_interactions(X_arr, y)
    return users, items, weights, None


def drop_time(X: NDArray[np.generic]) -> NDArray[np.generic]:
    """The identifier columns of validated interactions or pairs, without their time.

    A time column next to identifiers of another type makes the whole array ``object``;
    identifiers that are all numbers are given back their numeric dtype, so that what a
    recommender is fitted on does not depend on whether its caller carried time.
    """
    ids = X[:, :_N_COLUMNS]
    if ids.dtype.kind != "O" or X.shape[1] == _N_COLUMNS or not ids.size:
        return ids
    if all(_is_number(v) for v in ids.ravel().tolist()):
        return np.array(ids.tolist())
    return ids


def check_times(
    values: ArrayLike, *, name: str = "times", allow_missing: bool = False
) -> NDArray[np.generic]:
    """Validate a one-dimensional array of times.

    Parameters
    ----------
    values : array-like of shape (n,)
        Numbers, ``datetime64``, or objects that are ``datetime`` (including
        ``pandas.Timestamp``), ``date``, ``numpy.datetime64`` or numbers.
    name : str, default="times"
        How an error names ``values``.
    allow_missing : bool, default=False
        Whether NaN, NaT and ``None`` are accepted. A missing time in a query or a
        candidate pair means "latest"; an interaction must have happened at some time.

    Returns
    -------
    times : ndarray of shape (n,)
        ``int64``, ``float64`` (when a number is missing or fractional) or
        ``datetime64``; an object column of datetimes becomes ``datetime64[ns]``.
    """
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}.")
    if arr.dtype.kind == "O":
        arr = _times_from_objects(arr, name)
    kind = arr.dtype.kind
    if kind in "iu":
        return arr.astype(np.int64, copy=False)
    if kind == "f":
        arr = arr.astype(np.float64, copy=False)
        if np.isinf(arr).any():
            raise ValueError(f"{name} contains infinite times.")
        missing = np.isnan(arr)
    elif kind == "M":
        missing = np.isnat(arr)
    elif kind in "US":
        raise TypeError(
            f"{name} must hold numbers or datetimes, got strings. A list mixing string "
            "identifiers with numeric times becomes all strings in numpy; pass an object "
            "array or a DataFrame instead."
        )
    else:
        raise TypeError(f"{name} must hold numbers or datetimes, got dtype {arr.dtype}.")
    if not allow_missing and missing.any():
        raise ValueError(f"{name} contains missing times.")
    return arr


def check_as_of(
    as_of: ArrayLike | None, n_queries: int, time_dtype: np.dtype[np.generic]
) -> NDArray[np.generic]:
    """The time each of ``n_queries`` queries is answered as of.

    ``None`` answers every query as of the latest data -- a missing time -- and a scalar
    answers all of them as of the same moment. Otherwise there must be one time per
    query, missing where a query wants the latest data. Times must be of the kind the
    recommender was fitted on: numbers against numbers, datetimes against datetimes.
    """
    datetime_fitted = time_dtype.kind == "M"
    if as_of is None:
        if datetime_fitted:
            return np.full(n_queries, "NaT", dtype=time_dtype)
        return np.full(n_queries, np.nan)
    if np.ndim(as_of) == 0:
        as_of = [as_of] * n_queries
    times = check_times(as_of, name="as_of", allow_missing=True)
    if len(times) != n_queries:
        raise ValueError(
            f"as_of must be a single time or one per query ({n_queries}), got {len(times)}."
        )
    if (times.dtype.kind == "M") != datetime_fitted:
        fitted = "datetimes" if datetime_fitted else "numbers"
        raise TypeError(
            f"as_of must hold {fitted}, as the fitted times do; got dtype {times.dtype}."
        )
    return times.astype(time_dtype) if datetime_fitted else times


def _is_number(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool | np.bool_)


def _is_missing(value: object) -> bool:
    # NaN, NaT and pandas.NaT are the values not equal to themselves.
    return value is None or value != value  # noqa: PLR0124


def _times_from_objects(values: NDArray[np.object_], name: str) -> NDArray[np.generic]:
    """Numbers or ``datetime64[ns]`` from an object column; ``None`` becomes missing."""
    items = values.tolist()
    present = [v for v in items if not _is_missing(v)]
    if all(isinstance(v, datetime.date | np.datetime64) for v in present) and present:
        return values.astype(_DATETIME)
    if all(_is_number(v) for v in present):
        if len(present) == len(items) and all(isinstance(v, numbers.Integral) for v in items):
            return np.array(items, dtype=np.int64)
        return np.array([np.nan if _is_missing(v) else v for v in items], dtype=np.float64)
    raise TypeError(
        f"{name} must hold numbers or datetimes, not a mix of them or of other objects."
    )


#: Dtype kinds an identifier array can already be in that ``check_array`` with
#: ``dtype=None`` hands back unchanged: integers, floats, strings and objects.
_PASSTHROUGH_KINDS = frozenset("iufUO")


def check_ids(ids: ArrayLike, *, name: str = "X") -> NDArray[np.generic]:
    """Validate a one-dimensional array of identifiers."""
    # The common case in a serving loop -- a non-empty 1-D ndarray -- is returned as is.
    # ``check_array`` would return the same array, but only after probing it for every
    # dataframe library it knows, which was most of a one-user ``recommend``.
    if (
        type(ids) is np.ndarray
        and ids.ndim == 1
        and ids.size
        and ids.dtype.kind in _PASSTHROUGH_KINDS
    ):
        return ids
    arr = check_array(ids, ensure_2d=False, dtype=None, ensure_all_finite=False)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array of identifiers.")
    return arr


def factorize(values: NDArray[np.generic]) -> tuple[NDArray[np.generic], NDArray[np.intp]]:
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


def _factorize_objects(values: NDArray[np.generic]) -> tuple[NDArray[np.generic], NDArray[np.intp]]:
    """Factorize an object array by hashing, then sort only the distinct values."""
    seen: dict[SortableId, int] = {}
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


def stack_pairs(users: NDArray[np.generic], items: NDArray[np.generic]) -> NDArray[np.generic]:
    """Column-stack identifiers without letting numpy coerce one namespace into the other.

    ``np.column_stack`` of integer users and string items would silently turn the users
    into strings, which no longer match the fitted ones; an object array keeps both.
    """
    if users.dtype == items.dtype:
        return np.column_stack([users, items])
    pairs = np.empty((len(users), 2), dtype=object)
    pairs[:, 0] = users
    pairs[:, 1] = items
    return pairs


def lookup_ids(
    ids: NDArray[np.generic], fitted_ids: NDArray[np.generic], *, name: str
) -> tuple[NDArray[np.intp], NDArray[np.bool_]]:
    """Positions of identifiers in the sorted array ``fitted_ids``, and which exist.

    ``positions[i]`` is meaningful only where ``known[i]`` is true. The lenient form of
    :func:`encode_ids`, for inputs that may legitimately name identifiers the model has
    never seen and are expected to skip them.

    Raises
    ------
    ValueError
        If the identifiers cannot be ordered against the fitted ones at all.
    """
    try:
        positions = np.searchsorted(fitted_ids, ids)
    except TypeError as exc:
        raise ValueError(f"Unknown {name} identifiers: incompatible types.") from exc
    if not len(fitted_ids):
        return np.zeros(len(ids), dtype=np.intp), np.zeros(len(ids), dtype=bool)
    positions = np.minimum(positions, len(fitted_ids) - 1)
    known = np.asarray(fitted_ids[positions] == ids, dtype=bool)
    return positions.astype(np.intp), known


def encode_ids(
    ids: NDArray[np.generic], fitted_ids: NDArray[np.generic], *, name: str
) -> NDArray[np.intp]:
    """Map identifiers to positions in the sorted array ``fitted_ids``.

    Raises
    ------
    ValueError
        If any identifier was not seen during ``fit``.
    """
    positions, known = lookup_ids(ids, fitted_ids, name=name)
    if not np.all(known):
        sample = list(np.asarray(ids)[~known][:5])
        raise ValueError(f"Unknown {name} identifiers: {sample}.")
    return positions
