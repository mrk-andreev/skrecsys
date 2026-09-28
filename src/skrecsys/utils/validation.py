"""Input validation for user-item interaction data.

Interactions are ``[user, item]`` rows, or ``[user, item, time]`` rows for a recommender
constructed with ``time=True``. A time is a number (such as Unix seconds) or a
``datetime64``; a column of ``datetime`` or ``pandas.Timestamp`` objects, which is what a
DataFrame's datetime column becomes, is read as ``datetime64[ns]``.

Any columns after these system columns are the *query context* of each interaction:
request-time attributes such as the page or the device. Every recommender accepts them,
and one that cannot use them ignores them. The queries of ``recommend`` are a vector of
user identifiers, or a matrix whose first column is the user and whose other columns are
the context of each query (see :func:`check_queries`).
"""

import datetime
import math
import numbers
from typing import Literal, TypeVar, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import check_array, check_consistent_length

from skrecsys import _core
from skrecsys._typing import DataFrameLike, SortableId

__all__ = [
    "check_as_of",
    "check_ids",
    "check_interactions",
    "check_optionally_timed",
    "check_queries",
    "check_rows",
    "check_times",
    "drop_time",
    "encode_ids",
    "factorize",
    "interaction_context",
    "lookup_ids",
    "stack_columns",
    "stack_pairs",
]

_N_COLUMNS = 2
_N_TIMED_COLUMNS = 3
_INT64_BYTES = 8
#: A table of rows: ``X``, or a list of query rows.
_MATRIX_NDIM = 2

#: A value of an input column, of whatever type it holds.
_T = TypeVar("_T")

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
    X : array-like of shape (n_interactions, n_system + n_context)
        ``X[:, 0]`` contains user identifiers, ``X[:, 1]`` item identifiers and, when
        ``time`` is true, ``X[:, 2]`` the time of each interaction. Any further columns
        are the query context of each interaction; they are not validated here, and
        :func:`interaction_context` returns them.

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
    X_arr = check_rows(X)
    if time:
        if X_arr.shape[1] < _N_TIMED_COLUMNS:
            raise ValueError(
                "X must have at least 3 columns (user identifiers, item identifiers, times), "
                f"got {X_arr.shape[1]}."
            )
    elif X_arr.shape[1] < _N_COLUMNS:
        raise ValueError(
            "X must have at least 2 columns (user identifiers, item identifiers), "
            f"got {X_arr.shape[1]}."
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


def check_rows(X: ArrayLike, *, ensure_min_samples: int = 1) -> NDArray[np.generic]:
    """``X`` as a two-dimensional array, each column keeping the kind of value it holds.

    numpy gives a table one dtype, so a time or context column would recast the
    identifiers next to it: ``[[1, 2, "web"]]`` becomes strings, where user 1 would be
    ``"1"``, and ``[[1, 2, 0.5]]`` floats, where identifiers past 2**53 collide. Rows of
    more than two columns whose identifiers are of another type than the rest are made
    an ``object`` array instead, from which :func:`drop_time`,
    :func:`interaction_context` and :func:`check_queries` give each column back its own
    numeric dtype. Two-column rows are left to numpy, as they always were.

    Raises
    ------
    ValueError
        If ``X`` is a float array whose identifiers are integers too large for a float
        to have kept exactly.
    """
    arr = check_array(
        _typed_rows(X, _N_COLUMNS),
        dtype=None,
        ensure_all_finite=False,
        ensure_min_samples=ensure_min_samples,
    )
    return _integer_ids_restored(arr, _N_COLUMNS)


#: The magnitude from which a float no longer holds every integer exactly.
_FLOAT_EXACT_INTEGERS = 2**53


def _typed_rows(X: ArrayLike, n_ids: int) -> ArrayLike:
    """``X`` as an ``object`` array when numpy would recast its first ``n_ids`` columns.

    That is a list of rows mixing numbers and strings, a list of rows mixing integer
    identifiers with floats, or a DataFrame whose integer identifier columns sit next to
    columns of another dtype. Anything else is returned as it is.
    """
    objects = _frame_as_objects(X, n_ids)
    if objects is not None:
        return objects
    if isinstance(X, list | tuple) and X:
        objects = _list_as_objects(X, n_ids)
        if objects is not None:
            return objects
    return X


def _frame_as_objects(X: ArrayLike, n_ids: int) -> NDArray[np.generic] | None:
    """A DataFrame whose integer identifiers numpy would recast, as objects; else None."""
    if not isinstance(X, DataFrameLike) or X.ndim != _MATRIX_NDIM:
        return None
    dtypes = list(X.dtypes)
    if len(dtypes) <= n_ids or len(set(dtypes)) == 1:
        return None
    if not all(isinstance(d, np.dtype) and d.kind in "iu" for d in dtypes[:n_ids]):
        return None
    return X.to_numpy(dtype=object)


def _list_as_objects(X: ArrayLike, n_ids: int) -> NDArray[np.generic] | None:
    """A list of rows whose identifiers numpy would recast, as objects; else None."""
    arr = np.asarray(X)
    if arr.ndim != _MATRIX_NDIM or arr.shape[1] <= n_ids or arr.dtype.kind not in "USf":
        return None
    objects = np.array(X, dtype=object)
    if arr.dtype.kind == "f":
        ids = objects[:, :n_ids].ravel().tolist()
        keep = all(_is_number(v) and isinstance(v, numbers.Integral) for v in ids)
    else:
        keep = not all(isinstance(v, str) for v in objects.ravel().tolist())
    return objects if keep else None


def _integer_ids_restored(arr: NDArray[np.generic], n_ids: int) -> NDArray[np.generic]:
    """A float table whose identifier columns hold whole numbers, with those as integers.

    The float dtype came from the other columns, so the identifiers are made integers
    again -- in an ``object`` array beside the float columns -- as long as a float can
    have held them exactly.
    """
    if arr.dtype.kind != "f" or arr.ndim != _MATRIX_NDIM or arr.shape[1] <= n_ids:
        return arr
    ids = arr[:, :n_ids].astype(np.float64, copy=False)
    if not ids.size or not np.all(np.isfinite(ids)) or not np.all(ids == np.round(ids)):
        return arr
    if np.any(np.abs(ids) >= _FLOAT_EXACT_INTEGERS):
        raise ValueError(
            "The identifiers are integers stored as floats because of the other columns of "
            "X, and some are too large for a float to hold exactly, so distinct ones may "
            "already have collided. Pass a DataFrame or an object array to keep them "
            "integers."
        )
    int_ids = ids.astype(np.int64)
    return stack_columns([*int_ids.T, *arr[:, n_ids:].T])


def check_optionally_timed(
    X: ArrayLike, y: ArrayLike | None = None
) -> tuple[
    NDArray[np.generic], NDArray[np.generic], NDArray[np.float64], NDArray[np.generic] | None
]:
    """:func:`check_interactions` for code that serves recommenders with and without time.

    A splitter or a scorer does not choose whether time is used; the recommender it is
    handed does. So a third column is read as time here, and ``times`` is ``None`` for
    two-column interactions.

    It cannot tell a time from a context column, so it reads exactly three columns as
    timed interactions and is not used where context may be present: code that knows the
    recommender calls :func:`check_interactions` with its ``time``.
    """
    X_arr = check_rows(X)
    if X_arr.shape[1] == _N_TIMED_COLUMNS:
        return check_interactions(X_arr, y, time=True)
    users, items, weights = check_interactions(X_arr, y)
    return users, items, weights, None


def drop_time(X: NDArray[np.generic]) -> NDArray[np.generic]:
    """The identifier columns of validated interactions or pairs, without time or context.

    A time or context column next to identifiers of another type makes the whole array
    ``object``; identifiers that are all numbers are given back their numeric dtype, so
    that what a recommender is fitted on does not depend on what its caller carried.
    """
    ids = X[:, :_N_COLUMNS]
    if X.shape[1] == _N_COLUMNS:
        return ids
    return _numeric_if_possible(ids)


def _numeric_if_possible(values: NDArray[np.generic]) -> NDArray[np.generic]:
    """An ``object`` array of numbers as a numeric array; anything else as it is."""
    if values.dtype.kind != "O" or not values.size:
        return values
    if all(_is_number(v) for v in values.ravel().tolist()):
        return np.array(values.tolist())
    return values


def interaction_context(X: NDArray[np.generic], *, time: bool) -> NDArray[np.generic] | None:
    """The query context columns of validated interactions or pairs, or None without any.

    The columns after the system ones: ``[user, item]``, and the time when ``time`` is
    true. Returned as given -- numbers stay numbers, other values stay objects -- for the
    features that read them to interpret.
    """
    n_system = _N_TIMED_COLUMNS if time else _N_COLUMNS
    if X.shape[1] <= n_system:
        return None
    return _numeric_if_possible(X[:, n_system:])


def check_queries(X: ArrayLike) -> tuple[NDArray[np.generic], NDArray[np.generic] | None]:
    """Validate the queries of ``recommend``: user identifiers, with or without context.

    Parameters
    ----------
    X : array-like of shape (n_queries,) or (n_queries, 1 + n_context)
        A vector of user identifiers asks without context. A matrix holds the user in
        ``X[:, 0]`` and the context of each query in the other columns, laid out like
        the context columns of the ``X`` of ``fit``.

    Returns
    -------
    queries : ndarray of shape (n_queries,)
    context : ndarray of shape (n_queries, n_context), or None
        None for a vector, or a matrix of a single column.

    Raises
    ------
    ValueError
        If ``X`` is a float matrix whose users are integers too large for a float to
        have kept exactly; see :func:`check_rows`.
    """
    if isinstance(X, np.ndarray) and X.ndim == 1:
        return check_ids(X), None
    # A query row mixes a user with context of any type; see check_rows.
    arr = check_array(_typed_rows(X, 1), ensure_2d=False, dtype=None, ensure_all_finite=False)
    if arr.ndim == 1:
        return check_ids(arr), None
    arr = _integer_ids_restored(arr, 1)
    if arr.ndim != _MATRIX_NDIM or arr.shape[1] == 0:
        raise ValueError(
            "X must be a vector of user identifiers or a matrix of shape "
            f"(n_queries, 1 + n_context), got shape {arr.shape}."
        )
    queries = check_ids(_numeric_if_possible(arr[:, 0]))
    if arr.shape[1] == 1:
        return queries, None
    return queries, _numeric_if_possible(arr[:, 1:])


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
            f"{name} must hold numbers or datetimes, got strings. An array mixing string "
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


def _is_number(value: _T) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool | np.bool_)


def _is_missing(value: _T) -> bool:
    """Whether ``value`` is ``None``, a NaN or a NaT."""
    if isinstance(value, np.datetime64):
        return bool(np.isnat(value))
    return value is None or (isinstance(value, numbers.Real) and math.isnan(value))


def _as_datetime64(value: _T) -> _T | np.datetime64:
    """A ``pandas.Timestamp`` or ``pandas.NaT`` as a ``datetime64``; anything else as it is."""
    to_datetime64 = getattr(value, "to_datetime64", None)
    return value if to_datetime64 is None else to_datetime64()


def _times_from_objects(values: NDArray[np.object_], name: str) -> NDArray[np.generic]:
    """Numbers or ``datetime64[ns]`` from an object column; ``None`` becomes missing."""
    items = values.tolist()
    present = [v for v in items if not _is_missing(v)]
    if all(isinstance(v, datetime.date | np.datetime64) for v in present) and present:
        # numpy reads a pandas.Timestamp as the datetime it subclasses, to the
        # microsecond; its own datetime64 keeps the nanoseconds.
        return np.array([_as_datetime64(v) for v in items], dtype=_DATETIME)
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


def stack_columns(columns: list[NDArray[np.generic]]) -> NDArray[np.generic]:
    """Columns side by side, each keeping what it holds.

    Columns of one dtype stack as they are. Otherwise the result is ``object``, as in
    :func:`stack_pairs`, and a ``datetime64`` column is stored as ``datetime64`` scalars
    -- NaT where missing -- because numpy would cast its values to bare integers, or to
    ``datetime`` objects that drop the nanoseconds.
    """
    if all(c.dtype == columns[0].dtype for c in columns):
        return np.column_stack(columns)
    out = np.empty((len(columns[0]), len(columns)), dtype=object)
    for j, column in enumerate(columns):
        if column.dtype.kind == "M":
            # A list of scalars is assigned element by element, each kept as it is.
            out[:, j] = list(column)
        else:
            out[:, j] = column
    return out


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
