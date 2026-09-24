"""Checks for estimator parameters, shared so every estimator words its errors alike.

Estimators call these from ``fit`` (typically in a ``_check_params`` method), never from
``__init__``: ``set_params`` and ``clone`` assign parameters without running
``__init__``, so only a check at fit time sees every value a parameter can take.

``bool`` is rejected wherever a number is expected even though Python counts it as one:
``n_factors=True`` is a mistake, not a request for one factor.
"""

import math
import numbers
from collections.abc import Callable
from typing import Literal, overload

__all__ = ["check_bool", "check_component", "check_int", "check_real", "resolve_n_jobs"]


def check_component(
    value: object, name: str, predicate: Callable[[object], bool], kind: str
) -> None:
    """Check that a parameter holding a part of a composite plays its role.

    Parameters
    ----------
    value : object
        The parameter's value.
    name : str
        The parameter's name, for the error message.
    predicate : callable
        Such as :func:`skrecsys.is_recommender` or :func:`skrecsys.base.is_ranker`.
    kind : str
        The role, with its article, as in ``"a recommender"``.

    Raises
    ------
    TypeError
        When ``predicate(value)`` is false.
    """
    if not predicate(value):
        raise TypeError(f"{name} must be {kind}, got {type(value).__name__}.")


def check_bool(value: object, name: str) -> bool:
    """Check that a parameter is a ``bool``, and return it."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool, got {value!r}.")
    return value


@overload
def check_int(
    value: object,
    name: str,
    *,
    min_value: int,
    max_value: int | None = ...,
    allow_none: Literal[False] = ...,
) -> int: ...
@overload
def check_int(
    value: object,
    name: str,
    *,
    min_value: int,
    max_value: int | None = ...,
    allow_none: Literal[True],
) -> int | None: ...
def check_int(
    value: object,
    name: str,
    *,
    min_value: int,
    max_value: int | None = None,
    allow_none: bool = False,
) -> int | None:
    """Check that a parameter is an integer in ``[min_value, max_value]``, and return it.

    ``max_value=None`` leaves the range open above; ``allow_none`` also accepts ``None``.
    """
    if value is None and allow_none:
        return None
    number = int(value) if isinstance(value, numbers.Integral) else None
    if (
        isinstance(value, bool)
        or number is None
        or number < min_value
        or (max_value is not None and number > max_value)
    ):
        bound = f">= {min_value}" if max_value is None else f"in [{min_value}, {max_value}]"
        prefix = "None or " if allow_none else ""
        raise ValueError(f"{name} must be {prefix}an integer {bound}, got {value!r}.")
    return number


def check_real(
    value: object,
    name: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> float:
    """Check that a parameter is a finite real number within the bounds, and return it.

    NaN and infinities are always rejected; a ``None`` bound leaves that side open.
    """
    number = float(value) if isinstance(value, numbers.Real) else math.nan
    ok = (
        not isinstance(value, bool)
        and math.isfinite(number)
        and (min_value is None or (number >= min_value if min_inclusive else number > min_value))
        and (max_value is None or (number <= max_value if max_inclusive else number < max_value))
    )
    if not ok:
        bounds = _bounds(
            min_value, max_value, min_inclusive=min_inclusive, max_inclusive=max_inclusive
        )
        # Two bounds already rule out the infinities; say so only when a side is open.
        kind = "a real number" if None not in (min_value, max_value) else "a finite real number"
        raise ValueError(f"{name} must be {kind}{bounds}, got {value!r}.")
    return number


def _bounds(
    min_value: float | None, max_value: float | None, *, min_inclusive: bool, max_inclusive: bool
) -> str:
    if min_value is not None and max_value is not None:
        left = "[" if min_inclusive else "("
        right = "]" if max_inclusive else ")"
        return f" in {left}{min_value:g}, {max_value:g}{right}"
    if min_value is not None:
        return f" {'>=' if min_inclusive else '>'} {min_value:g}"
    if max_value is not None:
        return f" {'<=' if max_inclusive else '<'} {max_value:g}"
    return ""


def resolve_n_jobs(n_jobs: object) -> int:
    """Check ``n_jobs`` and return it as a thread count, 0 meaning every core.

    ``None`` and -1 both ask for every core, as in scikit-learn.
    """
    if n_jobs is None or (not isinstance(n_jobs, bool) and n_jobs == -1):
        return 0
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, numbers.Integral) or n_jobs < 1:
        raise ValueError(f"n_jobs must be None, -1 or an integer >= 1, got {n_jobs!r}.")
    return int(n_jobs)
