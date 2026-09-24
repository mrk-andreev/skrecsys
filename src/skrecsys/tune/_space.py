"""Search-space distributions, declared as ``typing.Annotated`` metadata on ``__init__``.

An estimator states the range worth searching next to the parameter itself::

    def __init__(self, k1: Annotated[float, Float(0.05, 5.0, log=True)] = 1.2): ...

``Annotated`` leaves the parameter's type what it was, so type checkers and
scikit-learn's ``get_params``, which reads only the names, see no difference, while
:func:`search_space` finds the ranges without a second table that could drift from the
signature.
"""

import inspect
import math
import numbers
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Annotated, TypeAlias, get_args, get_origin

from skrecsys._typing import Estimator

__all__ = ["Categorical", "Distribution", "Float", "Int", "search_space"]


@dataclass(frozen=True)
class Float:
    """A real parameter in ``[low, high]``.

    Parameters
    ----------
    low, high : float
        Inclusive bounds, ``low <= high``.
    log : bool, default=False
        Search ``log(x)`` uniformly instead of ``x``, for parameters whose effect is
        multiplicative such as a regularization strength. Needs ``low > 0``.
    """

    low: float
    high: float
    log: bool = False

    def __post_init__(self) -> None:
        _check_bounds(self, self.low, self.high)
        if self.log and self.low <= 0:
            raise ValueError(f"A log-scaled Float needs low > 0, got {self.low}.")

    def contains(self, value: object) -> bool:
        """Whether ``value`` is a real number within the bounds."""
        return (
            isinstance(value, numbers.Real)
            and not isinstance(value, bool)
            and self.low <= float(value) <= self.high
        )


@dataclass(frozen=True)
class Int:
    """An integer parameter in ``[low, high]``.

    Parameters
    ----------
    low, high : int
        Inclusive bounds, ``low <= high``.
    log : bool, default=False
        Search ``log(x)`` uniformly instead of ``x``, for counts spanning orders of
        magnitude such as a number of neighbours. Needs ``low >= 1``.
    """

    low: int
    high: int
    log: bool = False

    def __post_init__(self) -> None:
        for bound in (self.low, self.high):
            if isinstance(bound, bool) or not isinstance(bound, numbers.Integral):
                raise TypeError(f"Int bounds must be integers, got {bound!r}.")
        _check_bounds(self, self.low, self.high)
        if self.log and self.low < 1:
            raise ValueError(f"A log-scaled Int needs low >= 1, got {self.low}.")

    def contains(self, value: object) -> bool:
        """Whether ``value`` is an integer within the bounds."""
        return (
            isinstance(value, numbers.Integral)
            and not isinstance(value, bool)
            and self.low <= int(value) <= self.high
        )


@dataclass(frozen=True)
class Categorical:
    """A parameter taking one of a fixed set of values, which need not be ordered.

    Parameters
    ----------
    choices : sequence
        The values, at least one, compared with ``==``.
    """

    choices: tuple[Hashable, ...]

    def __init__(self, choices: Sequence[Hashable]) -> None:
        object.__setattr__(self, "choices", tuple(choices))
        if not self.choices:
            raise ValueError("Categorical needs at least one choice.")

    def contains(self, value: object) -> bool:
        """Whether ``value`` is one of the choices."""
        return any(value == choice for choice in self.choices)

    def index(self, value: object) -> int:
        """The position of ``value`` among the choices."""
        return next(i for i, choice in enumerate(self.choices) if value == choice)


#: Any of the distributions above.
Distribution: TypeAlias = Float | Int | Categorical


def _check_bounds(dist: Float | Int, low: float, high: float) -> None:
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        raise ValueError(
            f"{type(dist).__name__} needs finite bounds with low <= high, got [{low}, {high}]."
        )


def search_space(estimator: Estimator) -> dict[str, Distribution]:
    """The tunable parameters an estimator declares, by ``set_params`` name.

    A parameter is tunable when its ``__init__`` annotation is ``Annotated[T, dist]``
    with ``dist`` a :class:`Float`, :class:`Int` or :class:`Categorical`. A parameter
    holding another estimator contributes that estimator's space under
    ``name__param``, the way scikit-learn names nested parameters.

    Parameters
    ----------
    estimator : estimator
        An instance, configured or not; only its class and its nested estimators are
        read.

    Returns
    -------
    space : dict of str to distribution

    Examples
    --------
    >>> from skrecsys.recommendation import BM25Recommender
    >>> sorted(search_space(BM25Recommender()))
    ['b', 'k1', 'n_neighbors']
    """
    space: dict[str, Distribution] = {}
    for name, param in inspect.signature(type(estimator).__init__).parameters.items():
        dist = _distribution(param.annotation)
        if dist is not None:
            space[name] = dist
    for name, value in estimator.get_params(deep=False).items():
        if hasattr(value, "get_params") and not isinstance(value, type):
            nested = search_space(value)  # ty: ignore[invalid-argument-type]
            space.update({f"{name}__{key}": dist for key, dist in nested.items()})
    return space


def _distribution(annotation: object) -> Distribution | None:
    if get_origin(annotation) is not Annotated:
        return None
    return next(
        (meta for meta in get_args(annotation)[1:] if isinstance(meta, Float | Int | Categorical)),
        None,
    )
