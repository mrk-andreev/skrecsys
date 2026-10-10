"""An ask-and-tell optimization study, in the manner of Optuna, sampled in Rust.

A :class:`Study` hands out :class:`Trial` objects; the objective asks each trial for the
parameter values it needs -- define-by-run, so which parameters exist may depend on
earlier answers -- and the study is told the score. Sampling is univariate: each value
comes from ``_core.tune_suggest`` given the history of that one parameter, which is what
lets the search space change from trial to trial.
"""

import math
import sys
import zlib
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.utils import check_random_state

from skrecsys import _core
from skrecsys.tune._space import Categorical, Distribution, Float, Int
from skrecsys.typing import RandomStateLike
from skrecsys.utils._param_validation import check_int

__all__ = ["Study", "Trial"]

_SAMPLERS = ("tpe", "random")
_DIRECTIONS = ("maximize", "minimize")


@dataclass
class Trial:
    """One evaluation of the objective: the values it was given and what it scored.

    Obtained from :meth:`Study.ask`, never constructed directly.

    Attributes
    ----------
    number : int
        Position in the study, from 0.
    params : dict
        The values suggested so far, by name.
    distributions : dict
        The distribution each value was drawn from, by name.
    value : float or None
        The score, once told; ``None`` while running and for a failed trial.
    state : {"running", "complete", "failed"}
    """

    number: int
    params: dict[str, object] = field(default_factory=dict)
    distributions: dict[str, Distribution] = field(default_factory=dict)
    value: float | None = None
    state: Literal["running", "complete", "failed"] = "running"
    _study: "Study | None" = field(default=None, repr=False, compare=False)
    _fixed: dict[str, object] = field(default_factory=dict, repr=False, compare=False)

    def suggest(self, name: str, distribution: Distribution) -> object:
        """A value for parameter ``name`` from ``distribution``.

        Asking twice for the same name returns the first answer, so helper functions
        can each ask for what they need.
        """
        _check_distribution(name, distribution)
        if name in self.params:
            if self.distributions[name] != distribution:
                raise ValueError(
                    f"Parameter {name!r} was already suggested from {self.distributions[name]}."
                )
            return self.params[name]
        if self._study is None:
            raise RuntimeError("This trial is not attached to a study.")
        fixed = self._fixed.get(name, _MISSING)
        if fixed is not _MISSING and distribution.contains(fixed):
            value = fixed
        else:
            value = self._study._sample(self.number, name, distribution)
        self.params[name] = value
        self.distributions[name] = distribution
        return value

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        """A real value in ``[low, high]``; see :class:`Float`."""
        return float(self.suggest(name, Float(low, high, log=log)))  # ty: ignore[invalid-argument-type]

    def suggest_int(self, name: str, low: int, high: int, *, log: bool = False) -> int:
        """An integer in ``[low, high]``; see :class:`Int`."""
        return int(self.suggest(name, Int(low, high, log=log)))  # ty: ignore[invalid-argument-type]

    def suggest_categorical(self, name: str, choices: Sequence[Hashable]) -> object:
        """One of ``choices``; see :class:`Categorical`."""
        return self.suggest(name, Categorical(choices))


_MISSING = object()


class Study:
    """Search for the parameters that optimize an objective, one trial at a time.

    Parameters
    ----------
    direction : {"maximize", "minimize"}, default="maximize"
        Whether higher or lower objective values are better.
    sampler : {"tpe", "random"}, default="tpe"
        ``"tpe"`` is the Tree-structured Parzen Estimator [1]_: after
        ``n_startup_trials`` random trials, each value is drawn where the best tenth of
        the trials so far concentrates relative to the rest. ``"random"`` draws every
        value uniformly (in log space for log-scaled parameters).
    n_startup_trials : int, default=10
        Random trials before TPE takes over.
    n_ei_candidates : int, default=24
        Candidates TPE scores per suggestion.
    random_state : int, RandomState instance or None, default=None
        Seeds the sampler; an integer makes the study reproducible.

    Attributes
    ----------
    trials : list of Trial
        Every trial asked for, in order.

    References
    ----------
    .. [1] J. Bergstra, R. Bardenet, Y. Bengio and B. Kégl, "Algorithms for
       Hyper-Parameter Optimization", NeurIPS 2011.

    Examples
    --------
    >>> study = Study(random_state=0)
    >>> study.optimize(lambda t: -(t.suggest_float("x", -10, 10) - 2) ** 2, n_trials=60)
    >>> round(study.best_params["x"])
    2
    """

    def __init__(
        self,
        *,
        direction: Literal["maximize", "minimize"] = "maximize",
        sampler: Literal["tpe", "random"] = "tpe",
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        random_state: RandomStateLike = None,
    ) -> None:
        if direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}.")
        if sampler not in _SAMPLERS:
            raise ValueError(f"sampler must be one of {_SAMPLERS}, got {sampler!r}.")
        self.direction = direction
        self.sampler = sampler
        self.n_startup_trials = check_int(n_startup_trials, "n_startup_trials", min_value=0)
        self.n_ei_candidates = check_int(n_ei_candidates, "n_ei_candidates", min_value=1)
        self._seed = int(check_random_state(random_state).randint(0, 2**31 - 1))
        self._queue: list[dict[str, object]] = []
        self.trials: list[Trial] = []

    def enqueue(self, params: Mapping[str, object]) -> None:
        """Have a coming trial use ``params`` where they fall inside its distributions.

        The usual use is to evaluate the defaults first, so the study can only improve
        on them. A value outside the distribution it is asked from is sampled instead.
        """
        self._queue.append(dict(params))

    def ask(self) -> Trial:
        """Start a new trial."""
        fixed = self._queue.pop(0) if self._queue else {}
        trial = Trial(number=len(self.trials), _study=self, _fixed=fixed)
        self.trials.append(trial)
        return trial

    def tell(self, trial: Trial, value: float | None) -> None:
        """Finish ``trial`` with its score; ``None``, NaN or an infinity marks it failed."""
        if trial._study is not self or trial.state != "running":
            raise ValueError(f"Trial {trial.number} is not a running trial of this study.")
        if value is None or not math.isfinite(value):
            trial.state = "failed"
            return
        trial.value = float(value)
        trial.state = "complete"

    def optimize(self, objective: Callable[[Trial], float], n_trials: int) -> None:
        """Run ``n_trials`` trials of ``objective`` one after another.

        An exception from ``objective`` marks its trial failed and propagates.
        """
        check_int(n_trials, "n_trials", min_value=1)
        for _ in range(n_trials):
            trial = self.ask()
            try:
                value = objective(trial)
            except BaseException:
                self.tell(trial, None)
                raise
            self.tell(trial, value)

    @property
    def best_trial(self) -> Trial:
        """The complete trial with the best value; the earliest among equals."""
        complete = [t for t in self.trials if t.state == "complete"]
        if not complete:
            raise ValueError("No trial has completed yet.")
        sign = 1.0 if self.direction == "maximize" else -1.0
        return max(complete, key=lambda t: (sign * float(t.value), -t.number))  # ty: ignore[invalid-argument-type]

    @property
    def best_params(self) -> dict[str, object]:
        """The parameters of :attr:`best_trial`."""
        return dict(self.best_trial.params)

    @property
    def best_value(self) -> float:
        """The value of :attr:`best_trial`."""
        return float(self.best_trial.value)  # ty: ignore[invalid-argument-type]

    def _sample(self, number: int, name: str, distribution: Distribution) -> object:
        observed, scores = [], []
        sign = 1.0 if self.direction == "maximize" else -1.0
        for trial in self.trials:
            if trial.state != "complete" or trial.distributions.get(name) != distribution:
                continue
            value = trial.params[name]
            observed.append(
                distribution.index(value) if isinstance(distribution, Categorical) else float(value)  # ty: ignore[invalid-argument-type]
            )
            scores.append(sign * float(trial.value))  # ty: ignore[invalid-argument-type]

        match distribution:
            case Float(low, high, log):
                kind, bounds, n_choices = "float", (float(low), float(high)), 0
            case Int(low, high, log):
                kind, bounds, n_choices = "int", (float(low), float(high)), 0
            case Categorical(choices):
                kind, bounds, log, n_choices = "categorical", (0.0, 0.0), False, len(choices)
            case _:
                raise _unknown_distribution(name, distribution)
        # One seed per (study, trial, parameter): reproducible, and independent of the
        # order in which the objective happens to ask for its parameters.
        seed = zlib.crc32(name.encode(), self._seed) * 1_000_003 + number
        value = _core.tune_suggest(
            kind,
            bounds[0],
            bounds[1],
            log,
            n_choices,
            np.asarray(observed, dtype=np.float64),
            np.asarray(scores, dtype=np.float64),
            sys.maxsize if self.sampler == "random" else self.n_startup_trials,
            self.n_ei_candidates,
            seed % 2**64,
        )
        match distribution:
            case Float():
                return value
            case Int():
                return round(value)
            case Categorical(choices):
                return choices[round(value)]
            case _:  # pragma: no cover - the match above has already raised
                raise _unknown_distribution(name, distribution)


def _check_distribution(name: str, distribution: object) -> None:
    """Reject anything but the three distributions before it reaches the sampler."""
    if not isinstance(distribution, Float | Int | Categorical):
        raise _unknown_distribution(name, distribution)


def _unknown_distribution(name: str, distribution: object) -> TypeError:
    return TypeError(
        f"Parameter {name!r} needs a Float, Int or Categorical distribution, "
        f"got {type(distribution).__name__}: {distribution!r}."
    )
