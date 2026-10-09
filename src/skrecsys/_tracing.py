"""The hooks estimators call to report what ``recommend`` decided, when someone listens.

Nothing here is public: :mod:`skrecsys.inspection` builds on it. The module sits below
:mod:`skrecsys.base` in the import graph -- it imports nothing of the package but
:mod:`skrecsys.utils.validation` -- so every estimator can report without an import cycle.

A :class:`Tracer` is made current through a :class:`~contextvars.ContextVar`, never stored
on an estimator, so tracing leaves cloning and pickling alone. Estimators reach it in two
ways:

- ``recommend`` methods are wrapped in :func:`traced_recommend`, which opens a *call*: it
  names the estimator in the current path and records the request and what was served.
- Inside a call, a composite asks :func:`active_tracer` for the tracer and reports its
  decisions -- a route, a candidate list, a feature matrix -- through the tracer's
  methods, and names its parts with :func:`span`.

When no tracer is current, :func:`active_tracer` is one ``ContextVar.get`` and the wrapper
one more, per ``recommend`` call rather than per query.

Events are columnar: one event holds a block of queries as numpy arrays, the way
``recommend`` works on them. Events with a list per query hold ``pairs`` of shape
``(n_pairs, >= 2)`` -- query, item -- and ``groups``, the length of each query's list; the
query identifiers are what ties the events of one query together across the stages.
"""

import contextlib
import functools
import inspect
from collections.abc import Callable, Iterator
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Concatenate, Literal, ParamSpec, Self, TypeAlias, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from skrecsys.utils.validation import (
    check_queries,
    factorize,
    lookup_ids,
    stable_unit_hash,
    stack_pairs,
)

#: How much a tracer records: ``"full"`` adds the feature matrices and the leaf
#: attributions to the ``"decisions"`` every stage takes.
Level: TypeAlias = Literal["decisions", "full"]

#: Which queries a tracer records: all of them (``None``), a fraction of them chosen by a
#: stable hash of their identifier, or those a callable marks True.
Sample: TypeAlias = float | Callable[[NDArray[np.generic]], ArrayLike] | None

_LEVELS = ("decisions", "full")


@dataclass(frozen=True, slots=True)
class Event:
    """Something a ``recommend`` call did.

    ``call`` numbers the outermost ``recommend`` calls a tracer has seen, from 0, so the
    events of one call can be told from another's. ``path`` names where the event
    happened: the estimators and parts it is nested in, ``/``-separated, for instance
    ``Switch/on_true/Cascade/generator/ItemKNNRecommender``.
    """

    call: int
    path: str


@dataclass(frozen=True, slots=True)
class Request(Event):
    """A ``recommend`` call: its queries, and every argument but ``X``, defaults included."""

    queries: NDArray[np.generic]
    params: dict[str, object]


@dataclass(frozen=True, slots=True)
class Served(Event):
    """What a ``recommend`` call returned, one row per query."""

    queries: NDArray[np.generic]
    items: NDArray[np.generic]
    scores: NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class Raised(Event):
    """A ``recommend`` call that raised, recorded where the exception was first seen."""

    error: str
    message: str


@dataclass(frozen=True, slots=True)
class Route(Event):
    """The branch a condition sent each query to: ``mask`` True for ``on_true``."""

    condition: str
    queries: NDArray[np.generic]
    mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Candidates(Event):
    """A candidate list per query, best first: what ``source`` retrieved.

    ``scores`` has one column per generator when ``source`` is the merge of several.
    """

    source: str
    pairs: NDArray[np.generic]
    scores: NDArray[np.float64]
    groups: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Fusion(Event):
    """What one list of a reciprocal rank fusion adds to each of its items.

    ``ranks`` are 1-based positions in ``source``'s list; ``contributions`` its weight
    over ``k + rank``.
    """

    source: str
    pairs: NDArray[np.generic]
    ranks: NDArray[np.int64]
    contributions: NDArray[np.float64]
    groups: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Features(Event):
    """The feature matrix a ranker was given, one row per candidate pair."""

    pairs: NDArray[np.generic]
    values: NDArray[np.floating]
    names: tuple[str, ...] | None
    groups: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class RankerScores(Event):
    """What a ranker scored each candidate pair, in candidate order."""

    pairs: NDArray[np.generic]
    scores: NDArray[np.float64]
    groups: NDArray[np.int64]
    #: What each feature added to each score, then the bias; "full" only, and only
    #: for a ranker that can say (see ``RankerMixin._contributions``).
    contributions: NDArray[np.float64] | None = None
    #: Each candidate's item in the fitted item order, which breaks score ties as
    #: ``recommend`` does; None when the caller did not say.
    positions: NDArray[np.intp] | None = None


@dataclass(frozen=True, slots=True)
class Attribution(Event):
    """Why a leaf recommender scored its served items as it did; "full" only.

    One row per served pair, laid out as :class:`skrecsys._attribution.Attributions`.
    """

    pairs: NDArray[np.generic]
    groups: NDArray[np.int64]
    kind: str
    exact: bool
    reason_items: NDArray[np.object_]
    weights: NDArray[np.float64]
    rest: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Postprocess(Event):
    """The ranked lists a ``postprocess`` callback was given, and those it returned."""

    before_pairs: NDArray[np.generic]
    before_scores: NDArray[np.float64]
    before_groups: NDArray[np.int64]
    after_pairs: NDArray[np.generic]
    after_scores: NDArray[np.float64]
    after_groups: NDArray[np.int64]


class Tracer:
    """Collects the events of the ``recommend`` calls made while it is current.

    Parameters
    ----------
    level : {"full", "decisions"}, default="full"
        ``"decisions"`` leaves out the feature matrices and the leaf attributions, which
        dominate the size of a trace.
    sample : float, callable or None, default=None
        Which queries to record. ``None`` records all. A float in [0, 1] records that
        fraction, chosen by a hash of the identifier that is stable across processes, so
        a sampled user is sampled every time. A callable takes the queries of an outermost
        ``recommend`` call and returns a boolean mask over them. Nested calls record the
        same queries: a query is traced through every stage or not at all.
    """

    def __init__(self, *, level: Level = "full", sample: Sample = None, n_reasons: int = 5) -> None:
        if level not in _LEVELS:
            raise ValueError(f"level must be one of {_LEVELS}, got {level!r}.")
        if isinstance(sample, bool) or not (
            sample is None or callable(sample) or isinstance(sample, int | float)
        ):
            raise TypeError(
                f"sample must be a float, a callable or None, got {type(sample).__name__}."
            )
        if isinstance(sample, int | float) and not 0 <= sample <= 1:
            raise ValueError(f"sample must be in [0, 1], got {sample}.")
        if isinstance(n_reasons, bool) or not isinstance(n_reasons, int) or n_reasons < 1:
            raise ValueError(f"n_reasons must be an integer >= 1, got {n_reasons!r}.")
        self.level: Level = level
        self.sample = sample
        self.n_reasons = n_reasons
        self.events: list[Event] = []
        self._path: list[str] = []
        self._depth = 0
        self._suspended = 0
        self._muted = False
        self._call = -1
        #: Sorted distinct identifiers the current call records; None records all.
        self._sampled: NDArray[np.generic] | None = None
        self._raised: BaseException | None = None
        self._tokens: list[Token[Tracer | None]] = []

    def __enter__(self) -> Self:
        self._tokens.append(_ACTIVE.set(self))
        return self

    def __exit__(self, *exc_info: object) -> None:
        _ACTIVE.reset(self._tokens.pop())

    @property
    def full(self) -> bool:
        """Whether feature matrices and attributions are recorded."""
        return self.level == "full"

    # -- what subclasses hook into ---------------------------------------------------------

    def _emit(self, event: Event) -> None:
        self.events.append(event)

    def _end_call(self, call: int) -> None:
        """Called once an outermost ``recommend`` call has returned or raised."""

    # -- calls and spans -------------------------------------------------------------------

    @property
    def _where(self) -> str:
        return "/".join(self._path)

    def _record_call(
        self,
        method: Callable[..., tuple[NDArray[np.generic], NDArray[np.floating]]],
        signature: inspect.Signature,
        estimator: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[NDArray[np.generic], NDArray[np.floating]]:
        bound = signature.bind(estimator, *args, **kwargs)
        bound.apply_defaults()
        params = dict(bound.arguments)
        params.pop(next(iter(signature.parameters)))  # self
        queries, _ = check_queries(params.pop("X"))
        outer = self._depth == 0
        if outer:
            self._call += 1
            self._sampled = self._sample(queries)
            if self._sampled is not None and not len(self._sampled):
                self._muted = True
                try:
                    return method(estimator, *args, **kwargs)
                finally:
                    self._muted = False
                    self._sampled = None
        self._path.append(type(estimator).__name__)
        self._depth += 1
        try:
            keep = self._rows(queries)
            self._emit(Request(self._call, self._where, _take(queries, keep), params))
            items, scores = method(estimator, *args, **kwargs)
            served = Served(
                self._call,
                self._where,
                _take(queries, keep),
                _take(np.asarray(items), keep),
                _take(np.asarray(scores, dtype=np.float64), keep),
            )
            self._emit(served)
            if self.full:
                self._attribute(estimator, served)
            return items, scores
        except Exception as exc:
            if exc is not self._raised:
                self._raised = exc
                self._emit(Raised(self._call, self._where, type(exc).__name__, str(exc)))
            raise
        finally:
            self._depth -= 1
            self._path.pop()
            if outer:
                self._sampled = None
                self._raised = None
                self._end_call(self._call)

    def _attribute(self, estimator: object, served: Served) -> None:
        """Report why a leaf scored what it served; a composite has no reasons of its own.

        The leaf is asked only about the queries it just served and the items it chose,
        so attribution cannot meet an input serving did not; a failure is a bug and
        propagates like one.
        """
        attribute = getattr(estimator, "_attribute", None)
        if attribute is None or not served.items.size:
            return
        k = served.items.shape[1]
        users = np.repeat(served.queries, k)
        found = attribute(users, served.items.ravel(), self.n_reasons)
        if found is None:
            return
        self._emit(
            Attribution(
                self._call,
                self._where,
                stack_pairs(users, served.items.ravel()),
                np.full(len(served.queries), k, dtype=np.int64),
                found.kind,
                found.exact,
                found.items,
                found.weights,
                found.rest,
            )
        )

    def _push(self, name: str) -> None:
        self._path.append(name)

    def _pop(self) -> None:
        self._path.pop()

    # -- sampling --------------------------------------------------------------------------

    def _sample(self, queries: NDArray[np.generic]) -> NDArray[np.generic] | None:
        """The distinct queries an outermost call records, or None for all of them."""
        sample = self.sample
        if sample is None:
            return None
        if isinstance(sample, int | float):
            if sample >= 1:
                return None
            mask = stable_unit_hash(queries) < sample
        else:
            mask = np.asarray(sample(queries))
            if mask.shape != queries.shape or mask.dtype != np.bool_:
                raise ValueError(
                    f"sample must return a boolean array of shape {queries.shape}, got "
                    f"{mask.dtype} of shape {mask.shape}."
                )
        return factorize(queries[mask])[0]

    def _rows(self, queries: NDArray[np.generic]) -> NDArray[np.bool_] | None:
        """Which rows of a per-query array belong to a recorded query; None for all."""
        if self._sampled is None:
            return None
        return lookup_ids(queries, self._sampled, name="query")[1]

    def _group_rows(
        self, pairs: NDArray[np.generic], groups: NDArray[np.int64]
    ) -> tuple[NDArray[np.bool_] | None, NDArray[np.int64]]:
        """Which rows of a grouped array are recorded, and the groups that remain."""
        if self._sampled is None or not len(groups):
            return None, groups
        starts = np.cumsum(groups) - groups
        kept = lookup_ids(pairs[starts, 0], self._sampled, name="query")[1]
        return np.repeat(kept, groups), groups[kept]

    # -- what estimators report ------------------------------------------------------------

    def route(
        self, condition: object, queries: NDArray[np.generic], mask: NDArray[np.bool_]
    ) -> None:
        """Report which queries ``condition`` sent to ``on_true``."""
        keep = self._rows(queries)
        self._emit(
            Route(self._call, self._where, repr(condition), _take(queries, keep), _take(mask, keep))
        )

    def candidates(
        self,
        source: str,
        pairs: NDArray[np.generic],
        scores: NDArray[np.float64],
        groups: NDArray[np.int64],
    ) -> None:
        """Report the candidate lists ``source`` retrieved."""
        keep, groups = self._group_rows(pairs, groups)
        self._emit(
            Candidates(
                self._call, self._where, source, _take(pairs, keep), _take(scores, keep), groups
            )
        )

    def fusion(
        self,
        source: str,
        pairs: NDArray[np.generic],
        ranks: NDArray[np.int64],
        contributions: NDArray[np.float64],
        groups: NDArray[np.int64],
    ) -> None:
        """Report what ``source``'s lists add to a reciprocal rank fusion."""
        keep, groups = self._group_rows(pairs, groups)
        self._emit(
            Fusion(
                self._call,
                self._where,
                source,
                _take(pairs, keep),
                _take(ranks, keep),
                _take(contributions, keep),
                groups,
            )
        )

    def features(
        self,
        pairs: NDArray[np.generic],
        values: NDArray[np.floating],
        groups: NDArray[np.int64],
        names: tuple[str, ...] | None,
    ) -> None:
        """Report the feature matrix a ranker scored the candidates from; "full" only."""
        if not self.full:
            return
        keep, groups = self._group_rows(pairs, groups)
        self._emit(
            Features(
                self._call,
                self._where,
                _take(pairs, keep),
                _take(np.asarray(values, dtype=np.float64), keep),
                names,
                groups,
            )
        )

    def ranker_scores(
        self,
        pairs: NDArray[np.generic],
        scores: NDArray[np.float64],
        groups: NDArray[np.int64],
        contributions: NDArray[np.float64] | None = None,
        positions: NDArray[np.intp] | None = None,
    ) -> None:
        """Report what the ranker scored each candidate, and why when it can say.

        ``positions``, the candidates' fitted item order, breaks ties among the scores.
        """
        keep, groups = self._group_rows(pairs, groups)
        self._emit(
            RankerScores(
                self._call,
                self._where,
                _take(pairs, keep),
                _take(scores, keep),
                groups,
                None if contributions is None or not self.full else _take(contributions, keep),
                None if positions is None else _take(positions, keep),
            )
        )

    def postprocess(
        self,
        before: tuple[NDArray[np.generic], NDArray[np.float64], NDArray[np.int64]],
        after: tuple[NDArray[np.generic], NDArray[np.float64], NDArray[np.int64]],
    ) -> None:
        """Report the lists ``postprocess`` was given and those it returned."""
        keep_before, groups_before = self._group_rows(before[0], before[2])
        keep_after, groups_after = self._group_rows(after[0], after[2])
        self._emit(
            Postprocess(
                self._call,
                self._where,
                _take(before[0], keep_before),
                _take(before[1], keep_before),
                groups_before,
                _take(after[0], keep_after),
                _take(after[1], keep_after),
                groups_after,
            )
        )


#: The tracer the current context reports to.
_ACTIVE: ContextVar[Tracer | None] = ContextVar("skrecsys_tracer", default=None)


def current_tracer() -> Tracer | None:
    """The tracer of the current context, whether or not a call is in progress."""
    return _ACTIVE.get()


def active_tracer() -> Tracer | None:
    """The tracer to report to from inside ``recommend``, or None when nobody listens.

    None also outside a ``recommend`` call -- a ``predict`` or a ``fit`` made while a
    tracer is current records nothing -- inside :func:`untraced`, and for a call whose
    queries are all sampled out.
    """
    tracer = _ACTIVE.get()
    if tracer is None or not tracer._depth or tracer._suspended or tracer._muted:
        return None
    return tracer


@contextlib.contextmanager
def span(name: str) -> Iterator[None]:
    """Name a part of a composite in the path of the events reported inside it."""
    tracer = active_tracer()
    if tracer is None:
        yield
        return
    tracer._push(name)
    try:
        yield
    finally:
        tracer._pop()


@contextlib.contextmanager
def untraced() -> Iterator[None]:
    """Record nothing inside: for work a composite does that is not serving its queries.

    ``fit`` asks its parts for recommendations to learn from, and ``_count_eligible``
    may retrieve candidates to count them; neither is what a trace is about. Usable as
    a decorator, ``@untraced()``.
    """
    tracer = _ACTIVE.get()
    if tracer is None:
        yield
        return
    tracer._suspended += 1
    try:
        yield
    finally:
        tracer._suspended -= 1


_Self = TypeVar("_Self")
_Scalar = TypeVar("_Scalar", bound=np.generic)
_P = ParamSpec("_P")
_Out: TypeAlias = tuple[NDArray[np.generic], NDArray[np.floating]]


def traced_recommend(
    method: Callable[Concatenate[_Self, _P], _Out],
) -> Callable[Concatenate[_Self, _P], _Out]:
    """Wrap a ``recommend`` method so that a current tracer records the call."""
    signature = inspect.signature(method)

    @functools.wraps(method)
    def recommend(self: _Self, *args: _P.args, **kwargs: _P.kwargs) -> _Out:
        tracer = _ACTIVE.get()
        if tracer is None or tracer._suspended or tracer._muted:
            return method(self, *args, **kwargs)
        return tracer._record_call(method, signature, self, args, dict(kwargs))

    return recommend


def feature_names(component: object) -> tuple[str, ...] | None:
    """``component.get_feature_names_out()``, or None when it cannot name its columns."""
    names_out = getattr(component, "get_feature_names_out", None)
    if names_out is None:
        return None
    # Naming is a courtesy; a trace must not fail on it.
    with contextlib.suppress(Exception):
        return tuple(str(name) for name in names_out())
    return None


def _take(values: NDArray[_Scalar], keep: NDArray[np.bool_] | None) -> NDArray[_Scalar]:
    return values if keep is None else values[keep]
