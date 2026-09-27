"""Tracing ``recommend``: what every stage of a pipeline did, query by query."""

import datetime
import math
from collections.abc import Hashable, Iterator
from dataclasses import dataclass, field, fields
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from skrecsys._tracing import (
    Attribution,
    Candidates,
    Event,
    Features,
    Fusion,
    Level,
    Postprocess,
    Raised,
    RankerScores,
    Request,
    Route,
    Sample,
    Served,
    Tracer,
)
from skrecsys._typing import override

__all__ = [
    "CandidatesStep",
    "FeaturesStep",
    "FusionStep",
    "PostprocessStep",
    "QueryTrace",
    "RaisedStep",
    "RankerStep",
    "RequestStep",
    "RouteStep",
    "ServedStep",
    "Step",
    "Trace",
    "trace",
]


@dataclass(frozen=True)
class Step:
    """One thing a stage did for one query; ``path`` names the stage."""

    path: str

    def to_dict(self) -> dict[str, object]:
        """The step as JSON-compatible values, its kind under ``"step"``."""
        out: dict[str, object] = {"step": _STEP_NAMES[type(self)]}
        out |= {f.name: jsonable(getattr(self, f.name)) for f in fields(self)}
        return out

    def describe(self) -> str:
        """One line saying what the step did."""
        return type(self).__name__


@dataclass(frozen=True)
class RequestStep(Step):
    """A ``recommend`` call the query was part of, with its arguments but ``X``."""

    params: dict[str, object]

    @override
    def describe(self) -> str:
        return f"request n_recommendations={self.params.get('n_recommendations')}"


@dataclass(frozen=True)
class RouteStep(Step):
    """The branch a :class:`~skrecsys.compose.Switch` sent the query to."""

    condition: str
    holds: bool

    @property
    def branch(self) -> str:
        return "on_true" if self.holds else "on_false"

    @override
    def describe(self) -> str:
        return f"route {self.condition} -> {self.branch}"


@dataclass(frozen=True)
class CandidatesStep(Step):
    """The candidates ``source`` retrieved for the query, best first.

    ``scores`` has one column per generator when ``source`` is ``"union"``, the merge of
    several generators' lists.
    """

    source: str
    items: NDArray[np.generic]
    scores: NDArray[np.float64]

    def rank_of(self, item: Hashable) -> int | None:
        """0-based position of ``item`` in the list, None when it is absent."""
        return _position(self.items, item)

    @override
    def describe(self) -> str:
        return f"candidates from {self.source} {_preview(self.items, self.scores)}"


@dataclass(frozen=True)
class FusionStep(Step):
    """What ``source``'s list adds to each of its items in a reciprocal rank fusion."""

    source: str
    items: NDArray[np.generic]
    ranks: NDArray[np.int64]
    contributions: NDArray[np.float64]

    @override
    def describe(self) -> str:
        return f"fusion of {self.source} {_preview(self.items, self.contributions)}"


@dataclass(frozen=True)
class FeaturesStep(Step):
    """The ranker's input row for each candidate of the query, in candidate order."""

    items: NDArray[np.generic]
    names: tuple[str, ...] | None
    values: NDArray[np.floating]

    def row(self, item: Hashable) -> dict[str, float] | None:
        """The features of ``item`` by name (``x0``, ``x1``... when unnamed)."""
        position = _position(self.items, item)
        if position is None:
            return None
        names = self.names or tuple(f"x{j}" for j in range(self.values.shape[1]))
        return dict(zip(names, self.values[position].tolist(), strict=True))

    @override
    def describe(self) -> str:
        return f"features {self.values.shape[1]} columns for {len(self.items)} candidates"


@dataclass(frozen=True)
class RankerStep(Step):
    """What the ranker scored each candidate of the query, in candidate order."""

    items: NDArray[np.generic]
    scores: NDArray[np.float64]
    #: Per candidate, what each feature added to its score, then the bias -- in the
    #: ranker's own space -- when the trace is "full" and the ranker can say.
    contributions: NDArray[np.float64] | None = None
    #: Per candidate, its item's position in the fitted item order, when known.
    positions: NDArray[np.intp] | None = None

    def ranked(self) -> NDArray[np.generic]:
        """The candidates best first by the ranker.

        Ties go by fitted item order, as ``recommend`` breaks them, or keep candidate
        order when the trace does not know it.
        """
        return self.items[self._order()]

    def _order(self) -> NDArray[np.intp]:
        if self.positions is None:
            return np.argsort(-self.scores, kind="stable")
        return np.lexsort((self.positions, -self.scores))

    @override
    def describe(self) -> str:
        order = self._order()
        return f"ranker {_preview(self.items[order], self.scores[order])}"


@dataclass(frozen=True)
class AttributionStep(Step):
    """Why the leaf recommender at ``path`` scored each item it served the query.

    See :mod:`skrecsys._attribution` for ``kind`` and ``exact``. Row ``r`` holds the
    reasons of ``items[r]``: history items and their weights, the largest first, and
    ``rest``, what they leave of an exact score.
    """

    kind: str
    exact: bool
    items: NDArray[np.generic]
    reason_items: NDArray[np.object_]
    weights: NDArray[np.float64]
    rest: NDArray[np.float64]

    def of(self, item: Hashable) -> list[tuple[object, float]] | None:
        """The ``(history item, weight)`` reasons of ``item``; None if it was not served."""
        row = _position(self.items, item)
        if row is None:
            return None
        return [
            (reason, float(weight))
            for reason, weight in zip(self.reason_items[row], self.weights[row], strict=True)
            if reason is not None
        ]

    @override
    def describe(self) -> str:
        exact = "exact" if self.exact else "approximate"
        if self.kind == "popularity" and len(self.items):
            return f"popularity reasons ({exact}); first item: popularity {self.rest[0]:.4g}"
        first = self.of(self.items[0]) if len(self.items) else None
        shown = ", ".join(f"{item!r} {weight:.4g}" for item, weight in first or [])
        return f"{self.kind} reasons ({exact}); first item: [{shown}]"


@dataclass(frozen=True)
class PostprocessStep(Step):
    """The query's list before and after the ``postprocess`` of a Cascade."""

    before_items: NDArray[np.generic]
    before_scores: NDArray[np.float64]
    after_items: NDArray[np.generic]
    after_scores: NDArray[np.float64]

    @property
    def dropped(self) -> list[object]:
        """Items of the ranked list that ``postprocess`` removed."""
        after = set(self.after_items.tolist())
        return [item for item in self.before_items.tolist() if item not in after]

    @property
    def added(self) -> list[object]:
        """Items ``postprocess`` served that were not candidates."""
        before = set(self.before_items.tolist())
        return [item for item in self.after_items.tolist() if item not in before]

    @override
    def describe(self) -> str:
        return f"postprocess dropped {self.dropped}, added {self.added}"


@dataclass(frozen=True)
class ServedStep(Step):
    """What a ``recommend`` call returned to the query."""

    items: NDArray[np.generic]
    scores: NDArray[np.floating]

    @override
    def describe(self) -> str:
        return f"served {_preview(self.items, self.scores)}"


@dataclass(frozen=True)
class RaisedStep(Step):
    """A ``recommend`` call of the query's that raised."""

    error: str
    message: str

    @override
    def describe(self) -> str:
        return f"raised {self.error}: {self.message}"


_StepT = TypeVar("_StepT", bound=Step)

_STEP_NAMES: dict[type[Step], str] = {
    RequestStep: "request",
    RouteStep: "route",
    CandidatesStep: "candidates",
    FusionStep: "fusion",
    FeaturesStep: "features",
    RankerStep: "ranker",
    PostprocessStep: "postprocess",
    ServedStep: "served",
    RaisedStep: "raised",
    AttributionStep: "attribution",
}


@dataclass(frozen=True)
class QueryTrace:
    """Everything a traced ``recommend`` call did for one query, in the order it happened.

    Attributes
    ----------
    query : hashable
        The query, as given to ``recommend``.
    call : int
        Which outermost ``recommend`` call of the trace this is, from 0.
    steps : tuple of Step
        The steps, each naming its stage by ``path``. A query repeated within one call
        is traced once, by its first occurrence.
    """

    query: object
    call: int
    steps: tuple[Step, ...] = field(repr=False)

    def _of(self, kind: type[_StepT]) -> list[_StepT]:
        return [step for step in self.steps if isinstance(step, kind)]

    @property
    def path(self) -> str:
        """The path of the outermost estimator: where the call was made."""
        return self.steps[0].path if self.steps else ""

    @property
    def routes(self) -> list[RouteStep]:
        return self._of(RouteStep)

    @property
    def candidates(self) -> list[CandidatesStep]:
        return self._of(CandidatesStep)

    @property
    def fusion(self) -> list[FusionStep]:
        return self._of(FusionStep)

    @property
    def features(self) -> list[FeaturesStep]:
        return self._of(FeaturesStep)

    @property
    def ranker(self) -> list[RankerStep]:
        return self._of(RankerStep)

    @property
    def postprocess(self) -> list[PostprocessStep]:
        return self._of(PostprocessStep)

    @property
    def served(self) -> list[ServedStep]:
        return self._of(ServedStep)

    @property
    def raised(self) -> list[RaisedStep]:
        return self._of(RaisedStep)

    @property
    def attributions(self) -> list[AttributionStep]:
        return self._of(AttributionStep)

    @property
    def final(self) -> ServedStep | None:
        """What the outermost estimator served; None if the call raised."""
        return next((step for step in self.served if step.path == self.path), None)

    def to_dict(self) -> dict[str, object]:
        """The trace as JSON-compatible values."""
        return {
            "query": jsonable(self.query),
            "call": self.call,
            "steps": [step.to_dict() for step in self.steps],
        }

    @override
    def __str__(self) -> str:
        lines = [f"query {self.query!r}, call {self.call}"]
        lines += [f"  {step.path}: {step.describe()}" for step in self.steps]
        return "\n".join(lines)


class Trace(Tracer):
    """What the ``recommend`` calls made inside ``with`` did, for each query.

    Build one with :func:`trace`. Every ``recommend`` of a skrecsys estimator, and every
    stage of a composite, reports to the trace current in its context; the trace is not
    stored on any estimator, so the estimators clone and pickle as they would otherwise.

    See :class:`skrecsys._tracing.Tracer` for ``level`` and ``sample``.

    Attributes
    ----------
    events : list of events
        What was recorded, columnar: each event holds a block of queries.
    """

    def __init__(self, *, level: Level = "full", sample: Sample = None, n_reasons: int = 5) -> None:
        super().__init__(level=level, sample=sample, n_reasons=n_reasons)
        self._indexes: dict[int, dict[object, int]] = {}

    @property
    def n_calls(self) -> int:
        """How many outermost ``recommend`` calls were traced."""
        return self._call + 1

    @property
    def queries(self) -> list[object]:
        """The distinct queries traced, in the order they were first asked about."""
        seen: dict[object, None] = {}
        for event in self.events:
            if isinstance(event, Request) and "/" not in event.path:
                seen |= dict.fromkeys(event.queries.tolist())
        return list(seen)

    def __iter__(self) -> Iterator[QueryTrace]:
        """The trace of every query of every call."""
        for call in range(self.n_calls):
            outer = [
                e
                for e in self.events
                if e.call == call and isinstance(e, Request) and "/" not in e.path
            ]
            for event in outer:
                for query in dict.fromkeys(event.queries.tolist()):
                    yield self.query(query, call=call)

    def __getitem__(self, query: object) -> QueryTrace:
        """The trace of ``query`` in the latest call that asked about it."""
        return self.query(query)

    def query(self, query: object, *, call: int | None = None) -> QueryTrace:
        """The trace of ``query`` in ``call``, by default the latest that asked about it.

        Raises
        ------
        KeyError
            If no traced call asked about ``query``.
        """
        if isinstance(query, np.generic):
            # The trace keys queries as Python values, which a numpy scalar equals.
            query = query.item()
        calls = range(self.n_calls - 1, -1, -1) if call is None else [call]
        for number in calls:
            steps = self._steps(query, number)
            if steps:
                return QueryTrace(query, number, tuple(steps))
        raise KeyError(query)

    def _steps(self, query: object, call: int) -> list[Step]:
        steps: list[Step] = []
        asked = False
        for position, event in enumerate(self.events):
            if event.call != call:
                continue
            if isinstance(event, Raised):
                if asked:
                    steps.append(RaisedStep(event.path, event.error, event.message))
                continue
            step = self._step_of(position, event, query)
            if step is not None:
                asked = True
                steps.append(step)
        return steps

    def _step_of(self, position: int, event: Event, query: object) -> Step | None:
        """What ``event`` holds for ``query``, or None when it holds nothing."""
        if isinstance(event, Request | Served | Route):
            row = self._index(position, event.queries).get(query)
            return None if row is None else _row_step(event, row)
        if isinstance(event, Postprocess):
            before = self._group(position, event.before_pairs, event.before_groups, query)
            if before is None:
                return None
            after = self._group(-position - 1, event.after_pairs, event.after_groups, query)
            after = after or slice(0, 0)
            return PostprocessStep(
                event.path,
                event.before_pairs[before, 1],
                event.before_scores[before],
                event.after_pairs[after, 1],
                event.after_scores[after],
            )
        if isinstance(event, Candidates | Fusion | Features | RankerScores | Attribution):
            rows = self._group(position, event.pairs, event.groups, query)
            return None if rows is None else _group_step(event, rows)
        return None

    def _index(self, key: int, queries: NDArray[np.generic]) -> dict[object, int]:
        """Where each query first appears in a per-query array; built once per event."""
        index = self._indexes.get(key)
        if index is None:
            index = {}
            for row, query in enumerate(queries.tolist()):
                index.setdefault(query, row)
            self._indexes[key] = index
        return index

    def _group(
        self, key: int, pairs: NDArray[np.generic], groups: NDArray[np.int64], query: object
    ) -> slice | None:
        """The rows of the first group of ``query`` in a grouped array."""
        starts = np.cumsum(groups) - groups
        group = self._index(key, pairs[starts, 0] if len(groups) else pairs[:0, 0]).get(query)
        if group is None:
            return None
        start = int(starts[group])
        return slice(start, start + int(groups[group]))


def trace(*, level: Level = "full", sample: Sample = None, n_reasons: int = 5) -> Trace:
    """Trace the ``recommend`` calls made inside a ``with`` block.

    Parameters
    ----------
    level : {"full", "decisions"}, default="full"
        What to record. ``"decisions"`` records every stage's requests, routes, candidate
        lists, fusion contributions, ranker scores, postprocessing and served lists;
        ``"full"`` adds the feature matrices the rankers were given, what each feature
        added to a ranker's score, and why each leaf recommender scored what it served
        (:class:`AttributionStep`).
    sample : float, callable or None, default=None
        Which queries to record: all of them (``None``); a fraction, chosen by a hash of
        the identifier that is stable across processes; or those a callable, given the
        queries of an outermost ``recommend`` call, marks True in the boolean array it
        returns. A query is recorded through every stage or not at all.
    n_reasons : int, default=5
        How many history items to record behind each served item, with ``"full"``.

    Returns
    -------
    trace : Trace
        A context manager; index it by query once the block is done.

    Examples
    --------
    >>> from skrecsys.compose import KnownUser, Switch
    >>> from skrecsys.inspection import trace
    >>> from skrecsys.recommendation import ItemKNNRecommender, MostPopularRecommender
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "b"], ["u2", "c"], ["u3", "c"]]
    >>> rec = Switch(KnownUser(), ItemKNNRecommender(), MostPopularRecommender()).fit(X)
    >>> with trace() as t:
    ...     _ = rec.recommend(["u3", "new-user"], n_recommendations=1)
    >>> t["new-user"].routes[0].branch
    'on_false'
    >>> t["new-user"].final.items.tolist()
    ['b']
    """
    return Trace(level=level, sample=sample, n_reasons=n_reasons)


def _row_step(event: Request | Served | Route, row: int) -> Step:
    """The step of the query in row ``row`` of a per-query event."""
    if isinstance(event, Request):
        return RequestStep(event.path, event.params)
    if isinstance(event, Route):
        return RouteStep(event.path, event.condition, bool(event.mask[row]))
    return ServedStep(event.path, event.items[row], event.scores[row])


def _group_step(
    event: Candidates | Fusion | Features | RankerScores | Attribution, rows: slice
) -> Step:
    """The step of the query whose list is ``rows`` of a grouped event."""
    items = event.pairs[rows, 1]
    if isinstance(event, Attribution):
        return AttributionStep(
            event.path,
            event.kind,
            event.exact,
            items,
            event.reason_items[rows],
            event.weights[rows],
            event.rest[rows],
        )
    if isinstance(event, Candidates):
        return CandidatesStep(event.path, event.source, items, event.scores[rows])
    if isinstance(event, Fusion):
        return FusionStep(
            event.path, event.source, items, event.ranks[rows], event.contributions[rows]
        )
    if isinstance(event, Features):
        return FeaturesStep(event.path, items, event.names, event.values[rows])
    contributions = None if event.contributions is None else event.contributions[rows]
    positions = None if event.positions is None else event.positions[rows]
    return RankerStep(event.path, items, event.scores[rows], contributions, positions)


def jsonable(value: object) -> object:
    """``value`` as what :func:`json.dumps` writes as valid JSON.

    Arrays become lists, numpy scalars Python ones, times ISO strings and NaN or an
    infinity ``None``. A value with no JSON counterpart becomes its ``repr``.
    """
    if isinstance(value, np.ndarray):
        value = list(value.ravel()) if value.dtype.kind == "M" else value.tolist()
    if isinstance(value, np.datetime64):
        return None if np.isnat(value) else str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): jsonable(v) for key, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, datetime.datetime | datetime.date):
        value = value.isoformat()
    return value if value is None or isinstance(value, bool | int | str) else repr(value)


def _position(items: NDArray[np.generic], item: Hashable) -> int | None:
    found = np.flatnonzero(items == item)
    return int(found[0]) if len(found) else None


def _preview(items: NDArray[np.generic], scores: NDArray[np.floating] | None, n: int = 5) -> str:
    shown = []
    for j, item in enumerate(items[:n].tolist()):
        if scores is None or scores.ndim != 1:
            shown.append(repr(item))
        else:
            shown.append(f"{item!r} {scores[j]:.4g}")
    more = f", ... {len(items)} in all" if len(items) > n else ""
    return "[" + ", ".join(shown) + more + "]"
