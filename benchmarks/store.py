"""Benchmark results on disk: one committed JSON file per result.

A result is the measurement of one *unit* -- a model on a dataset, or for the index
report one (model, index) pair and one of its tables -- together with the key it was
measured under. The file is named after the unit rather than the key, so each unit has
exactly one current result, a re-run overwrites it, and a diff in review shows which
numbers moved rather than a file appearing and another disappearing.

A result is **fresh** when its stored key equals the key the config now computes,
**stale** when a result exists under some other key, and **missing** when there is no
file. A stale result is still a real measurement and is still reported, with a note;
it is only the run that treats it as work to do.

Results hold table cells as they are printed, not raw numbers. The rendered README is
then a function of the stored cells alone, so rendering is deterministic, needs no
recomputation, and reproduces a table exactly -- which is also what let the results the
README already carried be imported rather than re-measured.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, TypedDict, cast

import spec

from skrecsys.typing import override

if TYPE_CHECKING:
    from leaderboard import HostFacts

RESULTS_DIR = Path(__file__).resolve().parent / "results"

FRESH, STALE, MISSING = "fresh", "stale", "missing"

#: One table row as printed: column header to cell.
Cells: TypeAlias = dict[str, str]


class RowPayload(TypedDict):
    """A leaderboard or sequential result: one model's quality and timing rows."""

    quality: Cells
    timing: Cells


class SweepRow(TypedDict):
    """One configuration of the index sweep: what it answered and what it cost."""

    quality: Cells
    cost: Cells


class SweepPayload(TypedDict):
    """An index sweep result, carrying the exact baseline it was measured against."""

    n_items: int
    exact: SweepRow
    rows: list[SweepRow]


class SeriesPayload(TypedDict):
    """A latency or scaling result: one point per request size or catalog size."""

    points: list[Cells]


class RerankingPayload(TypedDict):
    """A reranking result: one pipeline's quality per segment, its per-request latency,
    and its lists for a few cold users."""

    quality: list[Cells]
    latency: list[Cells]
    examples: list[Cells]


class CandidatesPayload(TypedDict):
    """A candidate generation result: one generator set's ceiling and request time, one
    cell per budget."""

    recall: Cells
    latency: Cells


#: What a result stores, by kind of unit.
Payload: TypeAlias = (
    RowPayload | SweepPayload | SeriesPayload | RerankingPayload | CandidatesPayload
)


class Provenance(TypedDict):
    """Where a stored result came from: measured here, or imported from elsewhere."""

    source: str
    at: str


class Result(TypedDict):
    """A stored result, as :func:`write` lays it out."""

    key: str
    unit: str
    provenance: dict[str, spec.JSON]
    host: dict[str, spec.JSON]
    inputs: dict[str, spec.JSON]
    payload: dict[str, spec.JSON]


@dataclass(frozen=True)
class Unit:
    """One result's worth of work: where it lives and what it is keyed on."""

    benchmark: str
    dataset: str
    model: str
    #: What the result depends on. Not part of the unit's identity: a unit is where its
    #: result lives, and the inputs are what that result is currently checked against.
    inputs: Mapping[str, spec.JSON] = field(compare=False, hash=False)

    @property
    def key(self) -> str:
        return spec.key(self.inputs)

    @property
    def path(self) -> Path:
        return RESULTS_DIR / self.benchmark / self.dataset / f"{self.model}.json"

    @property
    def label(self) -> str:
        """How a status line or an error names the unit."""
        return f"{self.benchmark}/{self.dataset}/{self.model}"


@dataclass(frozen=True, kw_only=True)
class IndexUnit(Unit):
    """A unit of the index report: one (model, index) pair in one of its tables."""

    index: str
    #: Which of the index report's tables.
    kind: str

    @property
    @override
    def path(self) -> Path:
        base = RESULTS_DIR / self.benchmark / self.dataset
        return base / self.kind / f"{self.model}.{self.index}.json"

    @property
    @override
    def label(self) -> str:
        return f"{super().label}/{self.index}/{self.kind}"


def read(unit: Unit) -> Result | None:
    """The stored result of ``unit``, or ``None`` when it has never been measured."""
    try:
        text = unit.path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    # Written by `write` alone, so its layout is known; which payload it holds is for
    # the reader below that matches the unit's kind to say.
    return cast(Result, spec.parse_json(text))


def row_payload(result: Result) -> RowPayload:
    """The payload of a leaderboard or sequential result."""
    return cast(RowPayload, result["payload"])


def reranking_payload(result: Result) -> RerankingPayload:
    """The payload of a reranking result."""
    return cast(RerankingPayload, result["payload"])


def candidates_payload(result: Result) -> CandidatesPayload:
    """The payload of a candidate generation result."""
    return cast(CandidatesPayload, result["payload"])


def sweep_payload(result: Result) -> SweepPayload:
    """The payload of an index sweep result."""
    return cast(SweepPayload, result["payload"])


def series_payload(result: Result) -> SeriesPayload:
    """The payload of an index latency or scaling result."""
    return cast(SeriesPayload, result["payload"])


def state(unit: Unit) -> str:
    """Whether ``unit``'s result is fresh, stale or missing."""
    result = read(unit)
    if result is None:
        return MISSING
    return FRESH if result.get("key") == unit.key else STALE


def write(
    unit: Unit,
    payload: Payload,
    *,
    host: HostFacts | Mapping[str, spec.JSON],
    provenance: Provenance | None = None,
    inputs: Mapping[str, spec.JSON] | None = None,
) -> Path:
    """Store ``payload`` as ``unit``'s result, under the key of ``inputs``.

    ``inputs`` defaults to the unit's own, which is what a fresh measurement is keyed
    on. An import passes the inputs the numbers were really produced under instead, so
    a result measured with an older version of an entry is stored as stale rather than
    passed off as current.
    """
    inputs = dict(unit.inputs if inputs is None else inputs)
    result = {
        "key": spec.key(inputs),
        "unit": unit.label,
        "provenance": dict(provenance or {"source": "measured", "at": _now()}),
        "host": dict(host),
        "inputs": inputs,
        "payload": payload,
    }
    unit.path.parent.mkdir(parents=True, exist_ok=True)
    unit.path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return unit.path


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
