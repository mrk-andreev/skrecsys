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
from typing import Any

import spec

from skrecsys._typing import override

RESULTS_DIR = Path(__file__).resolve().parent / "results"

FRESH, STALE, MISSING = "fresh", "stale", "missing"


@dataclass(frozen=True)
class Unit:
    """One result's worth of work: where it lives and what it is keyed on."""

    benchmark: str
    dataset: str
    model: str
    #: What the result depends on. Not part of the unit's identity: a unit is where its
    #: result lives, and the inputs are what that result is currently checked against.
    inputs: Mapping[str, Any] = field(compare=False, hash=False)

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


def read(unit: Unit) -> dict[str, Any] | None:
    """The stored result of ``unit``, or ``None`` when it has never been measured."""
    try:
        return json.loads(unit.path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def state(unit: Unit) -> str:
    """Whether ``unit``'s result is fresh, stale or missing."""
    result = read(unit)
    if result is None:
        return MISSING
    return FRESH if result.get("key") == unit.key else STALE


def write(
    unit: Unit,
    payload: Mapping[str, Any],
    *,
    host: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
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
