"""The benchmark reports, as units of work and as blocks of the README.

A config implies a set of *units*, each one result: for the leaderboard, the sequential,
the reranking and the candidate generation report a model on a dataset, for the index
report a (model, index) pair on a dataset and one of its three tables. A unit's key is a
hash of everything the config says about it (see :mod:`spec`), so the set of units that
need running is simply the set whose stored key differs from the one the config computes
now.

Measuring and reporting meet here in both directions. :func:`measure` runs units and
stores what they found; :func:`blocks` reads every stored result back and lays it out as
the data a README block needs -- rows in table order, the settings the captions quote,
where each row was measured, and what is missing or stale -- for the templates to phrase.
"""

from __future__ import annotations

import sys
import traceback
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Literal, Protocol, TypedDict, TypeVar, cast

sys.path.insert(0, str(Path(__file__).resolve().parent))

import candidates
import indexes
import leaderboard
import reranking
import sequential
import spec
import store

from skrecsys.base import RecommenderMixin
from skrecsys.indexing import HNSW
from skrecsys.recommendation._base import BaseRecommender
from skrecsys.typing import Recommender

#: The settings each kind of unit is keyed on. A setting outside a unit's list cannot
#: change its result, so it must not invalidate it either: widening the latency batches
#: should re-run the latency table and leave the hour-long sweep alone.
_SETTINGS: dict[tuple[str, str | None], tuple[str, ...]] = {
    ("leaderboard", None): ("k", "repeat", "rank_repeat", "budget", "warmup"),
    ("sequential", None): ("cutoffs", "repeat", "rank_repeat", "budget", "warmup"),
    ("reranking", None): ("k", "latency_repeat", "budget", "warmup"),
    ("candidates", None): ("n_retrieved", "latency_repeat", "budget", "warmup"),
    ("indexes", "sweep"): ("k", "repeat", "rank_repeat", "budget", "warmup"),
    ("indexes", "scaling"): ("k", "repeat", "rank_repeat", "budget", "warmup", "catalog_scale"),
    ("indexes", "latency"): ("k", "latency_repeat", "budget", "warmup", "latency_batch"),
}


def command(benchmark: str, dataset: str) -> str:
    """The command that brings one block of the report up to date."""
    return f"python benchmarks/run.py run {benchmark} --dataset {dataset}"


def _inputs(
    config: spec.Config,
    target: spec.Target,
    model: spec.Entry,
    index: spec.IndexEntry | None = None,
    kind: str | None = None,
) -> dict[str, spec.JSON]:
    """Everything a unit's result depends on, which is what its key hashes."""
    names = _SETTINGS[config.benchmark, kind]
    inputs: dict[str, spec.JSON] = {
        "schema": spec.SCHEMA,
        "benchmark": config.benchmark,
        "dataset": target.definition.spec(),
        "settings": {name: target.raw_settings[name] for name in names},
        "model": model.spec(),
    }
    if index is not None:
        # The single-setting tables run at the dial's middle value only, so the rest of
        # the dial must not be able to invalidate them.
        inputs["index"] = (index if kind == "sweep" else indexes._at_middle(index)).spec()
        inputs["kind"] = kind
    return inputs


def units(config: spec.Config, dataset: str) -> list[store.Unit]:
    """Every result ``config`` implies on ``dataset``, in report order."""
    if config.benchmark == "indexes":
        return list(_index_units(config, dataset))
    target = config.targets[dataset]
    return [
        store.Unit(config.benchmark, dataset, model.name, _inputs(config, target, model))
        for model in config.active_models(dataset)
    ]


def _index_units(config: spec.Config, dataset: str) -> list[store.IndexUnit]:
    """The units of the index report on ``dataset``, in report order."""
    target = config.targets[dataset]
    found: list[store.IndexUnit] = []
    for model in config.active_models(dataset):
        for kind in indexes.KINDS:
            if kind == "latency" and not target.settings.get("latency_batch"):
                continue
            if kind == "scaling" and not target.settings.get("catalog_scale"):
                continue
            found.extend(
                store.IndexUnit(
                    config.benchmark,
                    dataset,
                    model.name,
                    _inputs(config, target, model, index, kind),
                    index=index.name,
                    kind=kind,
                )
                for index in config.active_indexes(dataset)
            )
    return found


# --- Measuring --------------------------------------------------------------------------


R = TypeVar("R")


class UnavailableError(RuntimeError):
    """A unit that cannot be measured on this host, such as a model needing an extra."""


def _build(entry: spec.Entry, kind: type[R]) -> R:
    """Build ``entry`` as a ``kind``: any recommender for a row, a single model for an index."""
    try:
        return entry.build(kind)
    except ImportError as error:
        raise UnavailableError(f"{entry.name} needs {error.name or 'an extra'} ({error})") from None


class Sink(Protocol):
    """Where a measured result goes: ``store.write`` by default, or anything with its shape.

    It returns where the result went, if it went anywhere a user can look.
    """

    def __call__(
        self, unit: store.Unit, payload: store.Payload, *, host: leaderboard.HostFacts
    ) -> Path | None: ...


def measure(
    config: spec.Config,
    dataset: str,
    todo: Sequence[store.Unit],
    log: Callable[[str], None] = print,
    sink: Sink = store.write,
) -> list[tuple[store.Unit, str]]:
    """Run ``todo`` and hand each result to ``sink`` as soon as it exists.

    A result is stored the moment its unit finishes, so a long run that fails on its
    last model keeps everything it measured before. A unit that fails is logged and
    returned rather than raised, for the same reason.
    """
    if not todo:
        return []
    target = config.targets[dataset]
    log(f"loading {dataset} ...")
    data = target.definition.load()
    failures: list[tuple[store.Unit, str]] = []
    jobs: list[tuple[Sequence[store.Unit], Callable[[], None]]]
    if config.benchmark == "indexes":
        jobs = [
            (group, partial(_run_indexes, config, target, data, group, log, sink))
            for group in _index_groups(todo)
        ]
    elif config.benchmark == "reranking":
        jobs = [
            ([unit], partial(_run_reranking, config, target, data, unit, log, sink))
            for unit in todo
        ]
    elif config.benchmark == "candidates":
        jobs = [
            ([unit], partial(_run_candidates, config, target, data, unit, log, sink))
            for unit in todo
        ]
    else:
        jobs = [([unit], partial(_run_row, config, target, data, unit, log, sink)) for unit in todo]
    for group, job in jobs:
        try:
            job()
        except UnavailableError as reason:
            log(f"  skipped: {reason}")
            failures.extend((unit, str(reason)) for unit in group)
        except Exception:
            detail = traceback.format_exc()
            log(detail)
            failures.extend((unit, detail.strip().splitlines()[-1]) for unit in group)
    return failures


def _run_row(
    config: spec.Config,
    target: spec.Target,
    data: spec.Split,
    unit: store.Unit,
    log: Callable[[str], None],
    sink: Sink,
) -> None:
    """The leaderboard and the sequential report: one row per model, one unit each."""
    settings = target.settings
    log(f"fitting {unit.model} ...")
    # The mixin marks a recommender at runtime; its estimator half comes from
    # BaseEstimator, which the protocol spells out and the mixin alone does not.
    estimator = cast(Recommender, _build(config.model(unit.model), RecommenderMixin))
    sequential_report = config.benchmark == "sequential"
    # One ranking per model, as deep as the deepest cutoff; the metrics truncate it.
    k = max(settings["cutoffs"]) if sequential_report else settings["k"]
    evaluation = leaderboard.evaluate(
        estimator,
        data,
        k,
        settings["repeat"],
        settings["rank_repeat"],
        settings["budget"],
        target.definition.max_eval_users,
        settings["warmup"],
    )
    quality = (
        sequential.score(evaluation, settings["cutoffs"])
        if sequential_report
        else leaderboard.score(evaluation, k)
    )
    payload: store.RowPayload = {
        "quality": {"Model": unit.model} | quality,
        "timing": {"Model": unit.model} | leaderboard.timings(evaluation),
    }
    host = leaderboard.host_facts()
    if sequential_report:
        host["device"] = sequential.device_facts()
    _report(log, sink(unit, payload, host=host))


def _run_reranking(
    config: spec.Config,
    target: spec.Target,
    data: spec.Split,
    unit: store.Unit,
    log: Callable[[str], None],
    sink: Sink,
) -> None:
    """The reranking report: one pipeline per unit, built from the dataset it runs on."""
    log(f"fitting {unit.model} ...")
    try:
        pipeline = reranking.build(config.model(unit.model), data)
    except ImportError as error:
        raise UnavailableError(f"{unit.model} needs {error.name or 'an extra'} ({error})") from None
    payload = reranking.measure(unit.model, pipeline, data, target.settings)
    _report(log, sink(unit, payload, host=leaderboard.host_facts()))


def _run_candidates(
    config: spec.Config,
    target: spec.Target,
    data: spec.Split,
    unit: store.Unit,
    log: Callable[[str], None],
    sink: Sink,
) -> None:
    """The candidate generation report: one generator set per unit."""
    log(f"retrieving with {unit.model} ...")
    generators = candidates.build(config.model(unit.model))
    payload = candidates.measure(unit.model, generators, data, target.settings)
    _report(log, sink(unit, payload, host=leaderboard.host_facts()))


def _index_groups(todo: Sequence[store.Unit]) -> list[list[store.IndexUnit]]:
    """The index report's units by model and table: one fit serves each group."""
    grouped: dict[tuple[str, str], list[store.IndexUnit]] = defaultdict(list)
    for unit in todo:
        if not isinstance(unit, store.IndexUnit):
            raise TypeError(f"{unit.label} is not a unit of the index report.")
        grouped[unit.model, unit.kind].append(unit)
    return list(grouped.values())


def _run_indexes(
    config: spec.Config,
    target: spec.Target,
    data: spec.Split,
    group: Sequence[store.IndexUnit],
    log: Callable[[str], None],
    sink: Sink,
) -> None:
    """The index report: a model is fitted once for every stale index of one table."""
    model, kind = group[0].model, group[0].kind
    by_name = {index.name: index for index in config.active_indexes(target.name)}
    chosen = [by_name[unit.index] for unit in group]
    log(f"{kind}: {model} with {', '.join(index.name for index in chosen)} ...")
    estimator = _build(config.model(model), BaseRecommender)
    settings, max_eval_users = target.settings, target.definition.max_eval_users
    payloads: Mapping[str, store.Payload]
    if kind == "sweep":
        payloads = indexes.sweep_payloads(model, estimator, data, chosen, settings, max_eval_users)
    elif kind == "scaling":
        payloads = indexes.scaling_payloads(estimator, data, chosen, settings, max_eval_users)
    else:
        payloads = indexes.latency_payloads(estimator, data, chosen, settings)
    host = leaderboard.host_facts()
    for unit in group:
        _report(log, sink(unit, payloads[unit.index], host=host))


def _report(log: Callable[[str], None], where: Path | None) -> None:
    """Say where a result went, when the sink said: a stored one names its file."""
    if where is not None:
        log(f"  wrote {where.relative_to(store.RESULTS_DIR.parent.parent)}")


# --- Reporting --------------------------------------------------------------------------


class _HostGroup(TypedDict):
    facts: dict[str, spec.JSON]
    models: list[str]


class _IndexGroup(TypedDict):
    index: str
    models: list[str]


class RowsBlock(TypedDict):
    """What the template of a leaderboard or sequential block reads."""

    benchmark: str
    dataset: spec.DatasetDef
    settings: spec.Settings
    skip: Mapping[str, str]
    command: str
    quality: list[store.Cells]
    timing: list[store.Cells]
    hosts: list[_HostGroup]
    missing: list[str]
    stale: list[str]


class IndexesBlock(TypedDict):
    """What the template of an index report block reads."""

    benchmark: str
    dataset: spec.DatasetDef
    settings: spec.Settings
    skip: Mapping[str, str]
    command: str
    quality: list[store.Cells]
    cost: list[store.Cells]
    latency: list[store.Cells]
    scaling: list[store.Cells]
    n_items: int | None
    min_index_size: int
    hosts: list[_HostGroup]
    missing: list[_IndexGroup]
    stale: list[_IndexGroup]
    total_models: int


class RerankingBlock(TypedDict):
    """What the template of a reranking block reads."""

    benchmark: str
    dataset: spec.DatasetDef
    settings: spec.Settings
    command: str
    #: The pipelines, in config order, which is the order of every table.
    pipelines: list[str]
    quality: list[store.Cells]
    latency: list[store.Cells]
    examples: list[store.Cells]
    #: How much of each example list is shown.
    example_depth: int
    hosts: list[_HostGroup]
    missing: list[str]
    stale: list[str]


class CandidatesBlock(TypedDict):
    """What the template of a candidate generation block reads."""

    benchmark: str
    dataset: spec.DatasetDef
    settings: spec.Settings
    command: str
    #: One row per generator set, in config order.
    recall: list[store.Cells]
    latency: list[store.Cells]
    hosts: list[_HostGroup]
    missing: list[str]
    stale: list[str]


class Report(TypedDict):
    """Every block of the README, by benchmark and then by dataset."""

    leaderboard: dict[str, RowsBlock]
    sequential: dict[str, RowsBlock]
    reranking: dict[str, RerankingBlock]
    candidates: dict[str, CandidatesBlock]
    indexes: dict[str, IndexesBlock]


def _hosts(results: Iterable[tuple[str, store.Result]]) -> list[_HostGroup]:
    """Where the rows of a block were measured, one entry per distinct host.

    A block measured in one run names one host, as it always has. Once rows are
    re-measured separately they can come from different machines, and timings are only
    comparable within one, so the block then says which rows ran where.
    """
    groups: dict[str, _HostGroup] = {}
    for model, result in results:
        host = result["host"]
        entry = groups.setdefault(spec.canonical(host), {"facts": host, "models": []})
        if model not in entry["models"]:
            entry["models"].append(model)
    return list(groups.values())


_U = TypeVar("_U", bound=store.Unit)


def _status(block: Iterable[_U]) -> tuple[dict[_U, store.Result], list[_U], list[_U]]:
    """The results of a block, and which of its units are missing and which are stale."""
    found: dict[_U, store.Result] = {}
    missing: list[_U] = []
    stale: list[_U] = []
    for unit in block:
        result = store.read(unit)
        if result is None:
            missing.append(unit)
            continue
        found[unit] = result
        if result["key"] != unit.key:
            stale.append(unit)
    return found, missing, stale


def rank_column(benchmark: str, settings: spec.Settings) -> str:
    """The quality column a row report is ranked by, best first."""
    if benchmark == "sequential":
        return f"HR@{settings['cutoffs'][0]}"
    return f"NDCG@{settings['k']}"


def _rows_block(config: spec.Config, dataset: str) -> RowsBlock:
    target = config.targets[dataset]
    found, missing, stale = _status(units(config, dataset))
    settings = target.settings
    sort_by = rank_column(config.benchmark, settings)
    rows = {unit: store.row_payload(result) for unit, result in found.items()}
    # Stable, so models that tie keep the config's order -- as a single run's table did.
    ordered = sorted(
        rows.items(), key=lambda item: float(item[1]["quality"][sort_by]), reverse=True
    )
    return {
        "benchmark": config.benchmark,
        "dataset": target.definition,
        "settings": settings,
        "skip": target.skip,
        "command": command(config.benchmark, dataset),
        "quality": [payload["quality"] for _, payload in ordered],
        "timing": [payload["timing"] for _, payload in ordered],
        "hosts": _hosts((unit.model, found[unit]) for unit, _ in ordered),
        "missing": [unit.model for unit in missing],
        "stale": [unit.model for unit in stale],
    }


def _reranking_block(config: spec.Config, dataset: str) -> RerankingBlock:
    """The pipelines side by side: each table grouped by segment, pipelines in config order.

    A result is one pipeline, so the tables interleave them -- all users for every
    pipeline, then the warm ones, then the cold ones -- which puts the rows a reader
    compares next to each other. The examples become one row per user and one column per
    pipeline, for the same reason.
    """
    target = config.targets[dataset]
    found, missing, stale = _status(units(config, dataset))
    payloads = {unit.model: store.reranking_payload(result) for unit, result in found.items()}
    pipelines = [entry.name for entry in config.active_models(dataset) if entry.name in payloads]

    def interleaved(table: Literal["quality", "latency"]) -> list[store.Cells]:
        rows = [row for name in pipelines for row in payloads[name][table]]
        order = {segment: position for position, segment in enumerate(reranking.SEGMENTS)}
        return sorted(rows, key=lambda row: order[row["users"]])

    examples: dict[str, store.Cells] = {}
    for name in pipelines:
        for example in payloads[name]["examples"]:
            row = examples.setdefault(
                example["user"],
                {
                    "cold user": example["user"],
                    "age, gender, occupation": example["profile"],
                    "held out": example["held out"],
                },
            )
            row[name] = example["top"]
    return {
        "benchmark": config.benchmark,
        "dataset": target.definition,
        "settings": target.settings,
        "command": command(config.benchmark, dataset),
        "pipelines": pipelines,
        "quality": interleaved("quality"),
        "latency": interleaved("latency"),
        "examples": list(examples.values()),
        "example_depth": reranking.EXAMPLE_DEPTH,
        "hosts": _hosts((unit.model, result) for unit, result in found.items()),
        "missing": [unit.model for unit in missing],
        "stale": [unit.model for unit in stale],
    }


def _candidates_block(config: spec.Config, dataset: str) -> CandidatesBlock:
    """The generator sets in config order, which puts each set after its members."""
    target = config.targets[dataset]
    found, missing, stale = _status(units(config, dataset))
    payloads = {unit.model: store.candidates_payload(result) for unit, result in found.items()}
    sets = [entry.name for entry in config.active_models(dataset) if entry.name in payloads]
    return {
        "benchmark": config.benchmark,
        "dataset": target.definition,
        "settings": target.settings,
        "command": command(config.benchmark, dataset),
        "recall": [payloads[name]["recall"] for name in sets],
        "latency": [payloads[name]["latency"] for name in sets],
        "hosts": _hosts((unit.model, result) for unit, result in found.items()),
        "missing": [unit.model for unit in missing],
        "stale": [unit.model for unit in stale],
    }


def _indexes_block(config: spec.Config, dataset: str) -> IndexesBlock:
    target = config.targets[dataset]
    found, missing, stale = _status(_index_units(config, dataset))
    models = [entry.name for entry in config.active_models(dataset)]
    active = config.active_indexes(dataset)
    sweep: dict[tuple[str, str], store.SweepPayload] = {}
    series: dict[str, dict[tuple[str, str], store.SeriesPayload]] = {
        "latency": {},
        "scaling": {},
    }
    for unit, result in found.items():
        if unit.kind == "sweep":
            sweep[unit.model, unit.index] = store.sweep_payload(result)
        else:
            series[unit.kind][unit.model, unit.index] = store.series_payload(result)
    quality, cost = indexes.sweep_tables(models, active, sweep)
    sizes = {payload["n_items"] for payload in sweep.values()}
    labels = {index.name: index.label for index in active}
    return {
        "benchmark": config.benchmark,
        "dataset": target.definition,
        "settings": target.settings,
        "skip": target.skip,
        "command": command(config.benchmark, dataset),
        "quality": quality,
        "cost": cost,
        "latency": indexes.latency_table(models, active, series["latency"]),
        "scaling": indexes.scaling_table(models, active, series["scaling"]),
        # Every sweep of a dataset runs on the same catalog, so there is one size.
        "n_items": sizes.pop() if len(sizes) == 1 else None,
        "min_index_size": HNSW().min_index_size,
        "hosts": _hosts((unit.model, result) for unit, result in found.items()),
        "missing": _by_index(missing, labels, models),
        "stale": _by_index(stale, labels, models),
        "total_models": len(models),
    }


def _by_index(
    found: Sequence[store.IndexUnit], labels: Mapping[str, str], models: Sequence[str]
) -> list[_IndexGroup]:
    """Units grouped under their index, the models in config order, for one sentence each."""
    grouped: dict[str, set[str]] = defaultdict(set)
    for unit in found:
        grouped[unit.index].add(unit.model)
    return [
        {"index": labels[name], "models": [model for model in models if model in chosen]}
        for name, chosen in grouped.items()
    ]


def blocks() -> Report:
    """Every block of the report, by benchmark and then by dataset, for the templates."""

    def rows(benchmark: str) -> dict[str, RowsBlock]:
        config = spec.load(benchmark)
        return {dataset: _rows_block(config, dataset) for dataset in config.targets}

    config = spec.load("indexes")
    reranking_config = spec.load("reranking")
    candidates_config = spec.load("candidates")
    return {
        "leaderboard": rows("leaderboard"),
        "sequential": rows("sequential"),
        "reranking": {
            dataset: _reranking_block(reranking_config, dataset)
            for dataset in reranking_config.targets
        },
        "candidates": {
            dataset: _candidates_block(candidates_config, dataset)
            for dataset in candidates_config.targets
        },
        "indexes": {dataset: _indexes_block(config, dataset) for dataset in config.targets},
    }
