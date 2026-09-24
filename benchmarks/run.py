#!/usr/bin/env python
"""Run the benchmarks the configs describe, and render the README from their results.

    python benchmarks/run.py status                          # what is fresh, stale, missing
    python benchmarks/run.py run leaderboard                 # measure what is out of date
    python benchmarks/run.py run indexes --dataset amazon-books --only ALS
    python benchmarks/run.py run leaderboard --dry-run       # say what would run
    python benchmarks/run.py render                          # rewrite README.md
    python benchmarks/run.py render --check                  # fail if README.md is stale

What runs is ``benchmarks/config/<benchmark>.json``: each entry names a class, its
parameters and a version, and each result in ``benchmarks/results`` is keyed on all of
that. ``run`` measures only the units whose key has changed -- a new model, an edited
parameter, a bumped version -- and leaves every other number alone, then re-renders the
README. To add a model, add an entry and run; to re-measure one after changing code the
config cannot see, bump its ``version``.

``--no-store`` measures without keeping anything, which is what an exploratory run and
the profile-guided build's training run want. It is also the only way to pass ``--set``:
a result measured under settings the config does not state could not honestly be stored
as the config's.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import indexes
import leaderboard
import readme
import spec
import store
import suite

_MARK = {store.FRESH: " ", store.STALE: "~", store.MISSING: "+"}


def _override(value: str) -> tuple[str, Any]:
    """Parse ``name=value``, reading the value as JSON so numbers and lists survive."""
    name, sep, raw = value.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"expected name=value, got {value!r}")
    try:
        return name, json.loads(raw)
    except json.JSONDecodeError:
        return name, raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="list every result and whether it is current")
    status.add_argument(
        "benchmarks", nargs="*", metavar="BENCHMARK", help=f"any of {', '.join(spec.BENCHMARKS)}"
    )
    status.add_argument("--dataset", action="append", help="limit to these datasets")
    status.add_argument("--all", action="store_true", help="list fresh results too")

    run = commands.add_parser("run", help="measure what is out of date, then re-render")
    run.add_argument("benchmark", choices=spec.BENCHMARKS)
    run.add_argument("--dataset", action="append", help="limit to these datasets")
    run.add_argument("--only", nargs="+", metavar="MODEL", help="limit to these models")
    run.add_argument("--index", nargs="+", metavar="INDEX", help="limit to these indexes")
    run.add_argument("--kind", nargs="+", choices=indexes.KINDS, help="limit to these tables")
    run.add_argument("--force", action="store_true", help="re-measure fresh results too")
    run.add_argument("--dry-run", action="store_true", help="list what would run, run nothing")
    run.add_argument("--no-store", action="store_true", help="measure and print, keep nothing")
    run.add_argument("--no-render", action="store_true", help="leave README.md as it is")
    run.add_argument(
        "--set",
        dest="overrides",
        action="append",
        type=_override,
        default=[],
        metavar="NAME=VALUE",
        help="override a setting for this run; needs --no-store",
    )

    render = commands.add_parser("render", help="write README.md from the template and results")
    render.add_argument("--check", action="store_true", help="exit 1 if README.md is out of date")
    return parser


def _datasets(config: spec.Config, wanted: Sequence[str] | None) -> list[str]:
    if not wanted:
        return list(config.targets)
    unknown = sorted(set(wanted) - set(config.targets))
    if unknown:
        raise SystemExit(
            f"{config.benchmark} has no dataset {unknown}; choose from {sorted(config.targets)}."
        )
    return list(wanted)


def _status(args: argparse.Namespace) -> int:
    unknown = sorted(set(args.benchmarks) - set(spec.BENCHMARKS))
    if unknown:
        raise SystemExit(f"no benchmark {unknown}; choose from {list(spec.BENCHMARKS)}.")
    counts = {store.FRESH: 0, store.STALE: 0, store.MISSING: 0}
    for benchmark in args.benchmarks or spec.BENCHMARKS:
        config = spec.load(benchmark)
        for dataset in _datasets(config, args.dataset):
            for unit in suite.units(config, dataset):
                state = store.state(unit)
                counts[state] += 1
                if args.all or state != store.FRESH:
                    print(f"{_MARK[state]} {state:<7} {unit.label}")
    print(
        f"{counts[store.FRESH]} fresh, {counts[store.STALE]} stale, "
        f"{counts[store.MISSING]} missing  (~ stale, + missing)"
    )
    return 0


def _with_overrides(config: spec.Config, overrides: Sequence[tuple[str, Any]]) -> spec.Config:
    """``config`` with every dataset's settings overridden for this run only."""
    if not overrides:
        return config
    targets = {}
    for name, target in config.targets.items():
        settings = {**target.settings, **dict(overrides)}
        spec.check_settings(config.benchmark, f"--set on {name}", settings)
        targets[name] = dataclasses.replace(target, settings=settings)
    return dataclasses.replace(config, targets=targets)


def _selected(args: argparse.Namespace, unit: store.Unit) -> bool:
    if args.only and unit.model not in args.only:
        return False
    if not isinstance(unit, store.IndexUnit):
        return not args.index and not args.kind
    return (not args.index or unit.index in args.index) and (
        not args.kind or unit.kind in args.kind
    )


class _Collector:
    """A sink that keeps results in memory, for ``--no-store``."""

    def __init__(self) -> None:
        self.results: dict[store.Unit, dict[str, Any]] = {}

    def __call__(self, unit: store.Unit, payload: dict[str, Any], *, host: Any) -> None:
        self.results[unit] = payload


def _print_collected(config: spec.Config, dataset: str, collected: _Collector) -> None:
    """Show what a ``--no-store`` run measured, in the tables the README would use."""
    found = {
        unit: payload for unit, payload in collected.results.items() if unit.dataset == dataset
    }
    if not found:
        return
    if config.benchmark != "indexes":
        column = suite.rank_column(config.benchmark, config.targets[dataset].settings)
        rows = sorted(found.values(), key=lambda row: float(row["quality"][column]), reverse=True)
        print(leaderboard.render_box([row["quality"] for row in rows]))
        print(leaderboard.render_box([row["timing"] for row in rows]))
        return
    models = [entry.name for entry in config.active_models(dataset)]
    active = config.active_indexes(dataset)
    by_kind: dict[str, dict[tuple[str, str], Any]] = {kind: {} for kind in indexes.KINDS}
    for unit, payload in found.items():
        if isinstance(unit, store.IndexUnit):
            by_kind[unit.kind][unit.model, unit.index] = payload
    for table in (
        *indexes.sweep_tables(models, active, by_kind["sweep"]),
        indexes.latency_table(models, active, by_kind["latency"]),
        indexes.scaling_table(models, active, by_kind["scaling"]),
    ):
        if table:
            print(leaderboard.render_box(table))


def _run(args: argparse.Namespace) -> int:
    if args.overrides and not args.no_store:
        raise SystemExit(
            "--set changes what a result is measured under, and a result has to be stored "
            "under the config that produced it. Edit the config to change what is stored, "
            "or add --no-store to measure without keeping anything."
        )
    config = _with_overrides(spec.load(args.benchmark), args.overrides)
    collector = _Collector() if args.no_store else None
    failures: list[tuple[store.Unit, str]] = []
    measured = 0
    for dataset in _datasets(config, args.dataset):
        chosen = [unit for unit in suite.units(config, dataset) if _selected(args, unit)]
        todo = [unit for unit in chosen if args.force or store.state(unit) != store.FRESH]
        print(f"{config.benchmark} / {dataset}: {len(todo)} of {len(chosen)} to measure")
        for unit in todo:
            state = store.state(unit)
            print(f"  {_MARK[state]} {unit.label}")
        if args.dry_run or not todo:
            continue
        sink = collector if collector is not None else store.write
        failures += suite.measure(config, dataset, todo, sink=sink)
        measured += len(todo)
        if collector is not None:
            _print_collected(config, dataset, collector)

    if failures:
        print(f"\n{len(failures)} unit(s) failed:", file=sys.stderr)
        for unit, reason in failures:
            print(f"  {unit.label}: {reason}", file=sys.stderr)
    if measured and not args.no_store and not args.no_render and not args.dry_run:
        changed = readme.write()
        print("README.md re-rendered." if changed else "README.md already current.")
    return 1 if failures else 0


def _render(args: argparse.Namespace) -> int:
    if args.check:
        if readme.is_current():
            print("README.md is current.")
            return 0
        print(
            "README.md differs from what README.md.j2 and benchmarks/results render to; "
            "run `python benchmarks/run.py render`.",
            file=sys.stderr,
        )
        return 1
    changed = readme.write()
    print("README.md re-rendered." if changed else "README.md already current.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            return _status(args)
        if args.command == "run":
            return _run(args)
        return _render(args)
    except spec.ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
