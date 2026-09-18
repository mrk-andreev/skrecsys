#!/usr/bin/env python
"""Fit every recommender on MovieLens 100K and print a leaderboard.

    uv run python benchmarks/leaderboard.py
    uv run python benchmarks/leaderboard.py --k 20 --format markdown
    uv run python benchmarks/leaderboard.py --write-readme

Each model is fitted on the training half of an official split and evaluated on the
held-out half with every metric in :mod:`skrecsys.metrics`, so the table mixes ranking
quality with the beyond-accuracy metrics that explain it: a model can win on NDCG while
recommending nothing but the head of the catalog.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone

import skrecsys
from skrecsys import _core
from skrecsys.datasets import fetch_movielens_100k
from skrecsys.metrics import (
    average_precision_at_k,
    catalog_coverage_at_k,
    hit_rate_at_k,
    item_popularity,
    mean_popularity_at_k,
    ndcg_at_k,
    novelty_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    user_coverage_at_k,
)
from skrecsys.recommendation import (
    EASE,
    AlternatingLeastSquares,
    BayesianPersonalizedRanking,
    BM25Recommender,
    ItemKNNRecommender,
    MostPopularRecommender,
    RP3Beta,
    SLIMElasticNet,
)

README_MARKER = "<!-- leaderboard -->"
README_END_MARKER = "<!-- /leaderboard -->"


def default_models() -> dict[str, Any]:
    """The estimators the leaderboard compares, in no particular order."""
    return {
        "SLIM": SLIMElasticNet(),
        "EASE": EASE(),
        "RP3Beta": RP3Beta(),
        "BPR": BayesianPersonalizedRanking(random_state=0),
        "BM25": BM25Recommender(),
        "ItemKNN": ItemKNNRecommender(),
        "MostPopular": MostPopularRecommender(),
        "ALS": AlternatingLeastSquares(random_state=0),
    }


@dataclass(frozen=True)
class Column:
    """One leaderboard column: how to compute a value and how to print it."""

    header: str
    metric: Callable[..., Any] | None = None
    fmt: str = "{:.4f}"
    kwargs_from: str = "none"


def columns(k: int) -> list[Column]:
    """Column specification at cutoff ``k``, ranking metrics first."""
    return [
        Column(f"NDCG@{k}", ndcg_at_k),
        Column(f"P@{k}", precision_at_k),
        Column(f"R@{k}", recall_at_k),
        Column("hit rate", hit_rate_at_k),
        Column("MAP", average_precision_at_k),
        Column("MRR", reciprocal_rank_at_k),
        Column("cat cov", catalog_coverage_at_k, kwargs_from="catalog"),
        Column("user cov", user_coverage_at_k),
        Column("mean pop", mean_popularity_at_k, "{:.4f}", kwargs_from="popularity"),
        Column("novelty", novelty_at_k, "{:.2f}", kwargs_from="popularity"),
    ]


#: Statistics reported for each timed operation, in table order. A float is a
#: nearest-rank quantile; a string names a plain reduction.
#:
#: A quantile is only as meaningful as the number of samples behind it: over ``n``
#: samples, any quantile above ``1 - 1 / n`` is just the slowest sample. A fit is
#: expensive and is therefore timed a handful of times, which supports order statistics
#: but not a tail, so it reports the spread it can actually resolve. Ranking is cheap
#: enough to sample often, so its tail quantiles describe real behaviour.
STATISTICS: dict[str, tuple[tuple[str, str | float], ...]] = {
    "fit": (("min", "min"), ("median", "median"), ("max", "max")),
    "rank": (("mean", "mean"), ("median", "median"), ("q95", 0.95), ("q99", 0.99)),
}

#: Untimed calls made before sampling starts, to pay for cold caches, first-touch page
#: faults and any lazily built state. Without one the first sample is reliably the
#: slowest, which drags a short run's mean well above the steady state.
WARMUP = {"fit": 1, "rank": 10}

#: Samples taken even when the time budget is already spent, so that every row has a
#: median rather than a single observation. A cap below this wins, which is what lets a
#: caller ask for an exact handful of samples.
MIN_SAMPLES = 3


@dataclass(frozen=True)
class Timing:
    """Wall-clock samples of one repeated operation, and the batch each one covered."""

    seconds: NDArray[np.float64]
    #: What a single sample processed, for example interactions fitted or queries ranked.
    batch: str
    #: Which entry of :data:`STATISTICS` describes these samples.
    operation: str = "rank"

    def statistics(self) -> dict[str, str]:
        """The reductions and nearest-rank quantiles configured for this operation.

        Quantiles never interpolate, so each one is a call that really happened.
        """
        reductions = {"min": np.min, "median": np.median, "max": np.max, "mean": np.mean}
        values = {}
        for name, how in STATISTICS[self.operation]:
            seconds = (
                float(reductions[how](self.seconds))
                if isinstance(how, str)
                else float(np.quantile(self.seconds, how, method="higher"))
            )
            values[name] = _format_duration(seconds)
        return values


#: What one timed call of each operation processes, and hence what its batch counts.
BATCH_UNITS = {"fit": "interactions", "rank": "users x items"}


def batch_header(operation: str) -> str:
    """Header of a batch column, carrying the unit its cells are counted in."""
    return f"{operation} batch ({BATCH_UNITS[operation]})"


def timing_columns() -> list[str]:
    """Header of the timing table, fit first."""
    headers = ["Model"]
    for prefix in BATCH_UNITS:
        headers += [batch_header(prefix), f"{prefix} samples"]
        headers += [f"{prefix} {name}" for name, _ in STATISTICS[prefix]]
    return headers


def sample(call: Callable[[], Any], cap: int, budget: float) -> list[float]:
    """Time ``call`` until the sample cap or the time budget, whichever comes first.

    A cheap operation reaches the cap and is described by many samples; an expensive one
    stops at the budget with a handful. Spending the same wall clock on every model would
    mean either waiting on the slowest or starving the rest, and the slowest models are
    also the ones whose timing a few samples already pin down.
    """
    seconds: list[float] = []
    floor = min(MIN_SAMPLES, cap)
    deadline = time.perf_counter() + budget
    while len(seconds) < cap:
        start = time.perf_counter()
        call()
        seconds.append(time.perf_counter() - start)
        if len(seconds) >= floor and time.perf_counter() >= deadline:
            break
    return seconds


@dataclass(frozen=True)
class Evaluation:
    """Everything the metrics need about one fitted model's recommendations."""

    y_true: list[set[Any]]
    y_pred: NDArray[Any]
    catalog: NDArray[Any]
    popularity: Mapping[Any, float]
    fit: Timing
    rank: Timing


def _held_out_by_user(
    X_test: NDArray[Any], y_test: NDArray[np.float64] | None, known_users: NDArray[Any]
) -> tuple[NDArray[Any], list[set[Any]]]:
    """Group positive held-out interactions of known users into relevant item sets."""
    users, items = X_test[:, 0], X_test[:, 1]
    keep = np.isin(users, known_users)
    if y_test is not None:
        keep &= y_test > 0
    users, items = users[keep], items[keep]
    if len(users) == 0:
        raise ValueError("No held-out interactions to score.")

    query_users, codes = np.unique(users, return_inverse=True)
    order = np.argsort(codes, kind="stable")
    boundaries = np.cumsum(np.bincount(codes, minlength=len(query_users)))[:-1]
    return query_users, [set(group.tolist()) for group in np.split(items[order], boundaries)]


def evaluate(
    estimator: Any,
    dataset: Any,
    k: int,
    repeat: int,
    rank_repeat: int,
    budget: float = float("inf"),
) -> Evaluation:
    """Fit ``estimator`` on the training split and recommend for held-out users.

    ``repeat`` and ``rank_repeat`` cap the samples of each operation and ``budget``
    caps the seconds they may spend, so a cheap operation is described by many samples
    and an expensive one stops early. Each operation is run :data:`WARMUP` times first
    without being timed, so no sample carries the cold start of the one before it.
    """
    train, test = dataset.train_indices, dataset.test_indices
    X_train, y_train = dataset.data[train], dataset.target[train]
    X_test, y_test = dataset.data[test], dataset.target[test]

    fitted = None
    for _ in range(WARMUP["fit"]):
        fitted = clone(estimator).fit(X_train, y_train)

    def fit_once() -> None:
        nonlocal fitted
        candidate = clone(estimator)
        candidate.fit(X_train, y_train)
        fitted = candidate

    fit_seconds = sample(fit_once, repeat, budget)
    assert fitted is not None  # noqa: S101 - repeat >= 1 is enforced by the parser

    query_users, y_true = _held_out_by_user(X_test, y_test, fitted.user_ids_)
    # Ranking the whole catalog for every held-out user at once, the way a batch job
    # would; per-request latency of a single user is a different measurement.
    for _ in range(WARMUP["rank"]):
        y_pred, _ = fitted.recommend(query_users, n_recommendations=k, exclude_seen=True)

    def rank_once() -> None:
        nonlocal y_pred
        y_pred, _ = fitted.recommend(query_users, n_recommendations=k, exclude_seen=True)

    rank_seconds = sample(rank_once, rank_repeat, budget)

    popularity = item_popularity(X_train, y_train)
    catalog = np.asarray(fitted.item_ids_)
    # Novelty normalizes over its own support, so cold items must be in the mapping.
    popularity = {item: popularity.get(item, 0.0) for item in catalog.tolist()}
    return Evaluation(
        y_true,
        y_pred,
        catalog,
        popularity,
        Timing(np.asarray(fit_seconds), f"{len(X_train)}", "fit"),
        Timing(np.asarray(rank_seconds), f"{len(query_users)} x {fitted.n_items_}", "rank"),
    )


def score(evaluation: Evaluation, k: int) -> dict[str, str]:
    """Format every column of one row."""
    extra = {
        "none": {},
        "catalog": {"catalog": evaluation.catalog},
        "popularity": {"item_popularity": evaluation.popularity},
    }
    row = {}
    for column in columns(k):
        assert column.metric is not None  # noqa: S101 - every column has one
        value = column.metric(
            evaluation.y_true, evaluation.y_pred, k=k, **extra[column.kwargs_from]
        )
        row[column.header] = column.fmt.format(float(value))
    return row


def timings(evaluation: Evaluation) -> dict[str, str]:
    """Format the timing columns of one row."""
    row = {}
    for prefix, timing in (("fit", evaluation.fit), ("rank", evaluation.rank)):
        row[batch_header(prefix)] = timing.batch
        row[f"{prefix} samples"] = str(len(timing.seconds))
        row |= {f"{prefix} {name}": value for name, value in timing.statistics().items()}
    return row


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def build_rows(
    models: Mapping[str, Any],
    dataset: Any,
    k: int,
    repeat: int,
    rank_repeat: int,
    budget: float = float("inf"),
    *,
    verbose: bool = True,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Evaluate every model; return the quality rows and the timing rows.

    Both are ordered by descending NDCG, so the two tables line up row for row.
    """
    quality, measured = [], {}
    for name, estimator in models.items():
        if verbose:
            print(f"fitting {name} ...", file=sys.stderr)  # noqa: T201
        evaluation = evaluate(estimator, dataset, k, repeat, rank_repeat, budget)
        quality.append({"Model": name} | score(evaluation, k))
        measured[name] = evaluation
    ndcg = f"NDCG@{k}"
    quality.sort(key=lambda row: float(row[ndcg]), reverse=True)
    timed = [{"Model": row["Model"]} | timings(measured[row["Model"]]) for row in quality]
    return quality, timed


def render_box(rows: Sequence[Mapping[str, str]]) -> str:
    """Render rows as a box-drawing table: centered headers, left-aligned cells."""
    if not rows:
        return ""
    headers = list(rows[0])
    widths = [max(len(header), *(len(row[header]) for row in rows)) for header in headers]

    def rule(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (width + 2) for width in widths) + right

    def line(cells: Iterable[str], *, center: bool) -> str:
        rendered = (
            cell.center(width) if center else cell.ljust(width)
            for cell, width in zip(cells, widths, strict=True)
        )
        return "│ " + " │ ".join(rendered) + " │"

    out = [rule("┌", "┬", "┐"), line(headers, center=True)]
    for row in rows:
        out.append(rule("├", "┼", "┤"))
        out.append(line((row[header] for header in headers), center=False))
    out.append(rule("└", "┴", "┘"))
    return "\n".join(out)


def render_markdown(rows: Sequence[Mapping[str, str]]) -> str:
    """Render rows as a GitHub Markdown table."""
    if not rows:
        return ""
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(row[header] for header in headers) + " |" for row in rows]
    return "\n".join(lines)


def render_csv(rows: Sequence[Mapping[str, str]]) -> str:
    """Render rows as CSV."""
    if not rows:
        return ""
    headers = list(rows[0])
    lines = [",".join(headers)]
    lines += [",".join(row[header] for header in headers) for row in rows]
    return "\n".join(lines)


RENDERERS = {"box": render_box, "markdown": render_markdown, "csv": render_csv}


def cpu_model() -> str:
    """The CPU's marketing name, or the coarse platform name when it is not exposed."""
    probes = {
        "Darwin": ["sysctl", "-n", "machdep.cpu.brand_string"],
        "Linux": [
            "sh",
            "-c",
            "sed -n 's/^model name[ \t]*: //p; s/^Model[ \t]*: //p' /proc/cpuinfo | head -1",
        ],
    }
    probe = probes.get(platform.system())
    if probe is not None:
        try:
            out = subprocess.run(probe, capture_output=True, text=True, timeout=5, check=False)  # noqa: S603
        except (OSError, subprocess.SubprocessError):
            out = None
        if out is not None and out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return platform.processor() or platform.machine() or "unknown CPU"


def usable_cpus() -> int | None:
    """Cores the process may actually run on, which is what rayon sizes its pool from."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return len(affinity(0))
    return os.cpu_count()


def host_caption() -> str:
    """The machine and build the timings below came from.

    Quality metrics are deterministic, but every timing column is specific to this host
    and this build, so a published table is only comparable against one that names both.
    """
    cores = usable_cpus()
    cores_text = f"{cores} usable cores" if cores else "unknown core count"
    return (
        f"Measured on {cpu_model()} ({cores_text}), {platform.platform()}, "
        f"{platform.python_implementation()} {platform.python_version()}, "
        f"numpy {np.__version__}, skrecsys {skrecsys.__version__}, `_core` built in "
        f"{_core.__build_profile__} mode. The quality table above is deterministic and "
        f"portable; the timings below are not comparable across machines or builds."
    )


def timing_caption(repeat: int, rank_repeat: int, budget: float) -> str:
    """The sentence that says what the timing table measured."""
    return (
        f"Wall clock per call. Each operation is sampled until it has spent {budget:.0f}s or "
        f"reached its cap ({repeat} fits, {rank_repeat} `recommend` calls), after untimed warm-up "
        f"calls ({WARMUP['fit']} fit, {WARMUP['rank']} rank) so that no sample pays for a cold "
        f"start; the `samples` columns say how many each row actually got, which is why a slow "
        f"model shows fewer. The batch columns say what a single call processed: one fit covers "
        f"the whole training split, and one `recommend` call ranks the entire catalog for every "
        f"held-out user at once, so these are throughput numbers rather than single-request "
        f"latency. Fit reports the spread a handful of samples can resolve; ranking is sampled "
        f"often enough for nearest-rank quantiles, each of which is a call that really happened. "
        f"Compare `min` across machines and watch `max` for the variance a run saw."
    )


def write_readme(readme: Path, body: str) -> None:
    """Replace everything between the leaderboard markers in ``readme``."""
    text = readme.read_text()
    block = f"{README_MARKER}\n{body}\n{README_END_MARKER}"
    pattern = re.compile(re.escape(README_MARKER) + ".*?" + re.escape(README_END_MARKER), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(
            f"{readme} has no leaderboard block; add a {README_MARKER} / "
            f"{README_END_MARKER} pair where the table belongs."
        )
    readme.write_text(pattern.sub(lambda _: block, text, count=1))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=10, help="top-k cutoff (default: 10)")
    parser.add_argument(
        "--subset", default="ua", help="MovieLens 100K split to evaluate (default: ua)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1000,
        help="most timed fits per model; the time budget usually stops it first (default: 1000)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=20.0,
        help="seconds to spend timing each operation of each model (default: 20)",
    )
    parser.add_argument(
        "--rank-repeat",
        type=int,
        default=1000,
        help="timed recommend calls per model; the upper quantiles need many (default: 1000)",
    )
    parser.add_argument("--format", choices=sorted(RENDERERS), default="box")
    parser.add_argument(
        "--models", nargs="+", metavar="NAME", help="subset of models to run (default: all)"
    )
    parser.add_argument(
        "--write-readme",
        nargs="?",
        const="README.md",
        metavar="PATH",
        help="also write a Markdown table into the leaderboard block of README.md",
    )
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be >= 1")
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if args.rank_repeat < 1:
        parser.error("--rank-repeat must be >= 1")
    if args.budget <= 0:
        parser.error("--budget must be > 0")

    models = default_models()
    if args.models:
        unknown = sorted(set(args.models) - set(models))
        if unknown:
            parser.error(f"unknown models: {unknown}; choose from {sorted(models)}")
        models = {name: models[name] for name in args.models}

    dataset = fetch_movielens_100k(subset=args.subset)
    quality, timed = build_rows(models, dataset, args.k, args.repeat, args.rank_repeat, args.budget)
    render = RENDERERS[args.format]
    print(render(quality))  # noqa: T201
    print()  # noqa: T201
    print(host_caption())  # noqa: T201
    print()  # noqa: T201
    print(render(timed))  # noqa: T201

    if args.write_readme:
        caption = (
            f"MovieLens 100K, official `{args.subset}` split, k={args.k}, default "
            f"hyper-parameters. Regenerate with "
            f"`python benchmarks/leaderboard.py --write-readme`."
        )
        write_readme(
            Path(args.write_readme),
            f"{caption}\n\n{render_markdown(quality)}\n\n"
            f"{host_caption()}\n\n"
            f"{timing_caption(args.repeat, args.rank_repeat, args.budget)}\n\n"
            f"{render_markdown(timed)}",
        )
        print(f"wrote the leaderboard into {args.write_readme}", file=sys.stderr)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
