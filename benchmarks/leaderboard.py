"""Fit a recommender on a benchmark dataset, and score and time what it recommends.

Each model is fitted on the training half of an official split and evaluated on the
held-out half with every metric in :mod:`skrecsys.metrics`, so the table mixes ranking
quality with the beyond-accuracy metrics that explain it: a model can win on NDCG while
recommending nothing but the head of the catalog.

This module measures and formats rows; which models run on which dataset, and under
what settings, is ``benchmarks/config/leaderboard.json``, and ``benchmarks/run.py`` is
what runs them. The timing machinery here is shared by the sequential and index reports.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone

import skrecsys
from skrecsys import _core
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


@dataclass(frozen=True)
class Column:
    """One leaderboard column: how to compute a value and how to print it."""

    header: str
    metric: Callable[..., Any]
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
    headers: list[str] = ["Model"]
    for prefix in BATCH_UNITS:
        headers += [batch_header(prefix), f"{prefix} samples"]
        headers += [f"{prefix} {name}" for name, _ in STATISTICS[prefix]]
    return headers


_T = TypeVar("_T")


def sample(call: Callable[[], Any], cap: int, budget: float) -> list[float]:
    """Time ``call`` until the sample cap or the time budget, whichever comes first.

    A cheap operation reaches the cap and is described by many samples; an expensive one
    stops at the budget with a handful. Spending the same wall clock on every model would
    mean either waiting on the slowest or starving the rest, and the slowest models are
    also the ones whose timing a few samples already pin down.
    """
    seconds, _ = timed(call, cap, budget)
    return seconds


def timed(call: Callable[[], _T], cap: int, budget: float) -> tuple[list[float], _T]:
    """:func:`sample`, also returning what the last call returned.

    ``call`` always runs at least once, so a result is always there to return.
    """
    if cap < 1:
        raise ValueError(f"cap must be at least 1, got {cap}.")
    seconds: list[float] = []
    floor = min(MIN_SAMPLES, cap)
    deadline = time.perf_counter() + budget
    while True:
        start = time.perf_counter()
        result = call()
        seconds.append(time.perf_counter() - start)
        if len(seconds) >= cap or (len(seconds) >= floor and time.perf_counter() >= deadline):
            return seconds, result


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
    max_eval_users: int | None = None,
    warmup: Mapping[str, int] = WARMUP,
) -> Evaluation:
    """Fit ``estimator`` on the training split and recommend for held-out users.

    ``repeat`` and ``rank_repeat`` cap the samples of each operation and ``budget``
    caps the seconds they may spend, so a cheap operation is described by many samples
    and an expensive one stops early. Each operation is run ``warmup`` times first
    without being timed, so no sample carries the cold start of the one before it.

    ``max_eval_users`` scores a fixed random sample of the held-out users instead of
    all of them. Every quality metric averages over users, so the sample estimates the
    same quantity; the timing columns then describe that batch and not the full one.
    """
    train, test = dataset.train_indices, dataset.test_indices
    X_train, y_train = dataset.data[train], dataset.target[train]
    X_test, y_test = dataset.data[test], dataset.target[test]

    def fit_once() -> Any:
        return clone(estimator).fit(X_train, y_train)

    for _ in range(warmup["fit"]):
        fit_once()
    fit_seconds, model = timed(fit_once, repeat, budget)

    query_users, y_true = _held_out_by_user(X_test, y_test, model.user_ids_)
    query_users, y_true = _sample_users(query_users, y_true, max_eval_users)

    # Ranking the whole catalog for every held-out user at once, the way a batch job
    # would; per-request latency of a single user is a different measurement.
    def rank_once() -> NDArray[Any]:
        items, _ = model.recommend(query_users, n_recommendations=k, exclude_seen=True)
        return items

    for _ in range(warmup["rank"]):
        rank_once()
    rank_seconds, y_pred = timed(rank_once, rank_repeat, budget)

    popularity = item_popularity(X_train, y_train)
    catalog = np.asarray(model.item_ids_)
    # Novelty normalizes over its own support, so cold items must be in the mapping.
    popularity = {item: popularity.get(item, 0.0) for item in catalog.tolist()}
    return Evaluation(
        y_true,
        y_pred,
        catalog,
        popularity,
        Timing(np.asarray(fit_seconds), f"{len(X_train)}", "fit"),
        Timing(np.asarray(rank_seconds), f"{len(query_users)} x {model.n_items_}", "rank"),
    )


def _sample_users(
    query_users: NDArray[Any], y_true: list[set[Any]], max_eval_users: int | None
) -> tuple[NDArray[Any], list[set[Any]]]:
    """Keep a fixed random sample of the held-out users, in their original order."""
    if max_eval_users is None or len(query_users) <= max_eval_users:
        return query_users, y_true
    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(len(query_users), size=max_eval_users, replace=False))
    return query_users[keep], [y_true[index] for index in keep]


def score(evaluation: Evaluation, k: int) -> dict[str, str]:
    """Format every column of one row."""
    extra = {
        "none": {},
        "catalog": {"catalog": evaluation.catalog},
        "popularity": {"item_popularity": evaluation.popularity},
    }
    row = {}
    for column in columns(k):
        value = column.metric(
            evaluation.y_true, evaluation.y_pred, k=k, **extra[column.kwargs_from]
        )
        row[column.header] = column.fmt.format(float(value))
    return row


def timings(evaluation: Evaluation) -> dict[str, str]:
    """Format the timing columns of one row."""
    row: dict[str, str] = {}
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
    max_eval_users: int | None = None,
    warmup: Mapping[str, int] = WARMUP,
    score_row: Callable[[Evaluation, int], dict[str, str]] | None = None,
    sort_by: str | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Evaluate every model; return the quality rows and the timing rows.

    Both are ordered by descending ``sort_by`` (NDCG at the cutoff by default), so the
    two tables line up row for row. ``score_row`` is what turns one evaluation into the
    columns of a row, which is where the sequential benchmark reports its own.
    """
    score_row = score_row or score
    quality: list[dict[str, str]] = []
    measured: dict[str, Evaluation] = {}
    for name, estimator in models.items():
        evaluation = evaluate(
            estimator, dataset, k, repeat, rank_repeat, budget, max_eval_users, warmup
        )
        quality.append({"Model": name} | score_row(evaluation, k))
        measured[name] = evaluation
    ranked_on = sort_by or f"NDCG@{k}"
    quality.sort(key=lambda row: float(row[ranked_on]), reverse=True)
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
    system = platform.system()
    if system == "Darwin":
        try:
            out = subprocess.run(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            out = None
        if out is not None and out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    elif system == "Linux":
        name = _cpuinfo_model(Path("/proc/cpuinfo"))
        if name:
            return name
    return platform.processor() or platform.machine() or "unknown CPU"


def _cpuinfo_model(path: Path) -> str | None:
    """The first ``model name`` (x86) or ``Model`` (ARM) field of ``/proc/cpuinfo``."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        key, sep, value = line.partition(":")
        if sep and key.strip() in {"model name", "Model"} and value.strip():
            return value.strip()
    return None


def usable_cpus() -> int | None:
    """Cores the process may actually run on, which is what rayon sizes its pool from."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return len(affinity(0))
    return os.cpu_count()


def host_facts() -> dict[str, Any]:
    """The machine and build a timing was taken on, as facts rather than a sentence.

    Quality metrics are deterministic, but every timing is specific to its host and its
    build, so a stored result carries these and the report says where each row ran.
    """
    return {
        "cpu": cpu_model(),
        "cores": usable_cpus(),
        "platform": platform.platform(),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "numpy": np.__version__,
        "skrecsys": skrecsys.__version__,
        "build": _core.__build_profile__,
    }
