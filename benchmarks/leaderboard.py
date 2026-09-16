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
import re
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone

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
    BM25Recommender,
    ItemKNNRecommender,
    MostPopularRecommender,
    RP3Beta,
)

README_MARKER = "<!-- leaderboard -->"
README_END_MARKER = "<!-- /leaderboard -->"


def default_models() -> dict[str, Any]:
    """The estimators the leaderboard compares, in no particular order."""
    return {
        "EASE": EASE(),
        "RP3Beta": RP3Beta(),
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
    timing: str = ""


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
        Column("fit", timing="fit_seconds"),
        Column("rec", timing="recommend_seconds"),
    ]


@dataclass(frozen=True)
class Evaluation:
    """Everything the metrics need about one fitted model's recommendations."""

    y_true: list[set[Any]]
    y_pred: NDArray[Any]
    catalog: NDArray[Any]
    popularity: Mapping[Any, float]
    fit_seconds: float
    recommend_seconds: float


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


def evaluate(estimator: Any, dataset: Any, k: int, repeat: int) -> Evaluation:
    """Fit ``estimator`` on the training split and recommend for held-out users."""
    train, test = dataset.train_indices, dataset.test_indices
    X_train, y_train = dataset.data[train], dataset.target[train]
    X_test, y_test = dataset.data[test], dataset.target[test]

    timings = []
    fitted = None
    for _ in range(repeat):
        candidate = clone(estimator)
        start = time.perf_counter()
        candidate.fit(X_train, y_train)
        timings.append(time.perf_counter() - start)
        fitted = candidate
    assert fitted is not None  # noqa: S101 - repeat >= 1 is enforced by the parser

    query_users, y_true = _held_out_by_user(X_test, y_test, fitted.user_ids_)
    # Ranking the whole catalog for every held-out user at once, the way a batch job
    # would; per-request latency of a single user is a different measurement.
    recommend_timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        y_pred, _ = fitted.recommend(query_users, n_recommendations=k, exclude_seen=True)
        recommend_timings.append(time.perf_counter() - start)

    popularity = item_popularity(X_train, y_train)
    catalog = np.asarray(fitted.item_ids_)
    # Novelty normalizes over its own support, so cold items must be in the mapping.
    popularity = {item: popularity.get(item, 0.0) for item in catalog.tolist()}
    return Evaluation(
        y_true,
        y_pred,
        catalog,
        popularity,
        float(np.median(timings)),
        float(np.median(recommend_timings)),
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
        if column.timing:
            row[column.header] = _format_duration(getattr(evaluation, column.timing))
            continue
        assert column.metric is not None  # noqa: S101 - every other column has one
        value = column.metric(
            evaluation.y_true, evaluation.y_pred, k=k, **extra[column.kwargs_from]
        )
        row[column.header] = column.fmt.format(float(value))
    return row


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def build_rows(
    models: Mapping[str, Any], dataset: Any, k: int, repeat: int, *, verbose: bool = True
) -> list[dict[str, str]]:
    """Evaluate every model and return the rows, best NDCG first."""
    rows = []
    for name, estimator in models.items():
        if verbose:
            print(f"fitting {name} ...", file=sys.stderr)  # noqa: T201
        evaluation = evaluate(estimator, dataset, k, repeat)
        rows.append({"Model": name} | score(evaluation, k))
    ndcg = f"NDCG@{k}"
    return sorted(rows, key=lambda row: float(row[ndcg]), reverse=True)


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


def write_readme(readme: Path, table: str, caption: str) -> None:
    """Replace the table between the leaderboard markers in ``readme``."""
    text = readme.read_text()
    block = f"{README_MARKER}\n{caption}\n\n{table}\n{README_END_MARKER}"
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
        "--repeat", type=int, default=3, help="fits per model; the median is timed (default: 3)"
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

    models = default_models()
    if args.models:
        unknown = sorted(set(args.models) - set(models))
        if unknown:
            parser.error(f"unknown models: {unknown}; choose from {sorted(models)}")
        models = {name: models[name] for name in args.models}

    dataset = fetch_movielens_100k(subset=args.subset)
    rows = build_rows(models, dataset, args.k, args.repeat)
    print(RENDERERS[args.format](rows))  # noqa: T201

    if args.write_readme:
        caption = (
            f"MovieLens 100K, official `{args.subset}` split, k={args.k}, default "
            f"hyper-parameters. Regenerate with "
            f"`python benchmarks/leaderboard.py --write-readme`."
        )
        write_readme(Path(args.write_readme), render_markdown(rows), caption)
        print(f"wrote the leaderboard into {args.write_readme}", file=sys.stderr)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
