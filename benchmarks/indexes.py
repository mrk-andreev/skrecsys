"""What an approximate index buys, and what it costs, per model.

Every row here is a pair: a model ranking exactly, and the same model ranking through
an index. The point of putting them on one line is that the gain and the loss are read
together -- a speedup means nothing without the recall beside it, and a recall means
nothing without the quality delta beside *that*, because agreeing with an exact model is
not the same as being right.

Nothing here is clamped or hidden: a configuration that loses prints its ``0.31x`` next
to one that wins with ``48x``, under one methodology, on one page. For several of these
models losing is the expected outcome, and the table is how that gets established rather
than asserted.

This module measures and shapes tables; ``benchmarks/run.py`` decides what to run, from
``benchmarks/config/indexes.json``, and stores one result per (model, index, table).
"""

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone

# The shared timing machinery lives in the sibling module; importing it keeps one
# implementation of "what a benchmark row looks like".
sys.path.insert(0, str(Path(__file__).resolve().parent))

import leaderboard
from leaderboard import WARMUP, Timing, sample
from spec import IndexEntry

from skrecsys.metrics import catalog_coverage_at_k, ndcg_at_k

#: The tables of the report, each stored as its own result so that one can be re-run
#: without the others: the sweep table (answers and cost at every dial value), the
#: single-request latency table, and the catalog-scaling table.
KINDS = ("sweep", "latency", "scaling")


class Measurement:
    """One configuration of one model: what it recommended, and what that cost."""

    def __init__(
        self,
        items: NDArray[Any],
        scores: NDArray[np.float64],
        fit: Timing,
        build: Timing | None,
        rank: Timing,
        index_bytes: int,
        vectors_bytes: int,
    ) -> None:
        self.items = items
        self.scores = scores
        self.fit = fit
        self.build = build
        self.rank = rank
        self.index_bytes = index_bytes
        self.vectors_bytes = vectors_bytes

    @property
    def rank_median(self) -> float:
        return float(np.median(self.rank.seconds))


def measure_model(
    estimator: Any,
    query_users: NDArray[Any],
    X_train: NDArray[Any],
    y_train: NDArray[np.float64] | None,
    k: int,
    indexes: Sequence[IndexEntry],
    repeat: int,
    rank_repeat: int,
    budget: float,
    warmup: Mapping[str, int] = WARMUP,
) -> tuple[Measurement, list[tuple[IndexEntry, int, Measurement]]]:
    """One model, ranked exactly and then at every dial value of every index.

    The model is fitted once, and each index is built once, because a dial is a
    query-time width: sweeping it by refitting would spend an hour on a six-figure
    catalog measuring the same two things over and over. Building once per *index* is
    unavoidable and is exactly what the `build` column is there to price -- a graph
    costs minutes where a set of codes costs a pass of arithmetic, and that difference
    is a result rather than an overhead to hide.

    Fit and build are timed apart from each other. An index that doubles the time to fit
    is a different proposition from one that adds a twentieth, and one number hides
    which.
    """
    bare = clone(estimator).set_params(index=None)

    def fit_once() -> Any:
        return clone(bare).fit(X_train, y_train)

    for _ in range(warmup["fit"]):
        fit_once()
    fit_seconds, fitted = leaderboard.timed(fit_once, repeat, budget)
    fit = Timing(np.asarray(fit_seconds), f"{len(X_train)}", "fit")
    batch = f"{len(query_users)} x {fitted.n_items_}"

    def rank_once() -> tuple[NDArray[Any], NDArray[np.float64]]:
        return fitted.recommend(query_users, n_recommendations=k, exclude_seen=True)

    def rank() -> tuple[Timing, NDArray[Any], NDArray[np.float64]]:
        for _ in range(warmup["rank"]):
            rank_once()
        seconds, (items, scores) = leaderboard.timed(rank_once, rank_repeat, budget)
        return Timing(np.asarray(seconds), batch, "rank"), items, scores

    exact_rank, items, scores = rank()
    exact = Measurement(items, scores, fit, None, exact_rank, 0, 0)

    approximate: list[tuple[IndexEntry, int, Measurement]] = []
    for index in indexes:
        fitted.set_params(index=index.build())
        # Warmed like the fit is, and for the same reason; on a six-figure catalog a
        # build costs minutes, so a fit warm-up of 0 is how a caller buys one not two.
        for _ in range(warmup["fit"]):
            fitted._fit_index()
        build_seconds = sample(fitted._fit_index, repeat, budget)
        build = Timing(np.asarray(build_seconds), f"{fitted.n_items_}", "fit")
        index_bytes, vectors_bytes = fitted.index_.nbytes, fitted.index_.space_.nbytes

        for value in index.dial.values:
            fitted.index_.set_params(**{index.dial.param: value})
            index_rank, items, scores = rank()
            approximate.append(
                (
                    index,
                    value,
                    Measurement(items, scores, fit, build, index_rank, index_bytes, vectors_bytes),
                )
            )
    return exact, approximate


def recall_at_k(approx: NDArray[Any], exact: NDArray[Any]) -> float:
    """Share of the exact answer the approximate one kept, averaged over queries."""
    pairs = zip(approx, exact, strict=True)
    hits = sum(len(set(a.tolist()) & set(e.tolist())) for a, e in pairs)
    return hits / float(approx.size)


def top_one_churn(approx: NDArray[Any], exact: NDArray[Any]) -> float:
    """Share of queries whose first recommendation changed -- the one a user notices."""
    return float(np.mean(approx[:, 0] != exact[:, 0]))


def score_gap(approx: NDArray[np.float64], exact: NDArray[np.float64]) -> float:
    """Relative score lost at rank one.

    Says whether a churned top item was a near-tie or a real miss, which recall alone
    cannot: swapping two items a thousandth apart and dropping the best item by a third
    both cost the same recall.
    """
    denominator = np.abs(exact[:, 0])
    denominator = np.where(denominator > 0, denominator, 1.0)
    return float(np.mean((exact[:, 0] - approx[:, 0]) / denominator))


def quality_row(
    model: str,
    label: str,
    dial: str,
    measurement: Measurement,
    baseline: Measurement,
    y_true: list[set[Any]],
    catalog: NDArray[Any],
    k: int,
) -> dict[str, str]:
    """One row of the table that says what the index cost in answers."""
    ndcg = float(ndcg_at_k(y_true, measurement.items, k=k))
    base_ndcg = float(ndcg_at_k(y_true, baseline.items, k=k))
    coverage = float(catalog_coverage_at_k(y_true, measurement.items, k=k, catalog=catalog))
    base_coverage = float(catalog_coverage_at_k(y_true, baseline.items, k=k, catalog=catalog))
    return {
        "Model": model,
        "Index": label,
        "dial": dial,
        f"recall@{k}": f"{recall_at_k(measurement.items, baseline.items):.4f}",
        f"NDCG@{k}": f"{ndcg:.4f}",
        f"dNDCG@{k}": f"{ndcg - base_ndcg:+.4f}",
        "top-1 churn": f"{top_one_churn(measurement.items, baseline.items):.4f}",
        "score gap@1": f"{score_gap(measurement.scores, baseline.scores):+.4f}",
        "dcat cov": f"{coverage - base_coverage:+.4f}",
    }


def cost_row(
    model: str,
    label: str,
    dial: str,
    measurement: Measurement,
    baseline: Measurement,
) -> dict[str, str]:
    """One row of the table that says what the index saved in time."""
    fit_median = float(np.median(measurement.fit.seconds))
    build_median = float(np.median(measurement.build.seconds)) if measurement.build else 0.0
    rank = measurement.rank.statistics()
    users = float(measurement.items.shape[0])
    return {
        "Model": model,
        "Index": label,
        "dial": dial,
        "fit": leaderboard._format_duration(fit_median),
        "build": leaderboard._format_duration(build_median) if measurement.build else "-",
        "build %fit": f"{100 * build_median / fit_median:.1f}%" if measurement.build else "-",
        "index MB": f"{measurement.index_bytes / 1e6:.1f}" if measurement.build else "-",
        "vectors MB": f"{measurement.vectors_bytes / 1e6:.1f}" if measurement.build else "-",
        "rank median": rank["median"],
        "rank q95": rank["q95"],
        "speedup": f"{baseline.rank_median / measurement.rank_median:.2f}x",
        "users/s": f"{users / measurement.rank_median:,.0f}",
    }


def sweep_payloads(
    name: str,
    estimator: Any,
    data: Any,
    indexes: Sequence[IndexEntry],
    settings: Mapping[str, Any],
    max_eval_users: int | None,
) -> dict[str, dict[str, Any]]:
    """Measure one model exactly and through each of ``indexes``, one payload per index.

    Every payload carries the exact row it was measured against, so a result stays
    self-consistent when another index of the same model is re-run later: its recall and
    its speedup were computed against *this* baseline, and the baseline is right there.
    """
    k = settings["k"]
    X_train, y_train, X_test, y_test = _split(data)
    probe = clone(estimator).set_params(index=None).fit(X_train, y_train)
    query_users, y_true = leaderboard._held_out_by_user(X_test, y_test, probe.user_ids_)
    query_users, y_true = leaderboard._sample_users(query_users, y_true, max_eval_users)
    catalog = np.asarray(probe.item_ids_)
    n_items = int(probe.n_items_)
    del probe

    baseline, approximate = measure_model(
        estimator,
        query_users,
        X_train,
        y_train,
        k,
        indexes,
        settings["repeat"],
        settings["rank_repeat"],
        settings["budget"],
        settings["warmup"],
    )
    exact = {
        "quality": quality_row(name, "exact", "-", baseline, baseline, y_true, catalog, k),
        "cost": cost_row(name, "exact", "-", baseline, baseline),
    }
    payloads: dict[str, dict[str, Any]] = {
        index.name: {"n_items": n_items, "exact": exact, "rows": []} for index in indexes
    }
    for index, value, measurement in approximate:
        setting = index.dial.setting(value)
        payloads[index.name]["rows"].append(
            {
                "quality": quality_row(
                    name, index.label, setting, measurement, baseline, y_true, catalog, k
                ),
                "cost": cost_row(name, index.label, setting, measurement, baseline),
            }
        )
    return payloads


def scale_catalog(
    X: NDArray[Any], y: NDArray[np.float64] | None, share: float
) -> tuple[NDArray[Any], NDArray[np.float64] | None]:
    """Keep a share of the catalog, and the interactions that fall in it.

    The items kept are the most popular ones rather than a random sample: dropping the
    tail changes how big the catalog is without changing what the data looks like, while
    a random cut would thin every user's history at the same time and confound the two.
    """
    items, counts = np.unique(X[:, 1], return_counts=True)
    keep = items[np.argsort(-counts, kind="stable")[: max(int(len(items) * share), 1)]]
    mask = np.isin(X[:, 1], keep)
    return X[mask], None if y is None else y[mask]


def scaling_payloads(
    estimator: Any,
    data: Any,
    indexes: Sequence[IndexEntry],
    settings: Mapping[str, Any],
    max_eval_users: int | None,
) -> dict[str, dict[str, Any]]:
    """How ranking time answers to the size of the catalog, one payload per index.

    The falsifiable part of the report. An exact scan is linear in the catalog and a
    graph walk is meant not to be, so those two columns should diverge as the catalog
    grows. A quantized scan is linear *too* -- it narrows what each comparison costs,
    not how many there are -- so its column is predicted to stay parallel to the exact
    one, a constant factor below it. That is a claim this table can falsify, which is
    why it is run at one point of every index rather than at the graph's alone.

    Each index is measured at its dial's middle value, against an exact ranking of the
    same queries at the same catalog size taken in the same call.
    """
    k = settings["k"]
    X_train, y_train, X_test, y_test = _split(data)
    points: dict[str, list[dict[str, str]]] = {index.name: [] for index in indexes}
    operating = [_at_middle(index) for index in indexes]
    for share in settings["catalog_scale"]:
        X, y = scale_catalog(X_train, y_train, share)
        probe = clone(estimator).set_params(index=None).fit(X, y)
        users, _ = leaderboard._held_out_by_user(X_test, y_test, probe.user_ids_)
        # Shrinking the catalog leaves heavy users with fewer than `k` items they have
        # not already seen, and `recommend` rightly refuses those rather than returning
        # a short list. They are dropped from the timing instead, so the columns compare
        # the two paths on the same queries.
        seen = np.diff(probe.interactions_.indptr)
        eligible = probe.n_items_ - seen[np.searchsorted(probe.user_ids_, users)]
        users = users[eligible >= k]
        if len(users) == 0:
            del probe
            continue
        users, _ = leaderboard._sample_users(users, [set()] * len(users), max_eval_users)
        n_items = probe.n_items_
        del probe

        exact, approximate = measure_model(
            estimator,
            users,
            X,
            y,
            k,
            operating,
            settings["repeat"],
            settings["rank_repeat"],
            settings["budget"],
            settings["warmup"],
        )
        for index, value, measurement in approximate:
            points[index.name].append(
                {
                    "items": f"{n_items:,}",
                    "users": f"{len(users):,}",
                    "exact rank": leaderboard._format_duration(exact.rank_median),
                    "setting": index.dial.setting(value),
                    "rank": leaderboard._format_duration(measurement.rank_median),
                    "speedup": f"{exact.rank_median / measurement.rank_median:.2f}x",
                }
            )
    return {name: {"points": series} for name, series in points.items()}


#: Where a duration stops being worth reading in microseconds.
MILLISECOND = 1e-3


def format_latency(seconds: float) -> str:
    """A duration at the resolution one request needs.

    The leaderboard's formatter reports whole milliseconds, which is right for a batch
    of ten thousand and rounds a single request to ``0 ms``.
    """
    if seconds < MILLISECOND:
        return f"{seconds * 1e6:.0f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def latency_payloads(
    estimator: Any,
    data: Any,
    indexes: Sequence[IndexEntry],
    settings: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """What one request costs, rather than what ten thousand of them cost together.

    The sweep table ranks every held-out user in a single call, which is a batch job and
    is where the throughput numbers come from. A serving path asks a different question:
    one user arrives, and the answer is owed before the next one. The two can disagree,
    because both paths set up scratch proportional to the catalog before they score
    anything -- an accumulator for the exact path, a visited list and a scatter buffer
    for the graph. Across ten thousand queries that setup is amortised into nothing.
    Across one it is most of the measurement.
    """
    k, budget, warmup = settings["k"], settings["budget"], settings["warmup"]
    repeat = settings["latency_repeat"]
    X_train, y_train, X_test, y_test = _split(data)
    fitted = clone(estimator).set_params(index=None).fit(X_train, y_train)
    query_users, _ = leaderboard._held_out_by_user(X_test, y_test, fitted.user_ids_)
    # One fitted copy per index, each holding its own built index, so the timing below
    # is a `recommend` call and never a build.
    indexed = {}
    for index in indexes:
        model = clone(estimator).set_params(index=None).fit(X_train, y_train)
        model.set_params(index=index.build())
        model._fit_index()
        model.index_.set_params(**{index.dial.param: index.dial.middle()})
        indexed[index.name] = (index, model)

    points: dict[str, list[dict[str, str]]] = {index.name: [] for index in indexes}
    for batch in settings["latency_batch"]:
        users = query_users[:batch]
        if len(users) < batch:
            continue

        def timed(model: Any, users: NDArray[Any] = users) -> float:
            def once() -> None:
                model.recommend(users, n_recommendations=k, exclude_seen=True)

            for _ in range(max(warmup["rank"], 1)):
                once()
            return float(np.median(sample(once, repeat, budget)))

        exact = timed(fitted)
        for name, (index, model) in indexed.items():
            approx = timed(model)
            points[name].append(
                {
                    "users/request": f"{batch:,}",
                    "exact": format_latency(exact),
                    "exact per user": format_latency(exact / batch),
                    "setting": index.dial.setting(index.dial.middle()),
                    "rank": format_latency(approx),
                    "speedup": f"{exact / approx:.2f}x",
                }
            )
    return {name: {"points": series} for name, series in points.items()}


def _at_middle(index: IndexEntry) -> IndexEntry:
    """``index`` swept at one value only, the one the single-setting tables use."""
    return index.with_values([index.dial.middle()])


def _split(data: Any) -> tuple[Any, Any, Any, Any]:
    train, test = data.train_indices, data.test_indices
    return data.data[train], data.target[train], data.data[test], data.target[test]


# --- Tables, shaped from stored payloads ------------------------------------------------
#
# A result holds one (model, index) pair, but a table reads across them: every index of a
# model on one line, every model in one block. These put the stored cells back together
# in the order the config lists models and indexes, so the rendered table reads the same
# however many runs its rows came from.


def sweep_tables(
    models: Sequence[str],
    indexes: Sequence[IndexEntry],
    payloads: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """The answers table and the cost table, each model's exact row first.

    The exact row printed is the baseline of the first index that has a result, which
    is the one its own rows were measured against.
    """
    quality: list[dict[str, str]] = []
    cost: list[dict[str, str]] = []
    for model in models:
        present = [
            payloads[model, index.name] for index in indexes if (model, index.name) in payloads
        ]
        if not present:
            continue
        quality.append(present[0]["exact"]["quality"])
        cost.append(present[0]["exact"]["cost"])
        for payload in present:
            quality.extend(row["quality"] for row in payload["rows"])
            cost.extend(row["cost"] for row in payload["rows"])
    return quality, cost


def scaling_table(
    models: Sequence[str],
    indexes: Sequence[IndexEntry],
    payloads: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, str]]:
    """One row per model and catalog size, a pair of columns per index."""
    return _wide(
        models,
        indexes,
        payloads,
        lead=("items", "users"),
        exact=("exact rank",),
        column=lambda index, point: f"{index.short} rank ({point['setting']})",
    )


def latency_table(
    models: Sequence[str],
    indexes: Sequence[IndexEntry],
    payloads: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, str]]:
    """One row per model and request size, a pair of columns per index."""
    return _wide(
        models,
        indexes,
        payloads,
        lead=("users/request",),
        exact=("exact", "exact per user"),
        column=lambda index, point: f"{index.short} ({point['setting']})",
    )


def _wide(
    models: Sequence[str],
    indexes: Sequence[IndexEntry],
    payloads: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    lead: tuple[str, ...],
    exact: tuple[str, ...],
    column: Any,
) -> list[dict[str, str]]:
    """Lay per-index series side by side, keyed on the ``lead`` cells they share.

    The exact cells come from the first index measured at that point. Only indexes with
    a result for some model get columns, and a point one of them lacks prints ``-``.
    """
    measured = [index for index in indexes if any((m, index.name) in payloads for m in models)]
    header_of: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    for model in models:
        series = {
            index.name: payloads[model, index.name]["points"]
            for index in measured
            if (model, index.name) in payloads
        }
        if not series:
            continue
        order: list[tuple[str, ...]] = []
        by_point: dict[tuple[str, ...], dict[str, dict[str, str]]] = {}
        for index in measured:
            for point in series.get(index.name, []):
                where = tuple(point[name] for name in lead)
                if where not in by_point:
                    order.append(where)
                    by_point[where] = {}
                by_point[where][index.name] = point
                header_of.setdefault(index.name, column(index, point))
        for where in order:
            first = next(iter(by_point[where].values()))
            row = {"Model": model} | dict(zip(lead, where, strict=True))
            row |= {name: first[name] for name in exact}
            for index in measured:
                point = by_point[where].get(index.name)
                header = header_of.get(index.name, column(index, {"setting": "-"}))
                row[header] = point["rank"] if point else "-"
                row[f"{index.short} speedup"] = point["speedup"] if point else "-"
            rows.append(row)
    return rows
