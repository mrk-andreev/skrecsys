"""Score a composed recommender separately on the users it knows and on the users it does not.

The leaderboard asks which model ranks best, and every user it scores was seen in
training. This report asks what a second stage buys a production pipeline, which has two
kinds of user: the ones with a history, served by a personalized model, and the ones
arriving cold, served by whatever a :class:`skrecsys.compose.Switch` falls back to. A
gain on one kind can hide a loss on the other, so every quality row is reported for all
held-out users, for the warm ones and for the cold ones.

Time is reported one way only: what one user's request costs, since a reranker is paid
for per request -- retrieval, featurizing a hundred candidates, and a tree ensemble over
them. Fit time and batch throughput are the leaderboard's questions.

The split is the dataset's, and for this report it is
:class:`skrecsys.model_selection.ColdStartSplit`: some users are held out whole, the rest
lose their latest interactions. Which pipelines run, and under what settings, is
``benchmarks/config/reranking.json``; ``benchmarks/run.py`` runs them.
"""

from __future__ import annotations

import sys
from collections.abc import Hashable, Mapping
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent))

import leaderboard
import spec
from indexes import format_latency
from store import Cells, RerankingPayload

from skrecsys.base import fit_clone, is_recommender
from skrecsys.metrics import item_popularity
from skrecsys.typing import FittedRecommender, Recommender

#: The rows of the quality table, in order: every held-out user, then each kind.
SEGMENTS = ("all", "warm", "cold")

#: Cold users shown by name, and how much of each list is shown.
N_EXAMPLES = 3
EXAMPLE_DEPTH = 5

#: The quality columns left out. ``recommend`` raises rather than return a short list,
#: so user coverage is 1 by construction and would only take up a column.
_OMITTED = ("user cov",)

#: The ``user_info`` columns that describe a cold user in the examples.
_PROFILE = ("age", "gender", "occupation")


class Builder(Protocol):
    """A builder of :mod:`pipelines`: the dataset, then the entry's params."""

    def __call__(self, data: object, /, **params: spec.JSON) -> object: ...


def build(entry: spec.Entry, data: object) -> Recommender:
    """The unfitted pipeline an entry names: a builder of :mod:`pipelines` and its params."""
    found = spec.resolve(f"{entry.package}.{entry.cls}")
    if not callable(found):
        raise spec.ConfigError(f"{entry.package}.{entry.cls} is not a pipeline builder.")
    # Imported by name, so only the call below can show it takes these arguments.
    pipeline = cast(Builder, found)(data, **entry.params)
    if not is_recommender(pipeline):
        raise spec.ConfigError(f"{entry.package}.{entry.cls} did not build a recommender.")
    return pipeline


def measure(
    name: str, pipeline: Recommender, data: spec.Split, settings: spec.Settings
) -> RerankingPayload:
    """Fit ``pipeline`` once, then score it per segment, time it per request, and show it."""
    k = settings["k"]
    train, test = data.train_indices, data.test_indices
    X_train, y_train = data.data[train], data.target[train]
    X_test, y_test = data.data[test], data.target[test]
    model = fit_clone(pipeline, X_train, y_train)

    # Every held-out user, known to the model or not: the cold ones are half the point.
    query_users, y_true = leaderboard._held_out_by_user(X_test, y_test, np.unique(X_test[:, 0]))
    warm = np.isin(query_users, model.user_ids_)
    y_pred, _ = model.recommend(query_users, n_recommendations=k, exclude_seen=True)

    popularity = item_popularity(X_train, y_train)
    catalog = np.asarray(model.item_ids_)
    popularity = {item: popularity.get(item, 0.0) for item in catalog.tolist()}
    masks = {"all": np.ones_like(warm), "warm": warm, "cold": ~warm}
    quality: list[Cells] = []
    for segment in SEGMENTS:
        rows = np.flatnonzero(masks[segment])
        if not len(rows):
            continue
        scored = leaderboard.score_lists(
            [y_true[row] for row in rows], y_pred[rows], k, catalog, popularity
        )
        cells = {"Pipeline": name, "users": segment, "n users": f"{len(rows):,}"}
        quality.append(cells | {h: v for h, v in scored.items() if h not in _OMITTED})

    latency = [
        _latency(name, segment, model, query_users[masks[segment]], settings)
        for segment in ("warm", "cold")
        if masks[segment].any()
    ]
    cold = np.flatnonzero(~warm)[:N_EXAMPLES]
    examples = [_example(data, query_users[row], y_true[row], y_pred[row]) for row in cold]
    return {"quality": quality, "latency": latency, "examples": examples}


def _latency(
    name: str,
    segment: str,
    model: FittedRecommender,
    users: NDArray[np.generic],
    settings: spec.Settings,
) -> Cells:
    """What one request costs: ``recommend`` for a single user, cycling over ``users``.

    Each call asks for a different user, in a fixed order, so the samples cover the
    segment rather than one user's history repeated from a warm cache.
    """
    k, budget = settings["k"], settings["budget"]
    position = 0

    def once() -> None:
        nonlocal position
        start = position % len(users)
        model.recommend(users[start : start + 1], n_recommendations=k)
        position += 1

    for _ in range(max(settings["warmup"]["rank"], 1)):
        once()
    position = 0
    seconds = np.asarray(leaderboard.sample(once, settings["latency_repeat"], budget))
    return {
        "Pipeline": name,
        "users": segment,
        "requests": f"{len(seconds):,}",
        "median": format_latency(float(np.median(seconds))),
        "q95": format_latency(float(np.quantile(seconds, 0.95, method="higher"))),
        "q99": format_latency(float(np.quantile(seconds, 0.99, method="higher"))),
    }


def _example(
    data: object, user: Hashable, relevant: set[Hashable], items: NDArray[np.generic]
) -> Cells:
    """One cold user: who they are, and the top of their list with the held-out hits bold."""
    titles = _lookup(data, "item_info", "item_id", "title")
    shown = []
    for item in items[:EXAMPLE_DEPTH].tolist():
        title = titles.get(item, str(item))
        shown.append(f"**{title}**" if item in relevant else title)
    return {
        "user": str(user),
        "profile": _profile(data, user),
        "held out": f"{len(relevant):,}",
        "top": "<br>".join(shown),
    }


def _lookup(data: object, table: str, key: str, column: str) -> dict[Hashable, str]:
    """``column`` of a side table by ``key``, or nothing when the dataset has no such table."""
    found = getattr(data, table, None)
    if not isinstance(found, Mapping) or key not in found or column not in found:
        return {}
    return {k: str(v) for k, v in zip(found[key].tolist(), found[column].tolist(), strict=True)}


def _profile(data: object, user: Hashable) -> str:
    """Age, gender and occupation -- everything a cold user brings -- when the dataset has them."""
    fields = (_lookup(data, "user_info", "user_id", c).get(user) for c in _PROFILE)
    return ", ".join(field for field in fields if field)
