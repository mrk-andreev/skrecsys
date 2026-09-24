"""How much of what users go on to like each set of candidate generators can retrieve.

A :class:`skrecsys.compose.Cascade` ranker only reorders what its generators retrieved,
so the share of a user's held-out items among the candidates -- the candidate recall,
or *ceiling* -- bounds what any ranker can reach. Merging several generators is a bet
about that ceiling: at a fixed budget of ``n_retrieved`` distinct items, each generator
gives up slots to the others, and the merge pays only when what the others bring is
worth more than what it displaces. This report measures the bet directly, for every set
of generators in ``benchmarks/config/candidates.json`` and at every budget of the
``n_retrieved`` setting, against the best of the set's own members at the same budget.

The ceiling involves no ranker, so it is deterministic and free of the noise a trained
second stage adds; what a ranker then makes of it is the reranking report's question.
Time is reported per request, as retrieval for one user, since that is what a merge
multiplies: every generator retrieves the full budget before duplicates are dropped.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent))

import leaderboard
import spec
from indexes import format_latency
from store import CandidatesPayload, Cells

from skrecsys._typing import FittedRecommender, Recommender
from skrecsys.base import fit_clone, is_recommender
from skrecsys.compose import ReciprocalRankFusion
from skrecsys.compose._candidates import concat_ids, retrieve, retrieve_union
from skrecsys.utils.validation import factorize, lookup_ids

#: Where the members of a generator set are looked up by class name.
_MEMBERS_FROM = "skrecsys.recommendation"


def generators(*, members: list[str]) -> list[Recommender]:
    """A generator set: classes of :mod:`skrecsys.recommendation` at their defaults.

    ``members`` is in the order a :class:`~skrecsys.compose.Cascade` would be given
    them, which is the order the merge interleaves them in.
    """
    found = []
    for name in members:
        member = getattr(importlib.import_module(_MEMBERS_FROM), name, None)
        generator = member() if isinstance(member, type) else None
        if not is_recommender(generator):
            raise spec.ConfigError(f"{_MEMBERS_FROM}.{name} is not a recommender.")
        found.append(generator)
    return found


def fused(*, members: list[str], k: float = 60.0) -> list[Recommender]:
    """The members of :func:`generators` merged by reciprocal rank fusion, not round-robin.

    The set is the one :class:`~skrecsys.compose.ReciprocalRankFusion`; at a budget of
    ``N`` it asks every member for ``N`` items, as the round-robin merge does, and keeps
    the ``N`` with the highest fused score.
    """
    return [ReciprocalRankFusion(generators(members=members), k=k)]


class Builder(Protocol):
    """A builder an entry names, such as :func:`generators`: the entry's params in."""

    def __call__(self, **params: spec.JSON) -> list[Recommender]: ...


def build(entry: spec.Entry) -> list[Recommender]:
    """The unfitted generators an entry names: a builder and its params."""
    found = spec.resolve(f"{entry.package}.{entry.cls}")
    if not callable(found):
        raise spec.ConfigError(f"{entry.package}.{entry.cls} is not a generator-set builder.")
    # Imported by name, so only the call below can show it takes these arguments.
    built = cast(Builder, found)(**entry.params)
    if not built:
        raise spec.ConfigError(f"{entry.name}: a generator set needs at least one member.")
    return built


def measure(
    name: str, generators: Sequence[Recommender], data: spec.Split, settings: spec.Settings
) -> CandidatesPayload:
    """Fit the members once, then take the ceiling and the request time at every budget."""
    budgets = settings["n_retrieved"]
    train, test = data.train_indices, data.test_indices
    X_train, y_train = data.data[train], data.target[train]
    fitted = [fit_clone(generator, X_train, y_train) for generator in generators]

    # Warm users only: a generator that cannot serve a cold one would score it zero,
    # which says nothing about merging.
    X_test, y_test = data.data[test], data.target[test]
    known = np.unique(X_train[:, 0])
    keep = np.isin(X_test[:, 0], known) & (y_test > 0)
    held_users, held_items = X_test[keep, 0], X_test[keep, 1]
    users = np.unique(held_users)

    members = _members(fitted)
    recall: Cells = {"Generators": name}
    latency: Cells = {"Generators": name}
    for n in budgets:
        ceiling = _ceiling(fitted, users, held_users, held_items, n)
        cell = f"{ceiling:.3f}"
        if len(members) > 1:
            best = max(_ceiling([one], users, held_users, held_items, n) for one in members)
            cell += f" ({ceiling - best:+.3f})"
        recall[f"candidate recall@{n}"] = cell
        latency[f"median latency@{n}"] = format_latency(
            _request_seconds(fitted, users, n, settings)
        )
    return {"recall": recall, "latency": latency}


def _members(generators: Sequence[FittedRecommender]) -> list[FittedRecommender]:
    """The generators a set merges: its own, or those a fusion holds."""
    if len(generators) == 1 and isinstance(generators[0], ReciprocalRankFusion):
        return [member for _, member in generators[0].recommenders_]
    return list(generators)


def _retrieve(
    generators: Sequence[FittedRecommender], queries: NDArray[np.generic], n: int
) -> tuple[NDArray[np.generic], NDArray[np.int64], NDArray[np.intp]]:
    """What a Cascade retrieves for ``queries``: one generator's list, or the merge."""
    if len(generators) == 1 and isinstance(generators[0], ReciprocalRankFusion):
        # Every member lists as deep as the budget, as in the round-robin merge; the
        # depth is read only by ``recommend``, so the fitted fusion takes it as it is.
        generators[0].set_params(n_retrieved=n)
    if len(generators) == 1:
        pairs, _, groups, kept = retrieve(generators[0], queries, n_retrieved=n, min_retrieved=0)
    else:
        pairs, _, groups, kept = retrieve_union(
            list(generators), queries, n_retrieved=n, min_retrieved=0
        )
    return pairs, groups, kept


def _ceiling(
    generators: Sequence[FittedRecommender],
    users: NDArray[np.generic],
    held_users: NDArray[np.generic],
    held_items: NDArray[np.generic],
    n: int,
) -> float:
    """Mean over ``users`` of the share of their held-out items among the candidates."""
    pairs, _, _ = _retrieve(generators, users, n)
    item_ids, codes = factorize(concat_ids([pairs[:, 1], held_items]))
    candidate_items, held_codes = codes[: len(pairs)], codes[len(pairs) :]
    width = max(len(item_ids), 1)
    candidate_keys = lookup_ids(pairs[:, 0], users, name="user")[0].astype(np.int64) * width
    held_user = lookup_ids(held_users, users, name="user")[0]
    held_keys = held_user.astype(np.int64) * width + held_codes
    hit = np.isin(held_keys, candidate_keys + candidate_items)
    hits = np.bincount(held_user, weights=hit, minlength=len(users))
    return float(np.mean(hits / np.bincount(held_user, minlength=len(users))))


def _request_seconds(
    generators: Sequence[FittedRecommender],
    users: NDArray[np.generic],
    n: int,
    settings: spec.Settings,
) -> float:
    """Median time to retrieve one user's candidates, cycling over ``users``."""
    position = 0

    def once() -> None:
        nonlocal position
        start = position % len(users)
        _retrieve(generators, users[start : start + 1], n)
        position += 1

    for _ in range(settings["warmup"]["rank"]):
        once()
    position = 0
    seconds = leaderboard.sample(once, settings["latency_repeat"], settings["budget"])
    return float(np.median(seconds))
