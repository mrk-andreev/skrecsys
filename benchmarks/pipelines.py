"""The pipelines the reranking report compares, built from the dataset they run on.

A leaderboard entry is a class and its parameters, which JSON can say. A reranking
pipeline also needs tables -- who each user is, what genres each movie has -- and those
come from the dataset, not from the config. So an entry of ``reranking.json`` names a
builder here instead of a class: ``builder(data, **params)`` returns the unfitted
pipeline, and the params stay in the config, where they key the result like any other.

Everything a builder reads from ``data`` beyond the interactions is MovieLens 100K's
``user_info`` and ``item_info``; the split the pipeline is fitted and scored on is the
dataset's, so nothing here can see a held-out row.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Protocol

import numpy as np
import spec
from numpy.typing import NDArray

from skrecsys.compose import (
    BlendRanker,
    Cascade,
    ConcatFeatures,
    GeneratorScores,
    InteractionCounts,
    JoinStaticFeatures,
    KnownUser,
    ReciprocalRankFusion,
    ReciprocalRankRanker,
    RecommenderScores,
    SegmentPopularity,
    Switch,
)
from skrecsys.model_selection import ColdStartSplit
from skrecsys.recommendation import EASE, BM25Recommender, MostPopularRecommender
from skrecsys.typing import Features, Ranker

#: The age bands MovieLens 1M publishes, applied to MovieLens 100K's exact ages.
AGE_BANDS = (18, 25, 35, 45, 50, 56)


class SideInfo(Protocol):
    """What a builder reads of MovieLens 100K beyond its interactions."""

    user_info: Mapping[str, NDArray[np.generic]]
    item_info: Mapping[str, NDArray[np.generic]]


def user_table(data: SideInfo) -> NDArray[np.float64]:
    """User id, age, gender (1 for female) and occupation code, one row per user."""
    users = data.user_info
    occupations = {name: code for code, name in enumerate(sorted(set(users["occupation"])))}
    return np.column_stack(
        [
            users["user_id"],
            users["age"],
            users["gender"] == "F",
            [occupations[name] for name in users["occupation"]],
        ]
    ).astype(np.float64)


def item_table(data: SideInfo) -> NDArray[np.float64]:
    """Item id and one 0/1 column per genre, one row per movie."""
    items = data.item_info
    return np.column_stack([items["item_id"], items["genres"]]).astype(np.float64)


def segments(data: SideInfo) -> NDArray[np.object_]:
    """User id, gender, age band and occupation: the segments a cold user has."""
    users = data.user_info
    table = np.empty((len(users["user_id"]), 4), dtype=object)
    table[:, 0] = users["user_id"]
    table[:, 1] = users["gender"]
    table[:, 2] = np.digitize(np.asarray(users["age"], dtype=np.float64), AGE_BANDS)
    table[:, 3] = users["occupation"]
    return table


def switch_bm25(data: object, *, k1: float = 1.2, b: float = 0.75) -> Switch:
    """The baseline: BM25 for the users it knows, the most popular items for the rest."""
    del data
    return Switch(KnownUser(), BM25Recommender(k1=k1, b=b), MostPopularRecommender())


def switch_bm25_catboost(
    data: SideInfo,
    *,
    n_retrieved: int = 100,
    iterations: int = 300,
    cold_users: float = 0.2,
    random_state: int = 0,
) -> Switch:
    """The same switch with a CatBoost ranker behind each branch.

    Known users: BM25 retrieves ``n_retrieved`` candidates, and CatBoost reorders them on
    who the user is, what the movie is, how much history each has, a second opinion from
    EASE and how popular the movie is in the user's segments. Cold users: the most popular
    items are the candidates, and CatBoost reorders them on segment popularity -- the one
    thing known about a user with no history. That ranker is fitted on users held out
    whole by :class:`ColdStartSplit`, so it learns from candidates exactly as a cold user
    is served them.
    """

    def ranker() -> Ranker:
        return _catboost(iterations, random_state)

    return _switch_reranked(data, ranker, n_retrieved, cold_users, random_state)


def switch_bm25_xgboost(
    data: SideInfo,
    *,
    n_retrieved: int = 100,
    cold_users: float = 0.2,
    random_state: int = 0,
    **xgboost: spec.JSON,
) -> Switch:
    """:func:`switch_bm25_catboost` with XGBoost's ``rank:ndcg`` as both rankers.

    ``xgboost`` holds the ranker's parameters, as :class:`XGBRanker` takes them.
    """

    def ranker() -> Ranker:
        return _booster("xgboost", "XGBRanker", xgboost, random_state)

    return _switch_reranked(data, ranker, n_retrieved, cold_users, random_state)


def switch_bm25_lightgbm(
    data: SideInfo,
    *,
    n_retrieved: int = 100,
    cold_users: float = 0.2,
    random_state: int = 0,
    **lightgbm: spec.JSON,
) -> Switch:
    """:func:`switch_bm25_catboost` with LightGBM's ``lambdarank`` as both rankers.

    ``lightgbm`` holds the ranker's parameters, as :class:`LGBMRanker` takes them.
    """

    def ranker() -> Ranker:
        return _booster("lightgbm", "LGBMRanker", lightgbm, random_state)

    return _switch_reranked(data, ranker, n_retrieved, cold_users, random_state)


def switch_bm25_blend(
    data: SideInfo,
    *,
    catboost: Mapping[str, spec.JSON],
    xgboost: Mapping[str, spec.JSON],
    lightgbm: Mapping[str, spec.JSON],
    n_retrieved: int = 100,
    cold_users: float = 0.2,
    normalize: str = "rank",
    cv: int = 3,
    random_state: int = 0,
) -> Switch:
    """:func:`switch_bm25_catboost` with the three boosters blended behind each branch.

    Each branch's :class:`BlendRanker` fits CatBoost, XGBoost and LightGBM -- each with
    the parameters of its own row -- ``cv`` times on all but one fold of users to score
    the fold left out, learns a logistic regression on those out-of-fold scores, and
    refits the three on every user for serving.
    """

    def ranker() -> Ranker:
        return BlendRanker(
            [
                ("catboost", _booster("catboost", "CatBoostRanker", catboost, random_state)),
                ("xgboost", _booster("xgboost", "XGBRanker", xgboost, random_state)),
                ("lightgbm", _booster("lightgbm", "LGBMRanker", lightgbm, random_state)),
            ],
            normalize=normalize,
            cv=cv,
            random_state=random_state,
        )

    return _switch_reranked(data, ranker, n_retrieved, cold_users, random_state)


def switch_ease(data: object) -> Switch:
    """EASE for the users it knows, the most popular items for the rest: BM25's partner
    in the fusions below, on its own."""
    del data
    return Switch(KnownUser(), EASE(), MostPopularRecommender())


def switch_bm25_ease_rrf(data: object, *, k: float = 60.0, n_retrieved: int = 100) -> Switch:
    """BM25 and EASE fused by reciprocal rank for known users: no ranker, nothing learned.

    Cold users get the most popular items, as in the baseline, so the rows differ only in
    how known users are served.
    """
    del data
    fusion = ReciprocalRankFusion(
        [("bm25", BM25Recommender()), ("ease", EASE())], k=k, n_retrieved=n_retrieved
    )
    return Switch(KnownUser(), fusion, MostPopularRecommender())


def switch_bm25_rrf(
    data: SideInfo,
    *,
    k: float = 60.0,
    n_retrieved: int = 100,
    cold_users: float = 0.2,
    random_state: int = 0,
) -> Switch:
    """:func:`switch_bm25_catboost` with a training-free :class:`ReciprocalRankRanker`.

    Known users: BM25's ``n_retrieved`` candidates, reordered by the reciprocal ranks of
    their BM25 and EASE scores. Cold users: the most popular items, reordered by the
    reciprocal ranks of their popularity and their popularity in the user's segments.
    """
    segment_table = segments(data)
    warm = Cascade(
        generator=BM25Recommender(),
        features=ConcatFeatures([("bm25", GeneratorScores()), ("ease", RecommenderScores(EASE()))]),
        ranker=ReciprocalRankRanker(k=k),
        n_retrieved=n_retrieved,
    )
    cold = Cascade(
        generator=MostPopularRecommender(),
        features=ConcatFeatures(
            [("popularity", GeneratorScores()), ("segment", SegmentPopularity(segment_table))]
        ),
        ranker=ReciprocalRankRanker(k=k),
        n_retrieved=n_retrieved,
        split=ColdStartSplit(cold_users, 0.0, random_state=random_state),
    )
    return Switch(KnownUser(), warm, cold)


def switch_bm25_rrf_boosters(
    data: SideInfo,
    *,
    catboost: Mapping[str, spec.JSON],
    xgboost: Mapping[str, spec.JSON],
    lightgbm: Mapping[str, spec.JSON],
    k: float = 60.0,
    n_retrieved: int = 100,
    cold_users: float = 0.2,
    random_state: int = 0,
) -> Switch:
    """:func:`switch_bm25_blend` with the boosters fused by rank instead of stacked.

    Each booster is fitted once on every user, and their ranks are fused: no out-of-fold
    refits and no blender to learn.
    """

    def ranker() -> Ranker:
        return ReciprocalRankRanker(
            [
                ("catboost", _booster("catboost", "CatBoostRanker", catboost, random_state)),
                ("xgboost", _booster("xgboost", "XGBRanker", xgboost, random_state)),
                ("lightgbm", _booster("lightgbm", "LGBMRanker", lightgbm, random_state)),
            ],
            k=k,
        )

    return _switch_reranked(data, ranker, n_retrieved, cold_users, random_state)


def _catboost(iterations: int, random_state: int) -> Ranker:
    return _booster("catboost", "CatBoostRanker", {"iterations": iterations}, random_state)


def _booster(extra: str, cls: str, params: Mapping[str, spec.JSON], random_state: int) -> Ranker:
    """A ranker of :mod:`skrecsys.integrations`, imported by name so that a builder needs
    only its own extra and the baseline builds with none."""
    module = importlib.import_module(f"skrecsys.integrations.{extra}")
    return getattr(module, cls)(**params, random_state=random_state)


def _switch_reranked(
    data: SideInfo,
    ranker: Callable[[], Ranker],
    n_retrieved: int,
    cold_users: float,
    random_state: int,
) -> Switch:
    """The reranked switch every booster row shares: only ``ranker`` differs between them."""
    users, items, segment_table = user_table(data), item_table(data), segments(data)
    warm_features: list[tuple[str, Features]] = [
        ("user", JoinStaticFeatures("user", users)),
        ("item", JoinStaticFeatures("item", items)),
        ("bm25", GeneratorScores()),
        ("user_count", InteractionCounts("user")),
        ("item_count", InteractionCounts("item")),
        ("ease", RecommenderScores(EASE())),
        ("segment", SegmentPopularity(segment_table)),
    ]
    cold_features: list[tuple[str, Features]] = [
        ("popularity", GeneratorScores()),
        ("segment", SegmentPopularity(segment_table)),
    ]
    warm = Cascade(
        generator=BM25Recommender(),
        features=ConcatFeatures(warm_features),
        ranker=ranker(),
        n_retrieved=n_retrieved,
    )
    cold = Cascade(
        generator=MostPopularRecommender(),
        features=ConcatFeatures(cold_features),
        ranker=ranker(),
        n_retrieved=n_retrieved,
        split=ColdStartSplit(cold_users, 0.0, random_state=random_state),
    )
    return Switch(KnownUser(), warm, cold)
