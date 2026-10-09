"""Composites small enough for the shared fixture, used by several test modules."""

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from skrecsys.compose import (
    Backfill,
    Cascade,
    ConcatFeatures,
    GeneratorScores,
    ItemListRecommender,
    JoinStaticFeatures,
    KnownUser,
    MinInteractions,
    PointwiseRanker,
    ReciprocalRankFusion,
    ReciprocalRankRanker,
    ReservedSlots,
    Switch,
)
from skrecsys.recommendation import (
    AlternatingLeastSquares,
    ItemKNNRecommender,
    MostPopularRecommender,
)

#: One feature per fixture item except i5, so a row of NaN is exercised too.
ITEM_TABLE = np.array(
    [["i0", 0.0], ["i1", 1.0], ["i2", 0.0], ["i3", 1.0], ["i4", 0.5]], dtype=object
)

#: The fixture's items as a list of its own order, for the composites that take one: the
#: shared checks expect a recommender to know exactly the items it was fitted on.
ITEM_LIST = ["i5", "i3", "i0", "i4", "i1", "i2"]


def demote_i0(pairs, scores, groups):
    """A business rule for the fixture: i0 last in each list, scored by position."""
    group_of_row = np.repeat(np.arange(len(groups)), groups)
    order = np.lexsort((pairs[:, 1] == "i0", group_of_row))
    starts = np.concatenate([[0], np.cumsum(groups)[:-1]])
    position = np.arange(len(pairs)) - np.repeat(starts, groups)
    return pairs[order], -position.astype(float), groups


def cascade(**params):
    """A cascade for the fixture, holding out half of each user's history."""
    defaults: dict[str, Any] = {
        "generator": ItemKNNRecommender(),
        "features": GeneratorScores(),
        "ranker": PointwiseRanker(LogisticRegression()),
        "split": 0.5,
    }
    return Cascade(**{**defaults, **params})


COMPOSITES = [
    Switch(
        KnownUser(),
        AlternatingLeastSquares(n_factors=2, n_iter=5, random_state=0),
        MostPopularRecommender(),
    ),
    Switch(MinInteractions(3), ItemKNNRecommender(), MostPopularRecommender()),
    Switch(~KnownUser(), MostPopularRecommender(), ItemKNNRecommender()),
    cascade(),
    cascade(generator=MostPopularRecommender()),
    cascade(
        features=ConcatFeatures([JoinStaticFeatures("item", ITEM_TABLE), GeneratorScores()]),
        ranker=PointwiseRanker(HistGradientBoostingClassifier(max_iter=5)),
    ),
    Switch(KnownUser(), cascade(), MostPopularRecommender()),
    cascade(
        generator=[ItemKNNRecommender(), MostPopularRecommender()],
        features=GeneratorScores(n_generators=2),
        ranker=PointwiseRanker(HistGradientBoostingClassifier(max_iter=5)),
    ),
    ReciprocalRankFusion([ItemKNNRecommender(), MostPopularRecommender()]),
    ReciprocalRankFusion(
        [
            ("knn", ItemKNNRecommender()),
            ("als", AlternatingLeastSquares(n_factors=2, n_iter=5, random_state=0)),
        ],
        k=1,
        weights=[2.0, 1.0],
        n_retrieved=6,
    ),
    cascade(ranker=ReciprocalRankRanker()),
    cascade(
        generator=[ItemKNNRecommender(), MostPopularRecommender()],
        features=GeneratorScores(n_generators=2),
        ranker=ReciprocalRankRanker(),
    ),
    cascade(generator=ReciprocalRankFusion([ItemKNNRecommender(), MostPopularRecommender()])),
    cascade(postprocess=demote_i0),
    ItemListRecommender(ITEM_LIST),
    ItemListRecommender(ITEM_LIST, rotate=True, random_state=3),
    Backfill([ItemKNNRecommender(), MostPopularRecommender(), ItemListRecommender(ITEM_LIST)]),
    Backfill(
        [("ranked", cascade(n_retrieved=2)), ("catalog", ItemListRecommender(ITEM_LIST))],
        skip_insufficient=True,
    ),
    ReservedSlots(
        ItemKNNRecommender(), ItemListRecommender(ITEM_LIST, rotate=True), n_slots=1, head=2
    ),
    ReservedSlots(
        Backfill([Switch(KnownUser(), cascade(), MostPopularRecommender())]),
        ItemListRecommender(ITEM_LIST),
        n_slots=2,
        head=3,
    ),
]
