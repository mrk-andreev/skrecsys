"""Compose recommenders into pipelines, in the spirit of :mod:`sklearn.pipeline`.

Every composite here is itself a recommender, so composites nest, clone, pickle and take
part in :class:`~sklearn.model_selection.GridSearchCV` through nested parameters such as
``on_true__n_factors`` or ``ranker__estimator__C``. The parts play five roles:

- **recommenders** (:mod:`skrecsys.recommendation`, and :class:`Switch`, :class:`Cascade`,
  :class:`ReciprocalRankFusion`);
- **conditions** routing queries: :class:`KnownUser`, :class:`MinInteractions`,
  :class:`QueryIn`, combined with ``~``, ``&`` and ``|``;
- **candidates**, passed between stages as plain numpy arrays: ``pairs`` of shape
  ``(n_pairs, 2)`` -- ``(n_pairs, 3)``, with the time each is ranked as of, in a
  :class:`Cascade` constructed with ``time=True`` --, the generator ``scores`` and the
  group sizes ``groups``;
- **features** of candidate pairs: :class:`JoinStaticFeatures`,
  :class:`JoinDynamicFeatures`, :class:`GeneratorScores`, :class:`InteractionCounts`,
  :class:`RecommenderScores`, :class:`SegmentPopularity`, :class:`ConcatFeatures`;
- **rankers** ordering candidates: :class:`PointwiseRanker`, :class:`GroupRanker`,
  :class:`BlendRanker` (which blends other rankers), :class:`AugmentedRanker` (which
  gives one ranker features the others do not see), :class:`ReciprocalRankRanker` (which
  fuses feature columns or other rankers by rank, with nothing to learn), and
  the third-party rankers of :mod:`skrecsys.integrations` (CatBoost, XGBoost, LightGBM),
  each behind its own extra.

A :class:`Cascade` may also end with business rules: its ``postprocess`` callback turns
each query's ranked candidates into the list to serve.

Examples
--------
>>> import numpy as np
>>> from sklearn.linear_model import LogisticRegression
>>> from skrecsys.compose import (
...     Cascade, ConcatFeatures, GeneratorScores, JoinStaticFeatures, KnownUser,
...     PointwiseRanker, Switch,
... )
>>> from skrecsys.recommendation import BM25Recommender, MostPopularRecommender
>>> X = [[u, i] for u in range(30) for i in (u % 6, u % 6 + 1, (u + 2) % 6 + 2)]
>>> item_table = np.array([[i, i % 2] for i in range(8)], dtype=float)
>>> rec = Switch(
...     condition=KnownUser(),
...     on_true=Cascade(
...         generator=BM25Recommender(),
...         features=ConcatFeatures([JoinStaticFeatures("item", item_table), GeneratorScores()]),
...         ranker=PointwiseRanker(LogisticRegression()),
...         n_retrieved=4,
...         split=0.4,
...     ),
...     on_false=MostPopularRecommender(),
... ).fit(X)
>>> rec.recommend([0, 1000], n_recommendations=2)[0].shape
(2, 2)
"""

from skrecsys.base import AllOf, AnyOf, Not
from skrecsys.compose._cascade import Cascade
from skrecsys.compose._conditions import KnownUser, MinInteractions, QueryIn
from skrecsys.compose._features import (
    ConcatFeatures,
    GeneratorScores,
    InteractionCounts,
    JoinDynamicFeatures,
    JoinStaticFeatures,
    RecommenderScores,
    SegmentPopularity,
)
from skrecsys.compose._fusion import ReciprocalRankFusion, ReciprocalRankRanker
from skrecsys.compose._rankers import AugmentedRanker, BlendRanker, GroupRanker, PointwiseRanker
from skrecsys.compose._switch import Switch

__all__ = [
    "AllOf",
    "AnyOf",
    "AugmentedRanker",
    "BlendRanker",
    "Cascade",
    "ConcatFeatures",
    "GeneratorScores",
    "GroupRanker",
    "InteractionCounts",
    "JoinDynamicFeatures",
    "JoinStaticFeatures",
    "KnownUser",
    "MinInteractions",
    "Not",
    "PointwiseRanker",
    "QueryIn",
    "ReciprocalRankFusion",
    "ReciprocalRankRanker",
    "RecommenderScores",
    "SegmentPopularity",
    "Switch",
]
