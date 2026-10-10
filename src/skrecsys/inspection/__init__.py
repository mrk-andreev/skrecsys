"""Inspect what recommenders and pipelines did, and why.

:func:`trace` records every stage of the ``recommend`` calls made inside a ``with``
block -- the request, the branch a :class:`~skrecsys.compose.Switch` took, the candidates
each generator retrieved, the features and scores of a ranker, what a ``postprocess``
changed and what was served -- and gives it back query by query as a
:class:`QueryTrace`.

:func:`explain` builds on it to say why each item was served -- and why an item you
expected was not: which stage let it go, where each generator placed it, which features
weighed most with the ranker, and which of the user's history items made each leaf
recommender score it as it did.

Examples
--------
>>> from skrecsys.inspection import trace
>>> from skrecsys.recommendation import ItemKNNRecommender
>>> rec = ItemKNNRecommender().fit([["u1", "a"], ["u1", "b"], ["u2", "b"], ["u2", "c"]])
>>> with trace() as t:
...     _ = rec.recommend(["u1"], n_recommendations=1)
>>> print(t["u1"])
query 'u1', call 0
  ItemKNNRecommender: request n_recommendations=1
  ItemKNNRecommender: served ['c' 0.7071]
  ItemKNNRecommender: history reasons (exact); first item: ['b' 0.7071]
"""

from skrecsys.inspection._explain import (
    Explanation,
    LeafReasons,
    RankerDecision,
    Retrieval,
    Status,
    explain,
)
from skrecsys.inspection._trace import (
    AttributionStep,
    CandidatesStep,
    FeaturesStep,
    FusionStep,
    Level,
    PostprocessStep,
    QueryTrace,
    RaisedStep,
    RankerStep,
    RequestStep,
    RouteStep,
    Sample,
    ServedStep,
    Step,
    Trace,
    trace,
)

__all__ = [
    "AttributionStep",
    "CandidatesStep",
    "Explanation",
    "FeaturesStep",
    "FusionStep",
    "LeafReasons",
    "Level",
    "PostprocessStep",
    "QueryTrace",
    "RaisedStep",
    "RankerDecision",
    "RankerStep",
    "RequestStep",
    "Retrieval",
    "RouteStep",
    "Sample",
    "ServedStep",
    "Status",
    "Step",
    "Trace",
    "explain",
    "trace",
]
