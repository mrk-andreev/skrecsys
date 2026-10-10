"""Integrations with third-party libraries, each behind its own optional extra.

Every submodule adapts one external library to the skrecsys protocols and imports it on
import, so this package imports none of them itself; import the submodule you need:

- :mod:`skrecsys.integrations.catboost`: :class:`~skrecsys.integrations.catboost.CatBoostRanker`,
  requires ``pip install skrecsys[catboost]``;
- :mod:`skrecsys.integrations.xgboost`: :class:`~skrecsys.integrations.xgboost.XGBRanker`,
  requires ``pip install skrecsys[xgboost]``;
- :mod:`skrecsys.integrations.lightgbm`: :class:`~skrecsys.integrations.lightgbm.LGBMRanker`,
  requires ``pip install skrecsys[lightgbm]``.

Each is a ranker for :class:`~skrecsys.compose.Cascade`, taking the group sizes of the
candidates and missing features (NaN) as they are.

One submodule needs no extra, as it adapts no library but asyncio:

- :mod:`skrecsys.integrations.batching`: :class:`~skrecsys.integrations.batching.MicroBatcher`
  and :class:`~skrecsys.integrations.batching.AsyncRecommender`, which rank the requests of an
  async service that are in flight together in one ``recommend`` call.
"""
