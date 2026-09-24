"""Every integration ranker, built small enough to fit the tests' tiny fixtures."""

import importlib

import pytest

#: (extra, class name, parameters keeping a fit fast and able to split six rows).
RANKERS = [
    ("catboost", "CatBoostRanker", {"iterations": 20}),
    ("xgboost", "XGBRanker", {"n_estimators": 20, "min_child_weight": 0}),
    ("lightgbm", "LGBMRanker", {"n_estimators": 20, "min_child_samples": 1}),
]


def make_ranker(extra, name, params, *args, **overrides):
    """Build the ranker, skipping the test where its extra is not installed."""
    pytest.importorskip(extra)
    cls = getattr(importlib.import_module(f"skrecsys.integrations.{extra}"), name)
    return cls(*args, **{**params, **overrides})
