"""Every integration ranker, built small enough to fit the tests' tiny fixtures."""

import importlib
import json

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


def library_params(ranker):
    """What the fitted ranker's library was trained with, keyed as the ranker names it.

    CatBoost reports its parameters as given; LightGBM's booster keeps them as passed to
    ``lightgbm.train``; XGBoost's lives in its JSON config, as strings, which come back as
    numbers where they parse. The number of boosting rounds requested, an argument of
    ``train`` rather than a parameter, is reported as ``n_estimators``.
    """
    model = ranker.model_
    if type(ranker).__name__ == "CatBoostRanker":
        return model.get_params()
    if type(ranker).__name__ == "LGBMRanker":
        # The rounds asked for: on tiny data LightGBM stops early once no split is left.
        return {**model.params, "n_estimators": model.params["num_iterations"]}
    params = {}
    _flatten(json.loads(model.save_config()), params)
    return {**params, "n_estimators": model.num_boosted_rounds()}


def _flatten(config, into):
    for key, value in config.items():
        if isinstance(value, dict):
            _flatten(value, into)
        else:
            into[key] = _number(value)


def _number(value):
    for parse in (int, float):
        try:
            return parse(value)
        except (TypeError, ValueError):
            continue
    return value
