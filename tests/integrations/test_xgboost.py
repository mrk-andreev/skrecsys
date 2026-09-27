"""The XGBRanker's own objectives, run only where the `xgboost` extra is installed."""

import json

import numpy as np
import pytest

pytest.importorskip("xgboost")

import xgboost

from skrecsys.integrations.xgboost import XGBRanker
from tests.integrations._rankers import library_params

X = np.array([[0.0], [1.0], [0.2], [0.9], [0.1], [np.nan]])
Y = np.array([0, 1, 0, 1, 0, 1])
GROUPS = np.array([2, 2, 2])


@pytest.mark.parametrize("objective", ["rank:ndcg", "rank:pairwise", "rank:map"])
def test_every_objective(objective):
    ranker = XGBRanker(objective, n_estimators=20, min_child_weight=0, random_state=0)
    scores = ranker.fit(X, Y, groups=GROUPS).predict(X, groups=GROUPS)
    assert np.isfinite(scores).all()
    assert scores[1] > scores[0]


def test_the_pair_count_is_passed_only_when_set():
    unset = xgboost.train({"objective": "rank:ndcg"}, xgboost.DMatrix(X, Y, group=GROUPS), 1)
    default = library_params(XGBRanker(n_estimators=1).fit(X, Y, groups=GROUPS))
    ranker = XGBRanker(n_estimators=1, lambdarank_num_pair_per_sample=2).fit(X, Y, groups=GROUPS)
    assert default["lambdarank_num_pair_per_sample"] == _pairs_of(unset)
    assert library_params(ranker)["lambdarank_num_pair_per_sample"] == 2


def _pairs_of(booster):
    objective = json.loads(booster.save_config())["learner"]["objective"]
    return int(objective["lambdarank_param"]["lambdarank_num_pair_per_sample"])
