"""The XGBRanker's own objectives, run only where the `xgboost` extra is installed."""

import numpy as np
import pytest

pytest.importorskip("xgboost")

from skrecsys.integrations.xgboost import XGBRanker

X = np.array([[0.0], [1.0], [0.2], [0.9], [0.1], [np.nan]])
Y = np.array([0, 1, 0, 1, 0, 1])
GROUPS = np.array([2, 2, 2])


@pytest.mark.parametrize("objective", ["rank:ndcg", "rank:pairwise", "rank:map"])
def test_every_objective(objective):
    ranker = XGBRanker(objective, n_estimators=20, min_child_weight=0, random_state=0)
    scores = ranker.fit(X, Y, groups=GROUPS).predict(X, groups=GROUPS)
    assert np.isfinite(scores).all()
    assert scores[1] > scores[0]
