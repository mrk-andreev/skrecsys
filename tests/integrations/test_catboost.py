"""The CatBoostRanker's own loss_functions, run only where the `catboost` extra is installed."""

import numpy as np
import pytest

pytest.importorskip("catboost")

from skrecsys.integrations.catboost import CatBoostRanker

X = np.array([[0.0], [1.0], [0.2], [0.9], [0.1], [np.nan]])
Y = np.array([0, 1, 0, 1, 0, 1])
GROUPS = np.array([2, 2, 2])


@pytest.mark.parametrize("loss_function", ["YetiRank", "PairLogit", "Logloss"])
def test_every_loss_function(loss_function):
    ranker = CatBoostRanker(loss_function, iterations=20, random_state=0)
    scores = ranker.fit(X, Y, groups=GROUPS).predict(X, groups=GROUPS)
    assert np.isfinite(scores).all()
    assert scores[1] > scores[0]
