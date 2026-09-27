"""The LGBMRanker's own objectives, run only where the `lightgbm` extra is installed."""

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from skrecsys.integrations.lightgbm import LGBMRanker
from tests.integrations._rankers import library_params

X = np.array([[0.0], [1.0], [0.2], [0.9], [0.1], [np.nan]])
Y = np.array([0, 1, 0, 1, 0, 1])
GROUPS = np.array([2, 2, 2])


@pytest.mark.parametrize("objective", ["lambdarank", "rank_xendcg", "binary"])
def test_every_objective(objective):
    ranker = LGBMRanker(objective, n_estimators=20, min_child_samples=1, random_state=0)
    scores = ranker.fit(X, Y, groups=GROUPS).predict(X, groups=GROUPS)
    assert np.isfinite(scores).all()
    assert scores[1] > scores[0]


def test_untuned_parameters_reach_the_model():
    ranker = LGBMRanker(n_estimators=5, min_child_samples=1, min_split_gain=0.5, subsample_freq=0)
    passed = library_params(ranker.fit(X, Y, groups=GROUPS))
    assert (passed["min_split_gain"], passed["subsample_freq"]) == (0.5, 0)
