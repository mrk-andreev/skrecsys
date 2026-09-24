"""The nn models declare a search space, and every point of it trains."""

import numpy as np
import pytest
from sklearn.base import clone

pytest.importorskip("torch")

from skrecsys.nn import HSTU, Mamba4Rec, SimpleX, XSimGCL
from skrecsys.tune import AutoTune, Study, search_space

#: Short fits, so a trial costs a few epochs; the space itself is left as declared.
MODELS = [
    SimpleX(max_iter=2, random_state=0),
    XSimGCL(max_iter=2, random_state=0),
    HSTU(max_sequence_length=8, max_iter=2, random_state=0),
    Mamba4Rec(max_sequence_length=8, max_iter=2, random_state=0),
]


def _interactions(n_users=30, n_items=20, per_user=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = [
        [f"u{u}", f"i{i}"]
        for u in range(n_users)
        for i in rng.choice(n_items, size=per_user, replace=False)
    ]
    return np.array(rows, dtype=object)


@pytest.mark.parametrize("estimator", MODELS, ids=lambda e: type(e).__name__)
def test_the_defaults_lie_inside_the_declared_space(estimator):
    space = search_space(estimator)
    assert space
    params = estimator.get_params()
    assert [name for name, dist in space.items() if not dist.contains(params[name])] == []


@pytest.mark.parametrize("estimator", MODELS, ids=lambda e: type(e).__name__)
def test_random_points_of_the_space_fit(estimator):
    """No combination of declared ranges is one the estimator rejects."""
    space = search_space(estimator)
    study = Study(sampler="random", random_state=0)
    X = _interactions()
    for _ in range(4):
        trial = study.ask()
        params = {name: trial.suggest(name, dist) for name, dist in space.items()}
        model = clone(estimator).set_params(**params).fit(X)
        items, _ = model.recommend(["u0"], n_recommendations=3)
        assert items.shape == (1, 3)
        study.tell(trial, 0.0)


@pytest.mark.parametrize("estimator", MODELS, ids=lambda e: type(e).__name__)
def test_autotune_wraps_the_model(estimator):
    tuned = AutoTune(estimator, n_trials=2, cv=2, random_state=0).fit(_interactions())
    assert set(tuned.best_params_) == set(search_space(estimator))
    assert type(tuned.best_estimator_) is type(estimator)
    items, _ = tuned.recommend(["u0", "u1"], n_recommendations=3)
    assert items.shape == (2, 3)
