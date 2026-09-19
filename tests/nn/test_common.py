import numpy as np
import pytest
from sklearn.base import clone

pytest.importorskip("torch")

import torch

from skrecsys.nn import HSTU, Mamba4Rec, SimpleX, XSimGCL
from skrecsys.recommendation._base import BaseRecommender
from tests.estimator_checks import (
    _ITEMS,
    _NEXT_ITEMS,
    _interactions,
    _next_interactions,
    check_pandas_input_matches_numpy,
    yield_incremental_recommender_checks,
    yield_recommender_checks,
)

IMPLEMENTED = [
    SimpleX(n_factors=4, max_iter=3, n_negatives=2, random_state=0),
    SimpleX(n_factors=4, max_iter=3, n_negatives=2, aggregator="self_attention", random_state=0),
    SimpleX(n_factors=4, max_iter=3, n_negatives=2, aggregator="user_attention", random_state=0),
    SimpleX(n_factors=4, max_iter=2, n_negatives=2, gamma=1.0, batch_size=2, random_state=0),
    SimpleX(n_factors=4, max_iter=2, n_negatives=2, history_size=1, n_jobs=1, random_state=0),
    XSimGCL(n_factors=4, n_layers=2, max_iter=3, random_state=0),
    XSimGCL(n_factors=4, n_layers=1, contrastive_layer=0, max_iter=2, random_state=0),
    XSimGCL(n_factors=4, n_layers=3, contrastive_layer=3, eps=0.0, max_iter=2, random_state=0),
    XSimGCL(n_factors=4, n_layers=2, max_iter=2, batch_size=2, n_jobs=1, random_state=0),
    HSTU(n_factors=4, max_sequence_length=4, max_iter=3, n_negatives=2, random_state=0),
    HSTU(n_factors=4, n_blocks=1, n_heads=2, head_dim=2, max_sequence_length=2, random_state=0),
    HSTU(n_factors=4, max_sequence_length=3, max_iter=2, dropout=0.0, batch_size=2, n_jobs=1),
    Mamba4Rec(n_factors=4, max_sequence_length=4, max_iter=3, random_state=0),
    Mamba4Rec(
        n_factors=8,
        n_blocks=2,
        d_state=2,
        d_conv=2,
        expand=1,
        dt_rank=2,
        max_sequence_length=2,
        max_iter=2,
        random_state=0,
    ),
    Mamba4Rec(n_factors=4, max_sequence_length=3, max_iter=2, dropout=0.0, batch_size=2, n_jobs=1),
]


@pytest.mark.parametrize("estimator", IMPLEMENTED, ids=repr)
@pytest.mark.parametrize("check", list(yield_recommender_checks()), ids=lambda c: c.__name__)
def test_common_checks(estimator, check):
    check(type(estimator).__name__, estimator)


@pytest.mark.parametrize("estimator", IMPLEMENTED, ids=repr)
def test_pandas_input_matches_numpy(estimator):
    check_pandas_input_matches_numpy(type(estimator).__name__, estimator)


#: One configuration of each model, seeded, for the incremental checks.
INCREMENTAL = [
    SimpleX(n_factors=4, max_iter=3, n_negatives=2, random_state=0),
    SimpleX(n_factors=4, max_iter=2, n_negatives=2, aggregator="self_attention", random_state=0),
    XSimGCL(n_factors=4, n_layers=2, max_iter=3, random_state=0),
    HSTU(n_factors=4, max_sequence_length=4, max_iter=3, n_negatives=2, random_state=0),
    Mamba4Rec(n_factors=4, max_sequence_length=4, max_iter=3, random_state=0),
]


@pytest.mark.parametrize("estimator", INCREMENTAL, ids=repr)
@pytest.mark.parametrize(
    "check", list(yield_incremental_recommender_checks()), ids=lambda c: c.__name__
)
def test_incremental_checks(estimator, check):
    check(type(estimator).__name__, estimator)


@pytest.mark.parametrize("estimator", INCREMENTAL, ids=repr)
def test_a_plain_fit_can_be_continued(estimator):
    """`fit` then `partial_fit` is the same interface every other recommender has.

    It works because what `fit` keeps is the training state as numpy, not a live module:
    the module is rebuilt from the grown data on every call anyway, so there is nothing
    about having fitted in one go that a later batch cannot pick up.
    """
    est = clone(estimator).fit(_interactions())
    est.partial_fit(_next_interactions())
    np.testing.assert_array_equal(est.item_ids_, np.unique(_ITEMS + _NEXT_ITEMS))
    items, _ = est.recommend(np.array(["u4"], dtype=object), n_recommendations=2)
    assert items.shape == (1, 2)


@pytest.mark.parametrize("estimator", INCREMENTAL, ids=repr)
def test_a_fitted_model_holds_no_torch(estimator):
    """Nothing torch-typed survives a fit, which is what keeps the pickle portable.

    The check is on the whole instance rather than on a filtered `__getstate__`: there
    is no filter any more, so anything torch-typed left on the estimator would go
    straight into the pickle and make it unreadable without the extra installed.
    """
    est = clone(estimator).partial_fit(_interactions())
    for name, value in vars(est).items():
        assert not torch.is_tensor(value), name
        assert not isinstance(value, torch.nn.Module | torch.optim.Optimizer), name
    learned, moments = est._training_state_
    assert all(isinstance(v, np.ndarray) for v in learned.values())
    assert all(isinstance(v["exp_avg"], np.ndarray) for v in moments.values())


@pytest.mark.parametrize("estimator", INCREMENTAL, ids=repr)
def test_partial_fit_carries_the_training_across(estimator):
    """A batch with no epochs in it must leave the model exactly as it was.

    Rebuilding the module is how `partial_fit` keeps every derived buffer honest, and it
    is also the easy way to lose what was learned. Zero epochs isolates the transfer:
    whatever comes out is what went in, or the parameters were not carried.
    """
    est = clone(estimator).partial_fit(_interactions())
    table = "item_factors_" if hasattr(est, "item_factors_") else "item_embeddings_"
    before = getattr(est, table).copy()
    est.set_params(max_iter=0).partial_fit(_interactions()[:3])
    np.testing.assert_allclose(getattr(est, table), before)


@pytest.mark.parametrize("estimator", INCREMENTAL, ids=repr)
def test_partial_fit_carries_adams_moments(estimator):
    """Every parameter keeps its moments and its step count across a batch.

    Adam keys its state by the parameter object, and `partial_fit` builds new ones. A
    batch that started from a zero second moment would take its first steps as large as
    the step size allows, on a model that had already converged.
    """
    est = clone(estimator).partial_fit(_interactions())
    learned, moments = est._training_state_
    steps = {state["step"] for state in moments.values()}
    est.partial_fit(_next_interactions())

    learned, moments = est._training_state_
    assert set(moments) == set(learned), "a parameter lost its moments across the batch."
    assert all(state["step"] > max(steps) for state in moments.values())
    assert all(np.abs(state["exp_avg_sq"]).sum() > 0.0 for state in moments.values())


@pytest.mark.parametrize("estimator", IMPLEMENTED, ids=repr)
def test_predict_matches_full_catalog_scoring(estimator):
    """`_score_pairs` must agree with scoring the whole catalog and reading cells back."""
    users = ["u0", "u0", "u1", "u1", "u1", "u2", "u2", "u3", "u3", "u3"]
    items = ["i0", "i1", "i1", "i2", "i3", "i0", "i4", "i2", "i4", "i5"]
    X = np.column_stack([users, items]).astype(object)
    est = clone(estimator).fit(X)

    pairs = np.array([[u, i] for u in sorted(set(users)) for i in sorted(set(items))], dtype=object)
    user_idx = np.searchsorted(est.user_ids_, pairs[:, 0])
    item_idx = np.searchsorted(est.item_ids_, pairs[:, 1])
    expected = BaseRecommender._score_pairs(est, user_idx, item_idx)

    np.testing.assert_allclose(est._score_pairs(user_idx, item_idx), expected, rtol=1e-12)
    np.testing.assert_allclose(est.predict(pairs), expected, rtol=1e-12)


#: One cheap, correctly-shaped estimator of each class, for the training-loop tests below.
#: They exercise `TorchRecommender._train`, which every one of them shares, so what is
#: under test is the stopping rule rather than any particular model.
ONE_OF_EACH = [
    SimpleX(n_factors=4, max_iter=40, n_negatives=2, random_state=0),
    XSimGCL(n_factors=4, n_layers=2, max_iter=40, random_state=0),
    HSTU(n_factors=4, max_sequence_length=4, max_iter=40, n_negatives=2, random_state=0),
    Mamba4Rec(n_factors=4, max_sequence_length=4, max_iter=40, random_state=0),
]


def _training_data():
    """A handful of users with repeated items, enough for one epoch of every model."""
    users = [f"u{user}" for user in range(8) for _ in range(4)]
    items = [f"i{(user + step) % 6}" for user in range(8) for step in range(4)]
    return np.column_stack([users, items]).astype(object)


@pytest.mark.parametrize("estimator", ONE_OF_EACH, ids=repr)
def test_n_iter_and_best_loss_describe_the_curve(estimator):
    """The two fitted scalars are the curve's length and its minimum, by definition."""
    est = clone(estimator).fit(_training_data())
    assert est.n_iter_ == len(est.loss_curve_)
    assert est.best_loss_ == pytest.approx(est.loss_curve_.min())
    assert est.n_iter_ <= est.get_params()["max_iter"]


@pytest.mark.parametrize("estimator", ONE_OF_EACH, ids=repr)
def test_a_fit_that_may_not_stop_runs_every_epoch(estimator):
    """`early_stopping=False` with a constant step size is the old loop, unchanged."""
    est = (
        clone(estimator)
        .set_params(early_stopping=False, learning_rate_schedule="constant")
        .fit(_training_data())
    )
    assert est.n_iter_ == est.get_params()["max_iter"]


@pytest.mark.parametrize("estimator", ONE_OF_EACH, ids=repr)
def test_early_stopping_ends_a_fit_that_cannot_improve(estimator):
    """A `tol` no epoch can ever meet stalls the fit immediately, so it stops at `patience`.

    This pins the arithmetic of the rule rather than a model's convergence: with the step
    size held constant there is nothing to do about a stall but stop, and it takes
    `n_iter_no_change` stalled epochs to decide that. The first epoch is never one of
    them -- it has no previous best to fail against -- so the fit runs `patience + 1`.
    """
    est = (
        clone(estimator)
        .set_params(
            early_stopping=True, learning_rate_schedule="constant", tol=1e9, n_iter_no_change=3
        )
        .fit(_training_data())
    )
    assert est.n_iter_ == 4


@pytest.mark.parametrize("estimator", ONE_OF_EACH, ids=repr)
def test_the_adaptive_schedule_spends_the_step_size_before_stopping(estimator):
    """Adaptive must outlast constant: every cut buys the fit another `n_iter_no_change`.

    Same unmeetable `tol`, so both fits stall on every epoch. The constant one has no
    move left and stops at `patience`; the adaptive one keeps dividing the step size by
    five until it reaches the 1e-6 floor, which is several more rounds of patience.
    """
    params = {"early_stopping": True, "tol": 1e9, "n_iter_no_change": 3}
    constant = clone(estimator).set_params(learning_rate_schedule="constant", **params)
    adaptive = clone(estimator).set_params(learning_rate_schedule="adaptive", **params)
    data = _training_data()
    assert adaptive.fit(data).n_iter_ > constant.fit(data).n_iter_


@pytest.mark.parametrize("estimator", ONE_OF_EACH, ids=repr)
@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"learning_rate_schedule": "cosine"}, "learning_rate_schedule must be one of"),
        ({"learning_rate_schedule": None}, "learning_rate_schedule must be one of"),
        ({"early_stopping": 1}, "early_stopping must be a boolean"),
        ({"tol": -1e-4}, "tol must be a finite real number >= 0"),
        ({"tol": float("nan")}, "tol must be a finite real number >= 0"),
        ({"tol": True}, "tol must be a finite real number >= 0"),
        ({"n_iter_no_change": 0}, "n_iter_no_change must be an integer >= 1"),
        ({"n_iter_no_change": 1.5}, "n_iter_no_change must be an integer >= 1"),
    ],
)
def test_invalid_stopping_parameters(estimator, params, match):
    """The stopping parameters are validated once in the base, so once for every model."""
    with pytest.raises(ValueError, match=match):
        clone(estimator).set_params(**params).fit(_training_data())
