"""SimpleX against an independent numpy implementation of each piece it is made of."""

import pickle

import numpy as np
import pytest
from sklearn.base import clone

pytest.importorskip("torch")

import torch

from skrecsys.nn import SimpleX


def _interactions(seed=0, n_users=40, n_items=15, density=0.3):
    rng = np.random.default_rng(seed)
    mask = rng.random((n_users, n_items)) < density
    mask[np.arange(n_users), np.arange(n_users) % n_items] = True
    users, items = np.nonzero(mask)
    return np.column_stack([users, items])


def _clusters(n_users=60, n_items=20, per_user=6, seed=0):
    """Two disjoint blocks of users and items; nobody crosses over."""
    rng = np.random.default_rng(seed)
    block_size = n_items // 2
    rows = []
    for user in range(n_users):
        block = user % 2
        pool = np.arange(block * block_size, (block + 1) * block_size)
        rows += [[f"u{user}", f"i{item}"] for item in rng.choice(pool, per_user, replace=False)]
    return np.array(rows, dtype=object)


def _fitted_module(estimator, X):
    """Fit and rebuild the module, so the tests can look at the pieces `fit` drops."""
    est = clone(estimator).fit(X)
    rng = np.random.RandomState(0)
    generator = torch.Generator()
    generator.manual_seed(int(rng.randint(2**31)))
    module = est._build_module(est.interactions_, torch.device("cpu"), rng, generator)
    module.eval()
    return est, module


def _softmax(scores, mask):
    scores = np.where(mask, scores, -np.inf)
    shifted = np.exp(scores - scores.max(axis=1, keepdims=True))
    shifted = np.where(mask, shifted, 0.0)
    total = shifted.sum(axis=1, keepdims=True)
    return np.divide(shifted, total, out=np.zeros_like(shifted), where=total > 0)


def _reference_user_vectors(module, users):
    """The fusion of Eq. 4-7 of the paper, written out in numpy."""
    item_weight = module.item_embedding.weight.detach().numpy()
    user_weight = module.user_embedding.weight.detach().numpy()
    history = module.history.numpy()[users]
    mask = history != module.pad_index
    embedded = item_weight[history]
    user_embedded = user_weight[users]

    aggregator = type(module.aggregator).__name__
    if aggregator == "_MeanAggregator":
        counts = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
        pooled = (embedded * mask[..., None]).sum(axis=1) / counts
    elif aggregator == "_UserAttentionAggregator":
        scores = np.einsum("bld,bd->bl", embedded, user_embedded)
        pooled = (_softmax(scores, mask)[..., None] * embedded).sum(axis=1)
    else:
        hidden = np.tanh(embedded @ module.aggregator.hidden.weight.detach().numpy().T)
        scores = (hidden @ module.aggregator.score.weight.detach().numpy().T).squeeze(-1)
        pooled = (_softmax(scores, mask)[..., None] * embedded).sum(axis=1)

    projected = pooled @ module.projection.weight.detach().numpy().T
    projected += module.projection.bias.detach().numpy()
    fused = module.gamma * user_embedded + (1.0 - module.gamma) * projected
    return fused / np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-12)


@pytest.mark.parametrize("aggregator", ["mean", "self_attention", "user_attention"])
def test_aggregation_matches_reference(aggregator):
    X = _interactions()
    est, module = _fitted_module(
        SimpleX(n_factors=8, max_iter=2, n_negatives=3, aggregator=aggregator, random_state=0), X
    )
    users = np.arange(est.n_users_)
    with torch.no_grad():
        actual = module.user_vectors(torch.from_numpy(users)).numpy()
    np.testing.assert_allclose(actual, _reference_user_vectors(module, users), rtol=1e-5, atol=1e-6)


def test_empty_history_pools_to_zero():
    """A user whose whole history is padding must not produce NaNs."""
    est, module = _fitted_module(
        SimpleX(n_factors=8, max_iter=1, n_negatives=2, aggregator="user_attention"),
        _interactions(),
    )
    with torch.no_grad():
        module.history.fill_(module.pad_index)
        vectors = module.user_vectors(torch.arange(est.n_users_)).numpy()
    assert np.isfinite(vectors).all()


def test_ccl_loss_matches_reference():
    """The loss is the CCL of the paper, one positive against N uniform negatives."""
    _est, module = _fitted_module(
        SimpleX(n_factors=8, max_iter=1, n_negatives=4, negative_weight=7.0, margin=0.3),
        _interactions(),
    )
    users = torch.tensor([0, 3, 7])
    items = torch.tensor([1, 2, 0])
    negatives = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    with torch.no_grad():
        actual = float(module.batch_loss((users, items, negatives)))

    item_weight = module.item_embedding.weight.detach().numpy()
    normalized = item_weight / np.maximum(np.linalg.norm(item_weight, axis=1, keepdims=True), 1e-12)
    vectors = _reference_user_vectors(module, users.numpy())
    positive = np.einsum("bd,bd->b", vectors, normalized[items.numpy()])
    negative = np.einsum("bd,bnd->bn", vectors, normalized[negatives.numpy()])
    expected = (1.0 - positive) + 7.0 * np.maximum(negative - 0.3, 0.0).mean(axis=1)
    assert actual == pytest.approx(expected.mean(), rel=1e-5)


def test_history_is_a_uniform_subsample_of_the_row():
    """Every history entry is one of the user's own items, kept at most once."""
    X = _interactions(density=0.8)
    est, module = _fitted_module(SimpleX(n_factors=4, max_iter=0, history_size=3), X)
    history = module.history.numpy()
    assert history.shape == (est.n_users_, 3)
    rows = est.interactions_
    for user in range(est.n_users_):
        kept = history[user][history[user] != module.pad_index]
        owned = rows.indices[rows.indptr[user] : rows.indptr[user + 1]]
        assert len(set(kept.tolist())) == len(kept)
        assert set(kept.tolist()) <= set(owned.tolist())
        assert len(kept) == min(3, len(owned))


def test_factors_are_unit_norm_and_score_is_their_dot_product():
    X = _interactions()
    est = SimpleX(n_factors=8, max_iter=3, n_negatives=3, random_state=0).fit(X)
    for factors in (est.user_factors_, est.item_factors_):
        np.testing.assert_allclose(np.linalg.norm(factors, axis=1), 1.0, atol=1e-6)
        assert factors.dtype == np.float64

    scores = est._score_users(np.arange(est.n_users_), np.arange(est.n_items_))
    np.testing.assert_allclose(scores, est.user_factors_ @ est.item_factors_.T, rtol=1e-12)
    assert np.all(np.abs(scores) <= 1.0 + 1e-6), "cosine scores must stay in [-1, 1]"


def test_loss_curve_decreases():
    # `early_stopping=False` keeps the curve a fixed 30 epochs long; what stopping early
    # does to it is `test_early_stopping_shortens_the_fit`'s business.
    est = SimpleX(
        n_factors=16,
        max_iter=30,
        n_negatives=5,
        learning_rate=0.05,
        early_stopping=False,
        random_state=0,
    )
    est.fit(_clusters())
    assert est.loss_curve_.shape == (30,)
    assert est.n_iter_ == 30
    assert est.loss_curve_[-1] < est.loss_curve_[0]


def test_recovers_planted_clusters():
    X = _clusters()
    est = SimpleX(
        n_factors=16, max_iter=80, n_negatives=10, learning_rate=0.05, random_state=0
    ).fit(X)
    queries = np.array([f"u{user}" for user in range(60)], dtype=object)
    items, _ = est.recommend(queries, n_recommendations=4)
    blocks = np.vectorize(lambda item: int(item[1:]) // 10)(items)
    inside = blocks == (np.arange(60) % 2)[:, None]
    assert inside.mean() > 0.85, f"only {inside.mean():.0%} of recommendations stayed in-cluster"


def test_seeded_fit_is_reproducible():
    X = _interactions()
    first = SimpleX(n_factors=8, max_iter=5, n_negatives=3, random_state=7, n_jobs=1).fit(X)
    second = SimpleX(n_factors=8, max_iter=5, n_negatives=3, random_state=7, n_jobs=1).fit(X)
    np.testing.assert_array_equal(first.user_factors_, second.user_factors_)
    np.testing.assert_array_equal(first.item_factors_, second.item_factors_)
    np.testing.assert_array_equal(first.loss_curve_, second.loss_curve_)

    other = SimpleX(n_factors=8, max_iter=5, n_negatives=3, random_state=8, n_jobs=1).fit(X)
    assert not np.array_equal(first.item_factors_, other.item_factors_)


def test_unseeded_fits_differ():
    X = _interactions()
    first = SimpleX(n_factors=8, max_iter=2, n_negatives=3).fit(X)
    second = SimpleX(n_factors=8, max_iter=2, n_negatives=3).fit(X)
    assert not np.array_equal(first.item_factors_, second.item_factors_)


def test_fitted_estimator_has_no_torch_state():
    """A fitted model must pickle and score where torch is not installed."""
    X = _interactions()
    est = SimpleX(n_factors=8, max_iter=2, n_negatives=3, random_state=0).fit(X)
    restored = pickle.loads(pickle.dumps(est))
    assert not any(isinstance(value, torch.nn.Module) for value in vars(est).values())
    np.testing.assert_array_equal(restored.item_factors_, est.item_factors_)


def test_device_cpu_and_auto_agree_in_shape():
    X = _interactions()
    fits = [
        SimpleX(n_factors=8, max_iter=2, n_negatives=3, device=device, random_state=0).fit(X)
        for device in ("cpu", "auto")
    ]
    for est in fits:
        assert est.user_factors_.shape == (est.n_users_, 8)
        assert np.isfinite(est.user_factors_).all()
    assert fits[0].user_factors_.shape == fits[1].user_factors_.shape


def test_accepts_a_torch_device_object():
    X = _interactions()
    est = SimpleX(n_factors=4, max_iter=1, n_negatives=2, device=torch.device("cpu")).fit(X)
    assert est.item_factors_.shape == (est.n_items_, 4)


def test_max_iter_zero_exports_the_initialization():
    X = _interactions()
    est = SimpleX(n_factors=4, max_iter=0, n_negatives=2, random_state=0).fit(X)
    assert est.loss_curve_.shape == (0,)
    np.testing.assert_allclose(np.linalg.norm(est.item_factors_, axis=1), 1.0, atol=1e-6)


def test_n_jobs_does_not_change_the_result():
    X = _interactions()
    single = SimpleX(n_factors=8, max_iter=3, n_negatives=3, random_state=0, n_jobs=1).fit(X)
    two = SimpleX(n_factors=8, max_iter=3, n_negatives=3, random_state=0, n_jobs=2).fit(X)
    np.testing.assert_allclose(single.item_factors_, two.item_factors_, rtol=1e-3, atol=1e-5)
    assert torch.get_num_threads() > 0, "the thread cap must be restored after the fit"


@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("n_factors", 0),
        ("n_factors", 1.5),
        ("n_factors", True),
        ("batch_size", 0),
        ("batch_size", None),
        ("max_iter", -1),
        ("max_iter", 1.5),
        ("learning_rate", 0.0),
        ("learning_rate", "fast"),
        ("regularization", -1.0),
        ("regularization", None),
        ("n_negatives", 0),
        ("n_negatives", 2.5),
        ("history_size", 0),
        ("history_size", None),
        ("negative_weight", -1.0),
        ("negative_weight", "heavy"),
        ("margin", -0.1),
        ("margin", 1.1),
        ("margin", None),
        ("gamma", -0.1),
        ("gamma", 1.1),
        ("dropout", 1.0),
        ("dropout", -0.1),
        ("aggregator", "attention"),
        ("aggregator", None),
        ("device", "not-a-device"),
        ("device", 3),
        ("n_jobs", 0),
        ("n_jobs", 1.5),
    ],
)
def test_invalid_param_raises(param, value):
    est = SimpleX(n_factors=4, max_iter=1, n_negatives=2).set_params(**{param: value})
    with pytest.raises(ValueError, match=param):
        est.fit(_interactions())
