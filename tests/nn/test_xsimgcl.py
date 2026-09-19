"""XSimGCL against an independent numpy implementation of each piece it is made of."""

import pickle

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone

pytest.importorskip("torch")

import torch

from skrecsys.nn import XSimGCL
from skrecsys.nn._xsimgcl import info_nce, normalized_adjacency


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


def test_normalized_adjacency_matches_scipy():
    """`D^-1/2 A D^-1/2` over the bipartite graph, written out with scipy."""
    X = _interactions()
    est = XSimGCL(n_factors=4, n_layers=1, max_iter=0).fit(X)
    interactions = est.interactions_
    n_users, n_items = interactions.shape

    structure = (interactions > 0).toarray().astype(np.float64)
    adjacency = np.block(
        [
            [np.zeros((n_users, n_users)), structure],
            [structure.T, np.zeros((n_items, n_items))],
        ]
    )
    degree = adjacency.sum(axis=1)
    scale = np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1e-12)), 0.0)
    expected = adjacency * scale[:, None] * scale[None, :]

    actual = normalized_adjacency(interactions).to_dense().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_adjacency_is_symmetric_and_bipartite():
    est = XSimGCL(n_factors=4, n_layers=1, max_iter=0).fit(_interactions())
    dense = normalized_adjacency(est.interactions_).to_dense().numpy()
    np.testing.assert_allclose(dense, dense.T, atol=1e-7)
    # No user-user or item-item edge: the graph only ever links the two sides.
    assert not dense[: est.n_users_, : est.n_users_].any()
    assert not dense[est.n_users_ :, est.n_users_ :].any()


def test_isolated_node_does_not_divide_by_zero():
    """An item nobody touched keeps a zero row rather than a NaN one."""
    interactions = sp.csr_array(np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]))
    dense = normalized_adjacency(interactions).to_dense().numpy()
    assert np.isfinite(dense).all()
    assert not dense[:, -1].any()


@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_propagation_matches_reference(n_layers):
    """Noise-free propagation is the layer average of repeated `A @ E`."""
    X = _interactions()
    est, module = _fitted_module(
        XSimGCL(n_factors=8, n_layers=n_layers, max_iter=2, random_state=0), X
    )
    with torch.no_grad():
        ranking, contrastive = module.propagate(perturbed=False)

    adjacency = normalized_adjacency(est.interactions_).to_dense().numpy().astype(np.float64)
    embeddings = np.concatenate(
        [
            module.user_embedding.weight.detach().numpy(),
            module.item_embedding.weight.detach().numpy(),
        ]
    ).astype(np.float64)
    layers = []
    for _ in range(n_layers):
        embeddings = adjacency @ embeddings
        layers.append(embeddings)

    np.testing.assert_allclose(ranking.numpy(), np.mean(layers, axis=0), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(contrastive.numpy(), layers[0], rtol=1e-4, atol=1e-5)


def test_contrastive_layer_selects_that_layer():
    X = _interactions()
    _est, module = _fitted_module(XSimGCL(n_factors=8, n_layers=3, contrastive_layer=3), X)
    with torch.no_grad():
        ranking, contrastive = module.propagate(perturbed=False)
        embeddings = torch.cat([module.user_embedding.weight, module.item_embedding.weight])
        for _ in range(3):
            embeddings = torch.sparse.mm(module.adjacency, embeddings)
    np.testing.assert_allclose(contrastive.numpy(), embeddings.numpy(), rtol=1e-5, atol=1e-6)
    assert not np.allclose(contrastive.numpy(), ranking.numpy())


def test_contrastive_layer_zero_is_the_raw_embeddings():
    X = _interactions()
    _est, module = _fitted_module(XSimGCL(n_factors=8, n_layers=2, contrastive_layer=0), X)
    with torch.no_grad():
        _, contrastive = module.propagate(perturbed=False)
        expected = torch.cat([module.user_embedding.weight, module.item_embedding.weight])
    np.testing.assert_allclose(contrastive.numpy(), expected.numpy(), rtol=1e-6)


def test_perturbation_has_the_requested_length_and_sign():
    """Each layer's noise is a unit vector scaled by `eps`, aligned with the embedding."""
    X = _interactions()
    _est, module = _fitted_module(XSimGCL(n_factors=8, n_layers=1, eps=0.3), X)
    with torch.no_grad():
        clean, _ = module.propagate(perturbed=False)
        noisy, _ = module.propagate(perturbed=True)
    delta = (noisy - clean).numpy()
    lengths = np.linalg.norm(delta, axis=1)
    live = np.linalg.norm(clean.numpy(), axis=1) > 0
    np.testing.assert_allclose(lengths[live], 0.3, rtol=1e-4)
    # The noise never flips a coordinate's sign, which is what keeps the view usable.
    sign = np.sign(clean.numpy())
    assert np.all(np.sign(delta)[sign != 0] == sign[sign != 0])


def test_eps_zero_is_a_plain_lightgcn_pass():
    X = _interactions()
    _est, module = _fitted_module(XSimGCL(n_factors=8, n_layers=2, eps=0.0), X)
    with torch.no_grad():
        clean, _ = module.propagate(perturbed=False)
        noisy, _ = module.propagate(perturbed=True)
    np.testing.assert_allclose(noisy.numpy(), clean.numpy(), rtol=1e-6)


def test_info_nce_matches_reference():
    rng = np.random.default_rng(0)
    first, second = rng.normal(size=(6, 5)), rng.normal(size=(6, 5))
    actual = float(info_nce(torch.tensor(first), torch.tensor(second), 0.3))

    def unit(values):
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    view1, view2 = unit(first), unit(second)
    positive = np.exp(np.einsum("ij,ij->i", view1, view2) / 0.3)
    total = np.exp(view1 @ view2.T / 0.3).sum(axis=1)
    assert actual == pytest.approx(float(np.mean(-np.log(positive / total))), rel=1e-6)


def test_info_nce_is_minimal_for_identical_views():
    """Agreement is the best the term can do, and it is the same for any aligned pair."""
    rng = np.random.default_rng(1)
    view = torch.tensor(rng.normal(size=(8, 4)))
    aligned = float(info_nce(view, view.clone(), 0.2))
    shuffled = float(info_nce(view, view[torch.randperm(8)], 0.2))
    assert aligned < shuffled


def test_negatives_are_never_the_users_own_items():
    X = _interactions(density=0.5)
    est, module = _fitted_module(XSimGCL(n_factors=4, n_layers=1, max_iter=0), X)
    generator = torch.Generator()
    generator.manual_seed(3)
    rows = est.interactions_
    for users, items, negatives in module.iter_batches(16, generator):
        assert len(negatives) == len(users) == len(items)
        for user, negative in zip(users.tolist(), negatives.tolist(), strict=True):
            owned = rows.indices[rows.indptr[user] : rows.indptr[user + 1]]
            assert negative not in owned


def test_every_interaction_is_visited_once_per_epoch():
    X = _interactions()
    est, module = _fitted_module(XSimGCL(n_factors=4, n_layers=1, max_iter=0), X)
    generator = torch.Generator()
    generator.manual_seed(5)
    seen = [
        (user, item)
        for users, items, _ in module.iter_batches(16, generator)
        for user, item in zip(users.tolist(), items.tolist(), strict=True)
    ]
    assert len(seen) == est.interactions_.nnz
    assert len(set(seen)) == est.interactions_.nnz


def test_score_is_the_dot_product_of_the_exported_factors():
    X = _interactions()
    est = XSimGCL(n_factors=8, n_layers=2, max_iter=3, random_state=0).fit(X)
    assert est.user_factors_.shape == (est.n_users_, 8)
    assert est.item_factors_.shape == (est.n_items_, 8)
    assert est.user_factors_.dtype == est.item_factors_.dtype == np.float64
    scores = est._score_users(np.arange(est.n_users_), np.arange(est.n_items_))
    np.testing.assert_allclose(scores, est.user_factors_ @ est.item_factors_.T, rtol=1e-12)


def test_export_is_the_noise_free_propagation():
    """Inference must not perturb, or two `recommend` calls would disagree."""
    X = _interactions()
    est, module = _fitted_module(XSimGCL(n_factors=8, n_layers=2, max_iter=2, random_state=0), X)
    est_again = clone(est).fit(X)
    np.testing.assert_array_equal(est.user_factors_, est_again.user_factors_)
    with torch.no_grad():
        ranking, _ = module.propagate(perturbed=False)
    assert np.isfinite(ranking.numpy()).all()


def test_loss_curve_decreases():
    # `early_stopping=False` keeps the curve a fixed 40 epochs long; what stopping early
    # does to it is `test_early_stopping_shortens_the_fit`'s business.
    est = XSimGCL(
        n_factors=16,
        n_layers=2,
        max_iter=40,
        learning_rate=0.05,
        early_stopping=False,
        random_state=0,
    )
    est.fit(_clusters())
    assert est.loss_curve_.shape == (40,)
    assert est.n_iter_ == 40
    assert est.loss_curve_[-1] < est.loss_curve_[0]


def test_recovers_planted_clusters():
    X = _clusters()
    est = XSimGCL(n_factors=16, n_layers=2, max_iter=80, learning_rate=0.05, random_state=0).fit(X)
    queries = np.array([f"u{user}" for user in range(60)], dtype=object)
    items, _ = est.recommend(queries, n_recommendations=4)
    blocks = np.vectorize(lambda item: int(item[1:]) // 10)(items)
    inside = blocks == (np.arange(60) % 2)[:, None]
    assert inside.mean() > 0.85, f"only {inside.mean():.0%} of recommendations stayed in-cluster"


def test_seeded_fit_is_reproducible():
    X = _interactions()
    first = XSimGCL(n_factors=8, n_layers=2, max_iter=5, random_state=7, n_jobs=1).fit(X)
    second = XSimGCL(n_factors=8, n_layers=2, max_iter=5, random_state=7, n_jobs=1).fit(X)
    np.testing.assert_array_equal(first.user_factors_, second.user_factors_)
    np.testing.assert_array_equal(first.loss_curve_, second.loss_curve_)

    other = XSimGCL(n_factors=8, n_layers=2, max_iter=5, random_state=8, n_jobs=1).fit(X)
    assert not np.array_equal(first.item_factors_, other.item_factors_)


def test_fitted_estimator_has_no_torch_state():
    """A fitted model must pickle and score where torch is not installed."""
    X = _interactions()
    est = XSimGCL(n_factors=8, n_layers=2, max_iter=2, random_state=0).fit(X)
    restored = pickle.loads(pickle.dumps(est))
    assert not any(isinstance(value, torch.nn.Module) for value in vars(est).values())
    np.testing.assert_array_equal(restored.item_factors_, est.item_factors_)


def test_weight_decay_is_left_to_the_loss():
    """The reference penalizes the batch's embeddings, not every parameter each step."""
    assert XSimGCL(regularization=0.5)._weight_decay() == 0.0


def test_max_iter_zero_exports_the_initialization():
    est = XSimGCL(n_factors=4, n_layers=2, max_iter=0, random_state=0).fit(_interactions())
    assert est.loss_curve_.shape == (0,)
    assert np.isfinite(est.user_factors_).all()


@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("n_factors", 0),
        ("n_factors", 1.5),
        ("batch_size", 0),
        ("max_iter", -1),
        ("learning_rate", 0.0),
        ("regularization", -1.0),
        ("n_layers", 0),
        ("n_layers", 1.5),
        ("n_layers", None),
        ("contrastive_layer", -1),
        ("contrastive_layer", 99),
        ("contrastive_layer", 1.5),
        ("contrastive_weight", -0.1),
        ("contrastive_weight", "strong"),
        ("temperature", 0.0),
        ("temperature", -1.0),
        ("temperature", None),
        ("eps", -0.1),
        ("eps", "noisy"),
        ("device", "not-a-device"),
        ("n_jobs", 0),
    ],
)
def test_invalid_param_raises(param, value):
    est = XSimGCL(n_factors=4, n_layers=2, max_iter=1).set_params(**{param: value})
    with pytest.raises(ValueError, match=param):
        est.fit(_interactions())
