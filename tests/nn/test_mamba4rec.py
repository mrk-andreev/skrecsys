from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from torch.nn import functional as F

from skrecsys.nn import Mamba4Rec
from skrecsys.nn._mamba4rec import _DT_MAX, _DT_MIN, _Mamba4RecModule, selective_scan

RING = [f"i{k}" for k in range(6)]


def ring_interactions(n_users=60, length=8, rng=None):
    """Sequences walking a ring of items, so the next item is fully determined."""
    rng = rng or np.random.default_rng(0)
    rows = []
    for user in range(n_users):
        start = int(rng.integers(len(RING)))
        rows += [[f"u{user}", RING[(start + step) % len(RING)]] for step in range(length)]
    return np.array(rows, dtype=object)


def build_module(estimator, sequences, n_items, seed=0):
    """A module wired to fixed sequences, for testing the encoder on its own."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    sequences = torch.as_tensor(sequences, dtype=torch.int32)
    lengths = (sequences != 0).sum(dim=1)
    module = _Mamba4RecModule(estimator, sequences, sequences[:, :-1], lengths, n_items, generator)
    module.eval()
    return module


def test_selective_scan_is_the_recurrence_it_documents():
    """The vectorized scan must equal the state space recurrence written out by hand."""
    torch.manual_seed(0)
    batch, length, channels, states = 2, 7, 3, 4
    x = torch.randn(batch, length, channels)
    delta = F.softplus(torch.randn(batch, length, channels))
    a = -torch.rand(channels, states)
    b = torch.randn(batch, length, states)
    c = torch.randn(batch, length, states)
    d = torch.randn(channels)

    expected = torch.zeros(batch, length, channels)
    for sample in range(batch):
        state = torch.zeros(channels, states)
        for position in range(length):
            step = delta[sample, position].unsqueeze(-1)
            state = torch.exp(step * a) * state + step * b[sample, position] * x[
                sample, position
            ].unsqueeze(-1)
            expected[sample, position] = state @ c[sample, position] + d * x[sample, position]

    torch.testing.assert_close(selective_scan(x, delta, a, b, c, d), expected)


def test_selective_scan_state_decays():
    """A step of zero keeps the state; a large one forgets it. That is the selectivity."""
    x = torch.zeros(1, 3, 1)
    x[0, 0, 0] = 1.0
    a, b, c, d = -torch.ones(1, 1), torch.ones(1, 3, 1), torch.ones(1, 3, 1), torch.zeros(1)

    kept = selective_scan(x, torch.full((1, 3, 1), 1e-4), a, b, c, d)
    forgotten = selective_scan(x, torch.full((1, 3, 1), 20.0), a, b, c, d)
    # the impulse is still readable two positions later ...
    assert kept[0, 2, 0] == pytest.approx(kept[0, 0, 0], rel=1e-3)
    # ... unless the steps in between are large, which decays it away
    assert forgotten[0, 2, 0] == pytest.approx(0.0, abs=1e-6)


def test_step_sizes_start_inside_their_range():
    """``dt`` is initialized in log space, so softplus must recover that range."""
    module = build_module(
        Mamba4Rec(n_factors=16, max_sequence_length=4), np.zeros((1, 5), dtype=np.int32), n_items=9
    )
    bias = module.layers[0].mamba.dt_proj.bias
    steps = F.softplus(bias.detach())
    assert float(steps.min()) >= _DT_MIN * 0.99
    assert float(steps.max()) <= _DT_MAX * 1.01
    # and they really span the range rather than collapsing onto one timescale
    assert float(steps.max()) / float(steps.min()) > 10


def test_encoding_is_causal():
    """The state at a position must not move when a later position changes.

    This is the property the whole training scheme rests on: every position is trained to
    predict the next item, which is only a prediction if it cannot see it.
    """
    estimator = Mamba4Rec(n_factors=8, n_blocks=2, max_sequence_length=6)
    module = build_module(estimator, np.zeros((1, 7), dtype=np.int32), n_items=9)
    prefix = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = torch.tensor([[1, 2, 3, 9, 8, 7]])
    with torch.no_grad():
        first, second = module.encode(prefix), module.encode(changed)
    torch.testing.assert_close(first[:, :3], second[:, :3])
    assert not torch.allclose(first[:, 3:], second[:, 3:])


def test_padding_does_not_reach_the_state():
    """A short sequence must encode as though the padding columns were not there."""
    estimator = Mamba4Rec(n_factors=8, n_blocks=2, max_sequence_length=6)
    module = build_module(estimator, np.zeros((1, 7), dtype=np.int32), n_items=9)
    padded = torch.tensor([[1, 2, 3, 0, 0, 0]])
    with torch.no_grad():
        encoded = module.encode(padded)
        longer = module.encode(torch.tensor([[1, 2, 3, 4, 5, 6]]))
    # the real positions agree with the same prefix inside a full sequence ...
    torch.testing.assert_close(encoded[:, :3], longer[:, :3])
    # ... and the padded ones carry nothing at all
    assert torch.all(encoded[:, 3:] == 0)


def test_learns_a_deterministic_next_item():
    """On a ring, the next item follows from the current one; Mamba4Rec should find it."""
    X = ring_interactions()
    rec = Mamba4Rec(
        n_factors=32,
        max_sequence_length=8,
        max_iter=40,
        learning_rate=5e-3,
        dropout=0.0,
        random_state=0,
    ).fit(X)
    queries = sorted({row[0] for row in X})
    items, _ = rec.recommend(queries, n_recommendations=1, exclude_seen=False)

    last_seen = dict(X.tolist())  # each user's final row, in order
    expected = [RING[(RING.index(last_seen[user]) + 1) % len(RING)] for user in queries]
    assert np.mean([got[0] == want for got, want in zip(items, expected, strict=True)]) > 0.9
    assert rec.loss_curve_[-1] < rec.loss_curve_[0]


def test_row_order_is_the_sequence():
    """Shuffling a user's rows changes what the model is told, and so what it learns."""
    X = ring_interactions(n_users=20, length=8)
    shuffled = X[np.random.default_rng(0).permutation(len(X))]
    kwargs: dict[str, Any] = {
        "n_factors": 8,
        "max_sequence_length": 8,
        "max_iter": 5,
        "dropout": 0.0,
        "random_state": 0,
    }
    ordered = Mamba4Rec(**kwargs).fit(X)
    scrambled = Mamba4Rec(**kwargs).fit(shuffled)
    assert not np.allclose(ordered.user_embeddings_, scrambled.user_embeddings_)
    # the identifiers are unaffected, so only the histories can have differed
    np.testing.assert_array_equal(ordered.user_ids_, scrambled.user_ids_)
    np.testing.assert_array_equal(ordered.item_ids_, scrambled.item_ids_)


def test_only_the_last_window_is_read():
    """An interaction older than ``max_sequence_length`` cannot reach the user's state."""
    rows = [["u0", f"i{k}"] for k in range(10)]
    tail = [["u1", f"i{k}"] for k in range(3, 10)]
    rec = Mamba4Rec(
        n_factors=8, max_sequence_length=4, max_iter=0, dropout=0.0, random_state=0
    ).fit(np.array(rows + tail, dtype=object))
    # u0 and u1 end on the same four items, so the window they are scored from matches
    np.testing.assert_allclose(rec.user_embeddings_[0], rec.user_embeddings_[1], atol=1e-6)


def test_export_is_numpy_and_holds_the_item_embeddings():
    rec = Mamba4Rec(n_factors=8, max_sequence_length=6, max_iter=2, random_state=0).fit(
        ring_interactions(n_users=10)
    )
    for name in ("user_embeddings_", "item_embeddings_"):
        exported = getattr(rec, name)
        assert isinstance(exported, np.ndarray)
        assert exported.dtype == np.float64
    assert rec.user_embeddings_.shape == (rec.n_users_, 8)
    assert rec.item_embeddings_.shape == (rec.n_items_, 8)


def test_scores_are_the_dot_product_of_the_exported_vectors():
    X = ring_interactions(n_users=10)
    rec = Mamba4Rec(n_factors=8, max_sequence_length=6, max_iter=2, random_state=0).fit(X)
    queries = sorted({row[0] for row in X})
    scored = rec.user_embeddings_ @ rec.item_embeddings_.T

    items, scores = rec.recommend(queries, n_recommendations=3, exclude_seen=False)
    np.testing.assert_allclose(np.sort(scores, axis=1)[:, ::-1], scores, rtol=1e-12)
    ranked = np.argsort(-scored, axis=1, kind="stable")[:, :3]
    np.testing.assert_array_equal(items, rec.item_ids_[ranked])
    np.testing.assert_allclose(scores, np.take_along_axis(scored, ranked, 1), rtol=1e-9)

    pairs = np.array([[user, item] for user in queries for item in rec.item_ids_], dtype=object)
    np.testing.assert_allclose(rec.predict(pairs), scored.ravel(), rtol=1e-9)


def test_device_cpu_and_auto_agree():
    """``device="auto"`` is what the benchmarks fit on, and it must not change the fit.

    Every draw comes from a CPU generator, so the batch order and the dropout masks of a
    seeded fit are the same wherever the arithmetic runs; only the last floating point
    digits of the result may differ. On a host without an accelerator ``"auto"`` *is*
    ``"cpu"`` and the two runs are identical.
    """
    X = ring_interactions(n_users=20)
    kwargs: dict[str, Any] = {
        "n_factors": 8,
        "max_sequence_length": 8,
        "max_iter": 3,
        "random_state": 0,
    }
    on_cpu = Mamba4Rec(device="cpu", **kwargs).fit(X)
    on_auto = Mamba4Rec(device="auto", **kwargs).fit(X)

    assert on_auto.user_embeddings_.shape == on_cpu.user_embeddings_.shape
    assert np.isfinite(on_auto.user_embeddings_).all()
    # float32 on another backend reorders the same arithmetic, nothing more.
    np.testing.assert_allclose(on_auto.loss_curve_, on_cpu.loss_curve_, rtol=1e-4)


def test_untrained_fit_still_exports():
    """``max_iter=0`` is a valid request: it exports the initialization."""
    rec = Mamba4Rec(n_factors=4, max_sequence_length=4, max_iter=0, random_state=0).fit(
        ring_interactions(n_users=5)
    )
    assert rec.loss_curve_.shape == (0,)
    assert rec.user_embeddings_.shape == (5, 4)


def test_single_interaction_users_are_harmless():
    """A user with one interaction supervises nothing but must not break the fit."""
    X = np.array([["u0", "a"], ["u1", "b"], ["u2", "c"]], dtype=object)
    rec = Mamba4Rec(
        n_factors=4, max_sequence_length=4, max_iter=2, batch_size=2, random_state=0
    ).fit(X)
    assert np.isfinite(rec.loss_curve_).all()
    assert np.isfinite(rec.user_embeddings_).all()


def test_stacking_adds_the_residual_connection():
    """One layer runs without a skip, several with one; the two are different models."""
    X = ring_interactions(n_users=20)
    kwargs: dict[str, Any] = {
        "n_factors": 8,
        "max_sequence_length": 8,
        "max_iter": 3,
        "dropout": 0.0,
        "random_state": 0,
    }
    single = Mamba4Rec(n_blocks=1, **kwargs).fit(X)
    stacked = Mamba4Rec(n_blocks=2, **kwargs).fit(X)
    assert not np.allclose(single.user_embeddings_, stacked.user_embeddings_)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"n_blocks": 0}, "n_blocks must be an integer >= 1"),
        ({"d_state": 0}, "d_state must be an integer >= 1"),
        ({"d_conv": -1}, "d_conv must be an integer >= 1"),
        ({"expand": 0}, "expand must be an integer >= 1"),
        ({"max_sequence_length": 0}, "max_sequence_length must be an integer >= 1"),
        ({"dt_rank": 0}, "dt_rank must be None or an integer >= 1"),
        ({"dropout": 1.0}, "dropout must be a real number in"),
        ({"n_factors": 0}, "n_factors must be an integer >= 1"),
        ({"max_iter": -1}, "max_iter must be an integer >= 0"),
    ],
)
def test_invalid_parameters(params, match):
    with pytest.raises(ValueError, match=match):
        Mamba4Rec(**params).fit(ring_interactions(n_users=4))
