from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from skrecsys.nn import HSTU
from skrecsys.nn._hstu import _HSTUModule, relative_position_bias, seeded_dropout

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
    module = _HSTUModule(estimator, sequences, sequences[:, :-1], lengths, n_items, generator)
    module.eval()
    return module


def test_relative_position_bias_depends_only_on_the_offset():
    weights = torch.arange(2 * 6 - 1, dtype=torch.float32)
    bias = relative_position_bias(weights, 6)
    assert bias.shape == (1, 6, 6)
    by_offset = {i - j: bias[0, i, j].item() for i in range(6) for j in range(6)}
    for i in range(6):
        for j in range(6):
            assert bias[0, i, j].item() == pytest.approx(by_offset[i - j])
    # distinct offsets get distinct weights, so the bias really carries the distance
    assert len(set(by_offset.values())) == len(by_offset)


def test_attention_is_causal():
    """The state at a position must not move when a later position changes.

    This is the property the whole training scheme rests on: every position is trained to
    predict the next item, which is only a prediction if it cannot see it.
    """
    estimator = HSTU(n_factors=8, n_blocks=2, n_heads=2, head_dim=4, max_sequence_length=6)
    module = build_module(estimator, np.zeros((1, 7), dtype=np.int32), n_items=9)
    prefix = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = torch.tensor([[1, 2, 3, 9, 8, 7]])
    with torch.no_grad():
        first, second = module.encode(prefix), module.encode(changed)
    torch.testing.assert_close(first[:, :3], second[:, :3])
    assert not torch.allclose(first[:, 3:], second[:, 3:])


def test_padding_does_not_reach_the_state():
    """A short sequence must encode as though the padding columns were not there."""
    estimator = HSTU(n_factors=8, n_blocks=2, max_sequence_length=6)
    module = build_module(estimator, np.zeros((1, 7), dtype=np.int32), n_items=9)
    padded = torch.tensor([[1, 2, 3, 0, 0, 0]])
    with torch.no_grad():
        encoded = module.encode(padded)
        longer = module.encode(torch.tensor([[1, 2, 3, 4, 5, 6]]))
    # the real positions agree with the same prefix inside a full sequence ...
    torch.testing.assert_close(encoded[:, :3], longer[:, :3])
    # ... and the padded ones carry nothing but the normalization floor
    assert torch.all(encoded[:, 3:].abs().sum(dim=-1) == 0)


def test_seeded_dropout_scales_and_is_reproducible():
    x = torch.ones(200, 8)
    generator = torch.Generator()
    generator.manual_seed(0)
    dropped = seeded_dropout(x, 0.5, generator, training=True)
    assert set(dropped.flatten().tolist()) == {0.0, 2.0}, "inverted dropout rescales what it keeps"
    assert 0.3 < (dropped == 0).float().mean() < 0.7
    generator.manual_seed(0)
    torch.testing.assert_close(seeded_dropout(x, 0.5, generator, training=True), dropped)
    # evaluation and p=0 are both the identity
    torch.testing.assert_close(seeded_dropout(x, 0.5, generator, training=False), x)
    torch.testing.assert_close(seeded_dropout(x, 0.0, generator, training=True), x)


def test_learns_a_deterministic_next_item():
    """On a ring, the next item follows from the current one; HSTU should find it."""
    X = ring_interactions()
    rec = HSTU(
        n_factors=32,
        max_sequence_length=8,
        max_iter=60,
        n_negatives=4,
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
        "n_negatives": 2,
        "dropout": 0.0,
        "random_state": 0,
    }
    ordered = HSTU(**kwargs).fit(X)
    scrambled = HSTU(**kwargs).fit(shuffled)
    assert not np.allclose(ordered.user_embeddings_, scrambled.user_embeddings_)
    # the identifiers are unaffected, so only the histories can have differed
    np.testing.assert_array_equal(ordered.user_ids_, scrambled.user_ids_)
    np.testing.assert_array_equal(ordered.item_ids_, scrambled.item_ids_)


def test_only_the_last_window_is_read():
    """An interaction older than ``max_sequence_length`` cannot reach the user's state."""
    rows = [["u0", f"i{k}"] for k in range(10)]
    tail = [["u1", f"i{k}"] for k in range(3, 10)]
    kwargs: dict[str, Any] = {
        "n_factors": 8,
        "max_sequence_length": 4,
        "max_iter": 0,
        "dropout": 0.0,
        "random_state": 0,
    }
    rec = HSTU(**kwargs).fit(np.array(rows + tail, dtype=object))
    # u0 and u1 end on the same four items, so the window they are scored from matches
    np.testing.assert_allclose(rec.user_embeddings_[0], rec.user_embeddings_[1], atol=1e-6)


def test_export_is_numpy_and_normalized():
    rec = HSTU(n_factors=8, max_sequence_length=6, max_iter=2, random_state=0).fit(
        ring_interactions(n_users=10)
    )
    for name in ("user_embeddings_", "item_embeddings_"):
        exported = getattr(rec, name)
        assert isinstance(exported, np.ndarray)
        assert exported.dtype == np.float64
        np.testing.assert_allclose(np.linalg.norm(exported, axis=1), 1.0, rtol=1e-6)
    assert rec.user_embeddings_.shape == (rec.n_users_, 8)
    assert rec.item_embeddings_.shape == (rec.n_items_, 8)


def test_scores_are_the_similarity_over_the_temperature():
    """Scoring is the cosine of the exported vectors, scaled; the scale cannot reorder it.

    Temperature also sharpens the training loss, so two fits that differ in it are two
    different models -- what is a pure rescaling is the scoring of one fitted model.
    """
    X = ring_interactions(n_users=10)
    rec = HSTU(n_factors=8, max_sequence_length=6, max_iter=2, temperature=0.05, random_state=0)
    rec = rec.fit(X)
    queries = sorted({row[0] for row in X})
    similarity = rec.user_embeddings_ @ rec.item_embeddings_.T

    items, scores = rec.recommend(queries, n_recommendations=3, exclude_seen=False)
    np.testing.assert_allclose(np.sort(scores, axis=1)[:, ::-1], scores, rtol=1e-12)
    ranked = np.argsort(-similarity, axis=1, kind="stable")[:, :3]
    np.testing.assert_array_equal(items, rec.item_ids_[ranked])
    np.testing.assert_allclose(scores, np.take_along_axis(similarity, ranked, 1) / 0.05, rtol=1e-9)

    pairs = np.array([[user, item] for user in queries for item in rec.item_ids_], dtype=object)
    np.testing.assert_allclose(rec.predict(pairs), (similarity / 0.05).ravel(), rtol=1e-9)


def test_untrained_fit_still_exports():
    """``max_iter=0`` is a valid request: it exports the initialization."""
    rec = HSTU(n_factors=4, max_sequence_length=4, max_iter=0, random_state=0).fit(
        ring_interactions(n_users=5)
    )
    assert rec.loss_curve_.shape == (0,)
    assert rec.user_embeddings_.shape == (5, 4)


def test_single_interaction_users_are_harmless():
    """A user with one interaction supervises nothing but must not break the fit."""
    X = np.array([["u0", "a"], ["u1", "b"], ["u2", "c"]], dtype=object)
    rec = HSTU(n_factors=4, max_sequence_length=4, max_iter=2, batch_size=2, random_state=0).fit(X)
    assert np.isfinite(rec.loss_curve_).all()
    assert np.isfinite(rec.user_embeddings_).all()


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"n_blocks": 0}, "n_blocks must be an integer >= 1"),
        ({"n_heads": -1}, "n_heads must be an integer >= 1"),
        ({"max_sequence_length": 0}, "max_sequence_length must be an integer >= 1"),
        ({"n_negatives": 0}, "n_negatives must be an integer >= 1"),
        ({"head_dim": 0}, "head_dim must be None or an integer >= 1"),
        ({"temperature": 0.0}, "temperature must be a real number > 0"),
        ({"dropout": 1.0}, "dropout must be a real number in"),
        ({"n_factors": 0}, "n_factors must be an integer >= 1"),
        ({"max_iter": -1}, "max_iter must be an integer >= 0"),
    ],
)
def test_invalid_parameters(params, match):
    with pytest.raises(ValueError, match=match):
        HSTU(**params).fit(ring_interactions(n_users=4))


def test_heads_split_the_projection():
    """Several heads must attend independently rather than share one attention map."""
    X = ring_interactions(n_users=20)
    kwargs: dict[str, Any] = {
        "n_factors": 8,
        "head_dim": 4,
        "max_sequence_length": 8,
        "max_iter": 3,
        "dropout": 0.0,
        "random_state": 0,
    }
    single = HSTU(n_heads=1, **kwargs).fit(X)
    multi = HSTU(n_heads=2, **kwargs).fit(X)
    assert not np.allclose(single.user_embeddings_, multi.user_embeddings_)
