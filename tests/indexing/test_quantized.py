"""The quantized flat index as a recommender uses it: exactness, recall, and pickling."""

import pickle

import numpy as np
import pytest

from skrecsys.indexing import QuantizedFlatIndex
from skrecsys.indexing._quantized import (
    SUPPORTED_BITS,
    decode,
    pack_codes,
    quantize_dense,
    unpack_codes,
)
from skrecsys.recommendation import (
    EASE,
    AlternatingLeastSquares,
    BayesianPersonalizedRanking,
    BM25Recommender,
    ItemKNNRecommender,
    RP3Beta,
)

#: Small enough to fit a test, large enough that the index is used rather than skipped.
SMALL_INDEX = {"min_index_size": 200}

#: Every model the mixin can index, and whether it needs an interaction value.
#:
#: All three kernels are covered here and it is worth saying which is which: the two
#: latent-factor models are dense vectors with dense queries, the three neighbourhood
#: models are sparse both sides, and EASE is the odd one -- catalog-wide *dense* item
#: rows scored against a query of a handful of nonzeros.
MODELS = [
    (AlternatingLeastSquares(n_factors=16, n_iter=8, random_state=0), True),
    (BayesianPersonalizedRanking(n_factors=16, max_iter=20, random_state=0), False),
    (ItemKNNRecommender(n_neighbors=50), False),
    (BM25Recommender(n_neighbors=20), False),
    (RP3Beta(n_neighbors=50), False),
    (EASE(l2_reg=50.0), False),
]

#: The shortlist each code width is measured with, and the guidance that goes with it.
#:
#: Narrow codes rank the shortlist worse, so they need a wider one to hold the same
#: answers -- which is the trade `oversample` exists to make, and the reason it is a
#: parameter rather than a constant. Four is enough at eight and four bits, where the
#: shortlist essentially always contains the true top ten; below that it is not.
OVERSAMPLE = {8: 4, 4: 4, 2: 8, 1: 32}

#: Recall floors at the shortlist above, per code width, over every model in `MODELS`.
#:
#: Set below what is measured on the fixture, not at it. Eight and four bits are
#: model-independent because they lose essentially nothing; one and two bits are where a
#: model's geometry starts to matter, and the floor there is the weakest of the six. It
#: is `AlternatingLeastSquares` at one bit, measured at 0.36 even with a thirty-two-fold
#: shortlist -- which is the number to look at before reaching for one-bit codes on a
#: latent-factor model, and the reason this table is in the test rather than the docs.
RECALL_FLOOR = {8: 1.0, 4: 0.99, 2: 0.75, 1: 0.30}


def _fit_pair(estimator, X, y, **index_kwargs):
    """The same estimator, exactly and approximately."""
    params = {k: v for k, v in estimator.get_params().items() if k != "index"}
    exact = estimator.__class__(**params).fit(X, y)
    index = QuantizedFlatIndex(**SMALL_INDEX, **index_kwargs)
    approx = estimator.__class__(**params, index=index).fit(X, y)
    return exact, approx


def _recall(approx_items, exact_items):
    k = exact_items.shape[1]
    pairs = zip(approx_items, exact_items, strict=True)
    return np.mean([len(set(a) & set(b)) for a, b in pairs]) / k


def _values(estimator, needs_y, interactions):
    rng = np.random.default_rng(1)
    return rng.integers(1, 6, len(interactions)).astype(float) if needs_y else None


@pytest.mark.parametrize(("estimator", "needs_y"), MODELS, ids=lambda v: repr(v)[:40])
def test_a_full_shortlist_at_eight_bits_is_the_exact_path(estimator, needs_y, interactions):
    """The property the rerank exists for, and the sharpest test of the whole index.

    A shortlist as wide as the catalog cannot drop a true top-k item whatever the codes
    say, and the rerank then scores those items with the original vectors. So the answer
    is not close to the exact path's -- it is the exact path's, items and scores both,
    down to the tie-break. Anything wrong in the affine identity, the packing, the
    candidate-position mapping or the ordering shows up here as an inequality rather
    than as a recall a threshold has to be chosen for.
    """
    y = _values(estimator, needs_y, interactions)
    n_items = len(np.unique(interactions[:, 1]))
    exact, approx = _fit_pair(estimator, interactions, y, bits=8, oversample=n_items)
    users = np.unique(interactions[:, 0])[:150]

    expected = exact.recommend(users, n_recommendations=10)
    got = approx.recommend(users, n_recommendations=10)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_allclose(got[1], expected[1], atol=1e-10)


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
@pytest.mark.parametrize(("estimator", "needs_y"), MODELS, ids=lambda v: repr(v)[:40])
def test_recall_against_the_exact_path(estimator, bits, needs_y, interactions):
    y = _values(estimator, needs_y, interactions)
    exact, approx = _fit_pair(estimator, interactions, y, bits=bits, oversample=OVERSAMPLE[bits])
    users = np.unique(interactions[:, 0])[:150]

    exact_items, _ = exact.recommend(users, n_recommendations=10)
    approx_items, _ = approx.recommend(users, n_recommendations=10)
    recall = _recall(approx_items, exact_items)
    floor = RECALL_FLOOR[bits]
    assert recall >= floor, (
        f"{estimator!r} at {bits} bits found {recall:.4f} of the exact top 10, "
        f"against a floor of {floor} at a shortlist of {OVERSAMPLE[bits]}k."
    )


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
def test_the_scores_are_exact_at_every_width(bits, interactions):
    """Approximate about *which* items, never about what they score.

    This is where the quantized index differs from the graph in kind rather than degree:
    a returned score is never a dequantized number, because the shortlist is rescored
    with the original vectors before anything is returned. One bit per value and the
    scores still agree with `predict` to the last place.
    """
    _, approx = _fit_pair(
        AlternatingLeastSquares(n_factors=16, n_iter=8, random_state=0),
        interactions,
        np.ones(len(interactions)),
        bits=bits,
    )
    users = np.unique(interactions[:, 0])[:50]
    items, scores = approx.recommend(users, n_recommendations=10)
    pairs = np.column_stack([np.repeat(users, 10), items.ravel()])
    np.testing.assert_allclose(scores.ravel(), approx.predict(pairs), atol=1e-10)


def test_a_wider_shortlist_never_lowers_recall(interactions):
    # The one dial that can be turned on a fitted model, and the direction it turns.
    exact = ItemKNNRecommender(n_neighbors=50).fit(interactions)
    approx = ItemKNNRecommender(
        n_neighbors=50, index=QuantizedFlatIndex(**SMALL_INDEX, bits=1)
    ).fit(interactions)
    users = np.unique(interactions[:, 0])[:100]
    exact_items, _ = exact.recommend(users, n_recommendations=10)

    recalls = []
    for oversample in (1, 2, 4, 16, 64):
        approx.index_.set_params(oversample=oversample)  # ty: ignore[unresolved-attribute]
        items, _ = approx.recommend(users, n_recommendations=10)
        recalls.append(_recall(items, exact_items))
    assert recalls == sorted(recalls), f"recall fell as the shortlist widened: {recalls}"


def test_the_index_is_built_and_costs_less_than_the_vectors(interactions):
    exact, approx = _fit_pair(
        AlternatingLeastSquares(n_factors=16, n_iter=8, random_state=0),
        interactions,
        np.ones(len(interactions)),
        bits=4,
    )
    assert exact.index_ is None
    assert approx.index_ is not None
    # Four-bit codes of a float64 space, so the codes are a sixteenth of the vectors
    # plus two small per-dimension arrays. The point of the index in one assertion.
    assert 0 < approx.index_.nbytes < approx.index_.space_.nbytes // 8


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
def test_narrower_codes_cost_proportionally_less(bits, interactions):
    _, approx = _fit_pair(
        AlternatingLeastSquares(n_factors=16, n_iter=8, random_state=0),
        interactions,
        np.ones(len(interactions)),
        bits=bits,
    )
    # Bit packing is what makes `bits` mean something: without it every width would
    # occupy a byte and the parameter would trade accuracy away for nothing.
    space = approx.index_.space_
    expected = space.n_items * -(-space.dim * bits // 8)
    assert approx.index_.codes_.nbytes == expected


def test_results_do_not_change_between_calls(interactions):
    _, approx = _fit_pair(ItemKNNRecommender(n_neighbors=50), interactions, None, bits=2)
    users = np.unique(interactions[:, 0])[:80]
    first = approx.recommend(users, n_recommendations=10)
    second = approx.recommend(users, n_recommendations=10)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


@pytest.mark.parametrize("n_jobs", [None, 1, 2, -1])
def test_the_answer_does_not_depend_on_the_thread_count(n_jobs, interactions):
    # The scan is sequential and the quantization is deterministic, so unlike the graph
    # there is nothing here for `n_jobs` or `random_state` to change. The kernels do not
    # even take a thread count; this is the test that keeps it that way.
    def build(jobs):
        return ItemKNNRecommender(
            n_neighbors=50,
            n_jobs=jobs,
            index=QuantizedFlatIndex(**SMALL_INDEX, bits=2),
        ).fit(interactions)

    users = np.unique(interactions[:, 0])[:40]
    baseline = build(None).recommend(users, n_recommendations=10)
    got = build(n_jobs).recommend(users, n_recommendations=10)
    np.testing.assert_array_equal(got[0], baseline[0])
    np.testing.assert_allclose(got[1], baseline[1])


def test_excluded_items_are_still_excluded(interactions):
    _, approx = _fit_pair(ItemKNNRecommender(n_neighbors=50), interactions, None, bits=2)
    users = np.unique(interactions[:, 0])[:60]
    items, _ = approx.recommend(users, n_recommendations=10, exclude_seen=True)
    for user, row in zip(users, items, strict=True):
        seen = set(interactions[interactions[:, 0] == user, 1])
        assert not (set(row) & seen)
        assert len(set(row)) == 10


def test_too_few_eligible_items_still_raises(interactions):
    _, approx = _fit_pair(ItemKNNRecommender(n_neighbors=50), interactions, None)
    users = np.unique(interactions[:, 0])[:5]
    with pytest.raises(ValueError, match="eligible"):
        approx.recommend(users, n_recommendations=approx.n_items_)


def test_candidates_below_the_floor_take_the_exact_path(interactions):
    exact, approx = _fit_pair(ItemKNNRecommender(n_neighbors=50), interactions, None, bits=1)
    users = np.unique(interactions[:, 0])[:40]
    candidates = np.unique(interactions[:, 1])[:150]
    expected = exact.recommend(users, n_recommendations=10, candidates=candidates)
    got = approx.recommend(users, n_recommendations=10, candidates=candidates)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_allclose(got[1], expected[1])


def test_a_small_catalog_takes_the_exact_path():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.integers(0, 40, 800), rng.integers(0, 60, 800)])
    exact = ItemKNNRecommender().fit(X)
    approx = ItemKNNRecommender(index="quantized-flat").fit(X)
    users = np.unique(X[:, 0])[:20]
    expected = exact.recommend(users, n_recommendations=5)
    got = approx.recommend(users, n_recommendations=5)
    np.testing.assert_array_equal(got[0], expected[0])
    np.testing.assert_allclose(got[1], expected[1])


def test_a_fitted_index_pickles_and_still_searches(interactions):
    _, approx = _fit_pair(
        AlternatingLeastSquares(n_factors=16, n_iter=8, random_state=0),
        interactions,
        np.ones(len(interactions)),
        bits=4,
    )
    users = np.unique(interactions[:, 0])[:40]
    before = approx.recommend(users, n_recommendations=10)
    restored = pickle.loads(pickle.dumps(approx))
    after = restored.recommend(users, n_recommendations=10)
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_allclose(before[1], after[1])


@pytest.mark.parametrize("break_it", ["codes", "scale", "bits"])
def test_a_corrupted_store_raises_rather_than_crashing(break_it, interactions):
    # A fitted index is numpy arrays, and numpy arrays come back from pickles that may
    # not be the ones we wrote. Every way of being wrong has to be a `ValueError`.
    _, approx = _fit_pair(
        AlternatingLeastSquares(n_factors=8, n_iter=5, random_state=0),
        interactions,
        np.ones(len(interactions)),
    )
    users = np.unique(interactions[:, 0])[:10]
    index = approx.index_
    if break_it == "codes":
        index.codes_ = index.codes_[: len(index.codes_) // 2]
    elif break_it == "scale":
        index.scale_ = index.scale_[:-1]
    else:
        index.bits = 3
    with pytest.raises(ValueError, match=r"quantized store|bits must be one of"):
        approx.recommend(users, n_recommendations=10)


def test_clipping_spends_the_code_range_on_the_bulk():
    """What `quantile` is for: one outlier must not cost every other value its levels.

    Without trimming, a dimension running over `[-1, 1]` with a single entry at `1000`
    has its whole eight-bit range stretched across that span, leaving the bulk about two
    levels to share. Trimming a tail puts the outlier outside the range, where it clips
    -- losing its magnitude but not its rank -- and gives the bulk the range back.
    """
    rng = np.random.default_rng(0)
    items = rng.uniform(-1.0, 1.0, size=(500, 4))
    items[0, :] = 1000.0

    errors = {}
    for quantile in (0.0, 0.01):
        codes, scale, offset = quantize_dense(items, 8, quantile)
        recovered = decode(codes, scale, offset)
        # The outlier itself is excluded: clipping is what it is *for*, and measuring
        # the bulk is the whole point of moving it out of the range.
        errors[quantile] = np.abs(recovered[1:] - items[1:]).max()

    assert errors[0.01] < errors[0.0] / 10, errors


@pytest.mark.parametrize("bits", SUPPORTED_BITS)
@pytest.mark.parametrize("dim", [1, 3, 8, 17])
def test_packing_round_trips_at_every_width_and_width_remainder(bits, dim):
    # The row padding and the byte boundary meet at every `dim` modulo `8 // bits`, so
    # the remainder is what this walks rather than one convenient size.
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 1 << bits, size=(7, dim)).astype(np.uint8)
    packed = pack_codes(codes, bits)
    np.testing.assert_array_equal(unpack_codes(packed, bits, codes.shape), codes)


def test_a_degenerate_dimension_codes_exactly():
    # Every item sharing a value leaves nothing to spread over the levels. A zero scale
    # sends them all to code zero, and the offset alone carries the value with no error.
    items = np.column_stack([np.full(50, 2.5), np.linspace(-1.0, 1.0, 50)])
    codes, scale, offset = quantize_dense(items, 8, 0.0)
    assert scale[0] == 0.0
    np.testing.assert_array_equal(codes[:, 0], 0)
    np.testing.assert_allclose(decode(codes, scale, offset)[:, 0], 2.5)


def test_exclude_interactions_is_honoured_by_the_index(interactions):
    _, approx = _fit_pair(
        AlternatingLeastSquares(n_factors=16, n_iter=8, random_state=0),
        interactions,
        _values(None, needs_y=True, interactions=interactions),
    )
    users = np.unique(interactions[:, 0])[:60]
    before, _ = approx.recommend(users, n_recommendations=10)
    # Each user's three best items, as if they had just been shown them.
    pairs = np.column_stack([np.repeat(users, 3), before[:, :3].ravel()])
    items, _ = approx.recommend(users, n_recommendations=10, exclude_interactions=pairs)
    for user, row, shown in zip(users, items, before[:, :3], strict=True):
        seen = set(interactions[interactions[:, 0] == user, 1])
        assert not (set(row) & (seen | set(shown)))
        assert len(set(row)) == 10
