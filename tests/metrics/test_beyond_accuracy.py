import math

import numpy as np
import pytest

from skrecsys.metrics import (
    catalog_coverage_at_k,
    item_popularity,
    mean_popularity_at_k,
    novelty_at_k,
    user_coverage_at_k,
)

# Two queries over a 4-item catalog; three distinct items recommended.
Y_PRED = np.array([["a", "b"], ["a", "c"]])
TRAIN = [["u1", "a"], ["u2", "a"], ["u2", "b"], ["u3", "c"]]
POPULARITY = {"a": 2.0, "b": 1.0, "c": 1.0, "d": 0.0}


def test_item_popularity_counts_and_sums():
    assert item_popularity(TRAIN) == {"a": 2.0, "b": 1.0, "c": 1.0}
    y = [1.0, 3.0, 2.0, 5.0]
    assert item_popularity(TRAIN, y, weighting="sum") == {"a": 4.0, "b": 2.0, "c": 5.0}
    assert item_popularity(TRAIN, normalize=True) == {"a": 0.5, "b": 0.25, "c": 0.25}


def test_item_popularity_rejects_unknown_weighting():
    with pytest.raises(ValueError, match="weighting"):
        item_popularity(TRAIN, weighting="mean")


@pytest.mark.parametrize(
    ("k", "expected"),
    [(None, 3 / 4), (2, 3 / 4), (1, 1 / 4)],
)
def test_catalog_coverage(k, expected):
    assert catalog_coverage_at_k(None, Y_PRED, k=k, catalog=["a", "b", "c", "d"]) == expected
    assert catalog_coverage_at_k(None, Y_PRED, k=k, n_catalog_items=4) == expected


def test_catalog_coverage_ignores_padding():
    padded = [["a", "b"], ["a", None]]
    assert catalog_coverage_at_k(None, padded, k=2, n_catalog_items=4) == 2 / 4


def test_catalog_coverage_needs_exactly_one_catalog_argument():
    with pytest.raises(ValueError, match="exactly one"):
        catalog_coverage_at_k(None, Y_PRED, k=2)
    with pytest.raises(ValueError, match="exactly one"):
        catalog_coverage_at_k(None, Y_PRED, k=2, catalog=["a"], n_catalog_items=1)


def test_catalog_coverage_rejects_items_outside_the_catalog():
    with pytest.raises(ValueError, match="outside the catalog"):
        catalog_coverage_at_k(None, Y_PRED, k=2, catalog=["a", "b"])
    with pytest.raises(ValueError, match="more than the catalog size"):
        catalog_coverage_at_k(None, Y_PRED, k=2, n_catalog_items=2)


@pytest.mark.parametrize(
    ("y_pred", "k", "expected"),
    [
        (Y_PRED, 2, [1.0, 1.0]),
        ([["a", "b"], ["a", None]], 2, [1.0, 0.0]),
        ([["a", "b"], ["a", np.nan]], 2, [1.0, 0.0]),
        ([["a", "b"], ["a"]], 2, [1.0, 0.0]),
        ([["a", "b"], ["a", "PAD"]], 2, [1.0, 1.0]),
        ([["a", "b"], ["a", None]], 1, [1.0, 1.0]),
    ],
)
def test_user_coverage(y_pred, k, expected):
    np.testing.assert_allclose(user_coverage_at_k(None, y_pred, k=k, average=None), expected)
    assert user_coverage_at_k(None, y_pred, k=k) == pytest.approx(np.mean(expected))


def test_user_coverage_honours_fill_value():
    y_pred = [["a", "b"], ["a", "PAD"]]
    scores = user_coverage_at_k(None, y_pred, k=2, average=None, fill_value="PAD")
    np.testing.assert_allclose(scores, [1.0, 0.0])


def test_mean_popularity():
    per_query = mean_popularity_at_k(None, Y_PRED, k=2, item_popularity=POPULARITY, average=None)
    np.testing.assert_allclose(per_query, [1.5, 1.5])
    top1 = mean_popularity_at_k(None, Y_PRED, k=1, item_popularity=POPULARITY, average=None)
    np.testing.assert_allclose(top1, [2.0, 2.0])


def test_mean_popularity_uses_default_for_unknown_items():
    scores = mean_popularity_at_k(
        None, [["a", "zzz"]], k=2, item_popularity=POPULARITY, average=None, default=7.0
    )
    np.testing.assert_allclose(scores, [4.5])


def test_mean_popularity_of_an_empty_row_is_nan():
    scores = np.asarray(
        mean_popularity_at_k(None, [["a"], [None]], k=1, item_popularity=POPULARITY, average=None)
    )
    assert scores[0] == 2.0
    assert math.isnan(scores[1])


def test_novelty_matches_smoothed_self_information():
    total = 4.0 + 1.0 * len(POPULARITY)
    expected_a = -math.log2(3.0 / total)
    expected_b = -math.log2(2.0 / total)
    per_query = novelty_at_k(None, Y_PRED, k=2, item_popularity=POPULARITY, average=None)
    np.testing.assert_allclose(per_query, [(expected_a + expected_b) / 2] * 2)


def test_novelty_without_smoothing_is_textbook_self_information():
    counts = {"a": 2.0, "b": 1.0, "c": 1.0}
    per_query = novelty_at_k(None, Y_PRED, k=2, item_popularity=counts, smoothing=0.0, average=None)
    np.testing.assert_allclose(per_query, [(1.0 + 2.0) / 2] * 2)


def test_novelty_ranks_a_rare_item_above_a_popular_one():
    popular = novelty_at_k(None, [["a"]], k=1, item_popularity=POPULARITY)
    rare = novelty_at_k(None, [["b"]], k=1, item_popularity=POPULARITY)
    assert rare > popular


def test_novelty_rejects_a_zero_probability_item_without_smoothing():
    with pytest.raises(ValueError, match="zero popularity"):
        novelty_at_k(None, [["d"]], k=1, item_popularity=POPULARITY, smoothing=0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"smoothing": -1.0}, "smoothing"),
        ({"item_popularity": {}}, "must not be empty"),
        ({"item_popularity": {"a": -1.0}}, "negative"),
        ({"item_popularity": {"a": 0.0}, "smoothing": 0.0}, "all zeros"),
    ],
)
def test_novelty_validates_its_arguments(kwargs, match):
    call = {"item_popularity": POPULARITY} | kwargs
    with pytest.raises(ValueError, match=match):
        novelty_at_k(None, [["a"]], k=1, **call)


@pytest.mark.parametrize(
    ("metric", "extra"),
    [
        (user_coverage_at_k, {}),
        (mean_popularity_at_k, {"item_popularity": POPULARITY}),
        (novelty_at_k, {"item_popularity": POPULARITY}),
        (catalog_coverage_at_k, {"n_catalog_items": 4}),
    ],
)
@pytest.mark.parametrize("bad_k", [0, 3, 1.5, True])
def test_invalid_k(metric, extra, bad_k):
    with pytest.raises(ValueError, match="k must be"):
        metric(None, Y_PRED, k=bad_k, **extra)


def test_rejects_three_dimensional_predictions():
    with pytest.raises(ValueError, match="two-dimensional"):
        user_coverage_at_k(None, np.zeros((2, 2, 2)), k=2)


def test_rejects_duplicate_recommendations():
    with pytest.raises(ValueError, match="duplicate items for query 1"):
        user_coverage_at_k(None, [["a", "b"], ["c", "c"]], k=2)


def test_sample_weight_reweights_queries():
    y_pred: list[list[str | None]] = [["a", "b"], ["a", None]]
    weighted = user_coverage_at_k(None, y_pred, k=2, sample_weight=[3.0, 1.0])
    assert weighted == pytest.approx(0.75)
