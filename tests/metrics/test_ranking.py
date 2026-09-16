import math

import numpy as np
import pytest

from skrecsys.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

ALL_METRICS = [
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    average_precision_at_k,
    reciprocal_rank_at_k,
    hit_rate_at_k,
]

# Query 0: hits at ranks 1 and 3 of 4, 3 relevant items.
# Query 1: no hits.
# Query 2: hit at rank 2, 1 relevant item.
Y_TRUE = [{"a", "c", "z"}, {"x"}, {"f"}]
Y_PRED = np.array([["a", "b", "c", "d"], ["a", "b", "c", "d"], ["e", "f", "g", "h"]])


def _log(rank):
    return 1 / math.log2(rank + 1)


@pytest.mark.parametrize(
    ("metric", "k", "expected"),
    [
        (precision_at_k, None, [2 / 4, 0, 1 / 4]),
        (precision_at_k, 2, [1 / 2, 0, 1 / 2]),
        (recall_at_k, None, [2 / 3, 0, 1]),
        (recall_at_k, 1, [1 / 3, 0, 0]),
        (
            ndcg_at_k,
            None,
            [(_log(1) + _log(3)) / (_log(1) + _log(2) + _log(3)), 0, _log(2) / _log(1)],
        ),
        (ndcg_at_k, 2, [_log(1) / (_log(1) + _log(2)), 0, _log(2) / _log(1)]),
        (average_precision_at_k, None, [(1 + 2 / 3) / 3, 0, (1 / 2) / 1]),
        (average_precision_at_k, 2, [1 / 2, 0, 1 / 2]),
        (reciprocal_rank_at_k, None, [1, 0, 1 / 2]),
        (reciprocal_rank_at_k, 1, [1, 0, 0]),
        (hit_rate_at_k, None, [1, 0, 1]),
        (hit_rate_at_k, 1, [1, 0, 0]),
    ],
)
def test_hand_computed(metric, k, expected):
    per_query = metric(Y_TRUE, Y_PRED, k=k, average=None)
    np.testing.assert_allclose(per_query, expected)
    assert metric(Y_TRUE, Y_PRED, k=k) == pytest.approx(np.mean(expected))


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_sample_weight(metric):
    per_query = metric(Y_TRUE, Y_PRED, average=None)
    weights = [1, 0, 3]
    assert metric(Y_TRUE, Y_PRED, sample_weight=weights) == pytest.approx(
        np.average(per_query, weights=weights)
    )


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_perfect_ranking(metric):
    y_true = [{"a", "b"}, {1}]
    y_pred = np.array([["a", "b"], [1, 2]], dtype=object)
    expected = 1 / 2 if metric is precision_at_k else 1.0
    assert metric(y_true, y_pred, average=None)[1] == pytest.approx(expected)
    assert metric(y_true, y_pred, average=None)[0] == pytest.approx(1.0)


@pytest.mark.parametrize("metric", ALL_METRICS)
@pytest.mark.parametrize(
    ("y_true", "y_pred", "kwargs", "match"),
    [
        ([{"a"}], [["a"]], {"k": 0}, "k must be"),
        ([{"a"}], [["a"]], {"k": 2}, "k must be"),
        ([{"a"}], ["a"], {}, "two-dimensional"),
        ([{"a"}, {"b"}], [["a"]], {}, "inconsistent"),
        ([set()], [["a"]], {}, "at least one relevant"),
        ([{"a"}], [["a", "a"]], {}, "duplicate"),
        ([{"a"}], [["a"]], {"average": "micro"}, "average"),
        ([{"a"}], [["a"]], {"sample_weight": [1, 2]}, "sample_weight"),
    ],
)
def test_invalid_inputs(metric, y_true, y_pred, kwargs, match):
    with pytest.raises(ValueError, match=match):
        metric(y_true, y_pred, **kwargs)
