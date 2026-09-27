import pickle

import numpy as np
import pytest
from sklearn.base import clone

from skrecsys.metrics import (
    MAP,
    MRR,
    NDCG,
    HitRate,
    Precision,
    Recall,
    average_precision_at_k,
    evaluate_recommender,
    get_scorer,
    hit_rate_at_k,
    make_recommender_scorer,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from skrecsys.recommendation import ItemKNNRecommender, MostPopularRecommender

TRAIN = np.array(
    [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "c"], ["u3", "b"], ["u3", "c"], ["u4", "a"]],
    dtype=object,
)
TEST = np.array([["u1", "c"], ["u2", "b"], ["u4", "c"], ["u4", "b"]], dtype=object)

METRICS = [
    (NDCG, ndcg_at_k),
    (Recall, recall_at_k),
    (Precision, precision_at_k),
    (MAP, average_precision_at_k),
    (MRR, reciprocal_rank_at_k),
    (HitRate, hit_rate_at_k),
]


@pytest.mark.parametrize(("cls", "metric"), METRICS, ids=lambda m: getattr(m, "__name__", m))
def test_a_metric_scores_as_its_function_does(cls, metric):
    # Each user has one unseen item left, so one is all there is to rank.
    rec = ItemKNNRecommender(n_neighbors=2).fit(TRAIN)
    expected = make_recommender_scorer(metric, k=1)(rec, TEST)
    assert cls(1)(rec, TEST) == expected


@pytest.mark.parametrize(("cls", "metric"), METRICS, ids=lambda m: getattr(m, "__name__", m))
def test_str_is_the_evaluate_recommender_key(cls, metric):
    rec = MostPopularRecommender().fit(TRAIN)
    scores = evaluate_recommender(rec, TEST, metrics=[metric], k=1)
    assert list(scores) == [str(cls(1))]


def test_repr_and_equality():
    assert repr(NDCG(10)) == "NDCG(k=10)"
    assert NDCG() == NDCG(10) != Recall(10)
    assert NDCG(5) != NDCG(10)


def test_a_metric_clones_and_pickles():
    # As scikit-learn clones a non-estimator parameter, such as AutoTune's scoring.
    assert clone(Recall(5), safe=False) == Recall(5)
    assert pickle.loads(pickle.dumps(MAP(3))) == MAP(3)


@pytest.mark.parametrize("k", [0, -1, 2.5, True, "10", None])
def test_k_must_be_a_positive_integer(k):
    with pytest.raises(ValueError, match="k must be an integer"):
        NDCG(k)


@pytest.mark.parametrize(
    ("scoring", "expected"),
    [
        (None, NDCG(10)),
        ("ndcg@10", NDCG(10)),
        ("ndcg", NDCG(10)),
        ("NDCG@5", NDCG(5)),
        (" recall@20 ", Recall(20)),
        ("precision@3", Precision(3)),
        ("average_precision@7", MAP(7)),
        ("map@5", MAP(5)),
        ("reciprocal_rank", MRR(10)),
        ("mrr@1", MRR(1)),
        ("hit_rate@4", HitRate(4)),
        ("hr", HitRate(10)),
    ],
)
def test_get_scorer_resolves_names(scoring, expected):
    assert get_scorer(scoring) == expected


def test_get_scorer_returns_callables_as_they_are():
    scorer = make_recommender_scorer(recall_at_k, k=5)
    assert get_scorer(scorer) is scorer
    metric = Recall(5)
    assert get_scorer(metric) is metric


@pytest.mark.parametrize(
    ("scoring", "match"),
    [
        ("auc", "Unknown metric 'auc'"),
        ("ndcg@", "Unknown metric"),
        ("ndcg@10@2", "k must be an integer"),
        ("ndcg@x", "k must be an integer"),
        ("ndcg@-1", "k must be an integer"),
        ("ndcg@0", "k must be an integer"),
        ("", "Unknown metric"),
    ],
)
def test_get_scorer_rejects_bad_names(scoring, match):
    with pytest.raises(ValueError, match=match):
        get_scorer(scoring)


def test_get_scorer_rejects_other_types():
    with pytest.raises(TypeError, match="scoring must be"):
        get_scorer(10)  # ty: ignore[invalid-argument-type]
