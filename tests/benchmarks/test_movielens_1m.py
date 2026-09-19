"""Quality benchmarks on the sequential split: data => leave-one-out => model => metrics.

This is the protocol of the sequential-recommendation literature, and the one a model in
:mod:`skrecsys.nn` that reads order rather than a bag of interactions belongs in. Each
user's last interaction is held out and ranked against the whole catalog, with the items
they already saw removed.

Thresholds sit about 10% below the scores measured when they were set, wider than the
ml-100k benchmarks because a neural fit moves more between releases. Raise them when a
model improves.
"""

import numpy as np
import pytest
from sklearn.base import clone

from skrecsys.datasets import fetch_movielens_1m
from skrecsys.metrics import hit_rate_at_k, ndcg_at_k
from skrecsys.recommendation import (
    BM25Recommender,
    ItemKNNRecommender,
    MostPopularRecommender,
)

try:
    from skrecsys import nn
except ImportError:  # the `nn` extra is optional; see src/skrecsys/nn/__init__.py
    nn = None

pytestmark = pytest.mark.benchmark

K = 10
#: History the model may read, the ``l200`` of ``ml-1m-l200``.
WINDOW = 200


def _sequential():
    """The neural sequential models at their defaults, or nothing without the extra.

    Both are fitted with ``device="auto"`` rather than the estimator default of
    ``"cpu"``. The default is the one every wheel and every CI runner can rely on, but a
    benchmark is the place to spend whatever the host has: ``"auto"`` takes CUDA, else
    MPS, else the CPU it would have used anyway, so a runner without an accelerator runs
    exactly as it did before. The thresholds are unchanged by the move -- the arithmetic
    is the same and only the last floating point digits differ -- and the floors below
    sit far enough under a measured score to absorb that.

    Mamba4Rec's floor is not the usual 10% under a measured score: its scan is an
    unfused PyTorch recurrence, so one fit at the default hundred epochs runs for hours
    and no run of it has been scored here yet. Until one has, the floor is what any
    working sequential model must clear -- well past `MostPopularRecommender` and into
    the range the classical baselines occupy. Raise it once a full fit is measured.
    """
    if nn is None:
        return {}
    return {
        "HSTU": (
            nn.HSTU(max_sequence_length=WINDOW, random_state=0, device="auto"),
            {"hit_rate": 0.258, "ndcg": 0.149},
        ),
        "Mamba4Rec": (
            nn.Mamba4Rec(max_sequence_length=WINDOW, random_state=0, device="auto"),
            {"hit_rate": 0.060, "ndcg": 0.030},
        ),
    }


#: Minimum scores at K on the leave-one-out split of ml-1m.
SEQUENTIAL_BENCHMARKS = {
    "MostPopularRecommender": (
        MostPopularRecommender(),
        {"hit_rate": 0.028, "ndcg": 0.013},
    ),
    "ItemKNNRecommender": (
        ItemKNNRecommender(),
        {"hit_rate": 0.064, "ndcg": 0.033},
    ),
    "BM25Recommender": (
        BM25Recommender(),
        {"hit_rate": 0.057, "ndcg": 0.029},
    ),
} | _sequential()


@pytest.fixture(scope="module")
def movielens_1m_sequential():
    """MovieLens 1M, last interaction held out, 200 of history before it."""
    return fetch_movielens_1m(subset="leave-one-out", max_sequence_length=WINDOW)


def _scores(estimator, dataset):
    train, test = dataset.train_indices, dataset.test_indices
    fitted = clone(estimator).fit(dataset.data[train], dataset.target[train])

    queries = dataset.data[test][:, 0]
    relevant = [{item} for item in dataset.data[test][:, 1].tolist()]
    recommended, _ = fitted.recommend(queries, n_recommendations=K, exclude_seen=True)
    return {
        "hit_rate": float(hit_rate_at_k(relevant, recommended, k=K)),
        "ndcg": float(ndcg_at_k(relevant, recommended, k=K)),
    }


@pytest.mark.parametrize("name", sorted(SEQUENTIAL_BENCHMARKS))
def test_meets_its_threshold(name, movielens_1m_sequential):
    estimator, thresholds = SEQUENTIAL_BENCHMARKS[name]
    scores = _scores(estimator, movielens_1m_sequential)
    for metric, floor in thresholds.items():
        assert scores[metric] >= floor, f"{name} {metric}={scores[metric]:.4f} < {floor}"


def test_the_split_is_one_held_out_interaction_per_user(movielens_1m_sequential):
    dataset = movielens_1m_sequential
    users = dataset.data[:, 0]
    assert len(dataset.test_indices) == len(np.unique(users))
    assert len(dataset.train_indices) + len(dataset.test_indices) == len(dataset.data)
    # every held-out row is its user's latest
    held_out = dataset.data[dataset.test_indices][:, 0]
    assert np.array_equal(np.unique(held_out), np.unique(users))
    for index in dataset.test_indices[:50]:
        same_user = users[index] == users
        assert dataset.timestamps[index] == dataset.timestamps[same_user].max()


def test_sequences_are_capped_at_the_window(movielens_1m_sequential):
    lengths = np.bincount(movielens_1m_sequential.data[:, 0])
    # the window is history; the held-out interaction sits on top of it
    assert lengths[lengths > 0].max() == WINDOW + 1
