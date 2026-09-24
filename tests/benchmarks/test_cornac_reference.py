"""Head-to-head benchmark of BayesianPersonalizedRanking against Cornac's ``BPR``.

Cornac pins an older numpy and pulls a deep-learning stack, so it runs in a throwaway
environment: ``uv run --isolated --with cornac``. The benchmark is skipped when ``uv`` is
missing or the environment cannot be built, for example offline.

Both sides take one gradient step per sampled triplet, but they sample from different
random streams and Cornac computes in float32, so the two fits are two draws from the
same training procedure rather than the same arithmetic. What has to agree is the model
they arrive at: ranking quality, how much of each user's top-K they share, and how well
both separate observed items from unobserved ones.
"""

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

from skrecsys.metrics import make_recommender_scorer, ndcg_at_k, precision_at_k
from skrecsys.recommendation import BayesianPersonalizedRanking

pytestmark = pytest.mark.benchmark

K = 10
SEED = 0
# Cornac drops to a single thread as soon as it is seeded, so ours does too: this
# compares the two kernels, not the two threading policies.
ESTIMATOR = BayesianPersonalizedRanking(
    n_factors=64, learning_rate=0.05, regularization=0.01, max_iter=100, random_state=SEED
)
SCRIPT = Path(__file__).parent / "_cornac_bpr.py"
#: Absolute tolerance on NDCG@K and P@K between two runs of the same procedure.
METRIC_TOLERANCE = 0.02
#: Share of each user's top-K list that both models must agree on, averaged over users.
SHARED_TOP_K = 0.45
#: Share of (observed, unobserved) pairs each model must rank the right way round.
TRAIN_AUC = 0.9


@pytest.fixture(scope="module")
def uv_bin():
    path = shutil.which("uv")
    if path is None:
        pytest.skip("uv not found; it runs Cornac in an isolated environment.")
    return path


def _run_cornac(uv_bin, interactions, estimator, tmp_path):
    """Fit Cornac's BPR in a throwaway environment; return its parameters and fit time."""
    np.savez(
        tmp_path / "input.npz",
        indptr=interactions.indptr,
        indices=interactions.indices,
        data=interactions.data,
        shape=np.asarray(interactions.shape),
    )
    command = [
        uv_bin, "run", "--isolated", "--quiet",
        "--with", "cornac", "--with", "numpy", "--with", "scipy",
        "python", str(SCRIPT), str(tmp_path / "input.npz"), str(tmp_path / "output.npz"),
        str(estimator.n_factors), str(estimator.learning_rate), str(estimator.regularization),
        str(estimator.max_iter), str(SEED),
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.skip(f"Could not run Cornac in an isolated environment:\n{result.stderr[-2000:]}")
    return np.load(tmp_path / "output.npz")


def _top_k(estimator, seen, k):
    """Rank items by descending score, ties by item index, excluding seen items."""
    users, items = np.arange(estimator.n_users_), np.arange(estimator.n_items_)
    scores = np.where(seen, -np.inf, estimator._score_users(users, items))
    return np.argsort(-scores, kind="stable")[:, :k]


def _train_auc(estimator, seen):
    """Share of (observed, unobserved) item pairs the model ranks the right way round."""
    scores = estimator._score_users(np.arange(estimator.n_users_), np.arange(estimator.n_items_))
    right, total = 0, 0
    for row, observed in zip(scores, seen, strict=True):
        positive, negative = row[observed], row[~observed]
        right += int((positive[:, None] > negative[None, :]).sum())
        total += positive.size * negative.size
    return right / total if total else 0.0


def test_bpr_matches_cornac(uv_bin, movielens_100k_ua, tmp_path, request):
    dataset = movielens_100k_ua
    train, test = dataset.train_indices, dataset.test_indices

    start = time.perf_counter()
    est = clone(ESTIMATOR).fit(dataset.data[train], dataset.target[train])
    ours_seconds = time.perf_counter() - start

    theirs = _run_cornac(uv_bin, est.interactions_, ESTIMATOR, tmp_path)
    theirs_est = clone(ESTIMATOR).fit(dataset.data[train], dataset.target[train])
    theirs_est.user_factors_ = theirs["user_factors"].astype(np.float64)
    theirs_est.item_factors_ = theirs["item_factors"].astype(np.float64)
    theirs_est.item_bias_ = theirs["item_bias"].astype(np.float64)

    seen = est.interactions_.toarray() > 0
    start = time.perf_counter()
    ours_top = _top_k(est, seen, K)
    ours_rank_seconds = time.perf_counter() - start
    start = time.perf_counter()
    theirs_top = _top_k(theirs_est, seen, K)
    theirs_rank_seconds = time.perf_counter() - start
    shared = float(
        np.mean([len(set(a) & set(b)) / K for a, b in zip(ours_top, theirs_top, strict=True)])
    )
    # Both learn the same popularity signal into the biases, up to their own scale.
    bias_correlation = float(np.corrcoef(est.item_bias_, theirs_est.item_bias_)[0, 1])

    scorers = {"ndcg": make_recommender_scorer(ndcg_at_k, k=K),
               "precision": make_recommender_scorer(precision_at_k, k=K)}  # fmt: skip
    ours_scores = {name: scorer(est, dataset.data[test]) for name, scorer in scorers.items()}
    theirs_scores = {
        name: scorer(theirs_est, dataset.data[test]) for name, scorer in scorers.items()
    }
    ours_auc, theirs_auc = _train_auc(est, seen), _train_auc(theirs_est, seen)

    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            f"\nBayesianPersonalizedRanking vs Cornac on ML-100k ua ({est.n_items_} items, "
            f"n_factors={ESTIMATOR.n_factors}, learning_rate={ESTIMATOR.learning_rate}, "
            f"regularization={ESTIMATOR.regularization}, max_iter={ESTIMATOR.max_iter}, "
            "one thread each)\n"
            f"  {'':10}{'NDCG@10':>10}{'P@10':>10}{'train AUC':>11}{'fit s':>10}{'rank s':>10}\n"
            f"  {'skrecsys':10}{ours_scores['ndcg']:>10.4f}{ours_scores['precision']:>10.4f}"
            f"{ours_auc:>11.4f}{ours_seconds:>10.3f}{ours_rank_seconds:>10.3f}\n"
            f"  {'cornac':10}{theirs_scores['ndcg']:>10.4f}{theirs_scores['precision']:>10.4f}"
            f"{theirs_auc:>11.4f}{float(theirs['seconds']):>10.3f}{theirs_rank_seconds:>10.3f}\n"
            f"  {shared:.1%} of each top-{K} list is shared, item biases correlate "
            f"{bias_correlation:.4f}"
        )

    for metric, ours in ours_scores.items():
        assert abs(ours - theirs_scores[metric]) <= METRIC_TOLERANCE, (metric, ours)
    assert shared >= SHARED_TOP_K, shared
    assert bias_correlation >= 0.9, bias_correlation
    assert min(ours_auc, theirs_auc) >= TRAIN_AUC, (ours_auc, theirs_auc)
    assert abs(ours_auc - theirs_auc) <= 0.02, (ours_auc, theirs_auc)
