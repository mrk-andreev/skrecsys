"""Head-to-head benchmark of EASE against the RecTools reference implementation.

RecTools pulls ``pm-implicit`` and an older pandas, which conflict with our ``reference``
group, so it runs in a throwaway environment: ``uv run --isolated --with rectools``. The
benchmark is skipped when ``uv`` is missing or the environment cannot be built.

RecTools inverts the regularized Gram matrix with a general-purpose routine and stores
the result as float32; we factor it by Cholesky and keep float64. The weights are
therefore compared at float32 precision, and the resulting rankings must agree.
"""

import copy
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

from skrecsys.metrics import make_recommender_scorer, ndcg_at_k, precision_at_k
from skrecsys.recommendation import EASE

pytestmark = pytest.mark.benchmark

ESTIMATOR = EASE(l2_reg=500.0)
K = 10
IDENTICAL_LISTS = 0.99
METRIC_TOLERANCE = 1e-3
SCRIPT = Path(__file__).parent / "_rectools_ease.py"


@pytest.fixture(scope="module")
def uv_bin():
    path = shutil.which("uv")
    if path is None:
        pytest.skip("uv not found; it runs RecTools in an isolated environment.")
    return path


def _run_rectools(uv_bin, interactions, l2_reg, tmp_path):
    """Fit RecTools' EASE in a throwaway environment; return its weights and fit time."""
    np.savez(
        tmp_path / "input.npz",
        indptr=interactions.indptr,
        indices=interactions.indices,
        data=interactions.data,
        shape=np.asarray(interactions.shape),
    )
    command = [
        uv_bin, "run", "--isolated", "--quiet", "--with", "rectools", "--with", "pandas",
        "python", str(SCRIPT),
        str(tmp_path / "input.npz"), str(tmp_path / "output.npz"), str(l2_reg),
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        pytest.skip(f"Could not run RecTools in an isolated environment:\n{result.stderr[-2000:]}")

    output = np.load(tmp_path / "output.npz")
    # RecTools numbers items in order of appearance; restore our sorted item order.
    order = np.argsort(output["item_ids"])
    return output["weight"][np.ix_(order, order)], float(output["seconds"])


def _top_k(scores, seen, k):
    """Rank items by descending score, ties by item index, excluding seen items."""
    masked = np.where(seen, -np.inf, scores)
    return np.argsort(-masked, kind="stable")[:, :k]


def test_ease_matches_rectools(uv_bin, movielens_100k_ua, tmp_path, request):
    dataset = movielens_100k_ua
    train, test = dataset.train_indices, dataset.test_indices

    start = time.perf_counter()
    est = clone(ESTIMATOR).fit(dataset.data[train], dataset.target[train])
    ours_seconds = time.perf_counter() - start

    theirs, rectools_seconds = _run_rectools(uv_bin, est.interactions_, ESTIMATOR.l2_reg, tmp_path)

    ratings = est.interactions_.toarray()
    seen = ratings > 0
    start = time.perf_counter()
    ours_top = _top_k(ratings @ est.similarity_, seen, K)
    ours_recommend_seconds = time.perf_counter() - start
    start = time.perf_counter()
    theirs_top = _top_k(ratings @ theirs, seen, K)
    theirs_recommend_seconds = time.perf_counter() - start

    identical = float(np.mean(np.all(ours_top == theirs_top, axis=1)))
    max_diff = float(np.abs(est.similarity_ - theirs).max())
    relative_error = float(
        np.linalg.norm(est.similarity_ - theirs) / np.linalg.norm(est.similarity_)
    )
    scorer_args = (est, dataset.data[test])
    ours_scores = {
        "ndcg": make_recommender_scorer(ndcg_at_k, k=K)(*scorer_args),
        "precision": make_recommender_scorer(precision_at_k, k=K)(*scorer_args),
    }
    theirs_est = copy.deepcopy(est)
    theirs_est.similarity_ = theirs.astype(np.float64)
    theirs_scores = {
        "ndcg": make_recommender_scorer(ndcg_at_k, k=K)(theirs_est, dataset.data[test]),
        "precision": make_recommender_scorer(precision_at_k, k=K)(theirs_est, dataset.data[test]),
    }

    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            f"\nEASE vs RecTools on ML-100k ua ({est.n_items_} items, l2_reg="
            f"{ESTIMATOR.l2_reg})\n"
            f"  {'':10}{'NDCG@10':>10}{'P@10':>10}{'fit s':>10}{'rank s':>10}\n"
            f"  {'skrecsys':10}{ours_scores['ndcg']:>10.4f}{ours_scores['precision']:>10.4f}"
            f"{ours_seconds:>10.3f}{ours_recommend_seconds:>10.3f}\n"
            f"  {'rectools':10}{theirs_scores['ndcg']:>10.4f}{theirs_scores['precision']:>10.4f}"
            f"{rectools_seconds:>10.3f}{theirs_recommend_seconds:>10.3f}\n"
            f"  weights: max abs diff {max_diff:.2e}, relative error {relative_error:.2e}, "
            f"{identical:.1%} identical top-{K} lists"
        )

    # RecTools stores float32, so agreement is bounded by its precision, not ours.
    assert relative_error < 1e-6, relative_error
    np.testing.assert_allclose(est.similarity_, theirs, rtol=1e-6, atol=1e-7)
    assert identical >= IDENTICAL_LISTS, identical
    for name, ours in ours_scores.items():
        assert abs(ours - theirs_scores[name]) <= METRIC_TOLERANCE, (name, ours)
