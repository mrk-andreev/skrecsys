"""Head-to-head benchmark of RP3Beta against the reference ``RP3betaRecommender``.

The reference lives in the evaluation framework of Ferrari Dacrema et al., which is a
research repository rather than a package, so the test clones it (sparsely: three
directories, a few megabytes) and runs it through ``uv run --isolated``. Nothing is
added to our own environment. The benchmark is skipped when ``git`` or ``uv`` is
missing, or when either the clone or the isolated run fails, for example offline.

The reference computes in float32, so the weights are compared at float32 precision and
the resulting rankings must agree.
"""

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone

from skrecsys.metrics import make_recommender_scorer, ndcg_at_k, precision_at_k
from skrecsys.recommendation import RP3Beta

pytestmark = pytest.mark.benchmark

ESTIMATOR = RP3Beta(n_neighbors=100, alpha=1.0, beta=0.6)
K = 10
IDENTICAL_LISTS = 0.99
SHARED_NEIGHBOURS = 0.99
METRIC_TOLERANCE = 1e-3
SCRIPT = Path(__file__).parent / "_dacrema_rp3beta.py"
REPOSITORY = "https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation.git"
# Everything RP3betaRecommender imports, and nothing else: the full history is ~900 MB.
CHECKOUT = ("Base", "GraphBased", "Utils")


@pytest.fixture(scope="module")
def tools():
    binaries = {name: shutil.which(name) for name in ("git", "uv")}
    missing = sorted(name for name, path in binaries.items() if path is None)
    if missing:
        pytest.skip(f"{', '.join(missing)} not found; needed to run the reference.")
    return binaries


@pytest.fixture(scope="module")
def reference_repo(tools, tmp_path_factory):
    """A sparse, shallow clone of the reference framework."""
    target = tmp_path_factory.mktemp("dacrema") / "framework"
    commands = [
        [tools["git"], "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         "--quiet", REPOSITORY, str(target)],
        [tools["git"], "-C", str(target), "sparse-checkout", "set", *CHECKOUT],
    ]  # fmt: skip
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
        if result.returncode != 0:
            pytest.skip(f"Could not clone the reference framework:\n{result.stderr[-2000:]}")
    return target


def _run_reference(uv_bin, repo, interactions, estimator, tmp_path):
    """Fit the reference in a throwaway environment; return its weights and fit time."""
    np.savez(
        tmp_path / "input.npz",
        indptr=interactions.indptr,
        indices=interactions.indices,
        data=interactions.data,
        shape=np.asarray(interactions.shape),
    )
    command = [
        uv_bin, "run", "--isolated", "--quiet",
        "--with", "numpy", "--with", "scipy", "--with", "scikit-learn", "--with", "pandas",
        "python", str(SCRIPT), str(repo),
        str(tmp_path / "input.npz"), str(tmp_path / "output.npz"),
        str(estimator.alpha), str(estimator.beta), str(estimator.n_neighbors),
    ]  # fmt: skip
    result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0:
        pytest.skip(
            f"Could not run the reference in an isolated environment:\n{result.stderr[-2000:]}"
        )

    output = np.load(tmp_path / "output.npz")
    weights = sp.csr_array(
        (output["values"], output["indices"], output["indptr"]), shape=tuple(output["shape"])
    )
    return weights, float(output["seconds"])


def _top_k(scores, seen, k):
    """Rank items by descending score, ties by item index, excluding seen items."""
    masked = np.where(seen, -np.inf, scores)
    return np.argsort(-masked, kind="stable")[:, :k]


def test_rp3beta_matches_the_reference(tools, reference_repo, movielens_100k_ua, tmp_path, request):
    dataset = movielens_100k_ua
    train, test = dataset.train_indices, dataset.test_indices

    start = time.perf_counter()
    est = clone(ESTIMATOR).fit(dataset.data[train], dataset.target[train])
    ours_seconds = time.perf_counter() - start

    theirs, reference_seconds = _run_reference(
        tools["uv"], reference_repo, est.interactions_, ESTIMATOR, tmp_path
    )

    ratings = est.interactions_.toarray()
    seen = ratings > 0
    start = time.perf_counter()
    ours_top = _top_k(ratings @ est.similarity_.toarray(), seen, K)
    ours_rank_seconds = time.perf_counter() - start
    start = time.perf_counter()
    theirs_top = _top_k(ratings @ theirs.toarray(), seen, K)
    theirs_rank_seconds = time.perf_counter() - start

    identical = float(np.mean(np.all(ours_top == theirs_top, axis=1)))
    ours_dense, theirs_dense = est.similarity_.toarray(), theirs.toarray()
    shared = (ours_dense != 0) & (theirs_dense != 0)
    agreement = float(shared.sum() / max(est.similarity_.nnz, theirs.nnz))
    residual = ours_dense[shared] - theirs_dense[shared]
    max_diff = float(np.abs(residual).max())
    relative_error = float(np.linalg.norm(residual) / np.linalg.norm(ours_dense[shared]))
    scorers = {"ndcg": make_recommender_scorer(ndcg_at_k, k=K),
               "precision": make_recommender_scorer(precision_at_k, k=K)}  # fmt: skip
    ours_scores = {name: scorer(est, dataset.data[test]) for name, scorer in scorers.items()}
    theirs_est = clone(ESTIMATOR).fit(dataset.data[train], dataset.target[train])
    theirs_est.similarity_ = sp.csr_array(theirs, dtype=np.float64)
    theirs_scores = {
        name: scorer(theirs_est, dataset.data[test]) for name, scorer in scorers.items()
    }

    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            f"\nRP3Beta vs Ferrari Dacrema et al. on ML-100k ua ({est.n_items_} items, "
            f"n_neighbors={ESTIMATOR.n_neighbors}, alpha={ESTIMATOR.alpha}, "
            f"beta={ESTIMATOR.beta})\n"
            f"  {'':10}{'NDCG@10':>10}{'P@10':>10}{'fit s':>10}{'rank s':>10}\n"
            f"  {'skrecsys':10}{ours_scores['ndcg']:>10.4f}{ours_scores['precision']:>10.4f}"
            f"{ours_seconds:>10.3f}{ours_rank_seconds:>10.3f}\n"
            f"  {'reference':10}{theirs_scores['ndcg']:>10.4f}{theirs_scores['precision']:>10.4f}"
            f"{reference_seconds:>10.3f}{theirs_rank_seconds:>10.3f}\n"
            f"  weights: {est.similarity_.nnz} vs {theirs.nnz} nonzeros, "
            f"{agreement:.2%} of them shared, max abs diff {max_diff:.2e}, "
            f"relative error {relative_error:.2e}, {identical:.1%} identical top-{K} lists"
        )

    # Both prune to the same size, so the kept neighbours can differ only where the
    # walk assigns two items exactly the same weight and the tie is broken differently.
    assert agreement >= SHARED_NEIGHBOURS, agreement
    # The reference computes in float32, so agreement is bounded by its precision.
    assert relative_error < 1e-6, relative_error
    np.testing.assert_allclose(ours_dense[shared], theirs_dense[shared], rtol=1e-5, atol=1e-9)
    assert identical >= IDENTICAL_LISTS, identical
    for name, ours in ours_scores.items():
        assert abs(ours - theirs_scores[name]) <= METRIC_TOLERANCE, (name, ours)
