"""Head-to-head benchmark of BM25Recommender against implicit's BM25Recommender.

Both fit the same interaction matrix of MovieLens 100K ``ua``. The algorithm is
deterministic, so the similarity matrices must be identical and recommendations must
agree. Install the reference with ``uv sync --group reference``; without it the
benchmark is skipped.
"""

import time
from collections import defaultdict

import numpy as np
import pytest
import scipy.sparse as sp

from skrecsys.metrics import ndcg_at_k, precision_at_k
from skrecsys.recommendation import BM25Recommender

nearest_neighbours = pytest.importorskip("implicit.nearest_neighbours")

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]

K = 10


def test_bm25_matches_implicit(movielens_100k_ua, request):
    dataset = movielens_100k_ua
    X_train, y_train = dataset.data[dataset.train_indices], dataset.target[dataset.train_indices]
    X_test = dataset.data[dataset.test_indices]

    start = time.perf_counter()
    est = BM25Recommender(n_neighbors=20, k1=1.2, b=0.75).fit(X_train, y_train)
    ours_fit = time.perf_counter() - start

    # Same user/item encoding for both: implicit gets our fitted interaction matrix.
    R = sp.csr_matrix(est.interactions_, dtype=np.float64)
    ref = nearest_neighbours.BM25Recommender(K=est.n_neighbors, K1=est.k1, B=est.b)
    start = time.perf_counter()
    ref.fit(R, show_progress=False)
    ref_fit = time.perf_counter() - start

    ours_sim = sp.csr_matrix(est.similarity_)
    ref_sim = ref.similarity.tocsr()
    for sim in (ours_sim, ref_sim):
        sim.eliminate_zeros()
        sim.sort_indices()
    np.testing.assert_array_equal(ours_sim.indptr, ref_sim.indptr, "kept neighbours differ")
    np.testing.assert_array_equal(ours_sim.indices, ref_sim.indices, "kept neighbours differ")
    assert abs(ours_sim - ref_sim).max() <= 1e-9

    relevant = defaultdict(set)
    for user, item in X_test:
        relevant[user].add(item)
    users = np.array(sorted(relevant))
    user_idx = np.searchsorted(est.user_ids_, users)

    start = time.perf_counter()
    ours_items, ours_scores = est.recommend(users, n_recommendations=K)
    ours_rec = time.perf_counter() - start
    start = time.perf_counter()
    ref_idx, ref_scores = ref.recommend(user_idx, R[user_idx], N=K)
    ref_rec = time.perf_counter() - start
    ref_items = est.item_ids_[ref_idx]

    # Position-wise scores agree regardless of how tied items are ordered.
    np.testing.assert_allclose(ours_scores, ref_scores, rtol=1e-12, atol=1e-9)
    identical = float(np.mean(np.all(ours_items == ref_items, axis=1)))
    assert identical >= 0.99, f"only {identical:.2%} of top-{K} lists are identical"

    y_true = [relevant[u] for u in users]
    ours_metrics = [float(m(y_true, ours_items, k=K)) for m in (ndcg_at_k, precision_at_k)]
    ref_metrics = [float(m(y_true, ref_items, k=K)) for m in (ndcg_at_k, precision_at_k)]
    assert max(abs(a - b) for a, b in zip(ours_metrics, ref_metrics, strict=True)) <= 1e-3

    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(
            f"\nBM25 vs implicit on ML-100k ua ({len(users)} users, "
            f"{identical:.2%} identical top-{K} lists)\n"
            f"  {'':9}{'ndcg@10':>10}{'prec@10':>10}{'fit s':>10}{'rec s':>10}\n"
            f"  {'skrecsys':9}{ours_metrics[0]:>10.4f}{ours_metrics[1]:>10.4f}"
            f"{ours_fit:>10.3f}{ours_rec:>10.3f}\n"
            f"  {'implicit':9}{ref_metrics[0]:>10.4f}{ref_metrics[1]:>10.4f}"
            f"{ref_fit:>10.3f}{ref_rec:>10.3f}"
        )
