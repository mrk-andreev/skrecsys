import heapq

import numpy as np
import pytest
import scipy.sparse as sp

from skrecsys.recommendation import BM25Recommender


def _counts(seed=0, n_users=40, n_items=15, density=0.3):
    rng = np.random.default_rng(seed)
    mask = rng.random((n_users, n_items)) < density
    mask[np.arange(n_users), np.arange(n_users) % n_items] = True
    users, items = np.nonzero(mask)
    return np.column_stack([users, items]), rng.integers(1, 6, len(users)).astype(float)


def _bm25_weight(X, K1, B):
    """``implicit.nearest_neighbours.bm25_weight`` (rows are documents, columns terms)."""
    X = sp.coo_matrix(X)
    idf = np.log(float(X.shape[0])) - np.log1p(np.bincount(X.col, minlength=X.shape[1]))
    row_sums = np.ravel(X.sum(axis=1))
    length_norm = (1.0 - B) + B * row_sums / row_sums.mean()
    X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
    return X


def _all_pairs_knn(W, K):
    """``implicit``'s ``all_pairs_knn``: per-row top K of ``W^T W`` with its tie rule."""
    W = sp.csr_matrix(W)
    W.sort_indices()
    items = W.T.tocsr()
    rows = []
    for i in range(W.shape[1]):
        sums, touched = {}, []
        for p in range(items.indptr[i], items.indptr[i + 1]):
            u, w1 = items.indices[p], items.data[p]
            for q in range(W.indptr[u], W.indptr[u + 1]):
                j = W.indices[q]
                if j not in sums:
                    sums[j] = 0.0
                    touched.append(j)
                sums[j] += W.data[q] * w1
        heap = []  # min-heap of (score, index), like std::greater on std::pair
        for j in reversed(touched):
            if len(heap) < K or sums[j] > heap[0][0]:
                if len(heap) >= K:
                    heapq.heappop(heap)
                heapq.heappush(heap, (sums[j], j))
        rows.append({j: s for s, j in heap})
    dense = np.zeros((W.shape[1], W.shape[1]))
    for i, row in enumerate(rows):
        for j, s in row.items():
            dense[i, j] = s
    return dense, [set(row) for row in rows]


def _reference_similarity(interactions, K, K1, B):
    weighted = _bm25_weight(sp.csr_matrix(interactions).T, K1, B).T
    return _all_pairs_knn(weighted, K)


def test_similarity_matches_implicit_reference():
    X, y = _counts()
    est = BM25Recommender(n_neighbors=5, k1=1.5, b=0.6).fit(X, y)
    expected, neighbours = _reference_similarity(est.interactions_, 5, 1.5, 0.6)
    np.testing.assert_allclose(est.similarity_.toarray(), expected, rtol=0, atol=1e-12)
    assert [set(est.similarity_[[i]].indices) for i in range(est.n_items_)] == neighbours


def test_ties_keep_the_same_neighbours_as_implicit():
    # Binary interactions with identical columns produce many tied similarities.
    X = [[u, i] for u in range(6) for i in range(5) if (u + i) % 3 != 0]
    est = BM25Recommender(n_neighbors=2).fit(X)
    _, neighbours = _reference_similarity(est.interactions_, 2, 1.2, 0.75)
    assert [set(est.similarity_[[i]].indices) for i in range(est.n_items_)] == neighbours


def test_scores_sum_neighbour_rows_of_user_items():
    X, y = _counts(seed=1)
    est = BM25Recommender(n_neighbors=4).fit(X, y)
    users = np.arange(est.n_users_)
    expected = est.interactions_.toarray() @ est.similarity_.toarray()
    np.testing.assert_allclose(est._score_users(users, np.arange(est.n_items_)), expected)


def test_n_neighbors_bounds_row_size_and_includes_self():
    X, y = _counts(seed=2)
    est = BM25Recommender(n_neighbors=3).fit(X, y)
    assert np.diff(est.similarity_.indptr).max() <= 3
    full = BM25Recommender(n_neighbors=est.n_items_).fit(X, y)
    assert np.all(full.similarity_.diagonal() > 0)


def test_thread_count_does_not_change_result():
    X, y = _counts(seed=3, n_users=300, n_items=120)
    one = BM25Recommender(n_neighbors=7, n_jobs=1).fit(X, y).similarity_
    many = BM25Recommender(n_neighbors=7, n_jobs=-1).fit(X, y).similarity_
    np.testing.assert_array_equal(one.indptr, many.indptr)
    np.testing.assert_array_equal(one.indices, many.indices)
    np.testing.assert_array_equal(one.data, many.data)


def test_b_zero_disables_length_normalization():
    X, y = _counts(seed=4)
    est = BM25Recommender(n_neighbors=15, b=0.0).fit(X, y)
    R = est.interactions_.toarray()
    idf = np.log(est.n_items_) - np.log1p((R != 0).sum(axis=1))
    W = np.where(R != 0, R * 2.2 / (1.2 + R), 0.0) * idf[:, None]
    np.testing.assert_allclose(est.similarity_.toarray(), W.T @ W, atol=1e-12)


@pytest.mark.parametrize(
    "params",
    [
        {"n_neighbors": 0},
        {"n_neighbors": 2.5},
        {"k1": -1.0},
        {"b": 1.5},
        {"b": float("nan")},
        {"n_jobs": 0},
        {"n_jobs": -2},
    ],
)
def test_invalid_params(params):
    X, y = _counts()
    with pytest.raises(ValueError, match=next(iter(params))):
        BM25Recommender(**params).fit(X, y)
