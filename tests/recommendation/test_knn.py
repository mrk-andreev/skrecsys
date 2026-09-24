import numpy as np
import pytest

from skrecsys.recommendation import ItemKNNRecommender

# Items: a, b, c. Users u1 and u2 both have a and b; u3 has a and c.
X = [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "b"], ["u3", "a"], ["u3", "c"]]


def _dense(rec):
    return rec.similarity_.toarray()


def test_cosine_similarity():
    rec = ItemKNNRecommender(n_neighbors=None).fit(X)
    norm_a, norm_b, norm_c = np.sqrt(3), np.sqrt(2), 1.0
    expected = np.array(
        [
            [0, 2 / (norm_a * norm_b), 1 / (norm_a * norm_c)],
            [2 / (norm_a * norm_b), 0, 0],
            [1 / (norm_a * norm_c), 0, 0],
        ]
    )
    np.testing.assert_allclose(_dense(rec), expected)


def test_shrink_damps_similarity():
    plain = _dense(ItemKNNRecommender(n_neighbors=None).fit(X))
    shrunk = _dense(ItemKNNRecommender(n_neighbors=None, shrink=5.0).fit(X))
    assert np.all(shrunk <= plain)
    assert shrunk[0, 1] == pytest.approx(2 / (np.sqrt(6) + 5))


def test_n_neighbors_prunes_rows():
    rec = ItemKNNRecommender(n_neighbors=1).fit(X)
    assert np.all(np.diff(rec.similarity_.indptr) <= 1)
    # Item a keeps its most similar neighbor b.
    assert np.flatnonzero(_dense(rec)[0]).tolist() == [1]


def test_scores_match_formula():
    rec = ItemKNNRecommender(n_neighbors=None).fit(X)
    S = _dense(rec)
    R = rec.interactions_.toarray()
    np.testing.assert_allclose(rec.predict([["u3", "b"], ["u1", "c"]]), [R[2] @ S[1], R[0] @ S[2]])


def test_recommend_neighbor():
    rec = ItemKNNRecommender().fit(X)
    items, _ = rec.recommend(["u1"], n_recommendations=1)
    assert items.tolist() == [["c"]]


@pytest.mark.parametrize("params", [{"n_neighbors": 0}, {"n_neighbors": 1.5}, {"shrink": -1.0}])
def test_invalid_params(params):
    with pytest.raises(ValueError, match=next(iter(params))):
        ItemKNNRecommender(**params).fit(X)


def test_partial_fit_recomputes_only_what_the_batch_can_reach():
    """Rows outside the batch's two-hop reach must be left byte-for-byte alone.

    Splicing is what makes the update cheap, and a row spliced in from the wrong place
    is invisible in an equality test against a refit -- both would be wrong together --
    so this pins which rows are touched rather than what they hold.
    """
    users = [f"u{u}" for u in range(6) for _ in range(2)]
    items = ["i0", "i1", "i1", "i2", "i2", "i3", "i4", "i5", "i5", "i6", "i6", "i7"]
    X = np.column_stack([users, items]).astype(object)
    est = ItemKNNRecommender(n_neighbors=5).fit(X)
    before = est.similarity_.toarray()

    # u5 holds i6 and i7, a component that never meets i0..i3, and gains an item.
    est.partial_fit(np.array([["u5", "i8"]], dtype=object))
    after = est.similarity_.toarray()
    untouched = [0, 1, 2, 3, 4, 5]
    np.testing.assert_array_equal(after[untouched, :8], before[untouched])
    assert not after[untouched, 8].any(), "a row the batch cannot reach gained a neighbour."
    assert after[[6, 7], 8].all(), "the rows the batch reaches did not gain the new item."


def test_partial_fit_lets_a_pruned_neighbour_return():
    """Rows are recomputed, not patched, so an entry pruned earlier can come back.

    Patching stored entries would be the obvious shortcut and would quietly make the
    model a one-way ratchet: whatever fell below `n_neighbors` once could never rise
    again, however much later evidence there was for it.
    """
    base = [["u0", "i0"], ["u0", "i1"], ["u1", "i0"], ["u1", "i1"]]
    base += [["u2", "i0"], ["u2", "i2"]]
    est = ItemKNNRecommender(n_neighbors=1).fit(np.array(base, dtype=object))
    neighbours = est.similarity_[[0]].indices
    assert neighbours.tolist() == [1], "i1 should start as i0's only kept neighbour."

    extra = [[f"u{u}", i] for u in range(3, 9) for i in ("i0", "i2")]
    est.partial_fit(np.array(extra, dtype=object))
    assert est.similarity_[[0]].indices.tolist() == [2], "i2 never displaced i1."
