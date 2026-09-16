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
