import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

import skrecsys
from skrecsys import RecommenderMixin, _core, is_recommender
from skrecsys._typing import override
from skrecsys.recommendation import ItemKNNRecommender, MostPopularRecommender


class _ConstantRecommender(RecommenderMixin, BaseEstimator):
    """Queries are ignored; every item scores zero."""

    def fit(self, X, y=None):
        self.item_ids_ = np.array(["a", "b", "c"], dtype=object)
        return self

    @override
    def _score_queries(self, X, item_indices, *, exclude_seen):
        shape = (len(X), len(item_indices))
        return np.zeros(shape), sp.csr_array(shape, dtype=bool)


def test_version():
    assert isinstance(skrecsys.__version__, str)


def test_is_recommender():
    assert is_recommender(_ConstantRecommender())
    assert is_recommender(MostPopularRecommender())
    assert not is_recommender(LinearRegression())


def test_ties_follow_fitted_item_order():
    rec = _ConstantRecommender().fit(None)
    items, scores = rec.recommend(["q1", "q2"], n_recommendations=3)
    assert items.tolist() == [["a", "b", "c"], ["a", "b", "c"]]
    np.testing.assert_array_equal(scores, 0.0)


def test_candidates_are_sorted_by_fitted_order():
    rec = _ConstantRecommender().fit(None)
    items, _ = rec.recommend(["q"], n_recommendations=2, candidates=["c", "a", "c"])
    assert items.tolist() == [["a", "c"]]


@pytest.mark.parametrize("n", [0, -1])
def test_invalid_n_recommendations_value(n):
    with pytest.raises(ValueError, match="n_recommendations"):
        _ConstantRecommender().fit(None).recommend(["q"], n_recommendations=n)


@pytest.mark.parametrize("n", [1.5, True, "3"])
def test_invalid_n_recommendations_type(n):
    with pytest.raises(TypeError, match="n_recommendations"):
        _ConstantRecommender().fit(None).recommend(["q"], n_recommendations=n)


def test_too_few_items_never_pads():
    with pytest.raises(ValueError, match="eligible"):
        _ConstantRecommender().fit(None).recommend(["q"], n_recommendations=4)


def test_recommend_requires_fit():
    with pytest.raises(NotFittedError):
        ItemKNNRecommender().recommend(["u"])


class _ScoredRecommender(RecommenderMixin, BaseEstimator):
    """Returns fixed scores and eligibility, to exercise ranking alone."""

    def __init__(self, scores, eligible):
        self.scores = np.asarray(scores, dtype=float)
        self.eligible = np.asarray(eligible, dtype=bool)

    def fit(self, X=None, y=None):
        self.item_ids_ = np.arange(self.scores.shape[1])
        return self

    @override
    def _score_queries(self, X, item_indices, *, exclude_seen):
        return self.scores[:, item_indices], sp.csr_array(~self.eligible[:, item_indices])


@pytest.mark.parametrize("n_recommendations", [1, 3, 10])
def test_ranking_matches_a_full_stable_sort(n_recommendations):
    rng = np.random.default_rng(0)
    # Few distinct scores, so ties are common and the tie rule is exercised.
    scores = rng.integers(0, 4, (25, 40)).astype(float)
    eligible = rng.random((25, 40)) < 0.8
    rec = _ScoredRecommender(scores, eligible).fit()

    items, top_scores = rec.recommend(np.arange(25), n_recommendations=n_recommendations)

    order = np.argsort(np.where(eligible, -scores, np.inf), axis=1, kind="stable")
    expected = order[:, :n_recommendations]
    np.testing.assert_array_equal(items, expected)
    np.testing.assert_array_equal(top_scores, np.take_along_axis(scores, expected, axis=1))


def test_ineligible_items_are_never_recommended():
    scores = np.array([[5.0, 4.0, 3.0]])
    eligible = np.array([[False, True, True]])
    items, top_scores = (
        _ScoredRecommender(scores, eligible).fit().recommend(["q"], n_recommendations=2)
    )
    assert items.tolist() == [[1, 2]]
    assert top_scores.tolist() == [[4.0, 3.0]]


def test_too_few_eligible_items_names_the_query():
    scores = np.zeros((2, 3))
    eligible = np.array([[True, True, True], [True, False, False]])
    rec = _ScoredRecommender(scores, eligible).fit()
    with pytest.raises(ValueError, match="query 1 has only 1 eligible"):
        rec.recommend(["q0", "q1"], n_recommendations=2)


def _order(scores, indptr, indices, k):
    return _core.top_k_per_row(
        np.ascontiguousarray(scores, dtype=np.float64),
        np.asarray(indptr, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
        k,
    )


def test_kernel_rejects_exclusions_that_do_not_match_the_scores():
    scores = np.zeros((2, 3))
    with pytest.raises(ValueError, match="one entry per row"):
        _order(scores, [0, 0], [], 1)
    with pytest.raises(ValueError, match="out of range"):
        _order(scores, [0, 1, 1], [3], 1)


def test_kernel_rejects_unsorted_exclusions():
    # The selection walks the exclusions alongside the scores, so a row that does not
    # ascend would silently keep an excluded item instead of dropping it.
    with pytest.raises(ValueError, match="ascend"):
        _order(np.zeros((1, 3)), [0, 2], [2, 0], 1)


def test_too_few_eligible_names_the_query_across_blocks(monkeypatch):
    """The reported query is the caller's, not the one inside the block."""
    eligible = np.ones((4, 3), dtype=bool)
    eligible[3] = [True, False, False]
    rec = _ScoredRecommender(np.zeros((4, 3)), eligible).fit()
    monkeypatch.setattr(type(rec), "_rank_chunk_size", lambda self, n_candidates, k: 2)
    with pytest.raises(ValueError, match="query 3 has only 1 eligible"):
        rec.recommend(["q0", "q1", "q2", "q3"], n_recommendations=2)
