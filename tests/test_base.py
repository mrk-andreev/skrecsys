import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression

import skrecsys
from skrecsys import RecommenderMixin, _core, is_recommender
from skrecsys.base import first_time_of, seen_among, uses_time
from skrecsys.compose import Cascade, GeneratorScores, PointwiseRanker
from skrecsys.recommendation import ItemKNNRecommender, MostPopularRecommender
from skrecsys.typing import override
from skrecsys.utils.validation import check_ids, check_interactions, encode_ids, factorize


class _ConstantRecommender(RecommenderMixin, BaseEstimator):
    """Queries are ignored; every item scores zero."""

    @override
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

    @override
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


class _MixinOnlyRecommender(RecommenderMixin, BaseEstimator):
    """Implements the mixin contract and nothing else, as a third-party recommender would.

    Scores items by popularity, excludes what a user has seen, ranks two queries per
    block, and records how many queries each call to `_score_queries` was handed.
    """

    @override
    def fit(self, X, y=None):
        users, items, _ = check_interactions(X, y)
        self.user_ids_, user_codes = factorize(users)
        self.item_ids_, item_codes = factorize(items)
        shape = (len(self.user_ids_), len(self.item_ids_))
        self.seen_ = sp.csr_array(
            (np.ones(len(user_codes), dtype=bool), (user_codes, item_codes)), shape=shape
        )
        self.popularity_ = np.bincount(item_codes, minlength=shape[1]).astype(float)
        self.blocks_ = []
        return self

    @override
    def _score_queries(self, X, item_indices, *, exclude_seen):
        rows = encode_ids(check_ids(X), self.user_ids_, name="user")
        self.blocks_.append(len(rows))
        scores = np.tile(self.popularity_[item_indices], (len(rows), 1))
        if not exclude_seen:
            return scores, sp.csr_array(scores.shape, dtype=bool)
        return scores, seen_among(self.seen_, rows, item_indices)

    @override
    def _rank_chunk_size(self, n_candidates, k):
        return 2


_MIXIN_X = np.array(
    [["u0", "a"], ["u0", "b"], ["u1", "b"], ["u2", "c"], ["u3", "a"], ["u3", "d"], ["u4", "e"]],
    dtype=object,
)


def test_eligible_counts_from_the_score_queries_fallback_are_exact_and_blocked():
    rec = _MixinOnlyRecommender().fit(_MIXIN_X)
    queries = np.array(["u0", "u1", "u2", "u3", "u4", "u0"], dtype=object)
    counts = rec._count_eligible(
        queries, candidates=["a", "b", "c", "d"], exclude_interactions=[["u1", "a"]]
    )
    # u0 saw a, b; u1 saw b and excludes a; u2 saw c; u3 saw a, d; u4 saw nothing listed.
    assert counts.tolist() == [2, 2, 3, 2, 4, 2]
    # Scored two queries at a time, like `recommend`, never all six at once.
    assert rec.blocks_ == [2, 2, 2]


def test_eligible_counts_without_exclude_seen_score_nothing():
    rec = _MixinOnlyRecommender().fit(_MIXIN_X)
    counts = rec._count_eligible(np.array(["u0", "u3"], dtype=object), exclude_seen=False)
    assert counts.tolist() == [5, 5]
    assert rec.blocks_ == []


def test_eligible_counts_agree_with_what_recommend_can_return():
    """The count is the most `recommend` serves: one more must raise."""
    rec = _MixinOnlyRecommender().fit(_MIXIN_X)
    for user, count in zip(["u0", "u4"], rec._count_eligible(["u0", "u4"]), strict=True):
        assert rec.recommend([user], n_recommendations=int(count))[0].shape == (1, count)
        with pytest.raises(ValueError, match="eligible items"):
            rec.recommend([user], n_recommendations=int(count) + 1)


def test_a_mixin_only_recommender_can_generate_for_a_cascade():
    """What the fallback is for: a Cascade generator that is not a BaseRecommender."""
    users = np.repeat([f"u{i}" for i in range(20)], 3)
    items = np.array([f"i{(i + j) % 8}" for i in range(20) for j in range(3)], dtype=object)
    X = np.column_stack([users.astype(object), items])
    cascade = Cascade(
        _MixinOnlyRecommender(),
        GeneratorScores(),
        PointwiseRanker(LogisticRegression()),
        n_retrieved=6,
        split=0.4,
    ).fit(X)
    recommended, _ = cascade.recommend(["u0", "u1"], n_recommendations=3)
    assert recommended.shape == (2, 3)
    assert not set(recommended[0]) & {"i0", "i1", "i2"}  # u0's history


def test_first_time_of_is_the_earliest_time_of_each_query():
    users = np.array(["b", "a", "b", "a", "c"])
    times = np.array([5, 3, 1, 9, 4])
    got = first_time_of(users, times, np.array(["a", "b"]))
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, [3, 1])


def test_first_time_of_a_query_without_rows_is_missing():
    np.testing.assert_array_equal(
        first_time_of(np.array([2, 2]), np.array([5, 3]), np.array([1, 2])), [np.nan, 3.0]
    )
    dates = np.array(["2024-01-02", "2024-01-01"], dtype="datetime64[ns]")
    got = first_time_of(np.array(["b", "b"]), dates, np.array(["a", "b"]))
    assert np.isnat(got[0])
    assert got[1] == np.datetime64("2024-01-01")


def test_uses_time_reads_the_time_parameter():
    assert not uses_time(MostPopularRecommender())
    ranker = PointwiseRanker(LogisticRegression())
    cascade = Cascade(MostPopularRecommender(), GeneratorScores(), ranker)
    assert not uses_time(cascade)
    assert uses_time(cascade.set_params(time=True))
