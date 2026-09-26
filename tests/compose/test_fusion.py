import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from skrecsys._typing import Recommender
from skrecsys.base import is_ranker, is_recommender, serves_unknown_users
from skrecsys.compose import (
    Cascade,
    ConcatFeatures,
    GeneratorScores,
    JoinStaticFeatures,
    KnownUser,
    PointwiseRanker,
    ReciprocalRankFusion,
    ReciprocalRankRanker,
    Switch,
)
from skrecsys.compose._fusion import ranks_per_group
from skrecsys.recommendation import (
    BM25Recommender,
    ItemKNNRecommender,
    MostPopularRecommender,
)
from tests.compose._data import N_USERS, TRENDING, trending_interactions, trending_table

INTERACTIONS = trending_interactions()

# --- the shared core -----------------------------------------------------------------------


def test_ranks_count_from_one_per_group_with_ties_averaged():
    scores = np.array([[3.0], [1.0], [2.0], [5.0], [5.0]])
    ranks = ranks_per_group(scores, np.array([3, 2]))
    np.testing.assert_array_equal(ranks[:, 0], [1.0, 3.0, 2.0, 1.5, 1.5])


def test_a_nan_score_is_unranked_and_pushes_nothing_down():
    scores = np.array([[np.nan], [1.0], [2.0]])
    ranks = ranks_per_group(scores, np.array([3]))
    np.testing.assert_array_equal(ranks[1:, 0], [2.0, 1.0])
    assert np.isnan(ranks[0, 0])


# --- ReciprocalRankRanker --------------------------------------------------------------------


def test_ranker_fuses_feature_columns_by_reciprocal_rank():
    F = np.array([[0.9, 0.1], [0.5, 0.8], [0.1, 0.2]])
    ranker = ReciprocalRankRanker(k=1).fit(F, [1, 0, 0], groups=[3])
    np.testing.assert_allclose(
        ranker.predict(F, groups=[3]), [1 / 2 + 1 / 4, 1 / 3 + 1 / 2, 1 / 4 + 1 / 3]
    )


def test_ranker_ranks_within_each_group():
    F = np.array([[1.0], [2.0], [100.0], [200.0]])
    scores = ReciprocalRankRanker(k=0).fit(F, [0, 1, 0, 1], groups=[2, 2]).predict(F, groups=[2, 2])
    # Only the order inside a group counts, never the scale across groups.
    np.testing.assert_allclose(scores, [1 / 2, 1, 1 / 2, 1])


def test_ranker_weights_the_columns():
    F = np.array([[1.0, 0.0], [0.0, 1.0]])
    scores = ReciprocalRankRanker(k=0, weights=[3, 1]).fit(F, [1, 0]).predict(F)
    np.testing.assert_allclose(scores, [3 + 1 / 2, 3 / 2 + 1])


def test_a_nan_feature_gives_its_row_nothing_from_that_column():
    F = np.array([[np.nan, 1.0], [1.0, 0.0]])
    scores = ReciprocalRankRanker(k=0).fit(F, [1, 0]).predict(F)
    np.testing.assert_allclose(scores, [1.0, 1 + 1 / 2])


def test_ranker_fuses_the_ranks_of_its_rankers():
    X = np.array([[0.0], [1.0], [0.2], [0.9], [0.1], [0.7]])
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array([2, 2, 2])
    fusion = ReciprocalRankRanker(
        [
            ("up", PointwiseRanker(LogisticRegression())),
            ("again", PointwiseRanker(LogisticRegression())),
        ],
        k=0,
    ).fit(X, y, groups=groups)
    assert [name for name, _ in fusion.rankers_] == ["up", "again"]
    np.testing.assert_allclose(fusion.predict(X, groups=groups), [1.0, 2.0] * 3)


def test_ranker_exposes_nested_params():
    fusion = ReciprocalRankRanker([PointwiseRanker(LogisticRegression())])
    assert "pointwiseranker__estimator__C" in fusion.get_params()
    fusion.set_params(pointwiseranker__estimator__C=0.5, k=10)
    assert fusion.get_params()["pointwiseranker__estimator__C"] == 0.5
    assert fusion.k == 10
    assert clone(fusion).set_params(rankers=None).rankers is None


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"k": -1}, ValueError, "k must be"),
        ({"weights": [1.0]}, ValueError, "one entry per source"),
        ({"weights": [1.0, -1.0]}, ValueError, "non-negative"),
        ({"rankers": []}, ValueError, "at least one ranker"),
        ({"rankers": [LogisticRegression()]}, TypeError, "not a ranker"),
    ],
)
def test_ranker_validates(params, error, match):
    with pytest.raises(error, match=match):
        ReciprocalRankRanker(**params).fit(np.zeros((2, 2)), [0, 1])


def test_ranker_checks_the_feature_count():
    ranker = ReciprocalRankRanker().fit(np.zeros((2, 2)), [0, 1])
    with pytest.raises(ValueError, match="fitted with 2"):
        ranker.predict(np.zeros((2, 3)))


def test_ranker_is_a_ranker():
    assert is_ranker(ReciprocalRankRanker())
    assert not is_recommender(ReciprocalRankRanker())


def test_ranker_is_the_training_free_second_stage_of_a_cascade():
    """The trending flag outranks BM25 order once both are fused, with nothing learned."""
    rec = Cascade(
        MostPopularRecommender(),
        ConcatFeatures([JoinStaticFeatures("item", trending_table()), GeneratorScores()]),
        ReciprocalRankRanker(weights=[2.0, 1.0]),
        n_retrieved=30,
        split=0.2,
    ).fit(INTERACTIONS)
    items, _ = rec.recommend(np.arange(N_USERS), n_recommendations=1)
    baseline, _ = (
        MostPopularRecommender()
        .fit(INTERACTIONS)
        .recommend(np.arange(N_USERS), n_recommendations=1)
    )
    assert np.isin(items, TRENDING).mean() > np.isin(baseline, TRENDING).mean()


# --- ReciprocalRankFusion --------------------------------------------------------------------


def _reference(members, users, k, weights, n_retrieved):
    """The fused lists, computed one user at a time from the members' own lists."""
    fitted = [clone(member).fit(INTERACTIONS) for member in members]
    out = []
    for user in users:
        scores: dict[int, float] = {}
        for member, weight in zip(fitted, weights, strict=True):
            items, _ = member.recommend([user], n_recommendations=n_retrieved)
            for rank, item in enumerate(items[0].tolist(), start=1):
                scores[item] = scores.get(item, 0.0) + weight / (k + rank)
        out.append(scores)
    return out


def test_fusion_scores_every_item_by_its_reciprocal_ranks():
    members: list[Recommender] = [BM25Recommender(), ItemKNNRecommender()]
    fusion = ReciprocalRankFusion(members, k=5, weights=[1.0, 2.0], n_retrieved=8)
    fusion.fit(INTERACTIONS)
    users = np.arange(10)
    items, scores = fusion.recommend(users, n_recommendations=5)
    for row, expected in enumerate(_reference(members, users, 5, [1.0, 2.0], 8)):
        best = sorted(expected.items(), key=lambda pair: (-pair[1], pair[0]))[:5]
        np.testing.assert_allclose(scores[row], [score for _, score in best])
        assert set(items[row]) == {item for item, _ in best}


def test_fusion_predict_agrees_with_recommend_without_exclusions():
    fusion = ReciprocalRankFusion([BM25Recommender(), MostPopularRecommender()]).fit(INTERACTIONS)
    items, scores = fusion.recommend([0, 1], n_recommendations=4, exclude_seen=False)
    pairs = np.column_stack([np.repeat([0, 1], 4), items.ravel()])
    np.testing.assert_allclose(fusion.predict(pairs), scores.ravel())


def test_a_zero_weight_leaves_the_other_list_alone():
    fusion = ReciprocalRankFusion(
        [BM25Recommender(), MostPopularRecommender()], weights=[1.0, 0.0], n_retrieved=30
    ).fit(INTERACTIONS)
    alone = BM25Recommender().fit(INTERACTIONS)
    users = np.arange(20)
    np.testing.assert_array_equal(
        fusion.recommend(users, n_recommendations=3)[0],
        alone.recommend(users, n_recommendations=3)[0],
    )


def test_an_unknown_user_is_served_by_the_members_that_serve_one():
    fusion = ReciprocalRankFusion([ItemKNNRecommender(), MostPopularRecommender()])
    assert serves_unknown_users(fusion)
    fusion.fit(INTERACTIONS)
    popular = MostPopularRecommender().fit(INTERACTIONS)
    np.testing.assert_array_equal(
        fusion.recommend([10_000], n_recommendations=3)[0],
        popular.recommend([10_000], n_recommendations=3)[0],
    )
    assert not serves_unknown_users(ReciprocalRankFusion([ItemKNNRecommender()]))
    knn_only = ReciprocalRankFusion([ItemKNNRecommender()]).fit(INTERACTIONS)
    with pytest.raises(ValueError, match="eligible items"):
        knn_only.recommend([10_000], n_recommendations=1)
    with pytest.raises(ValueError, match="Unknown user"):
        knn_only.predict([[10_000, 0]])


def test_fusion_caps_recommendations_at_n_retrieved():
    fusion = ReciprocalRankFusion([MostPopularRecommender()], n_retrieved=3).fit(INTERACTIONS)
    with pytest.raises(ValueError, match="exceeds n_retrieved"):
        fusion.recommend([0], n_recommendations=4)


def test_fusion_exposes_nested_params():
    fusion = ReciprocalRankFusion([("bm25", BM25Recommender()), ("pop", MostPopularRecommender())])
    assert fusion.get_params()["bm25__k1"] == 1.2
    fusion.set_params(bm25__k1=2.0, k=10)
    assert fusion.get_params()["bm25__k1"] == 2.0
    fusion.set_params(pop=ItemKNNRecommender())
    assert isinstance(fusion.get_params()["pop"], ItemKNNRecommender)


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        ({"k": -1}, ValueError, "k must be"),
        ({"weights": [1.0, 2.0]}, ValueError, "one entry per source"),
        ({"n_retrieved": 0}, ValueError, "n_retrieved"),
    ],
)
def test_fusion_validates(params, error, match):
    fusion = ReciprocalRankFusion([MostPopularRecommender()]).set_params(**params)
    with pytest.raises(error, match=match):
        fusion.fit(INTERACTIONS)


def test_fusion_needs_recommenders():
    with pytest.raises(ValueError, match="at least one recommender"):
        ReciprocalRankFusion([]).fit(INTERACTIONS)
    with pytest.raises(TypeError, match="not a recommender"):
        ReciprocalRankFusion([MostPopularRecommender()]).set_params(
            recommenders=[LogisticRegression()]
        )


def test_fusion_nests_in_a_switch_and_as_a_cascade_generator():
    fusion = ReciprocalRankFusion([BM25Recommender(), ItemKNNRecommender()])
    switch = Switch(KnownUser(), fusion, MostPopularRecommender()).fit(INTERACTIONS)
    assert switch.recommend([0, 10_000], n_recommendations=2)[0].shape == (2, 2)
    cascade = Cascade(
        fusion,
        ConcatFeatures([JoinStaticFeatures("item", trending_table()), GeneratorScores()]),
        ReciprocalRankRanker(),
        n_retrieved=20,
    ).fit(INTERACTIONS)
    restored = pickle.loads(pickle.dumps(cascade))
    np.testing.assert_array_equal(
        restored.recommend([0, 1], n_recommendations=3)[0],
        cascade.recommend([0, 1], n_recommendations=3)[0],
    )
