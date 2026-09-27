"""Why recommenders score what they score, and :func:`skrecsys.inspection.explain`."""

import json

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from skrecsys._attribution import Attributions
from skrecsys.base import RecommenderMixin
from skrecsys.compose import (
    Cascade,
    GeneratorScores,
    KnownUser,
    PointwiseRanker,
    ReciprocalRankFusion,
    Switch,
)
from skrecsys.inspection import QueryTrace, ServedStep, explain, trace
from skrecsys.metrics import make_recommender_scorer, ndcg_at_k
from skrecsys.recommendation import (
    EASE,
    AlternatingLeastSquares,
    BayesianPersonalizedRanking,
    BM25Recommender,
    ItemKNNRecommender,
    MostPopularRecommender,
    RP3Beta,
    SLIMElasticNet,
)
from skrecsys.tune import AutoTune, Int

X = np.array([[u, i] for u in range(40) for i in (u % 7, u % 7 + 1, (u * 3) % 9, (u + 2) % 11)])
WEIGHTS = np.random.default_rng(0).random(len(X)) + 0.5
USERS = np.repeat(np.arange(6), 4)
ITEMS = np.tile([0, 4, 9, 10], 6)

EXACT = [
    ItemKNNRecommender(),
    BM25Recommender(),
    RP3Beta(),
    SLIMElasticNet(alpha=0.01),
    EASE(),
]
APPROXIMATE = [
    AlternatingLeastSquares(n_factors=3, random_state=0),
    BayesianPersonalizedRanking(random_state=0),
]


def attribute(rec: RecommenderMixin, users, items, n_reasons: int) -> Attributions:
    found = rec._attribute(np.asarray(users), np.asarray(items), n_reasons)
    assert found is not None
    return found


def final_of(query_trace: QueryTrace) -> ServedStep:
    final = query_trace.final
    assert final is not None
    return final


class TestAttributions:
    @pytest.mark.parametrize("estimator", EXACT, ids=lambda e: type(e).__name__)
    def test_history_terms_add_up_to_the_score(self, estimator):
        rec = estimator.fit(X, WEIGHTS)
        found = attribute(rec, USERS, ITEMS, 2)
        assert (found.kind, found.exact) == ("history", True)
        total = np.nansum(found.weights, axis=1) + found.rest
        np.testing.assert_allclose(total, rec.predict(np.column_stack([USERS, ITEMS])))

    @pytest.mark.parametrize("estimator", EXACT, ids=lambda e: type(e).__name__)
    def test_reasons_are_history_items_largest_first(self, estimator):
        rec = estimator.fit(X, WEIGHTS)
        found = attribute(rec, USERS, ITEMS, 3)
        for user, items, weights in zip(USERS, found.items, found.weights, strict=True):
            history = set(X[X[:, 0] == user, 1].tolist())
            listed = [i for i in items.tolist() if i is not None]
            assert set(listed) <= history
            magnitudes = np.abs(weights[: len(listed)])
            assert np.all(np.diff(magnitudes) <= 0)
            assert np.all(weights[: len(listed)] != 0)
            assert np.isnan(weights[len(listed) :]).all()

    @pytest.mark.parametrize("estimator", APPROXIMATE, ids=lambda e: type(e).__name__)
    def test_factor_models_name_the_most_alike_history_items(self, estimator):
        rec = estimator.fit(X)
        found = attribute(rec, USERS, ITEMS, 2)
        assert (found.kind, found.exact) == ("similar_history", False)
        assert np.isnan(found.rest).all()
        for item, reasons in zip(ITEMS, found.items, strict=True):
            assert item not in reasons.tolist()

    def test_popularity_is_all_of_the_score(self):
        rec = MostPopularRecommender().fit(X)
        found = attribute(rec, np.array([0, 999]), np.array([0, 1]), 3)
        assert found.kind == "popularity"
        assert found.items.shape == (2, 0)
        np.testing.assert_array_equal(found.rest, rec.item_popularity_[[0, 1]])

    def test_composites_have_none_of_their_own(self):
        rec = ReciprocalRankFusion([ItemKNNRecommender(), MostPopularRecommender()]).fit(X)
        assert rec._attribute(np.array([0]), np.array([0]), 3) is None


class TestTracedReasons:
    def test_a_full_trace_records_why_each_leaf_served_its_items(self):
        rec = ItemKNNRecommender().fit(X)
        with trace(n_reasons=2) as t:
            items, _ = rec.recommend([0, 1], n_recommendations=3)
        (step,) = t[1].attributions
        np.testing.assert_array_equal(step.items, items[1])
        want = attribute(rec, np.full(3, 1), items[1], 2)
        np.testing.assert_array_equal(step.weights, want.weights)
        assert step.of(items[1][0]) == [
            (i, w) for i, w in zip(want.items[0], want.weights[0], strict=True) if i is not None
        ]

    def test_decisions_record_no_reasons(self):
        rec = ItemKNNRecommender().fit(X)
        with trace(level="decisions") as t:
            rec.recommend([0], n_recommendations=3)
        assert not t[0].attributions

    def test_a_failing_attribution_is_recorded_not_raised(self, monkeypatch):
        rec = ItemKNNRecommender().fit(X)
        want = rec.recommend([0], n_recommendations=3)

        def broken(*args):
            raise RuntimeError("boom")

        monkeypatch.setattr(rec, "_attribute", broken)
        with trace() as t:
            got = rec.recommend([0], n_recommendations=3)
        np.testing.assert_array_equal(got[0], want[0])
        (raised,) = t[0].raised
        assert raised.error == "AttributionError"
        assert "boom" in raised.message

    def test_a_linear_ranker_reports_its_log_odds_terms(self):
        rec = cascade(PointwiseRanker(LogisticRegression())).fit(X)
        with trace() as t:
            rec.recommend([0], n_recommendations=3)
        (ranker,) = t[0].ranker
        (features,) = t[0].features
        model = rec.ranker_.estimator_
        contributions = ranker.contributions
        assert contributions is not None
        np.testing.assert_allclose(
            contributions.sum(axis=1), model.decision_function(features.values)
        )

    def test_a_ranker_that_cannot_say_reports_nothing(self):
        rec = cascade(PointwiseRanker(HistGradientBoostingClassifier(max_iter=5))).fit(X)
        with trace() as t:
            rec.recommend([0], n_recommendations=3)
        assert t[0].ranker[0].contributions is None


def cascade(ranker, **params):
    return Cascade(
        [ItemKNNRecommender(), MostPopularRecommender()],
        GeneratorScores(n_generators=2),
        ranker,
        n_retrieved=5,
        split=0.4,
        **params,
    )


def drop_first(pairs, scores, groups):
    """Postprocess: drop each query's best-ranked candidate."""
    starts = np.cumsum(groups) - groups
    keep = np.ones(len(pairs), dtype=bool)
    keep[starts] = False
    return pairs[keep], scores[keep], groups - 1


def by_item(explanations):
    return {e.item: e for e in explanations}


class TestExplain:
    def test_served_items_come_first_best_first(self):
        rec = ItemKNNRecommender().fit(X)
        items, scores = rec.recommend([0], n_recommendations=3)
        found = explain(rec, [0], n_recommendations=3)
        assert [e.item for e in found] == items[0].tolist()
        assert [e.position for e in found] == [1, 2, 3]
        np.testing.assert_allclose([e.score for e in found], scores[0])
        assert all(e.status == "served" for e in found)
        (reasons,) = found[0].reasons
        assert reasons.exact
        assert reasons.rest is not None
        assert sum(w for _, w in reasons.reasons) + reasons.rest == pytest.approx(scores[0][0])

    def test_why_not_names_the_filter(self):
        rec = ItemKNNRecommender().fit(X)
        seen, other, unseen = 0, 4, 9
        found = by_item(
            explain(
                rec,
                [0],
                items=[seen, other, unseen, 999],
                n_recommendations=2,
                exclude_interactions=[[0, other]],
            )
        )
        assert (found[seen].status, found[seen].excluded_by) == ("excluded", "exclude_seen")
        assert (found[other].status, found[other].excluded_by) == (
            "excluded",
            "exclude_interactions",
        )
        assert found[999].status == "unknown_item"
        assert found[999].reasons == ()
        found = by_item(explain(rec, [0], items=[unseen], n_recommendations=1, candidates=[5, 6]))
        assert (found[unseen].status, found[unseen].excluded_by) == ("excluded", "candidates")

    def test_a_leaf_ranks_the_rest_out(self):
        rec = ItemKNNRecommender().fit(X)
        items, scores = rec.recommend([0], n_recommendations=1)
        others = [i for i in rec.item_ids_.tolist() if i not in X[X[:, 0] == 0, 1]]
        found = by_item(explain(rec, [0], items=others, n_recommendations=1))
        for item in others:
            if item == items[0][0]:
                continue
            assert found[item].status == "ranked_out"
            assert found[item].cutoff == scores[0][0]
            assert found[item].score <= scores[0][0]
            assert "below the cut-off" in found[item].detail

    def test_a_cascade_tells_retrieval_from_ranking(self):
        rec = cascade(PointwiseRanker(LogisticRegression())).fit(X)
        with trace() as t:
            rec.recommend([0], n_recommendations=2)
        union = next(c for c in t[0].candidates if c.source == "union")
        served = final_of(t[0]).items.tolist()
        ranked_out = next(i for i in union.items.tolist() if i not in served)
        unseen = set(rec.item_ids_.tolist()) - set(X[X[:, 0] == 0, 1].tolist())
        missed = next(i for i in sorted(unseen) if i not in union.items.tolist())

        found = by_item(explain(rec, [0], items=[ranked_out, missed], n_recommendations=2))
        out = found[ranked_out]
        assert out.status == "ranked_out"
        assert out.ranker.position > 2
        assert out.ranker.n_candidates == len(union.items)
        assert "bias" in out.ranker.contributions
        assert {r.source for r in out.retrieval} == {
            "itemknnrecommender",
            "mostpopularrecommender",
            "union",
        }
        assert found[missed].status == "not_retrieved"
        assert found[missed].ranker is None
        # A generator may have proposed it, but not within the merged budget.
        (merged,) = [r for r in found[missed].retrieval if r.source == "union"]
        assert merged.rank is None
        proposed = [r for r in found[missed].retrieval if r.rank is not None]
        want = "cut from the merged top 5" if proposed else "no generator retrieved it"
        assert want in found[missed].detail
        # Each leaf the query reached explains the item, retrieved or not.
        kinds = {r.path.rsplit("/", 1)[-1]: r.kind for r in found[missed].reasons}
        assert kinds == {
            "ItemKNNRecommender": "history",
            "MostPopularRecommender": "popularity",
        }

    def test_ranker_ties_are_ranked_as_they_are_served(self):
        # A constant ranker ties every candidate: recommend breaks ties by fitted item
        # order, not candidate order, and the explanation must say the same.
        rec = cascade(PointwiseRanker(DummyClassifier())).fit(X)
        with trace() as t:
            items, _ = rec.recommend([0], n_recommendations=2)
        union = next(c for c in t[0].candidates if c.source == "union")
        assert union.items.tolist() != sorted(union.items.tolist())
        found = by_item(explain(rec, [0], items=union.items.tolist(), n_recommendations=2))
        for item, e in found.items():
            assert e.ranker is not None
            served = item in items[0].tolist()
            assert (e.ranker.position <= 2) == served
            if served:
                assert e.ranker.position == e.position

    def test_postprocess_is_blamed_for_what_it_drops(self):
        rec = cascade(PointwiseRanker(LogisticRegression()), postprocess=drop_first).fit(X)
        with trace() as t:
            rec.recommend([0], n_recommendations=2)
        best = t[0].ranker[0].ranked()[0]
        (found,) = [
            e for e in explain(rec, [0], items=[best], n_recommendations=2) if e.item == best
        ]
        assert found.status == "dropped_by_postprocess"
        assert found.ranker is not None
        assert found.ranker.position == 1
        assert found.detail.startswith("ranked #1 of")

    def test_routes_and_reasons_through_nested_composites(self):
        rec = Switch(
            KnownUser(), cascade(PointwiseRanker(LogisticRegression())), MostPopularRecommender()
        ).fit(X)
        warm, cold = explain(rec, [0, 999], n_recommendations=1)
        assert [(r.condition, r.branch) for r in warm.routes] == [("KnownUser()", "on_true")]
        assert {r.path for r in warm.reasons} == {
            "Switch/on_true/Cascade/itemknnrecommender/ItemKNNRecommender",
            "Switch/on_true/Cascade/mostpopularrecommender/MostPopularRecommender",
        }
        assert warm.ranker is not None
        assert warm.ranker.path == "Switch/on_true/Cascade"
        assert [r.branch for r in cold.routes] == ["on_false"]
        assert [r.path for r in cold.reasons] == ["Switch/on_false/MostPopularRecommender"]

    def test_nested_fusion_follows_the_outer_fusion(self):
        # The nested Cascade retrieves by popularity only, so the KNN branch fuses items
        # its ranker never saw.
        nested_cascade = Cascade(
            MostPopularRecommender(),
            GeneratorScores(),
            PointwiseRanker(LogisticRegression()),
            n_retrieved=3,
            split=0.4,
        )
        rec = ReciprocalRankFusion([nested_cascade, ItemKNNRecommender()], n_retrieved=3).fit(X)
        with trace() as t:
            rec.recommend([0], n_recommendations=1)
        fused = {i for step in t[0].fusion if step.path == t[0].path for i in step.items.tolist()}
        (nested,) = t[0].ranker
        assert nested.path == "ReciprocalRankFusion/cascade/Cascade"
        served = final_of(t[0]).items.tolist()
        unseen = set(rec.item_ids_.tolist()) - set(X[X[:, 0] == 0, 1].tolist())
        found = by_item(explain(rec, [0], items=sorted(unseen), n_recommendations=1))
        for item in unseen - set(served):
            want = "ranked_out" if item in fused else "not_retrieved"
            assert found[item].status == want, item
            assert found[item].ranker is None
        # Retrieved by the KNN branch and fused, though the nested ranker never saw it.
        assert any(
            found[item].status == "ranked_out"
            for item in fused - set(served) - set(nested.items.tolist())
        )

    def test_nested_postprocess_does_not_decide_for_the_outer_stage(self):
        rec = ReciprocalRankFusion(
            [
                cascade(PointwiseRanker(LogisticRegression()), postprocess=drop_first),
                ItemKNNRecommender(),
            ],
            n_retrieved=3,
        ).fit(X)
        with trace() as t:
            rec.recommend([0], n_recommendations=1)
        (post,) = t[0].postprocess
        fused = {i for step in t[0].fusion if step.path == t[0].path for i in step.items.tolist()}
        served = final_of(t[0]).items.tolist()
        found = by_item(explain(rec, [0], items=np.asarray(post.dropped), n_recommendations=1))
        for item in set(post.dropped) - set(served):
            want = "ranked_out" if item in fused else "not_retrieved"
            assert found[item].status == want, item

    def test_fusion_tells_retrieval_from_ranking(self):
        rec = ReciprocalRankFusion(
            [ItemKNNRecommender(), MostPopularRecommender()], n_retrieved=3
        ).fit(X)
        with trace() as t:
            rec.recommend([0], n_recommendations=1)
        fused = {i for step in t[0].fusion for i in step.items.tolist()}
        served = final_of(t[0]).items.tolist()
        unseen = set(rec.item_ids_.tolist()) - set(X[X[:, 0] == 0, 1].tolist())
        found = by_item(explain(rec, [0], items=sorted(unseen), n_recommendations=1))
        for item in unseen - set(served):
            want = "ranked_out" if item in fused else "not_retrieved"
            assert found[item].status == want, item

    def test_autotune_is_explained_through_its_best_estimator(self):
        rec = AutoTune(
            ItemKNNRecommender(),
            search_space={"n_neighbors": Int(2, 4)},
            scoring=make_recommender_scorer(ndcg_at_k, k=2),
            n_trials=2,
            random_state=0,
        ).fit(X)
        (found,) = explain(rec, [0], n_recommendations=1)
        assert [r.path for r in found.reasons] == ["AutoTune/best_estimator/ItemKNNRecommender"]

    def test_explanations_are_json(self):
        rec = cascade(PointwiseRanker(LogisticRegression()), postprocess=drop_first).fit(X)
        for found in explain(rec, [0], items=[0, 9, 999], n_recommendations=2):
            record = json.loads(json.dumps(found.to_dict(), allow_nan=False))
            assert record["status"] == found.status
            assert str(found).startswith(f"query 0, item {found.item!r}")
