"""What :func:`skrecsys.inspection.trace` records of each stage of a pipeline."""

import json
import pickle
import zlib

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from skrecsys._tracing import (
    Attribution,
    Candidates,
    Features,
    RankerScores,
    Request,
    Route,
    Served,
    stable_unit_hash,
)
from skrecsys.compose import (
    AugmentedRanker,
    BlendRanker,
    Cascade,
    GeneratorScores,
    JoinStaticFeatures,
    KnownUser,
    PointwiseRanker,
    ReciprocalRankFusion,
    ReciprocalRankRanker,
    Switch,
)
from skrecsys.inspection import (
    CandidatesStep,
    FeaturesStep,
    PostprocessStep,
    QueryTrace,
    RankerStep,
    ServedStep,
    explain,
    trace,
)
from skrecsys.metrics import make_recommender_scorer, ndcg_at_k
from skrecsys.recommendation import ItemKNNRecommender, MostPopularRecommender
from skrecsys.tune import AutoTune, Int

#: SplitMix64 of 0, 1 and -1 over 2**64: 0xE220A8397B1DCDAF for 0 is the reference output.
PINNED_INT_HASHES = [0.8833108082136426, 0.5665615751722809, 0.8939429202831845]

X = np.array([[u, i] for u in range(30) for i in (u % 6, u % 6 + 1, (u + 2) % 6 + 2)])


def served(query_trace: QueryTrace) -> ServedStep:
    """What the outermost estimator served, which a call that returned always has."""
    final = query_trace.final
    assert final is not None
    return final


def drop_item_3(pairs, scores, groups):
    keep = pairs[:, 1] != 3
    group_of_row = np.repeat(np.arange(len(groups)), groups)
    return pairs[keep], scores[keep], np.bincount(group_of_row[keep], minlength=len(groups))


def pipeline(**params):
    cascade = Cascade(
        [ItemKNNRecommender(), MostPopularRecommender()],
        GeneratorScores(n_generators=2),
        PointwiseRanker(LogisticRegression()),
        n_retrieved=5,
        split=0.4,
        **params,
    )
    return Switch(KnownUser(), cascade, MostPopularRecommender()).fit(X)


def test_every_stage_of_a_pipeline_is_recorded():
    rec = pipeline(postprocess=drop_item_3)
    with trace() as t:
        items, scores = rec.recommend([0, 1000], n_recommendations=2)

    known = t[0]
    assert known.path == "Switch"
    assert [(r.path, r.condition, r.branch) for r in known.routes] == [
        ("Switch", "KnownUser()", "on_true")
    ]
    by_source = {step.source: step for step in known.candidates}
    assert set(by_source) == {"itemknnrecommender", "mostpopularrecommender", "union"}
    assert by_source["union"].scores.shape == (5, 2)
    assert {s.path for s in known.served} == {
        "Switch",
        "Switch/on_true/Cascade",
        "Switch/on_true/Cascade/itemknnrecommender/ItemKNNRecommender",
        "Switch/on_true/Cascade/mostpopularrecommender/MostPopularRecommender",
    }
    # The ranker scores and the features are of the merged candidates, in their order.
    (ranker,) = known.ranker
    (features,) = known.features
    np.testing.assert_array_equal(ranker.items, by_source["union"].items)
    np.testing.assert_array_equal(features.items, by_source["union"].items)
    assert features.names == ("generator_score_0", "generator_score_1")
    assert features.row(ranker.items[0]) is not None
    (post,) = known.postprocess
    assert 3 not in post.after_items.tolist()
    assert post.dropped == ([3] if 3 in post.before_items.tolist() else [])
    np.testing.assert_array_equal(post.before_items, ranker.ranked())
    final = served(known)
    np.testing.assert_array_equal(final.items, items[0])
    np.testing.assert_array_equal(final.scores, scores[0])

    cold = t[1000]
    assert [r.branch for r in cold.routes] == ["on_false"]
    assert not cold.candidates
    assert served(cold).items.tolist() == items[1].tolist()


def test_ranker_ties_go_by_fitted_item_order():
    items = np.array([30, 10, 20, 40])
    scores = np.array([1.0, 2.0, 1.0, 1.0])
    unknown = RankerStep("Cascade", items, scores)
    assert unknown.ranked().tolist() == [10, 30, 20, 40]
    known = RankerStep("Cascade", items, scores, positions=np.array([3, 1, 2, 0]))
    assert known.ranked().tolist() == [10, 40, 20, 30]


def test_decisions_level_leaves_out_the_features():
    with trace(level="decisions") as t:
        pipeline().recommend([0], n_recommendations=2)
    assert not t[0].features
    assert t[0].ranker


def test_fusion_contributions_add_up_to_the_served_scores():
    rec = ReciprocalRankFusion([ItemKNNRecommender(), MostPopularRecommender()]).fit(X)
    with trace() as t:
        items, scores = rec.recommend([3], n_recommendations=3)
    fused: dict[object, float] = {}
    for step in t[3].fusion:
        for item, contribution in zip(step.items.tolist(), step.contributions, strict=True):
            fused[item] = fused.get(item, 0.0) + contribution
    np.testing.assert_allclose([fused[i] for i in items[0].tolist()], scores[0])


def test_autotune_names_its_best_estimator():
    rec = AutoTune(
        ItemKNNRecommender(),
        search_space={"n_neighbors": Int(2, 4)},
        scoring=make_recommender_scorer(ndcg_at_k, k=2),
        n_trials=2,
        random_state=0,
    ).fit(X)
    with trace() as t:
        rec.recommend([0], n_recommendations=2)
    assert [s.path for s in t[0].served] == [
        "AutoTune/best_estimator/ItemKNNRecommender",
        "AutoTune",
    ]


def test_nothing_is_recorded_outside_recommend():
    rec = ReciprocalRankFusion([ItemKNNRecommender(), MostPopularRecommender()])
    with trace() as t:
        rec.fit(X)
        rec.predict([[0, 1]])
        pipeline().predict([[0, 1]])
    assert t.events == []


def test_each_outermost_call_is_numbered():
    rec = ItemKNNRecommender().fit(X)
    with trace() as t:
        rec.recommend([0], n_recommendations=1)
        rec.recommend([0, 1], n_recommendations=2)
    assert t.n_calls == 2
    assert t.queries == [0, 1]
    assert t[0].call == 1
    assert len(served(t.query(0, call=0)).items) == 1
    assert [(q.query, q.call) for q in t] == [(0, 0), (0, 1), (1, 1)]
    with pytest.raises(KeyError):
        t.query(1, call=0)


def test_numpy_queries_are_looked_up_and_shown_as_python_values():
    rec = ItemKNNRecommender().fit(X)
    queries = np.array([3, 4])
    with trace() as t:
        rec.recommend(queries, n_recommendations=1)
    found = t[queries[0]]
    assert type(found.query) is int
    assert str(found).startswith("query 3, call 0")


def test_a_raising_call_is_recorded_once_where_it_raised():
    rec = pipeline()
    with trace() as t, pytest.raises(ValueError, match="exceeds n_retrieved"):
        rec.recommend([0], n_recommendations=6)
    (raised,) = t[0].raised
    assert raised.path == "Switch/on_true/Cascade"
    assert raised.error == "ValueError"
    assert t[0].final is None


def test_traces_nest_and_restore_the_outer_one():
    rec = ItemKNNRecommender().fit(X)
    with trace() as outer:
        with trace() as inner:
            rec.recommend([0], n_recommendations=1)
        rec.recommend([1], n_recommendations=1)
    assert inner.queries == [0]
    assert outer.queries == [1]


def test_to_dict_is_json():
    with trace() as t:
        pipeline(postprocess=drop_item_3).recommend([0], n_recommendations=2)
    record = json.loads(json.dumps(t[0].to_dict(), allow_nan=False))
    kinds = [step["step"] for step in record["steps"]]
    assert kinds[0] == "request"
    assert kinds[-1] == "served"
    assert {"route", "candidates", "features", "ranker", "postprocess"} <= set(kinds)
    assert "ranker" in str(t[0])


def test_steps_are_of_the_expected_types():
    with trace() as t:
        pipeline(postprocess=drop_item_3).recommend([0], n_recommendations=2)
    steps = t[0]
    assert all(isinstance(s, CandidatesStep) for s in steps.candidates)
    assert all(isinstance(s, FeaturesStep) for s in steps.features)
    assert all(isinstance(s, RankerStep) for s in steps.ranker)
    assert all(isinstance(s, PostprocessStep) for s in steps.postprocess)
    assert all(isinstance(s, ServedStep) for s in steps.served)


class TestSampling:
    def test_a_rate_records_the_same_queries_every_time(self):
        rec = pipeline()
        queries = np.arange(30)
        with trace(sample=0.5) as first:
            rec.recommend(queries, n_recommendations=2)
        with trace(sample=0.5) as second:
            rec.recommend(queries[::-1], n_recommendations=2)
        assert 5 <= len(first.queries) <= 25
        assert set(first.queries) == set(second.queries)

    def test_a_sampled_query_is_recorded_through_every_stage(self):
        rec = pipeline()
        with trace(sample=lambda q: np.isin(q, np.arange(0, 30, 2))) as t:
            rec.recommend(np.arange(30), n_recommendations=2)
        assert t.queries == list(range(0, 30, 2))
        for event in t.events:
            if isinstance(event, Request | Served | Route):
                ids = event.queries
            else:
                assert isinstance(event, Attribution | Candidates | Features | RankerScores)
                ids = event.pairs[:, 0]
            assert np.isin(ids, np.arange(0, 30, 2)).all(), type(event).__name__

    def test_a_call_with_nothing_sampled_records_nothing(self):
        rec = pipeline()
        with trace(sample=0.0) as t:
            rec.recommend([0, 1], n_recommendations=2)
        assert t.events == []
        assert t.n_calls == 1

    def test_the_callable_must_return_a_mask(self):
        rec = ItemKNNRecommender().fit(X)
        with trace(sample=lambda q: [True]) as t, pytest.raises(ValueError, match="boolean"):
            rec.recommend([0, 1], n_recommendations=1)
        assert t._depth == 0

    @pytest.mark.parametrize(
        ("sample", "error"), [(True, TypeError), ("all", TypeError), (1.5, ValueError)]
    )
    def test_invalid_sample(self, sample, error):
        with pytest.raises(error):
            trace(sample=sample)

    def test_invalid_level(self):
        with pytest.raises(ValueError, match="level"):
            trace(level="everything")  # ty: ignore[invalid-argument-type]

    def test_the_hash_is_stable(self):
        # Pinned values: a change would resample every user of a running deployment.
        ints = stable_unit_hash(np.array([0, 1, -1]))
        strings = stable_unit_hash(np.array(["u1", 7], dtype=object))
        np.testing.assert_array_equal(ints, PINNED_INT_HASHES)
        np.testing.assert_array_equal(
            strings, [zlib.crc32(b"u1") / 2**32, zlib.crc32(b"7") / 2**32]
        )


def test_the_trace_is_not_part_of_the_estimator():
    rec = ItemKNNRecommender().fit(X)
    with trace():
        rec.recommend([0], n_recommendations=1)
        restored = pickle.loads(pickle.dumps(rec))
    assert not any("trace" in name.lower() for name in vars(restored))


# --- rankers with features of their own ------------------------------------------------

#: An item feature for every item of ``X``, which only an AugmentedRanker joins.
ITEM_FLAGS = np.column_stack([np.arange(8), np.arange(8) % 2]).astype(float)


def _augmented():
    return AugmentedRanker(
        PointwiseRanker(LogisticRegression()), JoinStaticFeatures("item", ITEM_FLAGS)
    )


def _cascade(ranker):
    return Cascade(
        MostPopularRecommender(), GeneratorScores(), ranker, n_retrieved=5, split=0.4
    ).fit(X)


def test_an_augmented_ranker_traces_the_features_it_joined():
    rec = _cascade(_augmented())
    with trace() as t:
        rec.recommend([0], n_recommendations=2)
    (features,) = t[0].features
    (ranker,) = t[0].ranker
    assert features.path == ranker.path == "Cascade"
    assert features.names == ("generator_score", "item_feature_0")
    np.testing.assert_array_equal(features.values[:, 1], features.items.astype(float) % 2)
    # The contributions are over every column the ranker saw, and add up to its log-odds.
    model = rec.ranker_.ranker_.estimator_
    assert ranker.contributions is not None
    assert ranker.contributions.shape == (len(ranker.items), 3)
    np.testing.assert_allclose(
        ranker.contributions.sum(axis=1), model.decision_function(features.values)
    )
    (why,) = explain(rec, [0], n_recommendations=1)
    assert why.ranker is not None
    assert why.ranker.contributions is not None
    assert "item_feature_0" in why.ranker.contributions


def test_a_blend_traces_each_ranker_with_the_features_it_saw():
    rec = _cascade(
        BlendRanker(
            [("plain", PointwiseRanker(LogisticRegression())), ("aug", _augmented())],
            random_state=0,
        )
    )
    with trace() as t:
        items, _ = rec.recommend([0], n_recommendations=2)
    rankers = {step.path: step for step in t[0].ranker}
    features = {step.path: step for step in t[0].features}
    assert (
        set(rankers) == set(features) == {"Cascade", "Cascade/ranker/plain", "Cascade/ranker/aug"}
    )
    assert features["Cascade"].names == ("generator_score",)
    assert features["Cascade/ranker/plain"].names == ("generator_score",)
    assert features["Cascade/ranker/aug"].names == ("generator_score", "item_feature_0")
    # Each ranker's own scores, before the blend normalizes them.
    fitted = dict(rec.ranker_.rankers_)
    np.testing.assert_allclose(
        rankers["Cascade/ranker/plain"].scores,
        fitted["plain"].predict(features["Cascade/ranker/plain"].values),
    )
    contributions = rankers["Cascade/ranker/aug"].contributions
    assert contributions is not None
    assert contributions.shape[1] == 3
    # What was served is the blend's order, which explain reads from the outermost ranker.
    np.testing.assert_array_equal(rankers["Cascade"].ranked()[:2], items[0])
    (why,) = explain(rec, [0], n_recommendations=1)
    assert why.ranker is not None
    assert why.ranker.position == 1


def test_a_decisions_trace_records_each_rankers_scores_only():
    rec = _cascade(ReciprocalRankRanker([("a", _augmented()), ("b", _augmented())]))
    with trace(level="decisions") as t:
        rec.recommend([0], n_recommendations=2)
    assert not t[0].features
    assert {step.path for step in t[0].ranker} == {
        "Cascade",
        "Cascade/ranker/a",
        "Cascade/ranker/b",
    }
    assert all(step.contributions is None for step in t[0].ranker)


def test_rankers_nested_in_an_augmented_one_see_its_columns():
    rec = _cascade(
        AugmentedRanker(
            BlendRanker([("inner", PointwiseRanker(LogisticRegression()))], blender=None),
            JoinStaticFeatures("item", ITEM_FLAGS),
        )
    )
    with trace() as t:
        rec.recommend([0], n_recommendations=2)
    features = {step.path: step.names for step in t[0].features}
    assert features == {
        "Cascade": ("generator_score", "item_feature_0"),
        "Cascade/ranker/inner": ("generator_score", "item_feature_0"),
    }


def test_an_injected_name_the_shared_features_have_is_prefixed():
    ranker = AugmentedRanker(PointwiseRanker(LogisticRegression()), GeneratorScores())
    rec = _cascade(ranker)
    with trace() as t:
        rec.recommend([0], n_recommendations=2)
    assert t[0].features[0].names == ("generator_score", "injected__generator_score")


def test_an_untraced_recommend_is_unchanged_by_member_tracing():
    rec = _cascade(BlendRanker([_augmented(), _augmented()], random_state=0))
    want = rec.recommend([0, 1], n_recommendations=2)
    with trace():
        got = rec.recommend([0, 1], n_recommendations=2)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_allclose(got[1], want[1])
