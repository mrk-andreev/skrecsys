"""Lists put together from several recommenders: a fixed list, a backfill, reserved slots."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier

from skrecsys._typing import override
from skrecsys.base import serves_unknown_users
from skrecsys.compose import (
    Backfill,
    Cascade,
    ConcatFeatures,
    GeneratorScores,
    InteractionCounts,
    ItemListRecommender,
    MinInteractions,
    PointwiseRanker,
    ProfileAffinity,
    ReservedSlots,
    Switch,
)
from skrecsys.exceptions import InsufficientDataError
from skrecsys.model_selection import LatestInteractionsSplit
from skrecsys.recommendation import BM25Recommender, ItemKNNRecommender, MostPopularRecommender
from tests.compose._composites import cascade
from tests.compose._data import N_ITEMS, N_USERS, trending_interactions

#: u1 saw a and b, u2 b and c, u3 c: popularity b=2, c=2, a=1.
X = np.array([["u1", "a"], ["u1", "b"], ["u2", "b"], ["u2", "c"], ["u3", "c"]])
#: A catalog in an order of its own, two of its items named by no interaction.
CATALOG = ["z", "c", "y", "b", "a"]
NO_ROWS = np.empty((0, 2), dtype=str)


def _lists(recommender, queries, n, **kwargs):
    return recommender.recommend(queries, n_recommendations=n, **kwargs)[0].tolist()


# --- ItemListRecommender


def test_item_list_serves_its_order_to_everyone_and_items_no_interaction_names():
    rec = ItemListRecommender(CATALOG).fit(X)
    assert rec.item_ids_.tolist() == ["a", "b", "c", "y", "z"]
    assert rec.rank_.tolist() == [4, 3, 1, 2, 0]
    assert rec.user_ids_.tolist() == ["u1", "u2", "u3"]
    assert serves_unknown_users(rec)
    items, scores = rec.recommend(["u1", "u2", "nobody"], n_recommendations=3)
    assert items.tolist() == [["z", "c", "y"], ["z", "y", "a"], ["z", "c", "y"]]
    # the scores are the places in the list, counted down from its length
    np.testing.assert_array_equal(scores, [[5, 4, 3], [5, 3, 1], [5, 4, 3]])
    np.testing.assert_array_equal(rec.predict([["u2", "y"], ["nobody", "a"]]), [3, 1])


def test_item_list_filters_like_any_recommender():
    rec = ItemListRecommender(CATALOG).fit(X)
    assert _lists(rec, ["u1"], 5, exclude_seen=False) == [CATALOG]
    assert _lists(rec, ["u1"], 2, candidates=["a", "y", "b", "c"]) == [["c", "y"]]
    assert _lists(rec, ["u1", "u3"], 2, exclude_interactions=[["u1", "z"]]) == [
        ["c", "y"],
        ["z", "y"],
    ]
    assert rec._count_eligible(["u1", "nobody"]).tolist() == [3, 5]
    with pytest.raises(ValueError, match="query 0 has only 3 eligible items"):
        rec.recommend(["u1"], n_recommendations=4)
    with pytest.raises(ValueError, match="Unknown item identifiers"):
        rec.predict([["u1", "not-listed"]])


def test_item_list_fits_on_no_interactions():
    """Before anyone has interacted there is still a catalog to show."""
    rec = ItemListRecommender(CATALOG).fit(NO_ROWS)
    assert (rec.n_users_, rec.n_items_) == (0, 5)
    assert _lists(rec, ["anyone"], 5) == [CATALOG]
    assert rec._count_eligible(["anyone"]).tolist() == [5]


def test_item_list_of_numeric_ids_and_dataframe_interactions():
    frame = pd.DataFrame({"user": [10, 10, 20], "item": [3, 99, 1]})
    rec = ItemListRecommender([3, 1, 2]).fit(frame)
    # item 99 is not in the list, so having interacted with it hides nothing
    assert rec.interactions_.nnz == 2
    assert _lists(rec, np.array([10, 20, 30]), 2) == [[1, 2], [3, 2], [3, 1]]


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"items": []}, "at least one item"),
        ({"items": ["a", "b", "a"]}, "duplicate"),
        ({"items": ["a"], "rotate": 1}, "rotate must be a bool"),
        ({"items": ["a"], "random_state": -1}, "random_state"),
    ],
)
def test_item_list_validates(params, match):
    with pytest.raises(ValueError, match=match):
        ItemListRecommender(**params).fit(X)


def test_rotation_reads_the_list_round_from_a_start_of_each_users_own():
    catalog = [f"i{k:02d}" for k in range(20)]
    users = np.array([f"user{k}" for k in range(400)], dtype=object)
    rec = ItemListRecommender(catalog, rotate=True).fit(X)
    items = _lists(rec, users, 20)
    starts = [catalog.index(row[0]) for row in items]
    for row, start in zip(items, starts, strict=True):
        assert row == catalog[start:] + catalog[:start]
    # every item leads someone's list, about as often as any other
    counts = np.bincount(starts, minlength=20)
    assert counts.min() >= 8
    assert counts.max() <= 40
    # nothing in the model moves: asking again, or after a refit, gives the same lists
    assert _lists(rec, users, 20) == items
    assert _lists(clone(rec).fit(X), users, 20) == items
    assert _lists(pickle.loads(pickle.dumps(rec)), users, 20) == items


def test_rotation_moves_with_random_state_and_applies_to_numeric_users():
    catalog = list(range(50))
    users = np.arange(200)
    first = _lists(ItemListRecommender(catalog, rotate=True).fit(NO_ROWS), users, 1)
    same = _lists(ItemListRecommender(catalog, rotate=True, random_state=0).fit(NO_ROWS), users, 1)
    other = _lists(ItemListRecommender(catalog, rotate=True, random_state=7).fit(NO_ROWS), users, 1)
    assert first == same
    moved = np.mean(np.ravel(first) != np.ravel(other))
    assert moved > 0.9
    assert len({row[0] for row in first}) > 40


def test_rotation_skips_what_a_user_has_seen_and_scores_as_it_ranks():
    rec = ItemListRecommender(CATALOG, rotate=True).fit(X)
    items, scores = rec.recommend(["u1", "u2"], n_recommendations=3)
    assert not {"a", "b"} & set(items[0])
    assert not {"b", "c"} & set(items[1])
    pairs = [[user, item] for user, row in zip(["u1", "u2"], items, strict=True) for item in row]
    np.testing.assert_array_equal(rec.predict(pairs), scores.ravel())


# --- Backfill


def _backfill(**params):
    return Backfill(
        [
            ("personal", ItemKNNRecommender()),
            ("popular", MostPopularRecommender()),
            ("catalog", ItemListRecommender(CATALOG)),
        ],
        **params,
    )


def test_backfill_tops_each_list_up_from_the_next_recommender():
    rec = _backfill().fit(X)
    assert rec.skipped_ == []
    assert rec.item_ids_.tolist() == ["a", "b", "c", "y", "z"]
    assert rec.user_ids_.tolist() == ["u1", "u2", "u3"]
    items, scores = rec.recommend(["u1", "u3", "nobody"], n_recommendations=3)
    assert items.tolist() == [
        # the one unseen neighbour, nothing new from popularity, then the catalog
        ["c", "z", "y"],
        # the neighbourhood model ranks both unseen items, and the catalog adds a third
        ["b", "a", "z"],
        # no personal model knows the user: the popular items
        ["b", "c", "a"],
    ]
    np.testing.assert_array_equal(scores, np.tile([3.0, 2.0, 1.0], (3, 1)))
    assert _lists(rec, ["u3", "nobody"], 4) == [["b", "a", "z", "y"], ["b", "c", "a", "z"]]


def test_backfill_fills_only_what_is_missing_and_never_repeats_an_item():
    rec = _backfill().fit(X)
    # u1 has seen a and b: the neighbourhood model offers c alone
    assert _lists(rec, ["u1"], 1) == [["c"]]
    assert _lists(rec, ["u1"], 3) == [["c", "z", "y"]]
    with pytest.raises(ValueError, match="query 0 has only 3 eligible items"):
        rec.recommend(["u1"], n_recommendations=4)
    # with nothing hidden, the personal list is long enough and nobody else is asked
    full = _lists(rec, ["u1", "u2", "nobody"], 5, exclude_seen=False)
    assert all(sorted(row) == ["a", "b", "c", "y", "z"] for row in full)
    assert full[2] == ["b", "c", "a", "z", "y"]


def test_backfill_passes_filters_to_every_recommender():
    rec = _backfill().fit(X)
    # "c" is the only personal item u1 has; excluded with "z", the catalog starts at "y".
    # The pairs are u1's alone: u3 is served as without them.
    recent = [["u1", "c"], ["u1", "z"]]
    assert _lists(rec, ["u1", "u3"], 1, exclude_interactions=recent) == [["y"], ["b"]]
    with pytest.raises(ValueError, match="query 0 has only 1 eligible items"):
        rec.recommend(["u1", "u3"], n_recommendations=2, exclude_interactions=recent)
    # a candidate only the catalog knows is given to the catalog alone
    assert _lists(rec, ["u3"], 3, candidates=["y", "a", "z"]) == [["a", "z", "y"]]
    assert _lists(rec, ["nobody"], 2, candidates=["y", "z"]) == [["z", "y"]]
    with pytest.raises(ValueError, match="Unknown item identifiers"):
        rec.recommend(["u1"], n_recommendations=1, candidates=["not-listed"])
    assert rec._count_eligible(["u1", "nobody"]).tolist() == [3, 5]
    assert rec._count_eligible(["u1"], candidates=["a", "c", "y"]).tolist() == [2]


class _Stubborn(MostPopularRecommender):
    """Serves what it was told to leave out, as business rules adding items of their own may."""

    @override
    def recommend(self, X, *, exclude_interactions=None, **kwargs):
        return super().recommend(X, **kwargs)

    @override
    def _count_eligible(self, X, *, exclude_interactions=None, **kwargs):
        return super()._count_eligible(X, **kwargs)


def test_backfill_drops_what_a_recommender_repeats():
    rec = Backfill([MostPopularRecommender(), _Stubborn(), ItemListRecommender(CATALOG)]).fit(X)
    assert _lists(rec, ["nobody"], 5) == [["b", "c", "a", "z", "y"]]


def test_backfill_skips_a_recommender_with_too_little_data_only_when_told_to():
    # one interaction per user leaves a cascade nothing to hold out for its ranker
    few = X[[0, 2]]
    sources = [("ranked", cascade()), ("catalog", ItemListRecommender(CATALOG))]
    with pytest.raises(InsufficientDataError, match="to hold out"):
        Backfill(sources).fit(few)
    rec = Backfill(sources, skip_insufficient=True).fit(few)
    assert rec.skipped_ == ["ranked"]
    assert [name for name, _ in rec.recommenders_] == ["catalog"]
    assert _lists(rec, ["u1"], 2) == [["z", "c"]]
    # the failure of a part inside another composite is skipped with that composite
    nested = [
        ("routed", Switch(MinInteractions(1), cascade(), MostPopularRecommender())),
        sources[1],
    ]
    assert Backfill(nested, skip_insufficient=True).fit(few).skipped_ == ["routed"]
    assert Backfill(sources, skip_insufficient=True).fit(X).skipped_ == []
    # any other error is a mistake in the arguments, and is raised
    wrong = [("popular", MostPopularRecommender(weighting="bogus")), sources[1]]
    with pytest.raises(ValueError, match="weighting"):
        Backfill(wrong, skip_insufficient=True).fit(few)


def test_backfill_before_the_first_interaction():
    with pytest.raises(InsufficientDataError, match="0 row"):
        _backfill().fit(NO_ROWS)
    rec = _backfill(skip_insufficient=True).fit(NO_ROWS)
    assert rec.skipped_ == ["personal", "popular"]
    assert _lists(rec, ["anyone"], 5) == [CATALOG]
    with pytest.raises(InsufficientDataError, match="No recommender of the Backfill"):
        Backfill([ItemKNNRecommender()], skip_insufficient=True).fit(NO_ROWS)


def test_backfill_predict_asks_the_first_recommender_that_knows_the_pair():
    rec = _backfill().fit(X)
    knn, _, catalog = (part for _, part in rec.recommenders_)
    scores = rec.predict([["u1", "c"], ["u1", "z"], ["u2", "a"]])
    np.testing.assert_allclose(scores[[0, 2]], knn.predict([["u1", "c"], ["u2", "a"]]))
    assert scores[1] == catalog.predict([["u1", "z"]])[0]
    with pytest.raises(ValueError, match="Unknown item identifiers"):
        rec.predict([["u1", "not-listed"]])
    with pytest.raises(ValueError, match="No recommender knows both"):
        Backfill([ItemKNNRecommender(), MostPopularRecommender()]).fit(X).predict([["new", "a"]])


def test_backfill_parameters_nest_by_name():
    rec = _backfill()
    assert rec.get_params()["popular__weighting"] == "count"
    rec.set_params(popular__weighting="sum", catalog__rotate=True)
    assert rec.recommenders[1][1].weighting == "sum"
    assert rec.recommenders[2][1].rotate is True
    unnamed = Backfill([MostPopularRecommender(), ItemListRecommender(CATALOG)])
    assert "itemlistrecommender__items" in unnamed.get_params()
    assert serves_unknown_users(unnamed)
    assert not serves_unknown_users(Backfill([ItemKNNRecommender()]))


@pytest.mark.parametrize(
    ("recommenders", "error", "match"),
    [
        ([], ValueError, "at least one recommender"),
        ([MostPopularRecommender(), "popular"], TypeError, "not a recommender"),
    ],
)
def test_backfill_validates(recommenders, error, match):
    with pytest.raises(error, match=match):
        Backfill(recommenders).fit(X)


# --- ReservedSlots


def _slots(base, inserted, **params):
    return ReservedSlots(ItemListRecommender(base), ItemListRecommender(inserted), **params)


BASE = ["b1", "b2", "b3", "b4", "b5", "b6"]


def test_reserved_slots_are_the_last_of_the_head():
    rec = _slots(BASE, ["e1", "e2", "e3"], n_slots=2, head=4).fit(NO_ROWS)
    items, scores = rec.recommend(["anyone"], n_recommendations=6)
    assert items.tolist() == [["b1", "b2", "e1", "e2", "b3", "b4"]]
    np.testing.assert_array_equal(scores, [[6, 5, 4, 3, 2, 1]])
    assert rec.item_ids_.tolist() == sorted([*BASE, "e1", "e2", "e3"])


def test_reserved_slots_stay_where_they_are_in_a_shorter_list():
    rec = _slots(BASE, ["e1", "e2", "e3"], n_slots=2, head=4).fit(NO_ROWS)
    assert _lists(rec, ["anyone"], 2) == [["b1", "b2"]]
    assert _lists(rec, ["anyone"], 3) == [["b1", "b2", "e1"]]
    assert _lists(rec, ["anyone"], 4) == [["b1", "b2", "e1", "e2"]]


def test_slots_without_an_item_stay_with_base():
    rec = _slots(BASE, ["e1"], n_slots=2, head=4).fit(NO_ROWS)
    assert _lists(rec, ["anyone"], 6) == [["b1", "b2", "e1", "b3", "b4", "b5"]]


def test_head_positions_base_cannot_fill_go_to_the_inserted_items():
    rec = _slots(["b1"], ["e1", "e2", "e3", "e4"], n_slots=1, head=3).fit(NO_ROWS)
    assert _lists(rec, ["anyone"], 3) == [["b1", "e1", "e2"]]
    # past the head nothing is inserted, and base has no more
    with pytest.raises(ValueError, match="query 0 has only 3 eligible items"):
        rec.recommend(["anyone"], n_recommendations=4)


def test_an_item_of_both_parts_is_listed_once():
    rec = _slots(["b1", "b2", "x", "b3"], ["b1", "x", "e1"], n_slots=1, head=3).fit(NO_ROWS)
    # b1 keeps its place; x takes the slot and leaves base's tail
    assert _lists(rec, ["anyone"], 4) == [["b1", "b2", "x", "b3"]]


def test_no_slots_inserts_only_where_base_has_nothing():
    rec = _slots(["b1"], ["e1", "e2"], n_slots=0, head=2).fit(NO_ROWS)
    assert _lists(rec, ["anyone"], 2) == [["b1", "e1"]]
    full = _slots(BASE, ["e1", "e2"], n_slots=0, head=2).fit(NO_ROWS)
    assert _lists(full, ["anyone"], 3) == [["b1", "b2", "b3"]]


def test_reserved_slots_filter_both_parts_per_query():
    seen = np.array([["u1", "b1"], ["u1", "e1"], ["u2", "b2"]])
    rec = _slots(BASE, ["e1", "e2", "e3"], n_slots=1, head=3).fit(seen)
    assert _lists(rec, ["u1", "u2", "new"], 4) == [
        ["b2", "b3", "e2", "b4"],
        ["b1", "b3", "e1", "b4"],
        ["b1", "b2", "e1", "b3"],
    ]
    assert _lists(rec, ["new"], 4, exclude_interactions=[["new", "e1"], ["new", "b1"]]) == [
        ["b2", "b3", "e2", "b4"]
    ]
    assert _lists(rec, ["new"], 3, candidates=["b6", "e3", "b5", "b4"]) == [["b4", "b5", "e3"]]
    assert rec._count_eligible(["u1", "new"]).tolist() == [5, 6]
    np.testing.assert_array_equal(rec.predict([["u1", "b2"]]), [5.0])


def test_reserved_slots_parameters_nest_and_are_validated():
    rec = ReservedSlots(MostPopularRecommender(), ItemListRecommender(["e1"]))
    assert rec.get_params()["inserted__rotate"] is False
    assert rec.set_params(base__weighting="sum").base.weighting == "sum"
    assert serves_unknown_users(rec)
    for params, error, match in (
        ({"n_slots": 11}, ValueError, "n_slots"),
        ({"head": 0}, ValueError, "head"),
        ({"inserted": "explore"}, TypeError, "inserted must be a recommender"),
    ):
        with pytest.raises(error, match=match):
            clone(rec).set_params(**params).fit(X)


# --- The serving model the blocks are for


def test_a_serving_model_is_composed_from_blocks():
    """Personal lists where there is history, always full, with room to explore.

    A ranked cascade for users with history, a rate-based popular list for the others,
    the whole catalog behind both, and three positions of the head for items nobody has
    been shown.
    """
    lone_user = N_USERS + 1
    likes = np.vstack([trending_interactions(), [[lone_user, 0]]])
    catalog = np.arange(N_ITEMS + 10)
    unexplored = catalog[N_ITEMS:]
    genres = np.column_stack([catalog, catalog % 2, catalog % 3 == 0]).astype(float)
    dislikes = np.array([[0, 3], [1, 3], [1, 4]])
    impressions = np.column_stack([np.arange(N_ITEMS), np.full(N_ITEMS, 30.0)])

    ranked = Cascade(
        generator=[("bm25", BM25Recommender()), ("popular", MostPopularRecommender())],
        features=ConcatFeatures(
            [
                ("gen", GeneratorScores(n_generators=2)),
                ("item_likes", InteractionCounts("item")),
                ("item_dislikes", InteractionCounts("item", interactions=dislikes)),
                ("genre", ProfileAffinity(genres)),
                ("genre_disliked", ProfileAffinity(genres, interactions=dislikes)),
            ]
        ),
        ranker=PointwiseRanker(HistGradientBoostingClassifier(max_iter=5, random_state=0)),
        n_retrieved=15,
        split=LatestInteractionsSplit(0.5, max_users=40),
    )
    popular = MostPopularRecommender(exposure=impressions, smoothing=20.0)
    model = ReservedSlots(
        Backfill(
            [
                ("personal", Switch(MinInteractions(2), ranked, popular)),
                ("popular", popular),
                ("catalog", ItemListRecommender(catalog)),
            ],
            skip_insufficient=True,
        ),
        ItemListRecommender(unexplored, rotate=True, random_state=1),
        n_slots=3,
        head=10,
    ).fit(likes)

    filled = model.base_
    assert isinstance(filled, Backfill)
    assert filled.skipped_ == []
    assert [name for name, _ in filled.recommenders_] == ["personal", "popular", "catalog"]
    # the ranker learned from the latest users only
    switch = dict(filled.recommenders_)["personal"]
    assert isinstance(switch, Switch)
    fitted_ranked = switch.on_true_
    assert isinstance(fitted_ranked, Cascade)
    assert 0 < fitted_ranked.n_ranker_groups_ <= 40
    assert model.item_ids_.tolist() == catalog.tolist()

    n = 30  # more than the cascade retrieves, so every source is needed
    queries = np.array([0, lone_user, 10_000])
    items, _ = model.recommend(queries, n_recommendations=n)
    assert items.shape == (3, n)
    for user, row in zip(queries, items, strict=True):
        assert len(set(row)) == n
        assert not set(row) & set(likes[likes[:, 0] == user, 1])
        assert set(row[7:10]) <= set(unexplored)
        assert not set(row[:7]) & set(unexplored)

    recent = np.array([[0, items[0, 0]], [0, items[0, 8]]])
    fresh, _ = model.recommend(queries[:1], n_recommendations=n, exclude_interactions=recent)
    assert not set(fresh[0]) & set(recent[:, 1])
    assert len(set(fresh[0])) == n

    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.recommend(queries, n_recommendations=n)[0], items)

    # the same blocks before anyone has liked anything: the catalog, explored
    cold_start = clone(model).fit(np.empty((0, 2), dtype=np.int64))
    assert cold_start.base_.skipped_ == ["personal", "popular"]
    first, _ = cold_start.recommend(np.array([0]), n_recommendations=10)
    assert first[0, :7].tolist() == catalog[:7].tolist()
    assert set(first[0, 7:]) <= set(unexplored)
