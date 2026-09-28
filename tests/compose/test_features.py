import datetime
import pickle

import numpy as np
import pytest
from sklearn.base import clone

from skrecsys.base import is_features
from skrecsys.compose import (
    ConcatFeatures,
    GeneratorScores,
    InteractionCounts,
    JoinDynamicFeatures,
    JoinStaticFeatures,
    RecommenderScores,
    SegmentPopularity,
)
from skrecsys.recommendation import ItemKNNRecommender, MostPopularRecommender

PAIRS = np.array([["u1", "a"], ["u2", "b"], ["u9", "a"]], dtype=object)
USERS = np.array([["u1", 1.0, 10.0], ["u2", 2.0, 20.0]], dtype=object)


def user_lengths(ids):
    """Module level, so that a component holding it pickles."""
    return np.array([[len(i)] for i in ids], dtype=float)


def test_static_join_fills_unknown_ids_with_nan():
    out = JoinStaticFeatures("user", USERS).fit().transform(PAIRS)
    np.testing.assert_array_equal(out[:2], [[1.0, 10.0], [2.0, 20.0]])
    assert np.isnan(out[2]).all()


def test_static_join_can_refuse_unknown_ids():
    join = JoinStaticFeatures("user", USERS, missing="error").fit()
    with pytest.raises(ValueError, match=r"No features for user identifiers: \['u9'\]"):
        join.transform(PAIRS)


def test_static_join_reads_items_and_numeric_ids():
    table = np.array([[3, 0.5], [1, 0.25]])
    out = JoinStaticFeatures("item", table).fit().transform(np.array([[7, 1], [7, 3]]))
    np.testing.assert_array_equal(out, [[0.25], [0.5]])


def test_static_join_names_features():
    join = JoinStaticFeatures("user", USERS).fit()
    assert join.get_feature_names_out().tolist() == ["user_feature_0", "user_feature_1"]


def test_static_join_keeps_dataframe_column_names():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"user": ["u1", "u2"], "age": [30, 40], "score": [0.5, 0.25]})
    join = JoinStaticFeatures("user", frame).fit()
    assert join.get_feature_names_out().tolist() == ["age", "score"]
    np.testing.assert_array_equal(join.transform(PAIRS[:2]), [[30.0, 0.5], [40.0, 0.25]])


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"kind": "session", "values": USERS}, "kind"),
        ({"values": USERS, "missing": "zero"}, "missing"),
        ({"values": None}, "values"),
        ({"values": USERS[:, :1]}, "at least one feature"),
        ({"values": np.array([["u1", 1.0], ["u1", 2.0]], dtype=object)}, "duplicate"),
    ],
)
def test_static_join_validates(params, match):
    with pytest.raises(ValueError, match=match):
        JoinStaticFeatures(**params).fit()


def pair_lengths(rows):
    """Module level, so that a component holding it pickles."""
    return np.array([[sum(len(str(i)) for i in row)] for row in rows], dtype=float)


DYNAMIC_PAIRS = np.array([["ann", "a"], ["bo", "bb"], ["ann", "a"], ["ann", "bb"]], dtype=object)


@pytest.mark.parametrize(
    ("kind", "callback", "expected_call", "expected_out"),
    [
        ("user", user_lengths, ["ann", "bo"], [3, 2, 3, 3]),
        ("item", user_lengths, ["a", "bb"], [1, 2, 1, 2]),
        (
            ("user", "item"),
            pair_lengths,
            [["ann", "a"], ["ann", "bb"], ["bo", "bb"]],
            [4, 4, 4, 5],
        ),
        (("item", "user"), pair_lengths, [["a", "ann"], ["bb", "ann"], ["bb", "bo"]], [4, 4, 4, 5]),
        (("user",), pair_lengths, [["ann"], ["bo"]], [3, 2, 3, 3]),
    ],
    ids=str,
)
def test_dynamic_join_calls_back_once_with_distinct_ids(
    kind, callback, expected_call, expected_out
):
    calls = []

    def recording(ids):
        calls.append(ids.tolist())
        return callback(ids)

    out = JoinDynamicFeatures(kind, recording, n_features=1).fit().transform(DYNAMIC_PAIRS)
    np.testing.assert_array_equal(out, np.array(expected_out, dtype=float)[:, None])
    assert calls == [expected_call]


def test_dynamic_join_on_pairs_of_numeric_ids():
    pairs = np.array([[7, 1], [3, 1], [7, 1], [7, 2]])
    calls = []

    def callback(rows):
        calls.append(rows.tolist())
        return rows[:, 0] * 10 + rows[:, 1]

    out = JoinDynamicFeatures(("user", "item"), callback).fit().transform(pairs)
    np.testing.assert_array_equal(out, [[71.0], [31.0], [71.0], [72.0]])
    assert calls == [[[3, 1], [7, 1], [7, 2]]]


def test_dynamic_join_on_no_pairs():
    empty = np.empty((0, 2), dtype=object)
    out = JoinDynamicFeatures(("user", "item"), pair_lengths, n_features=1).fit().transform(empty)
    assert out.shape == (0, 1)


@pytest.mark.parametrize(
    ("kind", "names"),
    [
        ("user", ["user_feature_0", "user_feature_1"]),
        ("item", ["item_feature_0", "item_feature_1"]),
        (("user", "item"), ["user_item_feature_0", "user_item_feature_1"]),
    ],
    ids=str,
)
def test_dynamic_join_names_features(kind, names):
    join = JoinDynamicFeatures(kind, user_lengths, n_features=2).fit()
    assert join.get_feature_names_out().tolist() == names


@pytest.mark.parametrize("kind", ["session", (), ("user", "user"), ("user", "session"), 1])
def test_dynamic_join_validates_kind(kind):
    with pytest.raises(ValueError, match="kind must be"):
        JoinDynamicFeatures(kind, user_lengths).fit()


def test_dynamic_join_checks_what_the_callback_returns():
    with pytest.raises(ValueError, match="shape"):
        JoinDynamicFeatures("user", lambda ids: np.zeros((1, 2))).fit().transform(PAIRS)
    with pytest.raises(ValueError, match="expected 3"):
        JoinDynamicFeatures("user", user_lengths, n_features=3).fit().transform(PAIRS)
    with pytest.raises(TypeError, match="callable"):
        JoinDynamicFeatures("user", None).fit()


@pytest.mark.parametrize(
    ("kind", "callback"),
    [("user", user_lengths), ("item", user_lengths), (("user", "item"), pair_lengths)],
    ids=str,
)
def test_dynamic_join_with_a_module_level_callback_pickles_and_clones(kind, callback):
    join = JoinDynamicFeatures(kind, callback, n_features=1).fit()
    restored = pickle.loads(pickle.dumps(join))
    np.testing.assert_array_equal(restored.transform(PAIRS), join.transform(PAIRS))
    assert clone(join).kind == kind


def test_generator_scores():
    out = GeneratorScores().fit().transform(PAIRS, scores=[3.0, 2.0, 1.0])
    np.testing.assert_array_equal(out, [[3.0], [2.0], [1.0]])
    with pytest.raises(ValueError, match="generator scores"):
        GeneratorScores().transform(PAIRS)
    with pytest.raises(ValueError, match="shape"):
        GeneratorScores().transform(PAIRS, scores=[1.0])


def test_concat_stacks_columns_and_names_them():
    concat = ConcatFeatures([JoinStaticFeatures("user", USERS), GeneratorScores()]).fit()
    out = concat.transform(PAIRS, scores=[3.0, 2.0, 1.0])
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out[:, 2], [3.0, 2.0, 1.0])
    assert concat.get_feature_names_out().tolist() == [
        "joinstaticfeatures__user_feature_0",
        "joinstaticfeatures__user_feature_1",
        "generatorscores__generator_score",
    ]


def test_concat_fits_clones_and_leaves_its_parameters_alone():
    part = JoinStaticFeatures("user", USERS)
    concat = ConcatFeatures([part]).fit()
    assert not hasattr(part, "table_")
    assert concat.features_[0][1] is not part


def test_concat_nested_params_with_automatic_names():
    concat = ConcatFeatures([JoinStaticFeatures("user", USERS), JoinStaticFeatures("item")])
    params = concat.get_params()
    assert params["joinstaticfeatures-1__kind"] == "user"
    assert params["joinstaticfeatures-2__kind"] == "item"
    concat.set_params(**{"joinstaticfeatures-2__missing": "error"})
    assert concat.get_params()["joinstaticfeatures-2__missing"] == "error"
    copy = clone(concat)
    assert copy.get_params()["joinstaticfeatures-2__missing"] == "error"
    assert copy.get_params()["joinstaticfeatures-2"] is not concat.features[1]


def test_concat_nested_params_with_explicit_names():
    concat = ConcatFeatures([("users", JoinStaticFeatures("user", USERS))])
    concat.set_params(users__missing="error")
    users = concat.get_params()["users"]
    assert isinstance(users, JoinStaticFeatures)
    assert users.missing == "error"
    concat.set_params(users=GeneratorScores())
    assert isinstance(concat.get_params()["users"], GeneratorScores)
    with pytest.raises(ValueError, match="Invalid parameter"):
        concat.set_params(nobody__kind="item")


def test_concat_validates():
    with pytest.raises(ValueError, match="at least one"):
        ConcatFeatures([]).fit()
    with pytest.raises(ValueError, match="all components or all"):
        ConcatFeatures([("a", GeneratorScores()), GeneratorScores()]).fit()  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="unique"):
        ConcatFeatures([("a", GeneratorScores()), ("a", GeneratorScores())]).fit()
    with pytest.raises(TypeError, match="not a feature component"):
        ConcatFeatures(["nope"]).fit()  # ty: ignore[invalid-argument-type]


def test_tags():
    for component in (
        JoinStaticFeatures(),
        JoinDynamicFeatures(),
        GeneratorScores(),
        InteractionCounts(),
        RecommenderScores(),
        SegmentPopularity(),
    ):
        assert is_features(component)
    assert is_features(ConcatFeatures([GeneratorScores()]))


def test_pairs_are_validated():
    with pytest.raises(ValueError, match="2 columns"):
        GeneratorScores().transform(np.zeros((2, 4)), scores=[1.0, 2.0])


X = np.array([["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "c"], ["u3", "a"], ["u1", "a"]])


def test_interaction_counts_count_rows_and_give_unseen_ids_zero():
    items = InteractionCounts("item").fit(X)
    np.testing.assert_array_equal(
        items.transform([["u9", "a"], ["u9", "c"], ["u9", "z"]]), [[4.0], [1.0], [0.0]]
    )
    users = InteractionCounts("user").fit(X)
    np.testing.assert_array_equal(users.transform([["u1", "z"], ["new", "a"]]), [[3.0], [0.0]])
    assert users.get_feature_names_out().tolist() == ["user_interactions"]


def test_interaction_counts_validate_kind():
    with pytest.raises(ValueError, match="kind"):
        InteractionCounts("both").fit(X)


def test_recommender_scores_match_predict_and_are_nan_where_it_cannot_score():
    feature = RecommenderScores(ItemKNNRecommender()).fit(X)
    model = ItemKNNRecommender().fit(X)
    out = feature.transform([["u2", "b"], ["new", "b"], ["u2", "z"]])
    assert out.shape == (3, 1)
    np.testing.assert_allclose(out[0], model.predict([["u2", "b"]]))
    assert np.isnan(out[1:]).all()
    assert feature.get_feature_names_out().tolist() == ["recommender_score"]


def test_recommender_scores_fit_a_clone():
    knn = ItemKNNRecommender()
    feature = RecommenderScores(knn).fit(X)
    assert feature.recommender_ is not knn
    assert not hasattr(knn, "item_ids_")


def test_recommender_scores_need_a_recommender():
    with pytest.raises(TypeError, match="recommender"):
        RecommenderScores(GeneratorScores()).fit(X)  # ty: ignore[invalid-argument-type]


SEGMENTS = np.array(
    [["u1", "kid", 1], ["u2", "kid", 2], ["u3", "adult", 1], ["new", "kid", 2]], dtype=object
)


def test_segment_popularity_is_the_share_of_the_segment_and_its_lift():
    feature = SegmentPopularity(SEGMENTS, smoothing=0.0).fit(X)
    out = feature.transform([["new", "b"], ["u3", "a"], ["u3", "b"]])
    # b: one kid of two, one user of three overall. a: every user.
    np.testing.assert_allclose(out[0, :2], [0.5, 1.5])
    np.testing.assert_allclose(out[1, :2], [1.0, 1.0])
    np.testing.assert_allclose(out[2, :2], [0.0, 0.0])
    # The second segmentation: u1 and u3 in group 1, u2 and the new user in group 2.
    np.testing.assert_allclose(out[0, 2:], [0.0, 0.0])
    assert feature.get_feature_names_out().tolist() == [
        "segment_0_share",
        "segment_0_lift",
        "segment_1_share",
        "segment_1_lift",
    ]


def test_segment_popularity_counts_users_not_rows():
    """u1 interacted with a twice, which is one user, not two."""
    feature = SegmentPopularity(SEGMENTS[:, :2], smoothing=0.0).fit(X)
    np.testing.assert_allclose(feature.transform([["u1", "a"]])[0, 0], 1.0)


def test_segment_popularity_smooths_towards_the_global_share():
    raw = SegmentPopularity(SEGMENTS[:, :2], smoothing=0.0).fit(X)
    smooth = SegmentPopularity(SEGMENTS[:, :2], smoothing=1000.0).fit(X)
    pair = [["new", "b"]]
    assert raw.transform(pair)[0, 0] == pytest.approx(0.5)
    assert smooth.transform(pair)[0, 0] == pytest.approx(1 / 3, abs=1e-3)


def test_segment_popularity_is_nan_for_users_without_a_segment_and_unseen_items():
    feature = SegmentPopularity(SEGMENTS[:, :2]).fit(X)
    out = feature.transform([["nobody", "a"], ["u1", "z"]])
    assert np.isnan(out).all()


def test_segment_popularity_keeps_dataframe_column_names():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"user": ["u1", "u2", "u3"], "age_band": ["kid", "kid", "adult"]})
    feature = SegmentPopularity(frame).fit(X)
    assert feature.get_feature_names_out().tolist() == ["age_band_share", "age_band_lift"]


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"segments": None}, "segments must be"),
        ({"segments": np.array([["u1"], ["u2"]])}, "at least one segment column"),
        ({"segments": np.array([["u1", "a"], ["u1", "b"]])}, "duplicate"),
        ({"segments": SEGMENTS, "smoothing": -1.0}, "smoothing"),
    ],
)
def test_segment_popularity_validates(params, match):
    with pytest.raises(ValueError, match=match):
        SegmentPopularity(**params).fit(X)


@pytest.mark.parametrize(
    "component",
    [
        InteractionCounts("user"),
        RecommenderScores(MostPopularRecommender()),
        SegmentPopularity(SEGMENTS),
    ],
    ids=lambda c: type(c).__name__,
)
def test_fitted_components_pickle_and_clone(component):
    fitted = clone(component).fit(X)
    restored = pickle.loads(pickle.dumps(fitted))
    pairs = [["u1", "c"], ["u2", "b"]]
    np.testing.assert_array_equal(restored.transform(pairs), fitted.transform(pairs))


@pytest.mark.parametrize(
    "component",
    [InteractionCounts(), RecommenderScores(MostPopularRecommender()), SegmentPopularity(SEGMENTS)],
    ids=lambda c: type(c).__name__,
)
def test_components_that_learn_need_interactions(component):
    with pytest.raises(ValueError, match="learns from interactions"):
        component.fit()


#: Pairs as a timed recommender hands them to its features: the time each is ranked as
#: of, NaN where the latest features are wanted.
TIMED_PAIRS = np.array([[1, 7, 3.0], [1, 7, 9.0], [2, 7, 3.0], [2, 8, np.nan]])


def module_level_time_callback(keys):
    """Module level, so that a component holding it pickles."""
    times = keys if keys.ndim == 1 else keys[:, -1]
    return np.nan_to_num(times.astype(float), nan=-1.0)


@pytest.mark.parametrize(
    ("kind", "expected_call"),
    [
        ("time", [3.0, 9.0, np.nan]),
        ("item-time", [[7, 3.0], [7, 9.0], [8, np.nan]]),
        (("user", "item", "time"), [[1, 7, 3.0], [1, 7, 9.0], [2, 7, 3.0], [2, 8, np.nan]]),
        ("time-user", [[3.0, 1], [3.0, 2], [9.0, 1], [np.nan, 2]]),
    ],
    ids=str,
)
def test_dynamic_join_keyed_by_time_calls_back_with_distinct_keys(kind, expected_call):
    calls = []

    def recording(keys):
        calls.append(np.asarray(keys, dtype=float))
        return np.arange(len(keys), dtype=float)

    out = JoinDynamicFeatures(kind, recording).fit().transform(TIMED_PAIRS)
    (call,) = calls
    np.testing.assert_array_equal(call, np.asarray(expected_call, dtype=float))
    # Each pair gets the row its key was answered with.
    keys = np.asarray(expected_call, dtype=float)
    parts = kind.split("-") if isinstance(kind, str) else kind
    columns = [("user", "item", "time").index(p) for p in parts]
    pair_keys = TIMED_PAIRS[:, columns].reshape(len(TIMED_PAIRS), -1)
    for row, key in zip(out.ravel(), pair_keys, strict=True):
        np.testing.assert_array_equal(keys.reshape(len(keys), -1)[int(row)], key)


def test_dynamic_join_passes_datetimes_beside_string_ids_as_datetime64_scalars():
    pairs = np.empty((3, 3), dtype=object)
    pairs[:, 0], pairs[:, 1] = ["u1", "u1", "u1"], ["a", "b", "c"]
    pairs[:, 2] = [
        datetime.datetime(2024, 1, 2),
        None,
        np.datetime64("2024-01-02T00:00:00.000000001"),
    ]
    calls = []

    def recording(keys):
        calls.append(keys.tolist())
        return np.zeros(len(keys))

    JoinDynamicFeatures("item-time", recording).fit().transform(pairs)
    [(a, a_time), (b, b_time), (c, c_time)] = calls[0]
    assert (a, b, c) == ("a", "b", "c")
    assert a_time == np.datetime64("2024-01-02", "ns")
    assert np.isnat(b_time)
    # The nanoseconds are kept.
    assert c_time == np.datetime64("2024-01-02T00:00:00.000000001")


def test_dynamic_join_keyed_by_time_needs_pairs_with_time():
    join = JoinDynamicFeatures("item-time", module_level_time_callback).fit()
    with pytest.raises(ValueError, match="pairs carry no time"):
        join.transform(PAIRS)


@pytest.mark.parametrize(
    ("kind", "names"),
    [
        ("item-time", ["item_time_feature_0"]),
        (("user", "item", "time"), ["user_item_time_feature_0"]),
    ],
    ids=str,
)
def test_dynamic_join_keyed_by_time_names_features(kind, names):
    join = JoinDynamicFeatures(kind, module_level_time_callback, n_features=1).fit()
    assert join.get_feature_names_out().tolist() == names


@pytest.mark.parametrize("kind", ["user-user", "item-session", "-", "user-"])
def test_dynamic_join_validates_joined_kinds(kind):
    with pytest.raises(ValueError, match="kind must be"):
        JoinDynamicFeatures(kind, module_level_time_callback).fit()


def test_dynamic_join_keyed_by_time_pickles():
    join = JoinDynamicFeatures("item-time", module_level_time_callback, n_features=1).fit()
    restored = pickle.loads(pickle.dumps(join))
    np.testing.assert_array_equal(restored.transform(TIMED_PAIRS), join.transform(TIMED_PAIRS))


TIMED_X = np.array(
    [["u1", "a", 1], ["u1", "b", 2], ["u2", "a", 3], ["u2", "c", 4], ["u3", "a", 5]],
    dtype=object,
)


@pytest.mark.parametrize(
    "component",
    [
        InteractionCounts("item"),
        RecommenderScores(ItemKNNRecommender()),
        SegmentPopularity(np.array([["u1", "x"], ["u2", "x"], ["u3", "y"]])),
        JoinStaticFeatures("item", np.array([["a", 1.0], ["b", 2.0]], dtype=object)),
    ],
    ids=lambda c: type(c).__name__,
)
def test_components_that_ignore_time_accept_timed_interactions_and_pairs(component):
    untimed = clone(component).fit(TIMED_X[:, :2])
    timed = clone(component).fit(TIMED_X)
    pairs = np.array([["u1", "c", 9], ["u3", "b", None]], dtype=object)
    np.testing.assert_array_equal(timed.transform(pairs), untimed.transform(pairs[:, :2]))


def test_concat_hands_its_parts_the_time():
    concat = ConcatFeatures(
        [JoinDynamicFeatures("time", module_level_time_callback), GeneratorScores()]
    ).fit()
    out = concat.transform(TIMED_PAIRS, scores=[0.1, 0.2, 0.3, 0.4])
    np.testing.assert_array_equal(out[:, 0], [3.0, 9.0, 3.0, -1.0])
