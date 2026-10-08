import numpy as np
import pandas as pd
import pytest

from skrecsys.recommendation import MostPopularRecommender

X = [["u1", "a"], ["u1", "b"], ["u2", "b"], ["u3", "c"], ["u3", "c"]]


def test_count_vs_sum_weighting():
    y = [5.0, 1.0, 1.0, 1.0, 2.0]
    count = MostPopularRecommender(weighting="count").fit(X, y)
    np.testing.assert_array_equal(count.item_popularity_, [1, 2, 1])
    total = MostPopularRecommender(weighting="sum").fit(X, y)
    np.testing.assert_array_equal(total.item_popularity_, [5, 2, 3])


def test_recommend_is_non_personalized():
    rec = MostPopularRecommender().fit(X)
    items, scores = rec.recommend(["u1", "u2", "u3"], n_recommendations=1, exclude_seen=False)
    assert items.tolist() == [["b"], ["b"], ["b"]]
    np.testing.assert_array_equal(scores, 2.0)


def test_predict_returns_popularity():
    rec = MostPopularRecommender().fit(X)
    np.testing.assert_array_equal(rec.predict([["u1", "c"], ["u3", "a"]]), [1.0, 1.0])


def test_invalid_weighting():
    with pytest.raises(ValueError, match="weighting"):
        MostPopularRecommender(weighting="bogus").fit(X)


def test_dataframe_feature_names():
    df = pd.DataFrame(X, columns=["user", "item"])
    rec = MostPopularRecommender().fit(df)
    assert rec.feature_names_in_.tolist() == ["user", "item"]
    assert rec.predict(df).shape == (len(df),)


def test_unknown_users_get_the_most_popular_items():
    """The cold-start policy: an unknown user has seen nothing, and asking is not an error."""
    rec = MostPopularRecommender().fit(X)  # popularity: a=1, b=2, c=1
    items, scores = rec.recommend(["new", "u1", "other"], n_recommendations=1)
    assert items.tolist() == [["b"], ["c"], ["b"]]
    np.testing.assert_array_equal(scores, [[2.0], [1.0], [2.0]])
    assert rec.recommend(["new"], n_recommendations=3)[0].tolist() == [["b", "a", "c"]]
    assert rec._count_eligible(["new", "u1"]).tolist() == [3, 1]
    with pytest.raises(ValueError, match="Unknown user"):
        rec.predict([["new", "a"]])


def test_exposure_ranks_by_rate():
    """Popularity per exposure: an item the table lacks was never shown."""
    shown = np.array([["b", 38.0], ["c", 0.0], ["zzz", 5.0]], dtype=object)
    rec = MostPopularRecommender(exposure=shown, smoothing=2.0).fit(X)
    np.testing.assert_array_equal(rec.item_counts_, [1, 2, 1])
    np.testing.assert_allclose(rec.item_popularity_, [1 / 2, 2 / 40, 1 / 2])
    assert rec.recommend(["new"], n_recommendations=3)[0].tolist() == [["a", "c", "b"]]
    np.testing.assert_allclose(rec.predict([["u1", "b"]]), [0.05])


def test_exposure_accepts_a_dataframe_and_plain_lists():
    frame = pd.DataFrame({"item": ["a", "b", "c"], "shown": [2, 40, 40]})
    want = MostPopularRecommender(exposure=frame, smoothing=1.0).fit(X).item_popularity_
    np.testing.assert_allclose(want, [1 / 3, 2 / 41, 1 / 41])
    lists = [["a", 2], ["b", 40], ["c", 40]]
    got = MostPopularRecommender(exposure=lists, smoothing=1.0).fit(np.array(X)).item_popularity_
    np.testing.assert_allclose(got, want)


def test_smoothing_is_ignored_without_exposure():
    rec = MostPopularRecommender(smoothing=3.0).fit(X)
    np.testing.assert_array_equal(rec.item_popularity_, [1, 2, 1])


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"exposure": [["a", 1.0]], "smoothing": 0.0}, "smoothing > 0"),
        ({"exposure": [["a", 1.0], ["a", 2.0]]}, "duplicate"),
        ({"exposure": [["a", -1.0]]}, ">= 0"),
        ({"exposure": [["a", 1.0, 2.0]]}, "2 columns"),
        ({"smoothing": -1.0}, "smoothing"),
    ],
)
def test_invalid_exposure(params, match):
    with pytest.raises(ValueError, match=match):
        MostPopularRecommender(**params).fit(X)


@pytest.mark.parametrize("weighting", ["count", "sum"])
def test_partial_fit_with_exposure_equals_fit_on_everything(weighting):
    """Counts accumulate exactly, and each call rates them by the exposure of the moment."""
    first, later = X[:3], [["u4", "a"], ["u1", "d"], ["u3", "c"]]
    shown = [["a", 4.0], ["b", 10.0]]
    rec = MostPopularRecommender(weighting, exposure=shown, smoothing=1.0).partial_fit(first)
    shown_later = [*shown, ["d", 7.0]]
    rec.set_params(exposure=shown_later).partial_fit(later)
    whole = MostPopularRecommender(weighting, exposure=shown_later, smoothing=1.0)
    whole.fit([*first, *later])
    assert rec.item_ids_.tolist() == whole.item_ids_.tolist()
    np.testing.assert_allclose(rec.item_counts_, whole.item_counts_)
    np.testing.assert_allclose(rec.item_popularity_, whole.item_popularity_)
