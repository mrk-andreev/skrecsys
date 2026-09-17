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
    assert rec.feature_names_in_.tolist()  # ty: ignore[unresolved-attribute] == ["user", "item"]
    assert rec.predict(df).shape == (len(df),)
