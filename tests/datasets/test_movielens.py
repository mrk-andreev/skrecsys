import builtins
import os

import numpy as np
import pandas as pd
import pytest

from skrecsys.datasets import fetch_movielens_100k
from skrecsys.model_selection import WarmStartKFold
from skrecsys.recommendation import ItemKNNRecommender

from .conftest import GENRES, RATINGS

SUBSETS = ["u1", "u2", "u3", "u4", "u5", "ua", "ub"]


def test_bunch_contents(fake_download, data_home):
    bunch = fetch_movielens_100k(data_home=data_home)
    # sorted by user, timestamp, item
    assert bunch.data.tolist() == [[1, 20], [1, 30], [1, 10], [2, 10], [2, 30], [3, 20]]
    assert bunch.data.dtype == np.int64
    assert bunch.target.dtype == np.float64
    np.testing.assert_array_equal(bunch.target, [3, 4, 5, 1, 2, 4])
    np.testing.assert_array_equal(bunch.timestamps, [100, 100, 300, 200, 250, 50])
    assert bunch.feature_names == ["user_id", "item_id"]
    assert bunch.target_names == ["rating"]
    assert bunch.genre_names == GENRES
    assert "MovieLens 100K" in bunch.DESCR
    assert "train_indices" not in bunch
    assert "frame" not in bunch


def test_metadata(fake_download, data_home):
    bunch = fetch_movielens_100k(data_home=data_home)
    users = bunch.user_info
    assert users.user_id.tolist() == [1, 2, 3]
    assert users.age.tolist() == [24, 53, 23]
    assert users.zip_code.tolist() == ["85711", "T8H1N", "32067"]
    items = bunch.item_info
    assert items.title.tolist() == ["Toy Story (1995)", "Amélie (2001)", "unknown"]
    assert items.release_date.dtype == np.dtype("datetime64[D]")
    assert items.release_date[1] == np.datetime64("2001-04-25")
    assert np.isnat(items.release_date[2])
    assert items.genres.dtype == bool
    np.testing.assert_array_equal(items.genres, [[0, 0, 1], [0, 1, 1], [1, 0, 0]])


def test_return_x_y(fake_download, data_home):
    X, y = fetch_movielens_100k(data_home=data_home, return_X_y=True)
    bunch = fetch_movielens_100k(data_home=data_home)
    np.testing.assert_array_equal(X, bunch.data)
    np.testing.assert_array_equal(y, bunch.target)


@pytest.mark.parametrize("subset", SUBSETS)
def test_subset_indices(fake_download, data_home, subset):
    bunch = fetch_movielens_100k(data_home=data_home, subset=subset)
    train, test = bunch.train_indices, bunch.test_indices
    assert np.intersect1d(train, test).size == 0
    assert len(train) + len(test) == len(RATINGS)
    assert np.all(np.diff(train) > 0)
    assert np.all(np.diff(test) > 0)
    rows = {tuple(r) for r in np.column_stack([bunch.data, bunch.target, bunch.timestamps])}
    assert len(rows) == len(RATINGS)


def test_subset_matches_fold_files(fake_download, data_home):
    bunch = fetch_movielens_100k(data_home=data_home, subset="ua")
    # ua.test holds the first rating (1, 10, 5, 300)
    assert bunch.data[bunch.test_indices].tolist() == [[1, 10]]
    assert bunch.target[bunch.test_indices].tolist() == [5.0]


def test_invalid_arguments(fake_download, data_home):
    with pytest.raises(ValueError, match="subset must be one of"):
        fetch_movielens_100k(data_home=data_home, subset="u6")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="return_X_y=True is only supported"):
        fetch_movielens_100k(data_home=data_home, subset="u1", return_X_y=True)  # ty: ignore[no-matching-overload]
    assert not fake_download


def test_download_if_missing(fake_download, data_home):
    with pytest.raises(OSError, match="Data not found"):
        fetch_movielens_100k(data_home=data_home, download_if_missing=False)
    assert not fake_download


def test_cache_is_reused(fake_download, data_home):
    fetch_movielens_100k(data_home=data_home)
    assert sorted(p.name for p in data_home.iterdir()) == ["movielens_100k.joblib"]
    bunch = fetch_movielens_100k(data_home=data_home, subset="u1", download_if_missing=False)
    assert len(fake_download) == 1
    assert len(bunch.test_indices) == 2


def test_as_frame(fake_download, data_home):
    bunch = fetch_movielens_100k(data_home=data_home, as_frame=True)
    assert isinstance(bunch.data, pd.DataFrame)
    assert bunch.data.columns.tolist() == ["user_id", "item_id"]
    assert isinstance(bunch.target, pd.Series)
    assert bunch.target.name == "rating"
    assert bunch.frame.columns.tolist() == ["user_id", "item_id", "rating", "timestamp"]
    assert len(bunch.frame) == len(RATINGS)
    assert bunch.user_info.columns.tolist() == [
        "user_id",
        "age",
        "gender",
        "occupation",
        "zip_code",
    ]
    items = bunch.item_info
    assert items.columns.tolist() == ["item_id", "title", "release_date", "imdb_url", *GENRES]
    assert items["Comedy"].dtype == bool
    assert pd.isna(items["release_date"].iloc[2])

    X, y = fetch_movielens_100k(data_home=data_home, as_frame=True, return_X_y=True)
    pd.testing.assert_frame_equal(X, bunch.data)
    pd.testing.assert_series_equal(y, bunch.target)


def test_as_frame_without_pandas(fake_download, data_home, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"skrecsys\[pandas\]"):
        fetch_movielens_100k(data_home=data_home, as_frame=True)


def test_fits_recommender(fake_download, data_home):
    X, y = fetch_movielens_100k(data_home=data_home, return_X_y=True)
    rec = ItemKNNRecommender().fit(X, y)
    assert rec.n_users_ == 3
    assert rec.n_items_ == 3


@pytest.mark.skipif(
    os.environ.get("SKRECSYS_NETWORK_TESTS") != "1",
    reason="test is enabled when SKRECSYS_NETWORK_TESTS=1",
)
def test_network_fetch(tmp_path):
    bunch = fetch_movielens_100k(data_home=tmp_path)
    assert bunch.data.shape == (100_000, 2)
    assert len(np.unique(bunch.data[:, 0])) == 943
    assert len(np.unique(bunch.data[:, 1])) == 1682
    assert bunch.user_info.user_id.shape == (943,)
    assert bunch.item_info.genres.shape == (1682, 19)
    assert set(np.unique(bunch.target)) == {1.0, 2.0, 3.0, 4.0, 5.0}

    folds = [
        fetch_movielens_100k(data_home=tmp_path, subset=subset)
        for subset in ("u1", "u2", "u3", "u4", "u5")
    ]
    assert all(len(fold.test_indices) == 20_000 for fold in folds)
    all_test = np.concatenate([fold.test_indices for fold in folds])
    np.testing.assert_array_equal(np.sort(all_test), np.arange(100_000))

    ua = fetch_movielens_100k(data_home=tmp_path, subset="ua")
    _, per_user = np.unique(bunch.data[ua.test_indices, 0], return_counts=True)
    assert np.all(per_user == 10)

    cv = WarmStartKFold(n_splits=2, shuffle=True, random_state=0)
    assert sum(1 for _ in cv.split(bunch.data)) == 2
