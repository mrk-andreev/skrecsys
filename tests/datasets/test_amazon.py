import numpy as np
import pandas as pd
import pytest

from skrecsys.datasets import _amazon, fetch_amazon_books
from skrecsys.recommendation import ItemKNNRecommender

from .conftest import AMAZON_ENCODED_ITEMS, AMAZON_ITEMS, AMAZON_RATINGS, AMAZON_USERS

# Every surviving user rates i1..i5, latest first in the file and hence last in the
# sorted rows; user codes follow the dropped d1..d5, which are encoded before they go.
FIRST_USER_CODE = 5
ITEMS_BY_TIME = [4, 3, 2, 1, 0]


def test_bunch_contents(fake_amazon_download, data_home):
    bunch = fetch_amazon_books(data_home=data_home)
    users = bunch.data[:, 0]
    assert np.unique(users).tolist() == [FIRST_USER_CODE + i for i in range(len(AMAZON_USERS))]
    # sorted by user, then timestamp: each user rates i5 first and i1 last
    assert bunch.data[:, 1].tolist() == ITEMS_BY_TIME * len(AMAZON_USERS)
    assert bunch.data.dtype == np.int64
    assert bunch.target.dtype == np.float64
    np.testing.assert_array_equal(bunch.target[:5], [1.0] * 5)
    np.testing.assert_array_equal(bunch.timestamps[:5], [996, 997, 998, 999, 1000])
    assert bunch.feature_names == ["user_id", "item_id"]
    assert bunch.target_names == ["rating"]
    assert "Amazon Books" in bunch.DESCR
    assert "train_indices" not in bunch
    assert "frame" not in bunch


def test_identifiers_are_kept(fake_amazon_download, data_home):
    bunch = fetch_amazon_books(data_home=data_home)
    items = bunch.item_info
    # x1..x4 clear the 5-interaction floor and are encoded, even though the only users
    # rating them are dropped right after; the reference preprocessing numbers them too.
    assert items.asin.tolist() == AMAZON_ENCODED_ITEMS
    assert items.item_id.tolist() == list(range(len(AMAZON_ENCODED_ITEMS)))
    assert set(bunch.data[:, 1].tolist()) == {AMAZON_ENCODED_ITEMS.index(a) for a in AMAZON_ITEMS}
    users = bunch.user_info
    assert users.reviewer_id[FIRST_USER_CODE:].tolist() == AMAZON_USERS
    assert users.user_id.tolist() == list(range(len(users.reviewer_id)))


def test_rare_users_and_items_are_dropped(fake_amazon_download, data_home):
    bunch = fetch_amazon_books(data_home=data_home, max_sequence_length=None)
    rated = set(bunch.data[:, 1].tolist())
    assert len(bunch.data) == len(AMAZON_USERS) * len(AMAZON_ITEMS)
    # a user left under the floor by the item filter keeps no interactions
    assert {AMAZON_ENCODED_ITEMS.index(a) for a in ("x1", "x2", "x3", "x4")} & rated == set()
    assert not any(asin.startswith("rare") for asin in bunch.item_info.asin.tolist())


@pytest.mark.parametrize("max_sequence_length", [1, 2, 4, 5, 50, None])
def test_truncation_keeps_the_latest(fake_amazon_download, data_home, max_sequence_length):
    bunch = fetch_amazon_books(data_home=data_home, max_sequence_length=max_sequence_length)
    kept = len(AMAZON_ITEMS) if max_sequence_length is None else min(max_sequence_length, 5)
    assert len(bunch.data) == kept * len(AMAZON_USERS)
    first_user = bunch.data[bunch.data[:, 0] == FIRST_USER_CODE]
    assert first_user[:, 1].tolist() == ITEMS_BY_TIME[len(ITEMS_BY_TIME) - kept :]


def test_leave_one_out_adds_a_history_slot(fake_amazon_download, data_home):
    """A held-out interaction sits outside the history it is predicted from."""
    bunch = fetch_amazon_books(data_home=data_home, subset="leave-one-out", max_sequence_length=2)
    assert len(bunch.data) == 3 * len(AMAZON_USERS)
    train, test = bunch.train_indices, bunch.test_indices
    assert np.intersect1d(train, test).size == 0
    assert len(train) + len(test) == len(bunch.data)
    assert np.all(np.diff(train) > 0)
    assert np.all(np.diff(test) > 0)
    # the last interaction of each user, which is i1 for everyone
    assert len(test) == len(AMAZON_USERS)
    assert bunch.data[test][:, 1].tolist() == [0] * len(AMAZON_USERS)
    np.testing.assert_array_equal(bunch.timestamps[test], [1000 - 10 * i for i in range(5)])


def test_leave_one_out_trains_a_recommender(fake_amazon_download, data_home):
    bunch = fetch_amazon_books(data_home=data_home, subset="leave-one-out")
    X, y = bunch.data, bunch.target
    model = ItemKNNRecommender(n_neighbors=2).fit(X[bunch.train_indices], y[bunch.train_indices])
    queries = np.unique(X[bunch.test_indices][:, 0])
    # every fixture user has rated the whole catalogue, so nothing is left to exclude
    recommended, _ = model.recommend(queries, n_recommendations=2, exclude_seen=False)
    assert recommended.shape == (len(AMAZON_USERS), 2)


def test_return_x_y(fake_amazon_download, data_home):
    X, y = fetch_amazon_books(data_home=data_home, return_X_y=True)
    bunch = fetch_amazon_books(data_home=data_home)
    np.testing.assert_array_equal(X, bunch.data)
    np.testing.assert_array_equal(y, bunch.target)


def test_as_frame(fake_amazon_download, data_home):
    bunch = fetch_amazon_books(data_home=data_home, as_frame=True)
    assert isinstance(bunch.data, pd.DataFrame)
    assert list(bunch.data.columns) == ["user_id", "item_id"]
    assert bunch.target.name == "rating"
    assert bunch.timestamps.name == "timestamp"
    assert list(bunch.frame.columns) == ["user_id", "item_id", "rating", "timestamp"]
    assert list(bunch.item_info.columns) == ["item_id", "asin"]
    assert list(bunch.user_info.columns) == ["user_id", "reviewer_id"]


def test_invalid_arguments(fake_amazon_download, data_home):
    with pytest.raises(ValueError, match="subset must be one of"):
        fetch_amazon_books(data_home=data_home, subset="loo")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="return_X_y=True is only supported"):
        fetch_amazon_books(data_home=data_home, subset="leave-one-out", return_X_y=True)  # ty: ignore[no-matching-overload]
    with pytest.raises(ValueError, match="max_sequence_length must be"):
        fetch_amazon_books(data_home=data_home, max_sequence_length=0)
    assert not fake_amazon_download


def test_downloads_once(fake_amazon_download, data_home):
    fetch_amazon_books(data_home=data_home)
    fetch_amazon_books(data_home=data_home, max_sequence_length=3)
    assert len(fake_amazon_download) == 1
    # the raw csv is removed once parsed; only the cache stays behind
    assert sorted(path.name for path in data_home.iterdir()) == ["amazon_books.joblib"]


def test_download_if_missing(fake_amazon_download, data_home):
    with pytest.raises(OSError, match="download_if_missing is False"):
        fetch_amazon_books(data_home=data_home, download_if_missing=False)
    assert not fake_amazon_download


def test_chunked_reading_matches_one_pass(tmp_path, monkeypatch):
    """A line split across two reads must be stitched back, not dropped or halved."""
    csv = tmp_path / "ratings.csv"
    csv.write_text("".join(f"{u},{i},{r},{t}\n" for u, i, r, t in AMAZON_RATINGS))
    whole = _amazon._read_columns(csv)
    for chunk_bytes in (1, 7, 13, 64):
        monkeypatch.setattr(_amazon, "_CHUNK_BYTES", chunk_bytes)
        for name, column in _amazon._read_columns(csv).items():
            np.testing.assert_array_equal(column, whole[name], err_msg=f"{name} @ {chunk_bytes}")


def test_missing_trailing_newline_keeps_the_last_row(tmp_path):
    csv = tmp_path / "ratings.csv"
    csv.write_text("u1,i1,5.0,100\nu2,i2,4.0,101")
    assert _amazon._read_columns(csv)["item"].tolist() == [b"i1", b"i2"]


def test_malformed_rows_are_rejected(tmp_path):
    csv = tmp_path / "ratings.csv"
    csv.write_text("u1,i1,5.0,100\nu2,i2,4.0\n")
    with pytest.raises(ValueError, match="Malformed ratings row"):
        _amazon._read_columns(csv)


def test_empty_file_is_rejected(tmp_path):
    csv = tmp_path / "ratings.csv"
    csv.write_text("")
    with pytest.raises(ValueError, match="holds no ratings"):
        _amazon._read_columns(csv)
