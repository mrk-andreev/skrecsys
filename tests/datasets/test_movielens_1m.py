import numpy as np
import pandas as pd
import pytest

from skrecsys.datasets import fetch_movielens_1m

from .conftest import ML1M_RATINGS

# Each user rates item 1 then item 2; item 2 carries the earlier timestamp, so sorting
# by user and time puts it first.
ITEMS_BY_TIME = [2, 1]


def test_bunch_contents(fake_ml1m_download, data_home):
    bunch = fetch_movielens_1m(data_home=data_home)
    assert bunch.data.tolist() == [[user, item] for user in (1, 2, 3) for item in ITEMS_BY_TIME]
    assert bunch.data.dtype == np.int64
    assert bunch.target.dtype == np.float64
    np.testing.assert_array_equal(bunch.target, [3, 4, 3, 4, 3, 4])
    np.testing.assert_array_equal(bunch.timestamps, [988, 989, 978, 979, 968, 969])
    assert bunch.feature_names == ["user_id", "item_id"]
    assert bunch.target_names == ["rating"]
    assert "MovieLens 1M" in bunch.DESCR
    assert "train_indices" not in bunch
    assert "frame" not in bunch


def test_metadata(fake_ml1m_download, data_home):
    bunch = fetch_movielens_1m(data_home=data_home)
    users = bunch.user_info
    assert users.user_id.tolist() == [1, 2, 3]
    assert users.gender.tolist() == ["F", "M", "M"]
    assert users.age.tolist() == [25, 35, 18]
    assert users.occupation.tolist() == [4, 7, 0]
    assert users.zip_code.tolist() == ["02138", "94043", "55455"]
    items = bunch.item_info
    # the movie table covers films nobody rated, which is what the id space means
    assert items.item_id.tolist() == [1, 2, 3]
    assert items.title[1] == "Amélie (2001)"
    assert items.year.tolist() == [1995, 2001, 0]
    assert bunch.genre_names == ["Animation", "Children's", "Comedy", "Drama", "Romance"]
    np.testing.assert_array_equal(
        items.genres,
        [[1, 1, 1, 0, 0], [0, 0, 1, 0, 1], [0, 0, 0, 1, 0]],
    )


def test_identifiers_are_not_renumbered(fake_ml1m_download, data_home):
    """Unlike the Amazon loader, ml-1m keeps the ids MovieLens ships."""
    bunch = fetch_movielens_1m(data_home=data_home)
    assert set(bunch.data[:, 0].tolist()) == {user for user, _, _, _ in ML1M_RATINGS}
    assert set(bunch.data[:, 1].tolist()) == {item for _, item, _, _ in ML1M_RATINGS}


@pytest.mark.parametrize("max_sequence_length", [1, 2, 200, None])
def test_truncation_keeps_the_latest(fake_ml1m_download, data_home, max_sequence_length):
    bunch = fetch_movielens_1m(data_home=data_home, max_sequence_length=max_sequence_length)
    kept = 2 if max_sequence_length is None else min(max_sequence_length, 2)
    assert len(bunch.data) == kept * 3
    assert bunch.data[bunch.data[:, 0] == 1][:, 1].tolist() == ITEMS_BY_TIME[2 - kept :]


def test_leave_one_out_adds_a_history_slot(fake_ml1m_download, data_home):
    bunch = fetch_movielens_1m(data_home=data_home, subset="leave-one-out", max_sequence_length=1)
    # one history slot plus the held-out rating, for each of the three users
    assert len(bunch.data) == 6
    train, test = bunch.train_indices, bunch.test_indices
    assert np.intersect1d(train, test).size == 0
    assert len(train) == len(test) == 3
    assert bunch.data[test][:, 1].tolist() == [1, 1, 1]
    np.testing.assert_array_equal(bunch.timestamps[test], [989, 979, 969])


def test_return_x_y(fake_ml1m_download, data_home):
    X, y = fetch_movielens_1m(data_home=data_home, return_X_y=True)
    bunch = fetch_movielens_1m(data_home=data_home)
    np.testing.assert_array_equal(X, bunch.data)
    np.testing.assert_array_equal(y, bunch.target)


def test_as_frame(fake_ml1m_download, data_home):
    bunch = fetch_movielens_1m(data_home=data_home, as_frame=True)
    assert isinstance(bunch.data, pd.DataFrame)
    assert list(bunch.frame.columns) == ["user_id", "item_id", "rating", "timestamp"]
    assert list(bunch.user_info.columns) == [
        "user_id",
        "gender",
        "age",
        "occupation",
        "zip_code",
    ]
    assert list(bunch.item_info.columns) == ["item_id", "title", "year", *bunch.genre_names]
    assert bunch.item_info["Comedy"].tolist() == [True, True, False]


def test_invalid_arguments(fake_ml1m_download, data_home):
    with pytest.raises(ValueError, match="subset must be one of"):
        fetch_movielens_1m(data_home=data_home, subset="loo")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="return_X_y=True is only supported"):
        fetch_movielens_1m(data_home=data_home, subset="leave-one-out", return_X_y=True)  # ty: ignore[no-matching-overload]
    with pytest.raises(ValueError, match="max_sequence_length must be"):
        fetch_movielens_1m(data_home=data_home, max_sequence_length=0)
    assert not fake_ml1m_download


def test_downloads_once(fake_ml1m_download, data_home):
    fetch_movielens_1m(data_home=data_home)
    fetch_movielens_1m(data_home=data_home, max_sequence_length=1)
    assert len(fake_ml1m_download) == 1
    assert sorted(path.name for path in data_home.iterdir()) == ["movielens_1m.joblib"]


def test_download_if_missing(fake_ml1m_download, data_home):
    with pytest.raises(OSError, match="download_if_missing is False"):
        fetch_movielens_1m(data_home=data_home, download_if_missing=False)
    assert not fake_ml1m_download
