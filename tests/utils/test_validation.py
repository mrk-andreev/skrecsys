import numpy as np
import pytest

from skrecsys.utils.validation import check_ids, check_interactions, encode_ids


def test_check_interactions_default_weights():
    users, items, y = check_interactions([["u", "a"], ["v", "b"]])
    assert users.tolist() == ["u", "v"]
    assert items.tolist() == ["a", "b"]
    np.testing.assert_array_equal(y, [1.0, 1.0])


def test_check_interactions_with_y():
    _, _, y = check_interactions([[1, 2], [3, 4]], [5, 0])
    assert y.dtype == np.float64
    np.testing.assert_array_equal(y, [5.0, 0.0])


@pytest.mark.parametrize("X", [[[1, 2, 3]], [[1]]])
def test_check_interactions_requires_two_columns(X):
    with pytest.raises(ValueError, match="2 columns"):
        check_interactions(X)


def test_check_interactions_inconsistent_length():
    with pytest.raises(ValueError, match="inconsistent"):
        check_interactions([[1, 2], [3, 4]], [1])


def test_check_interactions_non_finite_ids():
    with pytest.raises(ValueError, match="non-finite"):
        check_interactions([[1.0, np.nan]])


def test_check_ids_requires_1d():
    with pytest.raises(ValueError, match="one-dimensional"):
        check_ids([["a"]])


def test_encode_ids():
    fitted = np.array(["a", "b", "d"], dtype=object)
    np.testing.assert_array_equal(
        encode_ids(np.array(["d", "a"], dtype=object), fitted, name="item"), [2, 0]
    )
    with pytest.raises(ValueError, match="Unknown item"):
        encode_ids(np.array(["c"], dtype=object), fitted, name="item")
    with pytest.raises(ValueError, match="Unknown item"):
        encode_ids(np.array(["z"], dtype=object), fitted, name="item")
