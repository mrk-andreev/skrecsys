import numpy as np
import pytest

from skrecsys.model_selection import ColdStartSplit, LatestInteractionsSplit, WarmStartKFold
from skrecsys.model_selection._split import positions_in_user


def _dataset(seed=0):
    rng = np.random.default_rng(seed)
    users = rng.integers(20, size=300)
    items = rng.integers(40, size=300)
    return np.column_stack([users, items])


@pytest.mark.parametrize("shuffle", [False, True])
def test_warm_start_invariants(shuffle):
    X = _dataset()
    cv = WarmStartKFold(n_splits=4, shuffle=shuffle, random_state=0 if shuffle else None)
    tests = []
    for train, test in cv.split(X):
        assert len(test) > 0
        assert np.intersect1d(train, test).size == 0
        assert len(train) + len(test) == len(X)
        assert set(X[test, 0]) <= set(X[train, 0])
        assert set(X[test, 1]) <= set(X[train, 1])
        tests.append(test)
    all_test = np.concatenate(tests)
    assert len(all_test) == len(np.unique(all_test)), "test folds must be disjoint"
    assert cv.get_n_splits() == 4


def test_fold_sizes_are_balanced():
    sizes = [len(test) for _, test in WarmStartKFold(n_splits=3).split(_dataset())]
    assert max(sizes) - min(sizes) <= 1


def test_shuffle_is_reproducible():
    X = _dataset()
    a = [t.tolist() for _, t in WarmStartKFold(3, shuffle=True, random_state=1).split(X)]
    b = [t.tolist() for _, t in WarmStartKFold(3, shuffle=True, random_state=1).split(X)]
    c = [t.tolist() for _, t in WarmStartKFold(3, shuffle=True, random_state=2).split(X)]
    assert a == b
    assert a != c


def test_string_ids():
    X = [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "b"], ["u1", "c"], ["u2", "c"]]
    assert [t.tolist() for _, t in WarmStartKFold(n_splits=2).split(X)] == [[3], [5]]


def test_unsatisfiable_graph():
    X = [["u1", "a"], ["u2", "b"], ["u3", "c"], ["u1", "b"]]
    with pytest.raises(ValueError, match="warm-start folds"):
        list(WarmStartKFold(n_splits=2).split(X))


@pytest.mark.parametrize("n_splits", [1, 0, 2.0, True])
def test_invalid_n_splits(n_splits):
    with pytest.raises(ValueError, match="n_splits"):
        WarmStartKFold(n_splits=n_splits)


def test_random_state_without_shuffle():
    with pytest.raises(ValueError, match="shuffle"):
        WarmStartKFold(random_state=0)


def _timed_dataset():
    """30 users with 10 rows each, in time order within each user."""
    users = np.repeat(np.arange(30), 10)
    items = np.tile(np.arange(10), 30)
    return np.column_stack([users, items])


def test_cold_start_holds_out_cold_users_whole_and_the_latest_rows_of_the_rest():
    X = _timed_dataset()
    train, test = next(ColdStartSplit(cold_users=0.2, test_size=0.3, random_state=0).split(X))
    assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(len(X)))
    cold = np.setdiff1d(X[test, 0], X[train, 0])
    assert len(cold) == 6
    for user in np.unique(X[:, 0]):
        rows = np.flatnonzero(X[:, 0] == user)
        held = rows[np.isin(rows, test)]
        if user in cold:
            assert np.array_equal(held, rows)
        else:
            # The last three of ten rows, in the order X lists them.
            assert np.array_equal(held, rows[-3:])


def test_cold_start_without_a_test_size_holds_out_the_cold_users_only():
    X = _timed_dataset()
    train, test = next(ColdStartSplit(cold_users=0.1, test_size=0.0, random_state=0).split(X))
    assert len(np.unique(X[test, 0])) == 3
    assert not np.isin(X[test, 0], X[train, 0]).any()


def test_cold_start_is_reproducible_and_depends_on_the_seed():
    X = _timed_dataset()

    def test_of(seed):
        return next(ColdStartSplit(0.2, random_state=seed).split(X))[1]

    assert np.array_equal(test_of(0), test_of(0))
    assert not np.array_equal(test_of(0), test_of(1))
    assert ColdStartSplit().get_n_splits() == 1


def test_cold_start_string_ids():
    X = np.array([[f"u{u}", f"i{i}"] for u in range(10) for i in range(5)])
    train, test = next(ColdStartSplit(0.2, 0.2, random_state=0).split(X))
    assert len(train) + len(test) == len(X)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"cold_users": 0.0}, "cold_users"),
        ({"cold_users": 1.0}, "cold_users"),
        ({"test_size": 1.0}, "test_size"),
        ({"test_size": -0.1}, "test_size"),
    ],
)
def test_cold_start_validates(params, match):
    with pytest.raises(ValueError, match=match):
        ColdStartSplit(**params)


def test_cold_start_needs_a_cold_and_a_warm_user():
    X = _timed_dataset()[:20]  # two users
    with pytest.raises(ValueError, match="both must be non-empty"):
        next(ColdStartSplit(cold_users=0.2).split(X))


def test_positions_in_user_number_rows_in_row_order_or_by_time():
    codes = np.array([1, 0, 1, 1, 0])
    position, counts = positions_in_user(codes)
    assert position.tolist() == [0, 0, 1, 2, 1]
    assert counts.tolist() == [2, 3]
    # by time, ties in row order: user 1's rows at times 5, 1, 1 are its third, first, second
    position, _ = positions_in_user(codes, np.array([5.0, 2.0, 1.0, 1.0, 0.0]))
    assert position.tolist() == [2, 1, 0, 1, 0]


def test_latest_interactions_split_holds_out_the_last_rows_of_every_user():
    X = _dataset()
    train, test = next(LatestInteractionsSplit(test_size=0.25).split(X))
    assert np.intersect1d(train, test).size == 0
    assert len(train) + len(test) == len(X)
    for user in np.unique(X[:, 0]):
        rows = np.flatnonzero(X[:, 0] == user)
        n_held = len(rows) // 4
        assert sorted(np.intersect1d(test, rows)) == rows[len(rows) - n_held :].tolist()
    assert LatestInteractionsSplit().get_n_splits() == 1


def test_latest_interactions_split_of_the_most_recently_active_users():
    # last rows: u0 at 7, u1 at 5, u2 at 8, u3 (one row, nothing to hold out) at 9
    users = ["u0", "u1", "u0", "u2", "u1", "u1", "u2", "u0", "u2", "u3"]
    X = np.column_stack([users, [f"i{k}" for k in range(len(users))]])
    held = {
        max_users: next(LatestInteractionsSplit(0.5, max_users=max_users).split(X))[1].tolist()
        for max_users in (None, 3, 2, 1)
    }
    # each of u0, u1, u2 has three rows and gives its last one
    assert held[None] == held[3] == [5, 7, 8]
    assert held[2] == [7, 8]
    assert held[1] == [8]


def test_latest_interactions_split_accepts_context_columns():
    X = _dataset()
    plain = next(LatestInteractionsSplit(0.3, max_users=5).split(X))
    context = np.column_stack([X, np.arange(len(X)) % 3])
    with_context = next(LatestInteractionsSplit(0.3, max_users=5).split(context))
    for got, want in zip(with_context, plain, strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"test_size": 0.0}, "test_size"),
        ({"test_size": 1.0}, "test_size"),
        ({"max_users": 0}, "max_users"),
        ({"max_users": 2.5}, "max_users"),
    ],
)
def test_latest_interactions_split_validates(params, match):
    with pytest.raises(ValueError, match=match):
        LatestInteractionsSplit(**params)
