import numpy as np
import pytest

from skrecsys.model_selection import WarmStartKFold


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
