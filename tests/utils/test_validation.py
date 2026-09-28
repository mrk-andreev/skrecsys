import datetime

import numpy as np
import pytest
import scipy.sparse as sp

from skrecsys import _core
from skrecsys.recommendation import ItemKNNRecommender, MostPopularRecommender
from skrecsys.utils.validation import (
    check_as_of,
    check_ids,
    check_interactions,
    check_optionally_timed,
    check_queries,
    check_rows,
    check_times,
    drop_time,
    encode_ids,
    factorize,
    interaction_context,
    lookup_ids,
    stack_columns,
)


def test_check_interactions_default_weights():
    users, items, y = check_interactions([["u", "a"], ["v", "b"]])
    assert users.tolist() == ["u", "v"]
    assert items.tolist() == ["a", "b"]
    np.testing.assert_array_equal(y, [1.0, 1.0])


def test_check_interactions_with_y():
    _, _, y = check_interactions([[1, 2], [3, 4]], [5, 0])
    assert y.dtype == np.float64
    np.testing.assert_array_equal(y, [5.0, 0.0])


def test_check_interactions_requires_two_columns():
    with pytest.raises(ValueError, match="at least 2 columns"):
        check_interactions([[1]])


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


def test_lookup_ids_marks_unknown_identifiers_rather_than_raising():
    fitted = np.array(["a", "b", "d"], dtype=object)
    positions, known = lookup_ids(np.array(["d", "c", "a", "z"], dtype=object), fitted, name="item")
    np.testing.assert_array_equal(known, [True, False, True, False])
    np.testing.assert_array_equal(positions[known], [2, 0])


def test_lookup_ids_against_nothing_fitted():
    positions, known = lookup_ids(np.array([1, 2]), np.array([], dtype=np.int64), name="user")
    assert positions.shape == (2,)
    assert not known.any()


FACTORIZE_CASES = {
    "int64 dense": np.arange(2000) % 97,
    "int64 spread": (np.arange(2000) % 97) * 1_000_000_007,
    "int64 negative": (np.arange(2000) % 97) - 48,
    "int32": (np.arange(2000) % 97).astype(np.int32),
    "uint16": (np.arange(2000) % 97).astype(np.uint16),
    "uint64": (np.arange(2000) % 97).astype(np.uint64),
    "single value": np.array([7]),
    "constant": np.full(100, -3),
    "extremes": np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max, 0, -1]),
    "object str": np.array([f"u{v % 97}" for v in range(2000)], dtype=object),
    "object int": np.array([v % 97 for v in range(2000)], dtype=object),
    "unicode": np.array([f"u{v % 97}" for v in range(2000)]),
    "float": (np.arange(2000) % 97).astype(np.float64),
}


@pytest.mark.parametrize("values", FACTORIZE_CASES.values(), ids=FACTORIZE_CASES.keys())
def test_factorize_matches_numpy_unique(values):
    """`factorize` is `np.unique(..., return_inverse=True)`, down to the dtypes."""
    ids, codes = factorize(values)
    want_ids, want_codes = np.unique(values, return_inverse=True)
    assert ids.dtype == want_ids.dtype
    assert codes.dtype == want_codes.dtype
    np.testing.assert_array_equal(ids, want_ids)
    np.testing.assert_array_equal(codes, want_codes)
    np.testing.assert_array_equal(ids[codes], values)


def test_factorize_falls_back_for_identifiers_it_cannot_hash():
    """Hashing is only a shortcut; identifiers it cannot take go back to numpy."""
    values = np.empty(3, dtype=object)
    values[0], values[1], values[2] = [1], [2], [1]
    ids, codes = factorize(values)
    want_ids, want_codes = np.unique(values, return_inverse=True)
    np.testing.assert_array_equal(ids, want_ids)
    np.testing.assert_array_equal(codes, want_codes)


def test_factorize_raises_on_identifiers_it_cannot_order():
    """Sorting mixed identifiers fails in `factorize` exactly where numpy fails."""
    values = np.array([1, "a"], dtype=object)
    with pytest.raises(TypeError):
        np.unique(values)
    with pytest.raises(TypeError):
        factorize(values)


def _coo_to_csr(rows, cols, data, shape):
    indptr, indices, values = _core.coo_to_csr(
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(data, dtype=np.float64),
        shape[0],
        shape[1],
        0,
    )
    return sp.csr_array((values, indices, indptr), shape=shape)


@pytest.mark.parametrize(
    ("rows", "cols", "data", "shape"),
    [
        ([0, 1, 0, 1, 0], [2, 0, 2, 1, 0], [1.0, 2.0, 3.0, 4.0, 5.0], (2, 3)),
        ([2], [1], [0.0], (4, 3)),  # an explicit zero still marks the pair as observed
        ([], [], [], (3, 2)),
        ([0, 0, 0], [1, 1, 1], [1.0, -1.0, 0.5], (1, 2)),  # every entry a duplicate
    ],
)
def test_coo_to_csr_matches_scipy(rows, cols, data, shape):
    got = _coo_to_csr(rows, cols, data, shape)
    want = sp.csr_array((data, (rows, cols)), shape=shape)
    want.sum_duplicates()
    np.testing.assert_array_equal(got.indptr, want.indptr)
    np.testing.assert_array_equal(got.indices, want.indices)
    np.testing.assert_allclose(got.data, want.data, rtol=0, atol=1e-12)
    assert got.has_canonical_format


def test_fit_builds_the_matrix_scipy_would_have():
    """Duplicate pairs are summed and zero weights stay stored, as before."""
    X = np.array([["u0", "i1"], ["u0", "i1"], ["u1", "i0"], ["u0", "i0"]], dtype=object)
    y = np.array([2.0, 3.0, 0.0, 1.0])
    est = MostPopularRecommender().fit(X, y)

    users, items = np.unique(X[:, 0]), np.unique(X[:, 1])
    want = sp.csr_array(
        (y, (np.searchsorted(users, X[:, 0]), np.searchsorted(items, X[:, 1]))),
        shape=(len(users), len(items)),
    )
    want.sum_duplicates()
    np.testing.assert_array_equal(est.interactions_.indptr, want.indptr)
    np.testing.assert_array_equal(est.interactions_.indices, want.indices)
    np.testing.assert_allclose(est.interactions_.data, want.data)
    assert est.interactions_[0, 1] == 5.0
    assert est.interactions_.nnz == 3


@pytest.mark.parametrize("n_threads", [0, 1, 2, 8])
def test_coo_to_csr_is_the_same_whatever_the_thread_count(n_threads):
    """Blocks are cut by worker count, so the split must not reach the result."""
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 60, 5_000)
    cols = rng.integers(0, 40, 5_000)
    data = rng.random(5_000)
    got = _core.coo_to_csr(rows.astype(np.int64), cols.astype(np.int64), data, 60, 40, n_threads)
    want = sp.csr_array((data, (rows, cols)), shape=(60, 40))
    want.sum_duplicates()
    np.testing.assert_array_equal(got[0], want.indptr)
    np.testing.assert_array_equal(got[1], want.indices)
    np.testing.assert_allclose(got[2], want.data, rtol=0, atol=1e-12)


def test_fit_honours_n_jobs_for_the_matrix_it_builds():
    """A single-threaded estimator must not reach for every core while building."""
    X = np.array([[u, i] for u in range(20) for i in range(u % 7)])
    single = ItemKNNRecommender(n_neighbors=2, n_jobs=1).fit(X)
    every = ItemKNNRecommender(n_neighbors=2).fit(X)
    np.testing.assert_array_equal(single.interactions_.indptr, every.interactions_.indptr)
    np.testing.assert_array_equal(single.interactions_.indices, every.interactions_.indices)
    assert single._build_threads() == 1
    assert every._build_threads() == 0


def test_check_interactions_with_time_returns_the_times():
    X = np.array([["u", "a", 5], ["v", "b", 7]], dtype=object)
    users, items, y, times = check_interactions(X, time=True)
    assert users.tolist() == ["u", "v"]
    assert items.tolist() == ["a", "b"]
    np.testing.assert_array_equal(y, [1.0, 1.0])
    assert times.dtype == np.int64
    np.testing.assert_array_equal(times, [5, 7])


def test_check_interactions_with_time_requires_three_columns():
    with pytest.raises(ValueError, match="3 columns"):
        check_interactions([["u", "a"]], time=True)


def test_check_interactions_reads_further_columns_as_context():
    X = np.array([["u", "a", "mobile", 1.5], ["v", "b", "web", 0.0]], dtype=object)
    users, items, y = check_interactions(X)
    assert users.tolist() == ["u", "v"]
    assert items.tolist() == ["a", "b"]
    np.testing.assert_array_equal(y, [1.0, 1.0])
    context = interaction_context(X, time=False)
    timed = interaction_context(X, time=True)
    assert context is not None
    assert timed is not None
    assert context.tolist() == [["mobile", 1.5], ["web", 0.0]]
    assert timed.tolist() == [[1.5], [0.0]]
    assert interaction_context(X[:, :2], time=False) is None


def test_interaction_context_gives_numbers_back_their_dtype():
    X = np.empty((2, 3), dtype=object)
    X[:, 0], X[:, 1], X[:, 2] = ["u", "v"], ["a", "b"], [1, 2]
    context = interaction_context(X, time=False)
    assert context is not None
    assert context.dtype.kind == "i"
    np.testing.assert_array_equal(context, [[1], [2]])
    assert check_interactions(np.array([[1, 2, 7]]))[0].dtype.kind == "i"


def test_check_interactions_with_time_and_context():
    X = np.array([["u", "a", 5, "web"]], dtype=object)
    *_, times = check_interactions(X, time=True)
    np.testing.assert_array_equal(times, [5])
    context = interaction_context(X, time=True)
    assert context is not None
    assert context.tolist() == [["web"]]


def test_check_queries_reads_a_vector_as_users_without_context():
    queries, context = check_queries(np.array(["u", "v"]))
    assert queries.tolist() == ["u", "v"]
    assert context is None
    queries, context = check_queries([1, 2])
    assert queries.tolist() == [1, 2]
    assert context is None


def test_check_queries_reads_a_matrix_as_users_and_context():
    X = np.array([[1, "web", 0.5], [2, "app", 1.0]], dtype=object)
    queries, context = check_queries(X)
    assert queries.dtype.kind == "i"
    assert queries.tolist() == [1, 2]
    assert context is not None
    assert context.tolist() == [["web", 0.5], ["app", 1.0]]
    numeric = check_queries([[1, 3], [2, 4]])[1]
    assert numeric is not None
    assert numeric.dtype.kind == "i"


def test_check_queries_reads_a_single_column_as_users():
    queries, context = check_queries([[1], [2]])
    assert queries.tolist() == [1, 2]
    assert context is None


@pytest.mark.parametrize("X", [np.zeros((2, 0)), np.zeros((2, 2, 2))])
def test_check_queries_rejects_other_shapes(X):
    with pytest.raises(ValueError):
        check_queries(X)


def test_check_interactions_gives_numeric_ids_back_their_dtype():
    X = np.empty((2, 3), dtype=object)
    X[:, 0], X[:, 1] = [1, 2], [3, 4]
    X[:, 2] = [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)]
    users, items, _, times = check_interactions(X, time=True)
    assert users.dtype.kind == "i"
    assert items.dtype.kind == "i"
    assert times.dtype == np.dtype("datetime64[ns]")


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        (np.array([1, 2]), np.int64),
        (np.array([1.5, 2.0]), np.float64),
        (np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]"), "datetime64[D]"),
        (np.array([datetime.datetime(2024, 1, 1), datetime.date(2024, 1, 2)]), "datetime64[ns]"),
        (np.array([np.datetime64("2024-01-01"), None], dtype=object), "datetime64[ns]"),
        (np.array([1, 2], dtype=object), np.int64),
        (np.array([1, None], dtype=object), np.float64),
        (np.array([1, np.nan], dtype=object), np.float64),
        (
            np.array([np.datetime64("2024-01-01", "D"), np.datetime64("NaT", "D")], dtype=object),
            "datetime64[ns]",
        ),
    ],
    ids=str,
)
def test_check_times_accepts_numbers_and_datetimes(values, dtype):
    assert check_times(values, allow_missing=True).dtype == np.dtype(dtype)


def test_check_times_accepts_pandas_timestamps():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {"user": ["u", "v"], "item": [1, 2], "time": pd.to_datetime(["2024-01-01", "2024-01-02"])}
    )
    *_, times = check_interactions(frame, time=True)
    assert times.dtype == np.dtype("datetime64[ns]")
    assert times[1] == np.datetime64("2024-01-02")


@pytest.mark.parametrize(
    "values",
    [np.array(["a", "b"]), np.array([True, False]), np.array([1, "a"], dtype=object)],
    ids=str,
)
def test_check_times_rejects_what_is_not_a_time(values):
    with pytest.raises(TypeError, match="numbers or datetimes"):
        check_times(values)


@pytest.mark.parametrize(
    "values",
    [np.array([1.0, np.nan]), np.array(["2024-01-01", "NaT"], dtype="datetime64[D]")],
    ids=str,
)
def test_check_times_rejects_missing_times_unless_allowed(values):
    with pytest.raises(ValueError, match="missing"):
        check_times(values)
    assert len(check_times(values, allow_missing=True)) == 2


def test_check_times_rejects_infinite_and_non_1d_times():
    with pytest.raises(ValueError, match="infinite"):
        check_times([1.0, np.inf])
    with pytest.raises(ValueError, match="one-dimensional"):
        check_times([[1, 2]])


def test_check_as_of_broadcasts_and_marks_latest_as_missing():
    numbers = np.dtype(np.int64)
    assert np.isnan(check_as_of(None, 3, numbers)).all()
    np.testing.assert_array_equal(check_as_of(5, 3, numbers), [5, 5, 5])
    np.testing.assert_array_equal(check_as_of([1, np.nan], 2, numbers), [1.0, np.nan])
    dates = np.dtype("datetime64[ns]")
    latest = check_as_of(None, 2, dates)
    assert latest.dtype == dates
    assert np.isnat(latest).all()
    assert check_as_of(np.datetime64("2024-01-01"), 1, dates).dtype == dates


def test_check_as_of_checks_length_and_kind():
    with pytest.raises(ValueError, match="one per query"):
        check_as_of([1, 2], 3, np.dtype(np.int64))
    with pytest.raises(TypeError, match="datetimes"):
        check_as_of(5, 1, np.dtype("datetime64[ns]"))
    with pytest.raises(TypeError, match="numbers"):
        check_as_of(np.datetime64("2024-01-01"), 1, np.dtype(np.int64))


def test_check_optionally_timed_reads_a_third_column_as_time():
    assert check_optionally_timed([["u", "a"]])[3] is None
    timed = np.array([["u", "a", 3]], dtype=object)
    np.testing.assert_array_equal(check_optionally_timed(timed)[3], [3])


def test_check_interactions_explains_times_turned_into_strings():
    with pytest.raises(TypeError, match="object array or a DataFrame"):
        check_interactions(np.array([["u", "a", 5]]), time=True)
    # A list keeps its numbers, see check_rows.
    np.testing.assert_array_equal(check_interactions([["u", "a", 5]], time=True)[3], [5])


def test_drop_time_leaves_untimed_identifiers_alone():
    X = np.array([["u", "a"], ["v", "b"]], dtype=object)
    assert drop_time(X) is not None
    np.testing.assert_array_equal(drop_time(X), X)
    mixed = np.array([["u", 1, 5]], dtype=object)
    assert drop_time(mixed).dtype == object


def test_a_list_mixing_numbers_and_strings_keeps_numeric_ids():
    users, items, _ = check_interactions([[1, 2, "web"], [3, 4, "app"]])
    assert users.dtype.kind == "i"
    assert items.dtype.kind == "i"
    queries, context = check_queries([[1, "web", 0], [3, "app", 1]])
    assert queries.dtype.kind == "i"
    assert context is not None
    assert context.tolist() == [["web", 0], ["app", 1]]
    # Two-column lists are left to numpy, as before.
    assert check_interactions([[1, "a"]])[0].dtype.kind == "U"


_BIG = 2**53


def test_a_list_mixing_integer_ids_and_float_context_keeps_them_integers():
    X = [[_BIG, 1, 0.5], [_BIG + 1, 2, 0.25]]
    users, items, _ = check_interactions(X)
    assert users.dtype.kind == "i"
    assert users.tolist() == [_BIG, _BIG + 1]
    assert items.dtype.kind == "i"
    context = interaction_context(check_rows(X), time=False)
    assert context is not None
    assert context.dtype == np.float64
    assert context.ravel().tolist() == [0.5, 0.25]
    queries, query_context = check_queries([[_BIG + 1, 0.5]])
    assert queries.tolist() == [_BIG + 1]
    assert query_context is not None
    assert query_context.dtype == np.float64


def test_a_dataframe_mixing_integer_ids_and_float_context_keeps_them_integers():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"user": [_BIG, _BIG + 1], "item": [1, 2], "price": [0.5, 0.25]})
    users, items, _ = check_interactions(X)
    assert users.tolist() == [_BIG, _BIG + 1]
    assert items.dtype.kind == "i"
    queries, context = check_queries(X[["user", "price"]])
    assert queries.tolist() == [_BIG, _BIG + 1]
    assert context is not None
    assert context.dtype == np.float64


def test_a_float_array_gives_whole_number_ids_back_as_integers():
    X = np.array([[3.0, 1.0, 0.5], [4.0, 2.0, 0.25]])
    users, items, _ = check_interactions(X)
    assert users.dtype.kind == "i"
    assert items.dtype.kind == "i"
    queries, context = check_queries(np.array([[3.0, 0.5]]))
    assert queries.dtype.kind == "i"
    assert context is not None
    assert context.dtype == np.float64
    # Identifiers that are not whole numbers are floats on purpose.
    assert check_interactions(np.array([[0.5, 1.0, 0.5]]))[0].dtype.kind == "f"


def test_a_float_array_rejects_ids_a_float_cannot_hold_exactly():
    with pytest.raises(ValueError, match="too large for a float"):
        check_interactions(np.array([[_BIG, 1.0, 0.5]]))
    with pytest.raises(ValueError, match="too large for a float"):
        check_queries(np.array([[_BIG, 0.5]]))


def test_float_context_does_not_merge_or_recast_ids():
    X = [[_BIG, 1, 0.5], [_BIG + 1, 2, 0.5], [_BIG + 1, 1, 0.5]]
    rec = MostPopularRecommender().fit(X)
    assert rec.n_users_ == 2
    items, _ = rec.recommend([[_BIG, 0.5]], n_recommendations=1)
    assert items.dtype.kind == "i"
    assert items.tolist() == [[2]]


def test_a_dataframe_of_integer_ids_and_datetimes_reads_as_timed_interactions():
    pd = pytest.importorskip("pandas")
    times = pd.to_datetime(["2024-01-01", "2024-01-02"])
    X = pd.DataFrame({"user": [1, 2], "item": [3, 4], "time": times, "price": [0.5, 0.25]})
    users, items, _, found = check_interactions(X, time=True)
    assert users.dtype.kind == "i"
    assert items.dtype.kind == "i"
    assert found.dtype.kind == "M"
    context = interaction_context(check_rows(X), time=True)
    assert context is not None
    assert context.dtype == np.float64


def test_times_keep_their_nanoseconds():
    pd = pytest.importorskip("pandas")
    stamp = "2024-01-01T00:00:00.000000001"
    X = pd.DataFrame(
        {"user": [1, 2], "item": [3, 4], "time": pd.to_datetime([stamp, stamp]), "price": [0.5, 1]}
    )
    assert check_interactions(X, time=True)[3][0] == np.datetime64(stamp)
    missing = np.array([pd.Timestamp(stamp), pd.NaT], dtype=object)
    found = check_times(missing, allow_missing=True)
    assert found[0] == np.datetime64(stamp)
    assert np.isnat(found[1])
    objects = np.array([[1, 3, pd.Timestamp(stamp)]], dtype=object)
    assert check_interactions(objects, time=True)[3][0] == np.datetime64(stamp)


def test_stack_columns_keeps_datetimes_exact():
    times = np.array(["2024-01-01T00:00:00.000000001", "NaT"], dtype="datetime64[ns]")
    stacked = stack_columns([np.array(["u", "v"]), times])
    assert stacked.dtype == object
    assert stacked[0, 1] == times[0]
    assert np.isnat(stacked[1, 1])
    assert check_times(stacked[:, 1], allow_missing=True)[0] == times[0]
