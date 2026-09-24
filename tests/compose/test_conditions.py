import numpy as np
import pytest
from sklearn.base import clone

from skrecsys.base import is_condition
from skrecsys.compose import AllOf, AnyOf, KnownUser, MinInteractions, Not, QueryIn

X = np.array([["u1", "a"], ["u1", "b"], ["u1", "b"], ["u2", "a"]], dtype=object)
QUERIES = np.array(["u1", "u2", "new"], dtype=object)


def test_known_user():
    assert KnownUser().fit(X).evaluate(QUERIES).tolist() == [True, True, False]


def test_min_interactions_counts_rows_and_gives_unknown_users_none():
    condition = MinInteractions(3).fit(X)
    assert condition.evaluate(QUERIES).tolist() == [True, False, False]
    assert MinInteractions(0).fit(X).evaluate(QUERIES).tolist() == [True, True, True]


@pytest.mark.parametrize("n", [-1, 1.5, True, "3"])
def test_min_interactions_rejects_bad_counts(n):
    with pytest.raises(ValueError, match="n_interactions"):
        MinInteractions(n).fit(X)


def test_query_in_needs_no_fitted_users():
    condition = QueryIn(["new", "u2"]).fit(X)
    assert condition.evaluate(QUERIES).tolist() == [False, True, True]
    assert QueryIn().fit(X).evaluate(QUERIES).tolist() == [False, False, False]


def test_operators_build_combinators():
    known, heavy = KnownUser(), MinInteractions(3)
    assert isinstance(~known, Not)
    assert isinstance(known & heavy, AllOf)
    assert isinstance(known | heavy, AnyOf)
    assert (~known).fit(X).evaluate(QUERIES).tolist() == [False, False, True]
    assert (known & ~heavy).fit(X).evaluate(QUERIES).tolist() == [False, True, False]
    assert (~known | heavy).fit(X).evaluate(QUERIES).tolist() == [True, False, True]


def test_combinators_clone_and_expose_nested_params():
    inner = MinInteractions(3)
    condition = Not(inner)
    assert condition.get_params()["condition__n_interactions"] == 3
    copy = clone(condition).set_params(condition__n_interactions=1)
    assert copy.fit(X).evaluate(QUERIES).tolist() == [False, False, True]
    assert inner.n_interactions == 3


def test_combinators_reject_non_conditions():
    with pytest.raises(TypeError, match="not a condition"):
        Not("nope").fit(X)  # ty: ignore[invalid-argument-type]


def test_tags():
    assert is_condition(KnownUser())
    assert is_condition(~KnownUser())
