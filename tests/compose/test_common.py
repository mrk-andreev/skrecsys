"""The composites are recommenders, and are held to the same contract as every other."""

import numpy as np
import pytest
from sklearn.base import clone

from skrecsys.compose import Cascade
from tests.compose._composites import COMPOSITES
from tests.estimator_checks import (
    _interactions,
    check_numeric_ids,
    check_pandas_input_matches_numpy,
    yield_recommender_checks,
)


def _has_cascade(estimator):
    if isinstance(estimator, Cascade):
        return True
    return any(
        _has_cascade(v)
        for v in estimator.get_params(deep=False).values()
        if hasattr(v, "get_params")
    )


@pytest.mark.parametrize("estimator", COMPOSITES, ids=repr)
@pytest.mark.parametrize("check", list(yield_recommender_checks()), ids=lambda c: c.__name__)
def test_common_checks(estimator, check):
    if check is check_numeric_ids and _has_cascade(estimator):
        # Five interactions over three items leave a cascade's ranker only relevant
        # candidates to learn from, which it rightly refuses; numeric identifiers are
        # covered on a catalog large enough in test_cascade.py.
        pytest.skip("the fixture is too small to train a ranker on")
    check(type(estimator).__name__, estimator)


@pytest.mark.parametrize("estimator", COMPOSITES, ids=repr)
def test_pandas_input_matches_numpy(estimator):
    check_pandas_input_matches_numpy(type(estimator).__name__, estimator)


@pytest.mark.parametrize("estimator", COMPOSITES, ids=repr)
def test_context_columns_are_accepted(estimator):
    """None of these read context, so it changes nothing, in fit or in the queries."""
    X = _interactions()
    context = np.where(np.arange(len(X)) % 2, "web", "app")
    plain = clone(estimator).fit(X)
    contextual = clone(estimator).fit(np.column_stack([X, context]))
    users = np.unique(X[:, 0])
    queries = np.column_stack([users, np.full(len(users), "web")])
    want = plain.recommend(users, n_recommendations=1)
    for got in (
        contextual.recommend(users, n_recommendations=1),
        contextual.recommend(queries, n_recommendations=1),
    ):
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_allclose(got[1], want[1])
