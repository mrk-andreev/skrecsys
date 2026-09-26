"""The composites are recommenders, and are held to the same contract as every other."""

import pytest

from skrecsys.compose import Cascade
from tests.compose._composites import COMPOSITES
from tests.estimator_checks import (
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
