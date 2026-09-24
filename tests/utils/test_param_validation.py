"""The parameter checks every estimator shares."""

import numpy as np
import pytest

from skrecsys import is_recommender
from skrecsys.recommendation import MostPopularRecommender
from skrecsys.utils._param_validation import (
    check_bool,
    check_component,
    check_int,
    check_real,
    resolve_n_jobs,
)


def test_check_component():
    check_component(MostPopularRecommender(), "on_true", is_recommender, "a recommender")
    with pytest.raises(TypeError, match="on_true must be a recommender, got str"):
        check_component("popular", "on_true", is_recommender, "a recommender")


def test_check_bool():
    for value in (True, False):
        assert check_bool(value, "flag") is value
    for value in (1, 0, "yes", None, np.bool_(1)):
        with pytest.raises(ValueError, match="flag must be a bool"):
            check_bool(value, "flag")


@pytest.mark.parametrize("value", [1, 5, np.int64(3)])
def test_check_int_accepts(value):
    result = check_int(value, "k", min_value=1)
    assert result == value
    assert type(result) is int


@pytest.mark.parametrize("value", [0, -1, 1.0, 2.5, "3", None, True, False])
def test_check_int_rejects(value):
    with pytest.raises(ValueError, match=r"k must be an integer >= 1, got"):
        check_int(value, "k", min_value=1)


def test_check_int_bounds_and_none():
    assert check_int(3, "k", min_value=0, max_value=3) == 3
    with pytest.raises(ValueError, match=r"k must be an integer in \[0, 3\], got 4"):
        check_int(4, "k", min_value=0, max_value=3)
    assert check_int(None, "k", min_value=1, allow_none=True) is None
    with pytest.raises(ValueError, match="k must be None or an integer >= 1"):
        check_int(0, "k", min_value=1, allow_none=True)


@pytest.mark.parametrize("value", [0, 0.5, 1, np.float32(0.25), np.int64(1)])
def test_check_real_accepts(value):
    result = check_real(value, "x", min_value=0, max_value=1)
    assert result == float(value)
    assert type(result) is float


@pytest.mark.parametrize("value", [-0.1, 1.5, np.nan, np.inf, "0.5", None, True])
def test_check_real_rejects(value):
    with pytest.raises(ValueError, match=r"x must be a real number in \[0, 1\], got"):
        check_real(value, "x", min_value=0, max_value=1)


@pytest.mark.parametrize(
    ("bounds", "value", "message"),
    [
        ({"min_value": 0, "min_inclusive": False}, 0, "x must be a finite real number > 0"),
        ({"min_value": 0}, -1, "x must be a finite real number >= 0"),
        ({"max_value": 1, "max_inclusive": False}, 1, "x must be a finite real number < 1"),
        (
            {"min_value": 0, "max_value": 0.5, "max_inclusive": False},
            0.5,
            r"x must be a real number in \[0, 0.5\)",
        ),
        ({}, np.inf, "x must be a finite real number, got inf"),
    ],
)
def test_check_real_bounds(bounds, value, message):
    with pytest.raises(ValueError, match=message):
        check_real(value, "x", **bounds)


def test_resolve_n_jobs():
    assert resolve_n_jobs(None) == 0
    assert resolve_n_jobs(-1) == 0
    assert resolve_n_jobs(4) == 4
    assert resolve_n_jobs(np.int64(2)) == 2
    for value in (0, -2, 1.5, "4", True):
        with pytest.raises(ValueError, match="n_jobs must be None, -1 or an integer >= 1"):
            resolve_n_jobs(value)
