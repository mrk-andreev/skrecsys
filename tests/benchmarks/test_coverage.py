"""Runs in the regular suite: every implemented recommender must have a benchmark."""

import pytest

from skrecsys import recommendation

from .test_movielens_100k import BENCHMARKS


def _is_implemented(estimator_cls):
    try:
        estimator_cls().fit([["u", "i"]])
    except NotImplementedError:
        return False
    return True


@pytest.mark.parametrize("name", recommendation.__all__)
def test_implemented_recommender_is_benchmarked(name):
    if _is_implemented(getattr(recommendation, name)):
        assert name in BENCHMARKS, f"Add {name} to tests/benchmarks/test_movielens_100k.py."
