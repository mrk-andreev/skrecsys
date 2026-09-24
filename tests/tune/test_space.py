from typing import Annotated

import pytest
from sklearn.base import BaseEstimator

from skrecsys.compose import KnownUser, Switch
from skrecsys.recommendation import (
    BM25Recommender,
    ItemKNNRecommender,
    MostPopularRecommender,
)
from skrecsys.tune import Categorical, Float, Int, search_space


def test_bm25_declares_its_space():
    assert search_space(BM25Recommender()) == {
        "n_neighbors": Int(5, 1000, log=True),
        "k1": Float(0.05, 5.0, log=True),
        "b": Float(0.0, 1.0),
    }


def test_the_defaults_lie_inside_the_declared_space():
    for estimator in (BM25Recommender(), ItemKNNRecommender()):
        params = estimator.get_params()
        for name, dist in search_space(estimator).items():
            assert dist.contains(params[name]), (type(estimator).__name__, name)


def test_nested_estimators_are_prefixed():
    switch = Switch(KnownUser(), BM25Recommender(), MostPopularRecommender())
    space = search_space(switch)
    assert set(space) == {"on_true__n_neighbors", "on_true__k1", "on_true__b"}
    assert set(space) <= set(switch.get_params(deep=True))


def test_an_unannotated_estimator_has_an_empty_space():
    assert search_space(MostPopularRecommender()) == {}


def test_other_annotation_metadata_is_ignored():
    class Model(BaseEstimator):
        def __init__(
            self,
            a: Annotated[int, "doc", Int(1, 3)] = 1,
            b: Annotated[str, "doc"] = "x",
            c: float = 0.0,
        ) -> None:
            self.a, self.b, self.c = a, b, c

    assert search_space(Model()) == {"a": Int(1, 3)}


@pytest.mark.parametrize(
    ("make", "error"),
    [
        (lambda: Float(1.0, 0.0), ValueError),
        (lambda: Float(0.0, float("inf")), ValueError),
        (lambda: Float(0.0, 1.0, log=True), ValueError),
        (lambda: Int(0, 10, log=True), ValueError),
        (lambda: Int(0.5, 10), TypeError),  # ty: ignore[invalid-argument-type]
        (lambda: Categorical([]), ValueError),
    ],
)
def test_invalid_distributions_raise(make, error):
    with pytest.raises(error):
        make()


def test_contains():
    assert Float(0, 1).contains(0.5)
    assert not Float(0, 1).contains(value=True)
    assert not Float(0, 1).contains(2)
    assert Int(1, 5).contains(5)
    assert not Int(1, 5).contains(2.5)
    assert Categorical(["a", None]).contains(None)
    assert Categorical(["a", None]).index(None) == 1
