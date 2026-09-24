"""The protocols of ``skrecsys._typing`` against the estimators the package ships.

Each case list is annotated with its protocol, so the lists are what ``ty`` checks: an
estimator that drifts from its protocol stops type-checking here, before a composite
that takes it as a parameter notices. The tests tie that static contract to the runtime
tags the composites check with, one case per estimator.
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC

from skrecsys._typing import (
    Condition,
    DecisionClassifier,
    Features,
    PointwiseEstimator,
    ProbabilisticClassifier,
    Ranker,
    Recommender,
    Regressor,
)
from skrecsys.base import is_condition, is_features, is_ranker, is_recommender
from skrecsys.compose import (
    Cascade,
    ConcatFeatures,
    GeneratorScores,
    GroupRanker,
    JoinDynamicFeatures,
    JoinStaticFeatures,
    KnownUser,
    MinInteractions,
    PointwiseRanker,
    QueryIn,
    Switch,
)
from skrecsys.recommendation import (
    EASE,
    AlternatingLeastSquares,
    BayesianPersonalizedRanking,
    BM25Recommender,
    ItemKNNRecommender,
    MostPopularRecommender,
    RP3Beta,
    SLIMElasticNet,
)

RECOMMENDERS: list[Recommender] = [
    AlternatingLeastSquares(),
    BayesianPersonalizedRanking(),
    BM25Recommender(),
    EASE(),
    ItemKNNRecommender(),
    MostPopularRecommender(),
    RP3Beta(),
    SLIMElasticNet(),
    Switch(KnownUser(), ItemKNNRecommender(), MostPopularRecommender()),
    Cascade(MostPopularRecommender(), GeneratorScores(), PointwiseRanker(LinearRegression())),
]

CONDITIONS: list[Condition] = [
    KnownUser(),
    MinInteractions(),
    QueryIn(["u1"]),
    ~KnownUser(),
    KnownUser() & MinInteractions(),
    KnownUser() | QueryIn(["u1"]),
]

FEATURES: list[Features] = [
    JoinStaticFeatures("item", np.array([["a", 1.0]], dtype=object)),
    JoinDynamicFeatures("user", lambda ids: np.zeros(len(ids))),
    GeneratorScores(),
    ConcatFeatures([GeneratorScores()]),
]

RANKERS: list[Ranker] = [
    PointwiseRanker(LogisticRegression()),
    GroupRanker(LinearRegression(), group_param="sample_weight"),
]

#: Each scikit-learn estimator with the branch of ``PointwiseRanker.predict`` it takes.
POINTWISE: list[tuple[PointwiseEstimator, type]] = [
    (LogisticRegression(), ProbabilisticClassifier),
    (LinearSVC(), DecisionClassifier),
    (LinearRegression(), Regressor),
]


def _name(estimator: object) -> str:
    return type(estimator).__name__


@pytest.mark.parametrize("recommender", RECOMMENDERS, ids=_name)
def test_recommenders_satisfy_recommender(recommender):
    assert is_recommender(recommender)


@pytest.mark.parametrize("condition", CONDITIONS, ids=_name)
def test_conditions_satisfy_condition(condition):
    assert is_condition(condition)


@pytest.mark.parametrize("component", FEATURES, ids=_name)
def test_feature_components_satisfy_features(component):
    assert is_features(component)


@pytest.mark.parametrize("ranker", RANKERS, ids=_name)
def test_rankers_satisfy_ranker(ranker):
    assert is_ranker(ranker)


@pytest.mark.parametrize(("estimator", "protocol"), POINTWISE, ids=[_name(e) for e, _ in POINTWISE])
def test_scikit_learn_estimators_are_pointwise_estimators(estimator, protocol):
    assert isinstance(estimator, protocol)


def test_a_classifier_without_probabilities_is_scored_by_its_decision_function():
    assert not isinstance(LinearSVC(), ProbabilisticClassifier)
