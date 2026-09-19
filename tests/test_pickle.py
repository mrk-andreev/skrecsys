"""Every model this library ships must survive being stored and loaded again.

The checks themselves are in `tests/estimator_checks.py`, and the two `test_common.py` modules
run them across every *parameter configuration*. What neither of those has is a list of
every *model*: one covers the classical estimators, the other the neural ones behind the
`nn` extra, and an estimator added to neither would be missed by both without anything
failing.

So this is the single place that enumerates them. A new recommender belongs in the list
below, and the checks it is held to are the shared ones rather than a second copy of the
same assertions.
"""

import pytest

from skrecsys import recommendation
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
from tests.estimator_checks import check_partial_fit_survives_a_pickle, check_pickle_round_trip

try:
    from skrecsys import nn
except ImportError:  # the `nn` extra is optional; see src/skrecsys/nn/__init__.py
    nn = None

#: One configuration of every estimator in `skrecsys.recommendation`, kept small and
#: seeded so that the comparison is of the model and not of a fresh draw.
CLASSICAL = [
    MostPopularRecommender(),
    ItemKNNRecommender(n_neighbors=3),
    BM25Recommender(n_neighbors=3),
    RP3Beta(n_neighbors=3),
    EASE(l2_reg=1.0),
    SLIMElasticNet(alpha=0.01),
    AlternatingLeastSquares(n_factors=2, n_iter=5, random_state=0),
    BayesianPersonalizedRanking(n_factors=4, max_iter=10, random_state=0),
]


def _neural():
    """The `skrecsys.nn` models, or nothing when the extra is not installed."""
    if nn is None:
        return []
    return [
        nn.SimpleX(n_factors=4, max_iter=3, n_negatives=2, random_state=0),
        nn.XSimGCL(n_factors=4, n_layers=2, max_iter=3, random_state=0),
        nn.HSTU(n_factors=4, max_sequence_length=4, max_iter=3, n_negatives=2, random_state=0),
        nn.Mamba4Rec(n_factors=4, max_sequence_length=4, max_iter=3, random_state=0),
    ]


MODELS = CLASSICAL + _neural()


def test_every_shipped_recommender_is_listed():
    """The list above has to keep up with the package, or it quietly stops covering it."""
    listed = {type(estimator).__name__ for estimator in MODELS}
    expected = set(recommendation.__all__)
    if nn is not None:
        expected |= set(nn.__all__)
    assert listed == expected, (
        f"tests/test_pickle.py does not cover {sorted(expected - listed)}; "
        f"it lists {sorted(listed - expected)} that the package does not export."
    )


@pytest.mark.parametrize("estimator", MODELS, ids=lambda e: type(e).__name__)
def test_a_fitted_model_pickles_and_answers_the_same(estimator):
    """Fit, predict, store, load, predict again: the answers must not move."""
    check_pickle_round_trip(type(estimator).__name__, estimator)


@pytest.mark.parametrize("estimator", MODELS, ids=lambda e: type(e).__name__)
def test_a_model_pickled_mid_stream_goes_on_training(estimator):
    """Fit a batch, store, load, feed the next batch: training has to pick up where it was."""
    check_partial_fit_survives_a_pickle(type(estimator).__name__, estimator)
