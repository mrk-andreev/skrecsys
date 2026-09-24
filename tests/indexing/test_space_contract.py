"""Every indexable estimator's score must be an inner product in the space it declares.

This is the load-bearing test of the whole index layer. A search is approximate; the
*space* is not, and a mistake there -- a bias folded in wrongly, a similarity matrix
indexed in the wrong orientation -- shows up as mediocre recall rather than as an error,
which is exactly the kind of bug a recall threshold is too blunt to catch.
"""

import numpy as np
import pytest
import scipy.sparse as sp

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

INDEXABLE = [
    ItemKNNRecommender(n_neighbors=20),
    BM25Recommender(n_neighbors=10),
    RP3Beta(n_neighbors=20),
    SLIMElasticNet(alpha=0.1, max_iter=20),
    EASE(l2_reg=50.0),
    AlternatingLeastSquares(n_factors=8, n_iter=5, random_state=0),
    BayesianPersonalizedRanking(n_factors=8, max_iter=10, random_state=0),
]


def _small(rng):
    users = rng.integers(0, 60, 1200)
    items = rng.integers(0, 90, 1200)
    return np.column_stack([users, items])


@pytest.mark.parametrize("estimator", INDEXABLE, ids=repr)
def test_the_declared_space_reproduces_the_exact_scores(estimator):
    rng = np.random.default_rng(0)
    X = _small(rng)
    y = rng.integers(1, 6, len(X)).astype(float)
    params = {k: v for k, v in estimator.get_params().items() if k != "index"}
    estimator = estimator.__class__(**params)
    estimator.fit(X, y if isinstance(estimator, AlternatingLeastSquares) else None)

    user_indices = np.arange(min(40, estimator.n_users_))
    space = estimator._index_space()
    queries = estimator._index_queries(user_indices)
    assert space.n_items == estimator.n_items_

    scores = queries @ space.items.T
    scores = np.asarray(scores.todense()) if sp.issparse(scores) else np.asarray(scores)
    offset = estimator._index_score_offset(user_indices)
    if offset is not None:
        scores = scores + offset[:, None]

    expected = estimator._score_users(user_indices, np.arange(estimator.n_items_))
    np.testing.assert_allclose(scores, expected, atol=1e-10)


def test_a_model_without_a_vector_space_does_not_offer_an_index():
    # Popularity does not depend on the query at all, so there is nothing for a
    # nearest-neighbour index to narrow. It therefore has no `index` parameter, which
    # refuses one earlier and more plainly than any message could.
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError, match="unexpected keyword argument 'index'"):
        MostPopularRecommender(index="hnsw").fit(_small(rng))  # ty: ignore[unknown-argument]


def test_an_index_on_a_model_with_no_space_is_refused_with_a_reason():
    # The guard behind the missing parameter, for anything that takes an `index` it
    # cannot honour -- a third-party estimator, or one of these before its hooks land.
    class Popularity(MostPopularRecommender):
        def __init__(self, index=None):
            super().__init__()
            self.index = index

    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="does not support index='hnsw'"):
        Popularity(index="hnsw").fit(_small(rng))


def test_a_model_without_an_index_parameter_is_unaffected():
    rng = np.random.default_rng(0)
    estimator = MostPopularRecommender().fit(_small(rng))
    assert estimator.index_ is None
