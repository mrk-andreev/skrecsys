"""Common checks for recommender estimators.

Recommenders take ``(n_interactions, 2)`` identifier arrays, which scikit-learn's
``check_estimator`` does not generate. These checks cover the recommender contract.
"""

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from sklearn.base import clone

from skrecsys.base import is_recommender

__all__ = ["check_recommender", "yield_recommender_checks"]

_USERS = ["u0", "u0", "u1", "u1", "u1", "u2", "u2", "u3", "u3", "u3"]
_ITEMS = ["i0", "i1", "i1", "i2", "i3", "i0", "i4", "i2", "i4", "i5"]


def _interactions() -> np.ndarray:
    return np.column_stack([_USERS, _ITEMS]).astype(object)


def check_is_recommender(name: str, estimator: Any) -> None:
    assert is_recommender(estimator), f"{name} is not tagged as a recommender."
    assert not hasattr(estimator, "score"), f"{name} must not define a default score."


def check_fit_attributes(name: str, estimator: Any) -> None:
    X = _interactions()
    est = clone(estimator)
    assert est.fit(X) is est, f"{name}.fit must return self."
    np.testing.assert_array_equal(est.user_ids_, np.unique(_USERS))
    np.testing.assert_array_equal(est.item_ids_, np.unique(_ITEMS))
    assert est.n_users_ == len(set(_USERS))
    assert est.n_items_ == len(set(_ITEMS))


def check_recommend_output(name: str, estimator: Any) -> None:
    X = _interactions()
    est = clone(estimator).fit(X)
    queries = np.array(["u0", "u3", "u1"], dtype=object)
    items, scores = est.recommend(queries, n_recommendations=3)
    assert items.shape == (3, 3), f"{name}.recommend returned items of shape {items.shape}."
    assert scores.shape == (3, 3), f"{name}.recommend returned scores of shape {scores.shape}."
    assert np.all(np.isin(items, est.item_ids_))
    assert np.all(np.diff(scores, axis=1) <= 0), f"{name} scores are not descending."
    for row in items:
        assert len(set(row)) == len(row), f"{name} recommended duplicate items."

    items_again, scores_again = est.recommend(queries, n_recommendations=3)
    np.testing.assert_array_equal(items, items_again)
    np.testing.assert_array_equal(scores, scores_again)

    positions = np.searchsorted(est.item_ids_, items)
    ties = np.diff(scores, axis=1) == 0
    assert np.all(np.diff(positions, axis=1)[ties] > 0), (
        f"{name} does not break ties by fitted item order."
    )


def check_recommend_exclude_seen(name: str, estimator: Any) -> None:
    est = clone(estimator).fit(_interactions())
    seen = {u: set() for u in _USERS}
    for user, item in zip(_USERS, _ITEMS, strict=True):
        seen[user].add(item)
    users = np.array(sorted(seen), dtype=object)
    items, _ = est.recommend(users, n_recommendations=3, exclude_seen=True)
    for user, row in zip(users, items, strict=True):
        assert not seen[user] & set(row), f"{name} recommended seen items to {user}."
    items, _ = est.recommend(users, n_recommendations=est.n_items_, exclude_seen=False)
    assert items.shape == (len(users), est.n_items_)


def check_recommend_candidates(name: str, estimator: Any) -> None:
    est = clone(estimator).fit(_interactions())
    candidates = np.array(["i5", "i3", "i0"], dtype=object)
    items, _ = est.recommend(
        np.array(["u0"], dtype=object), n_recommendations=2, candidates=candidates
    )
    assert set(items[0]) <= set(candidates), f"{name} ignored candidates."


def check_recommend_errors(name: str, estimator: Any) -> None:
    est = clone(estimator).fit(_interactions())
    for kwargs, query in (
        ({"n_recommendations": 1}, ["unknown-user"]),
        ({"n_recommendations": est.n_items_}, ["u0"]),
        ({"n_recommendations": 1, "candidates": ["unknown-item"]}, ["u0"]),
    ):
        try:
            est.recommend(np.array(query, dtype=object), **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{name}.recommend({query}, **{kwargs}) did not raise ValueError.")


def check_predict(name: str, estimator: Any) -> None:
    est = clone(estimator).fit(_interactions())
    pairs = np.array([["u0", "i2"], ["u3", "i0"], ["u0", "i2"]], dtype=object)
    scores = est.predict(pairs)
    assert scores.shape == (3,), f"{name}.predict returned shape {scores.shape}."
    assert scores[0] == scores[2]
    try:
        est.predict(np.array([["u0", "unknown-item"]], dtype=object))
    except ValueError:
        return
    raise AssertionError(f"{name}.predict did not raise ValueError for an unknown item.")


def check_numeric_ids(name: str, estimator: Any) -> None:
    X = np.array([[10, 1], [10, 2], [20, 2], [20, 3], [30, 1]])
    est = clone(estimator).fit(X)
    items, _ = est.recommend(np.array([30]), n_recommendations=2)
    assert items.shape == (1, 2), f"{name} failed with numeric identifiers."
    assert 1 not in items[0]


_CHECKS: tuple[Callable[[str, Any], None], ...] = (
    check_is_recommender,
    check_fit_attributes,
    check_recommend_output,
    check_recommend_exclude_seen,
    check_recommend_candidates,
    check_recommend_errors,
    check_predict,
    check_numeric_ids,
)


def yield_recommender_checks() -> Iterator[Callable[[str, Any], None]]:
    """Yield the common recommender checks, each called as ``check(name, estimator)``."""
    yield from _CHECKS


def check_recommender(estimator: Any) -> None:
    """Run all common recommender checks on an unfitted estimator instance."""
    name = type(estimator).__name__
    for check in yield_recommender_checks():
        check(name, estimator)
