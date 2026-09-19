"""The checks every recommender in this repository is held to.

Recommenders take ``(n_interactions, 2)`` identifier arrays, which scikit-learn's
``check_estimator`` does not generate, so the contract is spelled out here: what
``recommend`` and ``predict`` must return, what ``partial_fit`` must do with a batch,
and what has to survive a pickle.

The two yielders are consumed by ``tests/recommendation/test_common.py`` and
``tests/nn/test_common.py``, which cross them with every parameter configuration, and by
``tests/test_pickle.py``, which runs the pickling pair over every model the library
ships. Parametrizing over the yielders rather than calling one entry point is what makes
a failure name the check that failed.

The incremental checks are yielded separately because ``partial_fit`` is an opt-in
capability: an estimator that does not define it is not thereby broken, and
``yield_recommender_checks`` has to stay runnable against every recommender.
"""

import pickle
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skrecsys.base import is_recommender

_USERS = ["u0", "u0", "u1", "u1", "u1", "u2", "u2", "u3", "u3", "u3"]
_ITEMS = ["i0", "i1", "i1", "i2", "i3", "i0", "i4", "i2", "i4", "i5"]

#: What counts as the same number once a batch has been through a sum and a kernel.
_TOLERANCE = 1e-12

#: A second batch for the incremental checks. It repeats a pair the first batch already
#: holds, names a user the model has never seen, and -- deliberately -- names a new item
#: that sorts *before* every fitted one, so the item codes handed out by the first call
#: all move. A batch whose new identifiers sort last would leave the permutation at the
#: identity and never exercise the relabelling at all.
_NEXT_USERS = ["u1", "u4", "u4", "u0", "u2"]
_NEXT_ITEMS = ["i2", "i5", "i0", "i-", "i-"]


def _interactions() -> np.ndarray:
    return np.column_stack([_USERS, _ITEMS]).astype(object)


def _next_interactions() -> np.ndarray:
    return np.column_stack([_NEXT_USERS, _NEXT_ITEMS]).astype(object)


def _both_interactions() -> np.ndarray:
    return np.concatenate([_interactions(), _next_interactions()])


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


def check_recommend_exclude_interactions(name: str, estimator: Any) -> None:
    """Excluded pairs act per query, like a candidate list written for that query alone.

    The pairs name a user outside the queries, an unknown user and an unknown item too,
    none of which may change anything, and ``u0`` is asked for twice, so both rows must
    carry its exclusion.
    """
    est = clone(estimator).fit(_interactions())
    queries = np.array(["u0", "u1", "u0"], dtype=object)
    pairs = np.array(
        [["u0", "i2"], ["u3", "i3"], ["u0", "unknown-item"], ["unknown-user", "i4"]],
        dtype=object,
    )
    for exclude_seen, kept, k in (
        (True, ["i3", "i4", "i5"], 2),
        (False, ["i0", "i1", "i3", "i4", "i5"], 4),
    ):
        items, scores = est.recommend(
            queries, n_recommendations=k, exclude_seen=exclude_seen, exclude_interactions=pairs
        )
        want_items, want_scores = est.recommend(
            np.array(["u0"], dtype=object),
            n_recommendations=k,
            exclude_seen=exclude_seen,
            candidates=np.array(kept, dtype=object),
        )
        other_items, other_scores = est.recommend(
            np.array(["u1"], dtype=object), n_recommendations=k, exclude_seen=exclude_seen
        )
        for row in (0, 2):
            np.testing.assert_array_equal(
                items[row], want_items[0], err_msg=f"{name} ignored exclude_interactions."
            )
            np.testing.assert_allclose(scores[row], want_scores[0], rtol=1e-9, atol=_TOLERANCE)
        np.testing.assert_array_equal(
            items[1], other_items[0], err_msg=f"{name} applied another user's exclusions."
        )
        np.testing.assert_allclose(scores[1], other_scores[0], rtol=1e-9, atol=_TOLERANCE)

    everything_unseen = np.array([["u0", i] for i in ("i2", "i3", "i4", "i5")], dtype=object)
    try:
        est.recommend(
            np.array(["u0"], dtype=object),
            n_recommendations=1,
            exclude_interactions=everything_unseen,
        )
    except ValueError:
        return
    raise AssertionError(f"{name} recommended an item it was told to exclude.")


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


def check_pandas_input_matches_numpy(name: str, estimator: Any) -> None:
    """DataFrame inputs produce the same scores and recommendations as arrays."""
    X = _interactions()
    # Two independent fits need the same random draws to isolate the input format.
    params = estimator.get_params(deep=False)
    if "random_state" in params and params["random_state"] is None:
        estimator = clone(estimator).set_params(random_state=0)
    array_est = clone(estimator).fit(X)
    frame_est = clone(estimator).fit(pd.DataFrame(X, columns=["user", "item"]))

    pairs = np.array([["u0", "i2"], ["u3", "i0"], ["u1", "i4"]], dtype=object)
    np.testing.assert_allclose(
        frame_est.predict(pd.DataFrame(pairs, columns=["user", "item"])),
        array_est.predict(pairs),
        err_msg=f"{name}.predict differs for DataFrame input.",
    )

    users = np.array(["u0", "u3", "u1"], dtype=object)
    array_items, array_scores = array_est.recommend(users, n_recommendations=3)
    frame_items, frame_scores = frame_est.recommend(pd.Series(users), n_recommendations=3)
    np.testing.assert_array_equal(
        frame_items, array_items, err_msg=f"{name}.recommend items differ for pandas input."
    )
    np.testing.assert_allclose(
        frame_scores, array_scores, err_msg=f"{name}.recommend scores differ for pandas input."
    )


def check_numeric_ids(name: str, estimator: Any) -> None:
    X = np.array([[10, 1], [10, 2], [20, 2], [20, 3], [30, 1]])
    est = clone(estimator).fit(X)
    items, _ = est.recommend(np.array([30]), n_recommendations=2)
    assert items.shape == (1, 2), f"{name} failed with numeric identifiers."
    assert 1 not in items[0]


def check_partial_fit_returns_self(name: str, estimator: Any) -> None:
    est = clone(estimator)
    assert est.partial_fit(_interactions()) is est, f"{name}.partial_fit must return self."
    assert est.partial_fit(_next_interactions()) is est


def check_partial_fit_first_call_equals_fit(name: str, estimator: Any) -> None:
    """The first call is `fit`, down to the recommendations it produces."""
    X = _interactions()
    incremental = clone(estimator).partial_fit(X)
    batch = clone(estimator).fit(X)
    np.testing.assert_array_equal(incremental.user_ids_, batch.user_ids_)
    np.testing.assert_array_equal(incremental.item_ids_, batch.item_ids_)
    queries = np.array(["u0", "u3"], dtype=object)
    items, scores = incremental.recommend(queries, n_recommendations=2)
    expected_items, expected_scores = batch.recommend(queries, n_recommendations=2)
    np.testing.assert_array_equal(items, expected_items, err_msg=f"{name}: first call differs.")
    np.testing.assert_allclose(scores, expected_scores)


def check_partial_fit_accumulates_interactions(name: str, estimator: Any) -> None:
    """The stored interactions are what one `fit` over every batch would have stored.

    The anchor of the incremental contract: whatever a model does with the batch, the
    history it keeps -- which is what `exclude_seen` and the similarity models score
    from -- must not depend on how the batches were cut.
    """
    est = clone(estimator).partial_fit(_interactions()).partial_fit(_next_interactions())
    batch = clone(estimator).fit(_both_interactions())
    np.testing.assert_array_equal(est.user_ids_, batch.user_ids_)
    np.testing.assert_array_equal(est.item_ids_, batch.item_ids_)
    assert (est.n_users_, est.n_items_) == (batch.n_users_, batch.n_items_)
    difference = abs(est.interactions_ - batch.interactions_)
    assert difference.nnz == 0 or difference.max() < _TOLERANCE, (
        f"{name}.partial_fit did not accumulate interactions the way fit does."
    )


def check_partial_fit_grows_vocabulary(name: str, estimator: Any) -> None:
    """New identifiers become queryable, and every fitted array grows with them."""
    est = clone(estimator).partial_fit(_interactions())
    before = (est.n_users_, est.n_items_)
    est.partial_fit(_next_interactions())
    assert est.n_users_ > before[0] and est.n_items_ > before[1], (
        f"{name}.partial_fit did not grow the vocabularies."
    )
    np.testing.assert_array_equal(est.user_ids_, np.unique(_USERS + _NEXT_USERS))
    np.testing.assert_array_equal(est.item_ids_, np.unique(_ITEMS + _NEXT_ITEMS))
    assert np.all(est.item_ids_[:-1] < est.item_ids_[1:]), f"{name}.item_ids_ is not sorted."

    # Scoring the whole catalog for every fitted user reaches every row and column of
    # every fitted array, which is what catches a model that grew some of them and not
    # others -- a state that stays plausible until something indexes the part that was
    # left behind.
    everyone = est.user_ids_
    items, scores = est.recommend(everyone, n_recommendations=est.n_items_, exclude_seen=False)
    assert items.shape == (len(everyone), est.n_items_)
    assert np.all(np.isfinite(scores)), f"{name} scores a fitted pair as non-finite."

    items, _ = est.recommend(np.array(["u4", "u0"], dtype=object), n_recommendations=2)
    assert items.shape == (2, 2), f"{name} cannot recommend to a user added by partial_fit."
    scores = est.predict(np.array([["u4", "i-"], ["u0", "i1"]], dtype=object))
    assert scores.shape == (2,) and np.all(np.isfinite(scores))


def check_partial_fit_scores_consistently(name: str, estimator: Any) -> None:
    """After a relabelling batch, `recommend` and `predict` still agree.

    Relabelling touches every fitted array, and a model that grew only some of them
    stays superficially usable: the shapes line up, the calls return. What does not
    survive is the agreement between the two ways of scoring the same pair, because one
    reads the interactions and the other the model.
    """
    est = clone(estimator).partial_fit(_interactions()).partial_fit(_next_interactions())
    queries = np.array(["u0", "u4"], dtype=object)
    items, scores = est.recommend(queries, n_recommendations=3, exclude_seen=False)
    pairs = np.column_stack([np.repeat(queries, 3), items.ravel()])
    np.testing.assert_allclose(
        est.predict(pairs),
        scores.ravel(),
        rtol=1e-9,
        atol=_TOLERANCE,
        err_msg=f"{name}.predict and .recommend disagree after partial_fit.",
    )


def check_partial_fit_after_fit(name: str, estimator: Any) -> None:
    """A model fitted in one go can still be brought up to date with a batch.

    ``fit`` and ``partial_fit`` are one interface, not two: whatever an estimator keeps
    in order to continue has to be kept by ``fit`` as well, or the first call would
    quietly decide whether there could ever be a second. What is asserted is the
    stronger form -- the two routes to the same data reach the same model -- because an
    estimator can satisfy the weaker one by starting the second call over from scratch.
    """
    est = clone(estimator).fit(_interactions())
    est.partial_fit(_next_interactions())
    np.testing.assert_array_equal(est.item_ids_, np.unique(_ITEMS + _NEXT_ITEMS))
    items, _ = est.recommend(np.array(["u4"], dtype=object), n_recommendations=2)
    assert items.shape == (1, 2), f"{name} cannot recommend after fit followed by partial_fit."

    batched = clone(estimator).partial_fit(_interactions())
    batched.partial_fit(_next_interactions())
    queries = est.user_ids_
    expected_items, expected_scores = batched.recommend(queries, n_recommendations=2)
    got_items, got_scores = est.recommend(queries, n_recommendations=2)
    np.testing.assert_array_equal(
        got_items,
        expected_items,
        err_msg=f"{name}: fit then partial_fit took a different path from two batches.",
    )
    np.testing.assert_allclose(got_scores, expected_scores, rtol=1e-9, atol=_TOLERANCE)


def check_pickle_round_trip(name: str, estimator: Any) -> None:
    """A fitted model must answer identically after a pickle round trip.

    Persisting a model and loading it somewhere else is the ordinary way one gets used,
    so what has to survive is not merely the attributes but the answers. Both are
    checked, because an estimator can keep every array and still change its mind: a
    memoized transpose, a lazily derived matrix or an index rebuilt on the wrong side of
    a round trip would all show up here and nowhere else.

    The tolerance is nominal. Pickling float64 is exact and nothing is re-derived on the
    way back in, so a difference at all is a defect rather than drift.
    """
    est = clone(estimator).fit(_interactions())
    queries = np.array(["u0", "u3", "u1"], dtype=object)
    pairs = np.array([["u0", "i2"], ["u3", "i0"], ["u1", "i1"]], dtype=object)
    scores = est.predict(pairs)
    items, ranked = est.recommend(queries, n_recommendations=3)

    restored = pickle.loads(pickle.dumps(est))
    check_is_fitted(restored)
    np.testing.assert_array_equal(restored.user_ids_, est.user_ids_)
    np.testing.assert_array_equal(restored.item_ids_, est.item_ids_)
    assert (restored.n_users_, restored.n_items_) == (est.n_users_, est.n_items_)

    np.testing.assert_allclose(
        restored.predict(pairs),
        scores,
        rtol=0,
        atol=_TOLERANCE,
        err_msg=f"{name}.predict changed across a pickle round trip.",
    )
    restored_items, restored_ranked = restored.recommend(queries, n_recommendations=3)
    np.testing.assert_array_equal(
        restored_items,
        items,
        err_msg=f"{name}.recommend returned different items across a pickle round trip.",
    )
    np.testing.assert_allclose(
        restored_ranked,
        ranked,
        rtol=0,
        atol=_TOLERANCE,
        err_msg=f"{name}.recommend returned different scores across a pickle round trip.",
    )


def check_partial_fit_survives_a_pickle(name: str, estimator: Any) -> None:
    """A model stored mid-stream must reload and carry on from the next batch.

    This is the shape the feature is for -- fit today, persist, reload tomorrow, take the
    next batch -- and it is the one an estimator can break while passing every other
    check here, by keeping the state that continues training somewhere that does not
    survive being written to disk.
    """
    est = clone(estimator).partial_fit(_interactions())
    restored = pickle.loads(pickle.dumps(est))
    restored.partial_fit(_next_interactions())

    straight = clone(estimator).partial_fit(_interactions())
    straight.partial_fit(_next_interactions())
    np.testing.assert_array_equal(restored.item_ids_, straight.item_ids_)
    queries = straight.user_ids_
    expected_items, expected_scores = straight.recommend(queries, n_recommendations=2)
    got_items, got_scores = restored.recommend(queries, n_recommendations=2)
    np.testing.assert_array_equal(
        got_items,
        expected_items,
        err_msg=f"{name} trained differently after being reloaded from a pickle.",
    )
    np.testing.assert_allclose(got_scores, expected_scores, rtol=1e-9, atol=_TOLERANCE)


_CHECKS: tuple[Callable[[str, Any], None], ...] = (
    check_is_recommender,
    check_fit_attributes,
    check_recommend_output,
    check_recommend_exclude_seen,
    check_recommend_candidates,
    check_recommend_exclude_interactions,
    check_recommend_errors,
    check_predict,
    check_numeric_ids,
    check_pickle_round_trip,
)

_INCREMENTAL_CHECKS: tuple[Callable[[str, Any], None], ...] = (
    check_partial_fit_returns_self,
    check_partial_fit_first_call_equals_fit,
    check_partial_fit_accumulates_interactions,
    check_partial_fit_grows_vocabulary,
    check_partial_fit_scores_consistently,
    check_partial_fit_after_fit,
    check_partial_fit_survives_a_pickle,
)


def yield_recommender_checks() -> Iterator[Callable[[str, Any], None]]:
    """Yield the common recommender checks, each called as ``check(name, estimator)``."""
    yield from _CHECKS


def yield_incremental_recommender_checks() -> Iterator[Callable[[str, Any], None]]:
    """Yield the checks for ``partial_fit``, each called as ``check(name, estimator)``."""
    yield from _INCREMENTAL_CHECKS
