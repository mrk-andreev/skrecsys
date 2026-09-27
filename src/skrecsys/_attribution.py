"""Why a recommender scored a user-item pair as it did, in terms of the user's history.

Nothing here is public: :mod:`skrecsys.inspection` builds on it, and the estimators'
``_attribute`` hooks call it. Like :mod:`skrecsys._tracing`, it sits below
:mod:`skrecsys.base` in the import graph.

Three kinds of answer, one per family of recommender:

``"history"``
    Exact. The score of an item-to-item model -- ItemKNN, BM25, RP3Beta, SLIM, EASE -- is
    a sum over the user's history, ``score(u, i) = sum_j x[u, j] * W[j, i]``, so each
    history item's term *is* its share of the score, and the terms add up to it.
``"similar_history"``
    Approximate. A factor model scores by a dot product that no history item owns; the
    reasons given are the history items whose vectors point the most like the target's,
    weighted by ``x[u, j] * cos(v_j, v_i)``. They do not add up to the score.
``"popularity"``
    The score is the item's popularity; no history item plays a part.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

#: ``weight(history_items, target_items)``: what one interaction with each history item
#: contributes to the score of the aligned target item, as fitted positions.
PairWeight = Callable[[NDArray[np.intp], NDArray[np.intp]], NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class Attributions:
    """The reasons behind the scores of aligned user-item pairs, one row per pair.

    Attributes
    ----------
    kind : {"history", "similar_history", "popularity"}
        See the module docstring.
    exact : bool
        Whether ``weights`` are shares of the score that add up to it with ``rest``.
    items : ndarray of object of shape (n_pairs, n_reasons)
        The history items behind each pair, the largest ``|weight|`` first; ``None``
        where a pair has fewer.
    weights : ndarray of shape (n_pairs, n_reasons)
        Their weights; NaN where ``items`` is ``None``.
    rest : ndarray of shape (n_pairs,)
        What the listed items leave of the score -- the rest of the history, or the
        popularity itself -- when ``exact``; NaN otherwise.
    """

    kind: str
    exact: bool
    items: NDArray[np.object_]
    weights: NDArray[np.float64]
    rest: NDArray[np.float64]


def history_attributions(
    history: sp.csr_array,
    rows: NDArray[np.intp],
    targets: NDArray[np.intp],
    weight: PairWeight,
    item_ids: NDArray[np.generic],
    n_reasons: int,
    *,
    kind: str,
    exact: bool,
) -> Attributions:
    """Attribute each pair ``(rows[p], targets[p])`` to the history row ``rows[p]``.

    ``history`` holds the interaction values ``x``; each stored ``x[u, j]`` contributes
    ``x[u, j] * weight(j, i)`` to target ``i``. Vectorized over every stored interaction
    of every pair, so a pair costs its user's history length.
    """
    block = sp.csr_array(history[rows])
    n_pairs = len(rows)
    owner = np.repeat(np.arange(n_pairs), np.diff(block.indptr))
    history_items = block.indices.astype(np.intp)
    terms = block.data * weight(history_items, targets[owner])
    total = np.bincount(owner, weights=terms, minlength=n_pairs)
    # A zero term explains nothing; nor, approximately, does the target resembling itself.
    keep = terms != 0
    if not exact:
        keep &= history_items != targets[owner]
    owner, history_items, terms = owner[keep], history_items[keep], terms[keep]

    # Each pair's terms, the largest magnitude first; ties by fitted item order.
    order = np.lexsort((history_items, -np.abs(terms), owner))
    counts = np.bincount(owner, minlength=n_pairs)
    starts = np.cumsum(counts) - counts
    position = np.arange(len(order)) - starts[owner[order]]
    shown = order[position < n_reasons]
    column = position[position < n_reasons]

    items = np.full((n_pairs, n_reasons), None, dtype=object)
    weights = np.full((n_pairs, n_reasons), np.nan)
    items[owner[shown], column] = item_ids[history_items[shown]]
    weights[owner[shown], column] = terms[shown]
    if exact:
        rest = total - np.bincount(owner[shown], weights=terms[shown], minlength=n_pairs)
    else:
        rest = np.full(n_pairs, np.nan)
    return Attributions(kind, exact, items, weights, rest)


def popularity_attributions(scores: NDArray[np.float64]) -> Attributions:
    """Pairs scored by their item's popularity alone: all of the score is ``rest``."""
    n_pairs = len(scores)
    return Attributions(
        "popularity",
        exact=True,
        items=np.empty((n_pairs, 0), dtype=object),
        weights=np.empty((n_pairs, 0)),
        rest=np.asarray(scores, dtype=np.float64),
    )


def cosine_weight(vectors: NDArray[np.floating]) -> PairWeight:
    """``weight(j, i)`` as the cosine of item vectors ``j`` and ``i``; 0 for a zero one."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, norms, out=np.zeros_like(vectors, dtype=np.float64), where=norms > 0)

    def weight(history_items: NDArray[np.intp], targets: NDArray[np.intp]) -> NDArray[np.float64]:
        return np.einsum("ij,ij->i", unit[history_items], unit[targets])

    return weight


def ranker_contributions(ranker: object, X: NDArray[np.floating]) -> NDArray[np.float64] | None:
    """Per-feature contributions of a ranker to each row's score, or None if it has none.

    The last column is the part no feature owns: a bias or an expected value. See the
    ``_contributions`` hook of :class:`skrecsys.base.RankerMixin`.
    """
    contributions = getattr(ranker, "_contributions", None)
    if contributions is None:
        return None
    out = contributions(X)
    return None if out is None else np.asarray(out, dtype=np.float64)
