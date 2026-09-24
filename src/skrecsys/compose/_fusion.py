"""Reciprocal rank fusion: combine ranked lists by rank alone, with nothing to learn.

Every source ranks the candidates of a query, and a candidate scores

    sum over sources of  weight / (k + rank)

where ``rank`` counts from 1 and a source that did not rank the candidate adds nothing
(Cormack, Clarke and Buettcher, 2009). Only ranks enter the sum, so sources whose scores
live on unrelated scales -- BM25 and EASE, CatBoost and LightGBM -- combine without
normalization or training, and a large ``k`` flattens the advantage of the top ranks.

The same fusion comes in the two shapes the composites take:

- :class:`ReciprocalRankFusion` is a recommender fusing the lists of other recommenders,
  and stands wherever a recommender does: on its own, as a :class:`Switch` branch, or as
  the generator of a :class:`Cascade`;
- :class:`ReciprocalRankRanker` is a ranker, the ``ranker`` of a :class:`Cascade`,
  fusing the columns of its features -- such as the scores of the generator and of other
  recommenders -- or the scores of other rankers.
"""

import sys
from typing import Annotated, Self, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import _check_feature_names, check_array, check_is_fitted

from skrecsys._typing import FittedRecommender, Ranker, Recommender, clone_as, override
from skrecsys.base import (
    RankerMixin,
    RecommenderMixin,
    check_n_recommendations,
    fit_clone,
    is_ranker,
    is_recommender,
    serves_unknown_users,
)
from skrecsys.compose._candidates import concat_ids, retrieve, top_k_per_group
from skrecsys.compose._named import ComponentList, NamedComponentsEstimator
from skrecsys.compose._rankers import check_groups, mean_positions
from skrecsys.tune._space import Float
from skrecsys.utils._param_validation import check_int, check_real
from skrecsys.utils.validation import (
    check_ids,
    check_interactions,
    encode_ids,
    factorize,
    lookup_ids,
)

if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs

#: The ``recommenders`` of :class:`ReciprocalRankFusion`: all bare or all named.
RecommenderList: TypeAlias = ComponentList[Recommender]

#: The ``rankers`` of :class:`ReciprocalRankRanker`: all bare or all named.
RankerList: TypeAlias = ComponentList[Ranker]

#: Queries fused at once by ``recommend``, in units of candidate rows per source.
_PAIRS_PER_BLOCK = 1_000_000


def check_fusion_weights(weights: ArrayLike | None, n_sources: int) -> NDArray[np.float64]:
    """One non-negative weight per source; ``None`` weighs every source 1."""
    if weights is None:
        return np.ones(n_sources, dtype=np.float64)
    values = np.ravel(np.asarray(weights, dtype=np.float64))
    if len(values) != n_sources:
        raise ValueError(
            f"weights must have one entry per source ({n_sources}), got {len(values)}."
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"weights must be finite and non-negative, got {values.tolist()}.")
    return values


def ranks_per_group(scores: NDArray[np.float64], groups: NDArray[np.int64]) -> NDArray[np.float64]:
    """Every column's 1-based rank within each group, the highest score ranked 1.

    Ties share the average of their ranks. A NaN score is not ranked: its rank is NaN,
    and it does not push any other row down.
    """
    group_of_row = np.repeat(np.arange(len(groups)), groups)
    sizes = groups[group_of_row]
    ranks = np.empty_like(scores)
    for column in range(scores.shape[1]):
        values = scores[:, column]
        missing = np.isnan(values)
        # Below every real score, a NaN takes the bottom positions and leaves the rest be.
        position = mean_positions(np.where(missing, -np.inf, values), group_of_row, groups)
        ranks[:, column] = np.where(missing, np.nan, sizes - position)
    return ranks


def reciprocal_rank_scores(
    ranks: NDArray[np.float64], k: float, weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``sum(weights / (k + ranks))`` across columns, a NaN rank adding nothing."""
    contributions = weights / (k + ranks)
    return np.nansum(contributions, axis=1)


class ReciprocalRankFusion(RecommenderMixin, NamedComponentsEstimator[Recommender]):
    """Recommend what several recommenders rank high, by reciprocal rank fusion.

    Every recommender retrieves its top ``n_retrieved`` items for a query, and an item
    scores ``sum(weight / (k + rank))`` over the lists it appears in, its rank counting
    from 1. Nothing is learned beyond the recommenders themselves: this is the blend to
    reach for before a :class:`Cascade`, or instead of one, when the recommenders are
    good in different ways -- BM25 and EASE, say -- and there is no side information for
    a ranker to use.

    A recommender that raises for users it was not fitted on (see
    :func:`skrecsys.base.serves_unknown_users`) is asked only about the users it knows,
    so a fusion with :class:`~skrecsys.recommendation.MostPopularRecommender` serves
    everyone, and cold users get its list alone.

    Parameters
    ----------
    recommenders : list of recommenders or of (name, recommender) tuples
        Cloned and fitted on the same interactions as ``recommenders_``. Unnamed ones are
        named after their class in lower case, numbered when a class repeats; names
        address nested parameters, as in ``ease__l2_reg``.
    k : float, default=60
        Added to every rank. The larger it is, the less the top of a list outweighs the
        rest of it; 60 is the value of the original paper.
    weights : array-like of shape (n_recommenders,), default=None
        How much each recommender's list counts; equal by default.
    n_retrieved : int, default=100
        How deep every recommender's list is. It is the most ``n_recommendations`` can
        be; an item below every list's cut-off scores nothing.

    Attributes
    ----------
    recommenders_ : list of (name, recommender) tuples
        The fitted recommenders.
    user_ids_, item_ids_ : ndarray
        The union of those of the recommenders, as in :class:`Cascade`.
    n_users_, n_items_ : int

    Examples
    --------
    >>> from skrecsys.compose import ReciprocalRankFusion
    >>> from skrecsys.recommendation import BM25Recommender, MostPopularRecommender
    >>> X = [[u, i] for u in range(20) for i in (u % 5, u % 5 + 1, u % 5 + 2)]
    >>> rec = ReciprocalRankFusion([BM25Recommender(), MostPopularRecommender()]).fit(X)
    >>> rec.recommend([0, 1000], n_recommendations=2)[0].shape
    (2, 2)
    """

    _components_param = "recommenders"
    _component_kind = "a recommender"

    def __init__(
        self,
        recommenders: RecommenderList,
        *,
        k: Annotated[float, Float(1.0, 1000.0, log=True)] = 60.0,
        weights: ArrayLike | None = None,
        n_retrieved: int = 100,
    ) -> None:
        self.recommenders: RecommenderList = recommenders
        self.k = k
        self.weights = weights
        self.n_retrieved = n_retrieved

    @override
    def _is_component(self, value: object) -> TypeIs[Recommender]:
        return is_recommender(value)

    @override
    def _components(self) -> RecommenderList:
        return self.recommenders

    @property
    def _serves_unknown_users(self) -> bool:
        """Whether any recommender answers unknown users, which the fusion then does too."""
        return any(serves_unknown_users(recommender) for _, recommender in self._named())

    def _check_params(self) -> list[tuple[str, Recommender]]:
        if not self.recommenders:
            raise ValueError("ReciprocalRankFusion needs at least one recommender.")
        named = self._named()
        for _, recommender in named:
            if not is_recommender(recommender):
                raise TypeError(f"{type(recommender).__name__} is not a recommender.")
        check_real(self.k, "k", min_value=0)
        check_fusion_weights(self.weights, len(named))
        check_int(self.n_retrieved, "n_retrieved", min_value=1)
        return named

    @override
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit every recommender on the interactions ``X``.

        Parameters
        ----------
        X : array-like of shape (n_interactions, 2)
            ``X[:, 0]`` contains user identifiers, ``X[:, 1]`` item identifiers.
        y : array-like of shape (n_interactions,), default=None
            Interaction values; ``None`` gives every interaction weight 1.

        Returns
        -------
        self : object
        """
        named = self._check_params()
        check_interactions(X, y)
        _check_feature_names(self, X, reset=True)
        X = check_array(X, dtype=None, ensure_all_finite=False)
        self.recommenders_ = [(name, fit_clone(rec, X, y)) for name, rec in named]
        fitted = [recommender for _, recommender in self.recommenders_]
        self.user_ids_ = factorize(concat_ids([rec.user_ids_ for rec in fitted]))[0]
        self.item_ids_ = factorize(concat_ids([rec.item_ids_ for rec in fitted]))[0]
        self.n_users_ = len(self.user_ids_)
        self.n_items_ = len(self.item_ids_)
        return self

    def _served(
        self, recommender: FittedRecommender, queries: NDArray[np.generic]
    ) -> NDArray[np.intp]:
        """Positions of the queries ``recommender`` can answer."""
        if serves_unknown_users(recommender):
            return np.arange(len(queries))
        return np.flatnonzero(lookup_ids(queries, recommender.user_ids_, name="user")[1])

    def _fuse(
        self,
        queries: NDArray[np.generic],
        *,
        candidates: ArrayLike | None,
        exclude_seen: bool,
        exclude_interactions: ArrayLike | None,
    ) -> tuple[NDArray[np.intp], NDArray[np.generic], NDArray[np.float64]]:
        """Every (query, item) any list holds: the query's position, the item, its score.

        Rows come sorted by query, so each query's rows are contiguous.
        """
        k = float(self.k)
        weights = check_fusion_weights(self.weights, len(self.recommenders_))
        rows, items, contributions = [], [], []
        for (_, recommender), weight in zip(self.recommenders_, weights, strict=True):
            served = self._served(recommender, queries)
            if not len(served):
                continue
            pairs, _, groups, kept = retrieve(
                recommender,
                queries[served],
                n_retrieved=int(self.n_retrieved),
                min_retrieved=0,
                candidates=candidates,
                exclude_seen=exclude_seen,
                exclude_interactions=exclude_interactions,
            )
            rank = np.arange(len(pairs)) - np.repeat(np.cumsum(groups) - groups, groups) + 1
            rows.append(np.repeat(served[kept], groups))
            items.append(pairs[:, 1])
            contributions.append(weight / (k + rank))
        if not rows:
            return (
                np.empty(0, dtype=np.intp),
                np.empty(0, dtype=self.item_ids_.dtype),
                np.empty(0, dtype=np.float64),
            )
        item_ids, item_codes = factorize(concat_ids(items))
        n_items = max(len(item_ids), 1)
        keys = np.concatenate(rows).astype(np.int64) * n_items + item_codes
        distinct, inverse = np.unique(keys, return_inverse=True)
        fused = np.bincount(np.ravel(inverse), weights=np.concatenate(contributions))
        return (distinct // n_items).astype(np.intp), item_ids[distinct % n_items], fused

    @override
    def recommend(
        self,
        X: ArrayLike,
        *,
        n_recommendations: int = 10,
        candidates: ArrayLike | None = None,
        exclude_seen: bool = True,
        exclude_interactions: ArrayLike | None = None,
    ) -> tuple[NDArray[np.generic], NDArray[np.floating]]:
        """Return the items with the highest fused scores, and those scores.

        Parameters are those of :meth:`skrecsys.base.RecommenderMixin.recommend`, and
        are passed to every recommender. Ties are resolved by fitted item order.

        Raises
        ------
        ValueError
            If ``n_recommendations`` exceeds ``n_retrieved``, or a query has fewer than
            ``n_recommendations`` eligible items.
        """
        check_is_fitted(self)
        check_n_recommendations(n_recommendations)
        if n_recommendations > self.n_retrieved:
            raise ValueError(
                f"n_recommendations={n_recommendations} exceeds n_retrieved={self.n_retrieved}."
            )
        queries = check_ids(X)
        items = np.empty((len(queries), n_recommendations), dtype=self.item_ids_.dtype)
        top_scores = np.empty((len(queries), n_recommendations), dtype=np.float64)
        size = max(1, _PAIRS_PER_BLOCK // (int(self.n_retrieved) * len(self.recommenders_)))
        for start in range(0, len(queries), size):
            stop = min(start + size, len(queries))
            row, item, fused = self._fuse(
                queries[start:stop],
                candidates=candidates,
                exclude_seen=exclude_seen,
                exclude_interactions=exclude_interactions,
            )
            groups = np.bincount(row, minlength=stop - start).astype(np.int64)
            short = np.flatnonzero(groups < n_recommendations)
            if short.size:
                query = int(short[0])
                raise ValueError(
                    f"Cannot recommend {n_recommendations} items: query {query + start} has "
                    f"only {groups[query]} eligible items."
                )
            positions = lookup_ids(item, self.item_ids_, name="item")[0]
            best = top_k_per_group(fused, groups, positions, n_recommendations)
            items[start:stop] = item[best]
            top_scores[start:stop] = fused[best]
        return items, top_scores

    @override
    def _count_eligible(
        self,
        X: ArrayLike,
        *,
        candidates: ArrayLike | None = None,
        exclude_seen: bool = True,
        exclude_interactions: ArrayLike | None = None,
    ) -> NDArray[np.int64]:
        check_is_fitted(self)
        queries = check_ids(X)
        counts = np.zeros(len(queries), dtype=np.int64)
        for _, recommender in self.recommenders_:
            served = self._served(recommender, queries)
            if not len(served):
                continue
            eligible = recommender._count_eligible(
                queries[served],
                candidates=candidates,
                exclude_seen=exclude_seen,
                exclude_interactions=exclude_interactions,
            )
            counts[served] = np.maximum(counts[served], eligible)
        return np.minimum(counts, int(self.n_retrieved))

    def predict(self, X: ArrayLike) -> NDArray[np.floating]:
        """Score user-item pairs with the fused score of the item in the user's lists.

        The lists are those ``recommend`` fuses with ``exclude_seen=False``, so a pair
        the user has interacted with is scored too; an item below every list's cut-off
        scores 0.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            User-item pairs; items must be known, and users known to at least one
            recommender unless one of them serves unknown users.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        users, pair_items, _ = check_interactions(X)
        _check_feature_names(self, X, reset=False)
        encode_ids(pair_items, self.item_ids_, name="item")
        queries, user_codes = factorize(users)
        served = np.zeros(len(queries), dtype=bool)
        for _, recommender in self.recommenders_:
            served[self._served(recommender, queries)] = True
        if not served.all():
            sample = list(queries[~served][:5])
            raise ValueError(f"Unknown user identifiers: {sample}.")
        row, item, fused = self._fuse(
            queries, candidates=None, exclude_seen=False, exclude_interactions=None
        )
        n_items = len(self.item_ids_)
        fused_keys = row.astype(np.int64) * n_items + encode_ids(item, self.item_ids_, name="item")
        pair_keys = user_codes.astype(np.int64) * n_items + encode_ids(
            pair_items, self.item_ids_, name="item"
        )
        order = np.argsort(fused_keys)
        position, found = lookup_ids(pair_keys, fused_keys[order], name="pair")
        return np.where(found, fused[order][position], 0.0)


class ReciprocalRankRanker(RankerMixin, NamedComponentsEstimator[Ranker]):
    """Rank candidates by reciprocal rank fusion of feature columns or of other rankers.

    Within each group, every source ranks the rows from 1, highest score first, and a row
    scores ``sum(weight / (k + rank))``. With ``rankers=None`` the sources are the
    columns of the features themselves, read as scores where higher is better, and
    nothing at all is learned: inside a :class:`Cascade`, features such as
    :class:`GeneratorScores` and :class:`RecommenderScores` become a training-free second
    stage. With ``rankers``, the sources are their predictions: each ranker is fitted on
    every row, and their ranks are fused -- the untrained alternative to
    :class:`BlendRanker`, with no out-of-fold refits.

    Parameters
    ----------
    rankers : list of rankers or of (name, ranker) tuples, default=None
        Cloned and fitted as ``rankers_``; named as in :class:`BlendRanker`. ``None``
        fuses the feature columns.
    k : float, default=60
        Added to every rank; see :class:`ReciprocalRankFusion`.
    weights : array-like of shape (n_sources,), default=None
        How much each source counts -- each ranker, or each feature column when
        ``rankers=None``; equal by default.

    Attributes
    ----------
    rankers_ : list of (name, ranker) tuples
        The fitted rankers; empty when fusing feature columns.
    n_features_in_ : int

    Notes
    -----
    A NaN feature, such as a :class:`RecommenderScores` column for a pair the recommender
    cannot score, does not rank its row, which then gets nothing from that column.

    Examples
    --------
    >>> from skrecsys.compose import ReciprocalRankRanker
    >>> F = [[0.9, 0.1], [0.5, 0.8], [0.1, 0.2]]
    >>> ReciprocalRankRanker(k=1).fit(F, [1, 0, 0], groups=[3]).predict(F, groups=[3])
    array([0.75      , 0.83333333, 0.58333333])
    """

    _components_param = "rankers"
    _component_kind = "a ranker"

    def __init__(
        self,
        rankers: RankerList | None = None,
        *,
        k: Annotated[float, Float(1.0, 1000.0, log=True)] = 60.0,
        weights: ArrayLike | None = None,
    ) -> None:
        self.rankers: RankerList | None = rankers
        self.k = k
        self.weights = weights

    @override
    def _is_component(self, value: object) -> TypeIs[Ranker]:
        return is_ranker(value)

    @override
    def _components(self) -> RankerList | None:
        return self.rankers

    def fit(self, X: ArrayLike, y: ArrayLike, *, groups: ArrayLike | None = None) -> Self:
        """Fit every ranker on all rows; with ``rankers=None``, only check the input."""
        if self.rankers is not None and not self.rankers:
            raise ValueError("rankers must be None or hold at least one ranker.")
        named = self._named()
        for _, ranker in named:
            if not is_ranker(ranker):
                raise TypeError(f"{type(ranker).__name__} is not a ranker.")
        check_real(self.k, "k", min_value=0)
        X = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")
        sizes = check_groups(groups, len(X))
        check_fusion_weights(self.weights, len(named) or X.shape[1])
        if named:
            labels = check_array(y, ensure_2d=False, dtype=np.float64)
            self.rankers_ = [
                (name, clone_as(ranker).fit(X, labels, groups=sizes)) for name, ranker in named
            ]
        else:
            self.rankers_ = []
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: ArrayLike, *, groups: ArrayLike | None = None) -> NDArray[np.float64]:
        """Score every row with the fused reciprocal ranks of its sources."""
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")
        sizes = check_groups(groups, len(X))
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the ranker was fitted with "
                f"{self.n_features_in_}."
            )
        if self.rankers_:
            scores = np.column_stack(
                [
                    np.asarray(ranker.predict(X, groups=sizes), dtype=np.float64)
                    for _, ranker in self.rankers_
                ]
            )
        else:
            scores = X
        weights = check_fusion_weights(self.weights, scores.shape[1])
        return reciprocal_rank_scores(ranks_per_group(scores, sizes), float(self.k), weights)
