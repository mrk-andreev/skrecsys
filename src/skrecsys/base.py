"""Base classes and helpers for recommender estimators."""

from typing import TYPE_CHECKING, Self, TypeVar, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils import Tags, get_tags
from sklearn.utils.validation import check_is_fitted

from skrecsys import _core
from skrecsys._attribution import Attributions
from skrecsys._tracing import traced_recommend
from skrecsys.typing import (
    Condition,
    Features,
    FittedRecommender,
    PairScorer,
    Ranker,
    Recommender,
    TimedRecommender,
    TypeIs,
    clone_as,
    override,
)
from skrecsys.utils.validation import (
    check_ids,
    check_interactions,
    check_queries,
    encode_ids,
    factorize,
    lookup_ids,
)

if TYPE_CHECKING:
    from skrecsys.compose._rankers import Candidates

__all__ = [
    "AllOf",
    "AnyOf",
    "ConditionMixin",
    "FeaturesMixin",
    "Not",
    "RankerMixin",
    "RecommenderMixin",
    "excluded_among",
    "first_row_of",
    "first_time_of",
    "is_condition",
    "is_features",
    "is_ranker",
    "is_recommender",
    "seen_among",
    "serves_unknown_users",
    "supports_partial_fit",
    "uses_time",
]


class _TagsMixin:
    """Base of the mixins below, which extend the tags of the estimator they join.

    Empty at runtime: ``super().__sklearn_tags__()`` in a mixin reaches whatever comes
    next in the estimator's MRO, which is ``BaseEstimator``. The declaration only tells
    the type checker that something there answers the call.
    """

    if TYPE_CHECKING:

        def __sklearn_tags__(self) -> Tags: ...


class RecommenderMixin(_TagsMixin):
    """Mixin class for all recommenders.

    Defines the ``recommend`` operation and sets the ``estimator_type`` tag to
    ``"recommender"``. The semantic type of a query is left to the estimator: subclasses
    implement ``_score_queries`` to score candidate items for each query in ``X``.

    The mixin deliberately provides no default ``score``: rating error, ranking quality
    and retrieval quality are different objectives. Use
    :func:`skrecsys.metrics.make_recommender_scorer` for model selection.

    Estimators using this mixin must expose a fitted ``item_ids_`` attribute.
    """

    #: Fitted by the estimator. Annotation only: a class attribute would be a value the
    #: estimator does not have until it is fitted, which is what ``check_is_fitted`` reads.
    item_ids_: NDArray[np.generic]

    if TYPE_CHECKING:
        # Every recommender implements it; declared so the mixins can call it. Declared
        # here, on a base that comes after the estimator's own class in its MRO, it never
        # stands in front of the real ``fit`` -- which a declaration on a mixin listed
        # first would, and IDEs then resolve ``Self`` against the mixin.
        def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self: ...

    @override
    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.estimator_type = "recommender"
        tags.target_tags.required = False
        tags.input_tags.string = True
        return tags

    def _score_queries(
        self, X: ArrayLike, item_indices: NDArray[np.intp], *, exclude_seen: bool
    ) -> tuple[NDArray[np.floating], sp.csr_array]:
        """Score ``item_indices`` for each query.

        Returns
        -------
        scores : ndarray of shape (n_queries, len(item_indices))
        excluded : scipy.sparse.csr_array of shape (n_queries, len(item_indices))
            Stored positions must not be recommended to the query. Sparse rather than a
            dense mask because the exclusions are the queries' own interactions, which
            are a vanishing fraction of the catalog.
        """
        raise NotImplementedError

    def _attribute(
        self, queries: NDArray[np.generic], items: NDArray[np.generic], n_reasons: int
    ) -> Attributions | None:
        """Why each aligned pair ``(queries[p], items[p])`` scores as it does.

        The ``n_reasons`` history items that weigh most in each pair's score, as
        :class:`skrecsys._attribution.Attributions`; see that module for the kinds of
        answer. None when the recommender cannot say, which is the default: a composite
        is explained through its parts. Read by :mod:`skrecsys.inspection`.
        """
        del queries, items, n_reasons
        return None

    @traced_recommend
    def recommend(
        self,
        X: ArrayLike,
        *,
        n_recommendations: int = 10,
        candidates: ArrayLike | None = None,
        exclude_seen: bool = True,
        exclude_interactions: ArrayLike | None = None,
    ) -> tuple[NDArray[np.generic], NDArray[np.floating]]:
        """Return recommended item identifiers and their scores.

        Parameters
        ----------
        X : array-like of shape (n_queries,) or (n_queries, 1 + n_context)
            Queries. For the estimators in :mod:`skrecsys.recommendation` these are
            user identifiers seen during ``fit``. A matrix holds the user in ``X[:, 0]``
            and the query context in the other columns, laid out like the context
            columns of the ``X`` of ``fit``; a vector asks without context. A recommender
            that cannot use context, such as every one in
            :mod:`skrecsys.recommendation`, ignores it.

        n_recommendations : int, default=10
            Number of items to return per query.

        candidates : array-like of shape (n_candidates,), default=None
            Item identifiers eligible for recommendation, shared by all queries.
            ``None`` means all fitted items.

        exclude_seen : bool, default=True
            Whether to remove items observed for the query during ``fit``.

        exclude_interactions : array-like of shape (n_interactions, 2), default=None
            Further user-item pairs that must not be recommended, laid out like the
            ``X`` of ``fit``: ``[:, 0]`` names a query, ``[:, 1]`` an item. Each pair
            removes its item from every query in ``X`` equal to its user, on top of
            ``exclude_seen`` and independently of it. Pairs whose user is not among the
            queries, or whose item the model does not know, are ignored, so the events a
            user produced since the model was fitted can be passed as they are -- the
            same rows a later ``partial_fit`` would take.

        Returns
        -------
        items : ndarray of shape (n_queries, n_recommendations)
            Recommended item identifiers ranked by descending score. Ties are resolved
            by fitted item order.

        scores : ndarray of shape (n_queries, n_recommendations)
            Scores of the recommended items.

        Raises
        ------
        ValueError
            If any query has fewer than ``n_recommendations`` eligible items.
        """
        check_is_fitted(self)
        item_ids = self.item_ids_
        check_n_recommendations(n_recommendations)

        item_indices = self._candidate_indices(candidates)

        # Queries are ranked in blocks: whatever a block scores is reduced to k columns
        # before the next one starts, so peak memory follows the block, not the query.
        queries, _ = check_queries(X)
        excluded = (
            None
            if exclude_interactions is None
            else excluded_among(queries, exclude_interactions, item_ids, item_indices)
        )
        size = self._rank_chunk_size(len(item_indices), n_recommendations)
        items = np.empty((len(queries), n_recommendations), dtype=item_ids.dtype)
        top_scores = np.empty((len(queries), n_recommendations), dtype=np.float64)
        for start in range(0, len(queries), size):
            stop = min(start + size, len(queries))
            order, scores = self._rank_queries(
                queries[start:stop],
                item_indices,
                n_recommendations,
                exclude_seen=exclude_seen,
                excluded=None if excluded is None else excluded[start:stop],
                first_query=start,
            )
            items[start:stop] = item_ids[item_indices[order]]
            top_scores[start:stop] = scores
        return items, top_scores

    def _candidate_indices(self, candidates: ArrayLike | None) -> NDArray[np.intp]:
        """Sorted, distinct fitted positions of ``candidates``; every item for ``None``."""
        item_ids = self.item_ids_
        if candidates is None:
            return np.arange(len(item_ids))
        candidate_ids = check_ids(candidates, name="candidates")
        return np.unique(encode_ids(candidate_ids, item_ids, name="item"))

    def _count_eligible(
        self,
        X: ArrayLike,
        *,
        candidates: ArrayLike | None = None,
        exclude_seen: bool = True,
        exclude_interactions: ArrayLike | None = None,
    ) -> NDArray[np.int64]:
        """How many items ``recommend`` could return to each query, given the same filters.

        A multi-stage model asks its first stage for as many candidates as a query *has*
        rather than a fixed number, because ``recommend`` raises instead of padding.
        Composites override this to answer for their parts.
        """
        check_is_fitted(self)
        item_ids = self.item_ids_
        item_indices = self._candidate_indices(candidates)
        queries, _ = check_queries(X)
        excluded = (
            None
            if exclude_interactions is None
            else excluded_among(queries, exclude_interactions, item_ids, item_indices)
        )
        seen = self._excluded_by_seen(queries, item_indices, exclude_seen=exclude_seen)
        union = union_of_exclusions(seen, excluded)
        return (len(item_indices) - np.diff(union.indptr)).astype(np.int64)

    def _excluded_by_seen(
        self, queries: NDArray[np.generic], item_indices: NDArray[np.intp], *, exclude_seen: bool
    ) -> sp.csr_array:
        """The candidate positions ``exclude_seen`` removes for each query.

        The fallback, for a recommender implementing only ``_score_queries``: the mask
        comes back with the scores, so getting it costs a scoring pass. The queries are
        scored in the blocks ``recommend`` uses, so the dense score matrix is bounded the
        same way, and not scored at all when ``exclude_seen`` is False. Estimators that
        keep their interactions override this and answer without scoring.
        """
        if not exclude_seen:
            return sp.csr_array((len(queries), len(item_indices)), dtype=bool)
        size = self._rank_chunk_size(len(item_indices), 1)
        blocks = []
        for start in range(0, len(queries), size):
            block = queries[start : start + size]
            _, seen = self._score_queries(block, item_indices, exclude_seen=True)
            blocks.append(seen.astype(bool))
        return sp.csr_array(sp.vstack(blocks, format="csr"))

    def _rank_chunk_size(self, n_candidates: int, k: int) -> int:
        """Queries ranked per block, holding the dense score matrix near 64 MB.

        ``k`` is unused here and is passed because an estimator that ranks without a
        dense matrix sizes its blocks by what it *does* build, which can depend on how
        many items each query asks for.
        """
        del k
        return max(1, min(8_000_000 // max(n_candidates, 1), 8192))

    def _rank_queries(
        self,
        queries: NDArray[np.generic],
        item_indices: NDArray[np.intp],
        k: int,
        *,
        exclude_seen: bool,
        excluded: sp.csr_array | None = None,
        first_query: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.floating]]:
        """Return the ``k`` best candidate positions of each query, and their scores.

        The one place ``recommend`` reaches for a ranking, and therefore the place an
        approximate index gets to intervene: :class:`skrecsys.indexing.VectorIndexMixin`
        overrides this and falls back to ``_rank_queries_exact``.
        """
        return self._rank_queries_exact(
            queries,
            item_indices,
            k,
            exclude_seen=exclude_seen,
            excluded=excluded,
            first_query=first_query,
        )

    def _rank_queries_exact(
        self,
        queries: NDArray[np.generic],
        item_indices: NDArray[np.intp],
        k: int,
        *,
        exclude_seen: bool,
        excluded: sp.csr_array | None = None,
        first_query: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.floating]]:
        """Rank by scoring every candidate, exactly.

        ``excluded`` holds, per query, further candidate positions to skip beyond what
        ``exclude_seen`` removes. ``first_query`` is the position of ``queries[0]`` among
        all the queries, so that an error names the query the caller asked about.
        Estimators that can rank without a dense score matrix override this.
        """
        scores, seen = self._score_queries(queries, item_indices, exclude_seen=exclude_seen)
        excluded = union_of_exclusions(seen, excluded)
        check_enough_eligible(excluded, len(item_indices), k, first_query)

        # Selecting k of n beats sorting all n. The kernel ranks by descending score and
        # breaks ties by column index, which is fitted item order because item_indices is
        # sorted.
        order = _core.top_k_per_row(
            np.ascontiguousarray(scores, dtype=np.float64),
            np.ascontiguousarray(excluded.indptr, dtype=np.int64),
            np.ascontiguousarray(excluded.indices, dtype=np.int64),
            k,
        )
        return order, np.take_along_axis(scores, order, axis=1)


def check_n_recommendations(n_recommendations: object) -> None:
    """Raise unless ``n_recommendations`` is an integer >= 1."""
    if isinstance(n_recommendations, bool) or not isinstance(n_recommendations, int | np.integer):
        raise TypeError(
            f"n_recommendations must be an integer, got {type(n_recommendations).__name__}."
        )
    if n_recommendations < 1:
        raise ValueError(f"n_recommendations must be >= 1, got {n_recommendations}.")


def check_enough_eligible(
    excluded: sp.csr_array, n_candidates: int, k: int, first_query: int
) -> None:
    """Raise if a query has fewer than ``k`` candidates left once its exclusions go.

    The kernels check this too, but only they know the row within the block; the count
    is exact here because the exclusions are candidate positions, each stored once.
    """
    available = n_candidates - np.diff(excluded.indptr)
    short = np.flatnonzero(available < k)
    if short.size:
        row = int(short[0])
        raise ValueError(
            f"Cannot recommend {k} items: query {row + first_query} has only "
            f"{available[row]} eligible items."
        )


class ConditionMixin(_TagsMixin):
    """Mixin class for conditions: fitted predicates over queries.

    A condition is fitted on the same interactions as the recommenders it routes between
    and answers, per query, which branch of a :class:`skrecsys.compose.Switch` serves it.
    Subclasses implement ``fit(X, y=None)`` and ``evaluate(X) -> ndarray of bool``.

    ``~a``, ``a & b`` and ``a | b`` combine conditions into new ones, which are estimators
    like any other, so they clone and take part in a grid search.
    """

    @override
    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.estimator_type = "condition"
        tags.target_tags.required = False
        tags.input_tags.string = True
        return tags

    def evaluate(self, X: ArrayLike) -> NDArray[np.bool_]:
        """Return, for each query in ``X``, whether the condition holds."""
        raise NotImplementedError

    def __invert__(self: Condition) -> "Not":
        return Not(self)

    def __and__(self: Condition, other: Condition) -> "AllOf":
        return AllOf([self, other])

    def __or__(self: Condition, other: Condition) -> "AnyOf":
        return AnyOf([self, other])


class Not(ConditionMixin, BaseEstimator):
    """Hold where ``condition`` does not. ``~condition`` builds one.

    Parameters
    ----------
    condition : condition
        The condition to negate. It is cloned and fitted as ``condition_``.
    """

    def __init__(self, condition: Condition) -> None:
        self.condition = condition

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the negated condition."""
        self.condition_ = _fitted_condition(self.condition, X, y)
        return self

    @override
    def evaluate(self, X: ArrayLike) -> NDArray[np.bool_]:
        check_is_fitted(self)
        return ~evaluate_condition(self.condition_, X)


class AllOf(ConditionMixin, BaseEstimator):
    """Hold where every one of ``conditions`` holds. ``a & b`` builds one.

    Parameters
    ----------
    conditions : list of conditions
        Cloned and fitted as ``conditions_``.
    """

    def __init__(self, conditions: list[Condition]) -> None:
        self.conditions = conditions

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit every condition."""
        self.conditions_ = [_fitted_condition(c, X, y) for c in self.conditions]
        return self

    @override
    def evaluate(self, X: ArrayLike) -> NDArray[np.bool_]:
        check_is_fitted(self)
        return np.logical_and.reduce([evaluate_condition(c, X) for c in self.conditions_])


class AnyOf(ConditionMixin, BaseEstimator):
    """Hold where at least one of ``conditions`` holds. ``a | b`` builds one.

    Parameters
    ----------
    conditions : list of conditions
        Cloned and fitted as ``conditions_``.
    """

    def __init__(self, conditions: list[Condition]) -> None:
        self.conditions = conditions

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit every condition."""
        self.conditions_ = [_fitted_condition(c, X, y) for c in self.conditions]
        return self

    @override
    def evaluate(self, X: ArrayLike) -> NDArray[np.bool_]:
        check_is_fitted(self)
        return np.logical_or.reduce([evaluate_condition(c, X) for c in self.conditions_])


#: The condition a combinator fits, kept as its own type through the clone.
ConditionT = TypeVar("ConditionT", bound=Condition)


def _fitted_condition(condition: ConditionT, X: ArrayLike, y: ArrayLike | None) -> ConditionT:
    if not is_condition(condition):
        raise TypeError(f"{type(condition).__name__} is not a condition.")
    return clone_as(condition).fit(X, y)


def evaluate_condition(condition: Condition, X: ArrayLike) -> NDArray[np.bool_]:
    """``condition.evaluate(X)``, checked to be one boolean per query; context is dropped."""
    queries, _ = check_queries(X)
    mask = np.asarray(condition.evaluate(queries))
    if mask.shape != queries.shape or mask.dtype != np.bool_:
        raise ValueError(
            f"{type(condition).__name__}.evaluate must return a boolean array of shape "
            f"{queries.shape}, got {mask.dtype} of shape {mask.shape}."
        )
    return mask


class FeaturesMixin(_TagsMixin):
    """Mixin class for features computed per candidate user-item pair.

    Subclasses implement ``fit(X, y=None)`` on the interactions and
    ``transform(pairs, *, scores=None, context=None)``, where ``pairs`` has shape
    ``(n_pairs, 2)`` laid out like the ``X`` of ``fit``, ``scores`` holds what the
    candidate generator scored each pair -- shape ``(n_pairs,)``, or
    ``(n_pairs, n_generators)`` when several generators propose candidates -- and
    ``context`` the query context of each pair, shape ``(n_pairs, n_context)``: the context
    of the query the pair was retrieved for, ``None`` when there is none. Every component
    takes both keywords, and one that does not read them ignores them. ``transform`` returns a float
    ndarray of shape ``(n_pairs, n_features)``.
    """

    @override
    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.estimator_type = "features"
        tags.target_tags.required = False
        tags.input_tags.string = True
        return tags

    def transform(
        self,
        pairs: ArrayLike,
        *,
        scores: ArrayLike | None = None,
        context: ArrayLike | None = None,
    ) -> NDArray[np.floating]:
        """Return the features of each pair."""
        raise NotImplementedError


class RankerMixin(_TagsMixin):
    """Mixin class for rankers: models ordering the candidates of each query.

    Subclasses implement ``fit(X, y, *, groups)`` and ``predict(X, *, groups)``. ``X`` is
    a feature matrix of shape ``(n_pairs, n_features)`` whose rows are contiguous per
    query, ``groups`` holds the number of rows of each query in order -- the convention
    of LightGBM and XGBoost -- and ``y`` the relevance of each pair. ``predict`` returns
    one score per row; only its order within a group matters.

    Like :class:`RecommenderMixin`, the mixin provides no default ``score``.
    """

    @override
    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.estimator_type = "ranker"
        return tags

    def _contributions(self, X: NDArray[np.float64]) -> NDArray[np.float64] | None:
        """What each feature adds to each row's score, or None when the ranker cannot say.

        Shape ``(n_rows, n_features + 1)``; the last column is what no feature owns, a
        bias or an expected value, and a row adds up to the ranker's score in its own
        space -- log-odds for a logistic model, the raw margin for a boosted one. Read by
        :mod:`skrecsys.inspection`.

        ``X`` is the matrix the ranker scores from: its input, unless it joins features
        of its own, when it is what :meth:`_ranker_input` returns.
        """
        del X
        return None

    def _takes_candidates(self) -> bool:
        """Whether ``fit`` and ``predict`` take the keyword ``candidates``.

        A ranker joining features of its own, such as
        :class:`~skrecsys.compose.AugmentedRanker`, needs the candidate pairs behind the
        rows of ``X``, and a composite of rankers passes them on; a
        :class:`~skrecsys.compose.Cascade` gives them to a ranker that says so.
        """
        return False

    def _ranker_input(
        self, X: NDArray[np.float64], candidates: "Candidates"
    ) -> tuple[NDArray[np.float64], tuple[str, ...] | None]:
        """The matrix the fitted ranker scores ``candidates`` from, and its column names.

        ``X`` and ``candidates.names`` themselves, unless the ranker joins features of
        its own onto the candidates. Read by :mod:`skrecsys.inspection`, which reports
        this matrix as the ranker's features.
        """
        return X, candidates.names

    def _fit_features(self, X: ArrayLike, y: ArrayLike | None) -> None:
        """Fit the features this ranker joins itself on the interactions ``X``.

        Called by a :class:`~skrecsys.compose.Cascade` on a clone before ``fit``, with
        the interactions the ranker is not labelled from, then on the fitted ranker with
        all of them, for serving. A ranker joining nothing does nothing.
        """
        del X, y


def is_recommender(estimator: object) -> TypeIs[Recommender]:
    """Return True if the given estimator is a recommender."""
    return _estimator_type(estimator) == "recommender"


def is_condition(estimator: object) -> TypeIs[Condition]:
    """Return True if the given estimator is a condition."""
    return _estimator_type(estimator) == "condition"


def is_features(estimator: object) -> TypeIs[Features]:
    """Return True if the given estimator computes features of candidate pairs."""
    return _estimator_type(estimator) == "features"


def is_ranker(estimator: object) -> TypeIs[Ranker]:
    """Return True if the given estimator is a ranker."""
    return _estimator_type(estimator) == "ranker"


def _estimator_type(estimator: object) -> str | None:
    """The ``estimator_type`` tag, or ``None`` for an object that is not an estimator."""
    if not hasattr(estimator, "__sklearn_tags__"):
        return None
    return get_tags(estimator).estimator_type


def fit_clone(recommender: Recommender, X: ArrayLike, y: ArrayLike | None) -> FittedRecommender:
    """Fit a clone of ``recommender``; a fitted recommender exposes its identifiers."""
    return cast(FittedRecommender, clone_as(recommender).fit(X, y))


def predict_pairs(recommender: Recommender, pairs: ArrayLike) -> NDArray[np.float64]:
    """``recommender.predict(pairs)``, for a composite whose part may not score pairs."""
    if not isinstance(recommender, PairScorer):
        raise TypeError(f"{type(recommender).__name__} cannot score pairs: it has no predict.")
    return np.asarray(recommender.predict(pairs), dtype=np.float64)


def supports_partial_fit(estimator: object) -> bool:
    """Return True if the estimator can be fitted one batch at a time.

    Probing for the method is how scikit-learn itself decides -- ``learning_curve`` and
    the common checks do exactly this -- because there is no tag for incremental
    learning and no ``PartialFitMixin`` to check against. It is also why
    :class:`skrecsys.recommendation.IncrementalRecommenderMixin` is mixed into the
    estimators that have an incremental update rather than being folded into
    :class:`~skrecsys.recommendation.BaseRecommender`: the probe has to stay truthful.
    """
    return hasattr(estimator, "partial_fit")


def serves_unknown_users(recommender: object) -> bool:
    """Return True if ``recommender.recommend`` answers users it was not fitted on.

    Most recommenders raise for an unknown user: they have nothing to score one with. A
    recommender that does not need the user, such as
    :class:`~skrecsys.recommendation.MostPopularRecommender`, declares
    ``_serves_unknown_users = True``, which is what lets :class:`skrecsys.compose.Cascade`
    train its ranker on held-out users the generator has never seen.
    """
    return bool(getattr(recommender, "_serves_unknown_users", False))


def uses_time(recommender: object) -> TypeIs[TimedRecommender]:
    """Return True if ``recommender`` was constructed with ``time=True``.

    Such a recommender is fitted on ``[user, item, time]`` rows and answers
    ``recommend(..., as_of=...)``. A composite hands a part that does not use time the
    identifier columns alone, so every other recommender keeps its two-column ``X``.
    """
    return getattr(recommender, "time", False) is True


def first_time_of(
    users: NDArray[np.generic], times: NDArray[np.generic], queries: NDArray[np.generic]
) -> NDArray[np.generic]:
    """The earliest of ``times`` among the rows of each of the sorted, distinct ``queries``.

    The moment a user's held-out interactions begin, which is the time their candidates
    are ranked as of when a ranker is trained on them and when they are evaluated: what
    the user was shown then, knowing only what happened before. A query without rows
    gets a missing time, which means "latest".
    """
    rows = first_row_of(users, times, queries)
    found = rows >= 0
    if found.all():
        # Every query has a time, so integer times need no room for a missing one.
        return times[rows]
    if times.dtype.kind == "M":
        out = np.full(len(queries), "NaT", dtype=times.dtype)
    else:
        out = np.full(len(queries), np.nan)
    out[found] = times[rows[found]]
    return out


def first_row_of(
    users: NDArray[np.generic], times: NDArray[np.generic] | None, queries: NDArray[np.generic]
) -> NDArray[np.intp]:
    """The row of each of the sorted, distinct ``queries`` among ``users`` that came first.

    The earliest by ``times``, ties to the earlier row, or the first row when there are no
    times; -1 for a query without rows. A held-out user's first row is where their
    held-out interactions begin, and its context is the one they are ranked with.
    """
    positions, known = lookup_ids(users, queries, name="user")
    rows = np.flatnonzero(known)
    positions = positions[known]
    order = (
        np.argsort(positions, kind="stable")
        if times is None
        else np.lexsort((rows, times[known], positions))
    )
    positions, rows = positions[order], rows[order]
    first = np.flatnonzero(np.diff(positions, prepend=-1))
    out = np.full(len(queries), -1, dtype=np.intp)
    out[positions[first]] = rows[first]
    return out


def seen_among(
    interactions: sp.csr_array,
    user_indices: NDArray[np.intp],
    item_indices: NDArray[np.intp],
) -> sp.csr_array:
    """Stored interactions of the given users, in candidate coordinates.

    ``item_indices`` must be sorted, which is what ``recommend`` passes, so the column
    remapping preserves the ascending order of the CSR indices. Only the interactions
    themselves are touched: nothing here is proportional to the catalog size except a
    single lookup table, and only when the candidates are a subset.
    """
    rows = sp.csr_array(interactions[user_indices])
    shape = (len(user_indices), len(item_indices))
    if len(item_indices) == interactions.shape[1]:
        # Every fitted item is a candidate, so the item index is its own position.
        indptr, columns = rows.indptr, rows.indices
    else:
        position = np.full(interactions.shape[1], -1, dtype=np.int64)
        position[item_indices] = np.arange(len(item_indices))
        columns = position[rows.indices]
        keep = columns >= 0
        owners = np.repeat(np.arange(shape[0]), np.diff(rows.indptr))[keep]
        columns = columns[keep]
        indptr = np.concatenate([[0], np.cumsum(np.bincount(owners, minlength=shape[0]))])
    return sp.csr_array((np.ones(len(columns), dtype=bool), columns, indptr), shape=shape)


def union_of_exclusions(first: sp.csr_array, second: sp.csr_array | None) -> sp.csr_array:
    """Both per-query exclusion masks as one, sorted, each position stored once."""
    if second is not None and second.nnz:
        first = sp.csr_array(first.astype(bool) + second.astype(bool))
    first.sort_indices()
    return first


def excluded_among(
    queries: NDArray[np.generic],
    exclude_interactions: ArrayLike,
    item_ids: NDArray[np.generic],
    item_indices: NDArray[np.intp],
) -> sp.csr_array:
    """The pairs of ``exclude_interactions`` that apply, in candidate coordinates.

    Row ``q`` holds the candidate positions that some pair ``(queries[q], item)`` names.
    Pairs are matched against the queries themselves rather than against every fitted
    user, so the matrix built here has a row per *distinct query*, however many users
    the model has; a pair for another user or an unknown item is dropped, not an error.
    """
    distinct, query_rows = factorize(queries)
    if np.shape(exclude_interactions)[:1] == (0,):  # a request with nothing to exclude
        pairs = sp.csr_array((len(distinct), len(item_ids)), dtype=bool)
        return seen_among(pairs, query_rows, item_indices)
    users, items, _ = check_interactions(exclude_interactions)
    user_pos, user_known = lookup_ids(users, distinct, name="user")
    item_pos, item_known = lookup_ids(items, item_ids, name="item")
    keep = user_known & item_known
    pairs = sp.csr_array(
        (np.ones(int(keep.sum()), dtype=bool), (user_pos[keep], item_pos[keep])),
        shape=(len(distinct), len(item_ids)),
    )
    pairs.sum_duplicates()
    return seen_among(pairs, query_rows, item_indices)
