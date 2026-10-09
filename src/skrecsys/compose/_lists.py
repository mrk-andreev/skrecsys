"""Lists put together from several recommenders, and a recommender that is a list.

:class:`~skrecsys.compose.Switch` picks one recommender per query and
:class:`~skrecsys.compose.ReciprocalRankFusion` scores every item by all of them. The
composites here concatenate instead: what one recommender has, then what the next has
that the list lacks. That is how a served list is kept full -- personal items, then
popular ones, then the rest of the catalog -- and how a few of its positions are given to
items no model would have put there.

Their parts need not know the same items, and their scores are not comparable, so the
scores ``recommend`` returns only restate the order of a list: ``n_recommendations`` for
its first item down to 1 for its last.
"""

import sys
from typing import Self, TypeAlias

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import _check_feature_names, check_is_fitted

from skrecsys._tracing import traced_recommend
from skrecsys._typing import FittedRecommender, Recommender, override
from skrecsys.base import (
    RecommenderMixin,
    check_n_recommendations,
    fit_clone,
    is_recommender,
    predict_pairs,
    seen_among,
    serves_unknown_users,
)
from skrecsys.compose._candidates import (
    concat_ids,
    not_among,
    ranks_in_rows,
    retrieve_lists,
    score_pairs,
    served_queries,
)
from skrecsys.compose._named import ComponentList, NamedComponentsEstimator
from skrecsys.exceptions import InsufficientDataError
from skrecsys.utils._param_validation import check_bool, check_component, check_int
from skrecsys.utils.validation import (
    check_ids,
    check_interactions,
    check_queries,
    check_rows,
    drop_time,
    encode_ids,
    factorize,
    lookup_ids,
    stable_unit_hash,
    stack_pairs,
)

if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs

#: The identifier columns of interactions.
_N_COLUMNS = 2

#: The ``recommenders`` of :class:`Backfill`: all bare recommenders or all named ones.
RecommenderList: TypeAlias = ComponentList[Recommender]


def _interaction_ids(
    X: ArrayLike, y: ArrayLike | None
) -> tuple[NDArray[np.generic], NDArray[np.generic], NDArray[np.generic]]:
    """The rows of ``X``, of which there may be none, and their users and items."""
    arr = check_rows(X, ensure_min_samples=0)
    if len(arr):
        users, items, _ = check_interactions(arr, y)
        return arr, users, items
    if arr.shape[1] < _N_COLUMNS:
        raise ValueError(
            "X must have at least 2 columns (user identifiers, item identifiers), "
            f"got {arr.shape[1]}."
        )
    ids = drop_time(arr)
    return arr, ids[:, 0], ids[:, 1]


def _union_ids(parts: list[NDArray[np.generic]]) -> NDArray[np.generic]:
    """The sorted distinct identifiers of ``parts``, some of which may hold none."""
    held = [part for part in parts if len(part)]
    if not held:
        return parts[0][:0]
    return factorize(concat_ids(held))[0]


def _id_pairs(exclude_interactions: ArrayLike | None) -> NDArray[np.generic] | None:
    """``exclude_interactions`` as identifier pairs, whatever columns followed them."""
    if exclude_interactions is None:
        return None
    arr = check_rows(exclude_interactions, ensure_min_samples=0)
    return drop_time(arr) if arr.shape[1] > _N_COLUMNS else arr


def _excluding(
    exclude_interactions: NDArray[np.generic] | None,
    queries: NDArray[np.generic],
    rows: NDArray[np.intp],
    items: NDArray[np.generic],
) -> NDArray[np.generic] | None:
    """``exclude_interactions`` and the items the lists of ``queries`` already hold."""
    if len(rows) == 0:
        return exclude_interactions
    taken = stack_pairs(queries[rows], items)
    if exclude_interactions is None or len(exclude_interactions) == 0:
        return taken
    if exclude_interactions.dtype == taken.dtype:
        return np.concatenate([exclude_interactions, taken])
    return np.concatenate([exclude_interactions.astype(object), taken.astype(object)])


def _split_lists(
    base: tuple[NDArray[np.intp], NDArray[np.generic]],
    inserted: tuple[NDArray[np.intp], NDArray[np.generic]],
    n_queries: int,
    head: int,
    n_kept: int,
) -> tuple[NDArray[np.bool_], NDArray[np.intp], NDArray[np.generic], NDArray[np.bool_]]:
    """What base keeps of the head, the slots after it, and the rest of base's list.

    Returns the mask of base's kept head, the rows and items of ``inserted`` that fill
    the slots, and the mask of base's remaining items.
    """
    base_rows, base_items = base
    rows, items = inserted
    n_base = np.bincount(base_rows, minlength=n_queries)
    n_top = np.minimum(n_base, n_kept)
    top = ranks_in_rows(base_rows, n_queries) < n_top[base_rows]
    fresh = not_among(rows, items, base_rows[top], base_items[top])
    rows, items = rows[fresh], items[fresh]
    slotted = ranks_in_rows(rows, n_queries) < np.maximum(head - n_top, 0)[rows]
    rows, items = rows[slotted], items[slotted]
    rest = ~top & not_among(base_rows, base_items, rows, items)
    return top, rows, items, rest


def _in_query_order(
    parts: list[tuple[NDArray[np.intp], NDArray[np.generic]]], fallback: NDArray[np.generic]
) -> tuple[NDArray[np.intp], NDArray[np.generic]]:
    """The ``(rows, items)`` parts as one pair, grouped by query.

    A stable sort by query keeps the parts in order, and each one's ranking. ``fallback``
    types the items when every part is empty.
    """
    rows = np.concatenate([part_rows for part_rows, _ in parts])
    held = [part_items for _, part_items in parts if len(part_items)]
    items = concat_ids(held) if held else fallback
    order = np.argsort(rows, kind="stable")
    return rows[order], items[order]


def _served_lists(
    rows: NDArray[np.intp],
    items: NDArray[np.generic],
    n_queries: int,
    n_recommendations: int,
    dtype: np.dtype[np.generic],
) -> tuple[NDArray[np.generic], NDArray[np.floating]]:
    """Lists of exactly ``n_recommendations`` items as ``recommend`` returns them.

    ``rows`` is ascending and names the query of each item, best first within a query;
    what a query has past ``n_recommendations`` is cut.
    """
    counts = np.bincount(rows, minlength=n_queries)
    short = np.flatnonzero(counts < n_recommendations)
    if short.size:
        query = int(short[0])
        raise ValueError(
            f"Cannot recommend {n_recommendations} items: query {query} has only "
            f"{counts[query]} eligible items."
        )
    keep = ranks_in_rows(rows, n_queries) < n_recommendations
    served = np.empty((n_queries, n_recommendations), dtype=dtype)
    served[:] = items[keep].reshape(n_queries, n_recommendations)
    scores = np.tile(np.arange(n_recommendations, 0, -1, dtype=np.float64), (n_queries, 1))
    return served, scores


class ItemListRecommender(RecommenderMixin, BaseEstimator):
    """Recommend from a fixed list of items, best first, the same to every user.

    The list is given, not learned: an editorial selection, a catalog in the order of
    its release, the items nobody has been shown yet. So it holds what no interaction
    names -- which every recommender fitted from interactions cannot recommend -- and it
    serves a user it has never seen, and can be fitted on no interactions at all. The
    interactions of ``fit`` only say what each user has seen.

    It is the last source of a :class:`~skrecsys.compose.Backfill`, which it keeps from
    ever running short, and with ``rotate`` the source of the items a
    :class:`~skrecsys.compose.ReservedSlots` explores.

    Parameters
    ----------
    items : array-like of shape (n_items,)
        Distinct item identifiers, best first.
    rotate : bool, default=False
        Whether each user starts at a place of their own in the list and reads it round
        from there. Without it every user is offered the head of the list, and its tail
        is reached by no one; with it the starts are spread evenly over the list, so
        every item is shown about as often as any other. The start depends on the user
        and ``random_state`` alone: ``recommend`` changes nothing in the model, and a
        user asking twice gets the same list.
    random_state : int, default=None
        Only with ``rotate``: an integer >= 0 that moves every user's start, ``None``
        being 0. Change it with each refit -- the version of the model, say -- for a
        user to be shown other items by each.

    Attributes
    ----------
    item_ids_ : ndarray of shape (n_items_,)
        Sorted identifiers of the list.
    rank_ : ndarray of shape (n_items_,)
        The place of each of ``item_ids_`` in the list, from 0.
    user_ids_ : ndarray of shape (n_users_,)
        Sorted users seen during ``fit``.
    n_users_, n_items_ : int
    interactions_ : scipy.sparse.csr_array of shape (n_users_, n_items_)
        Which items of the list each user has seen. Interactions with other items are
        dropped.

    Examples
    --------
    >>> from skrecsys.compose import ItemListRecommender
    >>> X = [["u1", "a"], ["u2", "b"]]
    >>> rec = ItemListRecommender(["new", "b", "a", "old"]).fit(X)
    >>> rec.recommend(["u1", "u2", "nobody"], n_recommendations=2)[0].tolist()
    [['new', 'b'], ['new', 'a'], ['new', 'b']]

    With ``rotate``, users start at different places:

    >>> spread = ItemListRecommender(list("abcdefgh"), rotate=True).fit(X)
    >>> spread.recommend(["u1", "u7"], n_recommendations=3, exclude_seen=False)[0].tolist()
    [['c', 'd', 'e'], ['f', 'g', 'h']]
    """

    #: Read by :func:`skrecsys.base.serves_unknown_users`: the list does not depend on
    #: the user, and a user never seen has seen nothing.
    _serves_unknown_users = True

    def __init__(
        self,
        items: ArrayLike = (),
        *,
        rotate: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.items = items
        self.rotate = rotate
        self.random_state = random_state

    @override
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Index the list, and note what each user of the interactions ``X`` has seen.

        Parameters
        ----------
        X : array-like of shape (n_interactions, 2 + n_context)
            ``X[:, 0]`` contains user identifiers, ``X[:, 1]`` item identifiers; further
            columns are ignored. It may have no rows, as an array of shape ``(0, 2)``.
        y : array-like of shape (n_interactions,), default=None
            Interaction values, which are not used.

        Returns
        -------
        self : object
        """
        check_bool(self.rotate, "rotate")
        check_int(self.random_state, "random_state", min_value=0, allow_none=True)
        listed = np.asarray(self.items)
        if not listed.size:
            raise ValueError("items must hold at least one item identifier.")
        listed = check_ids(listed, name="items")
        self.item_ids_, codes = factorize(listed)
        if len(self.item_ids_) != len(listed):
            raise ValueError("items holds duplicate identifiers.")
        self.rank_ = np.empty(len(listed), dtype=np.intp)
        self.rank_[codes] = np.arange(len(listed))

        _, users, items = _interaction_ids(X, y)
        _check_feature_names(self, X, reset=True)
        if len(users):
            self.user_ids_, user_codes = factorize(users)
            positions, known = lookup_ids(items, self.item_ids_, name="item")
        else:
            self.user_ids_, user_codes = users, np.empty(0, dtype=np.intp)
            positions, known = user_codes, np.empty(0, dtype=bool)
        self.n_users_ = len(self.user_ids_)
        self.n_items_ = len(self.item_ids_)
        seen = sp.csr_array(
            (np.ones(int(known.sum()), dtype=bool), (user_codes[known], positions[known])),
            shape=(self.n_users_, self.n_items_),
        )
        seen.sum_duplicates()
        seen.sort_indices()
        self.interactions_ = seen
        return self

    def _starts(self, queries: NDArray[np.generic]) -> NDArray[np.intp]:
        """The place in the list each query starts reading it from: 0 without ``rotate``."""
        if not self.rotate:
            return np.zeros(len(queries), dtype=np.intp)
        salt = 0 if self.random_state is None else int(self.random_state)
        starts = np.floor(stable_unit_hash(queries, salt) * self.n_items_).astype(np.intp)
        return np.minimum(starts, self.n_items_ - 1)

    @override
    def _score_queries(
        self, X: ArrayLike, item_indices: NDArray[np.intp], *, exclude_seen: bool
    ) -> tuple[NDArray[np.floating], sp.csr_array]:
        queries = check_ids(X)
        # An item's place as each query reads the list, from 0 where the query starts.
        places = (
            self.rank_[item_indices][None, :] - self._starts(queries)[:, None]
        ) % self.n_items_
        scores = (self.n_items_ - places).astype(np.float64)
        return scores, self._excluded_by_seen(queries, item_indices, exclude_seen=exclude_seen)

    @override
    def _excluded_by_seen(
        self, queries: NDArray[np.generic], item_indices: NDArray[np.intp], *, exclude_seen: bool
    ) -> sp.csr_array:
        if not exclude_seen or not self.n_users_:
            return sp.csr_array((len(queries), len(item_indices)), dtype=bool)
        # Unknown users read the one row appended past the fitted ones, which is empty.
        rows, known = lookup_ids(queries, self.user_ids_, name="user")
        rows = np.where(known, rows, self.n_users_)
        seen = self.interactions_
        padded = sp.csr_array(
            (seen.data, seen.indices, np.append(seen.indptr, seen.indptr[-1])),
            shape=(self.n_users_ + 1, self.n_items_),
        )
        return seen_among(padded, rows, item_indices)

    def predict(self, X: ArrayLike) -> NDArray[np.floating]:
        """Score user-item pairs as ``recommend`` scores them: ``n_items_`` for the first.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            User-item pairs; the items must be in the list, the users may be anyone.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        users, items, _ = check_interactions(X)
        _check_feature_names(self, X, reset=False)
        places = self.rank_[encode_ids(items, self.item_ids_, name="item")]
        return (self.n_items_ - (places - self._starts(users)) % self.n_items_).astype(np.float64)


class Backfill(RecommenderMixin, NamedComponentsEstimator[Recommender]):
    """Fill each list from the first recommender, then from the next what it lacks.

    Every recommender is asked in turn, only about the queries whose list is still
    short and only for items the list does not hold, until it has
    ``n_recommendations``. A recommender that cannot answer a query -- a user it was not
    fitted on, unless it serves those (see :func:`skrecsys.base.serves_unknown_users`)
    -- or has too few items for it is no error: the next one goes on. So a list is as
    personal as the data allows and always full, as long as the last recommender has
    enough for everyone, which an :class:`~skrecsys.compose.ItemListRecommender` of the
    whole catalog has.

    This is what ``recommend`` raising on a short list asks a caller to do, done once:
    a :class:`~skrecsys.compose.Cascade` reorders ``n_retrieved`` candidates and has no
    more, a neighbourhood model knows only the items someone interacted with, and none
    of them knows a user who arrived after ``fit``.

    Parameters
    ----------
    recommenders : list of recommenders or of (name, recommender) tuples
        The sources, the one whose items come first leading. Cloned and fitted on the
        same interactions as ``recommenders_``. Unnamed ones are named after their class
        in lower case, numbered when a class repeats; names address nested parameters,
        as in ``popular__weighting``.
    skip_insufficient : bool, default=False
        Whether a recommender whose ``fit`` raises
        :class:`~skrecsys.exceptions.InsufficientDataError` -- there are no
        interactions yet, or too few for the ranker of a
        :class:`~skrecsys.compose.Cascade` -- is left out rather than failing the fit.
        The others serve without it, and ``skipped_`` names it. Any other error is
        raised as it is.

    Attributes
    ----------
    recommenders_ : list of (name, recommender) tuples
        The fitted recommenders.
    skipped_ : list of str
        Names of the recommenders left out by ``skip_insufficient``.
    user_ids_, item_ids_ : ndarray
        The union of those of the recommenders.
    n_users_, n_items_ : int

    Notes
    -----
    ``candidates`` may name any fitted item; each recommender is given those it knows.
    The recommenders retrieve by user: a query context in ``X`` is accepted and not
    passed on, and there is no ``as_of``. An ``X`` without rows is passed on as it is,
    which only a recommender that needs no interactions accepts.

    Examples
    --------
    >>> from skrecsys.compose import Backfill, ItemListRecommender
    >>> from skrecsys.recommendation import ItemKNNRecommender, MostPopularRecommender
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "b"], ["u2", "c"], ["u3", "c"]]
    >>> rec = Backfill(
    ...     [
    ...         ("personal", ItemKNNRecommender()),
    ...         ("popular", MostPopularRecommender()),
    ...         ("catalog", ItemListRecommender(["z", "y", "c", "b", "a"])),
    ...     ]
    ... ).fit(X)
    >>> rec.recommend(["u1", "new-user"], n_recommendations=3)[0].tolist()
    [['c', 'z', 'y'], ['b', 'c', 'a']]

    Before the first interaction only the catalog can be fitted, and it serves alone:

    >>> import numpy as np
    >>> empty = Backfill(rec.recommenders, skip_insufficient=True).fit(np.empty((0, 2), dtype=str))
    >>> empty.skipped_
    ['personal', 'popular']
    >>> empty.recommend(["u1"], n_recommendations=3)[0].tolist()
    [['z', 'y', 'c']]
    """

    _components_param = "recommenders"
    _component_kind = "a recommender"

    def __init__(self, recommenders: RecommenderList, *, skip_insufficient: bool = False) -> None:
        self.recommenders: RecommenderList = recommenders
        self.skip_insufficient = skip_insufficient

    @override
    def _is_component(self, value: object) -> TypeIs[Recommender]:
        return is_recommender(value)

    @override
    def _components(self) -> RecommenderList:
        return self.recommenders

    @property
    def _serves_unknown_users(self) -> bool:
        """Whether any recommender answers unknown users, which the composite then does too."""
        return any(serves_unknown_users(recommender) for _, recommender in self._named())

    def _check_params(self) -> list[tuple[str, Recommender]]:
        if not self.recommenders:
            raise ValueError("Backfill needs at least one recommender.")
        named = self._named()
        for _, recommender in named:
            if not is_recommender(recommender):
                raise TypeError(f"{type(recommender).__name__} is not a recommender.")
        check_bool(self.skip_insufficient, "skip_insufficient")
        return named

    @override
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit every recommender on the interactions ``X``.

        Parameters
        ----------
        X : array-like of shape (n_interactions, 2 + n_context)
            ``X[:, 0]`` contains user identifiers, ``X[:, 1]`` item identifiers; any
            further columns are the query context, which every recommender is fitted
            with. It may have no rows, as an array of shape ``(0, 2)``.
        y : array-like of shape (n_interactions,), default=None
            Interaction values; ``None`` gives every interaction weight 1.

        Returns
        -------
        self : object

        Raises
        ------
        InsufficientDataError
            If a recommender raises it and ``skip_insufficient`` is false, or if every
            recommender does.
        """
        named = self._check_params()
        rows, _, _ = _interaction_ids(X, y)
        _check_feature_names(self, X, reset=True)
        self.recommenders_: list[tuple[str, FittedRecommender]] = []
        self.skipped_: list[str] = []
        for name, recommender in named:
            try:
                fitted = fit_clone(recommender, rows, y)
            except InsufficientDataError:
                if not self.skip_insufficient:
                    raise
                self.skipped_.append(name)
            else:
                self.recommenders_.append((name, fitted))
        if not self.recommenders_:
            raise InsufficientDataError(
                f"No recommender of the Backfill could be fitted: {self.skipped_} had too "
                "little data. End the list with a recommender that needs none, such as an "
                "ItemListRecommender."
            )
        fitted_parts = [recommender for _, recommender in self.recommenders_]
        self.user_ids_ = _union_ids([rec.user_ids_ for rec in fitted_parts])
        self.item_ids_ = _union_ids([rec.item_ids_ for rec in fitted_parts])
        self.n_users_ = len(self.user_ids_)
        self.n_items_ = len(self.item_ids_)
        return self

    def _candidates(self, candidates: ArrayLike | None) -> NDArray[np.generic] | None:
        """``candidates``, validated against the items any recommender knows."""
        if candidates is None:
            return None
        return self.item_ids_[self._candidate_indices(candidates)]

    @override
    @traced_recommend
    def recommend(  # pylint: disable=too-many-locals
        self,
        X: ArrayLike,
        *,
        n_recommendations: int = 10,
        candidates: ArrayLike | None = None,
        exclude_seen: bool = True,
        exclude_interactions: ArrayLike | None = None,
    ) -> tuple[NDArray[np.generic], NDArray[np.floating]]:
        """Return the items of the first recommender, topped up from the others in turn.

        Parameters are those of :meth:`skrecsys.base.RecommenderMixin.recommend`, and
        are passed to every recommender. The scores restate the order of each list,
        ``n_recommendations`` down to 1: those of the recommenders are not comparable.

        Raises
        ------
        ValueError
            If the recommenders together have fewer than ``n_recommendations`` eligible
            items for a query.
        """
        check_is_fitted(self)
        check_n_recommendations(n_recommendations)
        queries, _ = check_queries(X)
        known = self._candidates(candidates)
        excluded = _id_pairs(exclude_interactions)
        missing = np.full(len(queries), n_recommendations, dtype=np.int64)
        rows = np.empty(0, dtype=np.intp)
        items: NDArray[np.generic] = np.empty(0, dtype=self.item_ids_.dtype)
        for name, recommender in self.recommenders_:
            if not missing.any():
                break
            found_rows, found = retrieve_lists(
                recommender,
                queries,
                missing,
                candidates=known,
                exclude_seen=exclude_seen,
                exclude_interactions=_excluding(excluded, queries, rows, items),
                source=name,
            )
            # A recommender is told what the lists hold, and one that adds items of its
            # own all the same -- the business rules of a Cascade -- is not believed.
            fresh = not_among(found_rows, found, rows, items)
            found_rows, found = found_rows[fresh], found[fresh]
            missing -= np.bincount(found_rows, minlength=len(queries))
            rows = np.concatenate([rows, found_rows])
            items = concat_ids([items, found]) if len(items) else found
        # A stable sort by query keeps the recommenders in order, and each one's ranking.
        order = np.argsort(rows, kind="stable")
        return _served_lists(
            rows[order], items[order], len(queries), n_recommendations, self.item_ids_.dtype
        )

    @override
    def _count_eligible(
        self,
        X: ArrayLike,
        *,
        candidates: ArrayLike | None = None,
        exclude_seen: bool = True,
        exclude_interactions: ArrayLike | None = None,
    ) -> NDArray[np.int64]:
        """The most any one recommender has for each query: no more than the list can hold.

        What the recommenders have together is known only by putting the lists together,
        so this is a lower bound, exact whenever one recommender knows every item -- the
        catalog a backfill ends with.
        """
        check_is_fitted(self)
        queries, _ = check_queries(X)
        return _most_eligible(
            [recommender for _, recommender in self.recommenders_],
            queries,
            candidates=self._candidates(candidates),
            exclude_seen=exclude_seen,
            exclude_interactions=_id_pairs(exclude_interactions),
        )

    def predict(self, X: ArrayLike) -> NDArray[np.floating]:
        """Score user-item pairs, each with the first recommender that can score it.

        A recommender can score a pair whose user and item it was fitted on. The scores
        of different recommenders are not comparable, as in a
        :class:`~skrecsys.compose.Switch`.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            User-item pairs; the items must be known to some recommender.

        Returns
        -------
        scores : ndarray of shape (n_samples,)

        Raises
        ------
        ValueError
            If an item is unknown, or no recommender knows both the user and the item
            of a pair.
        """
        check_is_fitted(self)
        users, items, _ = check_interactions(X)
        _check_feature_names(self, X, reset=False)
        encode_ids(items, self.item_ids_, name="item")
        pairs = stack_pairs(users, items)
        scores = np.full(len(pairs), np.nan)
        for _, recommender in self.recommenders_:
            unscored = np.flatnonzero(np.isnan(scores))
            if not unscored.size:
                break
            scores[unscored] = score_pairs(recommender, pairs[unscored])
        unscored = np.isnan(scores)
        if unscored.any():
            sample = pairs[unscored][:5].tolist()
            raise ValueError(f"No recommender knows both the user and the item of {sample}.")
        return scores


class ReservedSlots(RecommenderMixin, BaseEstimator):
    """Give a few of the first positions of ``base``'s lists to items of ``inserted``.

    For exploration: a recommender shows what it already believes in, so an item nobody
    was shown never gets the interactions that would let it be recommended. Reserving
    positions for such items is how they earn them, at a known price -- ``n_slots`` of
    the ``head`` positions users react to.

    Of the first ``head`` positions of a list, ``base`` fills all but the last
    ``n_slots``, which go to the best items of ``inserted`` the list does not hold yet;
    the rest of ``base``'s items follow. Nothing is lost at either end: positions
    ``inserted`` has no item for stay with ``base``, and head positions ``base`` has no
    item for go to ``inserted`` too.

    Parameters
    ----------
    base : recommender
        Whose lists are served. Cloned and fitted as ``base_``.
    inserted : recommender
        Whose items take the reserved positions; for exploration, an
        :class:`~skrecsys.compose.ItemListRecommender` of the items to explore with
        ``rotate=True``, so that users are shown different ones. Cloned and fitted on
        the same interactions as ``inserted_``.
    n_slots : int, default=3
        Positions reserved in the head of each list, at most ``head``.
    head : int, default=10
        How many of the first positions count as the head: what a user sees without
        scrolling. The reserved positions are its last ``n_slots`` whatever
        ``n_recommendations`` is, so a list of ``head - n_slots`` items or fewer is
        ``base``'s alone.

    Attributes
    ----------
    base_, inserted_ : recommender
        The fitted parts.
    user_ids_, item_ids_ : ndarray
        The union of those of the parts.
    n_users_, n_items_ : int

    Notes
    -----
    ``candidates`` may name any fitted item; each part is given those it knows. The
    parts retrieve by user: a query context in ``X`` is accepted and not passed on, and
    there is no ``as_of``. The scores restate the order of each list.

    Examples
    --------
    >>> from skrecsys.compose import ItemListRecommender, ReservedSlots
    >>> from skrecsys.recommendation import MostPopularRecommender
    >>> X = [["u1", "a"], ["u2", "a"], ["u2", "b"], ["u3", "b"], ["u3", "c"], ["u4", "b"]]
    >>> rec = ReservedSlots(
    ...     MostPopularRecommender(), ItemListRecommender(["x", "y"]), n_slots=1, head=3
    ... ).fit(X)
    >>> rec.recommend(["new-user"], n_recommendations=4)[0].tolist()
    [['b', 'a', 'x', 'c']]
    """

    def __init__(
        self,
        base: Recommender,
        inserted: Recommender,
        *,
        n_slots: int = 3,
        head: int = 10,
    ) -> None:
        self.base = base
        self.inserted = inserted
        self.n_slots = n_slots
        self.head = head

    @property
    def _serves_unknown_users(self) -> bool:
        """Whether a part answers unknown users, which the composite then does too."""
        return serves_unknown_users(self.base) or serves_unknown_users(self.inserted)

    @override
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit both parts on the interactions ``X``.

        Parameters
        ----------
        X : array-like of shape (n_interactions, 2 + n_context)
            ``X[:, 0]`` contains user identifiers, ``X[:, 1]`` item identifiers; any
            further columns are the query context, which both parts are fitted with. It
            may have no rows, as an array of shape ``(0, 2)``, if both parts accept that.
        y : array-like of shape (n_interactions,), default=None
            Interaction values; ``None`` gives every interaction weight 1.

        Returns
        -------
        self : object
        """
        for name in ("base", "inserted"):
            check_component(getattr(self, name), name, is_recommender, "a recommender")
        head = check_int(self.head, "head", min_value=1)
        check_int(self.n_slots, "n_slots", min_value=0, max_value=head)
        rows, _, _ = _interaction_ids(X, y)
        _check_feature_names(self, X, reset=True)
        self.base_ = fit_clone(self.base, rows, y)
        self.inserted_ = fit_clone(self.inserted, rows, y)
        self.user_ids_ = _union_ids([self.base_.user_ids_, self.inserted_.user_ids_])
        self.item_ids_ = _union_ids([self.base_.item_ids_, self.inserted_.item_ids_])
        self.n_users_ = len(self.user_ids_)
        self.n_items_ = len(self.item_ids_)
        return self

    def _candidates(self, candidates: ArrayLike | None) -> NDArray[np.generic] | None:
        """``candidates``, validated against the items either part knows."""
        if candidates is None:
            return None
        return self.item_ids_[self._candidate_indices(candidates)]

    @override
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
        """Return ``base``'s lists with items of ``inserted`` in the reserved positions.

        Parameters are those of :meth:`skrecsys.base.RecommenderMixin.recommend`, and
        are passed to both parts. The scores restate the order of each list,
        ``n_recommendations`` down to 1.

        Raises
        ------
        ValueError
            If the parts together have fewer than ``n_recommendations`` eligible items
            for a query.
        """
        check_is_fitted(self)
        check_n_recommendations(n_recommendations)
        queries, _ = check_queries(X)
        n_queries = len(queries)
        known = self._candidates(candidates)
        excluded = _id_pairs(exclude_interactions)
        # The reserved positions are where they are however long a list is asked for: a
        # list cut before them holds none.
        head = min(int(self.head), n_recommendations)
        n_kept = max(int(self.head) - int(self.n_slots), 0)
        base_rows, base_items = retrieve_lists(
            self.base_,
            queries,
            n_recommendations,
            candidates=known,
            exclude_seen=exclude_seen,
            exclude_interactions=excluded,
            source="base",
        )
        rows, items = retrieve_lists(
            self.inserted_,
            queries,
            head,
            candidates=known,
            exclude_seen=exclude_seen,
            exclude_interactions=excluded,
            source="inserted",
        )
        top, rows, items, rest = _split_lists(
            (base_rows, base_items), (rows, items), n_queries, head, n_kept
        )
        all_rows, all_items = _in_query_order(
            [
                (base_rows[top], base_items[top]),
                (rows, items),
                (base_rows[rest], base_items[rest]),
            ],
            base_items,
        )
        return _served_lists(
            all_rows, all_items, n_queries, n_recommendations, self.item_ids_.dtype
        )

    @override
    def _count_eligible(
        self,
        X: ArrayLike,
        *,
        candidates: ArrayLike | None = None,
        exclude_seen: bool = True,
        exclude_interactions: ArrayLike | None = None,
    ) -> NDArray[np.int64]:
        """The most either part has for each query: no more than the list can hold.

        A lower bound, exact whenever one part knows every item; see
        :meth:`Backfill._count_eligible`.
        """
        check_is_fitted(self)
        queries, _ = check_queries(X)
        return _most_eligible(
            [self.base_, self.inserted_],
            queries,
            candidates=self._candidates(candidates),
            exclude_seen=exclude_seen,
            exclude_interactions=_id_pairs(exclude_interactions),
        )

    def predict(self, X: ArrayLike) -> NDArray[np.floating]:
        """Score user-item pairs with ``base``; the reserved positions play no part.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            User-item pairs, as ``base`` takes them.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        _check_feature_names(self, X, reset=False)
        return predict_pairs(self.base_, check_rows(X))


def _most_eligible(
    recommenders: list[FittedRecommender],
    queries: NDArray[np.generic],
    *,
    candidates: NDArray[np.generic] | None,
    exclude_seen: bool,
    exclude_interactions: NDArray[np.generic] | None,
) -> NDArray[np.int64]:
    """The most items any one of ``recommenders`` could return to each query."""
    counts = np.zeros(len(queries), dtype=np.int64)
    for recommender in recommenders:
        served = served_queries(recommender, queries)
        own = candidates
        if own is not None:
            own = own[lookup_ids(own, recommender.item_ids_, name="item")[1]]
            if len(own) == 0:
                continue
        if len(served) == 0:
            continue
        eligible = recommender._count_eligible(  # pylint: disable=protected-access
            queries[served],
            candidates=own,
            exclude_seen=exclude_seen,
            exclude_interactions=exclude_interactions,
        )
        counts[served] = np.maximum(counts[served], eligible)
    return counts
