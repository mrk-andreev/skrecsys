"""Features of candidate user-item pairs, for a ranker to learn from.

A feature component is fitted on interactions and turns candidate ``pairs`` -- shape
``(n_pairs, 2)``, laid out like the ``X`` of ``fit`` -- into a float matrix with one row per
pair. Everything here is numpy: a DataFrame is accepted wherever a table is, and its
column names are kept, but pandas is never imported.

Inside a recommender constructed with ``time=True``, interactions and pairs carry a third
column, the time: of an interaction, or the time a pair is ranked as of, missing where
the latest data is wanted. Components that do not read time ignore it.
"""

from collections.abc import Callable
from typing import Self, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

from skrecsys._typing import FeatureNamer, Features, Recommender, clone_as, override
from skrecsys.base import FeaturesMixin, fit_clone, is_features, is_recommender, predict_pairs
from skrecsys.compose._candidates import stack_columns
from skrecsys.compose._named import (
    ComponentList,
    check_component_list,
    name_components,
    nested_params,
    set_nested_params,
)
from skrecsys.utils._param_validation import check_component, check_real
from skrecsys.utils.validation import (
    check_optionally_timed,
    check_times,
    drop_time,
    factorize,
    lookup_ids,
)

_KINDS = ("user", "item")

#: The columns of ``pairs`` a :class:`JoinDynamicFeatures` key may be built from.
_KEY_PARTS = ("user", "item", "time")

#: The column of ``pairs`` that holds the time, when there is one.
_TIME_COLUMN = 2

#: How an error names what a :class:`ConcatFeatures` holds.
_FEATURE_KIND = "a feature component"

#: A joined table holds an identifier column and at least one feature column.
_MIN_TABLE_COLUMNS = 2

#: What ``transform`` returns: one row per pair, one column per feature.
_MATRIX_NDIM = 2


def check_pairs(pairs: ArrayLike) -> NDArray[np.generic]:
    """Validate candidate pairs: ``(n_pairs, 2)`` identifiers, or ``(n_pairs, 3)`` with time."""
    arr = check_array(pairs, dtype=None, ensure_all_finite=False, ensure_min_samples=0)
    if arr.shape[1] not in (len(_KINDS), len(_KEY_PARTS)):
        raise ValueError(
            "pairs must have 2 columns (user, item identifiers) or 3 (user, item, time), "
            f"got {arr.shape[1]}."
        )
    return arr


def pair_ids(pairs: ArrayLike) -> NDArray[np.generic]:
    """The identifier columns of candidate pairs, shape ``(n_pairs, 2)``."""
    return drop_time(check_pairs(pairs))


def pair_times(pairs: NDArray[np.generic]) -> NDArray[np.generic]:
    """The times of validated candidate pairs, missing where the latest data is wanted."""
    if pairs.shape[1] != len(_KEY_PARTS):
        raise ValueError(
            "pairs carry no time; construct the recommender with time=True and fit it on "
            "[user, item, time] rows."
        )
    return check_times(pairs[:, _TIME_COLUMN], name="pair times", allow_missing=True)


def _untimed(X: ArrayLike) -> ArrayLike:
    """Interactions without their time column, for a part that does not read time."""
    arr = check_array(X, dtype=None, ensure_all_finite=False)
    return drop_time(arr) if arr.shape[1] == len(_KEY_PARTS) else X


def _check_kind(kind: str) -> int:
    """The column of ``pairs`` that ``kind`` reads."""
    if kind not in _KINDS:
        raise ValueError(f"kind must be 'user' or 'item', got {kind!r}.")
    return _KINDS.index(kind)


def _check_kinds(kind: str | tuple[str, ...]) -> tuple[int, ...]:
    """The columns of ``pairs`` a :class:`JoinDynamicFeatures` key reads, in order.

    One part, a tuple of distinct parts, or distinct parts joined by ``"-"``; the parts
    are ``"user"``, ``"item"`` and ``"time"``.
    """
    parts = tuple(kind.split("-")) if isinstance(kind, str) else kind
    if (
        not isinstance(parts, tuple | list)
        or not parts
        or any(p not in _KEY_PARTS for p in parts)
        or len(set(parts)) != len(parts)
    ):
        raise ValueError(
            "kind must be 'user', 'item', 'time', or a tuple of distinct ones of them or "
            f"those joined by '-', such as ('item', 'time') or 'user-item-time'; got {kind!r}."
        )
    return tuple(_KEY_PARTS.index(p) for p in parts)


def _distinct_rows(
    columns: list[NDArray[np.generic]],
) -> tuple[NDArray[np.generic], NDArray[np.intp]]:
    """Distinct rows of ``columns`` side by side, sorted column by column, and each row's."""
    uniques, codes = zip(*(factorize(column) for column in columns), strict=True)
    key = np.zeros(len(columns[0]), dtype=np.int64)
    for u, c in zip(uniques, codes, strict=True):
        key = key * len(u) + c
    distinct, positions = factorize(key)
    keys = distinct.astype(np.int64, copy=False)
    stacked = []
    for u in reversed(uniques):
        stacked.append(u[keys % len(u)])
        keys = keys // len(u)
    return stack_columns(stacked[::-1]), positions


class JoinStaticFeatures(FeaturesMixin, BaseEstimator):
    """Join a fixed table of user or item features onto each pair.

    Parameters
    ----------
    kind : {"user", "item"}, default="user"
        Whether the table is keyed by the user or by the item of a pair.
    values : array-like of shape (n_rows, 1 + n_features)
        Column 0 holds the identifiers, the others the features, which must be numeric.
        A DataFrame's column names become the feature names. Identifiers must be
        distinct and need not match the fitted ones.
    missing : {"nan", "error"}, default="nan"
        What an identifier absent from the table gets: a row of NaN, which tree
        rankers handle natively, or a ``ValueError``.

    Attributes
    ----------
    ids_ : ndarray of shape (n_rows,)
        Sorted identifiers of the table.
    table_ : ndarray of shape (n_rows, n_features)
        Features, in the order of ``ids_``.
    feature_names_ : ndarray of str of shape (n_features,)

    Examples
    --------
    >>> import numpy as np
    >>> from skrecsys.compose import JoinStaticFeatures
    >>> table = np.array([["a", 1.0], ["b", 2.0]], dtype=object)
    >>> join = JoinStaticFeatures("item", table).fit()
    >>> join.transform([["u1", "b"], ["u1", "z"]]).tolist()
    [[2.0], [nan]]
    """

    def __init__(self, kind: str = "user", values: ArrayLike | None = None, missing: str = "nan"):
        self.kind = kind
        self.values = values
        self.missing = missing

    def fit(self, X: ArrayLike | None = None, y: ArrayLike | None = None) -> Self:
        """Index the table; the interactions are not used."""
        del X, y
        _check_kind(self.kind)
        if self.missing not in ("nan", "error"):
            raise ValueError(f"missing must be 'nan' or 'error', got {self.missing!r}.")
        if self.values is None:
            raise ValueError("values must be a table of identifiers and features, got None.")
        table = check_array(self.values, dtype=None, ensure_all_finite=False)
        if table.shape[1] < _MIN_TABLE_COLUMNS:
            raise ValueError(
                "values must have an identifier column and at least one feature column, "
                f"got {table.shape[1]} column(s)."
            )
        ids, codes = factorize(table[:, 0])
        if len(ids) != len(table):
            raise ValueError(f"values holds duplicate {self.kind} identifiers.")
        self.ids_ = ids
        self.table_ = np.empty((len(ids), table.shape[1] - 1), dtype=np.float64)
        self.table_[codes] = table[:, 1:].astype(np.float64)
        columns = getattr(self.values, "columns", None)
        names = (
            [str(c) for c in list(columns)[1:]]
            if columns is not None
            else [f"{self.kind}_feature_{i}" for i in range(self.table_.shape[1])]
        )
        self.feature_names_ = np.asarray(names, dtype=object)
        return self

    @override
    def transform(
        self, pairs: ArrayLike, *, scores: ArrayLike | None = None
    ) -> NDArray[np.floating]:
        del scores
        check_is_fitted(self)
        ids = pair_ids(pairs)[:, _check_kind(self.kind)]
        positions, known = lookup_ids(ids, self.ids_, name=self.kind)
        if self.missing == "error" and not np.all(known):
            sample = list(ids[~known][:5])
            raise ValueError(f"No features for {self.kind} identifiers: {sample}.")
        out = np.full((len(ids), self.table_.shape[1]), np.nan)
        out[known] = self.table_[positions[known]]
        return out

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> NDArray[np.object_]:
        """Names of the features ``transform`` returns."""
        del input_features
        check_is_fitted(self)
        return self.feature_names_.copy()


class JoinDynamicFeatures(FeaturesMixin, BaseEstimator):
    """Join user, item, pair or time features computed on demand by ``callback``.

    For features that live elsewhere -- a feature store, a cache, a model -- and cannot
    be materialized as a table when the recommender is fitted. A key that includes the
    time asks for the features as they stood when each pair is ranked, which keeps a
    ranker from learning on values that were known only later.

    Parameters
    ----------
    kind : str or tuple of str, default="user"
        What the callback is asked about, from the parts ``"user"``, ``"item"`` and
        ``"time"``. A single part passes the distinct users, items or times of the pairs,
        an array of shape ``(n_distinct,)``. A tuple such as ``("user", "item")``, or the
        same parts joined by ``"-"`` such as ``"item-time"``, passes their distinct
        combinations, an array of shape ``(n_distinct, n_parts)`` with the columns in the
        order of ``kind``. A key with ``"time"`` needs pairs with a time column, which a
        recommender constructed with ``time=True`` supplies.
    callback : callable
        ``callback(keys) -> array-like of shape (len(keys), n_features)``, called once per
        ``transform`` with the distinct keys described by ``kind``. A time is missing --
        NaN, NaT, or ``None`` among objects -- where the latest features are wanted, as
        for ``recommend`` without ``as_of``. The cut-off is the callback's to decide; a
        point-in-time lookup returns the latest value recorded strictly before the time.
        Times stay numbers or ``datetime64``, except beside identifiers of another type,
        where they are ``datetime`` objects. To pickle the recommender, it must be a
        module-level function or a picklable object, not a lambda.
    n_features : int, default=None
        The width ``callback`` returns. When given, it is checked and names the features.

    Examples
    --------
    >>> import numpy as np
    >>> from skrecsys.compose import JoinDynamicFeatures
    >>> def lengths(ids):
    ...     return np.array([[len(i)] for i in ids], dtype=float)
    >>> join = JoinDynamicFeatures("user", lengths).fit()
    >>> join.transform([["ann", "a"], ["bo", "a"]]).tolist()
    [[3.0], [2.0]]
    >>> def same_initial(pairs):
    ...     return np.array([[u[0] == i[0]] for u, i in pairs], dtype=float)
    >>> join = JoinDynamicFeatures(("user", "item"), same_initial).fit()
    >>> join.transform([["ann", "a"], ["bo", "a"]]).tolist()
    [[1.0], [0.0]]

    Keyed by item and time, on pairs whose third column is the time they are ranked as
    of, NaN for the latest price:

    >>> def price(keys):
    ...     times = keys[:, 1]
    ...     return np.where(np.isnan(times) | (times > 5), 2.0, 1.0)
    >>> join = JoinDynamicFeatures("item-time", price).fit()
    >>> join.transform(np.array([[1, 7, 3], [1, 7, 9], [2, 7, np.nan]])).tolist()
    [[1.0], [2.0], [2.0]]
    """

    def __init__(
        self,
        kind: str | tuple[str, ...] = "user",
        callback: Callable[[NDArray[np.generic]], ArrayLike] | None = None,
        n_features: int | None = None,
    ) -> None:
        self.kind = kind
        self.callback = callback
        self.n_features = n_features

    def fit(self, X: ArrayLike | None = None, y: ArrayLike | None = None) -> Self:
        """Validate the parameters; nothing is learned."""
        del X, y
        self._check_params()
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def _check_params(self) -> tuple[tuple[int, ...], Callable[[NDArray[np.generic]], ArrayLike]]:
        columns = _check_kinds(self.kind)
        if not callable(self.callback):
            raise TypeError(f"callback must be callable, got {type(self.callback).__name__}.")
        return columns, self.callback

    @override
    def transform(
        self, pairs: ArrayLike, *, scores: ArrayLike | None = None
    ) -> NDArray[np.floating]:
        del scores
        columns, callback = self._check_params()
        arr = check_pairs(pairs)
        ids = drop_time(arr)
        keys = [pair_times(arr) if c == _TIME_COLUMN else ids[:, c] for c in columns]
        if len(keys) == 1 and isinstance(self.kind, str):
            distinct, rows = factorize(keys[0])
        else:
            distinct, rows = _distinct_rows(keys)
        values = np.asarray(callback(distinct), dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != _MATRIX_NDIM or len(values) != len(distinct):
            raise ValueError(
                f"callback must return an array of shape ({len(distinct)}, n_features), "
                f"got {values.shape}."
            )
        if self.n_features is not None and values.shape[1] != self.n_features:
            raise ValueError(
                f"callback returned {values.shape[1]} features, expected {self.n_features}."
            )
        return values[rows]

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> NDArray[np.object_]:
        """Names of the features ``transform`` returns; needs ``n_features``."""
        del input_features
        if self.n_features is None:
            raise ValueError("Set n_features to name the features of a JoinDynamicFeatures.")
        prefix = self.kind.replace("-", "_") if isinstance(self.kind, str) else "_".join(self.kind)
        return np.asarray([f"{prefix}_feature_{i}" for i in range(self.n_features)], dtype=object)


class GeneratorScores(FeaturesMixin, BaseEstimator):
    """The score the candidate generator gave each pair, as a single feature.

    Nearly every ranker wants it: it is the whole of what the first stage knew. When a
    :class:`~skrecsys.compose.Cascade` has several generators, the scores have one column
    per generator, and so do the features.

    Parameters
    ----------
    n_generators : int, default=None
        How many generators score the pairs. When given, the width of ``scores`` is
        checked and the features are named ``generator_score_0``, ``generator_score_1``,
        and so on; otherwise a single feature is named ``generator_score``.

    Examples
    --------
    >>> from skrecsys.compose import GeneratorScores
    >>> GeneratorScores().fit().transform([["u", "a"]], scores=[0.5]).tolist()
    [[0.5]]
    >>> two = GeneratorScores(n_generators=2).fit()
    >>> two.transform([["u", "a"]], scores=[[0.5, 0.25]]).tolist()
    [[0.5, 0.25]]
    >>> two.get_feature_names_out().tolist()
    ['generator_score_0', 'generator_score_1']
    """

    def __init__(self, n_generators: int | None = None) -> None:
        self.n_generators = n_generators

    def fit(self, X: ArrayLike | None = None, y: ArrayLike | None = None) -> Self:
        """Nothing is learned."""
        del X, y
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    @override
    def transform(
        self, pairs: ArrayLike, *, scores: ArrayLike | None = None
    ) -> NDArray[np.floating]:
        n_pairs = len(check_pairs(pairs))
        if scores is None:
            raise ValueError("GeneratorScores needs the generator scores of the pairs.")
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != _MATRIX_NDIM or len(values) != n_pairs:
            raise ValueError(
                f"scores must have shape ({n_pairs},) or ({n_pairs}, n_generators), "
                f"got {np.shape(scores)}."
            )
        if self.n_generators is not None and values.shape[1] != self.n_generators:
            raise ValueError(
                f"scores has {values.shape[1]} generator column(s), expected "
                f"n_generators={self.n_generators}."
            )
        return values

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> NDArray[np.object_]:
        """Names of the features ``transform`` returns."""
        del input_features
        if self.n_generators is None:
            return np.asarray(["generator_score"], dtype=object)
        return np.asarray([f"generator_score_{i}" for i in range(self.n_generators)], dtype=object)


def _needs_interactions(component: object, X: ArrayLike | None) -> ArrayLike:
    """``X``, which a component that learns from interactions cannot be fitted without."""
    if X is None:
        raise ValueError(f"{type(component).__name__} learns from interactions; fit it on X.")
    return X


class InteractionCounts(FeaturesMixin, BaseEstimator):
    """How many interactions the user or the item of each pair had during ``fit``.

    A user's count says how much history a generator had to go on; an item's is its
    popularity. An identifier never seen during ``fit`` counts zero.

    Parameters
    ----------
    kind : {"user", "item"}, default="item"
        Whether to count the pair's user or its item.

    Attributes
    ----------
    ids_ : ndarray of shape (n_ids,)
        Sorted identifiers seen during ``fit``.
    counts_ : ndarray of shape (n_ids,)
        Interactions of each, counting a repeated pair each time.

    Examples
    --------
    >>> from skrecsys.compose import InteractionCounts
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "a"]]
    >>> InteractionCounts("item").fit(X).transform([["u9", "a"], ["u9", "z"]]).tolist()
    [[2.0], [0.0]]
    """

    def __init__(self, kind: str = "item") -> None:
        self.kind = kind

    def fit(self, X: ArrayLike | None = None, y: ArrayLike | None = None) -> Self:
        """Count the interactions of every user or item in ``X``."""
        column = _check_kind(self.kind)
        users, items, _, _ = check_optionally_timed(_needs_interactions(self, X), y)
        self.ids_, codes = factorize((users, items)[column])
        self.counts_ = np.bincount(codes, minlength=len(self.ids_)).astype(np.float64)
        return self

    @override
    def transform(
        self, pairs: ArrayLike, *, scores: ArrayLike | None = None
    ) -> NDArray[np.floating]:
        del scores
        check_is_fitted(self)
        ids = pair_ids(pairs)[:, _check_kind(self.kind)]
        positions, known = lookup_ids(ids, self.ids_, name=self.kind)
        return np.where(known, self.counts_[positions], 0.0)[:, None]

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> NDArray[np.object_]:
        """Names of the features ``transform`` returns."""
        del input_features
        return np.asarray([f"{self.kind}_interactions"], dtype=object)


class RecommenderScores(FeaturesMixin, BaseEstimator):
    """The score another recommender gives each pair, as a single feature.

    A ranker reordering one generator's candidates learns most from a second opinion:
    a model with a different inductive bias, whose score it can weigh against the
    generator's. The recommender is fitted on the interactions the features are fitted
    on, which inside a :class:`~skrecsys.compose.Cascade` keeps it off the held-out rows
    the ranker is labelled with.

    Parameters
    ----------
    recommender : recommender
        Any recommender with ``predict``. Cloned and fitted as ``recommender_``.

    Attributes
    ----------
    recommender_ : recommender
        The fitted recommender.

    Notes
    -----
    A pair whose user or item the recommender was not fitted on scores NaN, which tree
    rankers treat as missing, rather than raising as ``predict`` would.

    Examples
    --------
    >>> from skrecsys.compose import RecommenderScores
    >>> from skrecsys.recommendation import ItemKNNRecommender
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "c"]]
    >>> scores = RecommenderScores(ItemKNNRecommender()).fit(X)
    >>> scores.transform([["u1", "c"], ["new", "c"]]).round(3).tolist()
    [[0.707], [nan]]
    """

    def __init__(self, recommender: Recommender | None = None) -> None:
        self.recommender = recommender

    def fit(self, X: ArrayLike | None = None, y: ArrayLike | None = None) -> Self:
        """Fit a clone of ``recommender`` on the interactions ``X``."""
        check_component(self.recommender, "recommender", is_recommender, "a recommender")
        X = _untimed(_needs_interactions(self, X))
        self.recommender_ = fit_clone(cast(Recommender, self.recommender), X, y)
        return self

    @override
    def transform(
        self, pairs: ArrayLike, *, scores: ArrayLike | None = None
    ) -> NDArray[np.floating]:
        del scores
        check_is_fitted(self)
        pairs = pair_ids(pairs)
        known = lookup_ids(pairs[:, 0], self.recommender_.user_ids_, name="user")[1]
        known &= lookup_ids(pairs[:, 1], self.recommender_.item_ids_, name="item")[1]
        out = np.full(len(pairs), np.nan)
        if known.any():
            out[known] = predict_pairs(self.recommender_, pairs[known])
        return out[:, None]

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> NDArray[np.object_]:
        """Names of the features ``transform`` returns."""
        del input_features
        return np.asarray(["recommender_score"], dtype=object)


class SegmentPopularity(FeaturesMixin, BaseEstimator):
    """How popular each pair's item is among users of the same segment as its user.

    The cold-start feature: a user with no history still has a segment -- an age band,
    a country, a signup channel -- and what their segment watches is the best guess at
    what they will. For every segmentation, ``fit`` computes the share of a segment's
    users who interacted with each item, shrunk towards the share among all users by
    ``smoothing`` pseudo-users so that a small segment does not overfit, and
    ``transform`` returns two features per segmentation: that share, and its lift, the
    share over the global share.

    Parameters
    ----------
    segments : array-like of shape (n_users, 1 + n_segmentations)
        Column 0 holds user identifiers, each further column a segment label of any
        hashable type -- the segments of one segmentation. It may name users never seen
        in ``fit``, which is the point: a cold user is looked up here. A DataFrame's
        column names name the features.
    smoothing : float, default=10.0
        Pseudo-users, interacting at the global rate, added to every segment.

    Attributes
    ----------
    share_ : list of ndarray of shape (n_segments, n_items)
        Smoothed share of each segment's users who interacted with each item, one array
        per segmentation.
    global_share_ : ndarray of shape (n_items,)
        Share of all users who interacted with each item.
    item_ids_ : ndarray of shape (n_items,)
        Sorted items seen during ``fit``.

    Notes
    -----
    A pair whose user is missing from ``segments``, or whose item was not seen during
    ``fit``, gets NaN. A segment no fitted user belongs to gets the global share.

    Examples
    --------
    >>> import numpy as np
    >>> from skrecsys.compose import SegmentPopularity
    >>> segments = np.array([["u1", "kid"], ["u2", "kid"], ["u3", "adult"], ["new", "kid"]])
    >>> X = [["u1", "a"], ["u2", "a"], ["u3", "b"]]
    >>> feats = SegmentPopularity(segments, smoothing=0.0).fit(X)
    >>> feats.transform([["new", "a"], ["new", "b"]]).round(2).tolist()
    [[1.0, 1.5], [0.0, 0.0]]
    """

    def __init__(self, segments: ArrayLike | None = None, smoothing: float = 10.0) -> None:
        self.segments = segments
        self.smoothing = smoothing

    def fit(self, X: ArrayLike | None = None, y: ArrayLike | None = None) -> Self:
        """Count what every segment interacted with in ``X``."""
        smoothing = check_real(self.smoothing, "smoothing", min_value=0)
        if self.segments is None:
            raise ValueError("segments must be a table of user identifiers and labels.")
        table = check_array(self.segments, dtype=None, ensure_all_finite=False)
        if table.shape[1] < _MIN_TABLE_COLUMNS:
            raise ValueError(
                "segments must have a user column and at least one segment column, "
                f"got {table.shape[1]} column(s)."
            )
        self.user_ids_, rows = factorize(table[:, 0])
        if len(self.user_ids_) != len(table):
            raise ValueError("segments holds duplicate user identifiers.")

        users, items, _, _ = check_optionally_timed(_needs_interactions(self, X), y)
        self.item_ids_, item_codes = factorize(items)
        fitted_users, user_codes = factorize(users)
        # One count per distinct user-item pair: a share of users, not of interactions.
        pair_codes = np.unique(user_codes.astype(np.int64) * len(self.item_ids_) + item_codes)
        pair_users, pair_items = np.divmod(pair_codes, len(self.item_ids_))
        n_users = len(fitted_users)
        self.global_share_ = np.bincount(pair_items, minlength=len(self.item_ids_)) / n_users

        positions, known = lookup_ids(fitted_users, self.user_ids_, name="user")
        self.codes_ = []
        self.share_ = []
        for column in range(1, table.shape[1]):
            labels, codes = factorize(table[:, column])
            user_codes_of_table = np.empty(len(table), dtype=np.intp)
            user_codes_of_table[rows] = codes
            self.codes_.append(user_codes_of_table)
            # Users outside the table count towards the global share only.
            segment = np.full(n_users, -1, dtype=np.intp)
            segment[known] = user_codes_of_table[positions[known]]
            members = np.bincount(segment[known], minlength=len(labels)).astype(np.float64)
            pair_segment = segment[pair_users]
            inside = pair_segment >= 0
            counts = np.zeros((len(labels), len(self.item_ids_)))
            np.add.at(counts, (pair_segment[inside], pair_items[inside]), 1.0)
            share = (counts + smoothing * self.global_share_) / (members + smoothing)[:, None]
            empty = members + smoothing == 0
            share[empty] = self.global_share_
            self.share_.append(share)
        columns = getattr(self.segments, "columns", None)
        names = (
            [str(c) for c in list(columns)[1:]]
            if columns is not None
            else [f"segment_{i}" for i in range(table.shape[1] - 1)]
        )
        self.feature_names_ = np.asarray(
            [f"{name}_{what}" for name in names for what in ("share", "lift")], dtype=object
        )
        return self

    @override
    def transform(
        self, pairs: ArrayLike, *, scores: ArrayLike | None = None
    ) -> NDArray[np.floating]:
        del scores
        check_is_fitted(self)
        pairs = pair_ids(pairs)
        user_rows, user_known = lookup_ids(pairs[:, 0], self.user_ids_, name="user")
        item_cols, item_known = lookup_ids(pairs[:, 1], self.item_ids_, name="item")
        ok = user_known & item_known
        out = np.full((len(pairs), 2 * len(self.share_)), np.nan)
        global_share = self.global_share_[item_cols[ok]]
        for position, (codes, share) in enumerate(zip(self.codes_, self.share_, strict=True)):
            values = share[codes[user_rows[ok]], item_cols[ok]]
            out[ok, 2 * position] = values
            with np.errstate(divide="ignore", invalid="ignore"):
                out[ok, 2 * position + 1] = np.where(
                    global_share > 0, values / global_share, np.nan
                )
        return out

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> NDArray[np.object_]:
        """Names of the features ``transform`` returns: a share and a lift per column."""
        del input_features
        check_is_fitted(self)
        return self.feature_names_.copy()


#: The ``features`` of :class:`ConcatFeatures`: all bare components or all named ones.
FeatureList: TypeAlias = ComponentList[Features]


class ConcatFeatures(FeaturesMixin, BaseEstimator):
    """Concatenate the features of several components side by side.

    The composite of this module, as ``FeatureUnion`` is of scikit-learn's.

    Parameters
    ----------
    features : list of feature components or of (name, component) tuples
        Each is cloned and fitted as ``features_``. Unnamed components are named after
        their class in lower case, numbered when a class repeats. The names address
        nested parameters: ``name__param``.

    Attributes
    ----------
    features_ : list of (name, component) tuples
        The fitted components.

    Examples
    --------
    >>> import numpy as np
    >>> from skrecsys.compose import ConcatFeatures, GeneratorScores, JoinStaticFeatures
    >>> table = np.array([["a", 1.0], ["b", 2.0]], dtype=object)
    >>> both = ConcatFeatures([JoinStaticFeatures("item", table), GeneratorScores()]).fit()
    >>> both.transform([["u1", "b"]], scores=[0.5]).tolist()
    [[2.0, 0.5]]
    >>> both.get_params()["joinstaticfeatures__kind"]
    'item'
    """

    def __init__(self, features: FeatureList) -> None:
        self.features: FeatureList = features

    def _named(self) -> list[tuple[str, Features]]:
        """``features`` as (name, component) pairs, naming what the caller did not."""
        return name_components(self.features, "features")

    @override
    def get_params(self, deep: bool = True) -> dict[str, object]:
        params = dict[str, object](super().get_params(deep=False))
        if not deep:
            return params
        return params | nested_params(self._named())

    @override
    def set_params(self, **params: object) -> Self:
        if "features" in params:
            self.features = check_component_list(
                params.pop("features"), "features", is_features, _FEATURE_KIND
            )
        if not params:
            return self
        named = self._named()
        replaced = set_nested_params(type(self).__name__, named, params, is_features, _FEATURE_KIND)
        if replaced:
            explicit = isinstance(self.features[0], tuple)
            self.features = named if explicit else [component for _, component in named]
        return self

    def fit(self, X: ArrayLike | None = None, y: ArrayLike | None = None) -> Self:
        """Fit a clone of every component."""
        if not self.features:
            raise ValueError("ConcatFeatures needs at least one feature component.")
        self.features_ = []
        for name, component in self._named():
            if not is_features(component):
                raise TypeError(f"{type(component).__name__} is not a feature component.")
            self.features_.append((name, clone_as(component).fit(X, y)))
        return self

    @override
    def transform(
        self, pairs: ArrayLike, *, scores: ArrayLike | None = None
    ) -> NDArray[np.floating]:
        check_is_fitted(self)
        pairs = check_pairs(pairs)
        blocks = [component.transform(pairs, scores=scores) for _, component in self.features_]
        return np.hstack([np.asarray(b, dtype=np.float64).reshape(len(pairs), -1) for b in blocks])

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> NDArray[np.object_]:
        """Names of the features ``transform`` returns, prefixed with the component name."""
        del input_features
        check_is_fitted(self)
        names = []
        for name, component in self.features_:
            if not isinstance(component, FeatureNamer):
                raise TypeError(f"{name} does not name its features: no get_feature_names_out.")
            names.extend(f"{name}__{feature}" for feature in component.get_feature_names_out())
        return np.asarray(names, dtype=object)
