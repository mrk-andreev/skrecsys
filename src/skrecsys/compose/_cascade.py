"""Two-stage recommendation: generators propose candidates, a ranker orders them."""

from collections.abc import Callable
from typing import Self, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import _check_feature_names, check_array, check_is_fitted

from skrecsys._typing import (
    CrossValidator,
    Features,
    FittedRecommender,
    Ranker,
    Recommender,
    clone_as,
    override,
)
from skrecsys.base import (
    RecommenderMixin,
    check_n_recommendations,
    first_time_of,
    fit_clone,
    is_features,
    is_ranker,
    is_recommender,
    predict_pairs,
    serves_unknown_users,
)
from skrecsys.compose._candidates import (
    concat_ids,
    rank_within_groups,
    retrieve,
    retrieve_union,
    score_pairs,
    top_k_per_group,
    with_time,
)
from skrecsys.compose._named import (
    ComponentList,
    check_component_list,
    name_components,
    nested_params,
    set_nested_params,
)
from skrecsys.utils._param_validation import check_bool, check_component, check_int, check_real
from skrecsys.utils.validation import (
    check_as_of,
    check_ids,
    check_interactions,
    drop_time,
    encode_ids,
    factorize,
    lookup_ids,
)

#: Candidate pairs featurized and ranked at once by ``recommend``: the feature matrix of
#: a block is this many rows, whatever the number of queries asked about.
_PAIRS_PER_BLOCK = 1_000_000

#: The types a fractional ``split`` may have; anything else is a splitter.
_REAL = (int, float, np.integer, np.floating)

#: How an error names what ``generator`` holds.
_GENERATOR_KIND = "a recommender"

#: The ``generator`` of :class:`Cascade`: one recommender, or a list of them.
Generators: TypeAlias = Recommender | ComponentList[Recommender]

#: The ``postprocess`` of :class:`Cascade`: ranked candidate lists in, final lists out.
Postprocess: TypeAlias = Callable[
    [NDArray[np.generic], NDArray[np.float64], NDArray[np.int64]],
    tuple[ArrayLike, ArrayLike, ArrayLike],
]


class Cascade(RecommenderMixin, BaseEstimator):
    """Retrieve candidates with ``generator``, then reorder them with ``ranker``.

    ``recommend`` asks the generator for ``n_retrieved`` items per query, turns each
    candidate pair into a row of ``features``, and returns the ``n_recommendations`` pairs
    the ranker scores highest. A query with fewer eligible items gets them all as
    candidates.

    ``generator`` may also be a list of recommenders, whose candidates are merged: their
    lists are interleaved round-robin by rank, an item already proposed is skipped, and
    each query keeps the first ``n_retrieved`` distinct items. The generator scores then
    have one column per generator, in list order: the score a generator gave a candidate
    it retrieved, otherwise its ``predict`` of the pair, and NaN where it cannot score it
    -- an unknown user or item, or no ``predict``. A generator that does not serve
    unknown users (see :func:`skrecsys.base.serves_unknown_users`) is asked only about
    the users it was fitted on, so ``[ItemKNNRecommender(), MostPopularRecommender()]``
    serves a cold user the popular items alone.

    The ranker must not learn from interactions the generator was fitted on: the
    generator would place them among its candidates with a confidence it cannot have at
    serving time. So ``fit`` holds interactions out:

    1. ``split`` divides ``X`` into a fitting part and a held-out part.
    2. A generator fitted on the first part proposes candidates to the users of the
       second; a candidate is relevant when it is a held-out interaction with a positive
       value. Users whose candidates hold no relevant item teach nothing and are dropped.
       A user whose rows were all held out is kept only when the generator serves users
       it has never seen (see :func:`skrecsys.base.serves_unknown_users`): with
       :class:`~skrecsys.model_selection.ColdStartSplit` and a
       :class:`~skrecsys.recommendation.MostPopularRecommender` generator, the ranker
       learns from exactly the candidates a cold user is served.
    3. ``features`` fitted on the first part describe the candidates, and ``ranker`` is
       fitted on them, one group per user.
    4. The generator and the features are fitted again on all of ``X`` for serving.

    Parameters
    ----------
    generator : recommender, or list of recommenders or of (name, recommender) tuples
        The first stage. Fitted twice, the second time as ``generator_``. Unnamed
        generators of a list are named after their class in lower case, numbered when a
        class repeats; the names address nested parameters: ``generator__name__param``.
    features : feature component
        Turns candidate pairs into ranker inputs, for example a
        :class:`~skrecsys.compose.ConcatFeatures` of joined tables and
        :class:`~skrecsys.compose.GeneratorScores`. Fitted as ``features_``.
    ranker : ranker
        The second stage, following :class:`skrecsys.base.RankerMixin`:
        :class:`~skrecsys.compose.PointwiseRanker` around any scikit-learn classifier, or
        a third-party ranker of :mod:`skrecsys.integrations`, such as
        :class:`skrecsys.integrations.catboost.CatBoostRanker` from the ``catboost`` extra.
        Fitted as ``ranker_``.
    n_retrieved : int, default=100
        How many items the generator retrieves per query for the ranker to reorder --
        the width of the first stage, fixed at ``fit`` because the ranker learns from
        lists of this length. With several generators it is the budget of distinct items
        after merging, not a per-generator count. It is the most ``n_recommendations``
        can be, and is unrelated to the ``candidates`` argument of ``recommend``, which restricts
        *which* items may be retrieved.
    split : float or cross-validator, default=0.2
        How ``fit`` holds interactions out. A float in (0, 1) holds out that fraction of
        every user's interactions, rounded down, taking the *latest* ones: by their time
        when ``time`` is true, otherwise by row order, so sort ``X`` by timestamp when
        you have one. Otherwise any scikit-learn splitter, such as
        :class:`~skrecsys.model_selection.WarmStartKFold` or ``ShuffleSplit``, whose
        first ``(train, test)`` split is used;
        :class:`~skrecsys.model_selection.ColdStartSplit` trains a cold-start ranker.
    time : bool, default=False
        Whether interactions carry their time, as a third column of ``X``: ``[user,
        item, time]``, the time a number or a ``datetime64``. The generators still see
        ``[user, item]``; the features see the time of each candidate pair, so that a
        :class:`~skrecsys.compose.JoinDynamicFeatures` keyed by time can look features up
        as they stood then. In ``fit``, a held-out user's candidates are ranked as of
        their earliest held-out interaction -- what they would have been shown just
        before it -- and ``recommend`` ranks as of ``as_of``.
    postprocess : callable, default=None
        Business rules applied on top of the ranker by ``recommend``:
        ``postprocess(pairs, scores, groups) -> (pairs, scores, groups)``. It receives every
        candidate of each query, best first by the ranker -- ``pairs`` laid out as the
        features see them, ``scores`` the ranker's, ``groups`` the length of each query's
        list -- and returns the lists to serve in the same layout: reordered, shortened, or
        holding items that were never candidates, with scores of its choosing. The first
        ``n_recommendations`` of each returned list are recommended, and a list shorter
        than that raises, so leave ``n_retrieved`` room for what it drops. It never takes
        part in ``fit`` or ``predict``, is called once per block of queries, and must be a
        module-level function or another picklable callable for the cascade to pickle.

    Attributes
    ----------
    generator_, features_, ranker_ : estimator
        The fitted stages; ``generator_`` only when ``generator`` is a single recommender.
    generators_ : list of (name, recommender) tuples
        The fitted generators, a single one named ``"generator"``.
    user_ids_, item_ids_ : ndarray
        Those of ``generator_``, or the union of those of the generators.
    n_users_, n_items_ : int
    n_ranker_groups_ : int
        Users the ranker was fitted on.
    time_dtype_ : numpy.dtype
        Only when ``time`` is true: the dtype of the fitted times, which ``as_of`` must
        match in kind -- numbers or datetimes.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skrecsys.compose import Cascade, GeneratorScores, PointwiseRanker
    >>> from skrecsys.recommendation import ItemKNNRecommender
    >>> X = [[u, i] for u in range(20) for i in (u % 5, u % 5 + 1, u % 5 + 2)]
    >>> rec = Cascade(
    ...     ItemKNNRecommender(),
    ...     GeneratorScores(),
    ...     PointwiseRanker(LogisticRegression()),
    ...     n_retrieved=3,
    ...     split=0.4,
    ... ).fit(X)
    >>> rec.recommend([0], n_recommendations=2)[0].shape
    (1, 2)

    Several generators, whose scores :class:`~skrecsys.compose.GeneratorScores` turns
    into one feature each. A cold user's ItemKNN score is NaN, so the ranker must handle
    missing values:

    >>> from sklearn.ensemble import HistGradientBoostingClassifier
    >>> from skrecsys.recommendation import MostPopularRecommender
    >>> rec = Cascade(
    ...     [ItemKNNRecommender(), MostPopularRecommender()],
    ...     GeneratorScores(n_generators=2),
    ...     PointwiseRanker(HistGradientBoostingClassifier(max_iter=5)),
    ...     n_retrieved=4,
    ...     split=0.4,
    ... ).fit(X)
    >>> rec.recommend([0, 1000], n_recommendations=2)[0].shape
    (2, 2)

    Business rules on top of the ranker, here never recommending item 3:

    >>> import numpy as np
    >>> def drop_item_3(pairs, scores, groups):
    ...     keep = pairs[:, 1] != 3
    ...     group_of_row = np.repeat(np.arange(len(groups)), groups)
    ...     sizes = np.bincount(group_of_row[keep], minlength=len(groups))
    ...     return pairs[keep], scores[keep], sizes
    >>> rec = Cascade(
    ...     ItemKNNRecommender(),
    ...     GeneratorScores(),
    ...     PointwiseRanker(LogisticRegression()),
    ...     n_retrieved=3,
    ...     split=0.4,
    ...     postprocess=drop_item_3,
    ... ).fit(X)
    >>> bool((rec.recommend(range(20), n_recommendations=2)[0] != 3).all())
    True
    """

    def __init__(
        self,
        generator: Generators,
        features: Features,
        ranker: Ranker,
        n_retrieved: int = 100,
        split: float | CrossValidator = 0.2,
        *,
        time: bool = False,
        postprocess: Postprocess | None = None,
    ) -> None:
        self.generator: Generators = generator
        self.features = features
        self.ranker = ranker
        self.n_retrieved = n_retrieved
        self.split = split
        self.time = time
        self.postprocess = postprocess

    def _generators(self) -> list[tuple[str, Recommender]]:
        """``generator`` as (name, recommender) pairs; a single one is named "generator"."""
        if isinstance(self.generator, list):
            return name_components(self.generator, "generator")
        return [("generator", self.generator)]

    @property
    def _serves_unknown_users(self) -> bool:
        """A cascade serves unknown users when one of its generators does."""
        return any(serves_unknown_users(generator) for _, generator in self._generators())

    @override
    def get_params(self, deep: bool = True) -> dict[str, object]:
        params = dict[str, object](super().get_params(deep=deep))
        if deep and isinstance(self.generator, list):
            nested = nested_params(self._generators())
            params |= {f"generator__{key}": value for key, value in nested.items()}
        return params

    @override
    def set_params(self, **params: object) -> Self:
        if "generator" in params:
            generator = params.pop("generator")
            if isinstance(generator, list):
                generator = check_component_list(
                    generator, "generator", is_recommender, _GENERATOR_KIND
                )
            self.generator = cast(Generators, generator)
        if isinstance(self.generator, list):
            nested = {
                key.removeprefix("generator__"): params.pop(key)
                for key in list(params)
                if key.startswith("generator__")
            }
            if nested:
                named = self._generators()
                owner = type(self).__name__
                if set_nested_params(owner, named, nested, is_recommender, _GENERATOR_KIND):
                    explicit = isinstance(self.generator[0], tuple)
                    self.generator = named if explicit else [rec for _, rec in named]
        if params:
            super().set_params(**params)
        return self

    def _check_params(self) -> None:
        if isinstance(self.generator, list):
            if not self.generator:
                raise ValueError("generator must hold at least one recommender, got [].")
            for _, generator in self._generators():
                check_component(generator, "generator", is_recommender, _GENERATOR_KIND)
        else:
            check_component(self.generator, "generator", is_recommender, _GENERATOR_KIND)
        for name, check, kind in (
            ("features", is_features, "a feature component"),
            ("ranker", is_ranker, "a ranker"),
        ):
            check_component(getattr(self, name), name, check, kind)
        check_int(self.n_retrieved, "n_retrieved", min_value=1)
        check_bool(self.time, "time")
        if self.postprocess is not None and not callable(self.postprocess):
            raise TypeError(
                f"postprocess must be callable or None, got {type(self.postprocess).__name__}."
            )
        if isinstance(self.split, _REAL):
            check_real(
                self.split,
                "split",
                min_value=0,
                max_value=1,
                min_inclusive=False,
                max_inclusive=False,
            )

    def _check_interactions(
        self, X: ArrayLike, y: ArrayLike | None
    ) -> tuple[
        NDArray[np.generic], NDArray[np.generic], NDArray[np.float64], NDArray[np.generic] | None
    ]:
        """Users, items, weights and -- with ``time`` only -- the times of ``X``."""
        if self.time:
            return check_interactions(X, y, time=True)
        users, items, weights = check_interactions(X, y)
        return users, items, weights, None

    def _split_rows(
        self,
        X: NDArray[np.generic],
        y: ArrayLike | None,
        users: NDArray[np.generic],
        times: NDArray[np.generic] | None = None,
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """Row indices of the fitting part and of the held-out part."""
        split = self.split
        if not isinstance(split, _REAL):
            train, test = next(iter(split.split(X, y)))
            return np.asarray(train, dtype=np.intp), np.asarray(test, dtype=np.intp)
        _, codes = factorize(users)
        # Each user's rows by time when there is one -- ties in row order -- else by row.
        order = np.argsort(codes, kind="stable") if times is None else np.lexsort((times, codes))
        counts = np.bincount(codes)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        # A row's position among its user's rows, in time or row order.
        position = np.empty(len(codes), dtype=np.intp)
        position[order] = np.arange(len(codes)) - np.repeat(starts, counts)
        n_held = np.floor(split * counts).astype(np.intp)
        held = position >= (counts - n_held)[codes]
        return np.flatnonzero(~held), np.flatnonzero(held)

    @override
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the generator, the features and the ranker from interactions.

        Parameters
        ----------
        X : array-like of shape (n_interactions, 2) or (n_interactions, 3)
            ``X[:, 0]`` contains user identifiers, ``X[:, 1]`` item identifiers and, when
            ``time`` is true, ``X[:, 2]`` the time of each interaction.
        y : array-like of shape (n_interactions,), default=None
            Interaction values; ``None`` gives every interaction weight 1. A held-out
            interaction is relevant when its value is positive.

        Returns
        -------
        self : object
        """
        self._check_params()
        users, items, weights, times = self._check_interactions(X, y)
        _check_feature_names(self, X, reset=True)
        X_arr = check_array(X, dtype=None, ensure_all_finite=False)
        # The generators never see the time; the features see all of X.
        X_ids = drop_time(X_arr) if self.time else X_arr
        y_arr = None if y is None else weights
        train, held = self._split_rows(X_arr, y_arr, users, times)
        if not len(train) or not len(held):
            raise ValueError(
                f"split left {len(train)} interactions to fit on and {len(held)} to hold "
                "out; the ranker needs both."
            )
        X_train, y_train = X_arr[train], None if y_arr is None else y_arr[train]

        generators = [fit_clone(rec, X_ids[train], y_train) for _, rec in self._generators()]
        relevant = held[weights[held] > 0]
        # A splitter may hold out all of a user's rows. Such a user is a cold user, and the
        # ranker learns from them only when the generator can serve one -- which is what
        # teaches a cold-start ranker the candidates it will really see.
        queries = np.unique(users[relevant])
        if not self._serves_unknown_users:
            queries = np.intersect1d(queries, users[train])
        if not len(queries):
            raise ValueError("No held-out interaction belongs to a user with training rows.")
        pairs, scores, groups, kept = self._retrieve(generators, queries, min_retrieved=0)
        labels = _is_held_out(pairs, users[relevant], items[relevant], queries)
        if times is not None:
            # A user is ranked as of the start of their held-out interactions: the
            # features then are what the ranker could have known at the time.
            as_of = first_time_of(users[held], times[held], queries)
            pairs = with_time(pairs, np.repeat(as_of[kept], groups))
        pairs, scores, labels, groups = _groups_with_a_positive(pairs, scores, labels, groups)
        if not len(groups):
            raise ValueError(
                "No held-out interaction was among the generated candidates, so the ranker "
                "has nothing to learn from; increase n_retrieved."
            )
        if labels.min() == labels.max():
            raise ValueError(
                "Every generated candidate of the held-out users is relevant, so the ranker "
                "has nothing to tell apart; the catalog is too small for n_retrieved, or "
                "split holds out too much."
            )

        features = clone_as(self.features).fit(X_train, y_train)
        self.ranker_ = clone_as(self.ranker).fit(
            features.transform(pairs, scores=scores), labels, groups=groups
        )
        self.n_ranker_groups_ = len(groups)

        self.generators_ = [
            (name, fit_clone(rec, X_ids, y_arr)) for name, rec in self._generators()
        ]
        fitted = [rec for _, rec in self.generators_]
        if isinstance(self.generator, list):
            self.user_ids_ = factorize(concat_ids([rec.user_ids_ for rec in fitted]))[0]
            self.item_ids_ = factorize(concat_ids([rec.item_ids_ for rec in fitted]))[0]
        else:
            self.generator_ = fitted[0]
            self.user_ids_ = self.generator_.user_ids_
            self.item_ids_ = self.generator_.item_ids_
        self.features_ = clone_as(self.features).fit(X_arr, y_arr)
        if times is not None:
            self.time_dtype_ = times.dtype
        elif hasattr(self, "time_dtype_"):
            del self.time_dtype_
        self.n_users_ = len(self.user_ids_)
        self.n_items_ = len(self.item_ids_)
        return self

    def _retrieve(
        self,
        generators: list[FittedRecommender],
        queries: NDArray[np.generic],
        *,
        min_retrieved: int,
        candidates: ArrayLike | None = None,
        exclude_seen: bool = True,
        exclude_interactions: ArrayLike | None = None,
        first_query: int = 0,
    ) -> tuple[NDArray[np.generic], NDArray[np.float64], NDArray[np.int64], NDArray[np.intp]]:
        """Candidates of ``queries``: one generator's, or the merge of several."""
        if isinstance(self.generator, list):
            return retrieve_union(
                generators,
                queries,
                n_retrieved=int(self.n_retrieved),
                min_retrieved=min_retrieved,
                candidates=candidates,
                exclude_seen=exclude_seen,
                exclude_interactions=exclude_interactions,
                first_query=first_query,
            )
        return retrieve(
            generators[0],
            queries,
            n_retrieved=int(self.n_retrieved),
            min_retrieved=min_retrieved,
            candidates=candidates,
            exclude_seen=exclude_seen,
            exclude_interactions=exclude_interactions,
            first_query=first_query,
        )

    def _fitted_generators(self) -> list[FittedRecommender]:
        return [rec for _, rec in self.generators_]

    def _untimed(self, exclude_interactions: ArrayLike | None) -> ArrayLike | None:
        """``exclude_interactions`` as the generators take them, without a time column.

        With ``time``, the rows may be laid out like ``X`` -- the events since ``fit``,
        timed -- or as bare pairs.
        """
        if not self.time or exclude_interactions is None:
            return exclude_interactions
        arr = check_array(
            exclude_interactions, dtype=None, ensure_all_finite=False, ensure_min_samples=0
        )
        return drop_time(arr) if arr.shape[1] == 3 else arr  # noqa: PLR2004

    def _score_candidates(
        self, pairs: NDArray[np.generic], scores: NDArray[np.float64], groups: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        features = self.features_.transform(pairs, scores=scores)
        ranked = np.asarray(self.ranker_.predict(features, groups=groups), dtype=np.float64)
        if ranked.shape != (len(pairs),):
            raise ValueError(
                f"{type(self.ranker_).__name__}.predict must return one score per row, "
                f"got shape {ranked.shape} for {len(pairs)} rows."
            )
        return ranked

    def _postprocess(
        self,
        pairs: NDArray[np.generic],
        scores: NDArray[np.float64],
        groups: NDArray[np.int64],
        n_recommendations: int,
    ) -> tuple[NDArray[np.generic], NDArray[np.float64]]:
        """The first ``n_recommendations`` of each list ``postprocess`` makes of the ranked ones.

        ``pairs`` and ``scores`` are best first within each group.
        """
        postprocess = cast(Postprocess, self.postprocess)
        out = postprocess(pairs, scores, groups)
        if not isinstance(out, tuple) or len(out) != 3:  # noqa: PLR2004
            raise TypeError("postprocess must return a (pairs, scores, groups) tuple.")
        new_pairs = np.asarray(out[0])
        new_scores = np.asarray(out[1], dtype=np.float64)
        new_groups = np.asarray(out[2])
        if new_groups.shape != groups.shape or new_groups.dtype.kind not in "iu":
            raise ValueError(
                f"postprocess must return one integer group size per query, {len(groups)} "
                f"here, got {new_groups.dtype} of shape {new_groups.shape}."
            )
        n_rows = int(new_groups.sum())
        if new_pairs.ndim != 2 or len(new_pairs) != n_rows or new_scores.shape != (n_rows,):  # noqa: PLR2004
            raise ValueError(
                f"postprocess returned group sizes summing to {n_rows}, pairs of shape "
                f"{new_pairs.shape} and scores of shape {new_scores.shape}; they must agree."
            )
        if np.isnan(new_scores).any():
            raise ValueError("postprocess returned NaN scores.")
        queries = pairs[np.concatenate([[0], np.cumsum(groups)[:-1]]), 0]
        if not np.all(new_pairs[:, 0] == np.repeat(queries, new_groups)):
            raise ValueError(
                "postprocess must keep each query's rows in its own group, in the order the "
                "queries came in."
            )
        short = np.flatnonzero(new_groups < n_recommendations)
        if len(short):
            query = queries[short[0]]
            raise ValueError(
                f"postprocess left query {query} with {int(new_groups[short[0]])} items, "
                f"fewer than n_recommendations={n_recommendations}; increase n_retrieved "
                "or make it drop fewer."
            )
        starts = np.concatenate([[0], np.cumsum(new_groups)[:-1]]).astype(np.intp)
        rows = starts[:, None] + np.arange(n_recommendations)
        return new_pairs[rows, 1], new_scores[rows]

    @override
    def recommend(
        self,
        X: ArrayLike,
        *,
        n_recommendations: int = 10,
        candidates: ArrayLike | None = None,
        exclude_seen: bool = True,
        exclude_interactions: ArrayLike | None = None,
        as_of: ArrayLike | None = None,
    ) -> tuple[NDArray[np.generic], NDArray[np.floating]]:
        """Return the items the ranker scores highest among each query's candidates.

        Parameters are those of :meth:`skrecsys.base.RecommenderMixin.recommend`; the
        filters apply to the generator, so an excluded item is never a candidate.
        Scores are the ranker's, or those ``postprocess`` returns when there is one. Ties
        are resolved by fitted item order. With ``time``,
        ``exclude_interactions`` may be laid out like ``X``, time included, and:

        as_of : scalar or array-like of shape (n_queries,), default=None
            Only with ``time``: the time the queries are ranked as of, one for all or
            one per query, of the kind the fitted times are -- numbers or datetimes.
            It becomes the time of each candidate pair for the features to read.
            ``None``, or a missing time, ranks as of the latest data, as when serving;
            a past time replays what would have been served then.

        Raises
        ------
        ValueError
            If ``n_recommendations`` exceeds ``n_retrieved``, or a query has fewer than
            ``n_recommendations`` eligible items, or ``postprocess`` leaves it fewer, or
            ``as_of`` is given without ``time``.
        """
        check_is_fitted(self)
        check_n_recommendations(n_recommendations)
        if n_recommendations > self.n_retrieved:
            raise ValueError(
                f"n_recommendations={n_recommendations} exceeds n_retrieved={self.n_retrieved}."
            )
        if as_of is not None and not self.time:
            raise ValueError("as_of needs a Cascade constructed with time=True.")
        queries = check_ids(X)
        query_times = check_as_of(as_of, len(queries), self.time_dtype_) if self.time else None
        exclude_interactions = self._untimed(exclude_interactions)
        items = np.empty((len(queries), n_recommendations), dtype=self.item_ids_.dtype)
        top_scores = np.empty((len(queries), n_recommendations), dtype=np.float64)
        # What postprocess serves may be any item, so its blocks are joined at the end:
        # the fitted identifiers' dtype -- fixed-width strings, say -- may not hold them.
        served: list[tuple[NDArray[np.generic], NDArray[np.float64]]] = []
        generators = self._fitted_generators()
        # Each generator retrieves the full budget before the merge drops duplicates.
        size = max(1, _PAIRS_PER_BLOCK // (int(self.n_retrieved) * len(generators)))
        for start in range(0, len(queries), size):
            stop = min(start + size, len(queries))
            pairs, scores, groups, kept = self._retrieve(
                generators,
                queries[start:stop],
                min_retrieved=n_recommendations,
                candidates=candidates,
                exclude_seen=exclude_seen,
                exclude_interactions=exclude_interactions,
                first_query=start,
            )
            featurized = (
                pairs
                if query_times is None
                else with_time(pairs, np.repeat(query_times[start:stop][kept], groups))
            )
            ranked = self._score_candidates(featurized, scores, groups)
            positions = lookup_ids(pairs[:, 1], self.item_ids_, name="item")[0]
            if self.postprocess is not None:
                order = rank_within_groups(ranked, groups, positions)
                served.append(
                    self._postprocess(featurized[order], ranked[order], groups, n_recommendations)
                )
                continue
            best = top_k_per_group(ranked, groups, positions, n_recommendations)
            items[start:stop] = pairs[best, 1]
            top_scores[start:stop] = ranked[best]
        if self.postprocess is not None and served:
            served_items = np.concatenate([i for i, _ in served])
            if served_items.dtype == object and self.item_ids_.dtype != object:
                # With time the pairs are objects, users, items and times side by side;
                # rebuild the items' own dtype, wide enough for any item postprocess added.
                served_items = np.asarray(served_items.tolist())
            return served_items, np.concatenate([s for _, s in served])
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
        exclude_interactions = self._untimed(exclude_interactions)
        if not isinstance(self.generator, list):
            counts = self.generator_._count_eligible(
                X,
                candidates=candidates,
                exclude_seen=exclude_seen,
                exclude_interactions=exclude_interactions,
            )
            return np.minimum(counts, int(self.n_retrieved))
        # What the union holds is known only by merging the generators' candidates.
        queries = check_ids(X)
        counts = np.zeros(len(queries), dtype=np.int64)
        generators = self._fitted_generators()
        size = max(1, _PAIRS_PER_BLOCK // (int(self.n_retrieved) * len(generators)))
        for start in range(0, len(queries), size):
            stop = min(start + size, len(queries))
            _, _, groups, kept = retrieve_union(
                generators,
                queries[start:stop],
                n_retrieved=int(self.n_retrieved),
                min_retrieved=0,
                candidates=candidates,
                exclude_seen=exclude_seen,
                exclude_interactions=exclude_interactions,
            )
            counts[start + kept] = groups
        return counts

    def predict(self, X: ArrayLike) -> NDArray[np.floating]:
        """Score user-item pairs with the ranker; ``postprocess`` plays no part.

        The pairs of each user form one group, and the generator scores them as it would
        have scored them as candidates. Several generators score each pair with
        ``predict``, NaN where one cannot.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2) or (n_samples, 3)
            User-item pairs; with a single generator, both identifiers must be known to it.
            With ``time``, a third column holds the time each pair is scored as of,
            missing for the latest data.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        if self.time:
            X_arr = check_array(X, dtype=None, ensure_all_finite=False)
            if X_arr.shape[1] != 3:  # noqa: PLR2004
                raise ValueError(
                    "X must have exactly 3 columns (user identifiers, item identifiers, "
                    f"times), got {X_arr.shape[1]}."
                )
            pairs = drop_time(X_arr)
            users = check_interactions(pairs)[0]
            times = check_as_of(X_arr[:, 2], len(pairs), self.time_dtype_)
            featurized = with_time(pairs, times)
        else:
            users, _, _ = check_interactions(X)
            pairs = check_array(X, dtype=None, ensure_all_finite=False)
            featurized = pairs
        _check_feature_names(self, X, reset=False)
        if isinstance(self.generator, list):
            # Each generator scores what it can; a pair none of them can is an error, as
            # it is for a single generator.
            encode_ids(pairs[:, 1], self.item_ids_, name="item")
            if not self._serves_unknown_users:
                encode_ids(pairs[:, 0], self.user_ids_, name="user")
            generated = np.column_stack(
                [score_pairs(rec, pairs) for rec in self._fitted_generators()]
            )
        else:
            generated = predict_pairs(self.generator_, pairs)
        _, codes = factorize(users)
        order = np.argsort(codes, kind="stable")
        groups = np.bincount(codes).astype(np.int64)
        out = np.empty(len(users), dtype=np.float64)
        out[order] = self._score_candidates(featurized[order], generated[order], groups)
        return out


def _is_held_out(
    pairs: NDArray[np.generic],
    held_users: NDArray[np.generic],
    held_items: NDArray[np.generic],
    queries: NDArray[np.generic],
) -> NDArray[np.float64]:
    """1.0 for each candidate pair that is a held-out interaction, else 0.0.

    Pairs are matched as integer keys over the queries and the items the candidates
    name, which keeps the whole comparison in numpy.
    """
    item_ids, item_codes = factorize(pairs[:, 1])
    user_codes = lookup_ids(pairs[:, 0], queries, name="user")[0]
    held_user, user_known = lookup_ids(held_users, queries, name="user")
    held_item, item_known = lookup_ids(held_items, item_ids, name="item")
    keep = user_known & item_known
    n_items = max(len(item_ids), 1)
    candidate_keys = user_codes.astype(np.int64) * n_items + item_codes
    held_keys = held_user[keep].astype(np.int64) * n_items + held_item[keep]
    return np.isin(candidate_keys, held_keys).astype(np.float64)


def _groups_with_a_positive(
    pairs: NDArray[np.generic],
    scores: NDArray[np.float64],
    labels: NDArray[np.float64],
    groups: NDArray[np.int64],
) -> tuple[NDArray[np.generic], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Drop the groups without a relevant candidate: they rank nothing above anything."""
    group_of_row = np.repeat(np.arange(len(groups)), groups)
    has_positive = np.bincount(group_of_row, weights=labels, minlength=len(groups)) > 0
    rows = has_positive[group_of_row]
    return pairs[rows], scores[rows], labels[rows], groups[has_positive]
