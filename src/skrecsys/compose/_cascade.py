"""Two-stage recommendation: generators propose candidates, a ranker orders them."""

from collections.abc import Callable
from typing import Self, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import _check_feature_names, check_is_fitted

from skrecsys._tracing import (
    Tracer,
    active_tracer,
    feature_names,
    span,
    traced_recommend,
    untraced,
)
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
    first_row_of,
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
from skrecsys.compose._rankers import (
    Candidates,
    fit_features,
    fit_ranker,
    predict_ranker,
    trace_ranker,
)
from skrecsys.exceptions import InsufficientDataError
from skrecsys.model_selection._split import positions_in_user
from skrecsys.utils._param_validation import check_bool, check_component, check_int, check_real
from skrecsys.utils.validation import (
    check_as_of,
    check_interactions,
    check_queries,
    check_rows,
    drop_time,
    encode_ids,
    factorize,
    interaction_context,
    lookup_ids,
)

#: The system columns of ``X``: identifiers, and the time with ``time=True``.
_N_COLUMNS = 2
_N_TIMED_COLUMNS = 3

#: Candidate pairs featurized and ranked at once by ``recommend``: the feature matrix of
#: a block is this many rows, whatever the number of queries asked about.
_PAIRS_PER_BLOCK = 1_000_000

#: The types a fractional ``split`` may have; anything else is a splitter.
_REAL = (int, float, np.integer, np.floating)

#: How an error names what ``generator`` holds.
_GENERATOR_KIND = "a recommender"

#: What ``postprocess`` returns: ``(pairs, scores, groups)``.
_N_POSTPROCESS_OUTPUTS = 3

#: The ``pairs`` of ``postprocess``: one row per pair.
_MATRIX_NDIM = 2

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

    Columns of ``X`` after the system ones are the *query context* of each interaction
    -- ``[user, item, context...]``, or ``[user, item, time, context...]`` with ``time``
    -- such as the page or the device a request came from. The generators never see it:
    they retrieve by user. The features do, as the ``context`` of each candidate pair,
    and a :class:`~skrecsys.compose.JoinDynamicFeatures` keyed by ``"context"`` turns it
    into ranker inputs. In ``fit``, a held-out user's candidates carry the context of their
    first held-out interaction -- the earliest by time with ``time``, otherwise the first
    by row -- which is the request they are ranked for. ``recommend`` takes queries as a
    matrix ``[user, context...]``; a vector of users asks without context, which the
    features see as NaN. What they make of the NaN reaches the ranker, and a ranker that
    does not accept missing values, such as :class:`~sklearn.linear_model.LogisticRegression`,
    then raises. Without ``time``, a third column of ``X`` is context like any further one,
    not a time.

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
    n_context_ : int
        Context columns of the fitted ``X``, 0 without any. The context of a query
        passed to ``recommend`` must have as many.

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

    Query context: each interaction carries the shelf it was made from, 0 or 1, and shelf
    ``s`` holds the items of parity ``s``. A feature of the item and the context together
    tells the ranker whether a candidate is on the shelf the query comes from:

    >>> from skrecsys.compose import ConcatFeatures, JoinDynamicFeatures
    >>> X = [[u, i, u % 2] for u in range(60) for i in range(8) if i % 2 == u % 2 and i != u % 5]
    >>> def on_shelf(keys):
    ...     return np.array([[item % 2 == shelf] for item, shelf in keys], dtype=float)
    >>> rec = Cascade(
    ...     MostPopularRecommender(),
    ...     ConcatFeatures([GeneratorScores(), JoinDynamicFeatures("item-context", on_shelf)]),
    ...     PointwiseRanker(LogisticRegression()),
    ...     n_retrieved=8,
    ...     split=0.5,
    ... ).fit(X)
    >>> rec.recommend([[0, 0], [0, 1]], n_recommendations=2, exclude_seen=False)[0].tolist()
    [[0, 2], [1, 3]]
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
        position, counts = positions_in_user(codes, times)
        n_held = np.floor(split * counts).astype(np.intp)
        held = position >= (counts - n_held)[codes]
        return np.flatnonzero(~held), np.flatnonzero(held)

    @override
    @untraced()
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the generator, the features and the ranker from interactions.

        Parameters
        ----------
        X : array-like of shape (n_interactions, n_system + n_context)
            ``X[:, 0]`` contains user identifiers, ``X[:, 1]`` item identifiers and, when
            ``time`` is true, ``X[:, 2]`` the time of each interaction. Any further
            columns are the query context of each interaction.
        y : array-like of shape (n_interactions,), default=None
            Interaction values; ``None`` gives every interaction weight 1. A held-out
            interaction is relevant when its value is positive.

        Returns
        -------
        self : object

        Raises
        ------
        InsufficientDataError
            If the ranker has nothing to learn from: ``split`` leaves one of its parts
            empty, no held-out interaction is among the candidates of a user the
            generator serves, or every candidate is relevant. A :class:`ValueError`.
        """
        self._check_params()
        users, items, weights, times = self._check_interactions(X, y)
        _check_feature_names(self, X, reset=True)
        X_arr = check_rows(X)
        # The generators see neither the time nor the context. The features see the
        # time in X, and the context beside the pairs they describe.
        X_ids = drop_time(X_arr)
        context = interaction_context(X_arr, time=self.time)
        X_features = X_arr[:, :_N_TIMED_COLUMNS] if self.time else X_ids
        y_arr = None if y is None else weights
        train, held = self._split_rows(X_arr, y_arr, users, times)
        if not len(train) or not len(held):
            raise InsufficientDataError(
                f"split left {len(train)} interactions to fit on and {len(held)} to hold "
                "out; the ranker needs both."
            )
        X_train, y_train = X_features[train], None if y_arr is None else y_arr[train]

        generators = [fit_clone(rec, X_ids[train], y_train) for _, rec in self._generators()]
        relevant = held[weights[held] > 0]
        # A splitter may hold out all of a user's rows. Such a user is a cold user, and the
        # ranker learns from them only when the generator can serve one -- which is what
        # teaches a cold-start ranker the candidates it will really see.
        queries = np.unique(users[relevant])
        if not self._serves_unknown_users:
            queries = np.intersect1d(queries, users[train])
        if not len(queries):
            raise InsufficientDataError(
                "No held-out interaction belongs to a user with training rows."
            )
        pairs, scores, groups, kept = self._retrieve(generators, queries, min_retrieved=0)
        labels = _is_held_out(pairs, users[relevant], items[relevant], queries)
        pairs, pair_context = _as_requested(
            pairs,
            groups,
            queries[kept],
            users[held],
            None if times is None else times[held],
            None if context is None else context[held],
        )
        pairs, scores, labels, groups, pair_context = _groups_with_a_positive(
            pairs, scores, labels, groups, pair_context
        )
        if not len(groups):
            raise InsufficientDataError(
                "No held-out interaction was among the generated candidates, so the ranker "
                "has nothing to learn from; increase n_retrieved."
            )
        if labels.min() == labels.max():
            raise InsufficientDataError(
                "Every generated candidate of the held-out users is relevant, so the ranker "
                "has nothing to tell apart; the catalog is too small for n_retrieved, or "
                "split holds out too much."
            )

        features = clone_as(self.features).fit(X_train, y_train)
        ranker = clone_as(self.ranker)
        fit_features(ranker, X_train, y_train)
        self.ranker_ = fit_ranker(
            ranker,
            features.transform(pairs, scores=scores, context=pair_context),
            labels,
            groups,
            Candidates(pairs, scores, context=pair_context),
        )
        self.n_ranker_groups_ = len(groups)

        self._fit_generators(X_ids, y_arr)
        self.features_ = clone_as(self.features).fit(X_features, y_arr)
        # Features a ranker joins itself serve from all of X too, as the shared ones do.
        fit_features(self.ranker_, X_features, y_arr)
        if times is not None:
            self.time_dtype_ = times.dtype
        elif hasattr(self, "time_dtype_"):
            del self.time_dtype_
        self.n_context_ = 0 if context is None else context.shape[1]
        self.n_users_ = len(self.user_ids_)
        self.n_items_ = len(self.item_ids_)
        return self

    def _fit_generators(self, X: NDArray[np.generic], y: ArrayLike | None) -> None:
        """Fit the generators for serving, and take the identifiers they know."""
        self.generators_ = [(name, fit_clone(rec, X, y)) for name, rec in self._generators()]
        fitted = [rec for _, rec in self.generators_]
        if isinstance(self.generator, list):
            self.user_ids_ = factorize(concat_ids([rec.user_ids_ for rec in fitted]))[0]
            self.item_ids_ = factorize(concat_ids([rec.item_ids_ for rec in fitted]))[0]
        else:
            self.generator_ = fitted[0]
            self.user_ids_ = self.generator_.user_ids_
            self.item_ids_ = self.generator_.item_ids_

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
            merged = retrieve_union(
                generators,
                queries,
                n_retrieved=int(self.n_retrieved),
                min_retrieved=min_retrieved,
                candidates=candidates,
                exclude_seen=exclude_seen,
                exclude_interactions=exclude_interactions,
                first_query=first_query,
                names=[name for name, _ in self._generators()],
            )
            if (tracer := active_tracer()) is not None:
                tracer.candidates("union", merged[0], merged[1], merged[2])
            return merged
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

    def _id_pairs(self, exclude_interactions: ArrayLike | None) -> ArrayLike | None:
        """``exclude_interactions`` as the generators take them: identifiers only.

        The rows may be laid out like ``X`` -- the events since ``fit``, with their time
        and context -- or as bare pairs.
        """
        if exclude_interactions is None:
            return None
        arr = check_rows(exclude_interactions, ensure_min_samples=0)
        return drop_time(arr) if arr.shape[1] > _N_COLUMNS else arr

    def _query_context(
        self, context: NDArray[np.generic] | None, n_queries: int
    ) -> NDArray[np.generic] | None:
        """The context of each query as the features take it, or None if fitted without.

        A query asked without context gets NaN in every column. A cascade fitted without
        context ignores one, as every recommender that cannot use it does.
        """
        n_context = getattr(self, "n_context_", 0)
        if not n_context:
            return None
        if context is None:
            return np.full((n_queries, n_context), np.nan)
        if context.shape[1] != n_context:
            raise ValueError(
                f"The queries carry {context.shape[1]} context column(s), but the Cascade "
                f"was fitted with {n_context}."
            )
        return context

    def _score_candidates(
        self,
        pairs: NDArray[np.generic],
        scores: NDArray[np.float64],
        groups: NDArray[np.int64],
        tracer: Tracer | None = None,
        positions: NDArray[np.intp] | None = None,
        context: NDArray[np.generic] | None = None,
    ) -> NDArray[np.float64]:
        """The ranker's score of each candidate; ``tracer`` is told the features and scores.

        The features traced are those the ranker scored from, including any it joined
        itself; the rankers inside a composite ranker are traced under ``ranker``.
        ``positions``, the candidates' fitted item order, is traced with the scores so a
        trace breaks their ties as ``recommend`` does. ``context`` is the query context
        of each candidate.
        """
        features = self.features_.transform(pairs, scores=scores, context=context)
        names = None if tracer is None else feature_names(self.features_)
        candidates = Candidates(pairs, scores, names, positions, context)
        with span("ranker"):
            ranked = predict_ranker(self.ranker_, features, groups, candidates)
        if ranked.shape != (len(pairs),):
            raise ValueError(
                f"{type(self.ranker_).__name__}.predict must return one score per row, "
                f"got shape {ranked.shape} for {len(pairs)} rows."
            )
        if tracer is not None:
            trace_ranker(tracer, self.ranker_, features, ranked, groups, candidates)
        return ranked

    def _postprocess(
        self,
        pairs: NDArray[np.generic],
        scores: NDArray[np.float64],
        groups: NDArray[np.int64],
        n_recommendations: int,
        tracer: Tracer | None = None,
    ) -> tuple[NDArray[np.generic], NDArray[np.float64]]:
        """The first ``n_recommendations`` of each list ``postprocess`` makes of the ranked ones.

        ``pairs`` and ``scores`` are best first within each group. ``tracer`` is told the
        lists before and after.
        """
        postprocess = cast(Postprocess, self.postprocess)
        out = postprocess(pairs, scores, groups)
        if not isinstance(out, tuple) or len(out) != _N_POSTPROCESS_OUTPUTS:
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
        if (
            new_pairs.ndim != _MATRIX_NDIM
            or len(new_pairs) != n_rows
            or new_scores.shape != (n_rows,)
        ):
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
        if tracer is not None:
            tracer.postprocess(
                (pairs, scores, groups), (new_pairs, new_scores, new_groups.astype(np.int64))
            )
        starts = np.concatenate([[0], np.cumsum(new_groups)[:-1]]).astype(np.intp)
        rows = starts[:, None] + np.arange(n_recommendations)
        return new_pairs[rows, 1], new_scores[rows]

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
        as_of: ArrayLike | None = None,
    ) -> tuple[NDArray[np.generic], NDArray[np.floating]]:
        """Return the items the ranker scores highest among each query's candidates.

        Parameters are those of :meth:`skrecsys.base.RecommenderMixin.recommend`; the
        filters apply to the generator, so an excluded item is never a candidate.
        Scores are the ranker's, or those ``postprocess`` returns when there is one. Ties
        are resolved by fitted item order. ``X`` may be a matrix ``[user, context...]``
        with as many context columns as the fitted ``X`` had; a vector of users asks
        without context, which the features see as NaN. ``exclude_interactions`` may be
        laid out like ``X`` of ``fit``, time and context included. With ``time``:

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
            ``as_of`` is given without ``time``, or the queries carry a number of context
            columns other than the fitted one.
        """
        check_is_fitted(self)
        check_n_recommendations(n_recommendations)
        if n_recommendations > self.n_retrieved:
            raise ValueError(
                f"n_recommendations={n_recommendations} exceeds n_retrieved={self.n_retrieved}."
            )
        if as_of is not None and not self.time:
            raise ValueError("as_of needs a Cascade constructed with time=True.")
        queries, query_context = check_queries(X)
        query_context = self._query_context(query_context, len(queries))
        query_times = check_as_of(as_of, len(queries), self.time_dtype_) if self.time else None
        exclude_interactions = self._id_pairs(exclude_interactions)
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
            pair_context = (
                None
                if query_context is None
                else np.repeat(query_context[start:stop][kept], groups, axis=0)
            )
            tracer = active_tracer()
            positions = lookup_ids(pairs[:, 1], self.item_ids_, name="item")[0]
            ranked = self._score_candidates(
                featurized, scores, groups, tracer, positions, pair_context
            )
            if self.postprocess is not None:
                order = rank_within_groups(ranked, groups, positions)
                served.append(
                    self._postprocess(
                        featurized[order], ranked[order], groups, n_recommendations, tracer
                    )
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
    @untraced()
    def _count_eligible(
        self,
        X: ArrayLike,
        *,
        candidates: ArrayLike | None = None,
        exclude_seen: bool = True,
        exclude_interactions: ArrayLike | None = None,
    ) -> NDArray[np.int64]:
        check_is_fitted(self)
        exclude_interactions = self._id_pairs(exclude_interactions)
        if not isinstance(self.generator, list):
            counts = self.generator_._count_eligible(
                X,
                candidates=candidates,
                exclude_seen=exclude_seen,
                exclude_interactions=exclude_interactions,
            )
            return np.minimum(counts, int(self.n_retrieved))
        # What the union holds is known only by merging the generators' candidates.
        queries, _ = check_queries(X)
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
        X : array-like of shape (n_samples, n_system + n_context)
            User-item pairs; with a single generator, both identifiers must be known to it.
            With ``time``, a third column holds the time each pair is scored as of,
            missing for the latest data. Further columns are the query context of each
            pair, laid out like the context of the fitted ``X``; without them the
            features see NaN.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        X_arr = check_rows(X)
        if self.time:
            if X_arr.shape[1] < _N_TIMED_COLUMNS:
                raise ValueError(
                    "X must have at least 3 columns (user identifiers, item identifiers, "
                    f"times), got {X_arr.shape[1]}."
                )
            pairs = drop_time(X_arr)
            users = check_interactions(pairs)[0]
            times = check_as_of(X_arr[:, 2], len(pairs), self.time_dtype_)
            featurized = with_time(pairs, times)
        else:
            users, _, _ = check_interactions(X_arr)
            pairs = drop_time(X_arr)
            featurized = pairs
        context = self._query_context(interaction_context(X_arr, time=self.time), len(pairs))
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
        out[order] = self._score_candidates(
            featurized[order],
            generated[order],
            groups,
            context=None if context is None else context[order],
        )
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


def _as_requested(
    pairs: NDArray[np.generic],
    groups: NDArray[np.int64],
    queries: NDArray[np.generic],
    held_users: NDArray[np.generic],
    held_times: NDArray[np.generic] | None,
    held_context: NDArray[np.generic] | None,
) -> tuple[NDArray[np.generic], NDArray[np.generic] | None]:
    """The candidates of held-out users as the request their held-out items answer.

    A user is ranked as of their first held-out interaction -- the earliest by time,
    otherwise the first by row -- so the features are what the ranker could have known
    then, and with the context of that interaction. Returns ``pairs`` with the time of
    that interaction when there are times, and the context of each pair, if any.
    """
    if held_times is None and held_context is None:
        return pairs, None
    first = first_row_of(held_users, held_times, queries)
    if held_times is not None:
        pairs = with_time(pairs, np.repeat(held_times[first], groups))
    if held_context is None:
        return pairs, None
    return pairs, np.repeat(held_context[first], groups, axis=0)


def _groups_with_a_positive(
    pairs: NDArray[np.generic],
    scores: NDArray[np.float64],
    labels: NDArray[np.float64],
    groups: NDArray[np.int64],
    context: NDArray[np.generic] | None,
) -> tuple[
    NDArray[np.generic],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.generic] | None,
]:
    """Drop the groups without a relevant candidate: they rank nothing above anything."""
    group_of_row = np.repeat(np.arange(len(groups)), groups)
    has_positive = np.bincount(group_of_row, weights=labels, minlength=len(groups)) > 0
    rows = has_positive[group_of_row]
    kept_context = None if context is None else context[rows]
    return pairs[rows], scores[rows], labels[rows], groups[has_positive], kept_context
