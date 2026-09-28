"""Adapters turning scikit-learn estimators into rankers.

A ranker follows :class:`skrecsys.base.RankerMixin`: ``fit(X, y, *, groups)`` and
``predict(X, *, groups)``, with the rows of ``X`` contiguous per query and ``groups``
holding the number of rows of each. To write one directly::

    class MyRanker(RankerMixin, BaseEstimator):
        def fit(self, X, y, *, groups): ...
        def predict(self, X, *, groups): ...
"""

import copy
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol, Self, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, is_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from skrecsys._attribution import ranker_contributions
from skrecsys._tracing import Tracer, active_tracer, feature_names, span
from skrecsys._typing import (
    DecisionClassifier,
    Features,
    GroupEstimator,
    PointwiseEstimator,
    ProbabilisticClassifier,
    RandomStateLike,
    Ranker,
    Regressor,
    clone_as,
    override,
)
from skrecsys.base import RankerMixin, is_features, is_ranker
from skrecsys.compose._named import ComponentList, NamedComponentsEstimator
from skrecsys.utils._param_validation import check_component, check_int

if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    from typing_extensions import TypeIs


def check_groups(groups: ArrayLike | None, n_rows: int) -> NDArray[np.int64]:
    """Validate group sizes against the rows they partition; ``None`` means one group."""
    if groups is None:
        return np.array([n_rows], dtype=np.int64)
    sizes = np.asarray(groups)
    if sizes.ndim != 1 or sizes.dtype.kind not in "iu":
        raise ValueError("groups must be a one-dimensional array of integer group sizes.")
    if np.any(sizes < 1) or int(sizes.sum()) != n_rows:
        raise ValueError(
            f"groups must be positive sizes summing to the {n_rows} rows of X, "
            f"got a sum of {int(sizes.sum())}."
        )
    return sizes.astype(np.int64)


@dataclass(frozen=True, slots=True)
class Candidates:
    """The candidate pairs behind the rows of a ranker's ``X``, one pair per row.

    What a :class:`~skrecsys.compose.Cascade` hands a ranker that joins features of its
    own (see :meth:`skrecsys.base.RankerMixin._takes_candidates`): ``pairs`` laid out as
    the Cascade's features see them, the generator ``scores``, and -- while a trace is
    recording -- the ``names`` of the columns of ``X``, when they are known, and the
    ``positions`` of the items in the fitted item order, which break score ties. With
    query context, ``context`` holds the context of the query behind each row.
    """

    pairs: NDArray[np.generic]
    scores: NDArray[np.float64] | None = None
    names: tuple[str, ...] | None = None
    positions: NDArray[np.intp] | None = None
    context: NDArray[np.generic] | None = None

    def take(self, rows: NDArray[np.bool_] | NDArray[np.intp]) -> "Candidates":
        """The candidates of ``rows``, as ``X[rows]`` takes their features."""
        scores = None if self.scores is None else self.scores[rows]
        positions = None if self.positions is None else self.positions[rows]
        context = None if self.context is None else self.context[rows]
        return Candidates(self.pairs[rows], scores, self.names, positions, context)

    def renamed(self, names: tuple[str, ...] | None) -> "Candidates":
        """The same candidates behind a matrix whose columns are ``names``."""
        return replace(self, names=names)


class _CandidateRanker(Protocol):
    """A ranker whose ``fit`` and ``predict`` take the candidates behind ``X``."""

    def fit(
        self, X: ArrayLike, y: ArrayLike, *, groups: ArrayLike, candidates: Candidates | None
    ) -> Ranker: ...

    def predict(
        self, X: ArrayLike, *, groups: ArrayLike, candidates: Candidates | None
    ) -> ArrayLike: ...


def takes_candidates(ranker: object) -> bool:
    """Whether ``ranker`` takes ``candidates`` -- false for one outside :class:`RankerMixin`."""
    hook = getattr(ranker, "_takes_candidates", None)
    return bool(hook()) if callable(hook) else False


def fit_features(ranker: object, X: ArrayLike, y: ArrayLike | None) -> None:
    """Fit the features ``ranker`` joins itself, if it joins any."""
    hook = getattr(ranker, "_fit_features", None)
    if callable(hook):
        hook(X, y)


def fit_ranker(
    ranker: Ranker,
    X: ArrayLike,
    y: ArrayLike,
    groups: ArrayLike,
    candidates: Candidates | None,
) -> Ranker:
    """``ranker.fit``, passing ``candidates`` only to a ranker that takes them."""
    if takes_candidates(ranker):
        return cast(_CandidateRanker, ranker).fit(X, y, groups=groups, candidates=candidates)
    return ranker.fit(X, y, groups=groups)


def predict_ranker(
    ranker: Ranker, X: ArrayLike, groups: ArrayLike, candidates: Candidates | None
) -> NDArray[np.float64]:
    """``ranker.predict`` as floats, passing ``candidates`` only to a ranker that takes them."""
    if takes_candidates(ranker):
        scores = cast(_CandidateRanker, ranker).predict(X, groups=groups, candidates=candidates)
    else:
        scores = ranker.predict(X, groups=groups)
    return np.asarray(scores, dtype=np.float64)


def ranker_input(
    ranker: Ranker, X: ArrayLike, candidates: Candidates
) -> tuple[NDArray[np.float64], tuple[str, ...] | None]:
    """The matrix ``ranker`` scores ``candidates`` from, and its column names when known.

    ``X`` itself, unless the ranker joins features of its own; see
    :meth:`skrecsys.base.RankerMixin._ranker_input`.
    """
    X = np.asarray(X, dtype=np.float64)
    hook = getattr(ranker, "_ranker_input", None)
    return hook(X, candidates) if callable(hook) else (X, candidates.names)


def trace_ranker(
    tracer: Tracer,
    ranker: Ranker,
    X: ArrayLike,
    scores: NDArray[np.float64],
    groups: NDArray[np.int64],
    candidates: Candidates,
) -> None:
    """Report to ``tracer`` the features ``ranker`` scored the candidates from, and why.

    The features and the contributions are what the ranker itself saw -- the shared
    columns and those it joined -- and are recorded by a "full" trace only.
    """
    contributions = None
    if tracer.full:
        values, names = ranker_input(ranker, X, candidates)
        tracer.features(candidates.pairs, values, groups, names)
        contributions = ranker_contributions(ranker, values)
    tracer.ranker_scores(candidates.pairs, scores, groups, contributions, candidates.positions)


def member_scores(
    rankers: Sequence[tuple[str, Ranker]],
    X: NDArray[np.float64],
    groups: NDArray[np.int64],
    candidates: Candidates | None,
) -> NDArray[np.float64]:
    """Every ranker's scores of ``X``, one column each, as a composite of rankers needs.

    A trace sees each ranker at its own path, under its name: its features, its scores
    and, for a "full" trace, their contributions.
    """
    tracer = active_tracer()
    columns = []
    for name, ranker in rankers:
        with span(name):
            column = predict_ranker(ranker, X, groups, candidates).ravel()
            if tracer is not None and candidates is not None:
                trace_ranker(tracer, ranker, X, column, groups, candidates)
        columns.append(column)
    return np.column_stack(columns)


def prepare(ranker: Ranker, X: ArrayLike, y: ArrayLike | None) -> Ranker | None:
    """A clone of ``ranker`` whose own features are fitted, or None when it joins none.

    A composite fitting the same ranker several times -- :class:`BlendRanker` once per
    fold -- fits its features once, here, and each fit starts from :func:`fresh`.
    """
    if not takes_candidates(ranker):
        return None
    clone = clone_as(ranker)
    fit_features(clone, X, y)
    return clone


def fresh(ranker: Ranker, prepared: Ranker | None) -> Ranker:
    """An unfitted ranker to fit: a shallow copy of ``prepared``, sharing its features."""
    return clone_as(ranker) if prepared is None else copy.copy(prepared)


def _take(candidates: Candidates | None, rows: NDArray[np.bool_]) -> Candidates | None:
    return None if candidates is None else candidates.take(rows)


class PointwiseRanker(RankerMixin, BaseEstimator):
    """Rank by scoring every pair on its own with a classifier or a regressor.

    ``groups`` is validated and otherwise ignored: a pointwise model does not compare
    the candidates of a query with each other.

    Parameters
    ----------
    estimator : classifier or regressor
        Cloned and fitted as ``estimator_``. A classifier ranks by the probability of its
        greatest class -- the positive one, for 0/1 relevance -- or by its
        ``decision_function`` when it has no ``predict_proba``; a regressor by
        ``predict``. Estimators that handle NaN, such as
        ``HistGradientBoostingClassifier``, take missing joined features as they are.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skrecsys.compose import PointwiseRanker
    >>> ranker = PointwiseRanker(LogisticRegression()).fit(
    ...     [[0.0], [1.0], [0.2], [0.9]], [0, 1, 0, 1], groups=[2, 2]
    ... )
    >>> bool(ranker.predict([[0.1], [0.8]], groups=[2]).argmax() == 1)
    True
    """

    def __init__(self, estimator: PointwiseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X: ArrayLike, y: ArrayLike, *, groups: ArrayLike | None = None) -> Self:
        """Fit the estimator on every pair."""
        X = check_array(X, ensure_all_finite="allow-nan")
        check_groups(groups, len(X))
        self.estimator_ = clone_as(self.estimator).fit(X, y)
        return self

    def predict(self, X: ArrayLike, *, groups: ArrayLike | None = None) -> NDArray[np.float64]:
        """Score every pair."""
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite="allow-nan")
        check_groups(groups, len(X))
        estimator = self.estimator_
        classifier = is_classifier(estimator)
        if classifier and isinstance(estimator, ProbabilisticClassifier):
            return np.asarray(estimator.predict_proba(X)[:, -1], dtype=np.float64)
        if classifier and isinstance(estimator, DecisionClassifier):
            scores = np.asarray(estimator.decision_function(X), dtype=np.float64)
            return scores if scores.ndim == 1 else scores[:, -1]
        if not classifier and isinstance(estimator, Regressor):
            return np.asarray(estimator.predict(X), dtype=np.float64).ravel()
        kind = "predict_proba or decision_function" if classifier else "predict"
        raise TypeError(f"{type(estimator).__name__} cannot score pairs: it has no {kind}.")

    @override
    def _contributions(self, X: NDArray[np.float64]) -> NDArray[np.float64] | None:
        # A linear model scores by a sum of its coefficients times the features: in
        # log-odds for a logistic model, which ranks as its probability does.
        coef = getattr(self.estimator_, "coef_", None)
        intercept = getattr(self.estimator_, "intercept_", None)
        if coef is None or intercept is None or (np.ndim(coef) > 1 and np.shape(coef)[0] != 1):
            return None
        X = check_array(X, ensure_all_finite="allow-nan")
        terms = X * np.ravel(coef)
        bias = np.full((len(X), 1), float(np.ravel(intercept)[-1]))
        return np.hstack([terms, bias])


class GroupRanker(RankerMixin, BaseEstimator):
    """Rank with a learning-to-rank estimator that takes the group sizes in ``fit``.

    Parameters
    ----------
    estimator : estimator
        Cloned and fitted as ``estimator_``, with the group sizes passed as the keyword
        ``group_param``: ``LGBMRanker`` and ``XGBRanker`` both take ``group``.
    group_param : str, default="group"
        The keyword of ``estimator.fit`` receiving the group sizes.
    """

    def __init__(self, estimator: GroupEstimator, group_param: str = "group") -> None:
        self.estimator = estimator
        self.group_param = group_param

    def fit(self, X: ArrayLike, y: ArrayLike, *, groups: ArrayLike | None = None) -> Self:
        """Fit the estimator, passing the group sizes."""
        X = check_array(X, ensure_all_finite="allow-nan")
        sizes = check_groups(groups, len(X))
        self.estimator_ = clone_as(self.estimator).fit(X, y, **{self.group_param: sizes})
        return self

    def predict(self, X: ArrayLike, *, groups: ArrayLike | None = None) -> NDArray[np.float64]:
        """Score every pair."""
        check_is_fitted(self)
        X = check_array(X, ensure_all_finite="allow-nan")
        check_groups(groups, len(X))
        return np.asarray(self.estimator_.predict(X), dtype=np.float64).ravel()


#: How an error names what a :class:`BlendRanker` holds.
_RANKER_KIND = "a ranker"


class AugmentedRanker(RankerMixin, BaseEstimator):
    """Fit a ranker on the shared features and on ``features`` only it sees.

    In a :class:`~skrecsys.compose.Cascade`, every ranker of a :class:`BlendRanker` sees
    the Cascade's ``features``. Wrapping one of them injects features of its own -- a
    :class:`~skrecsys.compose.JoinStaticFeatures` table, a
    :class:`~skrecsys.compose.JoinDynamicFeatures` lookup, any feature component --
    joined onto the candidate pairs and appended after the shared columns, so that
    rankers of one blend can learn from different views of the candidates.

    The Cascade fits the injected features the way it fits its own: on the interactions
    the ranker is not labelled from, then on all of them for serving. They are joined
    onto the candidate pairs, which only a Cascade supplies, so an ``AugmentedRanker``
    ranks inside one -- as its ranker, or nested in its :class:`BlendRanker`.

    Parameters
    ----------
    ranker : ranker
        Cloned and fitted as ``ranker_`` on the shared features followed by the injected
        ones.
    features : feature component
        Cloned and fitted as ``features_``; nested parameters address it as
        ``features__param``.

    Attributes
    ----------
    ranker_ : ranker
        The fitted ranker.
    features_ : feature component
        The fitted injected features.
    n_features_in_ : int
        The shared features, the width of ``X``.
    n_injected_features_ : int
        The columns ``features_`` appends.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skrecsys.compose import (
    ...     AugmentedRanker, BlendRanker, Cascade, GeneratorScores, JoinStaticFeatures,
    ...     PointwiseRanker,
    ... )
    >>> from skrecsys.recommendation import MostPopularRecommender
    >>> X = [[u, i] for u in range(30) for i in (u % 6, u % 6 + 1, (u + 2) % 6 + 2)]
    >>> item_table = np.array([[i, i % 2] for i in range(8)], dtype=float)
    >>> rec = Cascade(
    ...     MostPopularRecommender(),
    ...     GeneratorScores(),
    ...     BlendRanker(
    ...         [
    ...             ("scores_only", PointwiseRanker(LogisticRegression())),
    ...             ("with_items", AugmentedRanker(
    ...                 PointwiseRanker(LogisticRegression()),
    ...                 JoinStaticFeatures("item", item_table),
    ...             )),
    ...         ],
    ...         blender=None,
    ...     ),
    ...     n_retrieved=4,
    ...     split=0.4,
    ... ).fit(X)
    >>> dict(rec.ranker_.rankers_)["with_items"].n_injected_features_
    1
    """

    def __init__(self, ranker: Ranker, features: Features) -> None:
        self.ranker = ranker
        self.features = features

    def _check_params(self) -> None:
        check_component(self.ranker, "ranker", is_ranker, _RANKER_KIND)
        check_component(self.features, "features", is_features, "a feature component")

    @override
    def _takes_candidates(self) -> bool:
        return True

    @override
    def _fit_features(self, X: ArrayLike, y: ArrayLike | None) -> None:
        self._check_params()
        self.features_ = clone_as(self.features).fit(X, y)
        if hasattr(self, "ranker_"):
            fit_features(self.ranker_, X, y)
        else:
            self._prepared = prepare(self.ranker, X, y)

    def _join(self, X: NDArray[np.float64], candidates: Candidates | None) -> NDArray[np.float64]:
        """``X`` followed by the injected features of the candidates behind its rows."""
        if candidates is None or not hasattr(self, "features_"):
            raise ValueError(
                "AugmentedRanker joins its features onto the candidate pairs, which only a "
                "Cascade supplies; use it as the ranker of a Cascade, or inside its BlendRanker."
            )
        if len(candidates.pairs) != len(X):
            raise ValueError(
                f"candidates hold {len(candidates.pairs)} pairs for the {len(X)} rows of X."
            )
        extra = self.features_.transform(
            candidates.pairs, scores=candidates.scores, context=candidates.context
        )
        return np.hstack([X, np.asarray(extra, dtype=np.float64).reshape(len(X), -1)])

    def _joined_names(self, n_shared: int, shared: tuple[str, ...] | None) -> tuple[str, ...]:
        """Names of the shared columns then the injected ones, ``x<j>`` where unknown.

        An injected name that a shared column already has is prefixed with
        ``injected__``, so that every column of the trace keeps its own name.
        """
        if shared is None or len(shared) != n_shared:
            shared = tuple(f"x{j}" for j in range(n_shared))
        injected = feature_names(self.features_)
        if injected is None or len(injected) != self.n_injected_features_:
            injected = tuple(f"x{n_shared + j}" for j in range(self.n_injected_features_))
        taken = set(shared)
        return (*shared, *(f"injected__{n}" if n in taken else n for n in injected))

    @override
    def _ranker_input(
        self, X: NDArray[np.float64], candidates: Candidates
    ) -> tuple[NDArray[np.float64], tuple[str, ...] | None]:
        check_is_fitted(self, "ranker_")
        names = self._joined_names(X.shape[1], candidates.names)
        return ranker_input(self.ranker_, self._join(X, candidates), candidates.renamed(names))

    @override
    def _contributions(self, X: NDArray[np.float64]) -> NDArray[np.float64] | None:
        # X is what _ranker_input returned: the injected columns are there already.
        return ranker_contributions(self.ranker_, X)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        groups: ArrayLike | None = None,
        candidates: Candidates | None = None,
    ) -> Self:
        """Fit the ranker on ``X`` and the injected features of ``candidates``."""
        self._check_params()
        X = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")
        sizes = check_groups(groups, len(X))
        prepared = self.__dict__.pop("_prepared", None)
        joined = self._join(X, candidates)
        self.ranker_ = fit_ranker(fresh(self.ranker, prepared), joined, y, sizes, candidates)
        self.n_features_in_ = X.shape[1]
        self.n_injected_features_ = joined.shape[1] - X.shape[1]
        return self

    def predict(
        self,
        X: ArrayLike,
        *,
        groups: ArrayLike | None = None,
        candidates: Candidates | None = None,
    ) -> NDArray[np.float64]:
        """Score every row from ``X`` and the injected features of ``candidates``."""
        check_is_fitted(self, "ranker_")
        X = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")
        sizes = check_groups(groups, len(X))
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the ranker was fitted with "
                f"{self.n_features_in_}."
            )
        joined = self._join(X, candidates)
        if candidates is not None and active_tracer() is not None:
            # A composite inside, such as a BlendRanker, traces the columns it sees.
            candidates = candidates.renamed(self._joined_names(X.shape[1], candidates.names))
        return predict_ranker(self.ranker_, joined, sizes, candidates).ravel()


#: How :class:`BlendRanker` puts its rankers' scores on one scale, per group.
_NORMALIZE = ("rank", "zscore", "none")

#: The ``rankers`` of :class:`BlendRanker`: all bare rankers or all named ones.
RankerList: TypeAlias = ComponentList[Ranker]


def normalize_per_group(
    scores: NDArray[np.float64], groups: NDArray[np.int64], how: str
) -> NDArray[np.float64]:
    """Put every column of ``scores`` on a common scale within each group.

    ``"rank"`` maps a group's scores to their percentile rank in [0, 1] -- ties share the
    average -- and a single-row group to 0.5; ``"zscore"`` centres and scales them, a
    constant group becoming zeros; ``"none"`` returns them as they are. Only the order
    within a group means anything for a ranker, so this is what lets scores from
    different models be added.
    """
    if how == "none":
        return scores
    group_of_row = np.repeat(np.arange(len(groups)), groups)
    out = np.empty_like(scores)
    for column in range(scores.shape[1]):
        values = scores[:, column]
        if how == "zscore":
            mean = np.bincount(group_of_row, weights=values) / groups
            centred = values - mean[group_of_row]
            std = np.sqrt(np.bincount(group_of_row, weights=centred**2) / groups)
            out[:, column] = np.divide(
                centred, std[group_of_row], out=np.zeros_like(centred), where=std[group_of_row] > 0
            )
        else:
            out[:, column] = _percentile_ranks(values, group_of_row, groups)
    return out


def _percentile_ranks(
    values: NDArray[np.float64], group_of_row: NDArray[np.intp], groups: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Each value's average rank within its group, scaled to [0, 1]."""
    position = mean_positions(values, group_of_row, groups)
    sizes = groups[group_of_row]
    return np.where(sizes > 1, position / np.maximum(sizes - 1, 1), 0.5)


def mean_positions(
    values: NDArray[np.float64], group_of_row: NDArray[np.intp], groups: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Each value's 0-based position within its group in ascending order, ties averaged."""
    order = np.lexsort((values, group_of_row))
    sorted_values, sorted_groups = values[order], group_of_row[order]
    starts = np.concatenate([[0], np.cumsum(groups)[:-1]])
    position = np.arange(len(values)) - starts[sorted_groups]
    # A run of equal values in one group shares the mean of its positions.
    new_run = np.ones(len(values), dtype=bool)
    new_run[1:] = (sorted_values[1:] != sorted_values[:-1]) | (
        sorted_groups[1:] != sorted_groups[:-1]
    )
    run = np.cumsum(new_run) - 1
    mean_position = np.bincount(run, weights=position) / np.bincount(run)
    out = np.empty(len(values), dtype=np.float64)
    out[order] = mean_position[run]
    return out


class BlendRanker(RankerMixin, NamedComponentsEstimator[Ranker]):
    """Blend several rankers into one, by stacking or by a weighted average.

    Rankers disagree in useful ways -- CatBoost, XGBoost and LightGBM grow different trees
    from the same features -- and a blend of them is usually steadier than the best one
    alone. Their raw scores are on unrelated scales, so each ranker's scores are first
    normalized within every group (see ``normalize``).

    With a ``blender`` (the default), the blend is learned. ``fit`` splits the *groups*
    into ``cv`` folds, fits every ranker on all folds but one and scores the one left out,
    so each row gets a score from rankers that never saw its group. The blender is fitted
    on those out-of-fold scores, and only then is every ranker refitted on all the rows,
    for ``predict``. A blender fitted on in-sample scores would learn to trust whichever
    ranker overfits most.

    With ``blender=None`` nothing is learned about the rankers: ``predict`` returns the
    ``weights``-weighted mean of their normalized scores.

    Parameters
    ----------
    rankers : list of rankers or of (name, ranker) tuples
        Cloned and fitted as ``rankers_``. Unnamed rankers are named after their class in
        lower case, numbered when a class repeats; names address nested parameters, as in
        ``catboostranker__iterations``.
    blender : ranker, "logistic" or None, default="logistic"
        Learns how to combine the normalized scores: ``"logistic"`` is a
        :class:`PointwiseRanker` around ``LogisticRegression``, and any ranker may be given
        instead. ``None`` averages instead of learning.
    normalize : {"rank", "zscore", "none"}, default="rank"
        ``"rank"`` turns each ranker's scores into percentile ranks within their group,
        which ignores how confident a ranker is; ``"zscore"`` standardizes them within
        their group, which keeps it; ``"none"`` passes them through.
    cv : int, default=3
        Folds of groups for the out-of-fold scores. Unused without a blender.
    weights : array-like of shape (n_rankers,), default=None
        Weights of the average when ``blender=None``; equal by default.
    passthrough : bool, default=False
        Whether the blender also sees the original features, beside the rankers' scores.
    random_state : int, RandomState instance or None, default=None
        Shuffles the groups into folds.

    Attributes
    ----------
    rankers_ : list of (name, ranker) tuples
        The rankers, fitted on every row.
    blender_ : ranker or None
        The fitted blender, or None when averaging.
    n_features_in_ : int

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from skrecsys.compose import BlendRanker, PointwiseRanker
    >>> X = [[0.0, 1.0], [1.0, 0.0], [0.2, 0.9], [0.9, 0.1], [0.1, 0.8], [0.8, 0.3]]
    >>> blend = BlendRanker(
    ...     [PointwiseRanker(LogisticRegression()), PointwiseRanker(DecisionTreeClassifier())],
    ...     blender=None,
    ... ).fit(X, [0, 1, 0, 1, 0, 1], groups=[2, 2, 2])
    >>> blend.predict([[0.1, 0.9], [0.9, 0.2]], groups=[2]).tolist()
    [0.0, 1.0]
    """

    def __init__(
        self,
        rankers: RankerList,
        blender: Ranker | str | None = "logistic",
        *,
        normalize: str = "rank",
        cv: int = 3,
        weights: ArrayLike | None = None,
        passthrough: bool = False,
        random_state: RandomStateLike = None,
    ) -> None:
        self.rankers: RankerList = rankers
        self.blender = blender
        self.normalize = normalize
        self.cv = cv
        self.weights = weights
        self.passthrough = passthrough
        self.random_state = random_state

    _components_param = "rankers"
    _component_kind = _RANKER_KIND

    @override
    def _is_component(self, value: object) -> TypeIs[Ranker]:
        return is_ranker(value)

    @override
    def _components(self) -> RankerList:
        return self.rankers

    def _check_params(self) -> tuple[list[tuple[str, Ranker]], Ranker | None]:
        """The named rankers and the blender to fit, once every parameter is checked."""
        if not self.rankers:
            raise ValueError("BlendRanker needs at least one ranker.")
        named = self._named()
        for _, ranker in named:
            if not is_ranker(ranker):
                raise TypeError(f"{type(ranker).__name__} is not a ranker.")
        if self.normalize not in _NORMALIZE:
            raise ValueError(f"normalize must be one of {_NORMALIZE}, got {self.normalize!r}.")
        blender: Ranker | None
        if self.blender is None:
            blender = None
            if self.weights is not None and len(np.ravel(self.weights)) != len(named):
                raise ValueError(
                    f"weights must have one entry per ranker ({len(named)}), "
                    f"got {len(np.ravel(self.weights))}."
                )
        elif isinstance(self.blender, str):
            if self.blender != "logistic":
                raise ValueError(
                    f"blender must be a ranker, 'logistic' or None, got {self.blender!r}."
                )
            blender = PointwiseRanker(LogisticRegression())
        else:
            check_component(self.blender, "blender", is_ranker, _RANKER_KIND)
            blender = self.blender
        check_int(self.cv, "cv", min_value=2)
        return named, blender

    @override
    def _takes_candidates(self) -> bool:
        return True

    @override
    def _fit_features(self, X: ArrayLike, y: ArrayLike | None) -> None:
        if hasattr(self, "rankers_"):
            for _, ranker in self.rankers_:
                fit_features(ranker, X, y)
            fit_features(self.blender_, X, y)
            return
        named, blender = self._check_params()
        members = {name: prepare(ranker, X, y) for name, ranker in named}
        self._prepared = (members, None if blender is None else prepare(blender, X, y))

    def _scores(
        self,
        rankers: Sequence[tuple[str, Ranker]],
        X: NDArray[np.float64],
        groups: NDArray[np.int64],
        candidates: Candidates | None,
    ) -> NDArray[np.float64]:
        """Every ranker's normalized scores, one column each."""
        raw = member_scores(rankers, X, groups, candidates)
        return normalize_per_group(raw, groups, self.normalize)

    def _blender_input(
        self, scores: NDArray[np.float64], X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.hstack([scores, X]) if self.passthrough else scores

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        groups: ArrayLike | None = None,
        candidates: Candidates | None = None,
    ) -> Self:
        """Fit the blender on out-of-fold scores, then every ranker on all rows.

        ``candidates``, the pairs behind the rows of ``X``, reach the rankers that join
        features of their own, such as an :class:`AugmentedRanker`; a
        :class:`~skrecsys.compose.Cascade` passes them.
        """
        named, blender = self._check_params()
        prepared, prepared_blender = self.__dict__.pop("_prepared", ({}, None))
        X = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")
        labels = check_array(y, ensure_2d=False, dtype=np.float64)
        sizes = check_groups(groups, len(X))
        if blender is not None:
            out_of_fold = self._out_of_fold(named, prepared, X, labels, sizes, candidates)
            self.blender_ = fit_ranker(
                fresh(blender, prepared_blender),
                self._blender_input(out_of_fold, X),
                labels,
                sizes,
                candidates,
            )
        else:
            self.blender_ = None
        self.rankers_ = [
            (name, fit_ranker(fresh(ranker, prepared.get(name)), X, labels, sizes, candidates))
            for name, ranker in named
        ]
        self.n_features_in_ = X.shape[1]
        return self

    def _out_of_fold(
        self,
        named: Sequence[tuple[str, Ranker]],
        prepared: dict[str, Ranker | None],
        X: NDArray[np.float64],
        labels: NDArray[np.float64],
        sizes: NDArray[np.int64],
        candidates: Candidates | None,
    ) -> NDArray[np.float64]:
        """Each row's normalized scores from rankers fitted without its group."""
        n_folds = int(self.cv)
        if n_folds > len(sizes):
            raise ValueError(f"cv={n_folds} folds need at least as many groups, got {len(sizes)}.")
        rng = check_random_state(self.random_state)
        fold_of_group = np.empty(len(sizes), dtype=np.intp)
        for fold, members in enumerate(np.array_split(rng.permutation(len(sizes)), n_folds)):
            fold_of_group[members] = fold
        fold_of_row = np.repeat(fold_of_group, sizes)
        scores = np.empty((len(X), len(named)), dtype=np.float64)
        for fold in range(n_folds):
            held, kept = fold_of_row == fold, fold_of_row != fold
            held_sizes = sizes[fold_of_group == fold]
            kept_sizes = sizes[fold_of_group != fold]
            fitted = [
                (
                    name,
                    fit_ranker(
                        fresh(ranker, prepared.get(name)),
                        X[kept],
                        labels[kept],
                        kept_sizes,
                        _take(candidates, kept),
                    ),
                )
                for name, ranker in named
            ]
            scores[held] = self._scores(fitted, X[held], held_sizes, _take(candidates, held))
        return scores

    def predict(
        self,
        X: ArrayLike,
        *,
        groups: ArrayLike | None = None,
        candidates: Candidates | None = None,
    ) -> NDArray[np.float64]:
        """Score every row: the blender's view of the rankers', or their weighted mean."""
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")
        sizes = check_groups(groups, len(X))
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the ranker was fitted with "
                f"{self.n_features_in_}."
            )
        scores = self._scores(self.rankers_, X, sizes, candidates)
        if self.blender_ is None:
            weights = None if self.weights is None else np.ravel(self.weights)
            return np.average(scores, axis=1, weights=weights)
        if candidates is not None and candidates.names is not None:
            # The blender's columns are the rankers' scores, then the features passed through.
            ranked = tuple(name for name, _ in self.rankers_)
            passed = candidates.names if self.passthrough else ()
            candidates = candidates.renamed((*ranked, *passed))
        blended = predict_ranker(self.blender_, self._blender_input(scores, X), sizes, candidates)
        return blended.ravel()
