"""Adapters turning scikit-learn estimators into rankers.

A ranker follows :class:`skrecsys.base.RankerMixin`: ``fit(X, y, *, groups)`` and
``predict(X, *, groups)``, with the rows of ``X`` contiguous per query and ``groups``
holding the number of rows of each. To write one directly::

    class MyRanker(RankerMixin, BaseEstimator):
        def fit(self, X, y, *, groups): ...
        def predict(self, X, *, groups): ...
"""

import sys
from collections.abc import Sequence
from typing import Self, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, is_classifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from skrecsys._typing import (
    DecisionClassifier,
    GroupEstimator,
    PointwiseEstimator,
    ProbabilisticClassifier,
    RandomStateLike,
    Ranker,
    Regressor,
    clone_as,
    override,
)
from skrecsys.base import RankerMixin, is_ranker
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


#: How :class:`BlendRanker` puts its rankers' scores on one scale, per group.
_NORMALIZE = ("rank", "zscore", "none")

#: How an error names what a :class:`BlendRanker` holds.
_RANKER_KIND = "a ranker"

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

    def _scores(
        self,
        rankers: Sequence[tuple[str, Ranker]],
        X: NDArray[np.float64],
        groups: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Every ranker's normalized scores, one column each."""
        raw = np.column_stack(
            [
                np.asarray(ranker.predict(X, groups=groups), dtype=np.float64)
                for _, ranker in rankers
            ]
        )
        return normalize_per_group(raw, groups, self.normalize)

    def _blender_input(
        self, scores: NDArray[np.float64], X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.hstack([scores, X]) if self.passthrough else scores

    def fit(self, X: ArrayLike, y: ArrayLike, *, groups: ArrayLike | None = None) -> Self:
        """Fit the blender on out-of-fold scores, then every ranker on all rows."""
        named, blender = self._check_params()
        X = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")
        labels = check_array(y, ensure_2d=False, dtype=np.float64)
        sizes = check_groups(groups, len(X))
        if blender is not None:
            out_of_fold = self._out_of_fold(named, X, labels, sizes)
            self.blender_ = clone_as(blender).fit(
                self._blender_input(out_of_fold, X), labels, groups=sizes
            )
        else:
            self.blender_ = None
        self.rankers_ = [
            (name, clone_as(ranker).fit(X, labels, groups=sizes)) for name, ranker in named
        ]
        self.n_features_in_ = X.shape[1]
        return self

    def _out_of_fold(
        self,
        named: Sequence[tuple[str, Ranker]],
        X: NDArray[np.float64],
        labels: NDArray[np.float64],
        sizes: NDArray[np.int64],
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
                (name, clone_as(ranker).fit(X[kept], labels[kept], groups=kept_sizes))
                for name, ranker in named
            ]
            scores[held] = self._scores(fitted, X[held], held_sizes)
        return scores

    def predict(self, X: ArrayLike, *, groups: ArrayLike | None = None) -> NDArray[np.float64]:
        """Score every row: the blender's view of the rankers', or their weighted mean."""
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")
        sizes = check_groups(groups, len(X))
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the ranker was fitted with "
                f"{self.n_features_in_}."
            )
        scores = self._scores(self.rankers_, X, sizes)
        if self.blender_ is None:
            weights = None if self.weights is None else np.ravel(self.weights)
            return np.average(scores, axis=1, weights=weights)
        blended = self.blender_.predict(self._blender_input(scores, X), groups=sizes)
        return np.asarray(blended, dtype=np.float64).ravel()
