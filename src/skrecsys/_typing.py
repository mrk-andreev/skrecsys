"""Typing constructs shared across the package.

The protocols spell out what a composite expects of the estimators it is given. They are
structural: an estimator satisfies one by having the methods, not by inheriting from it,
which is also how the ``is_recommender``-style checks in :mod:`skrecsys.base` decide at
runtime.
"""

import sys
from collections.abc import Callable, Hashable, Iterable, Iterator
from typing import Protocol, Self, TypeAlias, TypeVar, cast, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from sklearn.base import clone
from sklearn.utils import Tags

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover - exercised on 3.11 only
    from typing_extensions import override

if sys.version_info >= (3, 13):
    from typing import TypeIs
else:  # pragma: no cover - exercised on 3.11 and 3.12 only
    from typing_extensions import TypeIs

__all__ = [
    "Condition",
    "CrossValidator",
    "DataFrameLike",
    "DecisionClassifier",
    "Estimator",
    "FeatureNamer",
    "Features",
    "FittedRecommender",
    "GroupEstimator",
    "PairScorer",
    "PointwiseEstimator",
    "ProbabilisticClassifier",
    "RandomStateLike",
    "Ranker",
    "RankingMetric",
    "Recommender",
    "Regressor",
    "SortableId",
    "TypeIs",
    "clone_as",
    "override",
]

#: What ``random_state`` accepts across scikit-learn.
RandomStateLike: TypeAlias = int | np.random.RandomState | np.random.Generator | None


class Estimator(Protocol):
    """What :func:`sklearn.base.clone` and :func:`sklearn.utils.get_tags` need."""

    # Keyword-only here: scikit-learn always passes `deep` by name, and an estimator whose
    # `deep` is also positional still satisfies the protocol.
    def get_params(self, *, deep: bool = True) -> dict[str, object]: ...

    def set_params(self, **params: object) -> Self: ...

    def __sklearn_tags__(self) -> Tags: ...


class Recommender(Estimator, Protocol):
    """An estimator following :class:`skrecsys.base.RecommenderMixin`.

    ``_count_eligible`` comes with the mixin; a multi-stage model needs it because
    ``recommend`` raises instead of padding.
    """

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self: ...

    def recommend(
        self,
        X: ArrayLike,
        *,
        n_recommendations: int = ...,
        candidates: ArrayLike | None = ...,
        exclude_seen: bool = ...,
        exclude_interactions: ArrayLike | None = ...,
    ) -> tuple[NDArray[np.generic], NDArray[np.floating]]: ...

    def _count_eligible(
        self,
        X: ArrayLike,
        *,
        candidates: ArrayLike | None = ...,
        exclude_seen: bool = ...,
        exclude_interactions: ArrayLike | None = ...,
    ) -> NDArray[np.int64]: ...


class FittedRecommender(Recommender, Protocol):
    """A recommender after ``fit``, exposing the identifiers it was fitted on."""

    user_ids_: NDArray[np.generic]
    item_ids_: NDArray[np.generic]


class TimedRecommender(FittedRecommender, Protocol):
    """A fitted recommender constructed with ``time=True``: it answers as of a time.

    See :func:`skrecsys.base.uses_time`.
    """

    time: bool

    @override
    def recommend(
        self,
        X: ArrayLike,
        *,
        n_recommendations: int = ...,
        candidates: ArrayLike | None = ...,
        exclude_seen: bool = ...,
        exclude_interactions: ArrayLike | None = ...,
        as_of: ArrayLike | None = ...,
    ) -> tuple[NDArray[np.generic], NDArray[np.floating]]: ...


@runtime_checkable
class PairScorer(Protocol):
    """A recommender that also scores given user-item pairs, as every
    :class:`skrecsys.recommendation.BaseRecommender` does; the mixin alone does not."""

    def predict(self, X: ArrayLike) -> ArrayLike: ...


class Condition(Estimator, Protocol):
    """An estimator following :class:`skrecsys.base.ConditionMixin`."""

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self: ...

    def evaluate(self, X: ArrayLike) -> NDArray[np.bool_]: ...


class Features(Estimator, Protocol):
    """An estimator following :class:`skrecsys.base.FeaturesMixin`.

    ``fit`` takes ``X=None`` too: :class:`skrecsys.compose.ConcatFeatures` passes on what
    it was given, and a component that learns nothing may be fitted without data.
    """

    def fit(self, X: ArrayLike | None, y: ArrayLike | None = None) -> Self: ...

    def transform(
        self,
        pairs: ArrayLike,
        *,
        scores: ArrayLike | None = None,
        context: ArrayLike | None = None,
    ) -> NDArray[np.floating]: ...


@runtime_checkable
class FeatureNamer(Protocol):
    """A feature component that names the columns ``transform`` returns."""

    def get_feature_names_out(self) -> NDArray[np.generic]: ...


class Ranker(Estimator, Protocol):
    """An estimator following :class:`skrecsys.base.RankerMixin`.

    ``groups`` is always passed; an implementation may give it a default.
    """

    def fit(self, X: ArrayLike, y: ArrayLike, *, groups: ArrayLike) -> Self: ...

    def predict(self, X: ArrayLike, *, groups: ArrayLike) -> ArrayLike: ...


@runtime_checkable
class Regressor(Estimator, Protocol):
    """A supervised scikit-learn estimator scoring each row with ``predict``."""

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self: ...

    def predict(self, X: ArrayLike) -> ArrayLike: ...


@runtime_checkable
class ProbabilisticClassifier(Estimator, Protocol):
    """A classifier whose last-class probability serves as the score."""

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self: ...

    def predict_proba(self, X: ArrayLike) -> NDArray[np.floating]: ...


@runtime_checkable
class DecisionClassifier(Estimator, Protocol):
    """A classifier without probabilities, scored by its decision function."""

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self: ...

    def decision_function(self, X: ArrayLike) -> NDArray[np.floating]: ...


#: What :class:`skrecsys.compose.PointwiseRanker` can score pairs with.
PointwiseEstimator: TypeAlias = ProbabilisticClassifier | DecisionClassifier | Regressor


class GroupEstimator(Estimator, Protocol):
    """A learning-to-rank estimator taking the group sizes as a keyword of ``fit``."""

    # Called as ``fit(X, y, **{group_param: sizes})``: the keyword is a parameter of
    # GroupRanker, so the signature cannot be spelled past taking the call.
    @property
    def fit(self) -> Callable[..., "GroupEstimator"]: ...

    def predict(self, X: ArrayLike) -> ArrayLike: ...


class CrossValidator(Protocol):
    """A scikit-learn splitter; only its first ``(train, test)`` split is read."""

    def split(
        self, X: ArrayLike, y: ArrayLike | None = None
    ) -> Iterator[tuple[ArrayLike, ArrayLike]]: ...


#: A top-k metric such as :func:`skrecsys.metrics.ndcg_at_k`, called as
#: ``metric(y_true, y_pred, k=k, **kwargs)``. The metrics differ in their other keywords,
#: so the parameters beyond the call convention cannot be spelled as one protocol.
RankingMetric: TypeAlias = Callable[..., float | NDArray[np.float64]]


#: The dtype of a column of a :class:`DataFrameLike`: a numpy dtype, or a pandas
#: extension dtype.
DTypeT_co = TypeVar("DTypeT_co", covariant=True)


@runtime_checkable
class DataFrameLike(Protocol[DTypeT_co]):
    """What input validation reads of a DataFrame, without importing pandas.

    A Series has the same members; ``ndim`` tells the two apart.
    """

    @property
    def ndim(self) -> int: ...

    @property
    def dtypes(self) -> Iterable[DTypeT_co]: ...

    def to_numpy(self, dtype: DTypeLike | None = None) -> NDArray[np.generic]: ...


class SortableId(Hashable, Protocol):
    """An identifier that can key a dict and be sorted among its peers."""

    def __lt__(self, other: Self, /) -> bool: ...


EstimatorT = TypeVar("EstimatorT", bound=Estimator)


def clone_as(estimator: EstimatorT) -> EstimatorT:
    """:func:`sklearn.base.clone`, keeping the static type that ``clone`` loses."""
    return cast(EstimatorT, clone(estimator))
