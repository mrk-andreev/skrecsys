"""What the rankers of :mod:`skrecsys.integrations` share. Imports no third-party library."""

from collections.abc import Iterable, Mapping
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from skrecsys._typing import RandomStateLike, override
from skrecsys.base import RankerMixin
from skrecsys.compose._rankers import check_groups


class BoostedRanker(RankerMixin, BaseEstimator):
    """A ranker whose model comes from a gradient-boosting library.

    Validates ``X``, ``y`` and ``groups`` and checks the feature count, leaving a subclass
    to fit ``model_`` in :meth:`_fit_model` and to score with :meth:`_predict_model`.
    ``X`` may hold NaN: every library behind a subclass handles missing values natively.
    """

    random_state: RandomStateLike
    extra_params: Mapping[str, object] | None

    def _seed(self) -> int:
        """``random_state`` as the int seed the libraries take."""
        seed = self.random_state
        if not isinstance(seed, int | np.integer):
            seed = check_random_state(seed).randint(np.iinfo(np.int32).max)
        return int(seed)

    def _extra_params(self, aliases: Iterable[str]) -> dict[str, object]:
        """``extra_params``, checked not to name what the ranker already sets.

        The ranker's own parameters keep the library's names; ``aliases`` are the
        library's other names for them, and for what the ranker passes on its own, such
        as XGBoost's ``eta`` for ``learning_rate``. An extra parameter under any of those
        names would silently win or lose against the one of that meaning.
        """
        extra = {} if self.extra_params is None else self.extra_params
        if not isinstance(extra, Mapping) or not all(isinstance(key, str) for key in extra):
            raise TypeError(f"extra_params must be a mapping of parameter names, got {extra!r}.")
        set_here = set(self.get_params(deep=False)) - {"extra_params"} | set(aliases)
        clash = sorted(set(extra) & set_here)
        if clash:
            raise ValueError(
                f"extra_params sets {clash}, which {type(self).__name__} sets itself; "
                "use its parameter of that meaning instead."
            )
        return dict(extra)

    def _fit_model(
        self, X: NDArray[np.float64], labels: NDArray[np.float64], sizes: NDArray[np.int64]
    ) -> None:
        raise NotImplementedError

    def _predict_model(self, X: NDArray[np.float64]) -> ArrayLike:
        raise NotImplementedError

    def _contributions_model(self, X: NDArray[np.float64]) -> ArrayLike:
        """SHAP values of ``model_``: one column per feature, then the expected value."""
        raise NotImplementedError

    @override
    def _contributions(self, X: NDArray[np.float64]) -> NDArray[np.float64] | None:
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")
        return np.asarray(self._contributions_model(X), dtype=np.float64)

    def fit(self, X: ArrayLike, y: ArrayLike, *, groups: ArrayLike | None = None) -> Self:
        """Fit on the candidate features ``X`` and their relevance ``y``, per group."""
        X = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")
        labels = check_array(y, ensure_2d=False, dtype=np.float64)
        sizes = check_groups(groups, len(X))
        self._fit_model(X, labels, sizes)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: ArrayLike, *, groups: ArrayLike | None = None) -> NDArray[np.float64]:
        """Score every candidate; only the order within a group is meaningful."""
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")
        check_groups(groups, len(X))
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the ranker was fitted with "
                f"{self.n_features_in_}."
            )
        return np.asarray(self._predict_model(X), dtype=np.float64).ravel()
