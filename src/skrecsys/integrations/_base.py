"""What the rankers of :mod:`skrecsys.integrations` share. Imports no third-party library."""

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from skrecsys._typing import RandomStateLike
from skrecsys.base import RankerMixin
from skrecsys.compose._rankers import check_groups


class BoostedRanker(RankerMixin, BaseEstimator):
    """A ranker whose model comes from a gradient-boosting library.

    Validates ``X``, ``y`` and ``groups`` and checks the feature count, leaving a subclass
    to fit ``model_`` in :meth:`_fit_model` and to score with :meth:`_predict_model`.
    ``X`` may hold NaN: every library behind a subclass handles missing values natively.
    """

    random_state: RandomStateLike

    def _seed(self) -> int:
        """``random_state`` as the int seed the libraries take."""
        seed = self.random_state
        if not isinstance(seed, int | np.integer):
            seed = check_random_state(seed).randint(np.iinfo(np.int32).max)
        return int(seed)

    def _fit_model(
        self, X: NDArray[np.float64], labels: NDArray[np.float64], sizes: NDArray[np.int64]
    ) -> None:
        raise NotImplementedError

    def _predict_model(self, X: NDArray[np.float64]) -> ArrayLike:
        raise NotImplementedError

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
