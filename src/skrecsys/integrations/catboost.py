"""CatBoost integration: a ranker for :class:`~skrecsys.compose.Cascade` backed by CatBoost.

CatBoost is a third-party library and an optional extra::

    pip install skrecsys[catboost]

Nothing outside this module imports it.
"""

from typing import NotRequired, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from skrecsys._typing import RandomStateLike, override
from skrecsys.integrations._base import BoostedRanker

try:
    import catboost
except ModuleNotFoundError as _exc:  # pragma: no cover - covered by the catboost-free CI matrix
    if _exc.name != "catboost":
        raise
    raise ImportError(
        "skrecsys.integrations.catboost requires CatBoost. "
        "Install it with `pip install skrecsys[catboost]`."
    ) from _exc

__all__ = ["CatBoostRanker"]


class _CatBoostParams(TypedDict):
    """What ``catboost.CatBoost`` is constructed with."""

    loss_function: str
    iterations: int
    depth: int
    l2_leaf_reg: float
    random_seed: int
    thread_count: int
    verbose: bool
    allow_writing_files: bool
    learning_rate: NotRequired[float]


class CatBoostRanker(BoostedRanker):
    """Gradient-boosted trees trained to rank the candidates of each query.

    The group sizes become CatBoost's ``group_id``, so the pairwise and listwise losses
    compare candidates only within a query. Missing features -- NaN, as a
    :class:`~skrecsys.compose.JoinStaticFeatures` writes for an unknown identifier -- are
    handled natively.

    Parameters
    ----------
    loss_function : str, default="YetiRank"
        Any CatBoost ranking loss, such as ``"YetiRank"``, ``"PairLogit"``,
        ``"QuerySoftMax"`` or the pointwise ``"Logloss"``.
    iterations : int, default=500
        Number of trees.
    learning_rate : float, default=None
        ``None`` lets CatBoost choose.
    depth : int, default=6
        Depth of each tree.
    l2_leaf_reg : float, default=3.0
        L2 regularization of the leaf values.
    random_state : int, RandomState instance or None, default=None
        Seeds CatBoost.
    thread_count : int, default=-1
        Threads for training and prediction; -1 uses every core.
    verbose : bool, default=False
        Whether CatBoost logs its training.

    Attributes
    ----------
    model_ : catboost.CatBoost
        The fitted model.
    n_features_in_ : int

    Examples
    --------
    >>> from skrecsys.integrations.catboost import CatBoostRanker
    >>> ranker = CatBoostRanker(iterations=10, random_state=0).fit(
    ...     [[0.0], [1.0], [0.2], [0.9]], [0, 1, 0, 1], groups=[2, 2]
    ... )
    >>> ranker.predict([[0.1], [0.8]], groups=[2]).shape
    (2,)
    """

    def __init__(
        self,
        loss_function: str = "YetiRank",
        *,
        iterations: int = 500,
        learning_rate: float | None = None,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_state: RandomStateLike = None,
        thread_count: int = -1,
        verbose: bool = False,
    ) -> None:
        self.loss_function = loss_function
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.thread_count = thread_count
        self.verbose = verbose

    def _catboost_params(self) -> _CatBoostParams:
        params: _CatBoostParams = {
            "loss_function": self.loss_function,
            "iterations": self.iterations,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "random_seed": self._seed(),
            "thread_count": self.thread_count,
            "verbose": bool(self.verbose),
            "allow_writing_files": False,
        }
        if self.learning_rate is not None:
            params["learning_rate"] = self.learning_rate
        return params

    @override
    def _fit_model(
        self, X: NDArray[np.float64], labels: NDArray[np.float64], sizes: NDArray[np.int64]
    ) -> None:
        group_id = np.repeat(np.arange(len(sizes)), sizes)
        self.model_ = catboost.CatBoost(self._catboost_params())
        self.model_.fit(catboost.Pool(X, label=labels, group_id=group_id))

    @override
    def _predict_model(self, X: NDArray[np.float64]) -> ArrayLike:
        return self.model_.predict(X)
