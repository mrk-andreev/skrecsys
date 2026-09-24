"""XGBoost integration: a ranker for :class:`~skrecsys.compose.Cascade` backed by XGBoost.

XGBoost is a third-party library and an optional extra::

    pip install skrecsys[xgboost]

Nothing outside this module imports it.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from skrecsys._typing import RandomStateLike, override
from skrecsys.integrations._base import BoostedRanker

try:
    from xgboost.sklearn import XGBRanker as _XGBRanker
except ModuleNotFoundError as _exc:  # pragma: no cover - covered by the xgboost-free CI matrix
    if _exc.name != "xgboost":
        raise
    raise ImportError(
        "skrecsys.integrations.xgboost requires XGBoost. "
        "Install it with `pip install skrecsys[xgboost]`."
    ) from _exc

__all__ = ["XGBRanker"]


class XGBRanker(BoostedRanker):
    """Gradient-boosted trees from XGBoost, trained to rank the candidates of each query.

    The group sizes become XGBoost's ``group``, so the pairwise and listwise objectives
    compare candidates only within a query. Missing features -- NaN, as a
    :class:`~skrecsys.compose.JoinStaticFeatures` writes for an unknown identifier -- are
    handled natively. The parameters keep the names of ``xgboost.XGBRanker``.

    Parameters
    ----------
    objective : str, default="rank:ndcg"
        An XGBoost ranking objective: ``"rank:ndcg"``, ``"rank:pairwise"`` or
        ``"rank:map"``. ``"rank:ndcg"`` and ``"rank:map"`` need integer relevance labels.
    n_estimators : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=None
        ``None`` keeps XGBoost's default.
    max_depth : int, default=None
        Depth of each tree; ``None`` keeps XGBoost's default.
    min_child_weight : float, default=None
        Least hessian weight a leaf may hold; ``None`` keeps XGBoost's default of 1,
        which on a handful of candidates can leave the listwise objectives no split.
    reg_lambda : float, default=None
        L2 regularization of the leaf weights; ``None`` keeps XGBoost's default.
    random_state : int, RandomState instance or None, default=None
        Seeds XGBoost.
    n_jobs : int, default=None
        Threads for training and prediction; ``None`` uses every core.
    verbose : bool, default=False
        Whether XGBoost logs its warnings and progress.

    Attributes
    ----------
    model_ : xgboost.XGBRanker
        The fitted model.
    n_features_in_ : int

    Examples
    --------
    >>> from skrecsys.integrations.xgboost import XGBRanker
    >>> ranker = XGBRanker(n_estimators=10, min_child_weight=0, random_state=0).fit(
    ...     [[0.0], [1.0], [0.2], [0.9]], [0, 1, 0, 1], groups=[2, 2]
    ... )
    >>> ranker.predict([[0.1], [0.8]], groups=[2]).shape
    (2,)
    """

    def __init__(
        self,
        objective: str = "rank:ndcg",
        *,
        n_estimators: int = 100,
        learning_rate: float | None = None,
        max_depth: int | None = None,
        min_child_weight: float | None = None,
        reg_lambda: float | None = None,
        random_state: RandomStateLike = None,
        n_jobs: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    @override
    def _fit_model(
        self, X: NDArray[np.float64], labels: NDArray[np.float64], sizes: NDArray[np.int64]
    ) -> None:
        self.model_ = _XGBRanker(
            objective=self.objective,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            random_state=self._seed(),
            n_jobs=self.n_jobs,
            verbosity=1 if self.verbose else 0,
        )
        self.model_.fit(X, labels, group=sizes)

    @override
    def _predict_model(self, X: NDArray[np.float64]) -> ArrayLike:
        return self.model_.predict(X)
