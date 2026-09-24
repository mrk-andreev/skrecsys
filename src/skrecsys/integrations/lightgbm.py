"""LightGBM integration: a ranker for :class:`~skrecsys.compose.Cascade` backed by LightGBM.

LightGBM is a third-party library and an optional extra::

    pip install skrecsys[lightgbm]

Nothing outside this module imports it.
"""

from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from skrecsys._typing import RandomStateLike, override
from skrecsys.integrations._base import BoostedRanker

try:
    from lightgbm.sklearn import LGBMRanker as _LGBMRanker
except ModuleNotFoundError as _exc:  # pragma: no cover - covered by the lightgbm-free CI matrix
    if _exc.name != "lightgbm":
        raise
    raise ImportError(
        "skrecsys.integrations.lightgbm requires LightGBM. "
        "Install it with `pip install skrecsys[lightgbm]`."
    ) from _exc

__all__ = ["LGBMRanker"]


class LGBMRanker(BoostedRanker):
    """Gradient-boosted trees from LightGBM, trained to rank the candidates of each query.

    The group sizes become LightGBM's ``group``, so the listwise objectives compare
    candidates only within a query. Missing features -- NaN, as a
    :class:`~skrecsys.compose.JoinStaticFeatures` writes for an unknown identifier -- are
    handled natively. The parameters keep the names of ``lightgbm.LGBMRanker``.

    Parameters
    ----------
    objective : str, default="lambdarank"
        Any LightGBM objective, such as ``"lambdarank"``, ``"rank_xendcg"`` or the
        pointwise ``"binary"``. The ranking objectives need integer relevance labels.
    n_estimators : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=0.1
        Shrinkage of each tree.
    num_leaves : int, default=31
        Maximum leaves of each tree.
    max_depth : int, default=-1
        Depth limit of each tree; -1 means none.
    min_child_samples : int, default=20
        Fewest candidates a leaf may hold.
    reg_lambda : float, default=0.0
        L2 regularization of the leaf values.
    random_state : int, RandomState instance or None, default=None
        Seeds LightGBM.
    n_jobs : int, default=None
        Threads for training and prediction; ``None`` lets LightGBM choose.
    verbose : bool, default=False
        Whether LightGBM logs its warnings and progress.

    Attributes
    ----------
    model_ : lightgbm.LGBMRanker
        The fitted model.
    n_features_in_ : int

    Examples
    --------
    >>> from skrecsys.integrations.lightgbm import LGBMRanker
    >>> ranker = LGBMRanker(n_estimators=10, min_child_samples=1, random_state=0).fit(
    ...     [[0.0], [1.0], [0.2], [0.9]], [0, 1, 0, 1], groups=[2, 2]
    ... )
    >>> ranker.predict([[0.1], [0.8]], groups=[2]).shape
    (2,)
    """

    def __init__(
        self,
        objective: str = "lambdarank",
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        reg_lambda: float = 0.0,
        random_state: RandomStateLike = None,
        n_jobs: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    @override
    def _fit_model(
        self, X: NDArray[np.float64], labels: NDArray[np.float64], sizes: NDArray[np.int64]
    ) -> None:
        self.model_ = _LGBMRanker(
            objective=self.objective,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            reg_lambda=self.reg_lambda,
            random_state=self._seed(),
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else -1,
        )
        self.model_.fit(X, labels, group=sizes)

    @override
    def _predict_model(self, X: NDArray[np.float64]) -> ArrayLike:
        # A plain dense X always gives a dense array; sparse output needs pred_contrib.
        return cast("NDArray[np.float64]", self.model_.predict(X))
