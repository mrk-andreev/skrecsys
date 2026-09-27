"""XGBoost integration: a ranker for :class:`~skrecsys.compose.Cascade` backed by XGBoost.

XGBoost is a third-party library and an optional extra::

    pip install skrecsys[xgboost]

Nothing outside this module imports it.
"""

from collections.abc import Mapping
from typing import Annotated

import numpy as np
from numpy.typing import ArrayLike, NDArray

from skrecsys._typing import RandomStateLike, override
from skrecsys.integrations._base import BoostedRanker
from skrecsys.tune import Categorical, Float, Int

try:
    import xgboost
except ModuleNotFoundError as _exc:  # pragma: no cover - covered by the xgboost-free CI matrix
    if _exc.name != "xgboost":
        raise
    raise ImportError(
        "skrecsys.integrations.xgboost requires XGBoost. "
        "Install it with `pip install skrecsys[xgboost]`."
    ) from _exc

__all__ = ["XGBRanker"]


#: XGBoost's other names for the parameters :class:`XGBRanker` sets itself.
_ALIASES = (
    "eta",
    "num_boost_round",
    "lambda",
    "alpha",
    "min_split_loss",
    "seed",
    "nthread",
    "verbosity",
    "silent",
)


class XGBRanker(BoostedRanker):
    """Gradient-boosted trees from XGBoost, trained to rank the candidates of each query.

    The group sizes become XGBoost's ``group``, so the pairwise and listwise objectives
    compare candidates only within a query. Missing features -- NaN, as a
    :class:`~skrecsys.compose.JoinStaticFeatures` writes for an unknown identifier -- are
    handled natively. Training goes through XGBoost's core API, which works the same across
    its releases whatever the installed scikit-learn. The parameters keep the names of
    ``xgboost.XGBRanker`` and default
    to XGBoost's own values, and those worth tuning declare a search range for
    :class:`~skrecsys.tune.AutoTune`, which tunes them as ``ranker__<name>`` of the
    :class:`~skrecsys.compose.Cascade` holding the ranker.

    Parameters
    ----------
    objective : str, default="rank:ndcg"
        An XGBoost ranking objective: ``"rank:ndcg"``, ``"rank:pairwise"`` or
        ``"rank:map"``. ``"rank:ndcg"`` and ``"rank:map"`` need integer relevance labels.
    n_estimators : int, default=100
        Number of boosting rounds.
    learning_rate : float, default=0.3
        Shrinkage of each tree.
    max_depth : int, default=6
        Depth of each tree.
    min_child_weight : float, default=1.0
        Least hessian weight a leaf may hold; on a handful of candidates the default can
        leave the listwise objectives no split, where 0 does not.
    gamma : float, default=0.0
        Least loss reduction a split must bring.
    reg_alpha : float, default=0.0
        L1 regularization of the leaf weights.
    reg_lambda : float, default=1.0
        L2 regularization of the leaf weights.
    subsample : float, default=1.0
        Fraction of the candidates each tree is grown on.
    colsample_bytree : float, default=1.0
        Fraction of the features each tree may split on.
    lambdarank_pair_method : {"topk", "mean"}, default="topk"
        How the listwise objectives pick the pairs they compare: those involving the
        top ``lambdarank_num_pair_per_sample`` candidates, or that many random pairs per
        candidate.
    lambdarank_num_pair_per_sample : int, default=None
        See ``lambdarank_pair_method``; ``None`` keeps XGBoost's default.
    random_state : int, RandomState instance or None, default=None
        Seeds XGBoost.
    n_jobs : int, default=None
        Threads for training and prediction; ``None`` uses every core.
    verbose : bool, default=False
        Whether XGBoost logs its warnings and progress.
    extra_params : dict, default=None
        Further XGBoost parameters, passed to ``xgboost.train`` as they are, for instance
        ``{"max_bin": 64, "tree_method": "hist"}``. A parameter this class sets itself,
        or one of XGBoost's aliases for it, raises at ``fit``.

    Attributes
    ----------
    model_ : xgboost.Booster
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
        n_estimators: Annotated[int, Int(50, 2000, log=True)] = 100,
        learning_rate: Annotated[float, Float(1e-3, 0.3, log=True)] = 0.3,
        max_depth: Annotated[int, Int(2, 12)] = 6,
        min_child_weight: Annotated[float, Float(0.0, 20.0)] = 1.0,
        gamma: Annotated[float, Float(0.0, 5.0)] = 0.0,
        reg_alpha: Annotated[float, Float(0.0, 10.0)] = 0.0,
        reg_lambda: Annotated[float, Float(0.0, 10.0)] = 1.0,
        subsample: Annotated[float, Float(0.5, 1.0)] = 1.0,
        colsample_bytree: Annotated[float, Float(0.3, 1.0)] = 1.0,
        lambdarank_pair_method: Annotated[str, Categorical(("topk", "mean"))] = "topk",
        lambdarank_num_pair_per_sample: int | None = None,
        random_state: RandomStateLike = None,
        n_jobs: int | None = None,
        verbose: bool = False,
        extra_params: Mapping[str, object] | None = None,
    ) -> None:
        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.lambdarank_pair_method = lambdarank_pair_method
        self.lambdarank_num_pair_per_sample = lambdarank_num_pair_per_sample
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.extra_params = extra_params

    @override
    def _fit_model(
        self, X: NDArray[np.float64], labels: NDArray[np.float64], sizes: NDArray[np.int64]
    ) -> None:
        # The core API rather than xgboost.sklearn, whose wrapper follows scikit-learn's
        # interfaces and so ties the installable XGBoost to the installed scikit-learn;
        # ``train`` has kept its shape across every release since the floor. The names
        # are those of xgboost.XGBRanker, all aliases XGBoost's core accepts.
        params: dict[str, object] = {
            "objective": self.objective,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "lambdarank_pair_method": self.lambdarank_pair_method,
            "random_state": self._seed(),
            "verbosity": 1 if self.verbose else 0,
        }
        if self.n_jobs is not None:
            params["n_jobs"] = self.n_jobs
        if self.lambdarank_num_pair_per_sample is not None:
            params["lambdarank_num_pair_per_sample"] = self.lambdarank_num_pair_per_sample
        params |= self._extra_params(_ALIASES)
        self.model_ = xgboost.train(
            params,
            _matrix(X, label=labels, group=sizes),
            num_boost_round=self.n_estimators,
        )

    @override
    def _predict_model(self, X: NDArray[np.float64]) -> ArrayLike:
        return self.model_.predict(_matrix(X))

    @override
    def _contributions_model(self, X: NDArray[np.float64]) -> ArrayLike:
        return self.model_.predict(_matrix(X), pred_contribs=True)


def _matrix(
    X: NDArray[np.float64],
    label: NDArray[np.float64] | None = None,
    group: NDArray[np.int64] | None = None,
) -> xgboost.DMatrix:
    """``X`` as XGBoost's matrix, NaN marking a missing feature."""
    return xgboost.DMatrix(X, label=label, group=group, missing=np.nan)
