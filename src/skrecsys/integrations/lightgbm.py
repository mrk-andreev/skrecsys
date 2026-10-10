"""LightGBM integration: a ranker for :class:`~skrecsys.compose.Cascade` backed by LightGBM.

LightGBM is a third-party library and an optional extra::

    pip install skrecsys[lightgbm]

Nothing outside this module imports it.
"""

from collections.abc import Mapping
from typing import Annotated, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from skrecsys.integrations._base import BoostedRanker
from skrecsys.tune import Float, Int
from skrecsys.typing import RandomStateLike, override

try:
    import lightgbm
except ModuleNotFoundError as _exc:  # pragma: no cover - covered by the lightgbm-free CI matrix
    if _exc.name != "lightgbm":
        raise
    raise ImportError(
        "skrecsys.integrations.lightgbm requires LightGBM. "
        "Install it with `pip install skrecsys[lightgbm]`."
    ) from _exc

__all__ = ["LGBMRanker"]


#: LightGBM's other names for the parameters :class:`LGBMRanker` sets itself.
_ALIASES = (
    "num_iterations",
    "num_iteration",
    "n_iter",
    "num_tree",
    "num_trees",
    "num_round",
    "num_rounds",
    "num_boost_round",
    "eta",
    "shrinkage_rate",
    "num_leaf",
    "max_leaves",
    "max_leaf",
    "min_data_in_leaf",
    "min_data_per_leaf",
    "min_data",
    "lambda_l1",
    "l1_regularization",
    "lambda_l2",
    "lambda",
    "l2_regularization",
    "bagging_fraction",
    "sub_row",
    "bagging",
    "bagging_freq",
    "feature_fraction",
    "sub_feature",
    "min_gain_to_split",
    "seed",
    "random_seed",
    "num_threads",
    "num_thread",
    "nthread",
    "nthreads",
    "verbosity",
    "application",
    "app",
    "loss",
)


class LGBMRanker(BoostedRanker):
    """Gradient-boosted trees from LightGBM, trained to rank the candidates of each query.

    The group sizes become LightGBM's ``group``, so the listwise objectives compare
    candidates only within a query. Missing features -- NaN, as a
    :class:`~skrecsys.compose.JoinStaticFeatures` writes for an unknown identifier -- are
    handled natively. Training goes through LightGBM's core API, which works the same
    across its releases whatever the installed scikit-learn. The parameters keep the names
    of ``lightgbm.LGBMRanker``, and those
    worth tuning declare a search range for :class:`~skrecsys.tune.AutoTune`, which tunes
    them as ``ranker__<name>`` of the :class:`~skrecsys.compose.Cascade` holding the ranker.

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
    min_split_gain : float, default=0.0
        Least loss reduction a split must bring.
    reg_alpha : float, default=0.0
        L1 regularization of the leaf values.
    reg_lambda : float, default=0.0
        L2 regularization of the leaf values.
    subsample : float, default=1.0
        Fraction of the candidates each tree is grown on, redrawn every
        ``subsample_freq`` rounds; 1.0 uses them all.
    subsample_freq : int, default=1
        Rounds between redraws of ``subsample``; 0 turns subsampling off.
    colsample_bytree : float, default=1.0
        Fraction of the features each tree may split on.
    random_state : int, RandomState instance or None, default=None
        Seeds LightGBM.
    n_jobs : int, default=None
        Threads for training and prediction; ``None`` lets LightGBM choose.
    verbose : bool, default=False
        Whether LightGBM logs its warnings and progress.
    extra_params : dict, default=None
        Further LightGBM parameters, passed to ``lightgbm.train`` as they are, for instance
        ``{"lambdarank_truncation_level": 20}``. A parameter this class sets itself, or
        one of LightGBM's aliases for it, raises at ``fit``.

    Attributes
    ----------
    model_ : lightgbm.Booster
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
        n_estimators: Annotated[int, Int(50, 2000, log=True)] = 100,
        learning_rate: Annotated[float, Float(1e-3, 0.3, log=True)] = 0.1,
        num_leaves: Annotated[int, Int(8, 256, log=True)] = 31,
        max_depth: int = -1,
        min_child_samples: Annotated[int, Int(1, 200, log=True)] = 20,
        min_split_gain: float = 0.0,
        reg_alpha: Annotated[float, Float(0.0, 10.0)] = 0.0,
        reg_lambda: Annotated[float, Float(0.0, 10.0)] = 0.0,
        subsample: Annotated[float, Float(0.5, 1.0)] = 1.0,
        subsample_freq: int = 1,
        colsample_bytree: Annotated[float, Float(0.3, 1.0)] = 1.0,
        random_state: RandomStateLike = None,
        n_jobs: int | None = None,
        verbose: bool = False,
        extra_params: Mapping[str, object] | None = None,
    ) -> None:
        self.objective = objective
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.min_split_gain = min_split_gain
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.extra_params = extra_params

    @override
    def _fit_model(
        self, X: NDArray[np.float64], labels: NDArray[np.float64], sizes: NDArray[np.int64]
    ) -> None:
        # The core API rather than lightgbm.sklearn: the wrapper calls scikit-learn's
        # validation, whose signature changes under it (LightGBM 4.5 breaks on
        # scikit-learn 1.8+), while ``train`` has kept its shape across every release since
        # the floor. The names are those of lightgbm.LGBMRanker, all aliases LightGBM's core
        # accepts.
        params: dict[str, object] = {
            "objective": self.objective,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "min_split_gain": self.min_split_gain,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "subsample": self.subsample,
            "subsample_freq": self.subsample_freq,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self._seed(),
            "verbose": 1 if self.verbose else -1,
        }
        if self.n_jobs is not None:
            params["n_jobs"] = self.n_jobs
        params |= self._extra_params(_ALIASES)
        self.model_ = lightgbm.train(
            params,
            lightgbm.Dataset(X, label=labels, group=sizes),
            num_boost_round=self.n_estimators,
        )

    @override
    def _predict_model(self, X: NDArray[np.float64]) -> ArrayLike:
        # A plain dense X always gives a dense array; sparse output needs pred_contrib.
        return cast("NDArray[np.float64]", self.model_.predict(X))

    @override
    def _contributions_model(self, X: NDArray[np.float64]) -> ArrayLike:
        return cast("NDArray[np.float64]", self.model_.predict(X, pred_contrib=True))
