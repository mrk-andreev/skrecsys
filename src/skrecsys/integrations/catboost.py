"""CatBoost integration: a ranker for :class:`~skrecsys.compose.Cascade` backed by CatBoost.

CatBoost is a third-party library and an optional extra::

    pip install skrecsys[catboost]

Nothing outside this module imports it.
"""

from collections.abc import Mapping
from typing import Annotated

import numpy as np
from numpy.typing import ArrayLike, NDArray

from skrecsys._typing import RandomStateLike, override
from skrecsys.integrations._base import BoostedRanker
from skrecsys.tune import Float, Int

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


#: CatBoost's other names for the parameters :class:`CatBoostRanker` sets itself.
_ALIASES = (
    "objective",
    "num_boost_round",
    "n_estimators",
    "num_trees",
    "eta",
    "max_depth",
    "reg_lambda",
    "random_seed",
    "allow_writing_files",
    "colsample_bylevel",
    "max_bin",
    "n_jobs",
    "logging_level",
    "verbose_eval",
    "silent",
)

#: What CatBoost chooses when these are ``None``, from the data size and the device.
_LIBRARY_CHOSEN = ("learning_rate", "subsample", "rsm", "border_count")


class CatBoostRanker(BoostedRanker):
    """Gradient-boosted trees trained to rank the candidates of each query.

    The group sizes become CatBoost's ``group_id``, so the pairwise and listwise losses
    compare candidates only within a query. Missing features -- NaN, as a
    :class:`~skrecsys.compose.JoinStaticFeatures` writes for an unknown identifier -- are
    handled natively. The parameters worth tuning declare a search range for
    :class:`~skrecsys.tune.AutoTune`, which tunes them as ``ranker__<name>`` of the
    :class:`~skrecsys.compose.Cascade` holding the ranker. ``learning_rate`` and
    ``subsample`` default to ``None``, which no range contains, so the first trial of a
    search draws them rather than letting CatBoost choose.

    Parameters
    ----------
    loss_function : str, default="YetiRank"
        Any CatBoost ranking loss, such as ``"YetiRank"``, ``"PairLogit"``,
        ``"QuerySoftMax"`` or the pointwise ``"Logloss"``.
    iterations : int, default=500
        Number of trees.
    learning_rate : float, default=None
        ``None`` lets CatBoost choose from the data size.
    depth : int, default=6
        Depth of each tree.
    l2_leaf_reg : float, default=3.0
        L2 regularization of the leaf values.
    random_strength : float, default=1.0
        Scale of the noise added to split scores, against overfitting.
    subsample : float, default=None
        Fraction of the candidates each tree is grown on; ``None`` lets CatBoost choose,
        0.8 on large data and all of them on small.
    rsm : float, default=None
        Fraction of the features each split may consider; ``None`` is all of them.
    border_count : int, default=None
        Bins each numeric feature is split into; ``None`` keeps CatBoost's default.
    random_state : int, RandomState instance or None, default=None
        Seeds CatBoost.
    thread_count : int, default=-1
        Threads for training and prediction; -1 uses every core.
    verbose : bool, default=False
        Whether CatBoost logs its training.
    extra_params : dict, default=None
        Further ``catboost.CatBoost`` parameters, passed as they are, for instance
        ``{"grow_policy": "Lossguide", "max_leaves": 64}``. A parameter this class sets
        itself, or one of CatBoost's aliases for it, raises at ``fit``.

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
        iterations: Annotated[int, Int(100, 3000, log=True)] = 500,
        learning_rate: Annotated[float | None, Float(1e-3, 0.3, log=True)] = None,
        depth: Annotated[int, Int(4, 10)] = 6,
        l2_leaf_reg: Annotated[float, Float(1.0, 30.0, log=True)] = 3.0,
        random_strength: Annotated[float, Float(0.0, 10.0)] = 1.0,
        subsample: Annotated[float | None, Float(0.5, 1.0)] = None,
        rsm: float | None = None,
        border_count: int | None = None,
        random_state: RandomStateLike = None,
        thread_count: int = -1,
        verbose: bool = False,
        extra_params: Mapping[str, object] | None = None,
    ) -> None:
        self.loss_function = loss_function
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.subsample = subsample
        self.rsm = rsm
        self.border_count = border_count
        self.random_state = random_state
        self.thread_count = thread_count
        self.verbose = verbose
        self.extra_params = extra_params

    def _catboost_params(self) -> dict[str, object]:
        params: dict[str, object] = {
            "loss_function": self.loss_function,
            "iterations": self.iterations,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "random_strength": self.random_strength,
            "random_seed": self._seed(),
            "thread_count": self.thread_count,
            "verbose": bool(self.verbose),
            "allow_writing_files": False,
        }
        for name in _LIBRARY_CHOSEN:
            if (value := getattr(self, name)) is not None:
                params[name] = value
        return params | self._extra_params(_ALIASES)

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

    @override
    def _contributions_model(self, X: NDArray[np.float64]) -> ArrayLike:
        return self.model_.get_feature_importance(catboost.Pool(X), type="ShapValues")
