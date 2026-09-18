"""SLIM: one elastic-net regression per item, ported from Ferrari Dacrema's framework."""

import math
import numbers
import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.exceptions import ConvergenceWarning

from skrecsys import _core
from skrecsys._typing import override
from skrecsys.recommendation._base import SimilarityRecommender, kernel_csr


class SLIMElasticNet(SimilarityRecommender):
    """Sparse linear item-item model whose columns are elastic-net regressions [1]_.

    Item ``j`` is predicted from every other item by a linear model fitted on the
    interaction matrix ``R``: column ``j`` of the weight matrix minimizes

    ``1 / (2 n_users_) * ||R[:, j] - R[:, -j] w||^2 + alpha * l1_ratio * ||w||_1
    + 0.5 * alpha * (1 - l1_ratio) * ||w||^2``

    over ``w``, by default constrained to be non-negative [2]_. The L1 term is what
    makes the model sparse, and self-similarity is excluded rather than penalized, so
    the diagonal is zero by construction. The score of item ``j`` for user ``u`` is
    ``sum_i r_ui * w(i, j)`` over the items ``i`` of the user.

    This is a port of ``SLIMElasticNetRecommender`` from the evaluation framework of
    Ferrari Dacrema et al. [3]_, which fits each column with scikit-learn's
    :class:`~sklearn.linear_model.ElasticNet`; the parameters keep scikit-learn's names
    and meaning. The reference passes the whole interaction matrix as the design matrix
    once per item, while the kernel here forms the item Gram matrix ``R^T R`` once and
    runs coordinate descent against it, which is the same problem and the same solution.
    Two differences are visible in the fitted weights: the reference computes in float32
    and visits coordinates in a random order, so its weights agree with ours only to
    float32 precision; and it keeps ``min(nnz - 1, n_neighbors)`` weights per column,
    dropping the smallest one when a column has no more than ``n_neighbors`` of them,
    where this implementation keeps all ``n_neighbors``.

    Like :class:`~skrecsys.recommendation.EASE`, fitting needs ``O(n_items^2)`` memory
    for the Gram matrix, which is what bounds the usable catalog size.

    Parameters
    ----------
    alpha : float, default=1.0
        Strength of the elastic-net penalty. Must be positive; larger values give
        sparser, more shrunken columns.
    l1_ratio : float, default=0.1
        Mix of the two penalties, from 0 for ridge to 1 for lasso.
    n_neighbors : int, default=100
        Number of largest weights kept per column, the reference's ``topK``.
    positive : bool, default=True
        Constrain the weights to be non-negative, as the reference does by default.
    max_iter : int, default=100
        Maximum number of coordinate-descent sweeps per column.
    tol : float, default=1e-4
        Stop a column once its duality gap falls below ``tol * ||R[:, j]||^2``.
    n_jobs : int or None, default=None
        Number of threads used to fit. ``None`` or ``-1`` uses all cores.

    Attributes
    ----------
    similarity_ : scipy.sparse.csr_array of shape (n_items_, n_items_)
        Item-item weights with a zero diagonal; entry ``(i, j)`` is the contribution of
        item ``i`` to the score of item ``j``, so column ``j`` holds one fitted
        regression.
    n_unconverged_ : int
        Number of columns that reached ``max_iter`` without meeting ``tol``. A warning
        is raised during ``fit`` when this is nonzero.

    References
    ----------
    .. [1] X. Ning and G. Karypis, "SLIM: Sparse Linear Methods for Top-N Recommender
       Systems", ICDM 2011. https://doi.org/10.1109/ICDM.2011.134
    .. [2] M. Levy and K. Jack, "Efficient Top-N Recommendation by Linear Regression",
       LSRS 2013, which replaces SLIM's bound-constrained solver with elastic net.
    .. [3] M. Ferrari Dacrema et al., "Are We Really Making Much Progress? A Worrying
       Analysis of Recent Neural Recommendation Approaches", RecSys 2019.
       https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation

    Examples
    --------
    >>> from skrecsys.recommendation import SLIMElasticNet
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "b"], ["u3", "a"]]
    >>> rec = SLIMElasticNet(alpha=0.01).fit(X)
    >>> rec.recommend(["u3"], n_recommendations=1)[0].tolist()
    [['b']]
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.1,
        n_neighbors: int = 100,
        *,
        positive: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
        n_jobs: int | None = None,
    ) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_neighbors = n_neighbors
        self.positive = positive
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs

    @override
    def _fit(self, interactions: sp.csr_array) -> None:
        n_threads = self._check_params()
        n_items = interactions.shape[1]
        (indptr, indices, data), unconverged = _core.slim_elasticnet_weights(
            *kernel_csr(interactions),
            n_items,
            float(self.alpha),
            float(self.l1_ratio),
            bool(self.positive),
            int(self.max_iter),
            float(self.tol),
            int(self.n_neighbors),
            n_threads,
        )
        # The kernel solves one column per row, so row j holds the model of item j.
        columns = sp.csr_array((data, indices, indptr), shape=(n_items, n_items), dtype=np.float64)
        self.similarity_ = sp.csr_array(columns.T)
        self.n_unconverged_ = int(unconverged)
        if unconverged:
            warnings.warn(
                f"{unconverged} of {n_items} columns did not converge within "
                f"max_iter={self.max_iter}; raise max_iter, raise tol or raise alpha.",
                ConvergenceWarning,
                # _fit is called by fit, which the caller called.
                stacklevel=3,
            )

    @override
    def _check_params(self) -> int:
        """Validate parameters and return the thread count for the kernel (0 = all)."""
        if (
            not isinstance(self.alpha, numbers.Real)
            or not math.isfinite(float(self.alpha))
            or self.alpha <= 0
        ):
            raise ValueError(f"alpha must be a finite real number > 0, got {self.alpha!r}.")
        if not isinstance(self.l1_ratio, numbers.Real) or not 0 <= float(self.l1_ratio) <= 1:
            raise ValueError(f"l1_ratio must be a real number in [0, 1], got {self.l1_ratio!r}.")
        if not isinstance(self.n_neighbors, numbers.Integral) or self.n_neighbors < 1:
            raise ValueError(f"n_neighbors must be an integer >= 1, got {self.n_neighbors!r}.")
        if not isinstance(self.positive, bool):
            raise ValueError(f"positive must be a bool, got {self.positive!r}.")
        if not isinstance(self.max_iter, numbers.Integral) or self.max_iter < 1:
            raise ValueError(f"max_iter must be an integer >= 1, got {self.max_iter!r}.")
        if (
            not isinstance(self.tol, numbers.Real)
            or not math.isfinite(float(self.tol))
            or self.tol <= 0
        ):
            raise ValueError(f"tol must be a finite real number > 0, got {self.tol!r}.")
        if self.n_jobs is None or self.n_jobs == -1:
            return 0
        if not isinstance(self.n_jobs, numbers.Integral) or self.n_jobs < 1:
            raise ValueError(f"n_jobs must be None, -1 or an integer >= 1, got {self.n_jobs!r}.")
        return int(self.n_jobs)
