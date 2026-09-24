"""SLIM: one elastic-net regression per item, ported from Ferrari Dacrema's framework."""

import math
import numbers
import warnings
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning

from skrecsys import _core
from skrecsys._typing import override
from skrecsys.indexing import SparseSpace
from skrecsys.recommendation._base import SimilarityRecommender, kernel_csr, kernel_indices
from skrecsys.recommendation._incremental import (
    IncrementalRecommenderMixin,
    affected_item_rows,
    remap_dense_square,
    replace_rows,
)


class SLIMElasticNet(IncrementalRecommenderMixin, SimilarityRecommender):
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
    index : None, str or VectorIndex, default=None
        Approximate index used by ``recommend``. ``None`` scores every candidate
        exactly; ``"hnsw"`` or a configured :class:`~skrecsys.indexing.HNSW` walks a
        graph instead. See :mod:`skrecsys.indexing`, and note that the exact path here
        is already an inverted-index scan that never touches an item nobody reached --
        so whether a graph beats it is a question for ``benchmarks/indexes.py``.

    Attributes
    ----------
    gram_ : ndarray of shape (n_items_, n_items_)
        The item Gram matrix every column's regression is solved from. Present only once
        ``partial_fit`` has been used.
    similarity_ : scipy.sparse.csr_array of shape (n_items_, n_items_)
        Item-item weights with a zero diagonal; entry ``(i, j)`` is the contribution of
        item ``i`` to the score of item ``j``, so column ``j`` holds one fitted
        regression.
    n_unconverged_ : int
        Number of columns that reached ``max_iter`` without meeting ``tol``. A warning
        is raised during ``fit`` when this is nonzero.

    Notes
    -----
    ``partial_fit`` keeps ``gram_``, updates it by the rank the batch has, and re-solves
    only the columns the batch can reach, warm-starting each from the weights it already
    had. It is an approximation, not a reproduction of ``fit`` on every batch
    concatenated: coordinate descent stops at a tolerance rather than at the optimum, so
    a warm start lands somewhere else inside it, and the l1 penalty scales with the
    number of users, so every new user moves the optimum of every column -- the
    unreachable ones included. Keeping ``gram_`` also doubles the model's memory, which
    is already quadratic in the catalog.

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
        index: Any = None,
    ) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_neighbors = n_neighbors
        self.positive = positive
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.index = index

    _incremental_state_ = (("similarity_", "item_item"),)

    @override
    def _fit(self, interactions: sp.csr_array) -> None:
        n_threads = self._check_params()
        n_items = interactions.shape[1]
        gram = _core.item_gram(*kernel_csr(interactions), n_items, n_threads)
        columns = self._solve(gram, interactions.shape[0], None, None, n_threads, n_items)
        # The kernel solves one column per row, so row j holds the model of item j.
        self.similarity_ = sp.csr_array(columns.T)
        # A plain `fit` keeps the memory it always did. The Gram matrix is dense and
        # quadratic in the catalog, and only an incremental fit has any use for it.
        self.__dict__.pop("gram_", None)

    @override
    def _remap(
        self,
        *,
        user_perm: NDArray[np.intp],
        item_perm: NDArray[np.intp],
        n_users: int,
        n_items: int,
    ) -> None:
        super()._remap(user_perm=user_perm, item_perm=item_perm, n_users=n_users, n_items=n_items)
        if "gram_" in self.__dict__:
            # An item the model has not seen has an all-zero Gram row and column, which
            # is what `remap_dense_square` leaves behind.
            self.gram_ = remap_dense_square(self.gram_, item_perm, n_items)

    @override
    def _partial_fit(
        self,
        interactions: sp.csr_array,
        *,
        delta: sp.csr_array,
        new_user_indices: NDArray[np.intp],
        new_item_indices: NDArray[np.intp],
        touched_user_indices: NDArray[np.intp],
        touched_item_indices: NDArray[np.intp],
    ) -> None:
        """Re-solve the columns the batch can reach, warm-started, over an updated Gram.

        The Gram matrix is a sum of the users' outer products, so the batch changes it by
        removing what the touched users used to contribute and adding what they
        contribute now -- a rank-limited update rather than another pass over the
        interactions.

        This is the one estimator here whose ``partial_fit`` is **not** what ``fit`` on
        every batch concatenated would give, and for two reasons worth stating. The
        descent stops at a duality gap rather than at the exact optimum, so a warm start
        lands at a different point inside the same tolerance. And the l1 penalty is
        scaled by the number of users, so every new user moves every column's optimum --
        including the columns the batch cannot reach, which are not re-solved.
        """
        del new_user_indices, new_item_indices, touched_item_indices
        n_threads = self._check_params()
        n_users, n_items = interactions.shape
        if "gram_" not in self.__dict__:
            # The model was `fit`, not `partial_fit`: there is no Gram matrix to update,
            # and one pass over the interactions builds the one every later call keeps.
            self.gram_ = _core.item_gram(*kernel_csr(interactions), n_items, n_threads)
            columns = self._solve(self.gram_, n_users, None, None, n_threads, n_items)
            self.similarity_ = sp.csr_array(columns.T)
            return

        after = sp.csr_array(interactions[touched_user_indices])
        before = after - sp.csr_array(delta[touched_user_indices])
        # Kept sparse on the way in: the difference of the two outer products has only
        # the touched users' items in it, and densifying it would cost the catalog
        # squared twice over.
        difference = sp.coo_array(sp.csr_array(after.T @ after) - sp.csr_array(before.T @ before))
        self.gram_[difference.row, difference.col] += difference.data

        moved = np.unique(after.indices)
        targets = affected_item_rows(interactions, moved.astype(np.intp))
        stored = sp.csr_array(self.similarity_.T)
        block = self._solve(
            self.gram_, n_users, targets, sp.csr_array(stored[targets]), n_threads, n_items
        )
        self.similarity_ = sp.csr_array(replace_rows(stored, targets, block).T)

    def _solve(
        self,
        gram: NDArray[np.float64],
        n_users: int,
        targets: NDArray[np.intp] | None,
        warm: sp.csr_array | None,
        n_threads: int,
        n_items: int,
    ) -> sp.csr_array:
        """Solve the elastic net of each target column; row ``t`` is target ``t``."""
        (indptr, indices, data), unconverged = _core.slim_elasticnet_weights_from_gram(
            gram,
            float(n_users),
            float(self.alpha),
            float(self.l1_ratio),
            bool(self.positive),
            int(self.max_iter),
            float(self.tol),
            int(self.n_neighbors),
            n_threads,
            None if targets is None else kernel_indices(targets),
            *(
                (None, None, None)
                if warm is None
                else (
                    kernel_indices(warm.indptr),
                    kernel_indices(warm.indices),
                    np.ascontiguousarray(warm.data, dtype=np.float64),
                )
            ),
        )
        solved = n_items if targets is None else len(targets)
        self.n_unconverged_ = int(unconverged)
        if unconverged:
            warnings.warn(
                f"{unconverged} of {solved} columns did not converge within "
                f"max_iter={self.max_iter}; raise max_iter, raise tol or raise alpha.",
                ConvergenceWarning,
                # _fit is called by fit, which the caller called.
                stacklevel=4,
            )
        return sp.csr_array((data, indices, indptr), shape=(solved, n_items), dtype=np.float64)

    @override
    def _index_space(self) -> SparseSpace:
        # Row `j` of `_neighbors_of_items` holds exactly the weights that make up item
        # `j`'s score, which is the item vector, and it is already built and cached.
        return SparseSpace(sp.csr_array(self._neighbors_of_items()))

    @override
    def _index_queries(self, user_indices: NDArray[np.intp]) -> sp.csr_array:
        return sp.csr_array(self.interactions_[user_indices])

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
