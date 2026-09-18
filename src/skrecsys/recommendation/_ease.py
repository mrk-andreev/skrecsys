"""Embarrassingly shallow autoencoder: a closed-form linear item-item model."""

import math
import numbers

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from skrecsys import _core
from skrecsys._typing import override
from skrecsys.recommendation._base import BaseRecommender, kernel_csr, score_pairs_from_similarity


class EASE(BaseRecommender):
    """Closed-form linear item-item model for implicit feedback [1]_.

    A single linear layer reconstructs each user's interaction vector from itself, with
    self-reconstruction forbidden. With ``G = R^T R`` the item Gram matrix and
    ``A = G + l2_reg * I``, the minimizer has the closed form

    ``B = I - A^-1 diag(1 / diag(A^-1))``,

    that is ``B[i, j] = -P[i, j] / P[j, j]`` with ``P = A^-1``, and ``B[j, j] = 0``. The
    score of item ``j`` for user ``u`` is ``sum_i r_ui * B[i, j]``. Interaction values
    are used as given, not binarized.

    Ported from the reference implementations in UniRec [2]_ and RecTools [3]_. Those
    invert ``A`` with a general-purpose routine; since ``A`` is symmetric positive
    definite for ``l2_reg > 0``, this implementation factors it by Cholesky instead,
    which is cheaper and better conditioned. The two agree up to rounding. Ties rank by
    fitted item order.

    The model is dense in the number of items: fitting costs ``O(n_items^3)`` time and
    ``O(n_items^2)`` memory, which is what bounds the usable catalog size.

    Parameters
    ----------
    l2_reg : float, default=500.0
        L2 penalty added to the diagonal of the item Gram matrix. Must be positive;
        larger values shrink the weights.

    n_jobs : int or None, default=None
        Number of threads used to fit. ``None`` or ``-1`` uses all cores.

    Attributes
    ----------
    similarity_ : ndarray of shape (n_items_, n_items_)
        Item-item weights with a zero diagonal; entry ``(i, j)`` is the contribution of
        item ``i`` to the score of item ``j``. Dense, unlike the sparse ``similarity_``
        of the neighbourhood models.

    References
    ----------
    .. [1] H. Steck, "Embarrassingly Shallow Autoencoders for Sparse Data", WWW 2019.
       https://doi.org/10.1145/3308558.3313710
    .. [2] Microsoft, "UniRec", ``unirec/model/cf/ease.py``.
       https://github.com/microsoft/UniRec
    .. [3] MTS, "RecTools", ``rectools/models/ease.py``.
       https://github.com/MobileTeleSystems/RecTools

    Examples
    --------
    >>> from skrecsys.recommendation import EASE
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "b"], ["u3", "a"]]
    >>> rec = EASE(l2_reg=1.0).fit(X)
    >>> rec.recommend(["u3"], n_recommendations=1)[0].tolist()
    [['b']]
    """

    def __init__(self, l2_reg: float = 500.0, n_jobs: int | None = None) -> None:
        self.l2_reg = l2_reg
        self.n_jobs = n_jobs

    @override
    def _fit(self, interactions: sp.csr_array) -> None:
        n_threads = self._check_params()
        self.similarity_ = _core.ease_weights(
            *kernel_csr(interactions),
            interactions.shape[1],
            float(self.l2_reg),
            n_threads,
        )

    @override
    def _score_users(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        scores = self.interactions_[user_indices] @ self.similarity_
        return np.asarray(scores, dtype=np.float64)[:, item_indices]

    @override
    def _score_pairs(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        return score_pairs_from_similarity(
            self.interactions_, self.similarity_, user_indices, item_indices, by_row=False
        )

    def _check_params(self) -> int:
        """Validate parameters and return the thread count for the kernel (0 = all)."""
        if (
            not isinstance(self.l2_reg, numbers.Real)
            or not math.isfinite(float(self.l2_reg))
            or self.l2_reg <= 0
        ):
            raise ValueError(f"l2_reg must be a finite real number > 0, got {self.l2_reg!r}.")
        if self.n_jobs is None or self.n_jobs == -1:
            return 0
        if not isinstance(self.n_jobs, numbers.Integral) or self.n_jobs < 1:
            raise ValueError(f"n_jobs must be None, -1 or an integer >= 1, got {self.n_jobs!r}.")
        return int(self.n_jobs)
