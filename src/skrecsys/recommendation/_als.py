"""Factorization machine fitted by alternating least squares, ported from libFM."""

import numbers

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import check_random_state

from skrecsys import _core
from skrecsys._typing import override
from skrecsys.recommendation._base import BaseRecommender, kernel_indices


class AlternatingLeastSquares(BaseRecommender):
    """Biased matrix factorization fitted by alternating least squares, as in libFM.

    Each interaction is a factorization machine input that one-hot encodes the user
    and the item, so the model is

    ``r_ui = w0 + b_u + b_i + <p_u, q_i>``.

    Every parameter is set in turn to the minimizer of the squared loss over the
    observed interactions plus L2 penalties, with the other parameters held fixed
    [1]_. The Rust core is a port of libFM's ALS learner [2]_.

    The loss covers only observed entries, so the model is for explicit feedback
    such as ratings. When ``y`` is None, every target is 1 and the fit is degenerate.

    As in libFM, biases and factors start from ``N(0, init_stdev)`` and the fitted
    model holds the parameters from the final sweep.

    Parameters
    ----------
    n_factors : int, default=8
        Dimensionality of the latent factors.
    n_iter : int, default=100
        Number of alternating sweeps over all parameters.
    init_stdev : float, default=0.1
        Standard deviation of the normal distribution the biases and factors start from.
    reg_global : float, default=0.0
        L2 penalty on the global bias.
    reg_bias : float, default=1.0
        L2 penalty on user and item biases.
    reg_factors : float, default=10.0
        L2 penalty on user and item factors. libFM defaults to no regularization, but
        without it ALS overfits quickly.
    random_state : int, RandomState instance or None, default=None
        Seed for factor initialization.

    Attributes
    ----------
    global_bias_ : float
    user_bias_ : ndarray of shape (n_users_,)
    item_bias_ : ndarray of shape (n_items_,)
    user_factors_ : ndarray of shape (n_users_, n_factors)
    item_factors_ : ndarray of shape (n_items_, n_factors)
    loss_curve_ : ndarray of shape (n_iter,)
        Training RMSE after each sweep.
    target_range_ : tuple of float
        Minimum and maximum training target. ``predict`` clips to it, as libFM does;
        ``recommend`` uses unclipped scores.

    References
    ----------
    .. [1] S. Rendle, Z. Gantner, C. Freudenthaler, and L. Schmidt-Thieme, "Fast
       Context-aware Recommendations with Factorization Machines", SIGIR 2011.
       https://doi.org/10.1145/2009916.2010002
    .. [2] S. Rendle, "Factorization Machines with libFM", ACM TIST 2012.
       https://doi.org/10.1145/2168752.2168771

    Examples
    --------
    >>> from skrecsys.recommendation import AlternatingLeastSquares
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "b"], ["u3", "a"]]
    >>> y = [5, 1, 4, 2, 5]
    >>> rec = AlternatingLeastSquares(n_factors=2, random_state=0).fit(X, y)
    >>> rec.recommend(["u3"], n_recommendations=1, exclude_seen=False)[0].tolist()
    [['a']]
    """

    def __init__(
        self,
        n_factors: int = 8,
        n_iter: int = 100,
        init_stdev: float = 0.1,
        reg_global: float = 0.0,
        reg_bias: float = 1.0,
        reg_factors: float = 10.0,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.init_stdev = init_stdev
        self.reg_global = reg_global
        self.reg_bias = reg_bias
        self.reg_factors = reg_factors
        self.random_state = random_state

    @override
    def predict(self, X: ArrayLike) -> NDArray[np.floating]:
        return np.clip(super().predict(X), *self.target_range_)

    @override
    def _fit(self, interactions: sp.csr_array) -> None:
        self._check_params()
        n_users, n_items = interactions.shape
        n_features = n_users + n_items
        coo = interactions.tocoo()
        n_cases = coo.nnz
        indices = kernel_indices(np.column_stack([coo.row, n_users + coo.col]).ravel())
        indptr = np.arange(0, 2 * n_cases + 1, 2, dtype=np.int64)
        y = np.asarray(coo.data, dtype=np.float64)
        group = np.repeat(np.array([0, 1], dtype=np.int64), [n_users, n_items])

        rng = check_random_state(self.random_state)
        v = np.ascontiguousarray(rng.normal(0.0, self.init_stdev, (self.n_factors, n_features)))
        w = rng.normal(0.0, self.init_stdev, n_features)
        w0, loss_curve = _core.fm_als_fit(
            indptr,
            indices,
            np.ones(2 * n_cases),
            y,
            group,
            0.0,
            w,
            v,
            float(self.reg_global),
            np.full(2, float(self.reg_bias)),
            np.full(2, float(self.reg_factors)),
            int(self.n_iter),
        )

        self.global_bias_ = w0
        self.user_bias_, self.item_bias_ = w[:n_users], w[n_users:]
        self.user_factors_ = v[:, :n_users].T.copy()
        self.item_factors_ = v[:, n_users:].T.copy()
        self.loss_curve_ = np.asarray(loss_curve)
        self.target_range_ = (float(y.min()), float(y.max()))

    @override
    def _score_users(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        return (
            self.global_bias_
            + self.user_bias_[user_indices, None]
            + self.item_bias_[None, item_indices]
            + self.user_factors_[user_indices] @ self.item_factors_[item_indices].T
        )

    @override
    def _score_pairs(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        factors = np.einsum(
            "ij,ij->i", self.user_factors_[user_indices], self.item_factors_[item_indices]
        )
        return (
            self.global_bias_
            + self.user_bias_[user_indices]
            + self.item_bias_[item_indices]
            + factors
        )

    def _check_params(self) -> None:
        for name in ("n_factors", "n_iter"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0, got {value!r}.")
        for name in ("init_stdev", "reg_global", "reg_bias", "reg_factors"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Real) or not value >= 0:
                raise ValueError(f"{name} must be a real number >= 0, got {value!r}.")
