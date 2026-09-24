"""Factorization machine fitted by alternating least squares, ported from libFM."""

import numbers
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from sklearn.utils import check_random_state

from skrecsys import _core
from skrecsys._typing import override
from skrecsys.indexing import DenseSpace
from skrecsys.recommendation._base import BaseRecommender, kernel_indices
from skrecsys.recommendation._incremental import IncrementalRecommenderMixin


class AlternatingLeastSquares(IncrementalRecommenderMixin, BaseRecommender):
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
    n_iter_partial : int or None, default=None
        Sweeps run by ``partial_fit``, which start from the parameters already fitted
        rather than from fresh draws. ``None`` reuses ``n_iter``, which is generous for
        a small batch: a warm start has much less ground to cover than a cold one.
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
    index : None, str or VectorIndex, default=None
        Approximate index used by ``recommend``. ``None`` scores every candidate
        exactly; ``"hnsw"`` or a configured :class:`~skrecsys.indexing.HNSW` walks a
        graph instead, which is faster on a large catalog and no longer exact. See
        :mod:`skrecsys.indexing`.

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

    Notes
    -----
    ``partial_fit`` resumes the sweeps from the fitted parameters instead of redrawing
    them, over everything seen so far rather than over the batch alone -- libFM's
    per-parameter closed form is a minimizer over every case the parameter appears in,
    so a sweep restricted to a batch would be solving a different problem. The cost is
    linear in the stored interactions, not cubic in the catalog, and nothing learned is
    discarded. It is a fold-in, so the result is not what ``fit`` on every batch
    concatenated would have produced.

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
        n_iter_partial: int | None = None,
        init_stdev: float = 0.1,
        reg_global: float = 0.0,
        reg_bias: float = 1.0,
        reg_factors: float = 10.0,
        random_state: int | np.random.RandomState | None = None,
        index: Any = None,
    ) -> None:
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.n_iter_partial = n_iter_partial
        self.init_stdev = init_stdev
        self.reg_global = reg_global
        self.reg_bias = reg_bias
        self.reg_factors = reg_factors
        self.random_state = random_state
        self.index = index

    @override
    def predict(self, X: ArrayLike) -> NDArray[np.floating]:
        return np.clip(super().predict(X), *self.target_range_)

    _incremental_state_ = (
        ("user_bias_", "user"),
        ("item_bias_", "item"),
        ("user_factors_", "user"),
        ("item_factors_", "item"),
    )

    @override
    def _fit(self, interactions: sp.csr_array) -> None:
        self._check_params()
        n_users, n_items = interactions.shape
        n_features = n_users + n_items

        rng = check_random_state(self.random_state)
        self._rng = rng
        v = np.ascontiguousarray(rng.normal(0.0, self.init_stdev, (self.n_factors, n_features)))
        w = rng.normal(0.0, self.init_stdev, n_features)
        w0, loss_curve, y = self._sweep(interactions, 0.0, w, v, int(self.n_iter))

        self._store(n_users, w0, w, v)
        self.loss_curve_ = loss_curve
        self.target_range_ = (float(y.min()), float(y.max()))

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
        """Fold the batch in by resuming the sweeps from the fitted parameters.

        The sweeps run over everything seen so far, not over the batch alone, and that
        is not an oversight: libFM sets each parameter to the minimizer of the loss over
        every case the parameter appears in, using a residual cache it maintains over
        all of them, so a sweep restricted to the batch would be minimizing a different
        objective. The cost is linear in the stored interactions rather than cubic in
        the catalog, and nothing already learned is thrown away -- which is exactly what
        ``fit`` would do, since it redraws every parameter from ``N(0, init_stdev)``.

        The result is therefore a fold-in and not what ``fit`` on the concatenation of
        every batch would produce: the parameters start from a different point, and a
        finite number of sweeps does not forget where it started.
        """
        del delta, touched_user_indices, touched_item_indices
        n_users = interactions.shape[0]
        rng = self._incremental_rng()
        stdev = float(self.init_stdev)
        # `_remap` left the new rows at zero, which is a fixed point of the update for a
        # factor row, so they start from the same distribution a fresh fit draws.
        for indices, biases, factors in (
            (new_user_indices, self.user_bias_, self.user_factors_),
            (new_item_indices, self.item_bias_, self.item_factors_),
        ):
            if len(indices):
                biases[indices] = rng.normal(0.0, stdev, len(indices))
                factors[indices] = rng.normal(0.0, stdev, (len(indices), self.n_factors))

        w = np.concatenate([self.user_bias_, self.item_bias_])
        v = np.ascontiguousarray(np.concatenate([self.user_factors_, self.item_factors_]).T)
        w0, loss_curve, y = self._sweep(
            interactions, float(self.global_bias_), w, v, self._partial_sweeps()
        )

        self._store(n_users, w0, w, v)
        # The curve is the whole training history, so a caller can see every batch.
        self.loss_curve_ = np.concatenate([self.loss_curve_, loss_curve])
        low, high = self.target_range_
        self.target_range_ = (min(low, float(y.min())), max(high, float(y.max())))

    def _partial_sweeps(self) -> int:
        """Sweeps a ``partial_fit`` runs."""
        if self.n_iter_partial is None:
            return int(self.n_iter)
        if not isinstance(self.n_iter_partial, numbers.Integral) or self.n_iter_partial < 0:
            raise ValueError(
                f"n_iter_partial must be None or an integer >= 0, got {self.n_iter_partial!r}."
            )
        return int(self.n_iter_partial)

    def _sweep(
        self,
        interactions: sp.csr_array,
        w0: float,
        w: NDArray[np.float64],
        v: NDArray[np.float64],
        n_iter: int,
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        """Run ``n_iter`` ALS sweeps in place on ``w`` and ``v``.

        Each interaction is one factorization-machine case that one-hot encodes its user
        and its item, so the design matrix has exactly two nonzeros per row and the
        parameter vector is the users' followed by the items'.
        """
        n_users, n_items = interactions.shape
        coo = interactions.tocoo()
        n_cases = coo.nnz
        indices = kernel_indices(np.column_stack([coo.row, n_users + coo.col]).ravel())
        indptr = np.arange(0, 2 * n_cases + 1, 2, dtype=np.int64)
        y = np.asarray(coo.data, dtype=np.float64)
        group = np.repeat(np.array([0, 1], dtype=np.int64), [n_users, n_items])
        updated, loss_curve = _core.fm_als_fit(
            indptr,
            indices,
            np.ones(2 * n_cases),
            y,
            group,
            w0,
            w,
            v,
            float(self.reg_global),
            np.full(2, float(self.reg_bias)),
            np.full(2, float(self.reg_factors)),
            int(n_iter),
        )
        return updated, np.asarray(loss_curve), y

    def _store(
        self, n_users: int, w0: float, w: NDArray[np.float64], v: NDArray[np.float64]
    ) -> None:
        """Split the factorization-machine parameter vectors back into fitted arrays."""
        self.global_bias_ = w0
        self.user_bias_, self.item_bias_ = w[:n_users], w[n_users:]
        self.user_factors_ = v[:, :n_users].T.copy()
        self.item_factors_ = v[:, n_users:].T.copy()

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

    @override
    def _rank_chunk_size(self, n_candidates: int, k: int) -> int:
        # The kernel builds no dense score matrix, so a block only holds its results.
        del n_candidates, k
        return 65_536

    @override
    def _rank_queries_exact(
        self,
        queries: NDArray[Any],
        item_indices: NDArray[np.intp],
        k: int,
        *,
        exclude_seen: bool,
        excluded: sp.csr_array | None = None,
        first_query: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.floating]]:
        # `w0 + b_u + b_i + <p_u, q_i>`: the item bias rides along with the scan and the
        # user bias is the per-query offset.
        order, scores = self._rank_by_factors(
            queries,
            item_indices,
            k,
            exclude_seen=exclude_seen,
            excluded=excluded,
            first_query=first_query,
            user_factors=self.user_factors_,
            item_factors=self.item_factors_,
            item_bias=self.item_bias_,
            user_offset=self.user_bias_,
        )
        # Constant across every user and item, so it is added once to the results
        # rather than broadcast into the scan.
        scores += self.global_bias_
        return order, scores

    @override
    def _index_space(self) -> DenseSpace:
        # `r_ui = w0 + b_u + b_i + <p_u, q_i>`, of which only `b_i` and the dot product
        # vary with the item, so one extra dimension carries the item bias and the
        # global bias with it. `b_u` is the same for every item this query could return
        # and is added back afterwards.
        items = np.empty((self.n_items_, self.n_factors + 1), dtype=np.float64)
        items[:, : self.n_factors] = self.item_factors_
        items[:, self.n_factors] = self.item_bias_ + self.global_bias_
        return DenseSpace(items)

    @override
    def _index_queries(self, user_indices: NDArray[np.intp]) -> NDArray[np.float64]:
        queries = np.empty((len(user_indices), self.n_factors + 1), dtype=np.float64)
        queries[:, : self.n_factors] = self.user_factors_[user_indices]
        queries[:, self.n_factors] = 1.0
        return queries

    @override
    def _index_score_offset(self, user_indices: NDArray[np.intp]) -> NDArray[np.float64]:
        return np.asarray(self.user_bias_[user_indices], dtype=np.float64)

    def _check_params(self) -> None:
        for name in ("n_factors", "n_iter"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0, got {value!r}.")
        for name in ("init_stdev", "reg_global", "reg_bias", "reg_factors"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Real) or not value >= 0:
                raise ValueError(f"{name} must be a real number >= 0, got {value!r}.")
