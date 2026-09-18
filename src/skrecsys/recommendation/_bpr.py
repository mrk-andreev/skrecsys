"""BPR matrix factorization, ported from Cornac's ``BPR``."""

import numbers

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from skrecsys import _core
from skrecsys._typing import override
from skrecsys.recommendation._base import BaseRecommender, kernel_indices


class BayesianPersonalizedRanking(BaseRecommender):
    """Matrix factorization trained on pairwise ranking preferences [1]_.

    A user ``u`` scores item ``i`` as ``b_i + <p_u, q_i>``. Training draws triplets
    ``(u, i, j)`` where ``i`` is an item of the user and ``j`` is not, and takes a
    gradient step on

    ``log sigma(x_uij) - regularization * (||p_u||^2 + ||q_i||^2 + ||q_j||^2)``, where
    ``x_uij = b_i - b_j + <p_u, q_i - q_j>``,

    so the objective is how often the model ranks an observed item above an unobserved
    one, not how well it reconstructs the interaction values. Those values are ignored
    entirely: only which pairs are observed matters, which makes this an implicit
    feedback model and the natural counterpart of
    :class:`~skrecsys.recommendation.AlternatingLeastSquares`, whose squared loss
    covers observed ratings only.

    This is a port of ``BPR`` from Cornac [2]_, which follows implicit's: one epoch
    draws as many triplets as there are interactions, the positive is a uniformly drawn
    interaction — so active users are drawn more often — the negative a uniformly drawn
    item, and a triplet whose negative turns out to be one of the user's own items is
    skipped rather than redrawn. The defaults here are not Cornac's, which trains 100
    epochs at ``learning_rate=0.001`` and barely moves off its initialization.

    Threads update the factors without locking, as the reference does, so a fit on more
    than one thread is reproducible only up to the updates that races drop; see
    ``n_jobs``.

    Parameters
    ----------
    n_factors : int, default=64
        Dimensionality of the latent factors.
    learning_rate : float, default=0.05
        Step size of the gradient ascent.
    regularization : float, default=0.01
        L2 penalty on the factors of the three entities a triplet touches, and on the
        item biases.
    max_iter : int, default=100
        Number of epochs. Each draws one triplet per interaction.
    use_bias : bool, default=True
        Fit a per-item bias, as the reference does by default. It lets the model express
        item popularity without spending a factor on it.
    random_state : int, RandomState instance or None, default=None
        Seed of the factor initialization and of the triplet sampling.
    n_jobs : int or None, default=None
        Number of threads used to fit. ``None`` runs on one thread when
        ``random_state`` is set, so that the fit is reproducible, and on all cores
        otherwise; this is what the reference does. ``-1`` always uses all cores. Which
        triplets are drawn never depends on the thread count, but concurrent updates to
        the same user or item can overwrite each other, so only a single-threaded fit
        reproduces exactly.

    Attributes
    ----------
    user_factors_ : ndarray of shape (n_users_, n_factors)
    item_factors_ : ndarray of shape (n_items_, n_factors)
    item_bias_ : ndarray of shape (n_items_,)
        Zero throughout when ``use_bias`` is False.
    auc_curve_ : ndarray of shape (max_iter,)
        Share of the triplets of each epoch that the model already ranked the right way
        round, the reference's ``correct`` statistic. It estimates the training AUC, but
        each triplet is judged against the parameters as they stand when it is drawn, so
        the curve lags the model an epoch reports.

    References
    ----------
    .. [1] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme, "BPR:
       Bayesian Personalized Ranking from Implicit Feedback", UAI 2009.
       https://dl.acm.org/doi/10.5555/1795114.1795167
    .. [2] A. Salah, Q.-T. Truong, and H. W. Lauw, "Cornac: A Comparative Framework for
       Multimodal Recommender Systems", JMLR 2020.
       https://github.com/PreferredAI/cornac

    Examples
    --------
    >>> from skrecsys.recommendation import BayesianPersonalizedRanking
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "b"], ["u3", "a"]]
    >>> rec = BayesianPersonalizedRanking(n_factors=2, random_state=0).fit(X)
    >>> rec.recommend(["u3"], n_recommendations=1)[0].tolist()
    [['b']]
    """

    def __init__(
        self,
        n_factors: int = 64,
        learning_rate: float = 0.05,
        regularization: float = 0.01,
        max_iter: int = 100,
        *,
        use_bias: bool = True,
        random_state: int | np.random.RandomState | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_iter = max_iter
        self.use_bias = use_bias
        self.random_state = random_state
        self.n_jobs = n_jobs

    @override
    def _fit(self, interactions: sp.csr_array) -> None:
        n_threads = self._check_params()
        n_users, n_items = interactions.shape
        rng = check_random_state(self.random_state)
        # The reference's initialization: small uniform factors, zero biases.
        self.user_factors_ = self._initial(rng, n_users)
        self.item_factors_ = self._initial(rng, n_items)
        self.item_bias_ = np.zeros(n_items)
        # The kernel looks each user's items up by binary search, and ignores the values.
        structure = (
            interactions if interactions.has_sorted_indices else interactions.sorted_indices()
        )
        self.auc_curve_ = np.asarray(
            _core.bpr_fit(
                kernel_indices(structure.indptr),
                kernel_indices(structure.indices),
                n_items,
                self.user_factors_,
                self.item_factors_,
                self.item_bias_,
                float(self.learning_rate),
                float(self.regularization),
                bool(self.use_bias),
                int(self.max_iter),
                int(rng.randint(2**31)),
                n_threads,
            )
        )

    @override
    def _score_users(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        items = self.item_factors_[item_indices]
        return self.item_bias_[None, item_indices] + self.user_factors_[user_indices] @ items.T

    @override
    def _score_pairs(
        self, user_indices: NDArray[np.intp], item_indices: NDArray[np.intp]
    ) -> NDArray[np.floating]:
        factors = np.einsum(
            "ij,ij->i", self.user_factors_[user_indices], self.item_factors_[item_indices]
        )
        return self.item_bias_[item_indices] + factors

    def _initial(self, rng: np.random.RandomState, n_rows: int) -> NDArray[np.float64]:
        """Factors drawn as the reference draws them, ``(uniform - 0.5) / n_factors``."""
        draws = rng.uniform(size=(n_rows, self.n_factors))
        return np.ascontiguousarray((draws - 0.5) / max(self.n_factors, 1))

    def _check_params(self) -> int:
        """Validate parameters and return the thread count for the kernel (0 = all)."""
        for name in ("n_factors", "max_iter"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Integral) or value < 0:
                raise ValueError(f"{name} must be an integer >= 0, got {value!r}.")
        for name in ("learning_rate", "regularization"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Real) or not value >= 0:
                raise ValueError(f"{name} must be a real number >= 0, got {value!r}.")
        if not isinstance(self.use_bias, bool):
            raise ValueError(f"use_bias must be a bool, got {self.use_bias!r}.")
        if self.n_jobs == -1:
            return 0
        if self.n_jobs is None:
            # Racing updates would make a seeded fit irreproducible; the reference makes
            # the same trade.
            return 1 if self.random_state is not None else 0
        if not isinstance(self.n_jobs, numbers.Integral) or self.n_jobs < 1:
            raise ValueError(f"n_jobs must be None, -1 or an integer >= 1, got {self.n_jobs!r}.")
        return int(self.n_jobs)
