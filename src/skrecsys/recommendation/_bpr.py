"""BPR matrix factorization, ported from Cornac's ``BPR``."""

import numbers
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from skrecsys import _core
from skrecsys._typing import override
from skrecsys.indexing import DenseSpace
from skrecsys.recommendation._base import BaseRecommender, kernel_indices
from skrecsys.recommendation._incremental import IncrementalRecommenderMixin, entry_offsets


class BayesianPersonalizedRanking(IncrementalRecommenderMixin, BaseRecommender):
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
    index : None, str or VectorIndex, default=None
        Approximate index used by ``recommend``. ``None`` scores every candidate
        exactly; ``"hnsw"`` or a configured :class:`~skrecsys.indexing.HNSW` walks a
        graph instead, which is faster on a large catalog and no longer exact. See
        :mod:`skrecsys.indexing`.
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

    Notes
    -----
    ``partial_fit`` resumes the descent, drawing every positive from the batch while
    still rejecting negatives against the user's whole history, so an epoch costs the
    batch rather than the accumulated matrix. A user the batch does not mention is not
    updated, and stochastic gradient ascent resumed from where it stopped does not
    reproduce ``fit`` on every batch concatenated.

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
        max_iter_partial: int | None = None,
        *,
        use_bias: bool = True,
        random_state: int | np.random.RandomState | None = None,
        n_jobs: int | None = None,
        index: Any = None,
    ) -> None:
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.max_iter = max_iter
        self.max_iter_partial = max_iter_partial
        self.use_bias = use_bias
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.index = index

    _incremental_state_ = (
        ("user_factors_", "user"),
        ("item_factors_", "item"),
        ("item_bias_", "item"),
    )

    @override
    def _fit(self, interactions: sp.csr_array) -> None:
        n_threads = self._check_params()
        n_users, n_items = interactions.shape
        rng = check_random_state(self.random_state)
        self._rng = rng
        # The reference's initialization: small uniform factors, zero biases.
        self.user_factors_ = self._initial(rng, n_users)
        self.item_factors_ = self._initial(rng, n_items)
        self.item_bias_ = np.zeros(n_items)
        self.auc_curve_ = self._epochs(interactions, None, int(self.max_iter), rng, n_threads)

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
        """Resume the descent, drawing every positive from the batch.

        The epochs run over the batch's interactions alone -- one epoch draws as many
        triplets as the batch has entries, not as the history has -- while the negative
        is still rejected against everything the user has ever interacted with, so the
        model is never told to rank an item below one it already knows the user took.
        That is the whole of the kernel's ``positives`` argument.

        Two consequences to be clear about. A user the batch does not mention is not
        updated, so the model drifts toward recent behaviour; and because this is
        stochastic gradient ascent resumed from where it stopped, the result is not what
        ``fit`` on the concatenation of every batch would have produced. Replaying the
        whole matrix on every call would fix both and would be a refit.
        """
        del touched_user_indices, touched_item_indices
        n_threads = self._check_params()
        rng = self._incremental_rng()
        # `_remap` left the new rows at zero. Zero is the reference's initialization for
        # a bias but a fixed point for a factor row, so only the factors are drawn.
        if len(new_user_indices):
            self.user_factors_[new_user_indices] = self._initial(rng, len(new_user_indices))
        if len(new_item_indices):
            self.item_factors_[new_item_indices] = self._initial(rng, len(new_item_indices))
        curve = self._epochs(interactions, delta, self._partial_epochs(), rng, n_threads)
        self.auc_curve_ = np.concatenate([self.auc_curve_, curve])

    def _partial_epochs(self) -> int:
        """Epochs a ``partial_fit`` runs."""
        if self.max_iter_partial is None:
            return int(self.max_iter)
        if not isinstance(self.max_iter_partial, numbers.Integral) or self.max_iter_partial < 0:
            raise ValueError(
                f"max_iter_partial must be None or an integer >= 0, got {self.max_iter_partial!r}."
            )
        return int(self.max_iter_partial)

    def _epochs(
        self,
        interactions: sp.csr_array,
        positives: sp.csr_array | None,
        n_iter: int,
        rng: np.random.RandomState,
        n_threads: int,
    ) -> NDArray[np.float64]:
        """Run ``n_iter`` epochs in place, drawing positives from ``positives`` if given."""
        # The kernel looks each user's items up by binary search, and ignores the values.
        structure = (
            interactions if interactions.has_sorted_indices else interactions.sorted_indices()
        )
        drawn = None if positives is None else kernel_indices(entry_offsets(structure, positives))
        return np.asarray(
            _core.bpr_fit(
                kernel_indices(structure.indptr),
                kernel_indices(structure.indices),
                interactions.shape[1],
                self.user_factors_,
                self.item_factors_,
                self.item_bias_,
                float(self.learning_rate),
                float(self.regularization),
                bool(self.use_bias),
                int(n_iter),
                int(rng.randint(2**31)),
                n_threads,
                drawn,
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
        return self._rank_by_factors(
            queries,
            item_indices,
            k,
            exclude_seen=exclude_seen,
            excluded=excluded,
            first_query=first_query,
            user_factors=self.user_factors_,
            item_factors=self.item_factors_,
            item_bias=self.item_bias_,
        )

    @override
    def _index_space(self) -> DenseSpace:
        # `score(u, j) = b_j + <p_u, q_j>`: one extra dimension carries the item bias,
        # and there is no user bias to add back.
        items = np.empty((self.n_items_, self.n_factors + 1), dtype=np.float64)
        items[:, : self.n_factors] = self.item_factors_
        items[:, self.n_factors] = self.item_bias_
        return DenseSpace(items)

    @override
    def _index_queries(self, user_indices: NDArray[np.intp]) -> NDArray[np.float64]:
        queries = np.empty((len(user_indices), self.n_factors + 1), dtype=np.float64)
        queries[:, : self.n_factors] = self.user_factors_[user_indices]
        queries[:, self.n_factors] = 1.0
        return queries

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
