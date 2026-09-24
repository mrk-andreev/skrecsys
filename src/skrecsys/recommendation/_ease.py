"""Embarrassingly shallow autoencoder: a closed-form linear item-item model."""

import math
import numbers
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from skrecsys import _core
from skrecsys._typing import override
from skrecsys.indexing import DenseSpace
from skrecsys.recommendation._base import (
    BaseRecommender,
    kernel_candidates,
    kernel_csr,
    kernel_excluded,
    kernel_indices,
    kernel_matrix,
    score_pairs_from_similarity,
)
from skrecsys.recommendation._incremental import IncrementalRecommenderMixin
from skrecsys.utils.validation import check_ids, encode_ids


class EASE(IncrementalRecommenderMixin, BaseRecommender):
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
    index : None, str or VectorIndex, default=None
        Approximate index used by ``recommend``. ``None`` scores every candidate
        exactly; ``"hnsw"`` or a configured :class:`~skrecsys.indexing.HNSW` walks a
        graph instead. EASE's weight matrix is dense, so indexing it copies a matrix
        that is already quadratic in the catalog. See :mod:`skrecsys.indexing`.

    Attributes
    ----------
    similarity_ : ndarray of shape (n_items_, n_items_)
        Item-item weights with a zero diagonal; entry ``(i, j)`` is the contribution of
        item ``i`` to the score of item ``j``. Dense, unlike the sparse ``similarity_``
        of the neighbourhood models.
    inverse_gram_ : ndarray of shape (n_items_, n_items_)
        ``(G + l2_reg * I)^-1``, the matrix the weights are read off. Present only once
        ``partial_fit`` has been used, because it doubles the model's memory and only an
        incremental fit has anything to do with it.

    Notes
    -----
    ``partial_fit`` is exact: the weights are what ``fit`` on every batch concatenated
    would have produced, up to floating-point rounding. It keeps ``inverse_gram_`` and
    updates it in place, which costs ``O(n_items ** 2 * n_touched_users)`` against the
    ``O(n_items ** 3)`` of the factorization -- but doubles the memory, which for this
    model is already quadratic in the catalog. When a batch touches so many users that
    the update would cost as much as the factorization, it factorizes instead.

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

    def __init__(
        self,
        l2_reg: float = 500.0,
        n_jobs: int | None = None,
        index: Any = None,
    ) -> None:
        self.l2_reg = l2_reg
        self.n_jobs = n_jobs
        self.index = index

    @override
    def _fit(self, interactions: sp.csr_array) -> None:
        n_threads = self._check_params()
        self.similarity_ = _core.ease_weights(
            *kernel_csr(interactions),
            interactions.shape[1],
            float(self.l2_reg),
            n_threads,
        )
        # A plain `fit` keeps the memory it always did. The first `partial_fit` pays one
        # factorization to recover the inverse, and every call after that updates it.
        self.__dict__.pop("inverse_gram_", None)

    @override
    def _remap(
        self,
        *,
        user_perm: NDArray[np.intp],
        item_perm: NDArray[np.intp],
        n_users: int,
        n_items: int,
    ) -> None:
        """Grow the inverse Gram matrix, which is a closed form rather than a relabelling.

        An item the model has not seen has an all-zero Gram row and column, so bordering
        ``A = G + l2_reg * I`` with it leaves the Schur complement at ``l2_reg`` and the
        block inverse is simply the old one with ``1 / l2_reg`` on the new diagonal. No
        solve, and exact -- the batch's own interactions with that item arrive
        afterwards, as part of the low-rank update in ``_partial_fit``.

        ``similarity_`` is not relabelled at all: it is read back off the inverse at the
        end of every call, so permuting it here would be work thrown away.
        """
        del user_perm, n_users
        p = self._inverse_gram()
        grown = np.zeros((n_items, n_items), dtype=np.float64)
        grown[np.ix_(item_perm, item_perm)] = p
        fresh = np.setdiff1d(np.arange(n_items), item_perm)
        grown[fresh, fresh] = 1.0 / float(self.l2_reg)
        self.inverse_gram_ = grown

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
        """Update the inverse Gram matrix by the rank the batch actually has.

        Only the touched users' rows of ``R`` moved, and the Gram matrix is a sum of
        their outer products, so

            ``G_new = G_old - Z_old.T @ Z_old + Z_new.T @ Z_new``

        is a symmetric update of rank at most twice the number of touched users --
        remove what those users used to contribute, add what they contribute now. With
        ``C = [Z_new.T, Z_old.T]`` and ``S = diag(I, -I)`` that is ``A + C @ S @ C.T``,
        and Sherman-Morrison-Woodbury inverts it without refactorizing.

        The identity is exact, so this reaches the same weights the factorization would,
        and it is applied as one two-sided update rather than two rank-``|T|`` ones
        because the two-sided form keeps the arithmetic symmetric and the error small.
        """
        del new_user_indices, new_item_indices, touched_item_indices
        n_threads = self._check_params()
        n_items = interactions.shape[1]
        rank = 2 * len(touched_user_indices)
        if "inverse_gram_" not in self.__dict__ or rank >= n_items:
            # Either there is no inverse to update -- the model was `fit`, not
            # `partial_fit` -- or the update would cost what the factorization costs.
            self._factorize(interactions, n_threads)
            return

        after = sp.csr_array(interactions[touched_user_indices])
        before = after - sp.csr_array(delta[touched_user_indices])
        columns = sp.csr_array(sp.hstack([after.T, before.T], format="csr"))
        signs = np.concatenate([np.ones(rank // 2), -np.ones(rank // 2)])

        p = self.inverse_gram_
        transposed = np.asarray(columns.T @ p)  # C.T @ P, shape (rank, n_items)
        middle = np.diag(signs) + np.asarray(columns.T @ transposed.T)
        try:
            correction = transposed.T @ np.linalg.solve(middle, transposed)
        except np.linalg.LinAlgError:
            # The identity holds whenever the updated matrix is invertible, which it is
            # for l2_reg > 0; if the small system is too ill-conditioned to say so,
            # the factorization still can.
            self._factorize(interactions, n_threads)
            return
        updated = p - correction
        # The result is symmetric by construction, and saying so costs one pass and
        # keeps a long chain of updates from drifting off the symmetric cone.
        self.inverse_gram_ = 0.5 * (updated + updated.T)
        self.similarity_ = _core.ease_weights_from_inverse(self.inverse_gram_, n_threads)

    def _factorize(self, interactions: sp.csr_array, n_threads: int) -> None:
        """Recover both the inverse and the weights from the interactions."""
        self.inverse_gram_ = self._compute_inverse_gram(interactions, n_threads)
        self.similarity_ = _core.ease_weights_from_inverse(self.inverse_gram_, n_threads)

    def _inverse_gram(self) -> NDArray[np.float64]:
        """``inverse_gram_``, factorizing the stored interactions if it is not there."""
        if "inverse_gram_" not in self.__dict__:
            self.inverse_gram_ = self._compute_inverse_gram(
                self.interactions_, self._check_params()
            )
        return self.inverse_gram_

    def _compute_inverse_gram(
        self, interactions: sp.csr_array, n_threads: int
    ) -> NDArray[np.float64]:
        return _core.ease_inverse_gram(
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
        # Row `i` of `similarity_` is what an interaction with item `i` adds to every
        # score, so a query is one `axpy` per item it holds, straight into the ranking.
        user_idx = encode_ids(check_ids(queries), self.user_ids_, name="user")
        return _core.recommend_from_dense_rows(
            *kernel_csr(self.interactions_),
            kernel_indices(user_idx),
            kernel_matrix(self.similarity_),
            kernel_candidates(item_indices, self.n_items_),
            exclude_seen,
            k,
            self._check_params(),
            first_query,
            **kernel_excluded(excluded),
        )

    @override
    def _index_space(self) -> DenseSpace:
        # `score(u, j) = <interactions[u], similarity_[:, j]>`, so item `j`'s vector is
        # column `j`. The transpose is a copy of an already quadratic matrix, which is
        # what `index MB` in the report is there to make visible.
        return DenseSpace(np.ascontiguousarray(self.similarity_.T, dtype=np.float64))

    @override
    def _index_queries(self, user_indices: NDArray[np.intp]) -> sp.csr_array:
        # Left sparse: the item vectors are catalog-wide, so scattering a query into one
        # would cost a full pass per distance, while gathering over the query's own
        # handful of nonzeros costs what the user's history costs.
        return sp.csr_array(self.interactions_[user_indices])

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
