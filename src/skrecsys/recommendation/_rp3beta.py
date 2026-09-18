"""RP3beta: random-walk item-item similarity, ported from Ferrari Dacrema's framework."""

import math
import numbers

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from skrecsys import _core
from skrecsys._typing import override
from skrecsys.recommendation._base import (
    SimilarityRecommender,
    keep_top_k_per_row,
    kernel_data,
    kernel_indices,
)


class RP3Beta(SimilarityRecommender):
    """Item-item recommender from a popularity-damped random walk on the user-item graph.

    A three-step walk ``item -> user -> item`` gives the transition probabilities
    ``W = Piu Pui``, where ``Pui`` is the interaction matrix with rows normalized to
    sum to one and ``Piu`` the same for the binary transpose, both raised to the power
    ``alpha``. RP3beta [1]_ then damps popular destinations by dividing column ``j`` of
    ``W`` by ``popularity[j] ** beta``, which is what separates it from P3alpha; ``beta
    = 0`` recovers P3alpha [2]_. Each row is pruned to its ``n_neighbors`` largest
    entries, the diagonal is dropped, and the score of item ``j`` for user ``u`` is
    ``sum_i r_ui * s(i, j)`` over the items ``i`` of the user.

    This is a port of ``RP3betaRecommender`` from the evaluation framework of Ferrari
    Dacrema et al. [3]_, including its second, column-wise pruning pass after the rows
    are normalized. Two parameters of the reference are not exposed: ``min_rating``,
    which is left to the caller, and ``implicit``, which the reference only applies
    when ``min_rating > 0`` and which ``fit(X)`` without ``y`` already achieves. The
    reference computes in float32 and breaks ties by an unstable sort, so its weights
    agree with ours only to float32 precision and may differ on exactly tied entries,
    where we keep the lower item index.

    Parameters
    ----------
    n_neighbors : int, default=100
        Number of largest entries kept per row, the reference's ``topK``.
    alpha : float, default=1.0
        Power applied to both transition matrices. Values below 1 flatten the walk,
        spreading probability towards less obvious items.
    beta : float, default=0.6
        Popularity damping of the destination item. 0 disables it, larger values push
        recommendations further into the long tail.
    normalize_similarity : bool, default=True
        Normalize each row of the similarity matrix to sum to one before the
        column-wise pruning pass.
    n_jobs : int or None, default=None
        Number of threads for computing similarities. ``None`` or ``-1`` uses all cores.

    Attributes
    ----------
    similarity_ : scipy.sparse.csr_array of shape (n_items_, n_items_)
        Pruned item-item transition weights; row ``i`` holds the neighbours of item
        ``i``. The diagonal is zero.

    References
    ----------
    .. [1] F. Christoffel et al., "Blockbusters and Wallflowers: Accurate, Diverse and
       Scalable Recommendations with Random Walks", RecSys 2015.
       https://doi.org/10.1145/2792838.2800180
    .. [2] C. Cooper et al., "Random Walks in Recommender Systems: Exact Computation and
       Simulations", WWW 2014. https://doi.org/10.1145/2567948.2579244
    .. [3] M. Ferrari Dacrema et al., "Are We Really Making Much Progress? A Worrying
       Analysis of Recent Neural Recommendation Approaches", RecSys 2019.
       https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation
    .. [4] V. W. Anelli et al., "Challenging the Myth of Graph Collaborative Filtering:
       a Reasoned and Reproducibility-driven Analysis", RecSys 2023, which tunes RP3beta
       as a classic baseline against graph neural recommenders.
       https://github.com/sisinflab/Graph-RSs-Reproducibility

    Examples
    --------
    >>> from skrecsys.recommendation import RP3Beta
    >>> X = [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "b"], ["u3", "a"]]
    >>> rec = RP3Beta().fit(X)
    >>> rec.recommend(["u3"], n_recommendations=1)[0].tolist()
    [['b']]
    """

    def __init__(
        self,
        n_neighbors: int = 100,
        alpha: float = 1.0,
        beta: float = 0.6,
        *,
        normalize_similarity: bool = True,
        n_jobs: int | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.beta = beta
        self.normalize_similarity = normalize_similarity
        self.n_jobs = n_jobs

    @override
    def _fit(self, interactions: sp.csr_array) -> None:
        n_threads = self._check_params()
        n_users, n_items = interactions.shape
        observed = sp.csr_array(interactions, copy=True)
        observed.eliminate_zeros()
        rows = np.repeat(np.arange(n_users), np.diff(observed.indptr))

        # Pui, the user-to-item step: each row normalized to sum to one, then ``alpha``.
        user_sums = np.bincount(rows, weights=np.abs(observed.data), minlength=n_users)
        pui = observed.data * _reciprocal(user_sums)[rows]
        if self.alpha != 1.0:
            pui = pui**self.alpha

        # Every entry of row i of the item-to-user step is popularity[i] ** -alpha, so
        # the walk needs only the two scaling vectors below alongside Pui.
        popularity = np.bincount(observed.indices, minlength=n_items).astype(np.float64)
        seen = popularity > 0
        row_scale, col_scale = np.zeros(n_items), np.zeros(n_items)
        row_scale[seen] = popularity[seen] ** -self.alpha
        col_scale[seen] = popularity[seen] ** -self.beta

        indptr, indices, data = _core.rp3beta_similarity(
            kernel_indices(observed.indptr),
            kernel_indices(observed.indices),
            kernel_data(pui),
            n_items,
            row_scale,
            col_scale,
            int(self.n_neighbors),
            n_threads,
        )
        similarity = sp.csr_array(
            (data, indices, indptr), shape=(n_items, n_items), dtype=np.float64
        )

        if self.normalize_similarity:
            row_sums = np.bincount(
                np.repeat(np.arange(n_items), np.diff(similarity.indptr)),
                weights=np.abs(similarity.data),
                minlength=n_items,
            )
            similarity.data *= np.repeat(_reciprocal(row_sums), np.diff(similarity.indptr))

        # The reference prunes a second time after normalizing, now column-wise.
        pruned = keep_top_k_per_row(sp.csr_array(similarity.T), int(self.n_neighbors), n_threads)
        self.similarity_ = sp.csr_array(pruned.T)

    @override
    def _check_params(self) -> int:
        """Validate parameters and return the thread count for the kernel (0 = all)."""
        if not isinstance(self.n_neighbors, numbers.Integral) or self.n_neighbors < 1:
            raise ValueError(f"n_neighbors must be an integer >= 1, got {self.n_neighbors!r}.")
        for name in ("alpha", "beta"):
            value = getattr(self, name)
            if not isinstance(value, numbers.Real) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be a finite real number >= 0, got {value!r}.")
        if not isinstance(self.normalize_similarity, bool):
            raise ValueError(
                f"normalize_similarity must be a bool, got {self.normalize_similarity!r}."
            )
        if self.n_jobs is None or self.n_jobs == -1:
            return 0
        if not isinstance(self.n_jobs, numbers.Integral) or self.n_jobs < 1:
            raise ValueError(f"n_jobs must be None, -1 or an integer >= 1, got {self.n_jobs!r}.")
        return int(self.n_jobs)


def _reciprocal(values: NDArray[np.floating]) -> NDArray[np.float64]:
    """``1 / values``, leaving zeros in place of the entries that would divide by zero."""
    # An empty sum comes back as an integer array, which cannot hold the reciprocal.
    values = np.asarray(values, dtype=np.float64)
    return np.divide(1.0, values, out=np.zeros_like(values), where=values > 0)
