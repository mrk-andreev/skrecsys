"""Incremental fitting: ``partial_fit`` and the vocabulary growth behind it.

An estimator that mixes in :class:`IncrementalRecommenderMixin` accepts interactions in
batches. Each batch may name users and items the model has never seen, and the model is
brought up to date from the batch rather than refitted from the accumulated history --
which is the whole point, and which every ``_partial_fit`` below is held to.
"""

from typing import Any, ClassVar, Self

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike, NDArray
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_feature_names, check_is_fitted

from skrecsys import _core
from skrecsys.base import supports_partial_fit
from skrecsys.recommendation._base import kernel_data, kernel_indices
from skrecsys.utils.validation import check_interactions, factorize

__all__ = [
    "IncrementalRecommenderMixin",
    "affected_item_rows",
    "entry_offsets",
    "grow_vocabulary",
    "remap_dense",
    "remap_dense_square",
    "remap_sparse",
    "replace_rows",
    "supports_partial_fit",
]

#: What the entries of ``_incremental_state_`` may name as the meaning of an array's
#: axes. ``"user"`` and ``"item"`` index axis 0 by that code space and leave any
#: trailing axes alone, which covers a factor matrix as well as a vector.
_AXIS_KINDS = ("user", "item", "item_item", "item_item_dense")


def grow_vocabulary(
    fitted_ids: NDArray[Any], values: NDArray[Any], *, name: str
) -> tuple[NDArray[Any], NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Merge ``values`` into the sorted identifier array ``fitted_ids``.

    ``encode_ids`` finds a code by binary search, so the identifiers must stay sorted --
    and therefore an identifier that does not sort last renumbers the codes that were
    handed out before it. The permutation returned here is what every fitted array is
    relabelled through, once per batch, rather than being applied to every query for the
    rest of the model's life.

    Returns
    -------
    merged : ndarray
        The sorted union of both identifier sets.
    perm : ndarray of intp of shape (len(fitted_ids),)
        ``perm[old_code] == new_code``. The identity when every new identifier sorts
        after every fitted one, which is what a stream of monotone keys produces and
        what lets ``partial_fit`` skip the relabelling entirely.
    codes : ndarray of intp of shape (len(values),)
        Code of each value in the merged space.
    new_indices : ndarray of intp
        Codes in the merged space that no fitted identifier maps to.
    """
    if fitted_ids.dtype.kind != values.dtype.kind and "O" not in (
        fitted_ids.dtype.kind,
        values.dtype.kind,
    ):
        raise ValueError(
            f"Cannot add {name} identifiers of dtype {values.dtype} to a model fitted "
            f"with {fitted_ids.dtype}: the two would have to be merged into a common "
            "dtype, which does not preserve the fitted identifiers."
        )
    # One pass over the fitted identifiers and the batch's distinct ones yields the
    # permutation, the batch's codes and the new-identifier mask together; `factorize`
    # already sorts, and already has a native path for integers and a hashed one for
    # objects, so nothing here re-sorts what it returns.
    batch_ids, batch_codes = factorize(values)
    try:
        merged, joint = factorize(np.concatenate([fitted_ids, batch_ids]))
    except TypeError as exc:
        raise ValueError(
            f"Cannot add these {name} identifiers to the fitted ones: the two sets "
            f"cannot be ordered against each other, so they have no common sorted "
            f"vocabulary."
        ) from exc
    perm = np.asarray(joint[: len(fitted_ids)], dtype=np.intp)
    codes = np.asarray(joint[len(fitted_ids) :], dtype=np.intp)[batch_codes]
    is_new = np.ones(len(merged), dtype=bool)
    is_new[perm] = False
    return merged, perm, codes, np.flatnonzero(is_new)


def is_identity(perm: NDArray[np.intp]) -> bool:
    """Whether ``perm`` leaves every code where it was."""
    return bool(np.array_equal(perm, np.arange(len(perm))))


def remap_dense(values: NDArray[Any], perm: NDArray[np.intp], n_new: int) -> NDArray[Any]:
    """Relabel axis 0 of ``values`` through ``perm``, zero-filling the new rows."""
    out = np.zeros((n_new, *values.shape[1:]), dtype=values.dtype)
    out[perm] = values
    return out


def remap_dense_square(values: NDArray[Any], perm: NDArray[np.intp], n_new: int) -> NDArray[Any]:
    """Relabel both axes of a square item-item matrix, zero-filling the new band."""
    out = np.zeros((n_new, n_new), dtype=values.dtype)
    out[np.ix_(perm, perm)] = values
    return out


def remap_sparse(
    matrix: sp.csr_array,
    row_perm: NDArray[np.intp],
    col_perm: NDArray[np.intp],
    shape: tuple[int, int],
) -> sp.csr_array:
    """Relabel both axes of a sparse matrix, growing it to ``shape``.

    Routed through COO because relabelling columns reorders them within a row, so the
    result has to be rebuilt in canonical form rather than patched in place.
    """
    coo = matrix.tocoo()
    grown = sp.coo_array((coo.data, (row_perm[coo.row], col_perm[coo.col])), shape=shape)
    out = sp.csr_array(grown)
    out.sort_indices()
    return out


def entry_offsets(matrix: sp.csr_array, subset: sp.csr_array) -> NDArray[np.int64]:
    """Where each stored entry of ``subset`` sits in ``matrix.indices``.

    ``subset`` must be a sub-pattern of ``matrix``, which is what a batch is once it has
    been added to the accumulated interactions. Both have sorted column indices, so the
    entries of each are globally ordered by ``row * n_cols + column`` and one
    ``searchsorted`` locates all of them at once. That product is why both dimensions
    are cast to int64 first: it is bounded by the number of identifiers that fit in
    memory squared, which is far inside the range, but not inside int32's.
    """
    n_cols = np.int64(matrix.shape[1])
    keys = np.repeat(np.arange(matrix.shape[0], dtype=np.int64), np.diff(matrix.indptr))
    keys *= n_cols
    keys += matrix.indices
    wanted = np.repeat(np.arange(subset.shape[0], dtype=np.int64), np.diff(subset.indptr))
    wanted *= n_cols
    wanted += subset.indices
    return np.searchsorted(keys, wanted).astype(np.int64)


def affected_item_rows(
    interactions: sp.csr_array, touched_items: NDArray[np.intp]
) -> NDArray[np.intp]:
    """Items whose similarity row a batch touching ``touched_items`` can have changed.

    An item-item score is built from the co-occurrence of the pair and from per-item
    statistics of the two columns. A batch moves the column of a touched item, so it can
    move the entry ``(i, j)`` only when one of ``i`` and ``j`` is touched *and* the two
    co-occur -- a pair that never co-occurs scores zero and is not stored at all. Both
    cases put ``i`` within two hops of the batch: one hop to a user of a touched item,
    one more to the items that user also took.

    Every other row is therefore unchanged entry for entry, which is what makes an
    incremental neighbourhood fit exact rather than merely close: nothing is patched, and
    no pruned entry has to be recovered.

    The two hops are taken against the *accumulated* matrix, not the batch: a user who
    interacted with a touched item long ago co-occurs with it just as much as one the
    batch names.
    """
    columns = sp.csc_array(interactions)[:, touched_items]
    users = np.unique(columns.indices)
    reachable = np.unique(sp.csr_array(interactions[users]).indices)
    return np.union1d(touched_items, reachable).astype(np.intp)


def _row_positions(starts: NDArray[Any], lengths: NDArray[Any]) -> NDArray[np.int64]:
    """Where each stored entry of a run of rows lands, given each row's start."""
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return np.repeat(np.asarray(starts, dtype=np.int64) - offsets, lengths) + np.arange(total)


def replace_rows(matrix: sp.csr_array, rows: NDArray[np.intp], block: sp.csr_array) -> sp.csr_array:
    """``matrix`` with row ``rows[t]`` replaced by row ``t`` of ``block``.

    ``rows`` must be sorted and distinct, and ``block`` must have one row per entry of
    it -- which is exactly what the row-restricted kernels return. Built by scattering
    rather than by stacking, so the cost is the stored entries and not the row count.
    """
    counts = np.diff(matrix.indptr).astype(np.int64)
    counts[rows] = np.diff(block.indptr)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    indices = np.empty(int(indptr[-1]), dtype=np.int64)
    data = np.empty(int(indptr[-1]), dtype=np.float64)

    keep = np.ones(matrix.shape[0], dtype=bool)
    keep[rows] = False
    kept = np.flatnonzero(keep)
    lengths = np.diff(matrix.indptr)[kept]
    source = _row_positions(matrix.indptr[kept], lengths)
    target = _row_positions(indptr[kept], lengths)
    indices[target] = matrix.indices[source]
    data[target] = matrix.data[source]

    target = _row_positions(indptr[rows], np.diff(block.indptr))
    indices[target] = block.indices
    data[target] = block.data
    return sp.csr_array((data, indices, indptr), shape=matrix.shape)


def _csr_from_coo(
    rows: NDArray[Any],
    cols: NDArray[Any],
    data: NDArray[Any],
    n_rows: int,
    n_cols: int,
    n_threads: int,
) -> sp.csr_array:
    """The canonical CSR of the given entries, duplicates summed.

    The same kernel ``fit`` builds ``interactions_`` with, so a batch accumulated here
    and a single ``fit`` over the concatenation agree entry for entry.
    """
    indptr, indices, values = _core.coo_to_csr(
        kernel_indices(rows), kernel_indices(cols), kernel_data(data), n_rows, n_cols, n_threads
    )
    return sp.csr_array((values, indices, indptr), shape=(n_rows, n_cols))


def _reshaped(matrix: sp.csr_array, n_rows: int, n_cols: int) -> sp.csr_array:
    """``matrix`` widened to ``(n_rows, n_cols)`` without moving any stored entry.

    Only valid when the codes did not move: the column indices are already right and the
    new rows are empty, so the row pointer is extended with its own last value.
    """
    indptr = np.concatenate(
        [matrix.indptr, np.full(n_rows + 1 - len(matrix.indptr), matrix.indptr[-1])]
    )
    return sp.csr_array((matrix.data, matrix.indices, indptr), shape=(n_rows, n_cols))


class IncrementalRecommenderMixin:
    """Mixin giving a recommender ``partial_fit``.

    Subclasses declare which fitted arrays live in which code space through
    :attr:`_incremental_state_`, which is all the default :meth:`_remap` needs, and
    implement :meth:`_partial_fit` to bring the model up to date from one batch.

    The deliberate difference from scikit-learn's ``SGDClassifier``, whose
    ``partial_fit`` rejects a batch naming a class the first call did not: here a batch
    may name users and items that were never seen, and the fitted vocabularies grow to
    admit them. Growing them is the point of the method.
    """

    #: ``(attribute, axis kind)`` pairs the default ``_remap`` relabels, where the kind
    #: is one of ``_AXIS_KINDS``. Empty means the estimator relabels nothing, which is
    #: true only of a model whose every parameter is rebuilt by ``_partial_fit``.
    _incremental_state_: ClassVar[tuple[tuple[str, str], ...]] = ()

    #: Fitted by ``BaseRecommender``, which every estimator mixing this in derives from.
    #: Annotations only: a class attribute would be a value the estimator does not have
    #: until it is fitted, which is what ``check_is_fitted`` reads.
    user_ids_: NDArray[Any]
    item_ids_: NDArray[Any]
    n_users_: int
    n_items_: int
    interactions_: sp.csr_array

    def partial_fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Update the recommender from one batch of user-item interactions.

        The first call is equivalent to ``fit``. Every later call adds the batch to what
        the model has already seen: identifiers that are new extend ``user_ids_`` and
        ``item_ids_``, and a user-item pair that was already stored has the batch's
        weight *added* to it, which is the rule ``fit`` applies to a pair repeated
        within one array.

        Parameters
        ----------
        X : array-like of shape (n_interactions, 2)
            ``X[:, 0]`` contains user identifiers, ``X[:, 1]`` item identifiers. Unlike
            ``predict`` and ``recommend``, neither has to have been seen before.

        y : array-like of shape (n_interactions,), default=None
            Rating, relevance, interaction weight, or confidence. If None, every
            observed interaction has weight 1.

        Returns
        -------
        self : object

        Notes
        -----
        What this saves is the *model* fit, not every pass over the data: accumulating
        ``interactions_`` costs one pass over everything stored so far on any call that
        renumbers a code. How much of the model itself is recomputed, and whether the
        result is what ``fit`` on the concatenation of every batch would have produced,
        is documented on each estimator.

        An ``index`` is rebuilt from scratch at the end of every call, so on a model
        whose own update is cheap the graph build dominates.
        """
        if _is_unfitted(self):
            return self.fit(X, y)  # ty: ignore[unresolved-attribute, unsound-return-statement]

        # Each estimator validates in `_fit`, which this path does not reach.
        check_params = getattr(self, "_check_params", None)
        if check_params is not None:
            check_params()

        users, items, weights = check_interactions(X, y)
        _check_feature_names(self, X, reset=False)
        stored = self.interactions_

        self.user_ids_, user_perm, user_codes, new_users = grow_vocabulary(
            self.user_ids_, users, name="user"
        )
        self.item_ids_, item_perm, item_codes, new_items = grow_vocabulary(
            self.item_ids_, items, name="item"
        )
        self.n_users_ = n_users = len(self.user_ids_)
        self.n_items_ = n_items = len(self.item_ids_)

        n_threads = self._build_threads()  # ty: ignore[unresolved-attribute]
        delta = _csr_from_coo(user_codes, item_codes, weights, n_users, n_items, n_threads)

        relabelled = not (is_identity(user_perm) and is_identity(item_perm))
        if relabelled or (n_users, n_items) != stored.shape:
            self._remap(user_perm=user_perm, item_perm=item_perm, n_users=n_users, n_items=n_items)
        if relabelled:
            # A moved column lands out of order within its row, so the whole matrix is
            # rebuilt through the kernel that already sorts and sums.
            rows = user_perm[np.repeat(np.arange(stored.shape[0]), np.diff(stored.indptr))]
            self.interactions_ = _csr_from_coo(
                np.concatenate([rows, np.repeat(np.arange(n_users), np.diff(delta.indptr))]),
                np.concatenate([item_perm[stored.indices], delta.indices]),
                np.concatenate([stored.data, delta.data]),
                n_users,
                n_items,
                n_threads,
            )
        else:
            self.interactions_ = _reshaped(stored, n_users, n_items) + delta

        self._partial_fit(
            self.interactions_,
            delta=delta,
            new_user_indices=new_users,
            new_item_indices=new_items,
            touched_user_indices=np.flatnonzero(np.diff(delta.indptr)).astype(np.intp),
            touched_item_indices=np.unique(delta.indices).astype(np.intp),
        )
        self._fit_index()  # ty: ignore[unresolved-attribute]
        self.n_batches_seen_ = getattr(self, "n_batches_seen_", 1) + 1
        return self

    def _remap(
        self,
        *,
        user_perm: NDArray[np.intp],
        item_perm: NDArray[np.intp],
        n_users: int,
        n_items: int,
    ) -> None:
        """Relabel every fitted array into the grown coordinate space.

        New rows and columns are left at zero; initializing them is ``_partial_fit``'s
        business, because only it knows whether a new factor row wants a draw or a
        count wants nothing. Arrays are rebound rather than mutated, which is what
        invalidates the transpose memoized in ``SimilarityRecommender``.
        """
        for name, kind in self._incremental_state_:
            value = getattr(self, name)
            if kind == "user":
                grown: Any = remap_dense(value, user_perm, n_users)
            elif kind == "item":
                grown = remap_dense(value, item_perm, n_items)
            elif kind == "item_item":
                grown = remap_sparse(value, item_perm, item_perm, (n_items, n_items))
            elif kind == "item_item_dense":
                grown = remap_dense_square(value, item_perm, n_items)
            else:
                raise ValueError(
                    f"{type(self).__name__}._incremental_state_ names axis kind {kind!r} "
                    f"for {name!r}; it must be one of {_AXIS_KINDS}."
                )
            setattr(self, name, grown)

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
        """Bring the model up to date from one batch.

        Parameters
        ----------
        interactions : scipy.sparse.csr_array of shape (n_users_, n_items_)
            Everything seen so far, the batch included, in the grown coordinate space.

        delta : scipy.sparse.csr_array of shape (n_users_, n_items_)
            This batch alone, in the same space.

        new_user_indices, new_item_indices : ndarray of intp
            Codes that did not exist before this call. Their rows of every relabelled
            array are zero, and this is where they get their initial values.

        touched_user_indices, touched_item_indices : ndarray of intp
            Sorted codes with at least one entry in ``delta``. The new codes are a
            subset of these.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement incremental fitting.")

    def _incremental_rng(self) -> np.random.RandomState:
        """The draw stream shared by every call, so batches do not repeat each other.

        Rebuilding it from ``random_state`` per call would hand every batch the same
        numbers, which for a seeded estimator means every new user starting from the
        same factors as the last one.
        """
        rng = getattr(self, "_rng", None)
        if rng is None:
            rng = check_random_state(getattr(self, "random_state", None))
            self._rng = rng
        # `check_random_state` is untyped, so what it hands back is only known by name.
        return rng  # ty: ignore[unsound-return-statement]


def _is_unfitted(estimator: object) -> bool:
    """Whether nothing has been fitted yet, so ``partial_fit`` must behave as ``fit``."""
    try:
        check_is_fitted(estimator)
    except NotFittedError:
        return True
    return False
