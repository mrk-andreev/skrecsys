"""Hierarchical navigable small-world graphs, as a vector index."""

from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from skrecsys import _core
from skrecsys._typing import override
from skrecsys.indexing._base import (
    DenseSpace,
    SparseSpace,
    VectorIndex,
    VectorSpace,
    check_positive_int,
    register_index,
)

__all__ = ["HNSW"]


class HNSW(VectorIndex):
    """Approximate nearest neighbours over a hierarchical navigable small-world graph.

    The graph links each item to a spread of its nearest neighbours across a few levels,
    and a search walks it from the top down, so a query touches a few hundred items
    rather than the whole catalog [1]_. The implementation follows Qdrant's: ``m`` links
    above level 0 and ``2 * m`` at level 0, levels drawn geometrically, and neighbours
    chosen by the diversity heuristic rather than by score alone.

    What it costs is exactness. ``recommend`` still returns exactly the number of items
    asked for, still excludes what the query has seen, and still breaks ties by fitted
    item order -- but the items are the best the walk *found*, which is usually and not
    always the best there are. Run ``benchmarks/indexes.py`` to see what that difference
    is worth on a given model and dataset rather than assuming it.

    Two caveats worth knowing before reaching for it. Scores here are inner products,
    which are not a metric, so a graph over vectors whose norms vary widely reaches the
    low-norm ones less often -- though those are also the ones that lose to everything,
    so the recall cost is smaller than the connectivity one suggests. And a build on
    more than one thread is not reproducible, because concurrent insertions see
    different partial graphs; a recommender with ``random_state`` set therefore builds
    its index on one thread unless ``n_jobs`` says otherwise.

    Parameters
    ----------
    m : int, default=16
        Links kept per item above level 0; level 0 keeps ``2 * m``. Larger graphs
        recall more and cost more to build, search and hold.
    ef_construction : int, default=200
        Width of the candidate list an insertion considers. Affects build time and
        graph quality, not search time.
    ef_search : int, default=64
        Width of the candidate list a search keeps. The recall-for-latency dial, and
        the only one that can be turned on a fitted model.
    min_index_size : int, default=4096
        Candidates below which the exact path is used instead. A graph traversed with
        only a fraction of its nodes eligible needs proportionally more walking and
        falls apart when the fraction is small, while an exact scan over a short
        candidate list is cheap -- so below this the index is not worth its own risk.

    Attributes
    ----------
    space_ : DenseSpace or SparseSpace
        The item vectors the graph was built over.
    node_level_ : ndarray of shape (n_items,)
        The highest level each item lives on.
    links_indptr_ : ndarray of shape (n_slots + 1,)
        Start of each ``(item, level)`` slot's neighbours, slots ordered item-major.
    links_indices_ : ndarray
        The neighbours themselves.
    entry_point_ : int
        The item a search starts from, or ``-1`` when the catalog is empty.

    References
    ----------
    .. [1] Y. A. Malkov and D. A. Yashunin, "Efficient and robust approximate nearest
       neighbor search using Hierarchical Navigable Small World graphs", 2018.
       :doi:`10.1109/TPAMI.2018.2889473`
    """

    def __init__(
        self,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        min_index_size: int = 4096,
    ) -> None:
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.min_index_size = min_index_size

    def _check_params(self) -> None:
        """Validate parameters, as the estimators do, rather than by constraint table."""
        for name in ("m", "ef_construction", "ef_search", "min_index_size"):
            check_positive_int(getattr(self, name), name)

    @override
    def fit(self, space: VectorSpace, *, n_threads: int = 0, seed: int = 0) -> "HNSW":
        self._check_params()
        space = self.space_ = kernel_ready(equalize_norms(space))
        if isinstance(space, DenseSpace):
            graph = _core.hnsw_build_dense(
                np.ascontiguousarray(space.items, dtype=np.float64),
                int(self.m),
                int(self.ef_construction),
                int(seed),
                n_threads,
            )
        else:
            items = space.items
            graph = _core.hnsw_build_sparse(
                _as_int64(items.indptr),
                _as_int64(items.indices),
                np.ascontiguousarray(items.data, dtype=np.float64),
                space.dim,
                int(self.m),
                int(self.ef_construction),
                int(seed),
                n_threads,
            )
        (
            self.node_level_,
            self.links_indptr_,
            self.links_indices_,
            self.entry_point_,
        ) = graph
        return self

    @override
    def query(
        self,
        queries: NDArray[np.float64] | sp.csr_array,
        candidates: NDArray[np.intp],
        excluded: sp.csr_array,
        k: int,
        *,
        n_threads: int = 0,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        graph = (self.node_level_, self.links_indptr_, self.links_indices_, self.entry_point_)
        space = self.space_
        candidates = _as_int64(candidates)
        excluded_indptr, excluded_indices = _as_int64(excluded.indptr), _as_int64(excluded.indices)

        if isinstance(space, DenseSpace):
            items = np.ascontiguousarray(space.items, dtype=np.float64)
            if isinstance(queries, sp.csr_array):
                # EASE: catalog-wide dense item vectors, but a query of a few nonzeros.
                return _core.hnsw_search_sparse_dense(
                    *graph,
                    items,
                    _as_int64(queries.indptr),
                    _as_int64(queries.indices),
                    np.ascontiguousarray(queries.data, dtype=np.float64),
                    candidates,
                    excluded_indptr,
                    excluded_indices,
                    k,
                    int(self.ef_search),
                    n_threads,
                )
            return _core.hnsw_search_dense(
                *graph,
                items,
                _pad_queries(np.asarray(queries, dtype=np.float64), space.dim),
                candidates,
                excluded_indptr,
                excluded_indices,
                k,
                int(self.ef_search),
                n_threads,
            )

        if not isinstance(queries, sp.csr_array):
            raise TypeError("a sparse index takes sparse queries.")
        vectors = space.items
        return _core.hnsw_search_sparse(
            *graph,
            _as_int64(vectors.indptr),
            _as_int64(vectors.indices),
            np.ascontiguousarray(vectors.data, dtype=np.float64),
            space.dim,
            _as_int64(queries.indptr),
            _as_int64(queries.indices),
            np.ascontiguousarray(queries.data, dtype=np.float64),
            candidates,
            excluded_indptr,
            excluded_indices,
            k,
            int(self.ef_search),
            n_threads,
        )

    @property
    @override
    def nbytes(self) -> int:
        return int(self.node_level_.nbytes + self.links_indptr_.nbytes + self.links_indices_.nbytes)


def _as_int64(values: Any) -> NDArray[np.int64]:
    """The int64 layout the kernels borrow, copying only a narrower array."""
    return np.ascontiguousarray(values, dtype=np.int64)


def _pad_queries(queries: NDArray[np.float64], dim: int) -> NDArray[np.float64]:
    """Widen dense queries to the indexed width with zeros."""
    if queries.shape[1] == dim:
        return np.ascontiguousarray(queries)
    padded = np.zeros((queries.shape[0], dim), dtype=np.float64)
    padded[:, : queries.shape[1]] = queries
    return padded


def kernel_ready(space: VectorSpace) -> VectorSpace:
    """Put the item vectors in the layout the kernels borrow, once and for all.

    scipy stores its index arrays as ``int32`` whenever they fit, and the kernels borrow
    ``int64``. Converting at query time would mean rebuilding the whole index array on
    every call -- twelve million entries for a six-figure catalog, which dwarfs the
    search it is preparing for and is paid again by the next request. Doing it here
    makes the conversion in :meth:`HNSW.query` a no-op that returns the same array.
    """
    if isinstance(space, DenseSpace):
        return DenseSpace(np.ascontiguousarray(space.items, dtype=np.float64))
    items = space.items
    return SparseSpace(
        sp.csr_array(
            (
                np.ascontiguousarray(items.data, dtype=np.float64),
                _as_int64(items.indices),
                _as_int64(items.indptr),
            ),
            shape=items.shape,
        )
    )


def equalize_norms(space: VectorSpace) -> VectorSpace:
    """Give every item vector the same norm, without changing a single score.

    An inner product is not a metric, and a graph built under one navigates badly when
    the item norms vary: a high-norm item beats its neighbours for every query at once,
    so it collects the links and the rest of the catalog gets harder to walk to. The
    standard repair is to spend one dimension on the difference [1]_. Item ``j`` gets
    ``sqrt(max_norm**2 - ||v_j||**2)`` in the new dimension, which brings every item to
    ``max_norm``, and a query gets a zero there, which contributes nothing. Every
    query-item score is therefore *exactly* what it was, while the geometry the graph is
    built over becomes the sphere, where nearest by inner product and nearest by angle
    are the same question.

    It is not free -- one more dimension per distance, and one more stored value per
    sparse item row -- but it is the difference between a graph that finds four fifths
    of the right answers and one that finds most of the rest. A bias term makes the norm
    spread worse, so the models with biases need it most.

    References
    ----------
    .. [1] Y. Bachrach et al., "Speeding up the Xbox recommender system using a Euclidean
       transformation for inner-product spaces", RecSys 2014.
       :doi:`10.1145/2645710.2645741`
    """
    if isinstance(space, DenseSpace):
        items = np.ascontiguousarray(space.items, dtype=np.float64)
        slack = _slack(np.linalg.norm(items, axis=1))
        return DenseSpace(np.hstack([items, slack[:, None]]))

    items = space.items
    norms = np.sqrt(np.asarray(items.multiply(items).sum(axis=1)).ravel())
    slack = _slack(norms)
    # One stored value an item, in a column no query ever has an entry in.
    padding = sp.csr_array(
        (slack, np.zeros(len(slack), dtype=items.indices.dtype), np.arange(len(slack) + 1)),
        shape=(items.shape[0], 1),
    )
    widened = sp.csr_array(sp.hstack([items, padding], format="csr"))
    widened.sort_indices()
    return SparseSpace(widened)


def _slack(norms: NDArray[np.float64]) -> NDArray[np.float64]:
    """What each item vector is short of the longest one, as a length."""
    longest = float(norms.max(initial=0.0))
    return np.sqrt(np.maximum(longest * longest - norms * norms, 0.0))


register_index("hnsw", HNSW)
