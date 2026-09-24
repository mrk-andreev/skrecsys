"""The mixin that gives a recommender an optional vector index."""

import numbers
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from skrecsys.base import check_enough_eligible, seen_among, union_of_exclusions
from skrecsys.indexing._base import (
    VectorIndex,
    VectorSpace,
    make_index,
)
from skrecsys.utils.validation import check_ids, encode_ids

__all__ = ["VectorIndexMixin"]

#: Seed for an estimator that has no ``random_state`` of its own. Their fits are already
#: deterministic, so the index must not be the thing that makes a run vary.
_DEFAULT_SEED = 0


class VectorIndexMixin:
    """Mixin giving a recommender an ``index`` parameter and the machinery behind it.

    A recommender opts in by implementing :meth:`_index_space` and
    :meth:`_index_queries`, which together say that its score is an inner product:

        ``score(u, j) == <_index_queries(u), space.items[j]> + _index_score_offset(u)``

    for every fitted user and item. That identity is exact -- it is the *search* that
    approximates, not the space -- and ``tests/indexing/test_space_contract.py`` holds
    every estimator to it.

    An estimator whose score is not an inner product returns ``None`` from
    :meth:`_index_space` and refuses an index at ``fit`` time with a message saying so.

    Estimators using this mixin take an ``index`` parameter of ``None`` (exact, and the
    default), a registered name such as ``"hnsw"``, or a configured
    :class:`~skrecsys.indexing.VectorIndex`.
    """

    def _index_space(self) -> VectorSpace | None:
        """The space this model's score is an inner product in, or ``None``."""
        return None

    def _index_queries(self, user_indices: NDArray[np.intp]) -> NDArray[np.float64] | sp.csr_array:
        """Query vectors for those users, in the space :meth:`_index_space` returned."""
        raise NotImplementedError

    def _index_score_offset(self, user_indices: NDArray[np.intp]) -> NDArray[np.float64] | None:
        """A per-query constant added to every score, or ``None`` when there is none.

        A term that does not vary with the item cannot change a ranking, and making it a
        dimension would only inflate every item vector's norm by the same amount --
        which an inner-product graph is more sensitive to than the ranking is. So it is
        added back to the scores after the search instead of being indexed.
        """
        return None

    def _fit_index(self) -> None:
        """Build ``index_`` from the ``index`` parameter, or set it to ``None``."""
        spec = getattr(self, "index", None)
        index = make_index(spec)
        if index is None:
            self.index_: VectorIndex | None = None
            return
        space = self._index_space()
        if space is None:
            raise ValueError(
                f"{type(self).__name__} does not support index={spec!r}: "
                "its score is not an inner product of a query vector with an item "
                "vector, so a nearest-neighbour index cannot narrow the candidates. "
                "Pass index=None."
            )
        self.index_ = index.fit(
            space, n_threads=self._index_build_threads(), seed=self._index_seed()
        )

    def _index_jobs(self) -> int | None:
        """The estimator's ``n_jobs``, as a thread count, or ``None`` when unset."""
        n_jobs = getattr(self, "n_jobs", None)
        if n_jobs == -1:
            return 0
        if isinstance(n_jobs, numbers.Integral) and not isinstance(n_jobs, bool) and n_jobs >= 1:
            return int(n_jobs)
        return None

    def _index_build_threads(self) -> int:
        """Threads for the index build, where 0 means every core.

        Concurrent insertions see different partial graphs, so a seeded estimator builds
        its index on one thread and gets the same graph every run. The BPR sampler makes
        the same trade for the same reason: reproducibility is worth more than the cores
        when the caller has asked for it by name.
        """
        n_jobs = self._index_jobs()
        if n_jobs is not None:
            return n_jobs
        return 1 if getattr(self, "random_state", None) is not None else 0

    def _index_query_threads(self) -> int:
        """Threads for a search, where 0 means every core.

        Not the build's rule. A search is a pure function of a graph that is already
        fixed -- queries are independent and the heaps break ties by item index -- so
        the thread count cannot change its answer, and holding a seeded estimator to one
        thread here would cost most of what the index was built for. Only an explicit
        ``n_jobs`` narrows it.
        """
        n_jobs = self._index_jobs()
        return 0 if n_jobs is None else n_jobs

    def _index_seed(self) -> int:
        """The seed the index levels are drawn from."""
        random_state = getattr(self, "random_state", None)
        if random_state is None:
            return _DEFAULT_SEED
        return int(check_random_state(random_state).randint(np.iinfo(np.int32).max))

    def _worthwhile_index(self, n_candidates: int, k: int) -> VectorIndex | None:
        """The index to walk for this call, or ``None`` to score the candidates exactly.

        Three reasons converge on the same rule. A graph traversed with only a fraction
        of its nodes eligible needs roughly the reciprocal of that fraction in extra
        walking, and below about a tenth the eligible subgraph stops being connected
        enough to walk at all -- which is why Qdrant estimates a filter's cardinality
        before choosing. An exact scan over a short candidate list is linear in the list
        rather than the catalog, so there is nothing left to save. And asking for a
        large share of the catalog is asking for an ordering, not a search.
        """
        index = self.index_
        if index is None:
            return None
        floor = max(index.min_index_size, 10 * max(k, getattr(index, "ef_search", k)))
        return index if n_candidates >= floor else None

    def _rank_chunk_size(self, n_candidates: int, k: int) -> int:
        """Queries ranked per block.

        The exact path blocks queries to keep the dense score matrix it builds near
        64 MB, which on a six-figure catalog means about a dozen queries at a time. A
        graph walk builds no such matrix, and every block costs a kernel call that
        re-validates the whole index, so the same figure would have a large catalog pay
        that toll hundreds of times per `recommend`. When the index is what will run,
        the block is flat and large.
        """
        if self._worthwhile_index(n_candidates, k) is not None:
            return 65_536
        return int(super()._rank_chunk_size(n_candidates, k))  # ty: ignore[unresolved-attribute]

    def _rank_queries(
        self,
        queries: NDArray[Any],
        item_indices: NDArray[np.intp],
        k: int,
        *,
        exclude_seen: bool,
        excluded: sp.csr_array | None = None,
        first_query: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.floating]]:
        """Rank through the index when there is one worth using, else exactly.

        The only definition of ``_rank_queries`` in the hierarchy: the exact paths are
        ``_rank_queries_exact``, which subclasses override as they always did, so adding
        an index changed no estimator's exact code.
        """
        index = self._worthwhile_index(len(item_indices), k)
        if index is None:
            return self._rank_queries_exact(  # ty: ignore[unresolved-attribute]
                queries,
                item_indices,
                k,
                exclude_seen=exclude_seen,
                excluded=excluded,
                first_query=first_query,
            )
        return self._rank_queries_indexed(
            index,
            queries,
            item_indices,
            k,
            exclude_seen=exclude_seen,
            excluded=excluded,
            first_query=first_query,
        )

    def _rank_queries_indexed(
        self,
        index: VectorIndex,
        queries: NDArray[Any],
        item_indices: NDArray[np.intp],
        k: int,
        *,
        exclude_seen: bool,
        excluded: sp.csr_array | None = None,
        first_query: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.floating]]:
        """Rank by walking the index, with the exact path's contract kept intact."""
        user_idx = encode_ids(check_ids(queries), self.user_ids_, name="user")  # ty: ignore[unresolved-attribute]
        shape = (len(user_idx), len(item_indices))
        seen = (
            seen_among(self.interactions_, user_idx, item_indices)  # ty: ignore[unresolved-attribute]
            if exclude_seen
            else sp.csr_array(shape, dtype=bool)
        )
        excluded = union_of_exclusions(seen, excluded)
        # Eligibility is a property of the candidates and the exclusions, and has
        # nothing to do with how they are ranked, so the error is raised by the same
        # rule and names the same query whether an index is in play or not.
        check_enough_eligible(excluded, len(item_indices), k, first_query)

        order, scores = index.query(
            self._index_queries(user_idx),
            item_indices,
            excluded,
            k,
            n_threads=self._index_query_threads(),
        )
        offset = self._index_score_offset(user_idx)
        if offset is not None:
            scores = scores + offset[:, None]
        return order, scores
