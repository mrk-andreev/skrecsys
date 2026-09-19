"""Vector spaces, the index base class, and the registry that names the index types."""

import numbers
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone

__all__ = [
    "MIN_EXPECTED_OVERLAP",
    "DenseSpace",
    "SparseSpace",
    "VectorIndex",
    "VectorSpace",
    "available_indexes",
    "is_navigable",
    "make_index",
    "register_index",
]


@dataclass(frozen=True)
class DenseSpace:
    """Item vectors as one dense block, row ``j`` being item ``j``.

    What a latent-factor model has: a few dozen values an item, and the score of an item
    is the dot product of its row with the query's.
    """

    items: NDArray[np.float64]

    @property
    def n_items(self) -> int:
        return int(self.items.shape[0])

    @property
    def dim(self) -> int:
        return int(self.items.shape[1])

    @property
    def nbytes(self) -> int:
        return int(self.items.nbytes)


@dataclass(frozen=True)
class SparseSpace:
    """Item vectors as sparse rows, row ``j`` being item ``j``.

    What an item-item model has: a catalog-wide dimension of which an item stores only
    its neighbours. Qdrant indexes a sparse collection the same way round.
    """

    items: sp.csr_array

    @property
    def n_items(self) -> int:
        return int(self.items.shape[0])

    @property
    def dim(self) -> int:
        return int(self.items.shape[1])

    @property
    def nbytes(self) -> int:
        return int(self.items.data.nbytes + self.items.indices.nbytes + self.items.indptr.nbytes)


#: A space is whichever of the two an estimator's score is an inner product in.
VectorSpace = DenseSpace | SparseSpace


class VectorIndex(BaseEstimator):
    """Base class for the indexes :class:`~skrecsys.indexing.VectorIndexMixin` manages.

    An index is a parameter object that a recommender carries: it holds no fitted state
    of its own until ``fit`` builds one, and what ``fit`` returns is the object a fitted
    estimator keeps as ``index_``. Subclassing :class:`sklearn.base.BaseEstimator` is
    what makes ``clone``, ``get_params`` and ``set_params`` work, so an index parameter
    reaches ``GridSearchCV`` as ``index__<name>`` like any other nested estimator.

    Threads and the seed are not parameters here. They come from the recommender, which
    already has ``n_jobs`` and often ``random_state``, and two sources of truth for
    "is this fit reproducible" is one more than a user can keep track of.
    """

    #: Candidates below which the exact path is used instead. An index is only worth
    #: walking when there is much more catalog than answer; see ``_worthwhile_index``.
    min_index_size: int

    def fit(self, space: VectorSpace, *, n_threads: int = 0, seed: int = 0) -> "VectorIndex":
        """Build the index over ``space`` and return self."""
        raise NotImplementedError

    def query(
        self,
        queries: NDArray[np.float64] | sp.csr_array,
        candidates: NDArray[np.intp],
        excluded: sp.csr_array,
        k: int,
        *,
        n_threads: int = 0,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return the ``k`` best candidate positions of each query, and their scores."""
        raise NotImplementedError

    @property
    def nbytes(self) -> int:
        """Bytes the fitted index occupies, for the report to put a price on it."""
        raise NotImplementedError


#: Stored columns two item vectors are expected to share before a neighbour graph is
#: worth building over them. One is the point where a pair of items is as likely as not
#: to have anything in common at all.
MIN_EXPECTED_OVERLAP = 1.0


def is_navigable(space: VectorSpace) -> bool:
    """Whether a neighbour graph would have anything to navigate by in ``space``.

    Construction links an item to the items nearest it, so it needs the distances
    between *items* to mean something. In a sparse space that is not a given: if the
    item vectors are short relative to the dimension they live in, two of them almost
    never share a stored column, almost every pair scores zero, and the graph gets built
    out of ties. A search then walks a structure that carries no information, and returns
    whatever it happens to reach.

    The expected number of shared columns between two rows of ``r`` entries over ``d``
    columns is about ``r**2 / d``, which is cheap to compute and separates the measured
    cases cleanly. On MovieLens 100K, ``ItemKNNRecommender`` keeps 50 neighbours over
    1,680 items -- an expected overlap of 1.5, and a measured recall@10 of 0.998 --
    while ``BM25Recommender`` keeps 20, giving 0.24 and a recall of 0.65. On Amazon
    Books the same models sit at 0.004 and 0.0006, and their measured recall is 0.016
    and 0.0002: not approximation, but noise.

    A dense space always passes. Its vectors are dense by construction, so every pair of
    items has a distance; whether that distance is *useful* is a question about the
    model, and one ``benchmarks/indexes.py`` answers per model rather than in general.
    """
    if isinstance(space, DenseSpace):
        return True
    stored = space.items.indptr[-1] - space.items.indptr[0]
    if space.n_items == 0 or space.dim == 0:
        return False
    mean_row = float(stored) / space.n_items
    return mean_row * mean_row / space.dim >= MIN_EXPECTED_OVERLAP


#: The index types a string may name. One line per index; nothing else needs to know.
_REGISTRY: dict[str, type[VectorIndex]] = {}


def register_index(name: str, index: type[VectorIndex]) -> None:
    """Make ``index`` available as ``index="<name>"``."""
    _REGISTRY[name] = index


def available_indexes() -> list[str]:
    """The names ``index=`` accepts, sorted."""
    return sorted(_REGISTRY)


def make_index(spec: Any) -> VectorIndex | None:
    """Resolve an ``index`` parameter into an index to build, or ``None`` for exact.

    ``None`` keeps the exact path, a string names a registered index and takes its
    defaults, and an instance is cloned so that fitting never mutates a parameter the
    caller still holds.

    """
    if spec is None:
        return None
    if isinstance(spec, VectorIndex):
        return cast(VectorIndex, clone(spec))
    if isinstance(spec, str):
        if spec not in _REGISTRY:
            names = ", ".join(repr(name) for name in available_indexes())
            raise ValueError(f"Unknown index {spec!r}. Available indexes: {names}.")
        return _REGISTRY[spec]()
    raise TypeError(
        f"index must be None, one of {available_indexes()}, or a "
        f"VectorIndex instance, got {type(spec).__name__}."
    )


def check_positive_int(value: Any, name: str) -> int:
    """Validate a parameter that must be an integer of at least one."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}.")
    return int(value)


#: A tail fraction is taken off *each* end, so half of one is the whole distribution.
MAX_TAIL_FRACTION = 0.5


def check_tail_fraction(value: Any, name: str) -> float:
    """Validate a parameter naming a share of each tail, so under half in total."""
    in_range = isinstance(value, numbers.Real) and 0.0 <= value < MAX_TAIL_FRACTION
    if isinstance(value, bool) or not in_range:
        raise ValueError(f"{name} must be a float in [0, {MAX_TAIL_FRACTION}), got {value!r}.")
    return float(value)
