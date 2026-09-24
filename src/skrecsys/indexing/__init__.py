"""Optional approximate indexes for the ``recommend`` path.

Scoring a query against a whole catalog is exact and linear in the catalog. A vector
index trades the first for the second: it narrows the items a query is scored against,
so a large catalog costs a query far less than it used to, and in return the answer is
the best items the index *found* rather than the best items there are.

Whether that trade pays off depends on the model and the data. ``benchmarks/indexes.py``
measures it for each model: recall against the exact path, the ranking quality lost, and
the throughput gained. Without an index (``index=None``, the default) every model scores
exactly.
"""

from skrecsys.indexing._base import (
    MIN_EXPECTED_OVERLAP,
    DenseSpace,
    SparseSpace,
    VectorIndex,
    VectorSpace,
    available_indexes,
    is_navigable,
    make_index,
    register_index,
)
from skrecsys.indexing._hnsw import HNSW
from skrecsys.indexing._mixin import VectorIndexMixin
from skrecsys.indexing._quantized import QuantizedFlatIndex

__all__ = [
    "HNSW",
    "MIN_EXPECTED_OVERLAP",
    "DenseSpace",
    "QuantizedFlatIndex",
    "SparseSpace",
    "VectorIndex",
    "VectorIndexMixin",
    "VectorSpace",
    "available_indexes",
    "is_navigable",
    "make_index",
    "register_index",
]
