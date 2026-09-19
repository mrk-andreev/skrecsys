"""A flat scan over quantized codes, with an exact rerank, as a vector index."""

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from skrecsys import _core
from skrecsys._typing import override
from skrecsys.indexing._base import (
    DenseSpace,
    VectorIndex,
    VectorSpace,
    check_positive_int,
    check_tail_fraction,
    register_index,
)
from skrecsys.indexing._hnsw import _as_int64, _pad_queries, kernel_ready

__all__ = ["QuantizedFlatIndex"]

#: Code widths the kernel can pack and unpack, powers of two so a code never straddles a
#: byte. The widths in between would cost a two-byte read and a pair of shifts per value.
SUPPORTED_BITS = (1, 2, 4, 8)

#: The width at which a code is a byte and the packing is the identity.
BYTE_BITS = 8


class QuantizedFlatIndex(VectorIndex):
    """Nearest neighbours by scanning narrow codes, then reranking exactly.

    :class:`~skrecsys.indexing.HNSW` makes a query cheap by visiting fewer items. This
    index visits every candidate and makes each visit cheap instead. The item vectors are
    stored a second time as ``bits``-wide codes, so a pass over the catalog moves four to
    eight times less memory than a pass over the ``float64`` vectors would; the best
    ``oversample * k`` items that pass finds are then rescored with the *original*
    vectors, and the ``k`` returned are the best of those.

    Two consequences worth stating plainly, because they are what distinguish it from the
    graph. The scores are exact -- always, at every ``bits`` -- because nothing is ever
    ranked by a dequantized number; the only thing quantization can cost is a true top-k
    item that never made the shortlist. And a scan cannot fail to reach a candidate, so
    there is no navigability precondition, no fallback path, and no dependence on the
    thread count or the seed. What it does not do is beat a graph on a large catalog: it
    is still linear in the candidates, and ``benchmarks/indexes.py`` is where the two are
    put side by side.

    It also does not shrink a fitted model. The exact vectors are kept for the rerank, so
    the codes are an addition to the footprint rather than a replacement; :attr:`nbytes`
    reports the codes and their scales, which the benchmark prints beside the vectors'
    own megabytes. The win is scan bandwidth and a build that costs two NumPy passes.

    Quantization is affine, ``value ~= code * scale + offset``. A dense space gets a
    scale and an offset per dimension, so a latent dimension with a different spread from
    its neighbours keeps its own resolution. A sparse space gets one pair over all its
    stored values: a catalog-sized column dimension has no per-column statistics worth
    keeping, and per-column arrays would outweigh the codes they compress.

    Parameters
    ----------
    bits : int, default=8
        Code width, one of ``1``, ``2``, ``4`` or ``8``. Codes are bit-packed, so ``4``
        really is half the memory of ``8``. Narrower codes shortlist worse, which
        ``oversample`` is what buys back.
    quantile : float, default=0.0
        Share of each tail clipped when the bounds are chosen. ``0.0`` spans the exact
        minimum and maximum; ``0.001`` trims a thousandth off each end so a handful of
        outliers stop consuming the whole code range at the bulk's expense.
    oversample : int, default=4
        Candidates shortlisted per item asked for: the coarse pass keeps
        ``oversample * k`` and the rerank cuts them to ``k``. The recall-for-latency
        dial, and the only one that can be turned on a fitted model. At ``1`` the codes
        choose the answer outright and the rerank only corrects the scores; at
        ``oversample * k`` above the candidate count the result is exact.
    min_index_size : int, default=4096
        Candidates below which the exact path is used instead. A scan over a short
        candidate list is already cheap, so below this the codes buy nothing.

    Attributes
    ----------
    space_ : DenseSpace or SparseSpace
        The exact item vectors, which the shortlist is reranked against.
    codes_ : ndarray of uint8
        The packed codes, ``8 // bits`` to a byte, low-order code first.
    scale_ : ndarray of shape (dim,) or (1,)
        Per dimension for a dense space, a single value for a sparse one.
    offset_ : ndarray of shape (dim,) or (1,)
        The value a code of zero decodes to, in the same layout as ``scale_``.

    See Also
    --------
    HNSW : Approximate neighbours by walking a graph instead of scanning.

    Examples
    --------
    >>> import numpy as np
    >>> from skrecsys.indexing import QuantizedFlatIndex
    >>> from skrecsys.recommendation import AlternatingLeastSquares
    >>> rng = np.random.default_rng(0)
    >>> X = np.column_stack([rng.integers(0, 200, 4000), rng.integers(0, 500, 4000)])
    >>> index = QuantizedFlatIndex(bits=4, oversample=8, min_index_size=1)
    >>> rec = AlternatingLeastSquares(n_factors=8, random_state=0, index=index)
    >>> items, scores = rec.fit(X).recommend(np.arange(3), n_recommendations=5)
    >>> items.shape
    (3, 5)
    """

    def __init__(
        self,
        bits: int = 8,
        quantile: float = 0.0,
        oversample: int = 4,
        min_index_size: int = 4096,
    ) -> None:
        self.bits = bits
        self.quantile = quantile
        self.oversample = oversample
        self.min_index_size = min_index_size

    def _check_params(self) -> None:
        """Validate parameters, as the estimators do, rather than by constraint table."""
        for name in ("bits", "oversample", "min_index_size"):
            check_positive_int(getattr(self, name), name)
        if int(self.bits) not in SUPPORTED_BITS:
            raise ValueError(f"bits must be one of {list(SUPPORTED_BITS)}, got {self.bits!r}.")
        check_tail_fraction(self.quantile, "quantile")

    @override
    def fit(self, space: VectorSpace, *, n_threads: int = 0, seed: int = 0) -> "QuantizedFlatIndex":
        # Neither keyword is used. Quantization is a pair of deterministic NumPy passes,
        # so there is nothing for a thread count to divide and nothing for a seed to
        # settle -- a fit of this index is the same object however the caller is
        # configured. The signature keeps them because `VectorIndex.fit` promises them.
        del n_threads, seed
        self._check_params()
        # No `equalize_norms`: that transform exists to make an inner-product *graph*
        # navigable, by spending a dimension to bring every item to the same norm. A scan
        # visits every candidate whatever its norm, so here the extra dimension would be
        # one more code per item and one more multiply per score, for nothing.
        space = self.space_ = kernel_ready(space)
        bits = int(self.bits)
        if isinstance(space, DenseSpace):
            codes, self.scale_, self.offset_ = quantize_dense(
                space.items, bits, float(self.quantile)
            )
            self.codes_ = pack_codes(codes, bits)
        else:
            codes, scale, offset = quantize_sparse(space.items.data, bits, float(self.quantile))
            self.scale_, self.offset_ = np.array([scale]), np.array([offset])
            # One code per stored value, sharing the matrix's own rows. There is no row
            # to align -- the kernel walks the codes by the CSR's own positions -- so the
            # packing runs over the whole `data` array as a single run.
            self.codes_ = pack_codes(codes.reshape(1, -1), bits)
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
        # A scan is sequential by construction; the kernels take no thread count at all.
        del n_threads
        space = self.space_
        codes, bits, oversample = self.codes_, int(self.bits), int(self.oversample)
        candidates = _as_int64(candidates)
        excluded_indptr, excluded_indices = _as_int64(excluded.indptr), _as_int64(excluded.indices)

        if isinstance(space, DenseSpace):
            items = np.ascontiguousarray(space.items, dtype=np.float64)
            if isinstance(queries, sp.csr_array):
                # EASE: catalog-wide dense item vectors, but a query of a few nonzeros.
                return _core.quantized_search_sparse_dense(
                    codes,
                    bits,
                    self.scale_,
                    self.offset_,
                    items,
                    _as_int64(queries.indptr),
                    _as_int64(queries.indices),
                    np.ascontiguousarray(queries.data, dtype=np.float64),
                    candidates,
                    excluded_indptr,
                    excluded_indices,
                    k,
                    oversample,
                )
            return _core.quantized_search_dense(
                codes,
                bits,
                self.scale_,
                self.offset_,
                items,
                _pad_queries(np.asarray(queries, dtype=np.float64), space.dim),
                candidates,
                excluded_indptr,
                excluded_indices,
                k,
                oversample,
            )

        if not isinstance(queries, sp.csr_array):
            raise TypeError("a sparse index takes sparse queries.")
        vectors = space.items
        return _core.quantized_search_sparse(
            codes,
            bits,
            float(self.scale_[0]),
            float(self.offset_[0]),
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
            oversample,
        )

    @property
    @override
    def nbytes(self) -> int:
        return int(self.codes_.nbytes + self.scale_.nbytes + self.offset_.nbytes)


def clipping_bounds(
    values: NDArray[np.float64], quantile: float, axis: int | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The interval the codes span, trimming ``quantile`` off each tail.

    A code range is spent on whatever the bounds enclose, so one outlying value can cost
    every other value most of its resolution: a dimension running over ``[-1, 1]`` with a
    single entry at ``50`` quantizes the bulk into two of 256 levels. Trimming the tails
    moves the outliers outside the range, where they clip to its ends -- which loses
    their magnitude but not their sign or their rank against the bulk, and is the trade
    worth making whenever the distribution has tails at all.

    ``quantile=0.0`` asks for no trimming and takes the exact extremes, which is the
    right default: it is the only setting that cannot make an answer worse than the data.
    """
    if quantile == 0.0:
        return np.asarray(values.min(axis=axis)), np.asarray(values.max(axis=axis))
    return (
        np.asarray(np.quantile(values, quantile, axis=axis)),
        np.asarray(np.quantile(values, 1.0 - quantile, axis=axis)),
    )


def encode(
    values: NDArray[np.float64],
    lo: NDArray[np.float64],
    hi: NDArray[np.float64],
    bits: int,
) -> tuple[NDArray[np.uint8], NDArray[np.float64], NDArray[np.float64]]:
    """Affine codes over ``[lo, hi]``, with the scale and offset that decode them.

    Returns ``(codes, scale, offset)`` such that ``code * scale + offset`` is the value
    rounded to the nearest of ``2**bits`` levels, clipped to the interval. A degenerate
    interval -- every value in a dimension identical -- gets a zero scale, so every item
    codes to zero and the offset alone carries the value exactly.
    """
    levels = float((1 << bits) - 1)
    span = np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64)
    scale = np.where(span > 0.0, span / levels, 0.0)
    # Division by the zero scale is avoided rather than repaired: a degenerate dimension
    # has one value, and code zero plus the offset is that value with no error at all.
    safe = np.where(scale > 0.0, scale, 1.0)
    codes = np.rint((np.clip(values, lo, hi) - lo) / safe)
    return (
        np.clip(codes, 0.0, levels).astype(np.uint8),
        np.ascontiguousarray(scale, dtype=np.float64),
        np.ascontiguousarray(lo, dtype=np.float64),
    )


def quantize_dense(
    items: NDArray[np.float64], bits: int, quantile: float
) -> tuple[NDArray[np.uint8], NDArray[np.float64], NDArray[np.float64]]:
    """Code a dense item matrix, a scale and an offset per dimension."""
    lo, hi = clipping_bounds(items, quantile, axis=0)
    return encode(items, lo, hi, bits)


def quantize_sparse(
    data: NDArray[np.float64], bits: int, quantile: float
) -> tuple[NDArray[np.uint8], float, float]:
    """Code a sparse matrix's stored values, one scale and offset for all of them.

    The implicit zeros are left out on purpose. They are never visited -- the scan walks
    each item's stored columns -- so including them would only drag the bounds toward
    zero and spend the code range on a value that is never coded.
    """
    if data.size == 0:
        return np.zeros(0, dtype=np.uint8), 0.0, 0.0
    lo, hi = clipping_bounds(data, quantile, axis=None)
    # `encode` works per dimension, and a whole-matrix bound is that with one dimension;
    # `ascontiguousarray` lifts the two scalars to length-one arrays, which unwrap here.
    codes, scale, offset = encode(data, np.asarray(lo), np.asarray(hi), bits)
    return codes, float(scale.reshape(-1)[0]), float(offset.reshape(-1)[0])


def pack_codes(codes: NDArray[np.uint8], bits: int) -> NDArray[np.uint8]:
    """Pack a matrix of codes ``8 // bits`` to a byte, one aligned run per row.

    Low-order code first within a byte, and every row starting on a byte boundary. The
    alignment wastes under a byte per item and buys the scan a row address that is a
    multiplication rather than a bit offset.
    """
    if bits == BYTE_BITS:
        return np.ascontiguousarray(codes, dtype=np.uint8).ravel()
    n_rows, dim = codes.shape
    per_byte = BYTE_BITS // bits
    width = -(-dim // per_byte)
    # Pad the row out to a whole number of bytes, then fold the padded codes together:
    # column `s` of the reshaped block is the code that sits `s * bits` up in its byte.
    padded = np.zeros((n_rows, width * per_byte), dtype=np.uint8)
    padded[:, :dim] = codes
    blocks = padded.reshape(n_rows, width, per_byte)
    shifts = (np.arange(per_byte, dtype=np.uint8) * bits).astype(np.uint8)
    return np.ascontiguousarray(
        np.bitwise_or.reduce(blocks << shifts, axis=2).astype(np.uint8).ravel()
    )


def unpack_codes(packed: NDArray[np.uint8], bits: int, shape: tuple[int, int]) -> NDArray[np.uint8]:
    """Undo :func:`pack_codes`, for tests and for anyone inspecting a fitted index."""
    n_rows, dim = shape
    if bits == BYTE_BITS:
        return np.ascontiguousarray(packed, dtype=np.uint8).reshape(n_rows, dim)
    per_byte = BYTE_BITS // bits
    width = -(-dim // per_byte)
    blocks = packed.reshape(n_rows, width, 1)
    shifts = (np.arange(per_byte, dtype=np.uint8) * bits).astype(np.uint8)
    codes = (blocks >> shifts) & np.uint8((1 << bits) - 1)
    return np.ascontiguousarray(codes.reshape(n_rows, width * per_byte)[:, :dim])


def decode(
    codes: NDArray[np.uint8], scale: NDArray[np.float64], offset: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The values a store approximates, for measuring what the codes cost.

    Nothing in the search path calls this: the scan folds the scale into the query
    instead, and the rerank uses the exact vectors. It is here so that a caller weighing
    ``bits`` can see the error rather than infer it from recall.
    """
    return codes.astype(np.float64) * scale + offset


register_index("quantized-flat", QuantizedFlatIndex)
