"""Resolving the ``index`` parameter, and the registry behind it."""

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone

from skrecsys.indexing import (
    HNSW,
    QuantizedFlatIndex,
    SparseSpace,
    VectorIndex,
    available_indexes,
    is_navigable,
    make_index,
)
from skrecsys.recommendation import AlternatingLeastSquares


def test_the_registered_indexes_are_the_ones_a_name_may_pick():
    assert available_indexes() == ["hnsw", "quantized-flat"]


def test_none_means_the_exact_path():
    assert make_index(None) is None


def test_a_name_takes_the_defaults():
    index = make_index("hnsw")
    assert isinstance(index, HNSW)
    assert (index.m, index.ef_construction, index.ef_search) == (16, 200, 64)


def test_the_quantized_index_takes_its_defaults_too():
    index = make_index("quantized-flat")
    assert isinstance(index, QuantizedFlatIndex)
    assert (index.bits, index.quantile, index.oversample) == (8, 0.0, 4)


def test_an_instance_is_cloned_rather_than_kept():
    # Fitting must never touch a parameter the caller still holds a reference to.
    original = HNSW(m=8, ef_search=32)
    resolved = make_index(original)
    assert isinstance(resolved, HNSW)
    assert resolved is not original
    assert resolved.get_params() == original.get_params()


def test_an_unknown_name_says_what_is_available():
    with pytest.raises(ValueError, match=r"Unknown index 'ivf'\..*'hnsw'.*'quantized-flat'"):
        make_index("ivf")


@pytest.mark.parametrize("spec", [3, 4.5, object(), ["hnsw"]])
def test_anything_else_is_a_type_error(spec):
    with pytest.raises(TypeError, match="index must be None"):
        make_index(spec)


def test_index_parameters_are_reachable_for_model_selection():
    # The whole reason an index is an estimator: `GridSearchCV` addresses its
    # parameters as `index__<name>` like any other nested estimator.
    estimator = AlternatingLeastSquares(index=HNSW(ef_search=32))
    assert estimator.get_params(deep=True)["index__ef_search"] == 32
    estimator.set_params(index__ef_search=128)
    assert estimator.index.ef_search == 128


def test_cloning_an_estimator_carries_the_index():
    estimator = AlternatingLeastSquares(index=HNSW(m=8))
    copy = clone(estimator)
    assert isinstance(copy.index, HNSW)
    assert copy.index is not estimator.index
    assert copy.index.m == 8


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("bits", 0),
        ("bits", -1),
        ("bits", True),
        ("oversample", 0),
        ("oversample", 1.5),
        ("min_index_size", 0),
    ],
)
def test_quantized_integer_parameters_are_validated_when_the_index_is_built(name, value):
    index = QuantizedFlatIndex(**{name: value})
    with pytest.raises(ValueError, match=f"{name} must be an integer >= 1"):
        index._check_params()


@pytest.mark.parametrize("bits", [3, 5, 6, 7, 16])
def test_only_the_byte_aligned_code_widths_are_accepted(bits):
    # A code that straddles a byte would cost a two-byte read and a pair of shifts per
    # value, so the widths in between are refused rather than silently rounded.
    with pytest.raises(ValueError, match=r"bits must be one of \[1, 2, 4, 8\]"):
        QuantizedFlatIndex(bits=bits)._check_params()


@pytest.mark.parametrize("quantile", [-0.1, 0.5, 0.9, 1.0, True, "0.1"])
def test_the_clipping_quantile_is_a_share_of_one_tail(quantile):
    # Taken off *each* end, so half of one is the whole distribution and there is
    # nothing left to quantize.
    with pytest.raises(ValueError, match=r"quantile must be a float in \[0, 0.5\)"):
        QuantizedFlatIndex(quantile=quantile)._check_params()


def test_quantized_parameters_are_reachable_for_model_selection():
    estimator = AlternatingLeastSquares(index=QuantizedFlatIndex(bits=4))
    assert estimator.get_params(deep=True)["index__bits"] == 4
    estimator.set_params(index__oversample=16)
    assert estimator.index.oversample == 16


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("m", 0),
        ("m", -1),
        ("m", 1.5),
        ("m", True),
        ("ef_construction", 0),
        ("ef_search", 0),
        ("min_index_size", 0),
    ],
)
def test_parameters_are_validated_when_the_index_is_built(name, value):
    index = HNSW(**{name: value})
    with pytest.raises(ValueError, match=f"{name} must be an integer >= 1"):
        index._check_params()


def test_the_base_class_is_abstract_enough_to_subclass():
    assert issubclass(HNSW, VectorIndex)
    assert issubclass(QuantizedFlatIndex, VectorIndex)
    with pytest.raises(NotImplementedError):
        VectorIndex().fit(None)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("n_items", "dim", "per_row", "navigable"),
    [
        (1680, 1680, 50, True),  # MovieLens 100K item-KNN: measured recall 0.998
        (1680, 1680, 20, False),  # the same catalog at BM25's width: recall 0.65
        (660_940, 660_940, 50, False),  # Amazon Books: recall 0.016
    ],
)
def test_navigability_matches_what_was_measured(n_items, dim, per_row, navigable):
    """The threshold is set where the measurements separate, so it has to keep doing so."""
    indptr = np.arange(n_items + 1) * per_row
    indices = np.zeros(n_items * per_row, dtype=np.int64)
    data = np.ones(n_items * per_row)
    items = sp.csr_array((data, indices, indptr), shape=(n_items, dim))
    assert is_navigable(SparseSpace(items)) is navigable
