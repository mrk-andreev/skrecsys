"""The guard that turns a missing optional extra into a message that says what to do.

This is the one module in ``tests/nn`` that runs on the torch-free CI matrix, which is
exactly where the guard matters.
"""

import importlib
import importlib.util

import pytest

from skrecsys.recommendation import MostPopularRecommender

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is not None,
    reason="torch is installed, so the guard cannot fire",
)


def test_import_error_names_the_extra():
    with pytest.raises(ImportError, match=r"pip install skrecsys\[nn\]"):
        importlib.import_module("skrecsys.nn")


def test_the_rest_of_the_package_still_imports():
    """A missing extra must not cost anything outside :mod:`skrecsys.nn`."""
    MostPopularRecommender().fit([["u", "i"], ["u", "j"]])
