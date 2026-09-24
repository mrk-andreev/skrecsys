"""The guards that turn a missing extra into a message that says what to do.

Run on the extras-free CI matrix, which is exactly where the guards matter.
"""

import importlib
import importlib.util

import pytest

from skrecsys.compose import Cascade


@pytest.mark.parametrize("extra", ["catboost", "xgboost", "lightgbm"])
def test_import_error_names_the_extra(extra):
    if importlib.util.find_spec(extra) is not None:
        pytest.skip(f"{extra} is installed, so the guard cannot fire")
    with pytest.raises(ImportError, match=rf"pip install skrecsys\[{extra}\]"):
        importlib.import_module(f"skrecsys.integrations.{extra}")


def test_the_rest_still_imports():
    """A missing extra must not cost anything outside its :mod:`skrecsys.integrations` module."""
    importlib.import_module("skrecsys.integrations")
    assert Cascade.__name__ == "Cascade"
