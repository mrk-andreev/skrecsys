"""Runs in the regular suite: every implemented recommender must have a benchmark."""

import importlib

import pytest
import spec

from skrecsys import indexing, recommendation

from .test_movielens_1m import SEQUENTIAL_BENCHMARKS
from .test_movielens_100k import BENCHMARKS


def _implemented_names(module):
    names = []
    for name in module.__all__:
        try:
            getattr(module, name)().fit([["u", "i"]])
        except NotImplementedError:
            continue
        names.append(name)
    return names


def _nn_names():
    """The optional extra's estimators, or nothing when torch is not installed."""
    try:
        return _implemented_names(importlib.import_module("skrecsys.nn"))
    except ImportError:
        return []


@pytest.mark.parametrize("name", recommendation.__all__)
def test_implemented_recommender_is_benchmarked(name):
    if name in _implemented_names(recommendation):
        assert name in BENCHMARKS, f"Add {name} to tests/benchmarks/test_movielens_100k.py."


@pytest.mark.parametrize("name", _nn_names() or [pytest.param("", marks=pytest.mark.skip)])
def test_neural_recommender_is_benchmarked(name):
    """Every model is benchmarked somewhere: on ratings, or on next-item prediction.

    A sequential model belongs in the second, since the first asks it a question it was
    not trained to answer.
    """
    assert name in BENCHMARKS or name in SEQUENTIAL_BENCHMARKS, (
        f"Add {name} to tests/benchmarks/test_movielens_100k.py "
        f"or tests/benchmarks/test_movielens_1m.py."
    )


def _config_names(benchmark):
    return {entry.name: entry for entry in spec.load(benchmark).models}


def _classes(benchmark):
    """``(package, cls)`` of every model a report runs."""
    return {(entry.package, entry.cls) for entry in spec.load(benchmark).models}


@pytest.mark.parametrize("name", recommendation.__all__)
def test_implemented_recommender_is_on_the_readme_leaderboard(name):
    """A new estimator has to be added to the report, not just to the library."""
    if name in _implemented_names(recommendation):
        assert ("skrecsys.recommendation", name) in _classes("leaderboard"), (
            f"Add an entry for {name} to benchmarks/config/leaderboard.json."
        )


@pytest.mark.parametrize("name", _nn_names() or [pytest.param("", marks=pytest.mark.skip)])
def test_neural_recommender_is_in_a_readme_report(name):
    reported = _classes("leaderboard") | _classes("sequential")
    assert ("skrecsys.nn", name) in reported, (
        f"Add an entry for {name} to benchmarks/config/leaderboard.json or sequential.json."
    )


def _takes_an_index(entry):
    try:
        return "index" in entry.build().get_params()
    except ImportError:
        return None


@pytest.mark.parametrize("name", sorted(_config_names("leaderboard")))
def test_indexable_recommender_is_index_benchmarked(name):
    """Anything that can carry an index must say in the report what the index buys.

    ``indexes.json`` lists its models by name rather than taking every leaderboard model,
    because telling which estimators accept an index means importing them, and a report
    has to be listable on a host without the optional extras.
    """
    takes = _takes_an_index(_config_names("leaderboard")[name])
    if takes is None:
        pytest.skip(f"{name} needs an extra that is not installed here.")
    listed = name in _config_names("indexes")
    if takes:
        assert listed, f"{name} takes an index; add it to benchmarks/config/indexes.json."
    else:
        assert not listed, f"{name} has no index parameter; it cannot be in indexes.json."


def test_every_registered_index_is_in_the_report():
    """A new index type must reach the report, not just the library.

    It needs a decision about which dial to sweep before any of its rows mean anything,
    so registering one without adding an entry to ``indexes.json`` fails here.
    """
    reported = {entry.build().__class__ for entry in spec.load("indexes").indexes}
    registered = {indexing.make_index(name).__class__ for name in indexing.available_indexes()}
    assert registered <= reported, (
        f"{sorted(cls.__name__ for cls in registered - reported)} is registered but has no "
        "entry in benchmarks/config/indexes.json."
    )


@pytest.mark.parametrize("entry", spec.load("indexes").indexes, ids=lambda entry: entry.name)
def test_every_index_dial_is_a_parameter_its_index_really_has(entry):
    """A sweep sets its dial by name on a fitted index, which nothing else checks.

    The measurement turns it through ``set_params``, and sklearn raises on an unknown
    parameter -- but only minutes into a real run. This catches a rename in milliseconds.
    """
    assert entry.dial.param in entry.build().get_params()
