"""The benchmark configs: that they load, that mistakes in them are caught, and what a key
depends on -- which is the whole contract of the incremental harness.
"""

import copy
from typing import Any

import pytest
import spec

from skrecsys.indexing import HNSW


@pytest.mark.parametrize("benchmark", spec.BENCHMARKS)
def test_every_committed_config_loads(benchmark):
    config = spec.load(benchmark)
    assert config.models
    assert config.targets
    for dataset in config.targets:
        assert config.active_models(dataset)


def test_the_reports_inherit_their_models_from_the_leaderboard():
    leaderboard = spec.load("leaderboard").models
    sequential = spec.load("sequential").models
    indexes = spec.load("indexes").models
    # `"*"` expands to every leaderboard model, after the sequential models of its own.
    assert sequential[-len(leaderboard) :] == leaderboard
    # A name refers to the leaderboard's entry, so the two are the same object.
    by_name = {entry.name: entry for entry in leaderboard}
    for entry in indexes:
        assert entry == by_name[entry.name]


def test_a_dataset_overrides_settings_one_level_deep():
    target = spec.load("leaderboard").targets["amazon-books"]
    assert target.settings["repeat"] == 5
    # `warmup` is merged key by key, and `k` comes from the config's own settings.
    assert target.settings["warmup"] == {"fit": 1, "rank": 2}
    assert target.settings["k"] == 10


def test_a_dataset_can_narrow_an_index_dial():
    config = spec.load("indexes")
    hnsw = {entry.name: entry for entry in config.active_indexes("amazon-books")}["hnsw"]
    default = {entry.name: entry for entry in config.active_indexes("movielens-100k")}["hnsw"]
    assert hnsw.dial.values == (32, 64, 128)
    assert default.dial.values == (16, 32, 64, 128, 256)


def test_the_title_fills_in_the_dataset_s_own_parameters():
    title = spec.load_datasets()["movielens-1m"].title()
    assert title == "MovieLens 1M (`ml-1m-l200`), `leave-one-out` split"


def test_an_entry_builds_its_class_with_its_params():
    als = spec.load("leaderboard").model("ALS").build()
    assert type(als).__name__ == "AlternatingLeastSquares"
    assert als.get_params()["random_state"] == 0


def test_a_nested_object_spec_is_built_too():
    estimator = spec.build_object(
        "skrecsys.recommendation",
        "ItemKNNRecommender",
        {"index": {"package": "skrecsys.indexing", "cls": "HNSW", "params": {"m": 8}}},
    )
    assert isinstance(estimator.index, HNSW)
    assert estimator.index.m == 8


# --- What a key depends on --------------------------------------------------------------


def _key(entry: dict[str, Any]) -> str:
    return spec.key({"model": spec._entry("test", entry).spec()})


ENTRY = {
    "name": "ALS",
    "package": "skrecsys.recommendation",
    "cls": "AlternatingLeastSquares",
    "params": {"random_state": 0, "n_factors": 8},
    "version": "v1",
    "comment": "first",
}


def test_prose_does_not_move_a_key():
    edited = copy.deepcopy(ENTRY) | {"comment": "rewritten entirely"}
    assert _key(edited) == _key(ENTRY)


def test_key_order_does_not_move_a_key():
    reordered = copy.deepcopy(ENTRY)
    reordered["params"] = {"n_factors": 8, "random_state": 0}
    assert _key(reordered) == _key(ENTRY)


@pytest.mark.parametrize(
    "change",
    [
        {"params": {"random_state": 0, "n_factors": 16}},
        {"params": {"random_state": 0}},
        {"version": "v2"},
        {"cls": "BayesianPersonalizedRanking"},
    ],
    ids=["param-value", "param-removed", "version", "class"],
)
def test_anything_that_changes_a_measurement_moves_the_key(change):
    assert _key(copy.deepcopy(ENTRY) | change) != _key(ENTRY)


# --- Mistakes the loader refuses --------------------------------------------------------


def _refused(sandbox, name, change, match):
    sandbox.edit(name, change)
    with pytest.raises(spec.ConfigError, match=match):
        spec.load(name)


def test_a_misspelled_key_is_an_error_not_a_default(sandbox):
    # The likeliest unknown key is a misspelled known one; `parms` silently building the
    # model at its defaults is exactly the wrong number this format exists to prevent.
    def misspell(config):
        config["models"][1]["parms"] = config["models"][1].pop("params")

    _refused(sandbox, "leaderboard", misspell, r"missing \['params'\]")


def test_an_unknown_key_is_named(sandbox):
    _refused(
        sandbox,
        "leaderboard",
        lambda config: config["models"][0].update(notes="x"),
        r"unknown \['notes'\]",
    )


def test_a_version_must_be_given(sandbox):
    _refused(
        sandbox,
        "leaderboard",
        lambda config: config["models"][0].update(version=""),
        "version must be a non-empty string",
    )


def test_a_name_may_appear_once(sandbox):
    _refused(
        sandbox,
        "leaderboard",
        lambda config: config["models"].append(copy.deepcopy(config["models"][0])),
        "listed twice",
    )


def test_a_skip_must_name_a_model(sandbox):
    _refused(
        sandbox,
        "leaderboard",
        lambda config: config["datasets"]["toy"].update(skip={"ItemKNNN": "typo"}),
        "not models here",
    )


def test_a_dial_override_must_name_an_index(sandbox):
    _refused(
        sandbox,
        "indexes",
        lambda config: config["datasets"]["toy"].update(dials={"hsnw": [2]}),
        "not indexes here",
    )


def test_a_reference_must_name_an_inherited_model(sandbox):
    _refused(
        sandbox,
        "indexes",
        lambda config: config["models"].append("NoSuchModel"),
        "is not a model of leaderboard",
    )


def test_a_wildcard_needs_something_to_expand(sandbox):
    _refused(
        sandbox,
        "leaderboard",
        lambda config: config["models"].append("*"),
        "needs a models_from",
    )


def test_a_dataset_must_be_defined(sandbox):
    _refused(
        sandbox,
        "leaderboard",
        lambda config: config["datasets"].update(elsewhere={}),
        "not defined in datasets.json",
    )


def test_the_schema_is_checked(sandbox):
    _refused(sandbox, "leaderboard", lambda config: config.update(schema=2), "schema must be 1")


def test_a_dial_needs_values(sandbox):
    _refused(
        sandbox,
        "indexes",
        lambda config: config["indexes"][0]["dial"].update(values=[]),
        "non-empty list of integers",
    )


# --- Settings, which every run reads and nothing else checks ---------------------------


@pytest.mark.parametrize(
    ("change", "match"),
    [
        # A cap of zero samples would leave a model fitted zero times.
        ({"repeat": 0}, "repeat must be an integer >= 1"),
        ({"rank_repeat": 0}, "rank_repeat must be an integer >= 1"),
        ({"k": True}, "k must be an integer >= 1"),
        ({"budget": 0}, "budget must be a number of seconds > 0"),
        ({"warmup": {"fit": -1, "rank": 0}}, "warm-up counts must be integers >= 0"),
        ({"warmup": {"fit": 0}}, r"missing \['rank'\]"),
        ({"k_value": 3}, r"unknown \['k_value'\]"),
    ],
    ids=["repeat", "rank-repeat", "k-bool", "budget", "warmup-negative", "warmup-key", "typo"],
)
def test_settings_a_run_could_not_use_are_refused(sandbox, change, match):
    _refused(sandbox, "leaderboard", lambda config: config["settings"].update(change), match)


def test_a_setting_a_run_needs_is_refused_when_missing(sandbox):
    _refused(sandbox, "leaderboard", lambda config: config["settings"].pop("k"), r"missing \['k'\]")


def test_a_dataset_override_is_checked_after_merging(sandbox):
    # The override is what the run on that dataset reads, so that is what is checked.
    _refused(
        sandbox,
        "leaderboard",
        lambda config: config["datasets"]["toy"].update(settings={"repeat": 0}),
        "dataset 'toy': repeat must be an integer >= 1",
    )


def test_index_settings_are_checked_too(sandbox):
    _refused(
        sandbox,
        "indexes",
        lambda config: config["settings"].update(catalog_scale=[0.5, 2.0]),
        r"catalog_scale must list shares in \(0, 1\]",
    )
