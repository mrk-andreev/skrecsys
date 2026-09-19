"""The incremental harness end to end, in a sandbox: what runs, what is kept, and what an
edit to a config invalidates -- which should be exactly what it changes and nothing else.
"""

import spec
import store
import suite

from .conftest import stored


def _states(benchmark: str) -> dict[str, str]:
    config = spec.load(benchmark)
    return {
        unit.label.split("/", 2)[2]: store.state(unit)
        for dataset in config.targets
        for unit in suite.units(config, dataset)
    }


def _measure(benchmark: str, only: set[str] | None = None) -> list[tuple[store.Unit, str]]:
    config = spec.load(benchmark)
    failures: list[tuple[store.Unit, str]] = []
    for dataset in config.targets:
        todo = [
            unit
            for unit in suite.units(config, dataset)
            if store.state(unit) != store.FRESH and (only is None or unit.model in only)
        ]
        failures += suite.measure(config, dataset, todo, log=lambda _: None)
    return failures


def test_a_new_config_has_nothing_measured(sandbox):
    assert set(_states("leaderboard").values()) == {store.MISSING}


def test_measuring_stores_a_fresh_result_per_unit(sandbox):
    assert _measure("leaderboard") == []
    assert set(_states("leaderboard").values()) == {store.FRESH}
    result = stored(suite.units(spec.load("leaderboard"), "toy")[0])
    assert result["provenance"]["source"] == "measured"
    assert result["payload"]["quality"]["Model"] == "MostPopular"
    assert set(result["host"]) >= {"cpu", "platform", "build"}


def test_editing_a_parameter_invalidates_that_model_only(sandbox):
    _measure("leaderboard")
    sandbox.edit("leaderboard", lambda config: config["models"][1]["params"].update(n_neighbors=3))
    assert _states("leaderboard") == {"MostPopular": store.FRESH, "ItemKNN": store.STALE}


def test_editing_a_comment_invalidates_nothing(sandbox):
    _measure("leaderboard")
    sandbox.edit("leaderboard", lambda config: config["models"][1].update(comment="reworded"))
    assert set(_states("leaderboard").values()) == {store.FRESH}


def test_bumping_a_version_invalidates_that_model(sandbox):
    # The way to re-run an entry after a change its config cannot see.
    _measure("leaderboard")
    sandbox.edit("leaderboard", lambda config: config["models"][0].update(version="v2"))
    assert _states("leaderboard")["MostPopular"] == store.STALE


def test_a_new_model_is_all_there_is_to_run(sandbox):
    _measure("leaderboard")
    sandbox.edit(
        "leaderboard",
        lambda config: config["models"].append(
            {
                "name": "BM25",
                "package": "skrecsys.recommendation",
                "cls": "BM25Recommender",
                "params": {},
                "version": "v1",
            }
        ),
    )
    states = _states("leaderboard")
    assert states == {"MostPopular": store.FRESH, "ItemKNN": store.FRESH, "BM25": store.MISSING}


def test_a_leaderboard_edit_reaches_every_report_that_inherits_the_model(sandbox):
    _measure("sequential")
    sandbox.edit("leaderboard", lambda config: config["models"][1]["params"].update(n_neighbors=3))
    assert _states("sequential") == {"MostPopular": store.FRESH, "ItemKNN": store.STALE}


def test_a_setting_invalidates_only_the_tables_that_read_it(sandbox):
    """Widening the latency batches re-runs the latency table, and leaves the sweep alone.

    On a real catalog the sweep is the expensive table, so a setting that cannot change
    it must not invalidate it either.
    """
    assert _measure("indexes") == []
    sandbox.edit("indexes", lambda config: config["settings"].update(latency_batch=[1, 2]))
    states = _states("indexes")
    assert states["ItemKNN/hnsw/latency"] == store.STALE
    assert states["ItemKNN/hnsw/sweep"] == store.FRESH
    assert states["ItemKNN/hnsw/scaling"] == store.FRESH


def test_the_single_setting_tables_ignore_the_rest_of_the_dial(sandbox):
    # Latency and scaling run at the dial's middle value only.
    _measure("indexes")
    sandbox.edit("indexes", lambda config: config["indexes"][0]["dial"].update(values=[1, 4, 16]))
    states = _states("indexes")
    assert states["ItemKNN/hnsw/sweep"] == store.STALE
    assert states["ItemKNN/hnsw/latency"] == store.FRESH
    assert states["ItemKNN/hnsw/scaling"] == store.FRESH


def test_the_index_report_stores_every_table(sandbox):
    assert _measure("indexes") == []
    config = spec.load("indexes")
    results = {unit.kind: stored(unit)["payload"] for unit in suite._index_units(config, "toy")}
    sweep = results["sweep"]
    assert sweep["exact"]["quality"]["Index"] == "exact"
    assert [row["quality"]["dial"] for row in sweep["rows"]] == ["ef=2", "ef=4", "ef=8"]
    assert [point["setting"] for point in results["latency"]["points"]] == ["ef=4"]
    assert [point["setting"] for point in results["scaling"]["points"]] == ["ef=4"]


def test_a_model_this_host_cannot_build_is_reported_and_the_rest_still_run(sandbox, monkeypatch):
    """A model needing an extra that is not installed must not cost the rest of the run."""
    real_build = spec.Entry.build

    def build(entry):
        if entry.name == "ItemKNN":
            raise ImportError("No module named 'torch'", name="torch")
        return real_build(entry)

    monkeypatch.setattr(spec.Entry, "build", build)
    failures = _measure("leaderboard")
    assert [(unit.model, "torch" in reason) for unit, reason in failures] == [("ItemKNN", True)]
    assert _states("leaderboard") == {"MostPopular": store.FRESH, "ItemKNN": store.MISSING}


def test_a_crash_in_one_model_keeps_what_the_others_measured(sandbox, monkeypatch):
    real_build = spec.Entry.build

    def build(entry):
        if entry.name == "MostPopular":
            raise RuntimeError("boom")
        return real_build(entry)

    monkeypatch.setattr(spec.Entry, "build", build)
    failures = _measure("leaderboard")
    assert [unit.model for unit, _ in failures] == ["MostPopular"]
    assert "boom" in failures[0][1]
    assert _states("leaderboard")["ItemKNN"] == store.FRESH


def test_a_stale_result_is_still_reported(sandbox):
    """An outdated measurement is still a measurement; it is only the run that redoes it."""
    _measure("leaderboard")
    sandbox.edit("leaderboard", lambda config: config["models"][1].update(version="v2"))
    block = suite.blocks()["leaderboard"]["toy"]
    assert {row["Model"] for row in block["quality"]} == {"MostPopular", "ItemKNN"}
    assert block["stale"] == ["ItemKNN"]
    assert block["missing"] == []


def test_rows_are_ranked_best_first_whatever_order_they_were_measured_in(sandbox):
    _measure("leaderboard", only={"ItemKNN"})
    _measure("leaderboard", only={"MostPopular"})
    block = suite.blocks()["leaderboard"]["toy"]
    scores = [float(row["NDCG@3"]) for row in block["quality"]]
    assert scores == sorted(scores, reverse=True)
    assert [row["Model"] for row in block["timing"]] == [row["Model"] for row in block["quality"]]


def test_rows_from_two_hosts_are_kept_apart(sandbox):
    _measure("leaderboard")
    config = spec.load("leaderboard")
    unit = suite.units(config, "toy")[1]
    result = stored(unit)
    store.write(unit, result["payload"], host=result["host"] | {"cpu": "Another CPU"})
    hosts = suite.blocks()["leaderboard"]["toy"]["hosts"]
    assert len(hosts) == 2
    assert sorted(len(group["models"]) for group in hosts) == [1, 1]
