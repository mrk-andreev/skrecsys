"""The command line: that it runs what is stale, only that, and never stores what it shouldn't."""

import pytest
import run
import spec
import store
import suite


def _fresh(benchmark: str) -> bool:
    config = spec.load(benchmark)
    return all(
        store.state(unit) == store.FRESH
        for dataset in config.targets
        for unit in suite.units(config, dataset)
    )


def test_run_measures_what_is_missing_and_then_nothing(sandbox, capsys):
    assert run.main(["run", "leaderboard", "--no-render"]) == 0
    assert "2 of 2 to measure" in capsys.readouterr().out
    assert _fresh("leaderboard")
    assert run.main(["run", "leaderboard", "--no-render"]) == 0
    assert "0 of 2 to measure" in capsys.readouterr().out


def test_only_limits_the_run_to_the_models_named(sandbox):
    assert run.main(["run", "leaderboard", "--only", "ItemKNN", "--no-render"]) == 0
    states = {
        unit.model: store.state(unit) for unit in suite.units(spec.load("leaderboard"), "toy")
    }
    assert states == {"MostPopular": store.MISSING, "ItemKNN": store.FRESH}


def test_a_dry_run_measures_nothing(sandbox, capsys):
    assert run.main(["run", "leaderboard", "--dry-run"]) == 0
    assert "2 of 2 to measure" in capsys.readouterr().out
    assert not store.RESULTS_DIR.exists()


def test_force_re_measures_fresh_results(sandbox, capsys):
    run.main(["run", "leaderboard", "--no-render"])
    capsys.readouterr()
    run.main(["run", "leaderboard", "--force", "--dry-run"])
    assert "2 of 2 to measure" in capsys.readouterr().out


def test_no_store_measures_and_keeps_nothing(sandbox, capsys):
    assert run.main(["run", "leaderboard", "--no-store"]) == 0
    out = capsys.readouterr().out
    assert "┌" in out and "NDCG@3" in out, "the measured rows are printed as tables"
    assert not store.RESULTS_DIR.exists()


def test_a_setting_override_is_refused_unless_nothing_is_stored(sandbox):
    """A result measured under settings the config does not state must not be kept.

    It would be stored under a key its config does not compute, so it would read as
    stale forever while claiming to be a measurement of the config.
    """
    with pytest.raises(SystemExit, match="--no-store"):
        run.main(["run", "leaderboard", "--set", "k=5"])
    assert run.main(["run", "leaderboard", "--no-store", "--set", "k=2"]) == 0
    assert not store.RESULTS_DIR.exists()


def test_a_set_value_is_read_as_json():
    assert run._override("k=5") == ("k", 5)
    assert run._override('warmup={"fit": 0, "rank": 1}') == ("warmup", {"fit": 0, "rank": 1})
    assert run._override("name=plain") == ("name", "plain")


def test_status_counts_every_result(sandbox, capsys):
    run.main(["run", "leaderboard", "--only", "ItemKNN", "--no-render"])
    capsys.readouterr()
    assert run.main(["status", "leaderboard"]) == 0
    out = capsys.readouterr().out
    assert "+ missing leaderboard/toy/MostPopular" in out
    assert "1 fresh, 0 stale, 1 missing" in out


def test_status_rejects_an_unknown_benchmark(sandbox):
    with pytest.raises(SystemExit):
        run.main(["status", "leaderbord"])


def test_an_unknown_dataset_is_refused(sandbox):
    with pytest.raises(SystemExit, match="no dataset"):
        run.main(["run", "leaderboard", "--dataset", "elsewhere"])


def test_a_broken_config_is_reported_not_raised(sandbox, capsys):
    sandbox.edit("leaderboard", lambda config: config["models"][0].update(parms={}))
    assert run.main(["status"]) == 2
    assert "config error" in capsys.readouterr().err


def test_a_failed_unit_fails_the_run(sandbox, monkeypatch):
    def broken(entry):
        raise RuntimeError("boom")

    monkeypatch.setattr(spec.Entry, "build", broken)
    assert run.main(["run", "leaderboard", "--no-render"]) == 1


def test_a_stored_run_re_renders_the_readme(sandbox, monkeypatch):
    rendered = []
    monkeypatch.setattr(run.readme, "write", lambda: rendered.append(True) or True)
    run.main(["run", "leaderboard", "--only", "MostPopular"])
    assert rendered == [True]
    run.main(["run", "leaderboard", "--only", "MostPopular"])
    assert rendered == [True], "nothing was measured, so nothing to re-render"


def test_a_set_value_is_checked_like_a_configured_one(sandbox, capsys):
    assert run.main(["run", "leaderboard", "--no-store", "--set", "repeat=0"]) == 2
    assert "repeat must be an integer >= 1" in capsys.readouterr().err
