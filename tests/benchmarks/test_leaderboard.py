"""Unit tests for the leaderboard generator. No network and no `benchmark` marker:
the script is exercised on a synthetic dataset so that it cannot rot unnoticed.
"""

import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "leaderboard.py"
_spec = importlib.util.spec_from_file_location("leaderboard", _PATH)
assert _spec is not None and _spec.loader is not None
leaderboard = importlib.util.module_from_spec(_spec)
sys.modules["leaderboard"] = leaderboard
_spec.loader.exec_module(leaderboard)


@pytest.fixture
def dataset():
    """40 users over 12 items: each user likes one contiguous band of the catalog."""
    rows, targets = [], []
    for user in range(40):
        start = user % 6
        for item in range(start, start + 6):
            rows.append([f"u{user}", f"i{item}"])
            targets.append(5.0)
    data = np.array(rows)
    target = np.array(targets)
    # Hold out the last two interactions of every user.
    indices = np.arange(len(data))
    test = indices[(indices % 6) >= 4]
    train = indices[(indices % 6) < 4]
    return SimpleNamespace(data=data, target=target, train_indices=train, test_indices=test)


@pytest.fixture
def built(dataset):
    models = {
        "MostPopular": leaderboard.default_models()["MostPopular"],
        "ItemKNN": leaderboard.default_models()["ItemKNN"],
    }
    return leaderboard.build_rows(models, dataset, k=3, repeat=2, rank_repeat=3, verbose=False)


@pytest.fixture
def rows(built):
    return built[0]


def test_build_rows_reports_every_column(rows):
    expected = ["Model", *(column.header for column in leaderboard.columns(3))]
    assert {row["Model"] for row in rows} == {"MostPopular", "ItemKNN"}
    for row in rows:
        assert list(row) == expected


def test_build_rows_reports_every_timing_column(built):
    quality, timed = built
    assert [row["Model"] for row in timed] == [row["Model"] for row in quality]
    for row in timed:
        assert list(row) == leaderboard.timing_columns()
        # 40 users keep 4 interactions each for training, over the 9 items those cover.
        assert row[leaderboard.batch_header("fit")] == "160"
        assert row[leaderboard.batch_header("rank")] == "40 x 9"
        for prefix in ("fit", "rank"):
            for name, _ in leaderboard.STATISTICS[prefix]:
                assert row[f"{prefix} {name}"].endswith(("ms", "s"))


def test_quantiles_are_measured_samples_and_ordered():
    seconds = np.array([0.001, 0.002, 0.005, 0.100])
    statistics = leaderboard.Timing(seconds, "1", "rank").statistics()
    # Nearest-rank never interpolates, so the top quantiles are the slowest sample.
    assert statistics["q99"] == statistics["q95"] == "100 ms"
    assert statistics["mean"] == "27 ms"
    assert statistics["median"] == "4 ms"


def test_fit_reports_the_spread_a_few_samples_resolve():
    seconds = np.array([0.010, 0.012, 0.100])
    statistics = leaderboard.Timing(seconds, "1", "fit").statistics()
    # An outlier moves `max` but leaves the summary a reader compares on untouched.
    assert statistics["min"] == "10 ms"
    assert statistics["median"] == "12 ms"
    assert statistics["max"] == "100 ms"
    assert "mean" not in statistics


def test_sample_stops_at_the_cap_when_calls_are_cheap():
    calls = []
    seconds = leaderboard.sample(lambda: calls.append(1), cap=25, budget=60.0)
    assert len(seconds) == len(calls) == 25


def test_sample_stops_at_the_budget_when_calls_are_slow():
    # A call far slower than the budget still yields the floor, so the row has a median.
    seconds = leaderboard.sample(lambda: time.sleep(0.02), cap=1000, budget=0.03)
    assert leaderboard.MIN_SAMPLES <= len(seconds) < 1000


def test_sample_never_exceeds_a_cap_below_the_floor():
    # An explicit handful wins over the floor, which is what the tests above rely on.
    seconds = leaderboard.sample(lambda: time.sleep(0.02), cap=1, budget=0.001)
    assert len(seconds) == 1


def test_warmup_calls_are_not_timed(dataset):
    evaluation = leaderboard.evaluate(
        leaderboard.default_models()["MostPopular"], dataset, k=3, repeat=2, rank_repeat=3
    )
    # The warm-up runs in addition to the requested samples, never in place of them.
    assert evaluation.fit.seconds.shape == (2,)
    assert evaluation.rank.seconds.shape == (3,)


def test_build_rows_is_sorted_by_ndcg(rows):
    scores = [float(row["NDCG@3"]) for row in rows]
    assert scores == sorted(scores, reverse=True)


def test_metric_values_are_in_range(rows):
    for row in rows:
        for header in ("NDCG@3", "P@3", "R@3", "hit rate", "MAP", "MRR", "cat cov", "user cov"):
            assert 0.0 <= float(row[header]) <= 1.0
        assert float(row["novelty"]) > 0.0
        assert float(row["mean pop"]) > 0.0


def test_popularity_baseline_is_the_least_novel(dataset):
    models = leaderboard.default_models()
    popular = leaderboard.evaluate(models["MostPopular"], dataset, k=3, repeat=1, rank_repeat=1)
    knn = leaderboard.evaluate(models["ItemKNN"], dataset, k=3, repeat=1, rank_repeat=1)
    scored = [leaderboard.score(evaluation, 3) for evaluation in (popular, knn)]
    assert float(scored[0]["novelty"]) < float(scored[1]["novelty"])
    assert float(scored[0]["cat cov"]) < float(scored[1]["cat cov"])


def test_evaluate_covers_the_whole_catalog_in_popularity(dataset):
    evaluation = leaderboard.evaluate(
        leaderboard.default_models()["MostPopular"], dataset, k=3, repeat=2, rank_repeat=3
    )
    assert set(evaluation.popularity) == set(evaluation.catalog.tolist())
    assert evaluation.fit.seconds.shape == (2,)
    assert evaluation.rank.seconds.shape == (3,)
    assert (evaluation.fit.seconds > 0).all()
    assert (evaluation.rank.seconds > 0).all()


def test_render_box_matches_the_documented_shape():
    table = leaderboard.render_box([{"Model": "EASE", "NDCG@10": "0.2767"}])
    assert table.splitlines() == [
        "┌───────┬─────────┐",
        "│ Model │ NDCG@10 │",
        "├───────┼─────────┤",
        "│ EASE  │ 0.2767  │",
        "└───────┴─────────┘",
    ]


def test_renderers_handle_no_rows():
    for render in leaderboard.RENDERERS.values():
        assert render([]) == ""


def test_render_markdown_and_csv():
    rows = [{"Model": "EASE", "MRR": "0.5888"}, {"Model": "ALS", "MRR": "0.1119"}]
    assert leaderboard.render_markdown(rows).splitlines() == [
        "| Model | MRR |",
        "| --- | --- |",
        "| EASE | 0.5888 |",
        "| ALS | 0.1119 |",
    ]
    assert leaderboard.render_csv(rows).splitlines() == [
        "Model,MRR",
        "EASE,0.5888",
        "ALS,0.1119",
    ]


def test_write_readme_replaces_only_the_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("intro\n\n<!-- leaderboard -->\nstale\n<!-- /leaderboard -->\n\noutro\n")
    leaderboard.write_readme(readme, "caption\n\n| a |")
    assert readme.read_text() == (
        "intro\n\n<!-- leaderboard -->\ncaption\n\n| a |\n<!-- /leaderboard -->\n\noutro\n"
    )


def test_write_readme_requires_the_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n")
    with pytest.raises(ValueError, match="no leaderboard block"):
        leaderboard.write_readme(readme, "caption\n\n| a |")


def test_readme_leaderboard_is_up_to_date():
    readme = (_PATH.parents[1] / "README.md").read_text()
    assert leaderboard.README_MARKER in readme
    assert leaderboard.README_END_MARKER in readme
    block = readme.split(leaderboard.README_MARKER)[1].split(leaderboard.README_END_MARKER)[0]
    for header in [column.header for column in leaderboard.columns(10)] + (
        leaderboard.timing_columns()
    ):
        assert f"| {header} |" in block or f" {header} |" in block


def test_main_rejects_bad_arguments():
    for argv in (
        ["--k", "0"],
        ["--repeat", "0"],
        ["--rank-repeat", "0"],
        ["--models", "NoSuchModel"],
    ):
        with pytest.raises(SystemExit):
            leaderboard.main(argv)
