"""Unit tests for the leaderboard generator. No network and no `benchmark` marker:
the script is exercised on a synthetic dataset so that it cannot rot unnoticed.
"""

import importlib.util
import sys
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
def rows(dataset):
    models = {
        "MostPopular": leaderboard.default_models()["MostPopular"],
        "ItemKNN": leaderboard.default_models()["ItemKNN"],
    }
    return leaderboard.build_rows(models, dataset, k=3, repeat=1, verbose=False)


def test_build_rows_reports_every_column(rows):
    expected = ["Model", *(column.header for column in leaderboard.columns(3))]
    assert {row["Model"] for row in rows} == {"MostPopular", "ItemKNN"}
    for row in rows:
        assert list(row) == expected
        assert row["fit"].endswith(("ms", "s"))
        assert row["rec"].endswith(("ms", "s"))


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
    popular = leaderboard.evaluate(models["MostPopular"], dataset, k=3, repeat=1)
    knn = leaderboard.evaluate(models["ItemKNN"], dataset, k=3, repeat=1)
    scored = [leaderboard.score(evaluation, 3) for evaluation in (popular, knn)]
    assert float(scored[0]["novelty"]) < float(scored[1]["novelty"])
    assert float(scored[0]["cat cov"]) < float(scored[1]["cat cov"])


def test_evaluate_covers_the_whole_catalog_in_popularity(dataset):
    evaluation = leaderboard.evaluate(
        leaderboard.default_models()["MostPopular"], dataset, k=3, repeat=1
    )
    assert set(evaluation.popularity) == set(evaluation.catalog.tolist())
    assert evaluation.fit_seconds > 0
    assert evaluation.recommend_seconds > 0


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
    leaderboard.write_readme(readme, "| a |", "caption")
    assert readme.read_text() == (
        "intro\n\n<!-- leaderboard -->\ncaption\n\n| a |\n<!-- /leaderboard -->\n\noutro\n"
    )


def test_write_readme_requires_the_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n")
    with pytest.raises(ValueError, match="no leaderboard block"):
        leaderboard.write_readme(readme, "| a |", "caption")


def test_readme_leaderboard_is_up_to_date():
    readme = (_PATH.parents[1] / "README.md").read_text()
    assert leaderboard.README_MARKER in readme
    assert leaderboard.README_END_MARKER in readme
    block = readme.split(leaderboard.README_MARKER)[1].split(leaderboard.README_END_MARKER)[0]
    for column in leaderboard.columns(10):
        assert f"| {column.header} |" in block or f" {column.header} |" in block


def test_main_rejects_bad_arguments():
    for argv in (["--k", "0"], ["--repeat", "0"], ["--models", "NoSuchModel"]):
        with pytest.raises(SystemExit):
            leaderboard.main(argv)
