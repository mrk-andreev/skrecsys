"""Unit tests for the leaderboard's measurement: fitting, scoring, timing and tables.

No network and no `benchmark` marker: the module is exercised on a synthetic dataset so
that it cannot rot unnoticed. What runs, and how the README reports it, is tested in
``test_spec``, ``test_suite`` and ``test_readme``.
"""

import time

import leaderboard
import numpy as np
import pytest
import spec


def _model(name):
    """A fresh estimator as ``leaderboard.json`` configures it."""
    return spec.load("leaderboard").model(name).build()


@pytest.fixture
def built(dataset):
    models = {"MostPopular": _model("MostPopular"), "ItemKNN": _model("ItemKNN")}
    return leaderboard.build_rows(models, dataset, k=3, repeat=2, rank_repeat=3)


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
    evaluation = leaderboard.evaluate(_model("MostPopular"), dataset, k=3, repeat=2, rank_repeat=3)
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
    popular = leaderboard.evaluate(_model("MostPopular"), dataset, k=3, repeat=1, rank_repeat=1)
    knn = leaderboard.evaluate(_model("ItemKNN"), dataset, k=3, repeat=1, rank_repeat=1)
    scored = [leaderboard.score(evaluation, 3) for evaluation in (popular, knn)]
    assert float(scored[0]["novelty"]) < float(scored[1]["novelty"])
    assert float(scored[0]["cat cov"]) < float(scored[1]["cat cov"])


def test_evaluate_covers_the_whole_catalog_in_popularity(dataset):
    evaluation = leaderboard.evaluate(_model("MostPopular"), dataset, k=3, repeat=2, rank_repeat=3)
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


def test_sampling_users_is_a_stable_subset():
    users = np.arange(100)
    y_true = [{index} for index in users]
    sampled, sampled_true = leaderboard._sample_users(users, y_true, 10)
    assert len(sampled) == len(sampled_true) == 10
    assert sorted(sampled) == list(sampled), "users stay in their original order"
    assert [next(iter(s)) for s in sampled_true] == list(sampled), "sets follow their user"
    assert leaderboard._sample_users(users, y_true, 1000)[0] is users
    again, _ = leaderboard._sample_users(users, y_true, 10)
    np.testing.assert_array_equal(sampled, again)


def test_host_facts_name_the_machine_and_the_build():
    facts = leaderboard.host_facts()
    assert set(facts) == {"cpu", "cores", "platform", "python", "numpy", "skrecsys", "build"}
    assert facts["build"] in {"debug", "release"}
    assert facts["cores"] is None or facts["cores"] >= 1
