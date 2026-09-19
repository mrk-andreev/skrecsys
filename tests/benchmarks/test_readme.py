"""The README is rendered, not written: it must match its template and the stored results,
and the sentences around the tables must say what the results actually are.
"""

import jinja2
import pytest
import readme
import spec
import store
import suite

from .conftest import stored


def test_the_readme_is_what_its_template_and_results_render_to():
    """The guard that makes a generated README trustworthy.

    ``README.md`` is edited by editing ``README.md.j2`` or by storing a result, and this
    fails the moment the committed file drifts from either: a hand edit to the output,
    a template change nobody rendered, a result stored without re-rendering.
    """
    assert readme.is_current(), (
        "README.md differs from README.md.j2 rendered with benchmarks/results; run "
        "`python benchmarks/run.py render` and commit the result."
    )


def test_rendering_is_deterministic():
    assert readme.render() == readme.render()


def test_the_readme_says_it_is_generated():
    assert readme.render().startswith("<!-- Generated from README.md.j2 ")


def test_a_misspelled_name_in_a_template_fails_rather_than_renders_blank():
    template = readme.environment().from_string("{{ report.leaderboard.nope }}")
    with pytest.raises(jinja2.UndefinedError):
        template.render(report={"leaderboard": {}})


def test_every_dataset_of_every_report_is_in_the_readme():
    """A dataset added to a config has to be placed in the template to be reported."""
    template = (readme.REPO / readme.TEMPLATE).read_text(encoding="utf-8")
    for benchmark in spec.BENCHMARKS:
        for dataset in spec.load(benchmark).targets:
            assert f'report.{benchmark}["{dataset}"]' in template, (
                f"{benchmark} runs on {dataset}, but README.md.j2 never shows that block."
            )


def test_check_notices_a_hand_edited_readme(tmp_path, monkeypatch):
    target = tmp_path / "README.md"
    monkeypatch.setattr(readme, "README", target)
    target.write_text(readme.render() + "a hand edit\n", encoding="utf-8")
    assert not readme.is_current()
    assert readme.write() is True
    assert readme.is_current()
    assert readme.write() is False


# --- The sentences, against results in a sandbox -------------------------------------


def _block(benchmark: str) -> str:
    """One report's toy block, rendered through the real macros."""
    template = readme.environment().from_string(
        f'{{% import "report.md.j2" as r %}}{{{{ r.{benchmark}(report.{benchmark}["toy"]) }}}}'
    )
    return template.render(report=suite.blocks())


def _measure(benchmark: str, only=None) -> None:
    config = spec.load(benchmark)
    todo = [unit for unit in suite.units(config, "toy") if only is None or unit.model in only]
    assert suite.measure(config, "toy", todo, log=lambda _: None) == []


def test_a_block_names_its_dataset_settings_and_command(sandbox):
    _measure("leaderboard")
    text = _block("leaderboard")
    assert text.startswith("Toy `bands` split, k=3, default hyper-parameters.")
    assert "Regenerate with `python benchmarks/run.py run leaderboard --dataset toy`." in text
    assert "| Model | NDCG@3 |" in text
    assert "spent 0s or reached its cap (1 fits, 1 `recommend` calls)" in text
    assert "Not yet measured" not in text


def test_a_block_with_no_results_is_its_caption_and_a_note(sandbox):
    text = _block("leaderboard")
    assert text.endswith("Not yet measured: MostPopular, ItemKNN.")
    assert "|" not in text
    assert "Measured on" not in text


def test_what_has_not_been_measured_is_named(sandbox):
    _measure("leaderboard", only={"MostPopular"})
    assert "Not yet measured: ItemKNN." in _block("leaderboard")


def test_what_is_out_of_date_is_named_and_still_shown(sandbox):
    _measure("leaderboard")
    sandbox.edit("leaderboard", lambda config: config["models"][1].update(version="v2"))
    text = _block("leaderboard")
    assert "due a re-run: ItemKNN." in text
    assert "| ItemKNN |" in text


def test_what_was_left_out_on_purpose_is_named_with_its_reason(sandbox):
    sandbox.edit(
        "leaderboard",
        lambda config: config["datasets"]["toy"].update(skip={"ItemKNN": "too big"}),
    )
    _measure("leaderboard")
    text = _block("leaderboard")
    assert "Not run: ItemKNN (too big)." in text
    assert "Not yet measured" not in text


def test_one_host_is_named_once(sandbox):
    _measure("leaderboard")
    text = _block("leaderboard")
    assert text.count("Measured on ") == 1
    assert "more than one host" not in text


def test_rows_from_two_hosts_say_which_ran_where(sandbox):
    _measure("leaderboard")
    unit = suite.units(spec.load("leaderboard"), "toy")[1]
    result = stored(unit)
    store.write(unit, result["payload"], host=result["host"] | {"cpu": "Another CPU"})
    text = _block("leaderboard")
    assert "Measured on more than one host" in text
    assert "ItemKNN: Another CPU (" in text


def test_the_index_block_lays_out_every_table(sandbox):
    _measure("indexes")
    text = _block("indexes")
    for heading in (
        "**What the index costs in answers**",
        "**What it costs to build, hold and search**",
        "**What one request costs**",
        "**How ranking time answers to the size of the catalog**",
    ):
        assert heading in text
    assert "| ItemKNN | hnsw(m=4) | ef=2 |" in text
    assert "| hnsw (ef=4) | hnsw speedup |" in text
    # The toy training split covers 9 items, below the default floor, so the caption says
    # why the benchmark lowered it.
    assert "This catalog holds 9 items" in text


def test_an_unmeasured_index_is_named_for_every_model_at_once(sandbox):
    text = _block("indexes")
    assert "Not yet measured: hnsw(m=4) for every model." in text
