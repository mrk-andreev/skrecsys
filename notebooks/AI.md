# Notebooks

Every `.py` file here is a [marimo](https://docs.marimo.io) notebook, not a plain script. marimo
and the plotting libraries live in the `notebooks` dependency group, so a notebook opens with
`uv run --group notebooks marimo edit notebooks/<name>.py` and runs with
`uv run --group notebooks python notebooks/<name>.py`.

The notebooks are explanatory examples of skrecsys: each one teaches a part of the library on
real data. Their prose goes in `mo.md(...)` cells next to the code it explains, never in a
separate `.md` file — this file and `AGENTS.md`/`CLAUDE.md` are the only markdown here. Number
the files in reading order, and link each notebook to the next:

1. `01_intro_to_recommendations.py` — feedback types, collaborative filtering, metrics,
   baselines, evaluation;
2. `02_model_selection_and_tuning.py` — `GridSearchCV`, `AutoTune`, `Study`, time-aware splits;
3. `03_candidates_and_ranking.py` — `Cascade`, `ReciprocalRankFusion`, business rules with
   `Cascade(postprocess=...)`, cold start with `Switch`;
4. `04_production.py` — pickling, `partial_fit`, `exclude_interactions`/`candidates`, indexes;
5. `05_sequential_and_neural.py` — leave-last-out and `skrecsys.nn`. It needs the `nn` extra
   (`uv run --group notebooks --extra nn ...`) and imports `skrecsys.nn` in a `try` so that
   everything but the training still runs without torch;
6. `06_time_aware_recommendation.py` — backtests, time-based `cv=`, decay weights,
   `Cascade(time=True)`, bitemporal lookups, `as_of`;
7. `07_inspecting_recommendations.py` — `trace`, `level=`/`sample=`, `explain` and its
   statuses, exact and approximate reasons, ranker contributions, where relevant items are lost.
8. `08_query_context.py` — context columns in `fit` and `recommend`, context features with
   `JoinDynamicFeatures("context")` and `("item-context")`, per-request evaluation against the per-user scorers.

Every notebook ends with a "Check yourself" cell: 5–7 questions in an `mo.accordion`, each
answer hidden until opened. Ask about the concepts and the API the notebook teaches, and make
sure every answer agrees with what the notebook actually shows.

When a notebook quotes a number, compute it in an f-string from the cell's results rather
than typing it, and check every claim in the prose against the output after a run.

## Format

```python
import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from skrecsys.recommendation import EASE  # shared imports and one-time setup


@app.cell
def _(X_train):              # parameters: names this cell reads from other cells
    model = EASE().fit(X_train)
    mo.md("...")             # the last expression is the cell's output
    return (model,)          # names this cell defines for other cells


if __name__ == "__main__":
    app.run()
```

- Put imports in `with app.setup:`, not in cells. Classes passed as cell parameters trip ruff,
  and marimo forbids the same name being defined in two cells.
- Each global name is defined in exactly one cell. Use `_name` for cell-local temporaries.
- Cells form a dataflow graph, not a top-to-bottom order: don't mutate another cell's objects.
- Gate expensive work (cross-validation, large datasets) behind `mo.ui.run_button` +
  `mo.stop(...)`, and let it run straight through in script mode
  (`mo.app_meta().mode == "script"`).
- Load data through `skrecsys.datasets`, which caches downloads in `~/skrecsys_data`. Resolve
  any other path from `mo.notebook_dir()`, never the cwd, and write outputs under
  `.data/notebooks/<name>/`, never into `notebooks/`.
- Seed everything random (`random_state=0`) so the numbers the prose quotes stay true.
- Sort by time with a random tie-breaker: MovieLens users rate in batches, and the raw file
  orders same-second rows by item id, so "latest" would otherwise mean "highest id". Say
  whether a time split is per user or a global cutoff; only the latter has no leak.
- Choose models, generators and hyperparameters on validation data, never on the test set.
  When prose says one model beats another, back it with the evidence the cell computed (a
  paired per-user bootstrap, or the same order in every fold), and word it conditionally.

## Checks

```bash
uv run --group notebooks marimo check --strict notebooks/<name>.py
uv run ruff check notebooks/<name>.py
uv run --group notebooks python notebooks/<name>.py
```

Ruff formatting is excluded here (marimo owns it). The rules ignored for notebooks in
`pyproject.toml` are the ones marimo's format and `mo.md` prose trip (`B018`, `E501`, `RUF001`,
`PLC0415`, `PLR1711`); fix anything else at the root. Keep code lines within 100 characters even
though `E501` is off.
