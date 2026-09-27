import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    from functools import partial

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GridSearchCV

    from skrecsys.datasets import fetch_movielens_100k
    from skrecsys.metrics import (
        Recall,
        catalog_coverage_at_k,
        evaluate_recommender,
        get_scorer,
        make_recommender_scorer,
        ndcg_at_k,
        recall_at_k,
    )
    from skrecsys.model_selection import WarmStartKFold
    from skrecsys.recommendation import (
        EASE,
        BM25Recommender,
        ItemKNNRecommender,
        MostPopularRecommender,
        RP3Beta,
    )
    from skrecsys.tune import AutoTune, Study, search_space

    TEST_SHARE = 0.2  # share of each user's interactions held out in section 6
    N_STARTUP_TRIALS = 10


@app.cell
def _():
    mo.md(r"""
    # Model selection and tuning

    [Notebook 01](01_intro_to_recommendations.py) compared models with their default
    hyperparameters. Defaults are a reasonable start, but every model has knobs — how many
    neighbours ItemKNN keeps, how strongly EASE regularises — and the best setting depends
    on the data. This notebook shows how to choose them with skrecsys **without fooling
    yourself**:

    1. the three roles data plays: training, validation and test;
    2. a hand-made grid search with scikit-learn's `GridSearchCV`;
    3. the search spaces every skrecsys model declares;
    4. `AutoTune`, which tunes a model inside `fit`;
    5. `Study`, the tuner underneath, for objectives of your own;
    6. the pitfall that matters most: a split that does not match reality.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 1. Train, validation, test

    Tuning is itself a kind of fitting: trying 50 configurations and keeping the one with
    the best score on some data *fits* the hyperparameters to that data. So the data that
    picks the configuration cannot also be the data that reports how good it is. Three
    roles:

    - **training** interactions fit the model's parameters;
    - **validation** interactions pick the hyperparameters — here, by cross-validation
      *inside* the training set with `WarmStartKFold`;
    - **test** interactions are touched once, at the end, to report the result.

    MovieLens 100K ships an official split for this, `ua`: exactly 10 ratings of every user
    are held out as the test set. `fetch_movielens_100k(subset="ua")` returns the row
    positions of both parts. Everything below tunes on `ua`'s training part alone.
    """)
    return


@app.cell
def _():
    _ua = fetch_movielens_100k(subset="ua")
    X_train = _ua.data[_ua.train_indices]
    X_test = _ua.data[_ua.test_indices]
    n_items = len(np.unique(_ua.data[:, 1]))
    mo.md(f"""
    | | interactions | users |
    |---|---|---|
    | `ua` train | {len(X_train):,} | {len(np.unique(X_train[:, 0])):,} |
    | `ua` test | {len(X_test):,} | {len(np.unique(X_test[:, 0])):,} |
    """)
    return X_test, X_train, n_items


@app.cell
def _(X_test, X_train):
    defaults = {
        "MostPopular": MostPopularRecommender(),
        "ItemKNN": ItemKNNRecommender(),
        "BM25": BM25Recommender(),
        "RP3Beta": RP3Beta(),
        "EASE": EASE(),
    }
    default_scores = pd.Series(
        {
            _name: evaluate_recommender(
                _model.fit(X_train), X_test, metrics=[ndcg_at_k], k=10
            )["ndcg@10"]
            for _name, _model in defaults.items()
        },
        name="NDCG@10 on ua test, defaults",
    )
    mo.vstack(
        [
            mo.md("**Reference point:** every model with its defaults, on the `ua` test set."),
            default_scores.round(4).to_frame(),
        ]
    )
    return default_scores, defaults


@app.cell
def _():
    mo.md(r"""
    ## 2. Grid search with scikit-learn

    skrecsys models are scikit-learn estimators, so `GridSearchCV` works as it is. It needs
    two skrecsys pieces: a splitter that keeps every validation user and item in training
    (`WarmStartKFold`), and a scorer (`make_recommender_scorer`). Given a dict of metrics
    the scorer is multi-metric, and `refit=` names the one that picks the winner.

    Below, ItemKNN's two parameters — `n_neighbors`, how many similar items each item
    keeps, and `shrink`, which damps similarities supported by few co-occurrences — are
    searched on a grid, with catalog coverage recorded next to NDCG@10.
    """)
    return


@app.cell
def _(X_train, n_items):
    grid = GridSearchCV(
        ItemKNNRecommender(),
        {"n_neighbors": [5, 10, 20, 50, 100, 200, 500], "shrink": [0.0, 20.0, 100.0]},
        cv=WarmStartKFold(n_splits=3, shuffle=True, random_state=0),
        scoring=make_recommender_scorer(
            {
                "ndcg": ndcg_at_k,
                "coverage": partial(catalog_coverage_at_k, n_catalog_items=n_items),
            },
            k=10,
        ),
        refit="ndcg@10",
    ).fit(X_train)
    return (grid,)


@app.cell
def _(grid):
    _cv = pd.DataFrame(grid.cv_results_)
    _curves = pd.DataFrame(
        {
            "n_neighbors": _cv["param_n_neighbors"].astype(int),
            "shrink": _cv["param_shrink"].astype(float).astype(str),
            "NDCG@10": _cv["mean_test_ndcg@10"],
            "coverage@10": _cv["mean_test_coverage@10"],
        }
    )
    _base = alt.Chart(_curves).encode(
        x=alt.X("n_neighbors:Q", scale=alt.Scale(type="log"), title="n_neighbors (log scale)"),
        color=alt.Color("shrink:N", title="shrink"),
        tooltip=["n_neighbors", "shrink", "NDCG@10", "coverage@10"],
    )
    _ndcg = _base.mark_line(point=True).encode(
        y=alt.Y("NDCG@10:Q", scale=alt.Scale(zero=False))
    )
    _coverage = _base.mark_line(point=True).encode(y=alt.Y("coverage@10:Q"))
    mo.vstack(
        [
            alt.hconcat(
                _ndcg.properties(width=300, height=220, title="Validation NDCG@10"),
                _coverage.properties(width=300, height=220, title="Validation coverage@10"),
            ),
            mo.md(f"""
            `grid.best_params_` = `{grid.best_params_}`, with a mean validation NDCG@10 of
            {grid.best_score_:.4f}. The curves show the typical shape: too few neighbours
            is noisy, too many dilutes the signal with weakly related items. Coverage moves
            the other way — highest with few neighbours and no shrinkage, exactly where
            accuracy is worst. The metric you optimise decides which model you get.
            """),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 3. Declared search spaces

    Writing grids by hand means knowing each model's sensible ranges. skrecsys models
    declare them next to the defaults, as annotations on `__init__` parameters:

    ```python
    class ItemKNNRecommender(...):
        def __init__(
            self,
            n_neighbors: Annotated[int | None, Int(5, 1000, log=True)] = 50,
            shrink: Annotated[float, Float(0.0, 500.0)] = 0.0,
            ...
    ```

    `Annotated` changes nothing at runtime, and `skrecsys.tune.search_space(estimator)`
    reads the ranges back — including those of nested estimators, as `name__param`:
    """)
    return


@app.cell
def _(defaults):
    pd.DataFrame(
        [
            {"model": _name, "parameter": _param, "distribution": repr(_dist)}
            for _name, _model in defaults.items()
            for _param, _dist in search_space(_model).items()
        ]
    ).set_index(["model", "parameter"])
    return


@app.cell
def _():
    mo.md(r"""
    `MostPopularRecommender` has nothing to tune. `log=True` means the range is searched
    on a log scale: the difference between 5 and 10 neighbours matters as much as between
    500 and 1000.

    ## 4. `AutoTune`: tuning inside `fit`

    `AutoTune(model)` is itself a recommender. Its `fit` runs a search over the model's
    declared space, scoring each configuration by 3-fold `WarmStartKFold` NDCG@10 on the
    training interactions alone, then refits the best configuration on all of them. The
    first trial is always the model as given, so it cannot do worse than the defaults in
    cross-validation. The search is a Tree-structured Parzen Estimator (TPE), as in Optuna:
    10 random trials, then trials concentrated where the good ones were.

    Everything is configurable: `search_space=` overrides ranges, `freeze=` holds
    parameters at their current value, `scoring=` and `cv=` change the objective, and
    `n_trials=` the budget.

    `scoring=` takes the metric in whichever form is handiest: a name such as
    `"recall@20"` (a bare `"ndcg"` means cutoff 10), a metric object such as `Recall(20)`,
    or any scorer, such as `make_recommender_scorer` builds. `skrecsys.metrics.get_scorer`
    shows what each resolves to; the named forms are the same scorer as the long one, so
    they tune to the same result.
    """)
    return


@app.cell
def _():
    _forms = [
        None,
        "ndcg",
        "recall@20",
        "MAP@10",
        Recall(20),
        make_recommender_scorer(recall_at_k, k=20),
    ]
    pd.DataFrame(
        {
            "scoring=": [repr(_form) for _form in _forms],
            "resolves to": [repr(get_scorer(_form)) for _form in _forms],
        }
    )
    return


@app.cell
def _():
    tune_button = mo.ui.run_button(label="Tune four models (about 10 seconds)")
    tune_button
    return (tune_button,)


@app.cell
def _(X_train, defaults, tune_button):
    mo.stop(
        not (tune_button.value or mo.app_meta().mode == "script"),
        mo.md("*Press the button to run `AutoTune` with 30 trials per model.*"),
    )
    tuned = {}
    for _name in mo.status.progress_bar(["ItemKNN", "BM25", "RP3Beta", "EASE"], title="Tuning"):
        tuned[_name] = AutoTune(defaults[_name], n_trials=30, random_state=0).fit(X_train)
    return (tuned,)


@app.cell
def _(X_test, default_scores, tuned):
    tuned_results = pd.DataFrame(
        {
            _name: {
                "best parameters": ", ".join(
                    f"{_p}={_v:.3g}" for _p, _v in _model.best_params_.items()
                ),
                "validation NDCG@10 (CV)": _model.best_score_,
                "test NDCG@10, defaults": default_scores[_name],
                "test NDCG@10, tuned": evaluate_recommender(
                    _model, X_test, metrics=[ndcg_at_k], k=10
                )["ndcg@10"],
            }
            for _name, _model in tuned.items()
        }
    ).T.rename_axis("model")
    tuned_results.round(4)
    return (tuned_results,)


@app.cell
def _(tuned_results):
    _gain = tuned_results["test NDCG@10, tuned"] - tuned_results["test NDCG@10, defaults"]
    mo.md(f"""
    Tuning helps most where the defaults were furthest off — here **{_gain.idxmax()}**
    (+{_gain.max():.4f}) — and least where the defaults already sit near the optimum
    (**{_gain.idxmin()}**, {_gain.min():+.4f}). Two further things to notice:

    - The validation scores are higher than the test scores. They are not comparable:
      the validation folds hold out about a third of each user's training interactions,
      `ua` holds out exactly 10, and NDCG@10 depends on how many relevant items each user
      has. Compare models *within* one protocol, never across.
    - Even within one protocol, the best of 30 validation scores is optimistic: some of
      its margin is luck on those particular folds. That is why the test set is separate
      and touched once.
    """)
    return


@app.cell
def _(tuned):
    history_model = mo.ui.dropdown(options=list(tuned), value="RP3Beta", label="model")
    history_model
    return (history_model,)


@app.cell
def _(history_model, tuned):
    _trials = [
        _t for _t in tuned[history_model.value].study_.trials if _t.value is not None
    ]
    _history = pd.DataFrame(
        {
            "trial": [_t.number for _t in _trials],
            "NDCG@10": [_t.value for _t in _trials],
            "phase": ["random" if _t.number < N_STARTUP_TRIALS else "TPE" for _t in _trials],
        }
    )
    _history["best so far"] = _history["NDCG@10"].cummax()
    _points = (
        alt.Chart(_history)
        .mark_circle(size=60)
        .encode(
            x="trial:Q",
            y=alt.Y("NDCG@10:Q", scale=alt.Scale(zero=False), title="validation NDCG@10"),
            color=alt.Color("phase:N", title="sampler"),
            tooltip=["trial", "NDCG@10"],
        )
    )
    _best = alt.Chart(_history).mark_line(color="gray").encode(x="trial:Q", y="best so far:Q")
    mo.vstack(
        [
            (_points + _best).properties(
                width=560, height=240, title=f"AutoTune({history_model.value}) trials"
            ),
            mo.md("""
            Trial 0 is the default configuration. After the random start-up trials, TPE
            proposes values near the best ones seen so far, so the later trials cluster
            near the top.
            """),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 5. `Study`: your own objective

    `AutoTune` is built on `Study`, an ask-and-tell loop you can drive with any objective.
    It is *define-by-run*: the objective asks the trial for values as it goes, so which
    parameters exist can depend on earlier answers. Below, one study searches over the
    **model family and its hyperparameters at once**: the trial first picks a family,
    then draws that family's parameters from its declared search space. Each
    configuration is scored on one validation fold of the training set, to keep it quick.
    """)
    return


@app.cell
def _(X_train):
    _train, _valid = next(
        WarmStartKFold(n_splits=5, shuffle=True, random_state=0).split(X_train)
    )
    _families = {
        "ItemKNN": ItemKNNRecommender,
        "BM25": BM25Recommender,
        "RP3Beta": RP3Beta,
        "EASE": EASE,
    }

    def objective(trial):
        _family = trial.suggest_categorical("family", list(_families))
        _model = _families[_family]()
        _params = {
            _param: trial.suggest(f"{_family}.{_param}", _dist)
            for _param, _dist in search_space(_model).items()
        }
        _model.set_params(**_params).fit(X_train[_train])
        return evaluate_recommender(_model, X_train[_valid], metrics=[ndcg_at_k], k=10)[
            "ndcg@10"
        ]

    enqueued_defaults = [
        {"family": _family}
        | {
            f"{_family}.{_param}": _factory().get_params()[_param]
            for _param in search_space(_factory())
        }
        for _family, _factory in _families.items()
    ]
    return enqueued_defaults, objective


@app.cell
def _(enqueued_defaults, objective, tune_button):
    mo.stop(
        not (tune_button.value or mo.app_meta().mode == "script"),
        mo.md("*Press the tuning button in section 4 to run this study too.*"),
    )
    study = Study(direction="maximize", random_state=0)
    for _params in enqueued_defaults:
        study.enqueue(_params)
    study.optimize(objective, n_trials=40)
    return (study,)


@app.cell
def _(study):
    _rows = pd.DataFrame(
        {
            "trial": [_t.number for _t in study.trials],
            "family": [_t.params["family"] for _t in study.trials],
            "NDCG@10": [_t.value for _t in study.trials],
        }
    )
    _chart = (
        alt.Chart(_rows)
        .mark_circle(size=70)
        .encode(
            x="trial:Q",
            y=alt.Y("NDCG@10:Q", scale=alt.Scale(zero=False), title="validation NDCG@10"),
            color=alt.Color("family:N"),
            tooltip=["trial", "family", "NDCG@10"],
        )
        .properties(width=560, height=240, title="One study over model families")
    )
    mo.vstack(
        [
            _chart,
            mo.md(f"""
            Best trial: `{study.best_params}` with NDCG@10 {study.best_value:.4f}.
            Trials per family: {_rows["family"].value_counts().to_dict()}.

            The first four trials are each family's defaults, queued with
            `study.enqueue(params)`. That matters: TPE spends the budget where good
            scores appeared early, so a family that is unlucky in its first random draws
            can be starved. Enqueuing known-good starting points guarantees the study can
            only improve on them.

            The objective can be anything you can compute: a different metric or cutoff, a
            weighted mix of accuracy and coverage, a latency budget. `study.ask()` and
            `study.tell(trial, value)` let you drive the loop by hand, for instance from a
            job scheduler.
            """),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 6. The pitfall that matters most: time

    Every split so far is random: a user's held-out interactions are scattered through
    their history, so the model trains on some interactions that happened *after* the ones
    it is tested on. In production it is always the other way round — the model is fitted
    on the past and serves the future.

    The table below fits the same models holding out 20% of the data three ways:

    - **random 20%** of each user's interactions;
    - **latest 20% of each user** by timestamp — the common "temporal" split of the
      literature. It is only *per user*: the training set still holds other users'
      interactions from after a test interaction, so the model knows what became popular
      later. That is a leak a live service never has;
    - **global cutoff**: every interaction after one moment in time is held out, for the
      users seen before it — the only one of the three that matches production.

    MovieLens users rate many movies in the same second (the site asks new users to rate
    a batch), so timestamps have ties. Ties are broken at random: sorting them by item
    id, as the raw file does, would make "latest" mean "highest item id".
    """)
    return


@app.cell
def _():
    _movielens = fetch_movielens_100k()
    _rng = np.random.default_rng(0)
    _times = _movielens.timestamps
    # user, then time, then a random key that breaks ties in time
    _order = np.lexsort((_rng.random(len(_times)), _times, _movielens.data[:, 0]))
    _X, _t = _movielens.data[_order], _times[_order]
    _, _start, _counts = np.unique(_X[:, 0], return_index=True, return_counts=True)
    _position = np.arange(len(_X)) - np.repeat(_start, _counts)
    _n_test = np.repeat((_counts * TEST_SHARE).astype(int), _counts)
    _latest = _position >= np.repeat(_counts, _counts) - _n_test
    _random = np.zeros(len(_X), dtype=bool)
    for _s, _c in zip(_start, _counts, strict=True):
        _random[_s + _rng.choice(_c, int(_c * TEST_SHARE), replace=False)] = True
    _after = _t >= np.quantile(_t, 1 - TEST_SHARE)
    # held out after the cutoff; users with no interaction before it cannot be scored
    _known = np.isin(_X[:, 0], _X[~_after, 0])

    _splits = {
        "random 20%": (~_random, _random),
        "latest 20% of each user": (~_latest, _latest),
        "global cutoff": (~_after, _after & _known),
    }
    _scores = {_split: {} for _split in _splits}
    for _split, (_train, _test) in _splits.items():
        for _name, _factory in (
            ("MostPopular", MostPopularRecommender),
            ("ItemKNN", ItemKNNRecommender),
            ("RP3Beta", RP3Beta),
            ("EASE", EASE),
        ):
            _model = _factory().fit(_X[_train])
            _scores[_split][_name] = evaluate_recommender(
                _model, _X[_test], metrics=[ndcg_at_k], k=10
            )["ndcg@10"]
    time_scores = pd.DataFrame(_scores).rename_axis("model")
    global_test_users = len(np.unique(_X[_after & _known, 0]))
    time_scores.round(4)
    return global_test_users, time_scores


@app.cell
def _(global_test_users, time_scores):
    _ratio = (time_scores["latest 20% of each user"] / time_scores["random 20%"]).mean()
    _orders = {
        _split: " > ".join(time_scores[_split].sort_values(ascending=False).index)
        for _split in time_scores
    }
    _changed = len(set(_orders.values())) > 1
    mo.md(f"""
    Predicting the future is harder: holding out each user's latest interactions, the
    scores are {_ratio:.0%} of the random-split scores on average. The global cutoff
    scores differently again, and on only {global_test_users} test users — the ones active
    both before and after the cutoff. The order of models:

    | split | order by NDCG@10 |
    |---|---|
    | random 20% | {_orders["random 20%"]} |
    | latest 20% of each user | {_orders["latest 20% of each user"]} |
    | global cutoff | {_orders["global cutoff"]} |

    {"The **ranking of models changes** with the split, so" if _changed else "Here the order holds, but that is luck, not a rule:"}
    a model or hyperparameter chosen on one protocol is chosen for that protocol.
    [Notebook 06](06_time_aware_recommendation.py) tunes the same models both ways and
    measures the difference on the future.

    So choose the split to match the decision you are making. When the model will serve
    the future, hold out what happened **after a cutoff in time** — for validation as well
    as test. `AutoTune(cv=...)` and `GridSearchCV(cv=...)` accept any iterable of
    `(train, test)` index pairs, so a time-based split plugs straight in.
    `ColdStartSplit` reads row order as time, but per user, like the middle column: it
    adds users never seen in training, not a global cutoff.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Takeaways

    - Hyperparameters are fitted too: pick them on validation data, report on a test set
      touched once.
    - `GridSearchCV` + `WarmStartKFold` + `make_recommender_scorer` for explicit grids;
      `search_space(model)` shows what is worth searching; `AutoTune(model)` does the
      whole search inside `fit`; `Study` handles any other objective.
    - Tune for the metric, cutoff and split you will be judged on. A random split
      overestimates quality and can reorder models when the real task is predicting the
      future; holding out each user's latest interactions still leaks other users'
      future. A global cutoff in time is the split that matches production.

    Next: [notebook 03](03_candidates_and_ranking.py) combines models into a multi-stage
    recommender and handles users the models have never seen.
    """)
    return


@app.cell
def _():
    mo.vstack(
        [
            mo.md(
                "## Check yourself\n\n"
                "Try to answer each question before opening it."
            ),
            mo.accordion(
                {
                    "1. Why can't you choose hyperparameters by their score on the test set?": mo.md(
                        "Choosing the best of many configurations fits them to that data: part of the winner's margin is luck on those particular interactions. Reported on the same data, the score is optimistic. Choose on validation data inside the training set, and touch the test set once at the end."
                    ),
                    '2. What exactly does `AutoTune(model).fit(X)` do?': mo.md(
                        'It searches `search_space(model)` with a TPE sampler: 10 random trials, then trials concentrated where the good ones were. The first trial is the model as given. It scores each trial by 3-fold `WarmStartKFold` NDCG@10 on `X` alone, then refits the best configuration on all of `X`.'
                    ),
                    '3. How do you find out which hyperparameters of a model are worth tuning, and over what range?': mo.md(
                        "`skrecsys.tune.search_space(model)`. It reads the `Float`/`Int`/`Categorical` annotations on the model's `__init__` parameters, nested estimators included."
                    ),
                    '4. `AutoTune` reports a validation NDCG@10 of 0.40, but the `ua` test gives 0.32. Is something broken?': mo.md(
                        "Not necessarily. The two protocols differ: validation holds out a third of each user's training interactions, `ua` exactly 10, and NDCG depends on how many relevant items a user has. Also, the best of many validation scores is optimistic. Compare models within one protocol, never across."
                    ),
                    "5. Why does the study in section 5 enqueue each family's defaults first?": mo.md(
                        'TPE spends its budget where good scores appeared early, so a family that is unlucky in its first random draws can be starved. `study.enqueue(params)` guarantees known-good starting points are evaluated, so the study can only improve on them.'
                    ),
                    '6. How do you make `AutoTune` maximize Recall@20 instead of NDCG@10?': mo.md(
                        'Pass `scoring="recall@20"`, `scoring=Recall(20)` or `scoring=make_recommender_scorer(recall_at_k, k=20)`: the three are the same scorer, and `get_scorer` shows what a name resolves to. Leaving `scoring` out is NDCG@10.'
                    ),
                    '7. You hold out the latest 20% of every user\'s interactions. Can the model still see the future?': mo.md(
                        "Yes. The split is per user: the training set keeps other users' interactions from after a test interaction, so the model knows what became popular later. Only a global cutoff in time — everything after one moment held out — matches a deployed model fitted on the past and serving the future. See notebook 06."
                    ),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
