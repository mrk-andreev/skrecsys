import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import pickle
    import time

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    from skrecsys.datasets import fetch_movielens_100k
    from skrecsys.metrics import evaluate_recommender, hit_rate_at_k, ndcg_at_k
    from skrecsys.recommendation import EASE, ItemKNNRecommender, MostPopularRecommender

    # skrecsys.nn needs the optional torch extra; everything else here runs without it.
    try:
        from skrecsys.nn import HSTU, Mamba4Rec, SimpleX

        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False

    METRICS = {"hit_rate": hit_rate_at_k, "ndcg": ndcg_at_k}
    N_EPOCHS = 50


@app.cell
def _():
    mo.md(r"""
    # Sequential and neural recommenders

    Every model so far treats a user's history as a **set**: which movies they watched,
    not in which order. Often the order is the signal. Someone who just watched the first
    two films of a trilogy wants the third; a shopper who just bought a phone wants a case,
    not another phone. That is a different question:

    - **"What else would this user like?"** — the task of notebooks 01–04;
    - **"What will this user do next?"** — sequential recommendation.

    This notebook sets up the second question, shows how the classical models fare on it,
    and trains the neural models in `skrecsys.nn`. Those need PyTorch:

    ```sh
    uv run --group notebooks --extra nn marimo edit notebooks/05_sequential_and_neural.py
    ```

    Without the extra, the notebook still runs everything except the neural training.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 1. Leave-last-out: predicting the next item

    The protocol of the sequential-recommendation literature: hold out each user's
    **last** interaction, train on everything before it, and rank the held-out item
    against the whole catalog. With one relevant item per user, hit rate@10 is "was the
    next movie in the top 10?".

    **Check what "last" means before trusting it.** MovieLens users rate movies in
    batches, many in the same second, and `fetch_movielens_100k` breaks those ties by item
    id. Leave-last-out on the raw order would hold out *the highest item id of the last
    batch* — a pattern a sequential model can learn (it sees ids go up within a batch) and
    that has nothing to do with what the user wanted next. So ties are broken at random
    below, and the notebook reports how many users it concerns.

    For contrast, we also hold out one **random** interaction per user — the same amount
    of data, but no longer "the next one".

    Leave-last-out is also *per user*: the training set keeps other users' interactions
    from after a user's last one, so the model knows later popularity. That is the
    literature's protocol and we keep it here; [notebook 06](06_time_aware_recommendation.py)
    evaluates after a global cutoff in time instead.
    """)
    return


@app.cell
def _():
    _movielens = fetch_movielens_100k()
    _rng = np.random.default_rng(0)
    _times = _movielens.timestamps
    # user, then time, then a random key that breaks ties in time
    _order = np.lexsort((_rng.random(len(_times)), _times, _movielens.data[:, 0]))
    X, _t = _movielens.data[_order], _times[_order]
    _last = np.r_[X[1:, 0] != X[:-1, 0], True]
    _tied_with_last = np.r_[(X[1:, 0] == X[:-1, 0]) & (_t[1:] == _t[:-1]), False] & np.r_[
        _last[1:], False
    ]
    X_train, X_next = X[~_last], X[_last]
    _, _starts, _counts = np.unique(X[:, 0], return_index=True, return_counts=True)
    _random = np.zeros(len(X), dtype=bool)
    _random[_starts + _rng.integers(_counts)] = True
    X_train_random, X_random = X[~_random], X[_random]
    # The same protocol on the raw order, ties by item id, to show what the order does.
    _raw = _movielens.data
    _raw_last = np.r_[_raw[1:, 0] != _raw[:-1, 0], True]
    _raw_ease = evaluate_recommender(
        EASE().fit(_raw[~_raw_last]), _raw[_raw_last], metrics=[hit_rate_at_k], k=10
    )["hit_rate@10"]
    _random_ease = evaluate_recommender(
        EASE().fit(X_train), X_next, metrics=[hit_rate_at_k], k=10
    )["hit_rate@10"]
    mo.md(f"""
    Training rows: {len(X_train):,}. Held-out rows: {len(X_next):,}, one per user.
    For {_tied_with_last.sum() / len(X_next):.0%} of the users the last interaction shares
    its second with the one before it: their "next item" is really one of a batch, drawn
    at random. How much that matters: EASE finds the next item in its top 10 for
    {_raw_ease:.1%} of users when ties are ordered by item id, and {_random_ease:.1%} when
    they are broken at random. Same data, same model — only the meaning of "last"
    changed. On this dataset the next-item numbers partly measure how well a model
    predicts *a batch*, not a single next step.
    """)
    return X_next, X_random, X_train, X_train_random


@app.cell
def _(X_next, X_random, X_train, X_train_random):
    _rows = {}
    for _name, _factory in (  # defaults, untuned: see notebook 02 for tuning
        ("MostPopular", MostPopularRecommender),
        ("ItemKNN", ItemKNNRecommender),
        ("EASE", EASE),
    ):
        _random = evaluate_recommender(
            _factory().fit(X_train_random), X_random, metrics=METRICS, k=10
        )
        _next = evaluate_recommender(_factory().fit(X_train), X_next, metrics=METRICS, k=10)
        _rows[_name] = {
            "HR@10, random item": _random["hit_rate@10"],
            "HR@10, next item": _next["hit_rate@10"],
            "NDCG@10, next item": _next["ndcg@10"],
        }
    classical = pd.DataFrame(_rows).T.rename_axis("model")
    classical.round(4)
    return (classical,)


@app.cell
def _(classical):
    _spread = classical.max() - classical.min()
    mo.md(f"""
    The classical models are much worse at the next item than at a random one — EASE's
    hit rate drops from {classical.loc["EASE", "HR@10, random item"]:.1%} to
    {classical.loc["EASE", "HR@10, next item"]:.1%}. A set-based model scores a movie by
    its similarity to the *whole* history, so a user's taste from months ago counts as
    much as what they watched yesterday. It cannot tell what comes next from what came
    before. The gap between the best and the worst model is
    {_spread["HR@10, random item"]:.1%} of hit rate on a random item and
    {_spread["HR@10, next item"]:.1%} on the next one: the protocol, not just the model,
    decides the numbers. Never compare numbers across protocols — the README leaderboard
    and its sequential benchmark are separate tables for this reason.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 2. The models in `skrecsys.nn`

    | model | kind | reads |
    |---|---|---|
    | `HSTU` | sequential — attention over the last `max_sequence_length` interactions (the architecture behind Meta's generative recommenders) | the **order** of the history |
    | `Mamba4Rec` | sequential — a selective state-space recurrence, cost linear in the window length | the **order** of the history |
    | `SimpleX` | neural collaborative filtering — a user is the aggregate of their history's item embeddings, trained with a cosine contrastive loss | the history as a set |
    | `XSimGCL` | graph collaborative filtering — embeddings propagated over the user–item graph, with contrastive noise | the history as a set |

    They are ordinary skrecsys recommenders: same `fit`, `recommend`, `predict`,
    `partial_fit`; they pickle; `AutoTune` knows their search spaces. Things specific to
    them:

    - **Row order matters** for `HSTU` and `Mamba4Rec`: within a user, row $i$ precedes
      row $j$ when $i < j$. Sort by user and timestamp before `fit`. Shuffled rows train
      them on a wrong sequence they cannot detect.
    - **Training is expensive.** `device="auto"` uses CUDA or Apple's MPS when present;
      `max_iter` caps the epochs, and the default `early_stopping` ends a fit whose
      training loss has stopped improving.
    - **Torch is only needed to fit.** A fitted model keeps its parameters as numpy arrays,
      so it scores, pickles and unpickles in a service with no torch installed.
    """)
    return


@app.cell
def _():
    model_picker = mo.ui.multiselect(
        options=["HSTU", "Mamba4Rec", "SimpleX"],
        value=["HSTU"],
        label="neural models to train",
    )
    train_button = mo.ui.run_button(label=f"Train ({N_EPOCHS} epochs each)")
    mo.vstack(
        [
            mo.hstack([model_picker, train_button], justify="start"),
            mo.md(
                "Expect HSTU to take about a minute on an Apple-silicon GPU and a few minutes "
                "on CPU, and Mamba4Rec several times longer: its recurrence is plain PyTorch."
            ),
        ]
    )
    return model_picker, train_button


@app.cell
def _(X_train, model_picker, train_button):
    mo.stop(
        not HAS_TORCH,
        mo.md("*`skrecsys.nn` needs PyTorch: rerun with `--extra nn` to train these.*"),
    )
    mo.stop(
        not (train_button.value or mo.app_meta().mode == "script"),
        mo.md("*Choose models and press the button to train them.*"),
    )
    _factories = {"HSTU": HSTU, "Mamba4Rec": Mamba4Rec, "SimpleX": SimpleX}
    neural = {}
    fit_seconds = {}
    for _name in mo.status.progress_bar(model_picker.value, title="Training"):
        _model = _factories[_name](max_iter=N_EPOCHS, device="auto", random_state=0)
        _t0 = time.perf_counter()
        neural[_name] = _model.fit(X_train)
        fit_seconds[_name] = time.perf_counter() - _t0
    return fit_seconds, neural


@app.cell
def _(X_next, classical, fit_seconds, neural):
    _rows = {
        _name: evaluate_recommender(_model, X_next, metrics=METRICS, k=10)
        | {"fit (s)": fit_seconds[_name]}
        for _name, _model in neural.items()
    }
    _neural = pd.DataFrame(_rows).T.rename(
        columns={"hit_rate@10": "HR@10, next item", "ndcg@10": "NDCG@10, next item"}
    )
    next_item = pd.concat(
        [classical[["HR@10, next item", "NDCG@10, next item"]], _neural]
    ).rename_axis("model")
    next_item = next_item.sort_values("HR@10, next item", ascending=False)
    next_item.round(4)
    return (next_item,)


@app.cell
def _(X_next, X_train, fit_seconds, neural, next_item):
    mo.stop(not neural)
    _still_falling = all(
        _model.loss_curve_[-1] < _model.loss_curve_[-5:-1].min() for _model in neural.values()
    )
    _curves = pd.DataFrame(
        [
            {"model": _name, "epoch": _epoch + 1, "training loss": _loss}
            for _name, _model in neural.items()
            for _epoch, _loss in enumerate(_model.loss_curve_)
        ]
    )
    _chart = (
        alt.Chart(_curves)
        .mark_line()
        .encode(x="epoch:Q", y=alt.Y("training loss:Q", scale=alt.Scale(zero=False)))
        .properties(width=260, height=200)
        .facet(column=alt.Column("model:N", title="training loss per epoch"))
        .resolve_scale(y="independent")
    )
    _best = next_item["HR@10, next item"].idxmax()
    _sequential = _best in {"HSTU", "Mamba4Rec"}
    # Paired bootstrap over users: the best neural model against EASE, hit by hit.
    _top_neural = next_item.loc[list(neural), "HR@10, next item"].idxmax()
    _users, _relevant = X_next[:, 0], [{_i} for _i in X_next[:, 1]]
    _hits = {
        _name: hit_rate_at_k(
            _relevant, _model.recommend(_users, n_recommendations=10)[0], average=None
        )
        for _name, _model in (("EASE", EASE().fit(X_train)), (_top_neural, neural[_top_neural]))
    }
    _diff = _hits[_top_neural] - _hits["EASE"]
    _boot = _diff[np.random.default_rng(0).integers(len(_diff), size=(2000, len(_diff)))]
    _low, _high = np.quantile(_boot.mean(axis=1), [0.025, 0.975])
    _real = _low > 0 or _high < 0
    mo.vstack(
        [
            _chart,
            mo.md(f"""
            **{_best}** predicts the next movie best, with a hit rate of
            {next_item.loc[_best, "HR@10, next item"]:.1%} against EASE's
            {next_item.loc["EASE", "HR@10, next item"]:.1%}. Is that more than noise?
            With one held-out item per user, each user is a hit or a miss; resampling
            the {len(_diff)} users gives a 95% interval of [{_low:+.1%}, {_high:+.1%}]
            for {_top_neural} minus EASE.
            {("The interval excludes zero. " + ("A sequential model reads the history as a sequence and weighs the latest interactions most — the information a set-based model throws away." if _sequential else "")) if _real else "The interval contains zero: on this dataset, with ties broken at random, the two are not distinguishable. MovieLens histories are largely rating batches, so there is little order for a sequential model to read; with ties ordered by item id, it could learn that order instead, as section 1 warns."}
            The neural models run with their defaults and the classical ones too; a
            claim that one family beats another needs both tuned on validation data
            ([notebook 02](02_model_selection_and_tuning.py)). Losses are on each model's
            own scale. {"They are still falling at the end of the budget, so more epochs would help further." if _still_falling else "They have flattened out by the end of the budget."}

            The cost is the fit: {max(fit_seconds.values()):.0f} s of training against
            EASE's fraction of a second, and a GPU to make it bearable on real data. When the task is "what else", not
            "what next", the classical models of notebook 01 remain strong, cheap
            baselines. An order-agnostic neural model such as SimpleX (try it above)
            reads the history as a set too, so it runs into the same limit here.
            """),
        ]
    )
    return


@app.cell
def _(neural):
    mo.stop(not neural)
    _name, _model = next(iter(neural.items()))
    _payload = pickle.dumps(_model)
    mo.md(f"""
    **Shipping it.** The fitted `{_name}` pickles to {len(_payload) / 2**20:.1f} MB of
    numpy arrays. The service that loads it does not need torch; `partial_fit` resumes
    training from the stored parameters and optimizer state, as a warm start (see
    [notebook 04](04_production.py)).
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## The series so far

    1. **[Intro](01_intro_to_recommendations.py)** — recommendation is ranking; implicit
       vs explicit feedback; evaluate with a split, per-user top-$k$ lists and several
       metrics; always compare with `MostPopularRecommender`.
    2. **[Model selection](02_model_selection_and_tuning.py)** — tune on validation data,
       report on a test set touched once; `AutoTune` and `Study`; match the split to the
       decision, which usually means time.
    3. **[Candidates and ranking](03_candidates_and_ranking.py)** — generators judged by
       recall, rankers that use side information, `Switch` for cold users.
    4. **[Production](04_production.py)** — pickled versions, `partial_fit` between
       refits, request-time controls, indexes for large catalogs.
    5. **Sequential and neural** — when the order of the history is the signal, a
       sequential model such as `HSTU` can win, at the cost of training; first check
       what "order" means in your data, ties included.
    6. **[Time-aware recommendation](06_time_aware_recommendation.py)** — evaluate and
       tune on the future, weight recent data, and keep features free of leaks,
       bitemporal ones included.
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
                    '1. Why is "the next item" so much harder than "a random held-out item" for EASE or ItemKNN?': mo.md(
                        'They treat the history as a set and score an item by its similarity to all of it, so old and recent interactions weigh the same. The next item depends mostly on the latest ones, which they cannot single out.'
                    ),
                    '2. What must be true of the rows of `X` before fitting `HSTU` or `Mamba4Rec`?': mo.md(
                        'Within each user, rows must be in chronological order: row i precedes row j when i < j. Sort by user and timestamp. The model cannot detect shuffled rows; it silently learns the wrong sequence.'
                    ),
                    '3. Does the service that runs a fitted `HSTU` need PyTorch?': mo.md(
                        'No. A fitted `skrecsys.nn` model stores its parameters as numpy arrays; it scores, pickles and unpickles without torch. Only `fit` and `partial_fit` need it.'
                    ),
                    "4. Why can't you compare this notebook's HR@10 with the NDCG@10 in the README leaderboard?": mo.md(
                        'They answer different questions under different protocols: one held-out next item per user here, 10 random ratings per user (`ua`) there. Numbers from different protocols are not comparable.'
                    ),
                    '5. Why does this notebook break timestamp ties at random before holding out the last interaction?': mo.md(
                        'MovieLens users rate in batches, many movies in the same second, and the raw data orders ties by item id. The "last" item would then be the highest id of the last batch — a pattern a sequential model can learn without learning anything about what users want next.'
                    ),
                    '6. When is EASE still the better choice than HSTU?': mo.md(
                        'When the task is "what else" rather than "what next", when training time or a GPU is not available, or as a baseline to beat. It fits in milliseconds and is strong on set-based tasks.'
                    ),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
