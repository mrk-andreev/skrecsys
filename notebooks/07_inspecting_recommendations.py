import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import json
    import time

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from skrecsys.compose import (
        Cascade,
        ConcatFeatures,
        GeneratorScores,
        InteractionCounts,
        KnownUser,
        PointwiseRanker,
        Switch,
    )
    from skrecsys.datasets import fetch_movielens_100k
    from skrecsys.inspection import explain, trace
    from skrecsys.metrics import evaluate_recommender, ndcg_at_k, recall_at_k
    from skrecsys.model_selection import ColdStartSplit
    from skrecsys.recommendation import EASE, BayesianPersonalizedRanking, MostPopularRecommender

    N_RETRIEVED = 100
    K = 10
    #: Warm test users whose held-out items are traced through the pipeline in section 6.
    N_DIAGNOSED = 200


@app.cell
def _():
    mo.md(r"""
    # 7. Inspecting recommendations

    A pipeline from [notebook 03](03_candidates_and_ranking.py) returns two arrays: the
    items and their scores. When a user complains, or a metric drops, those arrays do not
    say what went wrong. Two questions come up again and again:

    1. **Why was this item recommended, exactly this way?** Which branch served the user,
       what each generator retrieved, what the ranker saw, what the business rules did —
       and which of the user's own interactions made a model score the item as it did.
    2. **When are the recommendations good, and when are they bad?** And when a relevant
       item is missing, which stage lost it?

    `skrecsys.inspection` answers both from the pipeline itself, without changing it:

    - `trace()` records every stage of the `recommend` calls made inside a `with` block;
    - `explain()` turns a trace into a verdict per item — why it was served, or where the
      pipeline let it go.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 1. The pipeline under inspection

    The split and the pipeline follow notebook 03. `ColdStartSplit` holds out 10% of users
    entirely and the latest 20% of everyone else's interactions. Known users get a cascade:
    EASE and popularity propose 100 candidates, and a logistic ranker reorders them using
    both generators' scores and how active the user is. Users the model has never seen get
    the most popular items.
    """)
    return


@app.cell
def _():
    _ratings = fetch_movielens_100k(as_frame=True).frame.sort_values("timestamp", kind="stable")
    X = _ratings[["user_id", "item_id"]].to_numpy()
    _train, _test = next(ColdStartSplit(cold_users=0.1, test_size=0.2, random_state=0).split(X))
    X_train, X_test = X[_train], X[_test]
    _cold = ~np.isin(X_test[:, 0], X_train[:, 0])
    X_warm, X_cold = X_test[~_cold], X_test[_cold]
    warm_users = np.unique(X_warm[:, 0])
    cold_users = np.unique(X_cold[:, 0])
    return X_train, X_warm, cold_users, warm_users


@app.cell
def _(X_train, X_warm):
    pipeline = Switch(
        KnownUser(),
        Cascade(
            [EASE(), MostPopularRecommender()],
            ConcatFeatures([GeneratorScores(n_generators=2), InteractionCounts("user")]),
            PointwiseRanker(LogisticRegression(max_iter=2000)),
            n_retrieved=N_RETRIEVED,
        ),
        MostPopularRecommender(),
    ).fit(X_train)
    _scores = evaluate_recommender(pipeline, X_warm, metrics=[ndcg_at_k, recall_at_k], k=K)
    mo.md(
        f"On warm test users the pipeline reaches NDCG@{K} = {_scores[f'ndcg@{K}']:.3f} and "
        f"recall@{K} = {_scores[f'recall@{K}']:.3f}. One number per pipeline — the rest of "
        "the notebook looks inside it."
    )
    return (pipeline,)


@app.cell
def _():
    mo.md(r"""
    ## 2. A trace, stage by stage

    Anything called inside `with trace() as t:` is recorded, however deeply the estimators
    nest. The trace lives in a context variable, not on the model, so the model clones and
    pickles as before, and a `recommend` outside the block costs nothing extra.

    `t[user]` is everything the call did for one query. Each step names the stage it comes
    from by a path: the estimators and parts it is nested in, such as
    `Switch/on_true/Cascade/ease/EASE`.
    """)
    return


@app.cell
def _(cold_users, pipeline, warm_users):
    warm_user, cold_user = int(warm_users[0]), int(cold_users[0])
    with trace() as first_trace:
        pipeline.recommend([warm_user, cold_user], n_recommendations=K)
    mo.vstack(
        [
            mo.md(f"**Warm user {warm_user}**"),
            mo.plain_text(str(first_trace[warm_user])),
            mo.md(f"**Cold user {cold_user}**"),
            mo.plain_text(str(first_trace[cold_user])),
        ]
    )
    return first_trace, warm_user


@app.cell
def _():
    mo.md(r"""
    The two users took different routes. The cold user went `on_false` and got the
    popularity list. The warm user went `on_true` into the cascade: each generator
    retrieved 100 items, the lists were merged into the `union` the ranker saw, the ranker
    scored every candidate, and the cascade served its top 10.

    The steps are typed, so the numbers are there to compute with. Here is where the
    served items stood at each stage:
    """)
    return


@app.cell
def _(first_trace, warm_user):
    _steps = first_trace[warm_user]
    _by_source = {step.source: step for step in _steps.candidates}
    (_ranker,) = _steps.ranker
    _ranked = _ranker.ranked().tolist()
    _final = _steps.final
    served_table = pd.DataFrame(
        {
            "item": _final.items,
            "score": _final.scores.round(4),
            "EASE rank": [_by_source["ease"].rank_of(i) for i in _final.items],
            "popularity rank": [
                _by_source["mostpopularrecommender"].rank_of(i) for i in _final.items
            ],
            "ranker rank": [_ranked.index(i) for i in _final.items],
        }
    )
    _promoted = int((served_table["EASE rank"] > served_table["ranker rank"]).sum())
    mo.vstack(
        [
            served_table,
            mo.md(
                f"Ranks are 0-based. The ranker placed {_promoted} of the {K} served items "
                "higher than EASE did: that is what the second stage is for."
            ),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 3. What a trace costs

    `level="full"`, the default, also records the feature matrix each ranker saw and why
    each model scored what it served (section 4). That is what a debugging session wants,
    and too much for a busy service. `level="decisions"` keeps every route, candidate list,
    ranker score and served list, and drops the two heavy parts.

    `sample=` records a fraction of users. The choice hashes the user id with a hash that
    is stable across processes, so the same users are traced on every request and every
    restart, and each is traced through every stage or not at all.
    """)
    return


@app.cell
def _(pipeline, warm_users):
    def _timed(**kwargs):
        _start = time.perf_counter()
        if kwargs:
            with trace(**kwargs) as _t:
                pipeline.recommend(warm_users, n_recommendations=K)
        else:
            _t = None
            pipeline.recommend(warm_users, n_recommendations=K)
        return time.perf_counter() - _start, _t

    _runs = {
        "no trace": {},
        'level="decisions", sample=0.1': {"level": "decisions", "sample": 0.1},
        'level="decisions"': {"level": "decisions"},
        'level="full"': {"level": "full"},
    }
    _rows = []
    for _name, _kwargs in _runs.items():
        _seconds, _t = _timed(**_kwargs)
        _rows.append(
            {
                "recording": _name,
                "seconds": round(_seconds, 3),
                "users traced": 0 if _t is None else len(_t.queries),
            }
        )
    _, _again = _timed(level="decisions", sample=0.1)
    _, _first = _timed(level="decisions", sample=0.1)
    cost_table = pd.DataFrame(_rows)
    mo.vstack(
        [
            mo.md(f"`recommend` for all {len(warm_users)} warm users:"),
            cost_table,
            mo.md(
                "Sampling twice traced the same users: "
                f"**{sorted(_again.queries) == sorted(_first.queries)}**."
            ),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 4. Why this item: `explain`

    `explain(model, users)` calls `recommend` under a full trace and explains every served
    item. For each item it gives the route, where each generator placed the item, what
    the ranker made of it, and the **reasons**: which of the user's interactions made each
    model score the item as it did.
    """)
    return


@app.cell
def _(pipeline, warm_user):
    served_explanations = explain(pipeline, [warm_user], n_recommendations=K, n_reasons=5)
    top = served_explanations[0]
    mo.plain_text(str(top))
    return (top,)


@app.cell
def _(first_trace, top, warm_user):
    (_ease_reasons,) = [r for r in top.reasons if r.path.endswith("EASE")]
    _ease_score = next(
        float(step.scores[step.rank_of(top.item)])
        for step in first_trace[warm_user].candidates
        if step.source == "ease"
    )
    _shown = sum(weight for _, weight in _ease_reasons.reasons)
    mo.md(
        f"""
    Each reason is a movie the user rated, with its share of the score. For EASE the
    shares are **exact**. An item-to-item model scores
    $\\text{{score}}(u, i) = \\sum_j x_{{uj}} W_{{ji}}$, so every rated movie $j$ owns one
    term. The five largest terms add up to {_shown:.4f}. The other
    {_ease_reasons.rest:.4f} is spread over the rest of the history. Together they give
    {_shown + _ease_reasons.rest:.4f}, the EASE score the trace recorded
    ({_ease_score:.4f}). ItemKNN, BM25, RP3Beta and SLIM are exact in the same way.

    The ranker's line lists its largest **contributions**. The ranker is a logistic
    regression, so these are coefficient × feature in log-odds, plus the bias. For
    CatBoost, XGBoost and LightGBM rankers these are the libraries' own SHAP values. A
    ranker that cannot decompose its score, such as `HistGradientBoostingClassifier`,
    reports none.
    """
    )
    return


@app.cell
def _(top):
    _contributions = pd.DataFrame(
        {"feature": list(top.ranker.contributions), "log-odds": list(top.ranker.contributions.values())}
    )
    alt.Chart(_contributions, title=f"Why the ranker scored item {top.item} as it did").mark_bar().encode(
        x=alt.X("log-odds:Q"),
        y=alt.Y("feature:N", sort="-x", title=None),
        color=alt.condition(alt.datum["log-odds"] > 0, alt.value("#4c78a8"), alt.value("#e45756")),
    ).properties(height=160)
    return


@app.cell
def _():
    mo.md(r"""
    A factor model has no such decomposition. BPR scores by a dot product of a user vector
    and an item vector, and no single rated movie owns a share of it. Its reasons are the
    rated movies whose vectors point most like the item's. They are useful as a "because
    you watched…" line, but they are marked **approximate** and do not add up to the
    score:
    """)
    return


@app.cell
def _(X_train, top, warm_user):
    _bpr = BayesianPersonalizedRanking(n_factors=32, random_state=0).fit(X_train)
    (_bpr_top,) = [
        e
        for e in explain(_bpr, [warm_user], items=[top.item], n_recommendations=K)
        if e.item == top.item
    ]
    _ease_movies = {m for r in top.reasons if r.exact for m, _ in r.reasons}
    _bpr_movies = [m for r in _bpr_top.reasons for m, _ in r.reasons]
    _shared = [m for m in _bpr_movies if m in _ease_movies]
    mo.vstack(
        [
            mo.plain_text(str(_bpr_top)),
            mo.md(
                f"{len(_shared)} of BPR's {len(_bpr_movies)} reasons "
                f"({', '.join(map(str, _shared)) or 'none'}) are also among EASE's exact "
                "top five. The two models were trained differently, and still point to "
                "some of the same movies behind this recommendation."
            ),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 5. Why not this item

    The more useful question is often the opposite one. Pass `items=` the items you
    expected, and each gets a **status**: the first stage, from the end of the pipeline
    back, that let it go.

    | status | the item… |
    |---|---|
    | `served` | was recommended |
    | `unknown_item` | was not in the training data |
    | `excluded` | was removed by a filter before scoring, named in `excluded_by` |
    | `not_retrieved` | was never among the candidates, or was cut when the lists merged |
    | `ranked_out` | was a candidate, but the ranker put it below the top 10 |
    | `dropped_by_postprocess` | was removed by the business rules |

    The natural items to ask about are the user's own held-out ones: what they went on to
    watch.
    """)
    return


@app.cell
def _(X_warm, pipeline, warm_user):
    _held_out = X_warm[X_warm[:, 0] == warm_user, 1]
    _found = explain(pipeline, [warm_user], items=_held_out, n_recommendations=K)
    _wanted = set(_held_out.tolist())
    user_verdicts = pd.DataFrame(
        [
            {"item": e.item, "status": e.status, "detail": e.detail}
            for e in _found
            if e.item in _wanted
        ]
    )
    _ranked_out = next(e for e in _found if e.status == "ranked_out" and e.item in _wanted)
    mo.vstack(
        [
            mo.md(f"User {warm_user} watched {len(_held_out)} movies after the split:"),
            user_verdicts["status"].value_counts().rename("movies").to_frame(),
            mo.md("One of them, a candidate the ranker put too low:"),
            mo.plain_text(str(_ranked_out)),
        ]
    )
    return


@app.cell
def _():
    mo.md(rf"""
    ## 6. Where relevant items are lost

    One user is an anecdote. Tracing the held-out items of {N_DIAGNOSED} warm users
    shows what a single metric hides: *which stage* fails, and for whom.

    This is diagnosis, not model selection: nothing below chooses a model on the test set.
    A fix it suggests is still tuned on validation data, as in notebook 02.
    """)
    return


@app.cell
def _(X_train, X_warm, pipeline, warm_users):
    _rng = np.random.default_rng(0)
    _users = _rng.choice(warm_users, size=min(N_DIAGNOSED, len(warm_users)), replace=False)
    _history = pd.Series(X_train[:, 0]).value_counts()
    _rows = []
    for _user in _users.tolist():
        _held_out = X_warm[X_warm[:, 0] == _user, 1]
        _wanted = set(_held_out.tolist())
        _found = explain(pipeline, [_user], items=_held_out, n_recommendations=K, n_reasons=1)
        _rows.extend(
            {"user": _user, "status": _e.status, "history": int(_history[_user])}
            for _e in _found
            if _e.item in _wanted
        )
    verdicts = pd.DataFrame(_rows)
    verdicts["history"] = pd.qcut(verdicts["history"], 3, labels=["short", "medium", "long"])
    status_share = verdicts["status"].value_counts(normalize=True)
    mo.vstack(
        [
            mo.md(f"{len(verdicts):,} held-out interactions of {len(_users)} users:"),
            (status_share * 100).round(1).rename("% of relevant items").to_frame(),
        ]
    )
    return status_share, verdicts


@app.cell
def _(verdicts):
    _by_history = (
        verdicts.groupby("history", observed=True)["status"]
        .value_counts(normalize=True)
        .rename("share")
        .reset_index()
    )
    alt.Chart(
        _by_history, title="What became of relevant items, by the user's history length"
    ).mark_bar().encode(
        x=alt.X("share:Q", axis=alt.Axis(format="%"), title=None),
        y=alt.Y("history:N", sort=["short", "medium", "long"], title="history"),
        color=alt.Color("status:N", sort=["served", "ranked_out", "not_retrieved", "unknown_item"]),
        order=alt.Order("status:N"),
    ).properties(height=140)
    return


@app.cell
def _(status_share, verdicts):
    _served_by_history = (
        verdicts.assign(served=verdicts["status"] == "served")
        .groupby("history", observed=True)["served"]
        .mean()
    )
    _lost = status_share.get("not_retrieved", 0.0)
    _ranked_out = status_share.get("ranked_out", 0.0)
    _served = status_share.get("served", 0.0)
    mo.md(
        f"""
    The diagnosis is plain. **{_lost:.0%}** of the relevant items never reach the ranker,
    and **{_ranked_out:.0%}** reach it and are ranked below the top {K}. Only
    **{_served:.0%}** are served. Most of what the pipeline misses, it misses at
    retrieval. A better ranker would at best move items within the
    {_ranked_out + _served:.0%} it sees. A wider or better first stage (a larger
    `n_retrieved`, another generator) goes after the larger share. That is a hypothesis
    to test on validation data, not a conclusion.

    The bars split the same verdicts by history length. Users with a short history get
    {_served_by_history["short"]:.0%} of their relevant items served, and users with a long
    one get {_served_by_history["long"]:.0%}. The share is not the whole story: a long
    history also holds out more items per user. Where the mix changes, the pipeline
    fails differently for different users. That is the second question from the start of
    the notebook, answered per stage rather than with one average.
    """
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 7. Keeping traces

    Every step, trace and explanation has a `to_dict()` with plain JSON values: numpy
    scalars become Python ones, NaN becomes `null`. So a sampled trace can go to a log or
    a table, next to the request it explains:
    """)
    return


@app.cell
def _(first_trace, top, warm_user):
    _record = first_trace[warm_user].to_dict()
    _kinds = [step["step"] for step in _record["steps"]]
    mo.vstack(
        [
            mo.md(f"The trace of user {warm_user} is {len(_kinds)} steps: `{', '.join(dict.fromkeys(_kinds))}`."),
            mo.plain_text(json.dumps(top.to_dict(), indent=1)[:900] + "\n..."),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## Takeaways

    - `with trace() as t:` records every stage of a pipeline. `t[user]` shows the route,
      the candidates of each generator, the ranker's scores, what postprocess changed and
      what was served. The model itself is untouched.
    - `level="decisions"` with `sample=` is cheap enough to leave on. `level="full"` adds
      feature matrices and reasons for debugging.
    - `explain` says why an item was served: its position at each stage, the ranker's
      feature contributions, and the rated movies behind a model's score. Those are exact
      for item-to-item models and approximate for factor models.
    - Asked about items that were **not** served, it names the stage that lost them.
      Aggregated over many users, that shows whether retrieval or ranking is the
      bottleneck, and for whom.
    """)
    return


@app.cell
def _():
    mo.vstack(
        [
            mo.md("## Check yourself\n\nTry to answer each question before opening it."),
            mo.accordion(
                {
                    "1. Does tracing change what `recommend` returns, or what gets pickled?": mo.md(
                        "No. The trace lives in a context variable, not on the estimator: `recommend` returns the same items and scores, and the model clones and pickles as before. Outside a `with trace()` block the only cost is a check that nobody is listening."
                    ),
                    '2. Why does `sample=0.1` trace the same users on every call, and why does that matter?': mo.md(
                        "Users are chosen by a hash of their id that is stable across processes, not by a random draw. So a sampled user is traced on every request and after every restart, and their traces can be followed over time. Each sampled user is also traced through every stage, never half-way."
                    ),
                    "3. EASE's reasons add up to its score and BPR's do not. Why?": mo.md(
                        "EASE scores by a sum over the user's history, $\\sum_j x_{uj} W_{ji}$, so every rated item owns one term. BPR scores by a dot product of learned vectors, which no rated item owns. Its reasons are the rated items most similar to the target, marked `exact=False`."
                    ),
                    '4. An expected item has status `not_retrieved`. Would a better ranker help?': mo.md(
                        "No. The ranker only reorders the candidates it is given, and this item was never among them, or was cut when the generators' lists merged. The fix is in the first stage: a larger `n_retrieved`, or a generator that finds such items."
                    ),
                    "5. `explain` says `excluded` with `excluded_by='exclude_seen'`. What happened?": mo.md(
                        "The user had already interacted with the item during training, and `recommend` removes seen items by default. Nothing ranked it low; it was never eligible."
                    ),
                    "6. Why is section 6's diagnosis on test users not model selection, and what should be done with it?": mo.md(
                        "It chooses nothing. It measures where the fixed pipeline loses relevant items. Any change it suggests, such as a larger `n_retrieved`, is a hypothesis to tune and compare on validation data, as in notebook 02, before touching the test set."
                    ),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
