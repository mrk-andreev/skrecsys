import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import pickle
    import time
    from itertools import pairwise

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd

    from skrecsys.datasets import fetch_movielens_100k
    from skrecsys.indexing import HNSW
    from skrecsys.metrics import evaluate_recommender, ndcg_at_k
    from skrecsys.recommendation import (
        EASE,
        BayesianPersonalizedRanking,
        ItemKNNRecommender,
    )

    N_BATCHES = 10


@app.cell
def _():
    mo.md(r"""
    # From notebook to production

    The previous notebooks fitted a model once and evaluated it. A live recommender is
    different: interactions keep arriving, new users and items appear, the model is
    serialized and served by another process, and every request needs an answer in
    milliseconds. This notebook covers the parts of skrecsys built for that:

    1. the lifecycle: trainer, updater and service;
    2. shipping a model as a pickle;
    3. keeping it fresh with `partial_fit`;
    4. serving-time controls: `exclude_interactions` and `candidates`;
    5. approximate retrieval with a vector index.

    Throughout, MovieLens 100K is replayed **in timestamp order**, as the log a service
    would have seen.
    """)
    return


@app.cell
def _():
    _movielens = fetch_movielens_100k(as_frame=True)
    _order = np.argsort(_movielens.timestamps.to_numpy(), kind="stable")
    log = _movielens.data.to_numpy()[_order]
    titles = _movielens.item_info.set_index("item_id")["title"]
    comedies = _movielens.item_info.loc[_movielens.item_info["Comedy"], "item_id"].to_numpy()
    return comedies, log, titles


@app.cell
def _():
    mo.md(r"""
    ## 1. Three jobs, one versioned model

    A deployment has three jobs, and they should not share a live object:

    - the **trainer** calls `fit` on a window of the interaction log, on a schedule;
    - the **updater** calls `partial_fit` on micro-batches of new events, between refits;
    - the **service** answers requests with `recommend`.

    What passes between them is a pickled, versioned model. The service never updates the
    model it is serving: the updater works on its own copy and publishes a new version,
    which the service swaps to in one step.
    """)
    return


@app.cell
def _():
    mo.mermaid(
        """
        flowchart LR
            log[("Interaction log")] -->|nightly| trainer["Trainer<br/>fit(window)"]
            trainer -->|publish v1| store[("Model store<br/>v1, v2, ...")]
            stream[["Event stream"]] -->|every few minutes| updater["Updater<br/>partial_fit(batch)"]
            store -->|load latest| updater
            updater -->|publish v(n+1)| store
            store -->|every new version| service(["Service<br/>recommend"])
        """
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 2. Shipping a model

    Every fitted skrecsys estimator pickles, composites and indexes included. A fitted
    model holds only numpy and scipy arrays, so the service needs skrecsys and its four
    dependencies, nothing else — even a `skrecsys.nn` model unpickles and scores without
    torch.
    """)
    return


@app.cell
def _(log):
    _model = EASE().fit(log)
    _payload = pickle.dumps(_model)
    _restored = pickle.loads(_payload)  # noqa: S301 -- we just made this pickle ourselves
    _users = _model.user_ids_[:100]
    _same = np.array_equal(
        _model.recommend(_users, n_recommendations=10)[0],
        _restored.recommend(_users, n_recommendations=10)[0],
    )
    mo.md(f"""
    `pickle.dumps(EASE().fit(log))` is **{len(_payload) / 2**20:.1f} MB** — mostly the
    dense {_model.n_items_:,} × {_model.n_items_:,} item–item matrix — and the reloaded
    model returns identical recommendations: **{_same}**.

    As with any pickle, load only models your own trainer produced. Record next to each
    version the window of the log it was fitted on; lists from two versions should never
    be mixed, because their scores are on different scales.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 3. Staying fresh with `partial_fit`

    `partial_fit(X)` adds a batch of interactions to a fitted model. Unlike scikit-learn's
    incremental estimators, a batch may bring **new users and new items**, and the model's
    vocabularies grow to admit them.

    To see what that buys, we fit ItemKNN on the first half of the log, then replay the
    second half in batches. Before each batch arrives, two versions of the model are asked
    to predict it:

    - **static** — fitted once and never updated;
    - **static + filter** — the same model, but each request excludes what the user did
      since the fit, with `exclude_interactions=` (section 4);
    - **incremental** — updated with `partial_fit` after every batch.

    The middle one matters for a fair comparison. The static model does not know what a
    user watched in the last batches, so it may recommend those movies again, and they can
    never be hits. Filtering them is cheap; the gap between it and the incremental model is
    what learning from the new data buys on top.

    All are scored on the users the static model knows, so the comparison is like for
    like, and we track separately how many of the batch's interactions come from users each
    model can serve at all.
    """)
    return


@app.cell
def _(log):
    _bounds = np.linspace(len(log) // 2, len(log), N_BATCHES + 1).astype(int)
    _static = ItemKNNRecommender().fit(log[: _bounds[0]])
    incremental = ItemKNNRecommender().fit(log[: _bounds[0]])
    _rows = []
    for _step, (_start, _stop) in enumerate(pairwise(_bounds)):
        _batch = log[_start:_stop]
        _shared = _batch[np.isin(_batch[:, 0], _static.user_ids_)]
        _since_fit = log[_bounds[0] : _start]
        for _name, _model, _exclude in (
            ("static", _static, None),
            ("static + filter", _static, _since_fit),
            ("incremental", incremental, None),
        ):
            _rows.append(
                {
                    "batch": _step + 1,
                    "model": _name,
                    "NDCG@10": evaluate_recommender(
                        _model, _shared, metrics=[ndcg_at_k], k=10, exclude_interactions=_exclude
                    )["ndcg@10"],
                    "servable share": np.isin(_batch[:, 0], _model.user_ids_).mean(),
                }
            )
        incremental.partial_fit(_batch)
    replay = pd.DataFrame(_rows)
    _mean = replay.groupby("model")["NDCG@10"].mean()
    _base = alt.Chart(replay).encode(
        x=alt.X("batch:O", title="next batch"),
        color=alt.Color("model:N"),
        tooltip=["batch", "model", alt.Tooltip("NDCG@10:Q", format=".4f")],
    )
    mo.vstack(
        [
            alt.hconcat(
                _base.mark_line(point=True)
                .encode(y="NDCG@10:Q")
                .properties(width=300, height=220, title="Predicting the next batch"),
                _base.mark_line(point=True)
                .encode(y=alt.Y("servable share:Q", axis=alt.Axis(format="%")))
                .properties(width=300, height=220, title="Interactions from known users"),
            ),
            mo.md(f"""
            On the very first batch the three are the same model. Over all batches the
            mean NDCG@10 is {_mean["static"]:.4f} static, {_mean["static + filter"]:.4f}
            with the filter and {_mean["incremental"]:.4f} incremental.
            {_mean["static + filter"] - _mean["static"]:+.4f} comes from no longer
            recommending what users just watched, and
            {_mean["incremental"] - _mean["static + filter"]:+.4f} from learning on the new
            interactions — who likes what now, and which movies go together. The
            incremental model can also serve the users who arrived after the fit, which the
            static model must hand to a fallback.
            """),
        ]
    )
    return (incremental,)


@app.cell
def _(incremental, log):
    _refit = ItemKNNRecommender().fit(log)
    _users = _refit.user_ids_
    _knn_exact = np.array_equal(
        incremental.recommend(_users, n_recommendations=10)[0],
        _refit.recommend(_users, n_recommendations=10)[0],
    )
    _timings = []
    for _name, _factory in (("ItemKNN", ItemKNNRecommender), ("EASE", EASE)):
        _model = _factory().fit(log[: len(log) - len(log) // N_BATCHES])
        _t0 = time.perf_counter()
        _model.partial_fit(log[len(log) - len(log) // N_BATCHES :])
        _partial = time.perf_counter() - _t0
        _t0 = time.perf_counter()
        _factory().fit(log)
        _full = time.perf_counter() - _t0
        _timings.append(
            {"model": _name, "partial_fit (ms)": _partial * 1e3, "full fit (ms)": _full * 1e3}
        )
    mo.vstack(
        [
            mo.md(f"""
            **Is it the same model?** For ItemKNN, `partial_fit` is *exact*: after all
            the batches, the incremental model recommends exactly what a fresh `fit` on
            the whole log does — **{_knn_exact}**. Most classical models are exact in
            this sense; the factor models and neural ones are *warm starts* instead:
            they continue training from where they stopped and drift from what a fresh
            fit would give.

            | model | after `partial_fit` | refit needed to |
            |---|---|---|
            | `MostPopularRecommender`, `ItemKNNRecommender`, `BM25Recommender`, `RP3Beta`, `EASE` | exact | forget |
            | `SLIMElasticNet`, `AlternatingLeastSquares`, `BayesianPersonalizedRanking`, `skrecsys.nn` | warm start | forget, correct drift |

            **Is it cheaper?** Timing the last batch against a full refit:
            """),
            pd.DataFrame(_timings).set_index("model").round(1),
            mo.md(f"""
            On a dataset this small a refit takes milliseconds, and `partial_fit` is
            {"no faster" if all(_t["partial_fit (ms)"] >= _t["full fit (ms)"] for _t in _timings) else "barely faster"}:
            it pays fixed costs per call (growing the vocabularies, recomputing
            every neighbour row the batch reaches). It pays off when a refit is
            expensive — a large log, a large catalog, a neural model — so call it on
            micro-batches every few minutes, not once per event.

            `partial_fit` only ever adds. Forgetting old interactions, honouring a
            deletion request or changing a hyperparameter all need a scheduled `fit` on
            a window of the log.
            """),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 4. Serving-time controls

    Two arguments of `recommend` handle what the model cannot know when it was fitted.

    - `exclude_interactions=` removes user–item pairs **per request** — typically the
      events a user produced after the serving version was fitted. `exclude_seen` only
      filters what the model saw; without this, a movie the user watched five minutes ago
      stays recommendable until the next update.
    - `candidates=` restricts the ranking to a set of items **for the whole call** —
      business rules such as "in stock", "available in this region" or "this page is
      about comedies".

    Both are **request-time** controls. A batch job that precomputes every user's list
    overnight cannot know what a user will do in the next hour, or which region and page
    a later request comes from; it can only store a longer list and filter it when the
    request arrives. Rules that hold for every request, such as licensing, belong in the
    model itself, with `Cascade(postprocess=...)` from
    [notebook 03](03_candidates_and_ranking.py).
    """)
    return


@app.cell
def _(comedies, incremental, titles):
    _user = incremental.user_ids_[0]
    _plain, _ = incremental.recommend([_user], n_recommendations=5)
    _just_watched = np.array([[_user, _item] for _item in _plain[0][:2]])
    _fresh, _ = incremental.recommend(
        [_user], n_recommendations=5, exclude_interactions=_just_watched
    )
    _funny, _ = incremental.recommend([_user], n_recommendations=5, candidates=comedies)
    pd.DataFrame(
        {
            "recommend(...)": [titles[_i] for _i in _plain[0]],
            "after watching the top 2": [titles[_i] for _i in _fresh[0]],
            "candidates=comedies": [titles[_i] for _i in _funny[0]],
        },
        index=pd.RangeIndex(1, 6, name=f"user {_user}, rank"),
    )
    return


@app.cell
def _():
    mo.md(r"""
    With `exclude_interactions`, the call still returns exactly five fresh items — no
    over-fetching and filtering afterwards. Users the model has never seen still raise
    `ValueError`; serve them with a fallback, as `Switch` does in
    [notebook 03](03_candidates_and_ranking.py).

    ## 5. Approximate retrieval with an index

    `recommend` scores every item in the catalog for every query. That is exact, and on a
    catalog of a few thousand items it takes a fraction of a millisecond per user. On
    millions of items it is the bottleneck, and an **index** helps:

    - `index="hnsw"` or `HNSW(m=..., ef_search=...)` builds a navigable graph over the item
      vectors at fit time; `recommend` walks the graph and touches a few hundred items
      instead of all of them. The price is exactness: the result is the best items the
      walk *found*.
    - `QuantizedFlatIndex(bits=...)` scans narrow codes and rescores a shortlist with the
      original vectors, so its scores are exact.

    `ef_search` is the recall-for-latency dial. Below, BPR is fitted with HNSW at several
    settings and compared with the exact top 10. MovieLens has only 1,682 movies — below
    the default `min_index_size` of 4,096, under which the index is skipped as not worth
    it — so we force it with `min_index_size=1` to see its behaviour.
    """)
    return


@app.cell
def _():
    index_button = mo.ui.run_button(label="Fit BPR with five index settings (about 10 s)")
    index_button
    return (index_button,)


@app.cell
def _(index_button, log):
    mo.stop(
        not (index_button.value or mo.app_meta().mode == "script"),
        mo.md("*Press the button to measure the index.*"),
    )

    def _timed_recommend(model, users):
        _best = np.inf
        for _ in range(5):
            _t0 = time.perf_counter()
            _items, _ = model.recommend(users, n_recommendations=10)
            _best = min(_best, time.perf_counter() - _t0)
        return _items, _best

    _exact_model = BayesianPersonalizedRanking(random_state=0).fit(log)
    _users = _exact_model.user_ids_
    _exact, _exact_time = _timed_recommend(_exact_model, _users)
    _rows = [{"ef_search": "exact", "recall of exact top 10": 1.0, "ms per call": _exact_time * 1e3}]
    for _ef in mo.status.progress_bar([10, 20, 40, 80, 160], title="Fitting"):
        _model = BayesianPersonalizedRanking(
            random_state=0, index=HNSW(ef_search=_ef, min_index_size=1)
        ).fit(log)
        _items, _seconds = _timed_recommend(_model, _users)
        _recall = np.mean(
            [len(set(_a) & set(_b)) / 10 for _a, _b in zip(_items, _exact, strict=True)]
        )
        _rows.append(
            {"ef_search": str(_ef), "recall of exact top 10": _recall, "ms per call": _seconds * 1e3}
        )
    index_results = pd.DataFrame(_rows).set_index("ef_search")
    return (index_results,)


@app.cell
def _(index_results):
    mo.vstack(
        [
            index_results.round(4),
            mo.md(f"""
            One call ranks every user in the dataset. Recall climbs to the exact answer as
            `ef_search` grows, and so does the time. On this tiny catalog the exact scan
            is {"faster than every index setting" if index_results["ms per call"].idxmin() == "exact" else "competitive"}
            — the graph only pays off when the catalog is large enough that scanning it
            is the expensive part. Measure on your own model and data before switching
            one on; the benchmark harness in `benchmarks/indexes.py` does exactly that.
            """),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## Takeaways

    - Separate the trainer, the updater and the service; pass **pickled, versioned**
      models between them and never mutate a model while it serves.
    - `partial_fit` brings in new users, items and interactions between refits. For most
      classical models it is exact; for factor and neural models it is a warm start. A
      scheduled `fit` is still needed to forget and to reset drift.
    - At request time, `exclude_interactions` removes what the user just did and
      `candidates` applies business rules; unknown users need a fallback.
    - Indexes trade exactness for speed on large catalogs; measure before using one.

    Next: [notebook 05](05_sequential_and_neural.py) looks at models that read the
    *order* of a user's history, and at the neural models in `skrecsys.nn`.
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
                    '1. Why should the service never call `partial_fit` on the model it is serving?': mo.md(
                        '`partial_fit` updates the model in place, rebinding vocabularies and matrices one after another. A concurrent `recommend` can read a mix of two versions. The updater works on its own copy and publishes a new version, which the service swaps to in one step.'
                    ),
                    '2. For which models is `partial_fit` exact, and what does "warm start" mean for the others?': mo.md(
                        'Exact (identical to `fit` on all the data): MostPopular, ItemKNN, BM25, RP3Beta, EASE. Warm start (SLIM, ALS, BPR, `skrecsys.nn`): training continues from the fitted parameters and gradually drifts from what a fresh fit would give.'
                    ),
                    '3. Name three things `partial_fit` cannot do, which need a scheduled `fit`.': mo.md(
                        'Forget old interactions (time windows, deletion requests), apply a hyperparameter change, and, for warm-start models, correct drift.'
                    ),
                    '4. A user watched a movie two minutes ago, and the serving model is an hour old. How do you avoid recommending that movie?': mo.md(
                        "Pass the user's recent events as `recommend(..., exclude_interactions=recent_pairs)`. `exclude_seen` only knows what the model was fitted on."
                    ),
                    '5. Why can a batch job not apply per-request business rules with `candidates=`?': mo.md(
                        '`candidates=` is shared by the whole call, and a batch job computes lists before any request arrives. Per-request eligibility (stock, region, page) needs realtime serving.'
                    ),
                    '6. On MovieLens, HNSW was no clear win over exact scoring. When would you use it anyway?': mo.md(
                        'When the catalog is large, hundreds of thousands to millions of items, and a full scan is the bottleneck. Below `min_index_size` skrecsys skips the index anyway. Measure recall and latency on your own model and data first.'
                    ),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
