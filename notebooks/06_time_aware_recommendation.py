import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier

    from skrecsys.compose import (
        Cascade,
        ConcatFeatures,
        GeneratorScores,
        JoinDynamicFeatures,
        PointwiseRanker,
    )
    from skrecsys.datasets import fetch_movielens_100k
    from skrecsys.metrics import evaluate_recommender, ndcg_at_k
    from skrecsys.recommendation import EASE, ItemKNNRecommender, MostPopularRecommender, RP3Beta
    from skrecsys.tune import AutoTune

    DAY = 86_400  # seconds; MovieLens timestamps are Unix seconds
    HORIZON = 14 * DAY  # how far into the future each evaluation looks
    LATE_SHARE = 0.15  # share of events that arrive late in the simulated feature store

    class GenreAffinity:
        """How much of a user's history, as of a time, falls in an item's genres.

        The callback of a ``JoinDynamicFeatures("user-item-time", ...)``: given rows of
        ``(user, item, time)`` it returns one column, the share of the user's genre
        mass that overlaps the item's genres. ``known_at`` holds the time each event
        became *known* to the feature store; an event counts for time ``t`` only when
        ``known_at < t``. ``leaky=True`` ignores ``t`` and uses the whole log.
        """

        def __init__(self, users, items, known_at, genres, *, leaky=False):
            self.genres = genres
            self.leaky = leaky
            self.history = {}
            for _user in np.unique(users):
                _mine = users == _user
                _order = np.argsort(known_at[_mine], kind="stable")
                _cumulative = np.cumsum(genres[items[_mine][_order]], axis=0)
                self.history[_user] = (
                    known_at[_mine][_order],
                    np.vstack([np.zeros(genres.shape[1]), _cumulative]),
                )

        def __call__(self, keys):
            out = np.zeros((len(keys), 1))
            for _row, (_user, _item, _time) in enumerate(keys):
                _times, _cumulative = self.history[int(_user)]
                _latest = self.leaky or np.isnan(_time)
                _n = len(_times) if _latest else np.searchsorted(_times, _time, side="left")
                _profile = _cumulative[_n]
                _total = _profile.sum()
                out[_row, 0] = _profile @ self.genres[int(_item)] / _total if _total else 0.0
            return out


@app.cell
def _():
    mo.md(r"""
    # Time-aware recommendation

    A recommender is always fitted on the past and used on the future. Every notebook in
    this series touched on that: [02](02_model_selection_and_tuning.py) showed that a
    random split overestimates quality, [04](04_production.py) replayed the log in
    order, and [05](05_sequential_and_neural.py) predicted the next item. This notebook
    puts time at the centre. Time can leak into an offline result in three ways:

    1. **The split** — the model trains on interactions that happened after the ones it
       is tested on.
    2. **The features** — a feature computed "as of today" is joined to training examples
       from the past, so the ranker learns from values nobody knew back then.
    3. **The data itself** — a record carries *when it happened*, but not always *when
       the system learned about it*. Data is **bitemporal**, and a lookup that ignores
       the second time leaks too.

    Each leak makes offline numbers look better than production ones. Along the way we
    use skrecsys's time support: time-based `cv=` for tuning, time-decay weights through
    `y`, `Cascade(time=True)` with point-in-time features, and `recommend(as_of=...)`.
    """)
    return


@app.cell
def _():
    _movielens = fetch_movielens_100k()
    _order = np.argsort(_movielens.timestamps, kind="stable")
    log = _movielens.data[_order]
    times = _movielens.timestamps[_order]
    genres = np.zeros((log[:, 1].max() + 1, len(_movielens.genre_names)))
    genres[_movielens.item_info.item_id] = _movielens.item_info.genres
    titles = dict(zip(_movielens.item_info.item_id, _movielens.item_info.title, strict=True))
    cutoff = np.quantile(times, 0.8)
    past = times < cutoff
    future = (times >= cutoff) & (times < cutoff + HORIZON) & np.isin(log[:, 0], log[past, 0])
    mo.md(f"""
    The MovieLens 100K log spans {(times.max() - times.min()) / DAY:.0f} days. For the
    sections on evaluation and tuning we cut it at the 80% point in time: the
    **past** is everything before the cutoff ({past.sum():,} interactions), and the
    **future** is the {HORIZON // DAY} days after it, for users the past knows
    ({future.sum():,} interactions, {len(np.unique(log[future, 0]))} users).
    """)
    return cutoff, future, genres, log, past, times, titles


@app.cell
def _():
    mo.md(r"""
    ## 1. Evaluating on the future: a rolling-origin backtest

    One cutoff gives one number, from one particular week. A **rolling-origin backtest**
    repeats the cut: fit on everything before day $d$, evaluate on the week after it,
    move $d$ forward, repeat. It shows how stable a result is over time — which is what a
    model in production lives through.
    """)
    return


@app.cell
def _(log, times):
    _rows = []
    for _day in range(56, 210, 21):
        _cut = times.min() + _day * DAY
        _train = times < _cut
        _test = (times >= _cut) & (times < _cut + 7 * DAY) & np.isin(log[:, 0], log[_train, 0])
        for _name, _factory in (
            ("MostPopular", MostPopularRecommender),
            ("ItemKNN", ItemKNNRecommender),
            ("EASE", EASE),
        ):
            _rows.append(
                {
                    "cutoff (day)": _day,
                    "model": _name,
                    "NDCG@10": evaluate_recommender(
                        _factory().fit(log[_train]), log[_test], metrics=[ndcg_at_k], k=10
                    )["ndcg@10"],
                    "test users": len(np.unique(log[_test, 0])),
                }
            )
    backtest = pd.DataFrame(_rows)
    _chart = (
        alt.Chart(backtest)
        .mark_line(point=True)
        .encode(
            x=alt.X("cutoff (day):Q"),
            y="NDCG@10:Q",
            color="model:N",
            tooltip=["cutoff (day)", "model", alt.Tooltip("NDCG@10:Q", format=".4f"), "test users"],
        )
        .properties(width=520, height=240, title="NDCG@10 on the week after each cutoff")
    )
    _wins = (
        backtest.loc[backtest.groupby("cutoff (day)")["NDCG@10"].idxmax(), "model"]
        .value_counts()
        .to_dict()
    )
    mo.vstack(
        [
            _chart,
            backtest.groupby("model")["NDCG@10"].agg(["mean", "std"]).round(4),
            mo.md(f"""
            Weekly results jump around: each week has only
            {backtest.groupby("cutoff (day)")["test users"].first().median():.0f} active
            users at the median, and what they watch changes. The best model per week is
            not always the same ({_wins}), and `MostPopular` is competitive far more often than on a random
            split. Decide on the average over many cutoffs, and look at the spread
            before believing a difference.
            """),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 2. Tuning on the future

    `AutoTune` and `GridSearchCV` take any `cv=` — an iterable of `(train, test)` index
    pairs. Below, the time-based splits are three cutoffs inside the past, each
    evaluated on the two weeks after it (test users must appear in their training fold).
    We tune RP3Beta and ItemKNN both ways and compare **what cross-validation promised**
    with **what the model then scored on the real future**.
    """)
    return


@app.cell
def _():
    tune_button = mo.ui.run_button(label="Tune two models both ways (about 20 s)")
    tune_button
    return (tune_button,)


@app.cell
def _(future, log, past, times, tune_button):
    mo.stop(
        not (tune_button.value or mo.app_meta().mode == "script"),
        mo.md("*Press the button to tune.*"),
    )
    _X, _t = log[past], times[past]
    time_splits = []
    for _end in np.quantile(_t, [0.6, 0.75, 0.9]):
        _train = np.flatnonzero(_t < _end)
        _test = np.flatnonzero((_t >= _end) & (_t < _end + HORIZON))
        time_splits.append((_train, _test[np.isin(_X[_test, 0], _X[_train, 0])]))
    _rows = []
    for _name, _factory in (("RP3Beta", RP3Beta), ("ItemKNN", ItemKNNRecommender)):
        for _cv_name, _cv in (("random (WarmStartKFold)", None), ("time-based", time_splits)):
            _tuned = AutoTune(_factory(), n_trials=30, cv=_cv, random_state=0).fit(_X)
            _rows.append(
                {
                    "model": _name,
                    "tuned with": _cv_name,
                    "best parameters": ", ".join(
                        f"{_p}={_v:.3g}" for _p, _v in _tuned.best_params_.items()
                    ),
                    "CV promised": _tuned.best_score_,
                    "future NDCG@10": evaluate_recommender(
                        _tuned, log[future], metrics=[ndcg_at_k], k=10
                    )["ndcg@10"],
                }
            )
    tuning = pd.DataFrame(_rows).set_index(["model", "tuned with"])
    tuning.round(4)
    return (tuning,)


@app.cell
def _(tuning):
    _random = tuning.xs("random (WarmStartKFold)", level="tuned with")
    _time = tuning.xs("time-based", level="tuned with")
    mo.md(f"""
    The random split promises NDCG@10 around {_random["CV promised"].mean():.2f}; the
    future delivers about {_random["future NDCG@10"].mean():.2f}. The time-based splits
    promise {_time["CV promised"].mean():.2f} — close to what actually happens. That is
    the main benefit: **a time-based validation score is an honest forecast**, so you can
    trust it when deciding whether a model is good enough to ship.

    The chosen parameters differ too, and on the future the time-tuned model wins for
    {", ".join(_n for _n in _time.index if _time.loc[_n, "future NDCG@10"] > _random.loc[_n, "future NDCG@10"]) or "neither model"}.
    With a few dozen users per fold, the gaps are small next to the noise of section 1 —
    the calibration of the estimate is the robust result.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 3. Recency: the past is not all equally useful

    Tastes and catalogs drift. Two ways to favour recent interactions without a
    sequential model, both through the `y` weights every skrecsys model accepts:

    - a **window**: weight 1 for the last $w$ days, 0 before;
    - a **time decay**: weight $0.5^{\text{age} / h}$, where $h$ is the half-life.

    `MostPopularRecommender(weighting="sum")` then ranks by *recent* popularity, and
    `EASE` weights the rows of its item–item regression.

    The half-life is a hyperparameter, so it is chosen on **validation** data: the last
    two weeks of the past play the future, with ages counted from their start. The
    chosen half-life is then refitted on the whole past and scored once on the
    real future.
    """)
    return


@app.cell
def _(cutoff, future, log, past, times):
    _valid_start = cutoff - HORIZON
    _before = times < _valid_start
    _valid = past & ~_before & np.isin(log[:, 0], log[_before, 0])

    def _score(model, train, test, as_of, half_life):
        _age = (as_of - times[train]) / DAY
        _y = 0.5 ** (_age / half_life) if half_life else np.ones(train.sum())
        return evaluate_recommender(
            model.fit(log[train], _y), log[test], metrics=[ndcg_at_k], k=10
        )["ndcg@10"]

    _rows = []
    for _half_life in (3, 7, 14, 30, 60, None):
        for _name, _params in (("MostPopular", {"weighting": "sum"}), ("EASE", {})):
            _factory = MostPopularRecommender if _name == "MostPopular" else EASE
            _rows.append(
                {
                    "half-life (days)": str(_half_life) if _half_life else "no decay",
                    "model": _name,
                    "validation NDCG@10": _score(
                        _factory(**_params), _before, _valid, _valid_start, _half_life
                    ),
                    "future NDCG@10": _score(
                        _factory(**_params), past, future, cutoff, _half_life
                    ),
                }
            )
    decay = pd.DataFrame(_rows)
    _chart = (
        alt.Chart(decay)
        .mark_bar()
        .encode(
            x=alt.X("half-life (days):N", sort=None),
            y="future NDCG@10:Q",
            color="model:N",
            xOffset="model:N",
            tooltip=["half-life (days)", "model", alt.Tooltip("future NDCG@10:Q", format=".4f")],
        )
        .properties(width=480, height=240, title="Time-decay weights, scored on the future")
    )
    _chosen = {}
    for _name, _group in decay.groupby("model"):
        _by_half_life = _group.set_index("half-life (days)")
        _pick = _by_half_life["validation NDCG@10"].idxmax()
        _chosen[_name] = {
            "half-life chosen on validation": _pick,
            "future NDCG@10, chosen": _by_half_life.loc[_pick, "future NDCG@10"],
            "future NDCG@10, no decay": _by_half_life.loc["no decay", "future NDCG@10"],
            "best possible on the future": _by_half_life["future NDCG@10"].max(),
        }
    _chosen = pd.DataFrame(_chosen).T.rename_axis("model")
    _gain = _chosen["future NDCG@10, chosen"] - _chosen["future NDCG@10, no decay"]
    _helped = [_n for _n in _chosen.index if _gain[_n] > 0]
    _pop_beats_ease = (
        _chosen.loc["MostPopular", "future NDCG@10, chosen"]
        > _chosen.loc["EASE", "future NDCG@10, no decay"]
    )
    mo.vstack(
        [
            _chart,
            _chosen.round(4),
            mo.md(f"""
            Chosen on validation and scored once on the future, decay helps
            {" and ".join(_helped) if _helped else "neither model"}
            ({", ".join(f"{_n} {_gain[_n]:+.3f}" for _n in _chosen.index)} against no
            decay).
            {"Decayed popularity even beats undecayed EASE: what is popular *this week* is a strong recommendation." if _pop_beats_ease else ""}
            The last column is what picking the half-life on the future itself would
            report — the optimistic number you get by tuning on the test set. Tune the
            half-life on time-based splits, never on a random one, where it cannot help.
            """),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 4. Features as of a time

    A two-stage recommender ([notebook 03](03_candidates_and_ranking.py)) ranks
    candidates with features. Some features change over time — a user's recent tastes, an
    item's trending score, a price. The ranker is trained on interactions from the past,
    so each training example must see the feature **as it was then**. `skrecsys`
    supports this directly:

    - give `X` a third column, the time of each interaction, and construct
      `Cascade(..., time=True)`;
    - `Cascade.fit` holds out each user's *latest* interactions by time, and featurizes
      the candidates **as of the user's first held-out interaction** — the moment they
      would have been shown;
    - `JoinDynamicFeatures("user-item-time", callback)` passes the callback rows of
      `(user, item, time)`, and the callback returns the values known at that time;
    - `evaluate_recommender` ranks a timed model as of each user's earliest held-out
      time, and `recommend(as_of=...)` replays any moment; without `as_of`, the time is
      missing (NaN) and the callback returns the latest values.

    The feature here is **genre affinity**: how much of the user's history so far falls in
    the candidate movie's genres. The callback, `GenreAffinity` in the setup cell, keeps
    each user's cumulative genre counts sorted by time and answers with a binary search.
    With `leaky=True` it ignores the time and uses the user's whole history — the
    everyday mistake of joining today's feature table to yesterday's training examples.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 5. Data is bitemporal

    "As of time $t$" hides a subtlety. Every record has **two** times:

    - **valid time** — when the fact happened or became true: the event timestamp, the
      moment a price took effect;
    - **transaction time** — when the system *recorded* it: when the event reached the
      warehouse, when a backfill loaded it, when an earlier value was corrected.

    They differ all the time in practice: mobile clients upload events hours late,
    pipelines backfill a broken day a week later, a price is corrected retroactively. What
    the system **knew** at time $t$ is every record with valid time $\le t$ **and**
    transaction time $\le t$, taking the latest correction recorded by $t$. A lookup on
    valid time alone answers "what was true at $t$", which includes facts nobody knew
    yet — a leak, and a train/serve skew, because in production the store can only return
    what it has already recorded.

    **skrecsys cannot resolve this for you.** `JoinDynamicFeatures` hands the callback a
    single time per pair — the time the pair is ranked as of — because only your data
    layer knows how its records are versioned. The callback must answer **as known at
    $t$**. In practice: store `(key, valid_from, recorded_at, value)` and filter on both
    times; many feature stores call this a point-in-time or "as-of" join, and pandas'
    `merge_asof` does it on one time column at a time. The same rule holds outside
    features: a late event belongs to the training data and to the `partial_fit` batch
    of the time it **arrived**, not the time it happened.

    MovieLens records only valid time, so we simulate transaction time: most events
    arrive within hours, and a fraction arrive days to weeks late, as with backfills.
    """)
    return


@app.cell
def _(times):
    _rng = np.random.default_rng(0)
    _late = _rng.random(len(times)) < LATE_SHARE
    lag_days = np.where(
        _late, _rng.uniform(3, 30, len(times)), _rng.exponential(0.25, len(times))
    )
    recorded_at = times + lag_days * DAY
    _counts, _edges = np.histogram(lag_days, bins=np.arange(0, 31))
    _hist = (
        alt.Chart(pd.DataFrame({"lag (days)": _edges[:-1], "events": _counts}))
        .mark_bar()
        .encode(x=alt.X("lag (days):Q", bin="binned"), x2="lag end:Q", y="events:Q")
        .transform_calculate(**{"lag end": "datum['lag (days)'] + 1"})
        .properties(width=420, height=180, title="Simulated recording lag")
    )
    mo.vstack(
        [
            _hist,
            mo.md(f"""
            {1 - LATE_SHARE:.0%} of the events are recorded within hours; {LATE_SHARE:.0%}
            arrive 3 to 30 days late.
            """),
        ]
    )
    return (recorded_at,)


@app.cell
def _():
    mo.md(r"""
    ## 6. Offline vs production, for three lookups

    We train the same cascade — EASE candidates, re-ranked by a boosted classifier on the
    generator score and genre affinity — three times, with three versions of the
    affinity lookup:

    | lookup | counts an event for time $t$ when |
    |---|---|
    | **leaky** | always: the whole log, including the future |
    | **valid time only** | it *happened* before $t$ |
    | **bitemporal** | it happened **and was recorded** before $t$ |

    The holdout is the latest 20% of every user's interactions — per user, the layout
    `Cascade(time=True)` trains on — since what this section measures is the features, not
    the split. Each model is evaluated twice on it:
    **offline**, with the lookup it was trained with, as a data scientist would; and in
    **production**, where the only lookup that exists is the bitemporal one — the store
    returns what it has recorded. We simulate production by pointing the fitted model's
    feature component at the bitemporal lookup.
    """)
    return


@app.cell
def _(genres, log, recorded_at, times):
    # user, then time, then a random key that breaks MovieLens's many ties in time
    _order = np.lexsort((np.random.default_rng(0).random(len(times)), times, log[:, 0]))
    _X, _t, _recorded = log[_order], times[_order].astype(float), recorded_at[_order]
    _users, _items = _X[:, 0], _X[:, 1]
    _, _starts, _counts = np.unique(_users, return_index=True, return_counts=True)
    _position = np.arange(len(_X)) - np.repeat(_starts, _counts)
    _n = np.repeat(_counts, _counts)
    _held = _position >= _n - (_n * 0.2).astype(int)
    _train = np.column_stack([_X[~_held], _t[~_held]])
    _test = np.column_stack([_X[_held], _t[_held]])
    lookups = {
        "leaky": GenreAffinity(_users, _items, _t, genres, leaky=True),
        "valid time only": GenreAffinity(_users, _items, _t, genres),
        "bitemporal": GenreAffinity(_users, _items, _recorded, genres),
    }
    _ease = evaluate_recommender(EASE().fit(_X[~_held]), _X[_held], metrics=[ndcg_at_k], k=10)
    _rows = {
        "EASE alone": {
            "offline NDCG@10": _ease["ndcg@10"],
            "production NDCG@10": _ease["ndcg@10"],
        }
    }
    timed_cascades = {}
    for _name, _lookup in lookups.items():
        _cascade = Cascade(
            EASE(),
            ConcatFeatures(
                [
                    ("generator", GeneratorScores()),
                    ("affinity", JoinDynamicFeatures("user-item-time", _lookup, n_features=1)),
                ]
            ),
            PointwiseRanker(HistGradientBoostingClassifier(random_state=0)),
            time=True,
        ).fit(_train)
        _offline = evaluate_recommender(_cascade, _test, metrics=[ndcg_at_k], k=10)["ndcg@10"]
        # Production: the fitted feature component now reads the real (bitemporal) store.
        dict(_cascade.features_.features_)["affinity"].callback = lookups["bitemporal"]
        _production = evaluate_recommender(_cascade, _test, metrics=[ndcg_at_k], k=10)["ndcg@10"]
        _rows[f"cascade, {_name}"] = {
            "offline NDCG@10": _offline,
            "production NDCG@10": _production,
        }
        timed_cascades[_name] = _cascade
    leak_results = pd.DataFrame(_rows).T.rename_axis("model")
    leak_results["gap"] = leak_results["production NDCG@10"] - leak_results["offline NDCG@10"]
    leak_results.round(4)
    return leak_results, timed_cascades


@app.cell
def _(leak_results):
    _r = leak_results
    _ease = _r.loc["EASE alone", "offline NDCG@10"]
    _leaky, _valid, _bi = (
        _r.loc["cascade, leaky"],
        _r.loc["cascade, valid time only"],
        _r.loc["cascade, bitemporal"],
    )
    mo.md(f"""
    - **Leaky.** Offline, the cascade scores {_leaky["offline NDCG@10"]:.4f} and
      {"beats" if _leaky["offline NDCG@10"] > _ease else "approaches"} EASE alone
      ({_ease:.4f}): the affinity already contains the held-out movies, so the ranker
      learns to trust it. In production, where the feature cannot see the future, it
      drops to {_leaky["production NDCG@10"]:.4f}. This is the classic "great offline,
      disappointing in the A/B test" result.
    - **Valid time only.** {_valid["offline NDCG@10"]:.4f} offline,
      {_valid["production NDCG@10"]:.4f} in production. The lookup counted events that had
      happened but had not yet arrived, so offline it was better informed than production
      will ever be — here by {-_valid["gap"]:.4f} of NDCG@10.
    - **Bitemporal.** Offline and production agree
      ({_bi["offline NDCG@10"]:.4f} and {_bi["production NDCG@10"]:.4f}): the offline
      number is a forecast you can trust.

    Once leakage is removed the cascade scores {_bi["production NDCG@10"]:.4f} against
    {_ease:.4f} for EASE alone:
    {"genre affinity does not beat EASE, which is exactly what the leaky offline result would have hidden." if _bi["production NDCG@10"] <= _ease else "genre affinity helps, but by far less than the leaky offline result promised."} With {LATE_SHARE:.0%} of
    events arriving late, the valid-time-only gap is small here; with late-arriving
    *corrections* to prices, stock or fraud labels it is often the largest leak in a
    pipeline, and no model code can detect it.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 7. Replaying the past with `as_of`

    A timed cascade can rank as of any moment: `recommend(users, as_of=t)` passes `t` to
    the feature callbacks, so the ranker sees the features as they were known then —
    handy for backtests and for debugging "why did user X see this on Tuesday?". Without
    `as_of`, as when serving, the callbacks get NaN and return the latest values.
    (The candidates still come from the generator as fitted: `as_of` replays the
    features, not the model version.)
    """)
    return


@app.cell
def _(log, timed_cascades, times, titles):
    _model = timed_cascades["bitemporal"]
    _user = np.unique(log[:, 0])[0]
    _theirs = np.sort(times[log[:, 0] == _user]).astype(float)
    _moments = {
        "after 5 ratings": _theirs[5],
        "after half": _theirs[len(_theirs) // 2],
        "latest (no as_of)": None,
    }
    _lists = {}
    for _label, _moment in _moments.items():
        _items, _ = (
            _model.recommend([_user], n_recommendations=5)
            if _moment is None
            else _model.recommend([_user], n_recommendations=5, as_of=_moment)
        )
        _lists[_label] = [titles[int(_item)] for _item in _items[0]]
    pd.DataFrame(_lists, index=pd.RangeIndex(1, 6, name=f"user {_user}, rank"))
    return


@app.cell
def _():
    mo.md(r"""
    ## Takeaways: a time-leak checklist

    - **Split** by time: hold out the future, for testing *and* for validation
      (`cv=` in `AutoTune` / `GridSearchCV`). Backtest over several cutoffs.
    - **Weight** recent interactions: windows or decay through `y`; tune the half-life on
      time-based splits.
    - **Features** as of the time of each training example: `Cascade(time=True)` and a
      `JoinDynamicFeatures` keyed by `"time"`, `"item-time"` or `"user-item-time"`.
    - **Bitemporal lookups**: the callback must return what was *known* at $t$ — valid
      time and recorded time both before $t$. skrecsys passes one time; resolving the
      two is the data layer's job.
    - **Late data** goes into the training window and the `partial_fit` batch of when it
      *arrived*.
    - If a feature makes offline results jump, suspect a leak before celebrating.
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
                    '1. Name the three ways time can leak into an offline result.': mo.md(
                        'The split (training on interactions after the tested ones); the features (values computed later joined to earlier training examples); and the data (using when something happened instead of when it became known, i.e. ignoring transaction time).'
                    ),
                    '2. In section 2, why did cross-validation on a random split promise a much higher NDCG@10 than the future delivered?': mo.md(
                        'A random split asks the model to fill gaps inside histories it has partly seen, which is easier than predicting the future. A time-based split asks the real question, so its estimate is close to what production gets.'
                    ),
                    "3. With `Cascade(time=True)`, as of which time are a training user's candidates featurized?": mo.md(
                        "As of that user's earliest held-out interaction, the moment the candidates would have been shown. A `JoinDynamicFeatures` keyed by time receives that time and must return the values known then."
                    ),
                    '4. Define valid time and transaction time. What was *known* at time t?': mo.md(
                        'Valid time is when a fact happened or became true; transaction time is when the system recorded it. Known at t: every record with valid time ≤ t **and** transaction time ≤ t, with the latest correction recorded by t.'
                    ),
                    "5. Why can't skrecsys resolve bitemporal data for you, and what must your callback do?": mo.md(
                        'skrecsys passes one time per pair, the time it is ranked as of; it cannot know how your store versions its records. The callback must answer "as known at t", filtering on both times, for example on `(key, valid_from, recorded_at, value)` rows.'
                    ),
                    '6. An event happened on Monday but reached the warehouse on Thursday. Which `partial_fit` batch does it belong to?': mo.md(
                        "Thursday's, the batch of when it arrived. Replaying it into Monday's batch would give a backtest information that production did not have on Monday."
                    ),
                    '7. A new feature raises offline NDCG by 3%. What should you check before shipping?': mo.md(
                        "That it is computed as of each example's time and as known at that time: no future values, no late-arriving records. Compare offline with a production-like evaluation, as in section 6."
                    ),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
