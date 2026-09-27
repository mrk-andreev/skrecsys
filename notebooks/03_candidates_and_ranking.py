import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import copy

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from skrecsys.compose import (
        Cascade,
        ConcatFeatures,
        GeneratorScores,
        InteractionCounts,
        JoinStaticFeatures,
        KnownUser,
        PointwiseRanker,
        ReciprocalRankFusion,
        RecommenderScores,
        SegmentPopularity,
        Switch,
    )
    from skrecsys.datasets import fetch_movielens_100k
    from skrecsys.metrics import evaluate_recommender, ndcg_at_k, recall_at_k
    from skrecsys.model_selection import ColdStartSplit
    from skrecsys.recommendation import (
        EASE,
        BM25Recommender,
        ItemKNNRecommender,
        MostPopularRecommender,
    )

    N_RETRIEVED = 100
    VALID_SHARE = 0.2


@app.function
def position_among(query, key):
    """Each row's position among the rows of its query sharing its ``key``, best first."""
    return (
        pd.DataFrame({"query": query, "key": key})
        .groupby(["query", "key"], sort=False)
        .cumcount()
        .to_numpy()
    )


@app.class_definition
class BusinessRules:
    """The last word after the ranker: a callable object, so a cascade holding it pickles.

    - licensing: films released before ``min_year`` (or of an unknown year) are dropped;
    - promotion: the best-ranked film released in ``new_from`` or later goes first;
    - diversity: films past the ``max_per_genre``-th of their primary genre go last.
    """

    def __init__(self, item_table, *, min_year, new_from, max_per_genre):
        ids = item_table["item_id"].to_numpy()
        self.year = pd.Series(item_table["year"].to_numpy(), index=ids)
        # the first genre a film lists; the genre columns follow the id and the year
        self.genre = pd.Series(item_table.iloc[:, 2:].to_numpy().argmax(axis=1), index=ids)
        self.min_year = min_year
        self.new_from = new_from
        self.max_per_genre = max_per_genre

    def __call__(self, pairs, scores, groups):
        # pairs, scores: every candidate, best first within each query; groups: list lengths
        query = np.repeat(np.arange(len(groups)), groups)
        year = self.year.reindex(pairs[:, 1]).to_numpy()
        keep = year >= self.min_year
        pairs, scores, query, year = pairs[keep], scores[keep], query[keep], year[keep]
        genre = self.genre.reindex(pairs[:, 1]).to_numpy()
        overflow = position_among(query, genre) >= self.max_per_genre
        is_new = year >= self.new_from
        promoted = is_new & (position_among(query, is_new) == 0)
        tier = np.where(promoted, 0, np.where(overflow, 2, 1))
        order = np.lexsort((tier, query))  # stable: the ranker's order within each tier
        return pairs[order], scores[order], np.bincount(query, minlength=len(groups))


@app.cell
def _():
    mo.md(r"""
    # Multi-stage recommenders and cold start

    The models in notebooks [01](01_intro_to_recommendations.py) and
    [02](02_model_selection_and_tuning.py) do everything in one step: score the whole
    catalog from interactions alone. Production systems usually split the work in two:

    1. **candidate generation** (retrieval): a cheap model picks a few hundred plausible
       items out of the whole catalog;
    2. **ranking**: a richer model reorders only those candidates, using anything known
       about the user, the item and the pair — metadata, counts, other models' opinions —
       which a collaborative-filtering model has no place for.

    And every production system meets users it has never seen. This notebook builds both
    with `skrecsys.compose`, whose pieces are all recommenders themselves: they fit,
    recommend, clone, pickle and grid-search like any other model.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 1. A split with warm and cold users

    `ColdStartSplit` asks the two questions a live service faces at once. It holds out 10%
    of users **entirely** — cold users, never seen in training — and the latest 20% of
    every other user's interactions — warm users. Row order is read as time, so the
    interactions are sorted by timestamp first. The warm holdout is *per user*: training
    still holds other users' later interactions, a leak [notebook
    02](02_model_selection_and_tuning.py) measures. It is the protocol `Cascade` itself
    uses inside `fit`, which keeps this notebook consistent; notebook 06 shows a global
    cutoff in time. We also keep the user and movie metadata that MovieLens provides,
    since the second stage can use it.
    """)
    return


@app.cell
def _():
    movielens = fetch_movielens_100k(as_frame=True)
    _ratings = movielens.frame.sort_values("timestamp", kind="stable")
    X = _ratings[["user_id", "item_id"]].to_numpy()
    train, test = next(ColdStartSplit(cold_users=0.1, test_size=0.2, random_state=0).split(X))
    X_train, X_test = X[train], X[test]
    _cold = ~np.isin(X_test[:, 0], X_train[:, 0])
    X_warm, X_cold = X_test[~_cold], X_test[_cold]
    mo.md(f"""
    | | interactions | users |
    |---|---|---|
    | train | {len(X_train):,} | {len(np.unique(X_train[:, 0])):,} |
    | test, warm users | {len(X_warm):,} | {len(np.unique(X_warm[:, 0])):,} |
    | test, cold users | {len(X_cold):,} | {len(np.unique(X_cold[:, 0])):,} |
    """)
    return X_cold, X_train, X_warm, movielens


@app.cell
def _(movielens):
    _users = movielens.user_info
    _items = movielens.item_info
    user_table = pd.DataFrame(
        {
            "user_id": _users["user_id"],
            "age": _users["age"].astype(float),
            "male": (_users["gender"] == "M").astype(float),
        }
    )
    item_table = pd.concat(
        [
            _items[["item_id"]],
            _items["release_date"].dt.year.astype(float).rename("year"),
            _items[movielens.genre_names].astype(float),
        ],
        axis=1,
    )
    segments = pd.DataFrame(
        {
            "user_id": _users["user_id"],
            "gender": _users["gender"],
            "age_band": (_users["age"] // 10 * 10).astype(str),
            "occupation": _users["occupation"],
        }
    )
    mo.vstack(
        [
            mo.md("**Side information** — tables keyed by the identifier in column 0:"),
            mo.hstack([user_table.head(3), item_table.iloc[:3, :8]], widths=[1, 2]),
        ]
    )
    return item_table, segments, user_table


@app.cell
def _():
    mo.md(r"""
    ## 2. Candidate generation

    The first stage only has to get the right items **somewhere** in its list: the
    ranker can reorder candidates, but never recover an item that was not retrieved. So a
    generator is judged by **recall at the retrieval depth** — how many of a user's
    relevant movies are among its top $k$ candidates.

    Choosing the generator is model selection, so it happens on **validation** data, not
    on the test set: the latest 20% of each user's *training* interactions. The test set
    stays untouched until the pipelines are compared.

    `ReciprocalRankFusion` combines several generators with nothing to train: an item
    scores $\sum 1/(60 + \text{rank})$ over the lists that contain it, so models whose
    scores live on unrelated scales combine by rank alone.
    """)
    return


@app.cell
def _(X_train):
    _users = pd.Series(X_train[:, 0])
    _position = _users.groupby(_users).cumcount().to_numpy()
    _size = _users.map(_users.value_counts()).to_numpy()
    _valid = _position >= _size - (_size * VALID_SHARE).astype(int)
    X_fit, X_valid = X_train[~_valid], X_train[_valid]
    generators = {
        "BM25": BM25Recommender(),
        "ItemKNN": ItemKNNRecommender(),
        "EASE": EASE(),
        "RRF(BM25, ItemKNN, EASE)": ReciprocalRankFusion(
            [BM25Recommender(), ItemKNNRecommender(), EASE()], n_retrieved=200
        ),
    }
    _depths = [10, 25, 50, 100, 200]
    candidate_recall = pd.DataFrame(
        [
            {"generator": _name, "k": _k, "recall": _value}
            for _name, _model in generators.items()
            for _k, _value in zip(
                _depths,
                evaluate_recommender(
                    _model.fit(X_fit), X_valid, metrics=[recall_at_k], k=_depths
                ).values(),
                strict=True,
            )
        ]
    )
    _chart = (
        alt.Chart(candidate_recall)
        .mark_line(point=True)
        .encode(
            x=alt.X("k:Q", title="candidates retrieved per user"),
            y=alt.Y("recall:Q", title="recall of validation movies"),
            color="generator:N",
            tooltip=["generator", "k", alt.Tooltip("recall:Q", format=".3f")],
        )
        .properties(width=520, height=260, title="How much can a ranker possibly find?")
    )
    _at_depth = candidate_recall[candidate_recall["k"] == N_RETRIEVED].set_index("generator")
    _best = _at_depth["recall"].idxmax()
    _members = _at_depth.drop(index=[_n for _n in _at_depth.index if _n.startswith("RRF")])
    _ease_gap = _at_depth["recall"].max() - _at_depth.loc["EASE", "recall"]
    mo.vstack(
        [
            _chart,
            mo.md(f"""
            At {N_RETRIEVED} candidates the best generator, **{_best}**, retrieves
            {_at_depth["recall"].max():.0%} of the validation movies, against
            {candidate_recall.query("k == 10")["recall"].max():.0%} in its top 10. That
            gap is the room a second stage has to work in. Fusion
            {"beats" if _best.startswith("RRF") else "does not beat"} its best member,
            {_members["recall"].idxmax()}, on this split; it pays off when the members
            make different mistakes. We use EASE as the generator below:
            {"it is the best one" if _ease_gap == 0 else f"it is {_ease_gap:.1%} of recall behind the best, and a single model is simpler to fit and serve"}.
            """),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 3. Features of candidate pairs

    The second stage sees each candidate as a `(user, item)` pair and describes it with a
    row of features. Feature components are small estimators; `ConcatFeatures` puts them
    side by side, like scikit-learn's `FeatureUnion`:

    - `GeneratorScores()` — the generator's own score, which almost every ranker wants;
    - `RecommenderScores(model)` — a second model's opinion of the pair;
    - `InteractionCounts("user")`, `InteractionCounts("item")` — how much history each
      side has;
    - `JoinStaticFeatures("user", table)`, `JoinStaticFeatures("item", table)` — metadata
      from a table keyed by identifier;
    - `SegmentPopularity(segments)` — how popular the item is among users of the same
      segment (gender, age band, occupation), the one signal a user with no history has.

    A missing value becomes NaN: `JoinStaticFeatures` does that for an identifier absent
    from its table, and here one movie has no release date, so no `year`. Tree ensembles
    handle NaN natively; a linear ranker needs an imputer in its pipeline.

    Demographics such as gender and age are sensitive attributes. Using them as ranking
    features is common and can help, but check whether the lists they produce differ
    between groups in ways you would not defend, and whether the law where you operate
    allows it.

    Here are the features of the top candidates for one warm user:
    """)
    return


@app.cell
def _(item_table, user_table):
    warm_features = ConcatFeatures(
        [
            ("generator", GeneratorScores()),
            ("itemknn", RecommenderScores(ItemKNNRecommender())),
            ("user_count", InteractionCounts("user")),
            ("item_count", InteractionCounts("item")),
            ("user", JoinStaticFeatures("user", user_table)),
            ("item", JoinStaticFeatures("item", item_table)),
        ]
    )
    return (warm_features,)


@app.cell
def _(X_train, X_warm, warm_features):
    _user = X_warm[0, 0]
    _items, _scores = EASE().fit(X_train).recommend([_user], n_recommendations=5)
    _pairs = np.column_stack([np.repeat(_user, 5), _items[0]])
    _features = clone(warm_features).fit(X_train)
    pd.DataFrame(
        _features.transform(_pairs, scores=_scores[0]),
        columns=_features.get_feature_names_out(),
        index=pd.Index(_items[0], name=f"user {_user}, item"),
    ).iloc[:, :12].round(3)
    return


@app.cell
def _():
    mo.md(r"""
    ## 4. `Cascade`: generator → features → ranker

    `Cascade(generator, features, ranker, n_retrieved=100)` is a two-stage recommender.
    `recommend` asks the generator for 100 candidates per user, featurizes each pair and
    returns the ones the ranker scores highest.

    Training the ranker needs labelled candidates, and here `Cascade.fit` is careful: it
    holds out the latest 20% of each user's rows, fits a generator on the rest, and labels
    its candidates by whether they were held out — so the ranker learns from candidates
    *exactly as they look when serving*, never from interactions the generator was fitted
    on. Only then does it refit the generator on everything.

    The ranker wraps any scikit-learn classifier with `PointwiseRanker`. Below: a
    logistic regression and a gradient-boosted tree ensemble, against EASE alone.
    """)
    return


@app.cell
def _(X_train, X_warm, warm_features):
    rankers = {
        "logistic regression": PointwiseRanker(
            make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression())
        ),
        "gradient boosting": PointwiseRanker(HistGradientBoostingClassifier(random_state=0)),
    }
    warm_cascades = {
        _name: Cascade(EASE(), warm_features, _ranker, n_retrieved=N_RETRIEVED).fit(X_train)
        for _name, _ranker in rankers.items()
    }
    _rows = {"EASE alone": EASE().fit(X_train)} | {
        f"EASE → {_name}": _model for _name, _model in warm_cascades.items()
    }
    warm_results = pd.DataFrame(
        {
            _name: evaluate_recommender(
                _model, X_warm, metrics=[ndcg_at_k, recall_at_k], k=10
            )
            for _name, _model in _rows.items()
        }
    ).T.rename_axis("warm users")
    warm_results.round(4)
    return warm_cascades, warm_results


@app.cell
def _(warm_results):
    _ndcg = warm_results["ndcg@10"]
    _alone = _ndcg["EASE alone"]
    _linear, _boosted = _ndcg["EASE → logistic regression"], _ndcg["EASE → gradient boosting"]

    def _versus(value):
        return f"{value:.4f} ({value - _alone:+.4f} against EASE alone)"

    mo.md(f"""
    NDCG@10 of EASE alone is {_alone:.4f}; the linear ranker gets {_versus(_linear)} and
    the boosted one {_versus(_boosted)}.
    {"Trees can use feature interactions — *this* user's age with *that* movie's genre and year — that a linear score cannot." if _boosted > _linear else "Here the linear ranker is as good as the trees: the features carry little signal a linear score misses."}
    Gradient-boosting libraries plug in the same way:
    `skrecsys.integrations` provides `CatBoostRanker`, `XGBRanker` and `LGBMRanker`
    (each behind its own extra, e.g. `pip install skrecsys[catboost]`), and `BlendRanker`
    stacks several rankers on out-of-fold scores. The three declare search ranges for
    their main knobs (learning rate, tree size, regularization, row and column sampling),
    so `AutoTune(Cascade(EASE(), features, LGBMRanker()))` tunes them as `ranker__<name>`
    without a `search_space=`; any other library parameter goes through `extra_params=`.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 5. Business rules on top of the ranker

    The ranker orders candidates by what a user is likely to watch. A service usually has
    the last word on top of that: titles it has no licence for must go, new releases get a
    push, and a list of ten thrillers looks broken even when each one is a good guess.

    `Cascade(..., postprocess=rules)` is where that logic lives. The callback is a black box
    over the ranked lists: it receives **every candidate of each query, best first** —
    `pairs`, the ranker's `scores`, and `groups`, the length of each list — and returns the
    lists to serve in the same layout. It may reorder, drop, or even add items. `recommend`
    serves the first 10 of what it returns and raises if a list comes back shorter, so
    `n_retrieved` must leave room for what the rules drop.

    `BusinessRules`, defined at the top of this notebook, applies three rules: films from
    before 1970 are unlicensed, the best-ranked 1998 release goes first, and films past the
    third of their genre go to the back. Years come from the `release_date` column, so a
    film that came out in January 1998 counts as a 1998 release even when its title says
    1997. The rules take no part in `fit`: the ranker still learns relevance, and only
    serving changes. So there is no need to refit — a copy of the fitted cascade with the
    rules set is the ruled model.
    """)
    return


@app.cell
def _(item_table, warm_cascades):
    rules = BusinessRules(item_table, min_year=1970, new_from=1998, max_per_genre=3)
    # postprocess plays no part in fit, so the fitted cascade takes the rules as it is
    ruled_cascade = copy.deepcopy(warm_cascades["gradient boosting"]).set_params(
        postprocess=rules
    )
    return ruled_cascade, rules


@app.cell
def _(X_warm, item_table, movielens, ruled_cascade, rules, warm_cascades):
    _models = {
        "ranker alone": warm_cascades["gradient boosting"],
        "with the rules": ruled_cascade,
    }
    _users = np.unique(X_warm[:, 0])
    _titles = movielens.item_info.set_index("item_id")["title"]
    _genre_names = np.array(item_table.columns[2:])
    _lists, _rows = {}, {}
    for _name, _model in _models.items():
        _items, _ = _model.recommend(_users, n_recommendations=10)
        _year = rules.year.reindex(_items.ravel()).to_numpy().reshape(_items.shape)
        _genre = rules.genre.reindex(_items.ravel()).to_numpy().reshape(_items.shape)
        _lists[_name] = _items
        _rows[_name] = evaluate_recommender(_model, X_warm, metrics=[ndcg_at_k], k=10) | {
            "films before 1970": (_year < rules.min_year).mean(),
            "a 1998 film first": (_year[:, 0] >= rules.new_from).mean(),
            "largest genre per list": np.mean([np.bincount(_g).max() for _g in _genre]),
        }
    rules_results = pd.DataFrame(_rows).T.rename_axis("warm users")
    # a user whose list the promotion changed, to look at side by side
    _promoted = _lists["with the rules"][:, 0] != _lists["ranker alone"][:, 0]
    _shown = int(np.flatnonzero(_promoted)[0]) if _promoted.any() else 0
    rules_example = pd.DataFrame(
        {
            _name: [
                f"{_titles[_i]} · {_genre_names[rules.genre[_i]]}" for _i in _items[_shown]
            ]
            for _name, _items in _lists.items()
        },
        index=pd.RangeIndex(1, 11, name=f"user {_users[_shown]}"),
    )
    mo.vstack([rules_results.round(3), rules_example])
    return (rules_results,)


@app.cell
def _(rules_results):
    _plain, _ruled = rules_results.iloc[0], rules_results.iloc[1]
    mo.md(f"""
    The rules do what they say. No film from before 1970 is served
    ({_plain["films before 1970"]:.1%} of the ranker's top-10 slots were). A 1998 release
    opens {_ruled["a 1998 film first"]:.0%} of the lists, against
    {_plain["a 1998 film first"]:.1%} before; the others had no 1998 film among their
    candidates. The largest genre in a list shrinks from {_plain["largest genre per list"]:.1f}
    films to {_ruled["largest genre per list"]:.1f}; it can pass 3 when the promoted film
    shares a genre with the top three, or a list runs out of other genres.

    {"They also cost accuracy" if _ruled["ndcg@10"] < _plain["ndcg@10"] else "Here they cost no accuracy"}:
    NDCG@10 goes from {_plain["ndcg@10"]:.3f} to {_ruled["ndcg@10"]:.3f},
    {"because the rules push aside films the ranker had good reason to rank high" if _ruled["ndcg@10"] < _plain["ndcg@10"] else "so the films they push aside were not the ones users went on to watch"}.
    Measuring that price is the point of putting the rules *inside* the
    recommender: `evaluate_recommender`, cross-validation and `AutoTune` all call
    `recommend`, so they score what users are really shown. The rules also travel with the
    cascade when it is pickled, as long as they are a module-level function or a callable
    object such as `BusinessRules` (a lambda will not pickle). In production, keep such a
    class in your own package, so the process that unpickles the model can import it.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 6. Cold start

    Now the users nobody has seen. EASE — like every collaborative-filtering model — has
    nothing to go on:
    """)
    return


@app.cell
def _(X_cold, X_train):
    try:
        EASE().fit(X_train).recommend(X_cold[:1, 0], n_recommendations=10)
        _message = "no error"
    except ValueError as _error:
        _message = str(_error)
    mo.md(f"`EASE().fit(X_train).recommend([cold user])` raises:\n\n> {_message}")
    return


@app.cell
def _():
    mo.md(r"""
    `Switch(condition, on_true, on_false)` routes each user: the personalized model where
    the condition holds, a fallback elsewhere. `KnownUser()` holds for users seen in
    training; `MinInteractions(n)` and `QueryIn(ids)` are the other conditions, and they
    combine with `~`, `&` and `|`. `MostPopularRecommender` can serve anyone, which makes
    it the natural fallback.

    The fallback can itself be a cascade. A **cold cascade** retrieves the most popular
    movies and reorders them with segment popularity. Its ranker must learn from users
    that look cold, so `Cascade(split=ColdStartSplit(...))` holds out whole users inside
    `fit`, instead of the latest rows of every user.
    """)
    return


@app.cell
def _(X_train, segments):
    cold_cascade = Cascade(
        MostPopularRecommender(),
        ConcatFeatures([GeneratorScores(), SegmentPopularity(segments)]),
        PointwiseRanker(make_pipeline(StandardScaler(), LogisticRegression())),
        n_retrieved=N_RETRIEVED,
        split=ColdStartSplit(cold_users=0.2, test_size=0.0, random_state=0),
    ).fit(X_train)
    return (cold_cascade,)


@app.cell
def _(X_cold, X_train, X_warm, cold_cascade, warm_cascades):
    pipelines = {
        "Switch(EASE, MostPopular)": Switch(
            KnownUser(), EASE(), MostPopularRecommender()
        ).fit(X_train),
        "Switch(EASE → boosting, MostPopular)": Switch(
            KnownUser(), warm_cascades["gradient boosting"], MostPopularRecommender()
        ).fit(X_train),
        "Switch(EASE → boosting, popular → segments)": Switch(
            KnownUser(), warm_cascades["gradient boosting"], cold_cascade
        ).fit(X_train),
    }
    _X_all = np.concatenate([X_warm, X_cold])
    pipeline_results = pd.DataFrame(
        {
            _name: {
                _group: evaluate_recommender(_model, _X, metrics=[ndcg_at_k], k=10)["ndcg@10"]
                for _group, _X in (("all", _X_all), ("warm", X_warm), ("cold", X_cold))
            }
            for _name, _model in pipelines.items()
        }
    ).T.rename_axis("NDCG@10")
    pipeline_results.round(4)
    return (pipeline_results,)


@app.cell
def _(pipeline_results):
    _base, _full = pipeline_results.iloc[0], pipeline_results.iloc[-1]
    mo.md(f"""
    Every pipeline now serves every user. Reading the table:

    - Cold users score *higher* than warm ones ({_base["cold"]:.3f} against
      {_base["warm"]:.3f} for the baseline). That is not because they are easier: a cold
      user's held-out set is their whole history, so many more items count as relevant,
      and most of those are popular. **Compare rows within a column, never across
      columns.**
    - The warm ranker lifts warm users from {_base["warm"]:.3f} to {_full["warm"]:.3f}.
    - The cold ranker {"lifts" if _full["cold"] > _base["cold"] else "moves"} cold users
      from {_base["cold"]:.3f} to {_full["cold"]:.3f} ({_full["cold"] / _base["cold"] - 1:+.1%})
      with nothing but gender, age band and occupation. Demographics say little about
      taste in movies; on a service where segments carry more signal (country, language,
      signup channel) they help more.

    Wrapped in a single `Switch`, the whole thing is still one recommender: it pickles as
    one object, clones for cross-validation, and exposes every inner parameter to
    `GridSearchCV` or `AutoTune` as a nested name such as
    `on_true__ranker__estimator__learning_rate`.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Takeaways

    - Split the work: a cheap **generator** judged by recall at the retrieval depth, then a
      **ranker** that can use side information and other models' opinions.
    - `Cascade` trains its ranker on held-out interactions only, so what it learns is what
      it will see when serving.
    - Business rules go in `Cascade(postprocess=...)`. They turn the ranked lists into the
      lists to serve, are left out of training, and are measured by every evaluation.
    - `Switch` routes users between models; `KnownUser` + `MostPopularRecommender` is the
      minimum any deployment needs, and a cold cascade with `SegmentPopularity` can do
      better when segments carry signal.
    - Evaluate warm and cold users separately: their numbers are not on the same scale.

    Next: [notebook 04](04_production.py) takes a fitted model to production — serving,
    incremental updates and fast retrieval.
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
                    '1. Why is a candidate generator judged by recall at the retrieval depth rather than by NDCG@10?': mo.md(
                        "The ranker can reorder candidates but never recover an item that was not retrieved. Recall@100 is the ceiling for the whole pipeline; the order inside the 100 is the ranker's job."
                    ),
                    '2. How does `Cascade.fit` stop the ranker from learning on interactions the generator has already seen?': mo.md(
                        "It holds out the last `split` fraction of each user's rows, fits a generator on the rest, and labels that generator's candidates by whether they were held out. The ranker learns from these. Only then does it refit the generator on everything, for serving."
                    ),
                    '3. `EASE().recommend([new_user])` raises. How do you serve users who are not in the training data?': mo.md(
                        'Route them: `Switch(KnownUser(), EASE(), MostPopularRecommender())`. `MostPopularRecommender` can serve any user, and the fallback may itself be a cascade.'
                    ),
                    '4. In section 6, cold users score far higher NDCG@10 than warm users. Are they easier to serve?': mo.md(
                        "No. A cold user's held-out set is their whole history, so many more items count as relevant, and most are popular. The numbers are on different scales: compare pipelines within a column, never cold against warm."
                    ),
                    '5. What does `SegmentPopularity` compute, and why is it the feature for cold users?': mo.md(
                        "For each pair it gives the share of users in the pair's user's segment (gender, age band, occupation...) who interacted with the item, and that share's lift over the global share. A user with no history still has a segment."
                    ),
                    '6. Why is the cold cascade fitted with `split=ColdStartSplit(...)`?': mo.md(
                        'Its ranker must learn from candidate lists as a cold user is served them. Holding out whole users inside `fit` produces exactly that; the default split would hold out the latest rows of users who still have a history.'
                    ),
                    '7. Why does `postprocess` run in `recommend` but not in `fit`, and why does it get the whole ranked list rather than the top 10?': mo.md(
                        "The ranker should learn which films a user will watch. A licence or a promotion is policy, not relevance, and training on it would teach the ranker the rules instead of the users. `recommend` is also what evaluation calls, so the rules are still measured. The rules get the whole list so that items after the top 10 can fill the places of the ones they drop; if too few are left, `recommend` raises."
                    ),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
