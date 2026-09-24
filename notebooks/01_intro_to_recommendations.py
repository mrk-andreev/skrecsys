import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import time
    from functools import partial

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import cross_validate

    from skrecsys.datasets import fetch_movielens_100k
    from skrecsys.metrics import (
        average_precision_at_k,
        catalog_coverage_at_k,
        evaluate_recommender,
        hit_rate_at_k,
        item_popularity,
        make_recommender_scorer,
        mean_popularity_at_k,
        ndcg_at_k,
        novelty_at_k,
        precision_at_k,
        recall_at_k,
        reciprocal_rank_at_k,
    )
    from skrecsys.model_selection import WarmStartKFold
    from skrecsys.recommendation import (
        EASE,
        AlternatingLeastSquares,
        BayesianPersonalizedRanking,
        BM25Recommender,
        ItemKNNRecommender,
        MostPopularRecommender,
        RP3Beta,
        SLIMElasticNet,
    )

    LOW_RATING = 2  # 1 and 2 stars
    LIKED_RATING = 4
    STABLE_RANK_CORRELATION = 0.8


@app.cell
def _():
    mo.md(r"""
    # An introduction to recommender systems with skrecsys

    A recommender system answers one question over and over: *out of everything in the
    catalog that this user has not seen yet, what should we show them first?* The answer is
    a short ranked list — the top $k$ items — and the quality of a recommender is the
    quality of that list.

    This notebook walks through the basics:

    1. how interactions are represented, and the difference between **explicit** and
       **implicit** feedback;
    2. **collaborative filtering** on a toy dataset you can check by hand;
    3. a real dataset, MovieLens 100K;
    4. how **evaluation** works: splitting interactions and grouping them per user;
    5. the **metrics**, computed by hand on a single ranking;
    6. the **baseline models** in `skrecsys.recommendation`, compared on the same split;
    7. **cross-validation** with scikit-learn.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 1. Interactions

    Everything a collaborative-filtering model knows comes from a log of interactions:
    *user $u$ did something with item $i$*. skrecsys takes that log exactly as it is stored
    in a database — one row per interaction:

    - `X` has two columns, user identifiers in column 0 and item identifiers in column 1.
      Identifiers can be strings or integers; skrecsys encodes them itself.
    - `y` is optional: one value per row, such as a rating, a play count or a weight.

    Conceptually the log is a sparse **user × item matrix** $R$, where $r_{ui}$ is the
    value of the interaction (or 1) and empty cells are pairs that never happened.
    """)
    return


@app.cell
def _():
    toy = pd.DataFrame(
        [
            ["alice", "The Matrix", 5],
            ["alice", "Alien", 4],
            ["alice", "Blade Runner", 5],
            ["bob", "The Matrix", 4],
            ["bob", "Blade Runner", 5],
            ["bob", "Heat", 3],
            ["carol", "Alien", 5],
            ["carol", "Blade Runner", 4],
            ["carol", "Heat", 4],
            ["dave", "Toy Story", 5],
            ["dave", "Finding Nemo", 4],
            ["dave", "The Matrix", 2],
            ["eve", "Toy Story", 4],
            ["eve", "Finding Nemo", 5],
            ["eve", "Heat", 2],
            ["frank", "Toy Story", 5],
            ["frank", "Finding Nemo", 5],
            ["grace", "Toy Story", 3],
            ["grace", "Finding Nemo", 4],
        ],
        columns=["user", "item", "rating"],
    )
    toy_X = toy[["user", "item"]].to_numpy()
    toy_y = toy["rating"].to_numpy(dtype=float)
    toy_matrix = toy.pivot_table(index="user", columns="item", values="rating")
    mo.hstack(
        [
            mo.vstack([mo.md("**`X`, `y`: the interaction log**"), toy]),
            mo.vstack([mo.md("**The same log as a user × item matrix**"), toy_matrix]),
        ],
        widths=[1, 2],
    )
    return toy_X, toy_matrix, toy_y


@app.cell
def _():
    mo.md(r"""
    ## 2. Explicit vs implicit feedback

    The value in a cell of $R$ can mean two very different things.

    | | **Explicit feedback** | **Implicit feedback** |
    |---|---|---|
    | What it is | the user *states* a preference | the user *does* something |
    | Examples | star ratings, thumbs up/down, reviews | views, clicks, plays, purchases, add-to-cart |
    | Negatives | yes: a 1-star rating is a clear "no" | none: a missing pair is *unknown*, not "disliked" |
    | Volume | small — rating is effort | large — a by-product of using the product |
    | What the value means | how much the user liked the item | how *confident* we are the user is interested |
    | Natural task | predict the rating (RMSE) | rank unseen items (top-$k$ metrics) |

    Most production systems run on implicit feedback: it is what you get for free, and in
    huge quantities. Even when explicit ratings exist, the goal is usually still a ranked
    list, not a predicted number of stars — nobody browses a catalog sorted by predicted
    rating of items they will never click.

    In skrecsys the difference is just whether you pass `y`:

    - `fit(X)` treats the data as **implicit**: every interaction weighs 1.
    - `fit(X, y)` passes a value per interaction. What the model does with it depends on
      the model:
        - `AlternatingLeastSquares` is an **explicit-feedback** model: it minimises the
          squared error of the observed ratings and is degenerate without `y`.
        - `BayesianPersonalizedRanking` is an **implicit-feedback** model: it ignores the
          values and learns to rank observed items above unobserved ones.
        - Neighbourhood and linear item models (`ItemKNNRecommender`, `EASE`, `RP3Beta`,
          …) use `y` as a weight on the corresponding cell of $R$.
        - `MostPopularRecommender(weighting="count")` counts users, while
          `weighting="sum"` sums the values.

    MovieLens 100K, which we use from here on, is an explicit dataset: 100,000 ratings on a
    1–5 scale. Here is how the ratings are distributed:
    """)
    return


@app.cell
def _():
    movielens = fetch_movielens_100k(as_frame=True)
    ratings = movielens.frame
    X = movielens.data.to_numpy()
    y = movielens.target.to_numpy()
    titles = movielens.item_info.set_index("item_id")["title"]
    return X, ratings, titles, y


@app.cell
def _(ratings):
    _counts = ratings["rating"].value_counts().sort_index().rename_axis("rating").reset_index()
    _chart = (
        alt.Chart(_counts)
        .mark_bar()
        .encode(
            x=alt.X("rating:O", title="rating (stars)"),
            y=alt.Y("count:Q", title="number of ratings"),
            tooltip=["rating", "count"],
        )
        .properties(width=360, height=200, title="MovieLens 100K rating distribution")
    )
    mo.vstack(
        [
            _chart,
            mo.md(f"""
            Only {(ratings["rating"] <= LOW_RATING).mean():.0%} of the ratings are 1 or 2 stars.
            People mostly rate what they chose to watch, and they mostly chose well — so
            *the fact that a user rated a movie at all* is a strong signal on its own.
            That is the **implicit view** of the same data: drop `y`, keep `X`, and read
            every row as "this user watched this movie".

            The rest of this notebook uses the implicit view: models are fitted on `X`
            only, and every held-out interaction counts as relevant. Section 7 comes back
            to the explicit view and shows what changes. (The README leaderboard fits with
            the ratings as `y`, so its numbers are not directly comparable with these.)
            """),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 3. Collaborative filtering, by hand

    **Collaborative filtering** recommends from the interactions of *other* users, with no
    knowledge of what the items are. The oldest form is the **item-based neighbourhood**
    model: two items are similar when the same users interact with both, and a user is
    recommended the items most similar to the ones they already have — "people who watched
    *Alien* also watched *Blade Runner*".

    With the implicit view of the toy data, the similarity of items $i$ and $j$ is the
    cosine of their columns in the binary matrix $R$:

    $$
    s(i, j) = \frac{\text{number of users with both } i \text{ and } j}
    {\sqrt{|\text{users of } i|}\,\sqrt{|\text{users of } j|}}
    $$

    and the score of an unseen item $j$ for user $u$ is the sum of its similarities to the
    user's items: $\hat r_{uj} = \sum_{i \in I_u} s(i, j)$.
    """)
    return


@app.cell
def _(toy_matrix):
    toy_binary = toy_matrix.notna().astype(float)
    _R = toy_binary.to_numpy()
    _norms = np.sqrt(_R.sum(axis=0))
    _cosine = (_R.T @ _R) / np.outer(_norms, _norms)
    np.fill_diagonal(_cosine, 0.0)
    toy_similarity = pd.DataFrame(_cosine, index=toy_binary.columns, columns=toy_binary.columns)
    mo.vstack(
        [
            mo.md("**Item–item cosine similarity** (diagonal set to zero)"),
            toy_similarity.round(3),
        ]
    )
    return toy_binary, toy_similarity


@app.cell
def _(toy_binary, toy_similarity):
    # Scores for every user and item, with the items a user already has masked out.
    toy_scores = (toy_binary @ toy_similarity).mask(toy_binary > 0)
    mo.vstack(
        [
            mo.md(
                "**Hand-computed scores** $\\hat r_{uj}$ "
                "(empty where the user already has the item)"
            ),
            toy_scores.round(3),
        ]
    )
    return (toy_scores,)


@app.cell
def _(toy_X, toy_scores, toy_y):
    knn = ItemKNNRecommender().fit(toy_X)
    popular = MostPopularRecommender().fit(toy_X)
    _users = ["alice", "carol", "dave"]
    _knn_items, _knn_scores = knn.recommend(_users, n_recommendations=2)
    _pop_items, _ = popular.recommend(_users, n_recommendations=2)
    _pairs = np.array(
        [[_u, _i] for _u, _row in zip(_users, _knn_items, strict=True) for _i in _row]
    )
    _by_hand = [toy_scores.loc[_u, _i] for _u, _i in _pairs]

    _by_weighting = {
        _weighting: ", ".join(
            MostPopularRecommender(weighting=_weighting)
            .fit(toy_X, toy_y)
            .recommend(["eve"], n_recommendations=6, exclude_seen=False)[0][0]
        )
        for _weighting in ("count", "sum")
    }

    mo.vstack(
        [
            mo.md("**skrecsys reproduces it**: `ItemKNNRecommender().fit(X)`"),
            pd.DataFrame(
                {
                    "user": _pairs[:, 0],
                    "ItemKNN recommends": _pairs[:, 1],
                    "ItemKNN score": _knn_scores.ravel().round(3),
                    "hand-computed score": np.round(_by_hand, 3),
                    "MostPopular recommends": _pop_items.ravel(),
                }
            ),
            mo.md(f"""
            `recommend(users, n_recommendations=k)` returns two `(n_users, k)` arrays —
            item identifiers and scores, best first, with each user's own items left out.
            The ItemKNN scores match the hand computation. Note how the recommendations
            are **personalized**: *alice* and *carol*, sci-fi fans, get what the other
            sci-fi fans watched, while *dave* gets *Heat*, which *eve* — who shares his
            taste in family movies — also watched. `MostPopularRecommender` gives
            everyone the same list, minus what they have already seen: the family movies
            that are popular overall. It is not personalized at all, and yet it is a
            surprisingly hard baseline to beat, as we will see.

            Passing ratings as `y` changes what "popular" means:
            `weighting="count"` ranks {_by_weighting["count"]};
            `weighting="sum"` ranks {_by_weighting["sum"]}.
            """),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 4. A real dataset: MovieLens 100K

    `skrecsys.datasets.fetch_movielens_100k` downloads the dataset once and caches it in
    `~/skrecsys_data`. With `return_X_y=True` it returns exactly the `X`, `y` layout from
    section 1; with `as_frame=True`, as used above, it also returns user and movie
    metadata as DataFrames.
    """)
    return


@app.cell
def _(X):
    _n_users = len(np.unique(X[:, 0]))
    _n_items = len(np.unique(X[:, 1]))
    _per_user = np.unique(X[:, 0], return_counts=True)[1]
    mo.md(f"""
    | | |
    |---|---|
    | interactions | {len(X):,} |
    | users | {_n_users:,} |
    | movies | {_n_items:,} |
    | density | {len(X) / (_n_users * _n_items):.1%} of the user × movie matrix is filled |
    | interactions per user | at least {_per_user.min()}, median {np.median(_per_user):.0f} |
    """)
    return


@app.cell
def _(X, titles):
    _popularity = item_popularity(X)
    _ranked = sorted(_popularity.values(), reverse=True)
    _share = np.cumsum(_ranked) / np.sum(_ranked)
    _head = int(np.searchsorted(_share, 0.5)) + 1
    _curve = pd.DataFrame(
        {
            "movie rank by popularity": np.arange(1, len(_ranked) + 1),
            "interactions": _ranked,
        }
    )
    _chart = (
        alt.Chart(_curve)
        .mark_area(opacity=0.7)
        .encode(
            x=alt.X("movie rank by popularity:Q"),
            y=alt.Y("interactions:Q"),
            tooltip=["movie rank by popularity", "interactions"],
        )
        .properties(width=560, height=220, title="The long tail of MovieLens 100K")
    )
    _top = sorted(_popularity, key=_popularity.get, reverse=True)[:5]
    mo.vstack(
        [
            _chart,
            mo.md(f"""
            Popularity is extremely skewed: the {_head} most popular movies
            ({_head / len(_ranked):.0%} of the catalog) collect half of all interactions,
            while most movies have only a handful. The head —
            {", ".join(titles[_i] for _i in _top)} — is what `MostPopularRecommender`
            shows everyone. This skew is why accuracy alone is not enough to judge a
            recommender: a model can score well by pushing the head, and we will measure
            that with *beyond-accuracy* metrics in section 6.
            """),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 5. How evaluation works

    Offline evaluation imitates the real situation: hide some interactions, fit the model
    on the rest, ask it for a top-$k$ list per user and check how many of the hidden
    interactions it recovers.

    **Splitting.** `skrecsys.model_selection.WarmStartKFold` is a scikit-learn
    cross-validator over interactions. It pins the first interaction of every user and of
    every item (after shuffling) to the training set, then deals the remaining
    interactions round-robin into `n_splits` test folds. Every user and item in a test
    fold therefore also appears in its training fold — models such as ItemKNN cannot
    score a user or an item they have never seen, so a plain `KFold` would leave some
    test users unanswerable.

    Two caveats worth knowing:

    - a random split ignores time: the model may learn from interactions that happened
      *after* the ones it is tested on. When the order matters, sort by timestamp and use
      a time-based holdout.
    - `ColdStartSplit` builds a different test set: a fraction of users held out
      entirely (**cold** users, only non-personalized or content models can serve them)
      plus the latest interactions of every other user.
    """)
    return


@app.cell
def _(X):
    splitter = WarmStartKFold(n_splits=5, shuffle=True, random_state=0)
    train, test = next(splitter.split(X))
    X_train, X_test = X[train], X[test]
    _train_users, _train_items = set(X_train[:, 0]), set(X_train[:, 1])
    mo.md(f"""
    `train, test = next(WarmStartKFold(n_splits=5, shuffle=True, random_state=0).split(X))`

    | | interactions | users | movies |
    |---|---|---|---|
    | train | {len(X_train):,} | {len(_train_users):,} | {len(_train_items):,} |
    | test | {len(X_test):,} | {len(set(X_test[:, 0])):,} | {len(set(X_test[:, 1])):,} |

    Test users missing from train: **{len(set(X_test[:, 0]) - _train_users)}**,
    test movies missing from train: **{len(set(X_test[:, 1]) - _train_items)}**.
    """)
    return X_test, X_train, splitter, test, train


@app.cell
def _(X_test):
    mo.md(rf"""
    **From a split to a score.** For every user in the test fold:

    1. `y_true` — the set of that user's held-out items (their *relevant* items);
    2. `y_pred` — the model's top-$k$ list, from `recommend(users, n_recommendations=k)`.
       Items the user has in the training fold are excluded (`exclude_seen=True`), since
       recommending something already watched is trivial and useless;
    3. a metric compares the two lists, and the per-user values are averaged over users
       (macro average), so every user counts the same no matter how active they are.

    `skrecsys.metrics.evaluate_recommender(model, X_test, metrics=..., k=...)` does all of
    this in one call, and asks the model for a single ranking of `max(k)` items that it
    reuses for every metric and cutoff. Here the test fold has
    {len(np.unique(X_test[:, 0])):,} users, each with
    {np.median(np.unique(X_test[:, 0], return_counts=True)[1]):.0f} held-out movies at the
    median.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 6. Metrics

    The ranking metrics all have the same signature, `metric(y_true, y_pred, k=k)`: a list
    of relevant-item sets and a 2-D array of ranked items, one row per user. Take one user
    whose relevant (held-out) items are **B, E and G**, and a model that ranked ten items as
    below. Move the slider to change $k$.
    """)
    return


@app.cell
def _():
    k_slider = mo.ui.slider(1, 10, value=5, label="cutoff $k$", show_value=True)
    k_slider
    return (k_slider,)


@app.cell
def _(k_slider):
    _relevant = {"B", "E", "G"}
    _ranking = list("ABCDEFGHIJ")
    _k = k_slider.value
    _y_true, _y_pred = [_relevant], [_ranking]
    _table = pd.DataFrame(
        {
            "rank": range(1, 11),
            "item": _ranking,
            "relevant": ["✔" if _i in _relevant else "" for _i in _ranking],
            "in top-k": ["●" if _r <= _k else "" for _r in range(1, 11)],
        }
    )
    _hits = [_r for _r, _i in enumerate(_ranking[:_k], start=1) if _i in _relevant]
    _values = {
        "precision@k": precision_at_k(_y_true, _y_pred, k=_k),
        "recall@k": recall_at_k(_y_true, _y_pred, k=_k),
        "hit_rate@k": hit_rate_at_k(_y_true, _y_pred, k=_k),
        "reciprocal_rank@k": reciprocal_rank_at_k(_y_true, _y_pred, k=_k),
        "average_precision@k": average_precision_at_k(_y_true, _y_pred, k=_k),
        "ndcg@k": ndcg_at_k(_y_true, _y_pred, k=_k),
    }
    _explanations = {
        "precision@k": "share of the top-k that is relevant: hits / k",
        "recall@k": "share of the relevant items found: hits / |relevant|",
        "hit_rate@k": "1 if at least one relevant item is in the top-k, else 0",
        "reciprocal_rank@k": "1 / rank of the first hit (0 if none): rewards an early first hit",
        "average_precision@k": "mean of precision@r at each hit r, / min(|relevant|, k)",
        "ndcg@k": "Σ 1/log₂(r+1) over hits r, / the same for an ideal ranking",
    }
    mo.hstack(
        [
            _table,
            mo.vstack(
                [
                    mo.md(f"Hits in the top-{_k} at ranks: **{_hits or 'none'}**"),
                    pd.DataFrame(
                        {
                            "metric": list(_values),
                            "value": np.round(list(_values.values()), 3),
                            "definition": [_explanations[_m] for _m in _values],
                        }
                    ),
                ]
            ),
        ],
        widths=[1, 3],
    )
    return


@app.cell
def _():
    mo.md(r"""
    What each one rewards:

    - **precision@k** and **recall@k** only count hits; the order *inside* the top-$k$ does
      not matter. Precision is capped when a user has fewer than $k$ relevant items,
      recall when they have more.
    - **hit_rate@k** asks the simplest question — did we show at least one good item? It
      saturates quickly and is best for small $k$.
    - **reciprocal_rank@k** (MRR when averaged) cares only about the first hit — the right
      metric when a user looks at one result.
    - **average_precision@k** (MAP) and **ndcg@k** reward putting *all* hits near the top.
      NDCG discounts a hit at rank $r$ by $1/\log_2(r+1)$ and normalises by the best
      possible ranking, so it lies in $[0, 1]$. It is the usual headline metric, and the one
      the skrecsys leaderboard reports.

    **Beyond accuracy.** Accuracy metrics favour whatever users already interact with most,
    so they are paired with metrics that describe *what* is recommended, using no ground
    truth:

    - `catalog_coverage_at_k` — the share of the catalog that appears in anyone's top-$k$;
    - `mean_popularity_at_k` — the average training popularity of recommended items
      (popularity bias);
    - `novelty_at_k` — mean self-information $-\log_2 p(i)$ of recommended items: high when
      the model recommends items few users know.

    The last two need the training popularity of each item, computed by
    `item_popularity(X_train)` and bound with `functools.partial`.
    """)
    return


@app.cell
def _(X, X_train):
    _train_popularity = item_popularity(X_train)
    metrics = {
        "precision": precision_at_k,
        "recall": recall_at_k,
        "hit_rate": hit_rate_at_k,
        "mrr": reciprocal_rank_at_k,
        "map": average_precision_at_k,
        "ndcg": ndcg_at_k,
        "coverage": partial(catalog_coverage_at_k, n_catalog_items=len(np.unique(X[:, 1]))),
        "mean_popularity": partial(mean_popularity_at_k, item_popularity=_train_popularity),
        "novelty": partial(novelty_at_k, item_popularity=_train_popularity),
    }
    return (metrics,)


@app.cell
def _():
    mo.md(r"""
    ## 7. Baseline models

    `skrecsys.recommendation` contains the classical collaborative-filtering models. All of
    them share the scikit-learn estimator API — `fit`, `recommend`, `predict`,
    `partial_fit`, `get_params` / `set_params` — so they are interchangeable in the code
    below.

    | model | idea |
    |---|---|
    | `MostPopularRecommender` | the same most-interacted items for everyone; the floor every model must beat |
    | `ItemKNNRecommender` | item–item cosine similarity, as in section 3, kept to the `n_neighbors` nearest items |
    | `BM25Recommender` | item–item similarity on BM25-weighted interactions, which dampens heavy users and popular items |
    | `RP3Beta` | a three-step random walk user → item → user → item, with popular items penalised by $\text{popularity}^\beta$ |
    | `EASE` | a closed-form linear item–item model: ridge regression of each item's column on all the others, zero diagonal |
    | `SLIMElasticNet` | the same idea as EASE with an elastic-net penalty, giving a sparse item–item matrix |
    | `BayesianPersonalizedRanking` | matrix factorization trained to rank observed items above sampled unobserved ones (implicit feedback) |
    | `AlternatingLeastSquares` | matrix factorization with biases, trained on the squared error of observed ratings (explicit feedback) |

    Every model below uses its default hyperparameters and is fitted on the training fold
    from section 5. ALS is fitted with the ratings as `y`, since it needs them; the others
    are fitted on `X` alone. Choose which models to compare:
    """)
    return


@app.cell
def _():
    model_factories = {
        "MostPopular": MostPopularRecommender,
        "ItemKNN": ItemKNNRecommender,
        "BM25": BM25Recommender,
        "RP3Beta": RP3Beta,
        "EASE": EASE,
        "SLIM": SLIMElasticNet,
        "BPR": partial(BayesianPersonalizedRanking, random_state=0),
        "ALS": partial(AlternatingLeastSquares, random_state=0),
    }
    explicit_models = {"ALS"}
    model_picker = mo.ui.multiselect(
        options=list(model_factories), value=list(model_factories), label="models"
    )
    model_picker
    return explicit_models, model_factories, model_picker


@app.cell
def _(X_train, explicit_models, model_factories, model_picker, train, y):
    fitted = {}
    fit_seconds = {}
    for _name in mo.status.progress_bar(model_picker.value, title="Fitting models"):
        _model = model_factories[_name]()
        _t0 = time.perf_counter()
        if _name in explicit_models:
            _model.fit(X_train, y[train])
        else:
            _model.fit(X_train)
        fit_seconds[_name] = time.perf_counter() - _t0
        fitted[_name] = _model
    return fit_seconds, fitted


@app.cell
def _(X_test, fitted, metrics):
    results = pd.DataFrame(
        {
            _name: evaluate_recommender(_model, X_test, metrics=metrics, k=[10, 20])
            for _name, _model in fitted.items()
        }
    ).T.rename_axis("model")
    results.round(4)
    return (results,)


@app.cell
def _(fit_seconds, results):
    mo.stop(results.empty, mo.md("*Pick at least one model.*"))
    _points = results[["ndcg@10", "coverage@10", "novelty@10"]].reset_index()
    _chart = (
        alt.Chart(_points)
        .mark_circle(size=140)
        .encode(
            x=alt.X("coverage@10:Q", title="catalog coverage@10", scale=alt.Scale(zero=False)),
            y=alt.Y("ndcg@10:Q", title="NDCG@10"),
            color=alt.Color("novelty@10:Q", title="novelty@10"),
            tooltip=["model", "ndcg@10", "coverage@10", "novelty@10"],
        )
        .properties(width=420, height=280, title="Accuracy vs. catalog coverage")
    )
    _labels = _chart.mark_text(align="left", dx=9).encode(text="model")
    _best = results["ndcg@10"].idxmax()
    _personalized = results.drop(index=["MostPopular", "ALS"], errors="ignore")
    _widest = _personalized["coverage@10"].idxmax() if len(_personalized) else _best
    _item_based = [_m for _m in ("ItemKNN", "BM25", "RP3Beta", "EASE") if _m in results.index]
    _factor = [_m for _m in ("BPR", "ALS") if _m in results.index]
    _most_novel = results["novelty@10"].idxmax()
    _als_rank = _novelty_rank = None
    if "ALS" in results.index:
        _als_rank = int(results["ndcg@10"].rank(ascending=False)["ALS"])
        _novelty_rank = int(results["novelty@10"].rank(ascending=False)["ALS"])
    mo.hstack(
        [
            _chart + _labels,
            mo.md(f"""
            **Reading the results**

            - **{_best}** has the best NDCG@10 on this split.
            - `MostPopular` is not personalized, yet it finds at least one relevant
              movie for {results.loc["MostPopular", "hit_rate@10"]:.0%} of the users
              — the popularity skew at work. It covers
              {results.loc["MostPopular", "coverage@10"]:.1%} of the catalog.
            - Accuracy and coverage pull apart: among the personalized models,
              **{_widest}** covers the most of the catalog
              ({results.loc[_widest, "coverage@10"]:.0%}) and **{_best}** only
              {results.loc[_best, "coverage@10"]:.0%}.
            - The item-based models ({", ".join(_item_based)}) fit in at most
              {max(fit_seconds[_m] for _m in _item_based):.2f} s each, and the best of
              them reaches NDCG@10 {results.loc[_item_based, "ndcg@10"].max():.4f}
              against {results.loc[_factor, "ndcg@10"].max():.4f} for the best
              latent-factor model — simple models are strong baselines.
            - `ALS` minimises rating error, not ranking error. It ranks
              {_als_rank} of {len(results)} by NDCG@10, and its novelty is
              {"the highest" if _most_novel == "ALS" else f"{_novelty_rank} of {len(results)}"}
              ({results.loc["ALS", "novelty@10"]:.2f} bits): it recommends obscure movies
              with high predicted ratings that few users ever watch — see below.
              This is *explicit* ALS; the implicit variant (iALS, Hu et al. 2008), which
              weighs every unobserved pair as a weak negative, is a strong ranking
              baseline, so do not read this as "matrix factorization cannot rank".
            """)
            if {"MostPopular", "ALS"} <= set(results.index)
            and set(_item_based) == {"ItemKNN", "BM25", "RP3Beta", "EASE"}
            and set(_factor) == {"BPR", "ALS"}
            else mo.md(f"**{_best}** has the best NDCG@10 of the models picked."),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ### Is the difference real?

    A table of means hides how noisy they are. Two models evaluated on the *same* users
    should be compared **per user**: `ndcg_at_k(..., average=None)` returns one value per
    user instead of the mean, and resampling users with replacement (a **paired
    bootstrap**) shows how much the difference of the means would move on another sample
    of users. If the 95% interval contains zero, the table's order is not evidence.
    """)
    return


@app.cell
def _(X_test, fitted, results):
    mo.stop(len(results) <= 1, mo.md("*Pick at least two models.*"))
    _users = np.unique(X_test[:, 0])
    _relevant = pd.Series(X_test[:, 1]).groupby(X_test[:, 0]).agg(set)[_users].tolist()
    _first, _second = results["ndcg@10"].nlargest(2).index
    _per_user = {
        _name: ndcg_at_k(
            _relevant, fitted[_name].recommend(_users, n_recommendations=10)[0], average=None
        )
        for _name in (_first, _second)
    }
    _diff = _per_user[_first] - _per_user[_second]
    _rng = np.random.default_rng(0)
    _boot = _diff[_rng.integers(len(_diff), size=(2000, len(_diff)))].mean(axis=1)
    _low, _high = np.quantile(_boot, [0.025, 0.975])
    mo.md(f"""
    **{_first}** against the runner-up **{_second}**: mean per-user difference in NDCG@10
    {_diff.mean():+.4f}, 95% bootstrap interval [{_low:+.4f}, {_high:+.4f}] over
    {len(_users):,} users. {_first} is better for {(_diff > 0).mean():.0%} of the users and
    worse for {(_diff < 0).mean():.0%}.
    {"The interval excludes zero, so on this split the gap is more than noise." if _low > 0 else "The interval contains zero: on this split the two are not distinguishable."}
    Resampling users answers "would it hold for other users?", not "would it hold on
    another split?" — cross-validation in section 9 addresses the second.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Explicit vs implicit, revisited

    **A good rating predictor is not a good recommender.** ALS was trained on the ratings,
    so let's judge it the explicit way — by the root-mean-square error of its predicted
    ratings on the held-out pairs, with `predict(pairs)` — and compare with the trivial
    prediction "every rating is the training average".
    """)
    return


@app.cell
def _(X_test, X_train, test, train, y):
    _als = AlternatingLeastSquares(random_state=0).fit(X_train, y[train])
    _rmse_als = np.sqrt(np.mean((_als.predict(X_test) - y[test]) ** 2))
    _rmse_mean = np.sqrt(np.mean((y[train].mean() - y[test]) ** 2))
    mo.md(f"""
    | predictor | RMSE on held-out ratings |
    |---|---|
    | training mean | {_rmse_mean:.3f} |
    | `AlternatingLeastSquares` | **{_rmse_als:.3f}** |

    ALS predicts ratings well, clearly better than the mean. It still ranks poorly,
    because it is only ever trained on movies users *chose* to rate: it learns how much a
    user will like a movie *given that they watch it*, and has never seen the pairs nobody
    picked. A ranking model has to separate "would watch" from "would never
    watch" across the whole catalog — the implicit-feedback question.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    **The ground truth is a choice, too.** So far every held-out rating counted as
    relevant, even a 1-star one. With explicit data you can instead count only the movies
    the user *liked*: `evaluate_recommender(model, X_test, y_test, ...)` treats pairs with
    `y <= 0` as not relevant, so passing `y_test = (rating >= 4)` evaluates against liked
    movies only. Users without a single liked movie in the test fold then have nothing to
    find and are left out.
    """)
    return


@app.cell
def _(X_test, fitted, test, y):
    _liked = (y[test] >= LIKED_RATING).astype(float)
    _compare = pd.DataFrame(
        {
            _name: {
                "NDCG@10, every rating relevant": evaluate_recommender(
                    _model, X_test, metrics=[ndcg_at_k], k=10
                )["ndcg@10"],
                "NDCG@10, rating ≥ 4 relevant": evaluate_recommender(
                    _model, X_test, _liked, metrics=[ndcg_at_k], k=10
                )["ndcg@10"],
            }
            for _name, _model in fitted.items()
        }
    ).T.rename_axis("model")
    _every, _liked_only = _compare.columns
    _rho = _compare[_every].rank().corr(_compare[_liked_only].rank())
    mo.vstack(
        [
            _compare.round(4),
            mo.md(f"""
            Scores drop from a mean of {_compare[_every].mean():.4f} to
            {_compare[_liked_only].mean():.4f}, since there are fewer relevant items per
            user, and the rank correlation of the two orders of models is
            {_rho:.2f}{": models fitted on the implicit signal recommend movies users go on to like, not just movies they go on to watch" if _rho > STABLE_RANK_CORRELATION else ", so the ground truth changes which model looks best"}.
            Whichever convention you pick, state it — numbers computed under different
            ground truths are not comparable.
            """),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 8. Look at the recommendations

    Metrics summarise thousands of lists; it is worth reading a few. Pick a user to see
    the movies they rated highest in the training fold and what three models recommend.
    """)
    return


@app.cell
def _(X_test):
    user_picker = mo.ui.dropdown(
        options=[str(_u) for _u in np.unique(X_test[:, 0])[:50]],
        value=str(np.unique(X_test[:, 0])[0]),
        label="user",
    )
    user_picker
    return (user_picker,)


@app.cell
def _(X_test, X_train, fitted, titles, train, user_picker, y):
    _user = int(user_picker.value)
    _own = X_train[:, 0] == _user
    _favourites = X_train[_own, 1][np.argsort(-y[train][_own], kind="stable")][:10]
    _held_out = set(X_test[X_test[:, 0] == _user, 1])
    _columns = {"rated highest (train)": [titles[_i] for _i in _favourites]}
    for _name in [_m for _m in ("MostPopular", "ItemKNN", "EASE") if _m in fitted]:
        _items, _ = fitted[_name].recommend([_user], n_recommendations=10)
        _columns[_name] = [
            f"✔ {titles[_i]}" if _i in _held_out else titles[_i] for _i in _items[0]
        ]
    mo.vstack(
        [
            pd.DataFrame(_columns, index=pd.RangeIndex(1, 11, name="rank")),
            mo.md("✔ marks a recommendation the user actually watched in the test fold."),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 9. Cross-validation with scikit-learn

    One split gives one number; a different random split gives a slightly different one.
    Because skrecsys recommenders are scikit-learn estimators and `WarmStartKFold` is a
    scikit-learn cross-validator, the standard tools work unchanged.
    `make_recommender_scorer` turns ranking metrics into a scorer, and with a list of
    metrics it becomes a multi-metric scorer:

    ```python
    cross_validate(
        EASE(),
        X,
        cv=WarmStartKFold(n_splits=5, shuffle=True, random_state=0),
        scoring=make_recommender_scorer([ndcg_at_k, recall_at_k], k=10),
    )
    ```

    The same scorer plugs into `GridSearchCV` for hyperparameter search.
    """)
    return


@app.cell
def _():
    cv_button = mo.ui.run_button(label="Run 5-fold cross-validation")
    cv_button
    return (cv_button,)


@app.cell
def _(X, cv_button, splitter):
    mo.stop(
        not (cv_button.value or mo.app_meta().mode == "script"),
        mo.md("*Press the button to cross-validate four models (a few seconds).*"),
    )
    _scorer = make_recommender_scorer([ndcg_at_k, recall_at_k], k=10)
    _rows = []
    for _name, _model in mo.status.progress_bar(
        {
            "MostPopular": MostPopularRecommender(),
            "ItemKNN": ItemKNNRecommender(),
            "RP3Beta": RP3Beta(),
            "EASE": EASE(),
        }.items(),
        title="Cross-validating",
    ):
        _scores = cross_validate(_model, X, cv=splitter, scoring=_scorer)
        _row = {"model": _name}
        for _metric in ("ndcg@10", "recall@10"):
            _fold_scores = _scores[f"test_{_metric}"]
            _row[_metric] = f"{_fold_scores.mean():.4f} ± {_fold_scores.std():.4f}"
            _row[f"_{_metric}"] = _fold_scores
        _row["fit time (s)"] = round(_scores["fit_time"].mean(), 3)
        _rows.append(_row)
    _table = pd.DataFrame(_rows).set_index("model")
    _folds = pd.DataFrame(dict(zip(_table.index, _table["_ndcg@10"], strict=True)))
    _means = _folds.mean().sort_values(ascending=False)
    _smallest_gap = (-_means.diff()).min()
    _largest_std = _folds.std(ddof=0).max()
    # The folds are shared, so the same order in every fold is the paired evidence.
    _ranks = _folds.rank(axis=1, ascending=False)
    _same_order = bool(_ranks.eq(_ranks.iloc[0]).all().all())
    mo.vstack(
        [
            _table.drop(columns=["_ndcg@10", "_recall@10"]),
            mo.md(f"""
            Mean ± standard deviation over the five folds. The smallest gap between two
            neighbouring models is {_smallest_gap:.4f} NDCG@10, the largest fold spread
            {_largest_std:.4f}. The folds are shared, so compare the models fold by fold:
            {"the order is the same in every fold, so the ranking from the single split above holds." if _same_order else "the order changes between folds for at least one pair, so treat the closest models as tied."}
            """),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 10. Takeaways

    - Recommendation is a **ranking** problem over what the user has not seen yet.
    - **Implicit feedback** (did the user interact?) is abundant and is what most models
      rank with; **explicit feedback** (how much did they like it?) answers a different
      question, as ALS shows. In skrecsys the difference is whether and how you pass `y`.
    - Evaluate with a split that keeps test users and items in training
      (`WarmStartKFold`), one top-$k$ list per user with seen items excluded, and several
      metrics at once (`evaluate_recommender`) — both accuracy and beyond-accuracy.
    - Always compare against `MostPopularRecommender`. Simple item-based models such as
      EASE and RP3Beta are strong baselines.
    - Before believing a difference, compare models **per user** on the same test set
      (`average=None` and a paired bootstrap) and fold by fold in cross-validation.

    Next in the series:

    2. [Model selection and tuning](02_model_selection_and_tuning.py) — `GridSearchCV`,
       `AutoTune` and `Study` from `skrecsys.tune`, and splits that respect time.
    3. [Candidates and ranking](03_candidates_and_ranking.py) — two-stage recommenders
       and cold start with `skrecsys.compose`.
    4. [From notebook to production](04_production.py) — pickled models, `partial_fit`,
       request-time controls and vector indexes.
    5. [Sequential and neural recommenders](05_sequential_and_neural.py) — predicting the
       next item with `skrecsys.nn`.
    6. [Time-aware recommendation](06_time_aware_recommendation.py) — time-based
       evaluation and tuning, recency, point-in-time features and bitemporal data.
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
                    '1. Why are recommenders usually judged with top-k ranking metrics rather than rating error (RMSE)?': mo.md(
                        'The product shows a short ranked list of items the user has not seen. RMSE only measures the ratings a user chose to give, and says nothing about the thousands of items they never picked. Section 7 shows this: ALS predicts ratings clearly better than the mean, yet ranks worst of all models.'
                    ),
                    '2. In implicit feedback, what does a missing (user, item) pair mean?': mo.md(
                        '*Unknown*, not *disliked*: the user may not know the item exists. Implicit data has no true negatives, which is why models such as BPR learn to rank observed items above unobserved ones rather than to predict a value.'
                    ),
                    "3. A user has 3 relevant items. A model's top 5 contains them at ranks 2 and 5. What are precision@5, recall@5 and the reciprocal rank?": mo.md(
                        'precision@5 = 2/5 = 0.4; recall@5 = 2/3 ≈ 0.667; reciprocal rank = 1/2 = 0.5 (the first hit is at rank 2). Check with the slider in section 6: the same example with relevant items B, E, G.'
                    ),
                    '4. Why does `WarmStartKFold` pin the first interaction of every user and every item to the training set?': mo.md(
                        'A collaborative-filtering model cannot score a user or an item it never saw in training. Pinning guarantees that every test user and item also appears in its training fold, so every test interaction can be answered.'
                    ),
                    '5. Why does evaluation call `recommend(..., exclude_seen=True)`?': mo.md(
                        "The held-out items are, by construction, not in the user's training history. Leaving training items in the list would fill top-k slots with things the user already has, which is useless as a recommendation and would score as misses."
                    ),
                    '6. `MostPopular` finds a relevant item for most users. Which metrics show what is wrong with it?': mo.md(
                        'Beyond-accuracy metrics: `catalog_coverage_at_k` (it recommends a few percent of the catalog), `mean_popularity_at_k` (high) and `novelty_at_k` (low). It is not personalized at all.'
                    ),
                    '7. You evaluate with `y_test = (rating >= 4)`. What happens to a test user whose held-out ratings are all 3 or below?': mo.md(
                        'Pairs with `y <= 0` are not relevant, so this user has nothing to find and is left out of the average. The scores then describe "finding liked movies" for a smaller set of users.'
                    ),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
