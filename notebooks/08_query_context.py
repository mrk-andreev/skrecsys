import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    from skrecsys.compose import (
        Cascade,
        ConcatFeatures,
        GeneratorScores,
        InteractionCounts,
        JoinDynamicFeatures,
        PointwiseRanker,
    )
    from skrecsys.datasets import fetch_movielens_100k
    from skrecsys.metrics import evaluate_recommender, hit_rate_at_k, ndcg_at_k
    from skrecsys.recommendation import EASE

    K = 10
    N_RETRIEVED = 100
    DAY = 86_400
    THURSDAY, SATURDAY = 3, 5

    def weekend_of(keys):
        """The weekend flag of each distinct ``[shelf, weekend]`` context row, as a feature.

        The callback of ``JoinDynamicFeatures("context", ...)``. A request without context
        has NaN for its flag, and so does its feature: this cascade's logistic ranker must
        be asked with context.
        """
        return np.asarray(keys[:, 1], dtype=float)

    class OnShelf:
        """Whether a movie is in the genre of the shelf a request came from.

        The callback of ``JoinDynamicFeatures("item-context", ...)``: it gets the distinct
        ``[item, shelf, weekend]`` rows of a batch of candidates and answers 1.0 when the
        item has the shelf's genre. A request without context has NaN for a shelf and
        gets 0.0. A class rather than a closure, so that the cascade pickles.
        """

        def __init__(self, genres, genre_names):
            self.genres = genres
            self.index = {name: j for j, name in enumerate(genre_names)}

        def __call__(self, keys):
            out = np.zeros((len(keys), 1))
            for row, (item, shelf, *_) in enumerate(keys):
                if isinstance(shelf, str):
                    out[row, 0] = self.genres[int(item), self.index[shelf]]
            return out


@app.cell
def _():
    mo.md(r"""
    # 8. Query context

    Every model so far answered one question: *what does this user like?* A real request
    says more than who is asking. It comes from a page, a search, a device, a time of day:
    its **query context**. The same user wants comedies on the comedy shelf and horror on
    the horror shelf. A model that knows only the user serves both requests the same list.

    In skrecsys the context is extra columns:

    - `fit(X)` takes `[user, item, context...]`, or `[user, item, time, context...]` for
      a recommender constructed with `time=True`: each interaction with the request it
      answered;
    - `recommend(X)` takes a vector of users, which asks without context, or a matrix
      `[user, context...]`, one row per request.

    Every recommender accepts the context. The ranker of a `Cascade` is what learns from it.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 1. A log with context

    MovieLens 100K has no request log, so we make one up. Each rating gets a **shelf**: one
    of the rated movie's genres, drawn at random, as if the user had found the movie while
    browsing that genre. The shelf is known before the click and the rated movie is on it,
    which is what makes it a useful context. Real contexts are rarely that informative, and
    how much a real one helps has to be measured, as below. The second context column is
    real: whether the rating was made on a **weekend**, from its timestamp.

    Ratings are sorted by time, with a random tie-breaker because MovieLens users rate in
    batches. The split is a **global** cutoff at the 80% point in time, so nothing after the
    cutoff leaks into training. The test set is the later ratings of users the training
    part knows.
    """)
    return


@app.cell
def _():
    _movielens = fetch_movielens_100k()
    _rng = np.random.default_rng(0)
    _order = np.lexsort((_rng.random(len(_movielens.data)), _movielens.timestamps))
    _log = _movielens.data[_order]
    _times = _movielens.timestamps[_order]
    genre_names = list(_movielens.genre_names)
    genres = np.zeros((_log[:, 1].max() + 1, len(genre_names)), dtype=bool)
    genres[_movielens.item_info.item_id] = _movielens.item_info.genres
    titles = dict(zip(_movielens.item_info.item_id, _movielens.item_info.title, strict=True))
    _names = np.array(genre_names, dtype=object)
    _shelf = np.array(
        [_names[_rng.choice(np.flatnonzero(genres[item]))] for item in _log[:, 1]], dtype=object
    )
    # 1970-01-01, day 0 of Unix time, was a Thursday: weekday 3 counting from Monday = 0.
    _weekend = ((_times // DAY + THURSDAY) % 7 >= SATURDAY).astype(int)
    X = np.empty((len(_log), 4), dtype=object)
    X[:, 0], X[:, 1], X[:, 2], X[:, 3] = _log[:, 0], _log[:, 1], _shelf, _weekend
    _cut = int(0.8 * len(X))
    X_train = X[:_cut]
    _later = X[_cut:]
    X_test = _later[np.isin(_later[:, 0].astype(int), X_train[:, 0].astype(int))]
    mo.vstack(
        [
            pd.DataFrame(X[:5], columns=["user", "item", "shelf", "weekend"]),
            mo.md(
                f"{len(X_train):,} training ratings and {len(X_test):,} test ratings from "
                f"{len(np.unique(X_test[:, 0]))} users the training part knows. "
                f"{X[:, 3].mean():.0%} of the ratings were made on a weekend."
            ),
        ]
    )
    return X_test, X_train, genre_names, genres, titles


@app.cell
def _():
    mo.md(r"""
    ## 2. Every recommender accepts context

    A dataset with context columns fits any model unchanged. `EASE`, like everything in
    `skrecsys.recommendation`, scores by user alone: it accepts the extra columns in `fit`
    and a matrix of queries in `recommend`, and ignores both.
    """)
    return


@app.cell
def _(X_test, X_train):
    _with = EASE().fit(X_train)
    _without = EASE().fit(X_train[:, :2])
    _user = X_test[0, 0]
    _asked = _with.recommend([[_user, "Comedy", 0], [_user, "Horror", 1]], n_recommendations=K)[0]
    _plain = _without.recommend([_user], n_recommendations=K)[0]
    _same = bool((_asked == _plain).all())
    mo.md(
        f"EASE fitted with and without the context, asked for user {_user} on the comedy and "
        f"the horror shelf: the lists are {'identical' if _same else 'different'}. Only a "
        "model that reads the context can use it."
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 3. A ranker that reads the context

    A `Cascade` retrieves candidates by user and then reranks them per request. The
    generator is EASE, retrieving 100 candidates. The ranker is a logistic regression,
    and it gets three versions of the features:

    1. **no context**: the EASE score and the movie's popularity;
    2. **+ weekend**: the same, plus the weekend flag as it is,
       `JoinDynamicFeatures("context", weekend_of)`;
    3. **+ shelf match**: the same as 1, plus a feature of the candidate *and* the request
       together, `JoinDynamicFeatures("item-context", OnShelf(...))`. It is 1 when the
       movie has the shelf's genre.

    To train the ranker, the cascade holds out each user's latest 20% of ratings. It ranks
    candidates for each user with the context of their **first** held-out rating, the
    request those ratings answer.

    These are three fixed designs, not tuned on the test set. Comparing them there is an
    evaluation, not a model selection.
    """)
    return


@app.cell
def _(X_train, genre_names, genres):
    _base = [GeneratorScores(), InteractionCounts("item")]
    _designs = {
        "no context": ConcatFeatures(_base),
        "+ weekend": ConcatFeatures([*_base, JoinDynamicFeatures("context", weekend_of)]),
        "+ shelf match": ConcatFeatures(
            [*_base, JoinDynamicFeatures("item-context", OnShelf(genres, genre_names))]
        ),
    }
    cascades = {
        _name: Cascade(
            EASE(),
            _features,
            PointwiseRanker(LogisticRegression(max_iter=2000)),
            n_retrieved=N_RETRIEVED,
        ).fit(X_train)
        for _name, _features in _designs.items()
    }
    return (cascades,)


@app.cell
def _():
    mo.md(r"""
    ## 4. Evaluating per request

    Each test rating is a request: the user, the shelf and the weekend flag are the query,
    and the rated movie is the one it should find. `recommend` gets one row per request,
    `[user, shelf, weekend]`, so one user with ten test ratings is asked ten times, each
    time with its own context.
    """)
    return


@app.cell
def _(X_test, cascades):
    _queries = X_test[:, [0, 2, 3]]
    _wanted = [{item} for item in X_test[:, 1]]
    per_request = {}
    served = {}
    _rows = []
    for _name, _rec in cascades.items():
        _items, _ = _rec.recommend(_queries, n_recommendations=K)
        served[_name] = _items
        per_request[_name] = hit_rate_at_k(_wanted, _items, k=K, average=None)
        _rows.append(
            {
                "features": _name,
                f"hit rate@{K}": per_request[_name].mean(),
                f"NDCG@{K}": ndcg_at_k(_wanted, _items, k=K),
            }
        )
    request_table = pd.DataFrame(_rows).set_index("features").round(4)
    request_table
    return per_request, request_table, served


@app.cell
def _(X_test, per_request, request_table, served):
    # Requests of one user are not independent, so the bootstrap resamples users.
    _users, _codes = np.unique(X_test[:, 0].astype(int), return_inverse=True)
    _diff = per_request["+ shelf match"] - per_request["no context"]
    _per_user = np.bincount(_codes, weights=_diff) / np.bincount(_codes)
    _weights = np.bincount(_codes) / len(_codes)
    _rng = np.random.default_rng(0)
    _draws = _rng.integers(len(_users), size=(2000, len(_users)))
    _boot = (_per_user[_draws] * _weights[_draws]).sum(axis=1) / _weights[_draws].sum(axis=1)
    _low, _high = np.quantile(_boot, [0.025, 0.975])
    _hits = request_table[f"hit rate@{K}"]
    _plain, _weekend = served["no context"], served["+ weekend"]
    _same_order = (_plain == _weekend).all(axis=1).mean()
    _same_items = (np.sort(_plain, axis=1) == np.sort(_weekend, axis=1)).all(axis=1).mean()
    mo.md(f"""
    The shelf match lifts the hit rate from {_hits["no context"]:.1%} to
    {_hits["+ shelf match"]:.1%} of requests: a difference of {_diff.mean():+.2%}, with a 95%
    interval of [{_low:+.2%}, {_high:+.2%}] from a bootstrap over the {len(_users)} test
    users. {"The interval excludes zero, so on this split the gain is more than noise." if _low > 0 else "The interval contains zero, so on this split the gain is not distinguishable from noise."}

    The weekend flag, passed to the ranker as it is, does next to nothing: the two
    rankers serve the same top {K} in the same order for {_same_order:.1%} of requests,
    and the same items for {_same_items:.1%}. That is expected. A logistic ranker scores
    each candidate with a weighted sum, and the flag is the same for every candidate of a
    request. It adds one constant to all of their scores and cannot change their order.
    What differs comes from training, where the extra column nudges the weights of the
    other features. A context feature reorders candidates only when it meets something
    that differs between them. That can be a feature built from the pair, as the shelf
    match is, or a ranker that learns interactions, such as gradient-boosted trees given
    the raw context next to item features.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 5. One user, several requests

    A vector asks without context. The shelf-match feature sees NaN and is 0 for every
    candidate, so the ranker falls back on the rest of its features. A matrix asks with
    context, one row per request, here for one user on three shelves:
    """)
    return


@app.cell
def _(X_test, cascades, genre_names, genres, titles):
    _rec = cascades["+ shelf match"]
    _user = X_test[0, 0]
    _shelves = ["Comedy", "Horror", "Children's"]
    _plain, _ = _rec.recommend([_user], n_recommendations=5)
    _asked, _ = _rec.recommend([[_user, _shelf, 0] for _shelf in _shelves], n_recommendations=5)
    _columns = {"no context": [titles[int(i)] for i in _plain[0]]}
    for _shelf, _items in zip(_shelves, _asked, strict=True):
        _columns[f"shelf {_shelf}"] = [titles[int(i)] for i in _items]
    _changed = [
        _shelf
        for _shelf, _items in zip(_shelves, _asked, strict=True)
        if (_items != _plain[0]).any()
    ]
    _unchanged = [_shelf for _shelf in _shelves if _shelf not in _changed]

    def _listed(shelves):
        if not shelves:
            return "no shelf"
        return ", ".join(shelves) + (" shelves" if len(shelves) > 1 else " shelf")

    _retrieved, _ = _rec.generator_.recommend([_user], n_recommendations=N_RETRIEVED)
    _on_shelf = {
        _shelf: int(genres[_retrieved[0].astype(int), genre_names.index(_shelf)].sum())
        for _shelf in _shelves
    }
    mo.vstack(
        [
            mo.md(f"**User {_user}, top 5**"),
            pd.DataFrame(_columns),
            mo.md(
                f"The list changes on the {_listed(_changed)} and stays as without context "
                f"on the {_listed(_unchanged)}. Of the {N_RETRIEVED} candidates EASE retrieved "
                "for this user, "
                + ", ".join(f"{n} are {_shelf}" for _shelf, n in _on_shelf.items())
                + " movies. The shelf match is one feature among three: a movie on the shelf "
                "rises only as far as its EASE score allows, and the ranker never sees a movie "
                "the generator did not retrieve."
            ),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## 6. What `evaluate_recommender` measures

    `evaluate_recommender` and `make_recommender_scorer`, which drive cross-validation and
    `AutoTune`, group the held-out rows **by user**. They ask each user once, with the
    context of the user's **first** held-out row, and count every held-out movie as
    relevant. That is how the cascade trains its ranker, and it keeps model selection
    working with context unchanged. But a user's later requests came from other shelves,
    so the per-user score understates what context does per request:
    """)
    return


@app.cell
def _(X_test, cascades, request_table):
    _by_user = {
        _name: evaluate_recommender(_rec, X_test, metrics=[ndcg_at_k], k=K)[f"ndcg@{K}"]
        for _name, _rec in cascades.items()
    }
    _table = pd.DataFrame(
        {
            f"per-user NDCG@{K}": pd.Series(_by_user),
            f"per-request NDCG@{K}": request_table[f"NDCG@{K}"],
        }
    ).round(4)
    _gain_user = _by_user["+ shelf match"] - _by_user["no context"]
    _gain_request = (
        request_table[f"NDCG@{K}"]["+ shelf match"] - request_table[f"NDCG@{K}"]["no context"]
    )
    mo.vstack(
        [
            _table,
            mo.md(
                f"Per user, the shelf match changes NDCG@{K} by {_gain_user:+.4f}; per "
                f"request, by {_gain_request:+.4f}. The per-user numbers are higher overall "
                "because a user's list may hit any of their held-out movies, not just one. "
                "When requests carry their own context, evaluate one row per request, as "
                "section 4 does."
            ),
        ]
    )
    return


@app.cell
def _():
    mo.md(r"""
    ## Takeaways

    - Context columns come after the system columns: `[user, item, context...]` in `fit`,
      `[user, context...]` in `recommend`. A vector of users asks without context.
    - Every recommender accepts context. Those that cannot use it ignore it, so one
      dataset serves every model.
    - A `Cascade` passes the context to its features and ranker, and its generators still
      retrieve by user. The ranker trains on each held-out user's first held-out request.
    - `JoinDynamicFeatures` keyed by `"context"` turns the context into features.
      `"item-context"` builds a feature of the candidate and the request together, which
      is what a linear ranker needs to reorder by context.
    - Evaluate per request when requests carry their own context. The per-user scorers ask
      each user once, with the first held-out context.
    """)
    return


@app.cell
def _():
    mo.vstack(
        [
            mo.md("## Check yourself\n\nTry to answer each question before opening it."),
            mo.accordion(
                {
                    "1. How do you ask `recommend` for the same user with two different contexts?": mo.md(
                        "Pass a matrix with one row per request: `rec.recommend([[user, 'Comedy', 0], [user, 'Horror', 0]])`. Column 0 is the user and the other columns are the context, laid out like the context columns of the `X` of `fit`. A vector `[user]` asks without context."
                    ),
                    "2. You fit `EASE` on `[user, item, shelf]` rows. What happens to the shelf?": mo.md(
                        "Nothing: EASE accepts the column and ignores it, and its recommendations are identical to a fit on `[user, item]` (section 2). Only a model that reads the context, such as the ranker of a `Cascade`, can use it."
                    ),
                    "3. Why did the weekend flag leave the logistic ranker's lists almost unchanged?": mo.md(
                        "The flag is the same for every candidate of a request, and a logistic ranker adds its weighted features up. So the flag adds one constant to every candidate's score and cannot change their order; the few lists that differ come from the other weights shifting in training. A context feature needs something that varies between candidates: a feature built from the pair, such as the shelf match, or a ranker that learns interactions, such as gradient-boosted trees."
                    ),
                    "4. Which context does `Cascade.fit` rank a held-out user's candidates with?": mo.md(
                        "The context of the user's first held-out interaction: the earliest by time with `time=True`, otherwise the first by row. That interaction is the request the held-out items answer, and it is the same moment `time=True` ranks as of."
                    ),
                    "5. What do the features see for a query asked without context?": mo.md(
                        "NaN in every context column. A `JoinDynamicFeatures` callback gets NaN in the context part of its keys, so it should handle that case: `OnShelf` answers 0, while `weekend_of` passes the NaN on, which only a ranker that accepts missing values, such as gradient-boosted trees, can take. The logistic ranker here would raise, so that cascade must be asked with context."
                    ),
                    "6. Why do the per-user and per-request NDCG disagree in section 6?": mo.md(
                        "`evaluate_recommender` asks each user once, with the context of their first held-out row, and counts every held-out movie as relevant. Later requests came from other shelves, so one context cannot serve them all. Per request, each test rating is asked with its own shelf and must find its own movie."
                    ),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
