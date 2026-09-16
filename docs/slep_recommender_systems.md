# SLEP XXX: A Recommender Estimator API and Classical Collaborative Filtering

| Field | Value |
| --- | --- |
| Author | Mark Andreev |
| Status | Draft |
| Type | Standards Track |
| Created | 2026-09-16 |

## Abstract

This SLEP proposes first-class support for classical recommender systems in scikit-learn: a `RecommenderMixin`, a common `fit`/`predict`/`recommend` contract, top-k ranking metrics, a warm-start cross-validation splitter, and a small set of established collaborative-filtering estimators.

The proposal targets in-memory, CPU-based recommendation from user-item interactions. It gives the Python ecosystem a stable, dependency-light API for common recommendation tasks and brings them under scikit-learn's estimator, validation, model-selection, documentation, and maintenance conventions.

## Motivation

Recommendation is a standard machine-learning task, yet scikit-learn currently directs users to separate projects such as `implicit`, LightFM, and Surprise [^1]. These projects provide useful implementations, but their training data, prediction, recommendation, and evaluation APIs differ. Users cannot exchange estimators or reuse a common model-selection workflow without adapters.

The gap is especially visible for long-lived applications. A model can depend on a recommender package, its data container, and its evaluation protocol at once. Independent project lifecycles then become application migration work. Scikit-learn is a suitable home for a durable baseline because it already owns the common estimator protocol, validation utilities, cross-validation, metrics, compatibility practices, and estimator checks[^2].

This proposal also accepts scikit-learn's maintenance constraint. The project only admits established algorithms that fit its `fit` and `predict`/`transform` model and NumPy/SciPy data model[^3]. The initial scope is therefore classical collaborative filtering. Deep, sequential, context-aware, knowledge-graph, distributed, and online-serving systems remain outside this SLEP.

## Detailed description

### Scope

The unit of training data is an interaction. For the estimators covered here, `X` contains user-item pairs and `y` contains the observed interaction value:

```text
X : array-like of shape (n_interactions, 2)
    X[:, 0] contains user identifiers.
    X[:, 1] contains item identifiers.

y : array-like of shape (n_interactions,), default=None
    Rating, relevance, interaction weight, or confidence. If None,
    every observed interaction has weight 1.
```

User and item identifiers may be strings or numbers. The two namespaces are encoded independently during `fit`. A user identifier equal to an item identifier has no special meaning. Dataframe column names are preserved through the existing `feature_names_in_` convention; positional semantics remain authoritative.

This representation preserves scikit-learn's invariant that `X` and `y` share their sample dimension[^2]. It covers popularity, neighborhood, matrix-factorization, pairwise-ranking, and linear item-item models. It is not declared to be a universal representation for every recommender family.

### Estimator protocol

`RecommenderMixin` defines the recommendation operation and identifies the estimator type. It does not prescribe the semantic type of a query, allowing a future SLEP to define session or context queries without changing the method:

```python
class RecommenderMixin:
    def recommend(
        self,
        X,
        *,
        n_recommendations=10,
        candidates=None,
        exclude_seen=True,
    ):
        """Return recommended item identifiers and their scores."""
```

For the estimators in this SLEP:

`fit(X, y=None)`  
Fits from user-item interactions.

`predict(X)`  
Scores user-item pairs in an array of shape `(n_samples, 2)` and returns an array of shape `(n_samples,)`. Unknown identifiers raise `ValueError` unless an estimator documents a cold-start policy.

`recommend(X, *, n_recommendations=10, candidates=None, exclude_seen=True)`  
`X` is a one-dimensional array of user identifiers. `candidates` is a shared one-dimensional item set; `None` means all fitted items. The method returns `(items, scores)`, each of shape `(n_queries, n_recommendations)`. `exclude_seen=True` removes items observed for that user during `fit`.

Ranking is descending by score. Ties are resolved by fitted item order so that results are deterministic. If fewer than `n_recommendations` eligible items exist, the method raises `ValueError`; it never pads with sentinel identifiers.

Fitted estimators expose `user_ids_`, `item_ids_`, `n_users_`, and `n_items_`. Algorithm-specific factors and similarities follow existing trailing-underscore conventions. `RecommenderMixin.__sklearn_tags__` sets `estimator_type="recommender"`; `is_recommender` and recommender-specific common checks are added with it.

The mixin does not provide a default `score`. Rating error, ranking quality, and retrieval quality are different objectives. Choosing one silently would make model selection unreliable.

### Metrics

The following binary-relevance metrics are added to `sklearn.metrics`:

- `precision_at_k`
- `recall_at_k`
- `ndcg_at_k`
- `average_precision_at_k`
- `reciprocal_rank_at_k`
- `hit_rate_at_k`

Their common contract is:

```python
metric(y_true, y_pred, *, k=None, average="macro", sample_weight=None)
```

`y_true` is a sequence containing the relevant item identifiers for each query. `y_pred` is a two-dimensional array of ranked item identifiers. With `average=None` the result is per-query; `"macro"` returns the mean across queries. Existing `root_mean_squared_error` and `mean_absolute_error` cover explicit-rating prediction. Existing `ndcg_score` remains the API for dense, graded candidate relevance.

`make_recommender_scorer` adapts the top-k metrics to `GridSearchCV` and `cross_validate`. It groups held-out interactions by user, calls `recommend`, and evaluates against the held-out item sets. Full fitted-catalog ranking is the default. Sampled-negative evaluation requires an explicit candidate set because its scores depend on the sampling protocol.

### Model selection

Ordinary `KFold` can leave a test user or item absent from training. `WarmStartKFold` partitions interactions while ensuring that every user and item in a test fold also occurs in its training fold. It raises a clear error when the interaction graph cannot satisfy this constraint. Cold-user and cold-item evaluation require side information or an explicit fallback policy and are not standardized here.

Time-aware splitting needs timestamps and is deferred. Documentation must state that random interaction splits are invalid when the production decision is time ordered.

### Initial estimator set

The proposed module is `sklearn.recommendation`. The initial estimator set is:

| Estimator                   | Feedback          | Purpose                     |
|-----------------------------|-------------------|-----------------------------|
| `MostPopularRecommender`    | implicit/explicit | deterministic baseline      |
| `ItemKNNRecommender`        | implicit/explicit | neighborhood model          |
| `AlternatingLeastSquares`   | implicit/explicit | latent-factor model         |
| `BPRMatrixFactorization`    | implicit          | pairwise ranking model      |
| `BiasedMatrixFactorization` | explicit          | rating prediction           |
| `EASE`                      | implicit          | closed-form item-item model |

These are established CPU algorithms with sparse-matrix formulations[^4][^5][^6][^7][^8]. Each estimator is still subject to scikit-learn's ordinary algorithm inclusion criteria, benchmarks, numerical tests, and maintenance review[^3]. The SLEP standardizes the shared contract; it does not waive per-estimator review.

### Non-goals

This SLEP does not include:

- user or item side-feature matrices;
- context-aware, sequential, session, graph, or neural recommenders;
- distributed training, GPU-specific kernels, approximate-nearest-neighbor indexes, or a production serving layer;
- catalog, feature-store, experimentation, or online-feedback infrastructure;
- user-specific candidate arrays or built-in negative-sampling benchmarks.

## Implementation

Implementation is split into reviewable changes:

1.  Add `RecommenderMixin`, estimator tags, `is_recommender`, validation helpers, and common estimator checks.
2.  Add ranking metrics, `make_recommender_scorer`, and `WarmStartKFold` with tests against hand-computed examples.
3.  Add `MostPopularRecommender` and `ItemKNNRecommender` as reference implementations proving the API.
4.  Add the remaining estimators in separate pull requests with numerical equivalence tests, sparse-input tests, complexity documentation, and benchmarks against maintained external implementations.
5.  Add a user-guide chapter covering implicit versus explicit feedback, leakage-safe evaluation, cold start, full-catalog ranking, and memory/time complexity.

At least one champion is expected to maintain the new module through its early releases. A prototype in a scikit-learn-compatible external package should precede acceptance, following the SLEP workflow recommendation[^9].

## Backward compatibility

The proposal is additive. Existing estimators, metrics, splitters, and public APIs keep their behavior. Introducing `"recommender"` as an estimator type requires auditing generic model-selection and inspection utilities so that they either support it or fail with a specific message.

The public API follows scikit-learn's normal deprecation policy after release. Serialized estimators retain the same cross-version limitations as other scikit-learn models.

## Alternatives

Keep recommendation in third-party packages  
This preserves scikit-learn's current scope and maintenance budget. It also preserves fragmented APIs and leaves model selection, metrics, and estimator checks without a common owner. A scikit-learn-contrib prototype remains the proposed incubation path, not the final interface boundary.

Treat recommendation as regression  
`predict([user, item])` fits pair scoring and explicit ratings. It does not express top-k retrieval, seen-item exclusion, candidate filtering, or ranking evaluation.

Require an `InteractionData` container  
A domain object can carry side features and timestamps. It would weaken interoperability with scikit-learn's array-based splitters and meta-estimators. `X` and `y` are sufficient for the stated scope.

Define one universal recommender input  
Hybrid, contextual, sequential, and knowledge-graph models require different row semantics and auxiliary datasets. Standardizing them in this SLEP would create a large, weakly tested abstraction. The mixin standardizes the operation while this SLEP standardizes classical interaction data.

Expose only `recommend`  
Omitting `predict` prevents pairwise diagnostics and reuse of explicit-rating metrics. Both operations are common and have distinct output shapes.

## Discussion

Before opening a SLEP pull request, the proposal should be discussed in a scikit-learn issue to confirm project interest, scope, champions, and the prototype location, as required by the SLEP workflow[^9]. Links to that issue, the prototype, and subsequent pull requests should be recorded here.

## References and footnotes

[^1]: [scikit-learn related recommendation projects](https://scikit-learn.org/stable/related_projects.html#recommendation-engine-packages).

[^2]: [Developing scikit-learn estimators](https://scikit-learn.org/stable/developers/develop.html).

[^3]: [scikit-learn algorithm inclusion criteria](https://scikit-learn.org/stable/faq.html#what-are-the-inclusion-criteria-for-new-algorithms).

[^4]: B. Sarwar et al., [Item-based collaborative filtering recommendation algorithms](https://doi.org/10.1145/371920.372071), 2001.

[^5]: Y. Hu, Y. Koren, and C. Volinsky, [Collaborative Filtering for Implicit Feedback Datasets](https://doi.org/10.1109/ICDM.2008.22), 2008.

[^6]: S. Rendle et al., [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://doi.org/10.5555/1795114.1795167), 2009.

[^7]: H. Steck, [Embarrassingly Shallow Autoencoders for Sparse Data](https://doi.org/10.1145/3308558.3313710), 2019.

[^8]: Y. Koren, R. Bell, and C. Volinsky, [Matrix Factorization Techniques for Recommender Systems](https://doi.org/10.1109/MC.2009.263), 2009.

[^9]: [SLEP000: SLEP and its workflow](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep000/proposal.html).

## Copyright

This document has been placed in the public domain.
