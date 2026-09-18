# skrecsys

[![PyPI](https://img.shields.io/pypi/v/skrecsys)](https://pypi.org/project/skrecsys/)
[![Python](https://img.shields.io/pypi/pyversions/skrecsys)](https://pypi.org/project/skrecsys/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Recommender systems in the scikit-learn style, built on NumPy, SciPy and scikit-learn.

skrecsys is a prototype of the recommender estimator API proposed in
[`docs/slep_recommender_systems.md`](docs/slep_recommender_systems.md): a `RecommenderMixin`
with a common `fit` / `predict` / `recommend` contract, top-k ranking metrics, a warm-start
cross-validation splitter and classical collaborative-filtering estimators.

The estimators are Python; the inner loops are not. The similarity kernels, the
elastic-net and least-squares solvers, the BPR sampler, identifier encoding, sparse
matrix assembly and top-k selection are all written in Rust and exposed through PyO3 as
`skrecsys._core`. They parallelize with [rayon](https://github.com/rayon-rs/rayon) and
release the GIL while they run, and the neighbourhood models score and rank a batch of
queries inside one kernel, without ever materializing a dense score matrix in NumPy.
Nothing in the crate uses `unsafe`.

The installed library is small: four runtime dependencies, all of them ones a
scikit-learn user already has — NumPy, SciPy, scikit-learn and joblib, plus
`typing-extensions` on Python 3.11 only. pandas is optional, and only needed for the
dataset loaders' `as_frame=True`.

## Installation

```sh
pip install skrecsys
```

or with [uv](https://docs.astral.sh/uv/):

```sh
uv add skrecsys
```

Requires Python 3.11+, on Linux, macOS or Windows. The wheels carry the compiled
extension, so installing needs no Rust toolchain; they are built against the stable ABI,
which means one wheel per platform covers every supported interpreter. Building from a
source checkout needs a Rust toolchain and [maturin](https://www.maturin.rs/).

## Usage

Training data is an array of shape `(n_interactions, 2)` of user and item identifiers,
with an optional interaction value `y`.

```python
import numpy as np
from sklearn.model_selection import GridSearchCV

from skrecsys.metrics import make_recommender_scorer, ndcg_at_k
from skrecsys.model_selection import WarmStartKFold
from skrecsys.recommendation import ItemKNNRecommender

X = np.array([["alice", "matrix"], ["alice", "alien"], ["bob", "matrix"], ...])

rec = ItemKNNRecommender(n_neighbors=50).fit(X)
items, scores = rec.recommend(["alice", "bob"], n_recommendations=5)

search = GridSearchCV(
    ItemKNNRecommender(),
    {"n_neighbors": [10, 50, 200], "shrink": [0.0, 10.0]},
    cv=WarmStartKFold(n_splits=5, shuffle=True, random_state=0),
    scoring=make_recommender_scorer(ndcg_at_k, k=10),
).fit(X)
```

### Datasets

Loaders download public datasets once, cache them in `~/skrecsys_data` (override with
`data_home=` or `SKRECSYS_DATA`) and return them as `(user_id, item_id)` pairs plus ratings,
like `sklearn.datasets`.

```python
from sklearn.model_selection import cross_validate

from skrecsys.datasets import fetch_movielens_100k

X, y = fetch_movielens_100k(return_X_y=True)
cross_validate(
    ItemKNNRecommender(),
    X,
    y,
    cv=WarmStartKFold(n_splits=5, shuffle=True, random_state=0),
    scoring=make_recommender_scorer(ndcg_at_k, k=10),
)

# official u1-u5 / ua / ub splits, timestamps and user/movie metadata
ml = fetch_movielens_100k(subset="u1")
cross_validate(
    ItemKNNRecommender(),
    ml.data,
    ml.target,
    cv=[(ml.train_indices, ml.test_indices)],
    scoring=make_recommender_scorer(ndcg_at_k, k=10),
)
```

`as_frame=True` returns pandas objects and requires `pip install skrecsys[pandas]`.

## Leaderboard

<!-- leaderboard -->
MovieLens 100K, official `ua` split, k=10, default hyper-parameters. Regenerate with `python benchmarks/leaderboard.py --write-readme`.

| Model | NDCG@10 | P@10 | R@10 | hit rate | MAP | MRR | cat cov | user cov | mean pop | novelty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SLIM | 0.3024 | 0.2558 | 0.2558 | 0.9226 | 0.1621 | 0.6418 | 0.2226 | 1.0000 | 247.5139 | 8.65 |
| BPR | 0.2800 | 0.2467 | 0.2467 | 0.9226 | 0.1451 | 0.5782 | 0.3542 | 1.0000 | 228.5243 | 8.84 |
| EASE | 0.2767 | 0.2382 | 0.2382 | 0.9035 | 0.1444 | 0.5888 | 0.3030 | 1.0000 | 228.8724 | 8.79 |
| RP3Beta | 0.2762 | 0.2407 | 0.2407 | 0.9215 | 0.1414 | 0.5911 | 0.2417 | 1.0000 | 235.4201 | 8.78 |
| BM25 | 0.2661 | 0.2292 | 0.2292 | 0.8918 | 0.1370 | 0.5788 | 0.1024 | 1.0000 | 293.0034 | 8.36 |
| ItemKNN | 0.2550 | 0.2200 | 0.2200 | 0.9173 | 0.1268 | 0.5624 | 0.2887 | 1.0000 | 217.0530 | 8.86 |
| MostPopular | 0.1331 | 0.1215 | 0.1215 | 0.7306 | 0.0545 | 0.3218 | 0.0530 | 1.0000 | 371.7549 | 7.97 |
| ALS | 0.0422 | 0.0408 | 0.0408 | 0.3160 | 0.0144 | 0.1119 | 0.1464 | 1.0000 | 155.0729 | 10.09 |

Measured on Apple M4 Pro (12 usable cores), macOS-26.6.2-arm64-arm-64bit-Mach-O, CPython 3.14.3, numpy 2.5.3, skrecsys 0.3.0, `_core` built in release mode. The quality table above is deterministic and portable; the timings below are not comparable across machines or builds.

Wall clock per call. Each operation is sampled until it has spent 20s or reached its cap (1000 fits, 1000 `recommend` calls), after untimed warm-up calls (1 fit, 10 rank) so that no sample pays for a cold start; the `samples` columns say how many each row actually got, which is why a slow model shows fewer. The batch columns say what a single call processed: one fit covers the whole training split, and one `recommend` call ranks the entire catalog for every held-out user at once, so these are throughput numbers rather than single-request latency. Fit reports the spread a handful of samples can resolve; ranking is sampled often enough for nearest-rank quantiles, each of which is a call that really happened. Compare `min` across machines and watch `max` for the variance a run saw.

| Model | fit batch (interactions) | fit samples | fit min | fit median | fit max | rank batch (users x items) | rank samples | rank mean | rank median | rank q95 | rank q99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SLIM | 90570 | 111 | 165 ms | 181 ms | 201 ms | 943 x 1680 | 1000 | 2 ms | 2 ms | 2 ms | 2 ms |
| BPR | 90570 | 16 | 1.29 s | 1.29 s | 1.42 s | 943 x 1680 | 1000 | 1 ms | 1 ms | 2 ms | 2 ms |
| EASE | 90570 | 691 | 27 ms | 29 ms | 49 ms | 943 x 1680 | 788 | 25 ms | 25 ms | 28 ms | 29 ms |
| RP3Beta | 90570 | 1000 | 8 ms | 8 ms | 13 ms | 943 x 1680 | 1000 | 2 ms | 2 ms | 2 ms | 2 ms |
| BM25 | 90570 | 1000 | 7 ms | 7 ms | 26 ms | 943 x 1680 | 1000 | 1 ms | 1 ms | 1 ms | 1 ms |
| ItemKNN | 90570 | 1000 | 6 ms | 6 ms | 27 ms | 943 x 1680 | 1000 | 2 ms | 2 ms | 3 ms | 3 ms |
| MostPopular | 90570 | 1000 | 1 ms | 1 ms | 1 ms | 943 x 1680 | 1000 | 1 ms | 1 ms | 1 ms | 1 ms |
| ALS | 90570 | 60 | 328 ms | 332 ms | 366 ms | 943 x 1680 | 1000 | 2 ms | 2 ms | 2 ms | 2 ms |
<!-- /leaderboard -->

Every model uses its default hyper-parameters, so this ranks the library's baselines,
not the best each method can do. `R@10` equals `P@10` because the `ua` split holds out
exactly 10 items per user, and `user cov` is 1 by construction: `recommend` raises
rather than return a short list. The beyond-accuracy columns are the interesting ones —
`MostPopular` has the highest `mean pop` and the lowest `novelty`, and `BM25` buys its
ranking score by covering a tenth of the catalog where `EASE` covers a third. `SLIM`
leads on every ranking metric while keeping a fifth of the catalog reachable, at a
similarity matrix a quarter as dense as `RP3Beta`'s. `BPR` is the only model here
trained on pairwise preferences rather than fitted in closed form, and it buys the
second-best ranking with the widest catalog coverage of all; it is also the slowest to
fit, and the entry is seeded, which pins it to a single thread.

The second table times the same runs. Both operations are batched — one fit over the
whole training split, one `recommend` call scoring all 943 held-out users against all
1680 items the model saw — so these are throughput numbers, and the per-request latency
of a single user is a different measurement. Ranking separates the models by what they
have to touch rather than by how good they are: the neighbourhood models go through a
kernel that accumulates a query, drops what the user has seen and keeps the best ten
before moving on, so no score matrix is ever built and they finish in 1–2 ms; the factor
models multiply a handful of factors per item and land in the same place; `EASE` is the
outlier at 25 ms because its similarity matrix is dense and every query is scored
against the whole of it. The `max` column is where a run's variance shows — `ItemKNN`
and `BM25` both fit in 6–7 ms but have a slowest sample four times that. How many
samples a row gets is the budget's doing, which is why `BPR` at 1.3 s a fit gets
sixteen and `MostPopular` gets its full thousand; `--budget` and `--repeat` buy more.

## Scope

| Module | Contents | Status |
| --- | --- | --- |
| `skrecsys` | `RecommenderMixin`, `is_recommender` | implemented |
| `skrecsys.metrics` | `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `average_precision_at_k`, `reciprocal_rank_at_k`, `hit_rate_at_k`, `make_recommender_scorer` | implemented |
| `skrecsys.metrics` | `catalog_coverage_at_k`, `user_coverage_at_k`, `mean_popularity_at_k`, `novelty_at_k`, `item_popularity` | implemented |
| `skrecsys.datasets` | `fetch_movielens_100k`, `get_data_home`, `clear_data_home` | implemented |
| `skrecsys.model_selection` | `WarmStartKFold` | implemented |
| `skrecsys.utils.estimator_checks` | `check_recommender`, `yield_recommender_checks` | implemented |
| `skrecsys.recommendation` | `MostPopularRecommender`, `ItemKNNRecommender`, `AlternatingLeastSquares`, `BM25Recommender`, `EASE`, `RP3Beta`, `SLIMElasticNet`, `BayesianPersonalizedRanking` | implemented |

```sh
uv run python benchmarks/leaderboard.py                  # print the leaderboard
uv run python benchmarks/leaderboard.py --write-readme   # regenerate the tables above
uv run python benchmarks/leaderboard.py --repeat 20 --rank-repeat 5000   # tighter timings
```

## Development

Native kernels (for example the libFM-style ALS solver) are written in Rust under `rust/` and built
by [maturin](https://www.maturin.rs) as `skrecsys._core`, so development needs a Rust toolchain
(`rustup` or your package manager). `uv sync` rebuilds the extension when Rust sources change.

```sh
git clone https://github.com/mrk-andreev/skrecsys.git
cd skrecsys
uv sync
uv run pytest
uv run pytest -m benchmark  # quality benchmarks on real datasets (downloads data)
LIBFM_BIN=/path/to/libfm/bin/libFM uv run pytest -m benchmark -k libfm  # ALS vs libFM
uv run --group reference pytest -m benchmark -k implicit  # BM25 vs implicit
uv run pytest -m benchmark -k rectools  # EASE vs RecTools (needs network on first run)
uv run pytest -m benchmark -k dacrema  # RP3Beta and SLIMElasticNet vs their reference framework (needs network)
uv run pytest -m benchmark -k cornac   # BayesianPersonalizedRanking vs Cornac (needs network on first run)
uv run ruff check
uv run ty check
cargo test
cargo clippy --all-targets -- -D warnings
```

The release profile uses fat LTO and a single codegen unit. `[lib] crate-type` deliberately
lists only `cdylib`: adding `rlib` back would make Cargo drop LTO silently, because it cannot
link-time-optimize a unit that also emits an rlib.

`python scripts/pgo.py` rebuilds the extension with profile-guided optimization, training on a
short leaderboard run. It is opt-in and not part of `uv sync` or the release workflow: an
interleaved A/B on an Apple M4 Pro put the kernels within a few percent either way for a 3%
smaller binary and roughly triple the build time, which is not a trade worth making by default.
Tight numeric loops with predictable branches give PGO little to work with, but that result is
specific to this workload and micro-architecture, so the script is there to re-measure with.

## Releasing

1. Bump the version: `uv version --bump patch` (or `minor` / `major`).
2. Commit, then tag and push: `git tag v$(uv version --short) && git push --tags`.
3. The `Release` GitHub Actions workflow builds and publishes to PyPI via Trusted Publishing.

## License

[MIT](LICENSE)
