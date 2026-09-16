# skrecsys

[![PyPI](https://img.shields.io/pypi/v/skrecsys)](https://pypi.org/project/skrecsys/)
[![Python](https://img.shields.io/pypi/pyversions/skrecsys)](https://pypi.org/project/skrecsys/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Recommender systems in the scikit-learn style, built on NumPy, SciPy and scikit-learn.

skrecsys is a prototype of the recommender estimator API proposed in
[`docs/slep_recommender_systems.md`](docs/slep_recommender_systems.md): a `RecommenderMixin`
with a common `fit` / `predict` / `recommend` contract, top-k ranking metrics, a warm-start
cross-validation splitter and classical collaborative-filtering estimators.

> **Status:** early alpha — the API is not stable yet.

## Installation

```sh
pip install skrecsys
```

or with [uv](https://docs.astral.sh/uv/):

```sh
uv add skrecsys
```

Requires Python 3.11+, on Linux, macOS or Windows.

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

| Model | NDCG@10 | P@10 | R@10 | hit rate | MAP | MRR | cat cov | user cov | mean pop | novelty | fit | rec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EASE | 0.2767 | 0.2382 | 0.2382 | 0.9035 | 0.1444 | 0.5888 | 0.3030 | 1.0000 | 228.8724 | 8.79 | 36 ms | 27 ms |
| RP3Beta | 0.2762 | 0.2407 | 0.2407 | 0.9215 | 0.1414 | 0.5911 | 0.2417 | 1.0000 | 235.4201 | 8.78 | 18 ms | 16 ms |
| BM25 | 0.2661 | 0.2292 | 0.2292 | 0.8918 | 0.1370 | 0.5788 | 0.1024 | 1.0000 | 293.0034 | 8.36 | 12 ms | 7 ms |
| ItemKNN | 0.2550 | 0.2200 | 0.2200 | 0.9173 | 0.1268 | 0.5624 | 0.2887 | 1.0000 | 217.0530 | 8.86 | 155 ms | 18 ms |
| MostPopular | 0.1331 | 0.1215 | 0.1215 | 0.7306 | 0.0545 | 0.3218 | 0.0530 | 1.0000 | 371.7549 | 7.97 | 6 ms | 1 ms |
| ALS | 0.0422 | 0.0408 | 0.0408 | 0.3160 | 0.0144 | 0.1119 | 0.1464 | 1.0000 | 155.0729 | 10.09 | 348 ms | 2 ms |
<!-- /leaderboard -->

Every model uses its default hyper-parameters, so this ranks the library's baselines,
not the best each method can do. `R@10` equals `P@10` because the `ua` split holds out
exactly 10 items per user, and `user cov` is 1 by construction: `recommend` raises
rather than return a short list. The beyond-accuracy columns are the interesting ones —
`MostPopular` has the highest `mean pop` and the lowest `novelty`, and `BM25` buys its
ranking score by covering a tenth of the catalog where `EASE` covers a third. `fit` and
`rec` are median wall-clock times over `--repeat` runs; `rec` ranks the whole catalog
for every held-out user in one batched call, which is not the same thing as the latency
of a single user's request.

## Scope

| Module | Contents | Status |
| --- | --- | --- |
| `skrecsys` | `RecommenderMixin`, `is_recommender` | implemented |
| `skrecsys.metrics` | `precision_at_k`, `recall_at_k`, `ndcg_at_k`, `average_precision_at_k`, `reciprocal_rank_at_k`, `hit_rate_at_k`, `make_recommender_scorer` | implemented |
| `skrecsys.metrics` | `catalog_coverage_at_k`, `user_coverage_at_k`, `mean_popularity_at_k`, `novelty_at_k`, `item_popularity` | implemented |
| `skrecsys.datasets` | `fetch_movielens_100k`, `get_data_home`, `clear_data_home` | implemented |
| `skrecsys.model_selection` | `WarmStartKFold` | implemented |
| `skrecsys.utils.estimator_checks` | `check_recommender`, `yield_recommender_checks` | implemented |
| `skrecsys.recommendation` | `MostPopularRecommender`, `ItemKNNRecommender`, `AlternatingLeastSquares`, `BM25Recommender`, `EASE`, `RP3Beta` | implemented |

```sh
uv run python benchmarks/leaderboard.py                  # print the leaderboard
uv run python benchmarks/leaderboard.py --write-readme   # regenerate the table above
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
uv run pytest -m benchmark -k dacrema  # RP3Beta vs its reference framework (needs network)
uv run ruff check
uv run ty check
cargo test
cargo clippy --all-targets -- -D warnings
```

## Releasing

1. Bump the version: `uv version --bump patch` (or `minor` / `major`).
2. Commit, then tag and push: `git tag v$(uv version --short) && git push --tags`.
3. The `Release` GitHub Actions workflow builds and publishes to PyPI via Trusted Publishing.

## License

[MIT](LICENSE)
