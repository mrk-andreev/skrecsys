# skrecsys

[![PyPI](https://img.shields.io/pypi/v/skrecsys)](https://pypi.org/project/skrecsys/)
[![Python](https://img.shields.io/pypi/pyversions/skrecsys)](https://pypi.org/project/skrecsys/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Recommender systems in the scikit-learn style, built on pandas.

> **Status:** early alpha — the API is not stable yet.

## Installation

```sh
pip install skrecsys
```

or with [uv](https://docs.astral.sh/uv/):

```sh
uv add skrecsys
```

Requires Python 3.14+.

## Usage

```python
import skrecsys

print(skrecsys.__version__)
```

## Development

```sh
git clone https://github.com/mrk-andreev/skrecsys.git
cd skrecsys
uv sync
uv run pytest
uv run ruff check
uv run ty check
```

## Releasing

1. Bump the version: `uv version --bump patch` (or `minor` / `major`).
2. Commit, then tag and push: `git tag v$(uv version --short) && git push --tags`.
3. The `Release` GitHub Actions workflow builds and publishes to PyPI via Trusted Publishing.

## License

[MIT](LICENSE)
