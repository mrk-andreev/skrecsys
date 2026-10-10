# Repository Guidelines

## Layout

`src/skrecsys/`: Python; `rust/kernels/`: algorithms; `rust/src/`: PyO3 bindings; `tests/`: matching tests; `notebooks/`: examples; `benchmarks/`: benchmarks.

## Workflow

Install `uv` and Rust. `uv sync` builds the extension. `uv run pytest` runs Python tests with coverage; `cargo test` runs Rust tests. `uv run pre-commit run --all-files` checks formatting, lint, types, and tests. Run dataset benchmarks separately: `uv run pytest -m benchmark`.

## Documentation

Edit Jinja sources `README.md.j2` or `docs/*.md.j2`, never generated `README.md` directly. Run `uv run python scripts/render_readme.py` to regenerate; add `--check` to verify. The website (ProperDocs (ReadTheDocs theme), published by `.github/workflows/docs.yml`) is rendered from the same sources by `scripts/build_site.py`; preview with `uv run --group docs properdocs serve` after running it.

## llms.txt

Keep `llms.txt` in the repository root up to date: when you add, remove, or rename public modules, docs pages, or other key files, update its links and descriptions in the same change.

## Conventions

Support Python 3.11, 3.12, 3.13, and 3.14. Use four spaces, Ruff's 100-character limit, `snake_case` functions/modules, and `PascalCase` classes. Avoid `typing.Any` in package code. Name tests `test_*.py`; test changed behavior in its matching feature area.

## Commits and PRs

Never commit; leave changes for the user. Commit history uses short, imperative subjects. PRs should describe changes, link relevant issues, list checks run, and include screenshots for visual changes.
