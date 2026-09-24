"""Render ``README.md`` from ``README.md.j2`` and the stored benchmark results.

The README is an output. Its prose lives in the template, its tables come from
``benchmarks/results``, and the sentences around the tables -- which dataset, which
settings, which host, what was left out -- are the macros of
``benchmarks/templates/report.md.j2``, written from the facts each result stores.
Nothing here runs a benchmark, so rendering takes a second and is the same on every
machine, which is what lets a test fail the moment ``README.md`` and its sources differ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jinja2

sys.path.insert(0, str(Path(__file__).resolve().parent))

import leaderboard
import suite

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = Path(__file__).resolve().parent / "templates"
TEMPLATE = "README.md.j2"
README = REPO / "README.md"


def environment() -> jinja2.Environment:
    """The Jinja environment the README is rendered in.

    ``StrictUndefined`` makes a misspelled name an error rather than an empty string --
    a silently blank caption is exactly the sort of drift a generated file is meant to
    rule out. Markdown is not HTML, so nothing is escaped.
    """
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader([REPO, TEMPLATES]),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,  # noqa: S701 - the output is Markdown, not HTML
    )
    env.filters["table"] = leaderboard.render_markdown
    return env


def render() -> str:
    """The README as the template and the stored results make it."""
    return environment().get_template(TEMPLATE).render(report=suite.blocks())


def write() -> bool:
    """Write the rendered README, returning whether it changed."""
    rendered = render()
    changed = not README.exists() or README.read_text(encoding="utf-8") != rendered
    if changed:
        README.write_text(rendered, encoding="utf-8", newline="\n")
    return changed


def is_current() -> bool:
    """Whether ``README.md`` is exactly what rendering would produce."""
    return README.exists() and README.read_text(encoding="utf-8") == render()
