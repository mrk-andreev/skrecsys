#!/usr/bin/env python
"""Render README.md from Jinja templates and stored benchmark results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import readme


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if README.md is stale")
    if parser.parse_args().check:
        if readme.is_current():
            print("README.md is current.")
            return 0
        print(
            "README.md differs from what README.md.j2 and benchmarks/results render to; "
            "run `uv run python scripts/render_readme.py`.",
            file=sys.stderr,
        )
        return 1
    changed = readme.write()
    print("README.md re-rendered." if changed else "README.md already current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
