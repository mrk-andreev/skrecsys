"""No Python source in the repository silences the linter with a ``noqa`` comment.

An inline suppression is invisible to review once it lands, and it outlives the reason it
was added. An exception the project does accept goes in ``[tool.ruff.lint.per-file-ignores]``
in ``pyproject.toml``, where every one sits in one place with its justification beside it.

Only comment tokens count, so a docstring or a string literal that mentions the word, like
this one, is not a suppression.
"""

import io
import re
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("src", "tests", "benchmarks", "notebooks", "scripts")
NOQA = re.compile(r"\bnoqa\b", re.IGNORECASE)


def _python_files():
    for name in SOURCE_DIRS:
        for path in sorted((ROOT / name).rglob("*.py")):
            parts = path.relative_to(ROOT).parts
            if not any(part.startswith(".") or part == "__pycache__" for part in parts):
                yield path


def _noqa_comments(path):
    source = path.read_text(encoding="utf-8")
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT and NOQA.search(token.string):
            yield token.start[0], token.string


def test_python_files_are_found():
    # Guards the scan itself: a moved directory must not turn the check below into a no-op.
    assert any(path.parent.name == "skrecsys" for path in _python_files())


def test_no_noqa_comments():
    found = [
        f"{path.relative_to(ROOT)}:{line}: {comment}"
        for path in _python_files()
        for line, comment in _noqa_comments(path)
    ]
    assert not found, (
        "Fix the code, or move the exception to [tool.ruff.lint.per-file-ignores] in "
        "pyproject.toml with a comment on why:\n" + "\n".join(found)
    )
