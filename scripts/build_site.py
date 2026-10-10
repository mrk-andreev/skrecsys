#!/usr/bin/env python
"""Render the MkDocs source tree in ``build/docs`` from the same Jinja sources as README.md.

``README.md.j2`` lists the ``docs/*.md.j2`` fragments in reading order; each one becomes
a page here, so the README and the website cannot disagree about content or order. The
fragments are written for one long page, so two things change on the way: headings move up
a level (the fragment's ``##`` becomes the page title), and ``](#anchor)`` links to a
heading on another page are pointed at that page. Run ``properdocs build`` afterwards.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import readme  # pylint: disable=wrong-import-position
import suite  # pylint: disable=wrong-import-position

REPO = readme.REPO
TITLE_LEVEL = 2
OUT = REPO / "build" / "docs"
GITHUB = "https://github.com/mrk-andreev/skrecsys/blob/main/"
INCLUDE = re.compile(r'^\{%-?\s*include\s+"docs/(\w+)\.md\.j2"\s*-?%\}\s*$', re.MULTILINE)
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
ANCHOR_LINK = re.compile(r"\]\(#([\w-]+)\)")
REPO_LINK = re.compile(r"\]\((notebooks/[^)#]+|LICENSE)\)")


def slugify(title: str) -> str:
    """The GitHub-style anchor of a heading, which the fragments' links already use."""
    title = re.sub(r"[`*_]", "", title).lower()
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", title))


def split_fences(text: str) -> list[tuple[bool, str]]:
    """Lines tagged with whether they are prose (True) or inside a code fence (False)."""
    lines, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            lines.append((False, line))
            fenced = not fenced
        else:
            lines.append((not fenced, line))
    return lines


def page_names() -> list[str]:
    """The fragments README.md.j2 includes, in order."""
    return INCLUDE.findall((REPO / readme.TEMPLATE).read_text(encoding="utf-8"))


def rewrite(
    line: str, page: str, owner: dict[str, str], titles: set[str], filenames: dict[str, str]
) -> str:
    """One prose line as the page needs it: heading demoted, links pointed at their pages."""
    match = HEADING.match(line)
    if match and len(match[1]) > 1:
        line = f"{match[1][1:]} {match[2]}"

    def anchor(link: re.Match[str]) -> str:
        slug = link[1]
        if owner.get(slug, page) == page and slug not in titles:
            return link[0]
        return f"]({filenames[owner[slug]]}#{slug})"

    line = ANCHOR_LINK.sub(anchor, line)
    return REPO_LINK.sub(lambda m: f"]({GITHUB}{m[1]})", line)


def main() -> int:
    env = readme.environment()
    blocks = suite.blocks()
    pages = {
        name: split_fences(env.get_template(f"docs/{name}.md.j2").render(report=blocks))
        for name in page_names()
    }

    # The first fragment is the landing page; the rest are named after their fragment.
    filenames = {name: f"{name}.md" for name in pages}
    filenames[next(iter(pages))] = "index.md"

    # A page's own title wins an anchor over a same-named subsection elsewhere, which is
    # what a link to "Query context" means; any other heading belongs to its first page.
    headings = {
        name: [m for prose, line in lines if prose and (m := HEADING.match(line))]
        for name, lines in pages.items()
    }
    owner: dict[str, str] = {}
    for name, found in headings.items():
        owner.update({slugify(m[2]): name for m in found[:1] if len(m[1]) == TITLE_LEVEL})
    for name, found in headings.items():
        for match in found:
            owner.setdefault(slugify(match[2]), name)
    titles = {slug for slug, name in owner.items() if slugify(headings[name][0][2]) == slug}

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    summary = []
    for name, lines in pages.items():
        out = [
            rewrite(line, name, owner, titles, filenames) if prose else line
            for prose, line in lines
        ]
        text = "\n".join(out) + "\n"
        title = next(
            (m[2] for prose, line in lines if prose and (m := HEADING.match(line))), "Overview"
        )
        filename = filenames[name]
        if filename == "index.md":
            text = "# skrecsys\n\n" + text
        (OUT / filename).write_text(text, encoding="utf-8", newline="\n")
        summary.append(f"* [{'Overview' if filename == 'index.md' else title}]({filename})")
    (OUT / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8", newline="\n")
    print(f"Rendered {len(summary)} pages to {OUT.relative_to(REPO)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
