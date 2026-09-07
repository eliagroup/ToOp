# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Check that every API-reference link in the architecture model still resolves.

The LikeC4 model links diagram boxes to mkdocstrings anchors on the published
docs site. Those anchors are derived from python dotted paths, so renaming or
moving a function silently breaks them and nothing in the normal build notices.

This resolves each link against a locally built site rather than the network, so
it fails fast, works offline, and cannot pass against a stale published site.

Usage: check_architecture_doc_links.py <built_site_dir> [source_dir]
"""

import re
import sys
from collections.abc import Iterator
from pathlib import Path

# A `link <url> 'title'` property in a .c4 file.
LINK = re.compile(r"link\s+(https://\S+)")

# A link into our own API reference: .../<optional version>/<page>/#<anchor>.
REFERENCE_URL = re.compile(r"https://[^/]+/ToOp/(?:[^/]+/)?(?P<page>.+?)/#(?P<anchor>[^'\"\s]+)")


def index_anchors(site_dir: Path) -> dict[str, set[str]]:
    """Collect every html anchor in the built site, keyed by page path.

    Parameters
    ----------
    site_dir : Path
        Root of a site built with `mkdocs build -d <dir>`.

    Returns
    -------
    dict[str, set[str]]
        Page path relative to the site root, mapped to the ids it defines.
    """
    anchors: dict[str, set[str]] = {}
    for html in site_dir.rglob("index.html"):
        page = html.parent.relative_to(site_dir).as_posix()
        text = html.read_text(encoding="utf8", errors="replace")
        anchors[page] = set(re.findall(r'id="([^"]+)"', text))
    return anchors


def suggest(anchor: str, known: set[str]) -> str:
    """Suggest a surviving anchor for one that no longer exists.

    A rename usually keeps the trailing symbol name, so an anchor with the same
    tail is very likely the intended target.

    Parameters
    ----------
    anchor : str
        The anchor that could not be found.
    known : set[str]
        The anchors the page actually defines.

    Returns
    -------
    str
        A hint to append to the error, or an empty string if nothing matches.
    """
    tail = anchor.rsplit(".", 1)[-1]
    near = sorted(a for a in known if a.rsplit(".", 1)[-1] == tail)
    return f"\n      did you mean: {near[0]}" if near else ""


def iter_links(source_dir: Path) -> Iterator[tuple[str, str]]:
    """Yield every documentation link declared in the LikeC4 sources.

    Parameters
    ----------
    source_dir : Path
        Folder holding the LikeC4 sources.

    Yields
    ------
    tuple[str, str]
        The `file:line` the link was found at, and the url itself.
    """
    repo_root = Path.cwd()
    for path in sorted(source_dir.rglob("*.c4")):
        try:
            location = path.relative_to(repo_root)
        except ValueError:
            location = path
        for lineno, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
            match = LINK.search(line)
            if match:
                yield f"{location}:{lineno}", match.group(1).rstrip("'\"")


def resolve(url: str, anchors: dict[str, set[str]]) -> str | None:
    """Check one link against the built site.

    Parameters
    ----------
    url : str
        The link to resolve.
    anchors : dict[str, set[str]]
        Anchors defined by the built site, keyed by page path.

    Returns
    -------
    str | None
        A description of the problem, or None if the link resolves. Links that
        do not point into our own API reference always resolve.
    """
    reference = REFERENCE_URL.match(url)
    if not reference:
        return None

    page, anchor = reference.group("page"), reference.group("anchor")
    if page not in anchors:
        return f"page not found: {page}/"
    if anchor not in anchors[page]:
        return f"anchor not found: #{anchor}{suggest(anchor, anchors[page])}"
    return None


def check(site_dir: Path, source_dir: Path) -> int:
    """Resolve every model link against the built site.

    Parameters
    ----------
    site_dir : Path
        Root of the built site.
    source_dir : Path
        Folder holding the LikeC4 sources.

    Returns
    -------
    int
        Process exit code: 0 if every link resolves, 1 otherwise.
    """
    anchors = index_anchors(site_dir)

    checked = external = 0
    failures: list[str] = []

    for where, url in iter_links(source_dir):
        checked += 1
        if not REFERENCE_URL.match(url):
            external += 1
            continue
        problem = resolve(url, anchors)
        if problem:
            failures.append(f"  {where}\n      {problem}\n      {url}")

    print(f"architecture doc links: {checked} checked, {external} external, {len(failures)} broken")

    if failures:
        print("\nBroken links:\n" + "\n".join(failures), file=sys.stderr)
        print(
            "\nThese point at the published API reference. If a module or function was "
            "renamed or moved,\nupdate the link in the .c4 file. If it is missing from the "
            "reference pages entirely,\nadd a ':::' directive for it under docs/references/.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    """Parse arguments and run the check.

    Returns
    -------
    int
        Process exit code.
    """
    if not 2 <= len(sys.argv) <= 3:
        print(__doc__, file=sys.stderr)
        return 2

    site_dir = Path(sys.argv[1])
    source_dir = Path(sys.argv[2]) if len(sys.argv) == 3 else Path("docs/architecture")

    if not site_dir.is_dir():
        print(f"Not a directory: {site_dir}", file=sys.stderr)
        print("Build the site first with: uv run mkdocs build -d <built_site_dir>", file=sys.stderr)
        return 2

    if not any(source_dir.rglob("*.c4")):
        print(f"No LikeC4 sources in '{source_dir}', nothing to check.")
        return 0

    return check(site_dir, source_dir)


if __name__ == "__main__":
    sys.exit(main())
