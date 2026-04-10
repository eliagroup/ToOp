#!/usr/bin/env python3

"""Patch uv lockfiles so FOSSA can parse editable workspace packages.

FOSSA's uv.lock parser currently expects every ``[[package]]`` entry to contain a
``version`` key. uv omits that key for editable workspace packages, which causes
FOSSA analysis to fail. This script injects a synthetic version into those
editable package entries in CI before running the FOSSA scan.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

LOCKFILE_NAME = "uv.lock"
PATCH_VERSION = "0.0.0+workspace"
PACKAGE_HEADER = "[[package]]"


def patch_package_block(block_lines: list[str], version: str) -> tuple[list[str], bool]:
    """Patch one ``[[package]]`` block if it is an editable package without version."""
    has_version = any(line.startswith("version = ") for line in block_lines)
    is_editable = any("source = { editable = " in line for line in block_lines)

    if has_version or not is_editable:
        return block_lines, False

    patched_lines: list[str] = []
    inserted_version = False
    for line in block_lines:
        patched_lines.append(line)
        if not inserted_version and line.startswith("name = "):
            patched_lines.append(f'version = "{version}"\n')
            inserted_version = True

    if not inserted_version:
        patched_lines.insert(1, f'version = "{version}"\n')

    return patched_lines, True


def patch_lockfile(lockfile_path: Path, version: str) -> bool:
    """Patch a uv.lock file in place if necessary."""
    original_lines = lockfile_path.read_text(encoding="utf-8").splitlines(keepends=True)
    patched_lines: list[str] = []
    current_block: list[str] = []
    changed = False

    def flush_block() -> None:
        nonlocal current_block, changed
        if not current_block:
            return
        if current_block[0].strip() == PACKAGE_HEADER:
            new_block, block_changed = patch_package_block(current_block, version)
            patched_lines.extend(new_block)
            changed = changed or block_changed
        else:
            patched_lines.extend(current_block)
        current_block = []

    for line in original_lines:
        if line.strip() == PACKAGE_HEADER:
            flush_block()
            current_block = [line]
            continue

        if current_block:
            current_block.append(line)
        else:
            patched_lines.append(line)

    flush_block()

    if changed:
        lockfile_path.write_text("".join(patched_lines), encoding="utf-8")

    return changed


def main() -> None:
    """Patch all uv.lock files below the repository root."""
    repo_root = Path(__file__).resolve().parent.parent
    version = PATCH_VERSION
    lockfiles = sorted(repo_root.rglob(LOCKFILE_NAME))

    changed_files = [lockfile for lockfile in lockfiles if patch_lockfile(lockfile, version)]

    if changed_files:
        logger.info("Patched uv.lock files for FOSSA:")
        for lockfile in changed_files:
            logger.info(f"- {lockfile.relative_to(repo_root)}")
    else:
        logger.info("No uv.lock patches needed for FOSSA.")


if __name__ == "__main__":
    main()
