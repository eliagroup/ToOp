#!/bin/bash
# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

# Thin wrapper around check_architecture_doc_links.py, so CI and local use have
# the same entry point as the other architecture scripts.
#
# Usage: scripts/check_architecture_doc_links.sh <built_site_dir> [source_dir]
#
# Build the site first:
#   uv run mkdocs build -d ./site
#   scripts/check_architecture_doc_links.sh ./site

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "$SCRIPT_DIR/check_architecture_doc_links.py" "$@"
