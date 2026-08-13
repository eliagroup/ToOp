#!/bin/bash
# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

# Serves the LikeC4 architecture model as a browsable app on http://localhost:5173
#
# The sources are bind-mounted, so edits to docs/architecture/*.c4 hot-reload in
# the browser. This is the way to work on the model; the PNGs under
# docs/architecture/generated are only what the docs build embeds.
#
# Usage: scripts/serve_architecture_diagrams.sh [source_dir] [port]

set -euo pipefail

SOURCE_DIR="${1:-docs/architecture}"
PORT="${2:-5173}"
HMR_PORT="${HMR_PORT:-24678}"

# Pinned by multi-arch index digest. Keep in sync with the version in
# scripts/render_architecture_diagrams.sh and docs/architecture/README.md.
LIKEC4_IMAGE="${LIKEC4_IMAGE:-likec4/likec4:1.59.2@sha256:d7ae4a95a488a7727af22f181db26c7804922f3154c317ac81170edd9fc73b12}"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "No '$SOURCE_DIR' found. Run this from the repository root." >&2
    exit 1
fi

# Only ask docker for a TTY when we actually have one, so the script still works
# from a non-interactive shell.
TTY_FLAGS=()
if [ -t 0 ] && [ -t 1 ]; then
    TTY_FLAGS=(-it)
fi

echo "Serving '$SOURCE_DIR' on http://localhost:$PORT  (Ctrl-C to stop)"

# Read-only mount: the dev server never needs to write into the repository.
# The image already listens on 0.0.0.0 when it detects a container.
docker run --rm \
    "${TTY_FLAGS[@]}" \
    -p "$PORT:$PORT" \
    -p "$HMR_PORT:$HMR_PORT" \
    -v "$PWD:/app:ro" \
    -w /app \
    "$LIKEC4_IMAGE" \
    start --port "$PORT" --hmr-port "$HMR_PORT" "$SOURCE_DIR"
