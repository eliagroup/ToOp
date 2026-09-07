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
# the browser. This is the way to work on the model. For a static build of the
# same app -- what gets published with the docs -- see
# scripts/build_architecture_app.sh.
#
# Usage: scripts/serve_architecture_diagrams.sh [source_dir] [port]

set -euo pipefail

SOURCE_DIR="${1:-docs/architecture}"
PORT="${2:-5173}"
HMR_PORT="${HMR_PORT:-24678}"

# Pinned by multi-arch index digest. This is now the only place the LikeC4
# version is pinned, so bumping it here is the whole upgrade.
LIKEC4_IMAGE="${LIKEC4_IMAGE:-likec4/likec4:1.59.2@sha256:d7ae4a95a488a7727af22f181db26c7804922f3154c317ac81170edd9fc73b12}"

if [ ! -d "$SOURCE_DIR" ]; then
    echo "No '$SOURCE_DIR' found. Run this from the repository root." >&2
    exit 1
fi

CONTAINER_NAME="likec4-diagrams-$PORT"

# Ctrl-C has to survive two hazards, hence the shape of the rest of this script.
#
# First, the server is PID 1 inside the container, and linux ignores signals
# with no explicit handler for PID 1 -- so a plain SIGINT is silently dropped
# and the container keeps serving. `--init` puts a real init process at PID 1
# to forward signals properly.
#
# Second, bash defers a trap until the current foreground command returns, so
# trapping around a foreground `docker run` deadlocks: the trap that would stop
# the container cannot run until the container stops. Running it in the
# background and waiting lets the trap fire immediately.
# `docker rm -f` rather than `docker stop`: stop leaves the --rm cleanup to run
# asynchronously, so restarting on the same port raced against the name still
# being held. rm -f is synchronous, and the dev server has no state to flush.
remove_server() {
    docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1 || true
}
trap remove_server EXIT INT TERM

# Clear a leftover container from a previous run that was killed outright.
remove_server

echo "Serving '$SOURCE_DIR' on http://localhost:$PORT  (Ctrl-C to stop)"

# Read-only mount: the dev server never needs to write into the repository.
# The image already listens on 0.0.0.0 when it detects a container.
docker run --rm --init \
    --name "$CONTAINER_NAME" \
    -p "$PORT:$PORT" \
    -p "$HMR_PORT:$HMR_PORT" \
    -v "$PWD:/app:ro" \
    -w /app \
    "$LIKEC4_IMAGE" \
    start --port "$PORT" --hmr-port "$HMR_PORT" "$SOURCE_DIR" &

wait $!
