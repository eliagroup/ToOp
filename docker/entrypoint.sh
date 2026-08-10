#!/usr/bin/env bash
# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

set -euo pipefail

# The repository is bind-mounted from the host, so its ownership does not match the container user.
# Without this, every git call (including the one uv-dynamic-versioning makes to resolve versions)
# fails with "detected dubious ownership".
git config --global --add safe.directory /app

# Reconcile the baked-in environment with whatever uv.lock is actually mounted. This is a ~1s no-op
# when nothing changed, and self-heals after a `git pull` that moved dependencies. Mirrors what
# .devcontainer/onCreateCommand.sh does. Set TOOP_SKIP_SYNC=1 to skip (e.g. when offline).
if [ "${TOOP_SKIP_SYNC:-0}" != "1" ]; then
    echo "entrypoint: syncing uv environment at ${UV_PROJECT_ENVIRONMENT:-/opt/venv}"
    uv sync --frozen --all-groups
fi

exec "$@"
