#!/bin/bash
# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

git config --global --add safe.directory /workspaces/ToOp

set -e

# Use custom .bashrc
cp "$PWD/.devcontainer/.bashrc" /root/.bashrc

# Install development dependencies from uv.lock
uv sync --all-groups --frozen

# GPU support, when the host has one. jax[cuda12] is hardware-specific and deliberately absent from
# uv.lock, so it is layered on top of the synced environment rather than pinned with everything else.
# nvidia-smi is present only when the container was started with the GPU attached, which
# "hostRequirements": {"gpu": "optional"} in devcontainer.json arranges when a GPU is available.
#
# This must come after the uv sync above: a later `uv sync` reconciles the environment exactly and
# removes these wheels again. See .devcontainer/README.md for how to get them back.
#
# uv pip needs an explicit --python: unlike uv sync and uv run it ignores UV_PROJECT_ENVIRONMENT and
# resolves ./.venv, which on a Windows host is the developer's own virtualenv seen through the bind
# mount, and it then fails on that Windows layout.
if command -v nvidia-smi > /dev/null 2>&1; then
    echo "GPU detected, installing jax[cuda12]"
    uv pip install --python "${UV_PROJECT_ENVIRONMENT:-/opt/venv}/bin/python" "jax[cuda12]"
fi

# Install pre-commit hooks & their virtual environments
uv run pre-commit install

# Install Azure ML extension
az extension add -n ml
