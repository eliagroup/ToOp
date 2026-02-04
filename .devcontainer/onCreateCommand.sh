#!/bin/bash
# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
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
# Install pre-commit hooks & their virtual environments
uv run pre-commit install

# Install Azure ML extension
az extension add -n ml
