#!/bin/bash
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
