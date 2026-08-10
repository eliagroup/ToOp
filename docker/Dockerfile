# syntax=docker/dockerfile:1
# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

# Runnable ToOp image, intended for a laptop (Jupyter / interactive shell / VS Code dev container).
#
# This is NOT the image used by the root ./Dockerfile. That one is a bare dev container base: it
# layers uv + protoc onto python:3.13 and copies no project files, leaving dependency installation
# to .devcontainer/onCreateCommand.sh. This image installs the project instead.
#
# The `uv` release is pinned by digest, matching the style of the root Dockerfile.
FROM ghcr.io/astral-sh/uv:0.11.19@sha256:b46b03ddfcfbf8f547af7e9eaefdf8a39c8cebcba7c98858d3162bd28cf536f6 AS uv

# Python 3.11 because every pyproject.toml in this repo declares requires-python ">=3.11,<3.12".
# Deliberately tag-pinned rather than digest-pinned: a digest must be verified against the registry,
# and inventing one would be worse than none. To harden, run
#   docker buildx imagetools inspect python:3.11-slim-bookworm
# and append the reported @sha256:... here.
FROM python:3.11-slim-bookworm

COPY --from=uv /uv /usr/local/bin/uv

# git is required at build time: uv-dynamic-versioning derives each package version from git tags.
# ca-certificates is required for any HTTPS the packages perform (e.g. CGMES boundary downloads).
RUN apt-get -qq update \
 && apt-get -qq -y install --no-install-recommends git ca-certificates \
 && apt-get -qq -y autoremove \
 && apt-get autoclean \
 && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

# Extra conveniences for interactive/dev-container use. Off by default to keep the runtime slim.
# Note: protoc is intentionally NOT installed here. packages/interfaces_pkg/src/compile_proto.sh
# needs the exact protoc 33.1 pinned by the root ./Dockerfile, and Debian ships a different version.
# The generated *_pb2.py files are committed, so regeneration is a rare, deliberate task -- use the
# root Dockerfile for it.
ARG DEV_TOOLS=false
RUN if [ "$DEV_TOOLS" = "true" ]; then \
      apt-get -qq update \
   && apt-get -qq -y install --no-install-recommends vim htop zip unzip curl less procps \
   && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log; \
    fi

# The project environment lives OUTSIDE /app on purpose.
#
# /app is bind-mounted from the Windows host at runtime. Creating the venv at /app/.venv would put a
# Linux virtualenv on the Windows filesystem: it would collide with any natively-created .venv, and
# every import would drag thousands of small files across the WSL2 9p/virtiofs bridge. Relocating it
# keeps `uv sync` / `uv run` as the only interface while the environment sits on Linux-native
# storage. UV_PROJECT_ENVIRONMENT is uv's supported way to do this.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_PREFERENCE=system \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# The whole context is copied (including .git, ~9MB) so that uv-dynamic-versioning can resolve
# versions and so the six editable path-dependencies under packages/ resolve at build time. At
# runtime the host repo is bind-mounted over /app; because the editable install records
# /app/packages/*/src, source edits on the host take effect with no rebuild.
COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-groups

# Bytecode is compiled once, above, and ships in the image (and therefore in the toop-venv volume,
# which Docker seeds from it). Turning the flag off from here on keeps the entrypoint's re-sync from
# recompiling all ~38k files on every container start, which costs ~6s each time. Placed after the
# sync so it does not invalidate that layer.
ENV UV_COMPILE_BYTECODE=0

# Optional NVIDIA GPU support. Requires an NVIDIA GPU, a Windows driver with WSL2 CUDA support, and
# `docker compose --profile gpu`. Roughly doubles the image size, hence opt-in.
ARG INSTALL_CUDA=false
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_CUDA" = "true" ]; then uv pip install "jax[cuda12]"; fi

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8888 6006

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uv", "run", "jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.password=", \
     "--ServerApp.root_dir=/app"]
