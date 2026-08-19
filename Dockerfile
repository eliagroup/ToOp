FROM ghcr.io/astral-sh/uv:0.11.19@sha256:b46b03ddfcfbf8f547af7e9eaefdf8a39c8cebcba7c98858d3162bd28cf536f6 AS uv

# Python 3.11 because every pyproject.toml in this repo declares requires-python ">=3.11,<3.12".
# Matching the base image means uv uses this interpreter instead of downloading a managed one; see
# .devcontainer/README.md for why that matters.
FROM python:3.11-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

# Declare environment variables
ENV PATH="/opt/venv/bin:/root/.local/bin:$PATH"

# The uv project environment lives outside the workspace on purpose; see .devcontainer/README.md.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_PYTHON_PREFERENCE=system

COPY --from=uv /uv /usr/local/bin/uv

# Install tooling. git is required, not a convenience: uv-dynamic-versioning derives every package
# version from git tags, so `uv sync` fails without it. The slim base ships neither it nor
# ca-certificates, which the packages need for any HTTPS they perform.
RUN apt-get -qq update && apt-get -qq -y install git ca-certificates vim zip unzip htop \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

WORKDIR /app
