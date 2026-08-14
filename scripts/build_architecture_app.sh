#!/bin/bash
# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

# Builds the interactive LikeC4 app -- the same browsable diagrams the serve
# script gives you locally -- as a static site inside the docs tree.
#
# The output lands in <source_dir>/app, which mkdocs copies verbatim into the
# built site, so it is published alongside the API reference on GitHub Pages.
# The docs workflow calls this before deploying the newest release.
#
# Deliberately not run on pull requests: it would mean pulling a 538 MiB image
# on every PR, and anonymous Docker Hub pulls from shared GitHub runner IPs are
# rate limited often enough to make the check flaky. A LikeC4 upgrade that
# breaks this therefore surfaces at release time, not in review.
#
# Usage: scripts/build_architecture_app.sh [source_dir]

set -euo pipefail

SOURCE_DIR="${1:-docs/architecture}"
OUTPUT_DIR="$SOURCE_DIR/app"

# Pinned by multi-arch index digest. Keep in sync with
# scripts/serve_architecture_diagrams.sh.
LIKEC4_IMAGE="${LIKEC4_IMAGE:-likec4/likec4:1.59.2@sha256:d7ae4a95a488a7727af22f181db26c7804922f3154c317ac81170edd9fc73b12}"

# Start clean so a deleted view does not linger in the published app.
rm -rf "$OUTPUT_DIR"

# The docs workflow rebuilds released tags, and tags that predate the
# architecture model have no sources. Skip rather than fail.
if ! compgen -G "$SOURCE_DIR/**/*.c4" > /dev/null; then
    echo "No LikeC4 sources in '$SOURCE_DIR', nothing to build."
    exit 0
fi

echo "Building the interactive architecture app into '$OUTPUT_DIR'"

# --base ./          relative asset paths, so the app works at whatever depth
#                    mike happens to publish this version under.
# --use-hash-history routing lives in the URL fragment, so deep links resolve
#                    without the server-side rewrites GitHub Pages cannot do.
docker run --rm --init \
    -v "$PWD:/app" \
    -w /app \
    "$LIKEC4_IMAGE" \
    build \
    --base ./ \
    --use-hash-history \
    --title 'ToOp Architecture' \
    -o "$OUTPUT_DIR" \
    "$SOURCE_DIR"

# The container writes as root; hand the output back to the calling user.
docker run --rm \
    -v "$PWD:/app" \
    -w /app \
    --entrypoint sh \
    "$LIKEC4_IMAGE" \
    -c "chown -R $(id -u):$(id -g) '$OUTPUT_DIR'"

echo "Built $(du -sh "$OUTPUT_DIR" | cut -f1) into $OUTPUT_DIR"
