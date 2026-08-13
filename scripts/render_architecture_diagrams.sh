#!/bin/bash
# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

# Renders the LikeC4 architecture model to the PNGs that the docs build embeds.
#
# The renders are generated, not committed, so both the docs CI job and the
# docs publishing workflow call this before running mkdocs.
#
# Usage: scripts/render_architecture_diagrams.sh [source_dir]

set -euo pipefail

SOURCE_DIR="${1:-docs/architecture}"
OUTPUT_DIR="$SOURCE_DIR/generated"

# Pinned by multi-arch index digest. Keep in sync with the version named in
# docs/architecture/README.md and in the likec4 VS Code extension.
LIKEC4_IMAGE="${LIKEC4_IMAGE:-likec4/likec4:1.59.2@sha256:d7ae4a95a488a7727af22f181db26c7804922f3154c317ac81170edd9fc73b12}"

# Never let renders from a previously built tag leak into this one. This runs
# before the check below because the renders are untracked, so a `git checkout`
# of another tag leaves them behind even when it removes the .c4 sources.
rm -rf "$OUTPUT_DIR"

# The docs workflow rebuilds every release tag in a loop, and tags that predate
# the architecture model have no sources. Skip rather than fail.
if ! compgen -G "$SOURCE_DIR/*.c4" > /dev/null; then
    echo "No LikeC4 sources in '$SOURCE_DIR' in this revision, skipping diagram rendering."
    exit 0
fi

echo "Rendering LikeC4 views from '$SOURCE_DIR' into '$OUTPUT_DIR'"

# The image must run as root to reach its bundled Chromium, which lives under
# /root/.cache with mode 0700 -- passing --user makes the export fail to find
# the browser. The trailing chown hands the output back to the calling user.
# LikeC4 mirrors the source folder structure in the output, so with views split
# across <source>/views/*.c4 the PNGs land in <output>/views/. The docs embed
# them by a flat path, so everything is flattened back afterwards.
# --shm-size: the headless Chromium the export drives will segfault on large
# diagrams with docker's default 64MB /dev/shm.
docker run --rm \
    --shm-size=1g \
    -v "$PWD:/app" \
    -w /app \
    --entrypoint sh \
    "$LIKEC4_IMAGE" \
    -c "
        set -eu
        likec4 validate '$SOURCE_DIR'
        likec4 export png -o '$OUTPUT_DIR' '$SOURCE_DIR'
        likec4 export png -f overview --sequence -o /tmp/sequence '$SOURCE_DIR'
        find /tmp/sequence -name overview.png -exec cp {} '$OUTPUT_DIR/overview-sequence.png' \;
        find '$OUTPUT_DIR' -mindepth 2 -name '*.png' -exec mv -f {} '$OUTPUT_DIR/' \;
        find '$OUTPUT_DIR' -mindepth 1 -type d -empty -delete
        chown -R $(id -u):$(id -g) '$OUTPUT_DIR'
    "

echo "Rendered:"
ls -1 "$OUTPUT_DIR"
