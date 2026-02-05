#!/bin/bash
# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

# filepath: /workspaces/ToOp/scripts/test_notebooks.sh
set -e

NOTEBOOK_DIR="${1:-notebooks}"
PATTERN="${2:-example*.ipynb}"

echo "Testing notebooks in '$NOTEBOOK_DIR' matching pattern: $PATTERN"

notebooks=$(find "$NOTEBOOK_DIR" -name "$PATTERN" -type f)

if [ -z "$notebooks" ]; then
    echo "No notebooks found matching pattern: $PATTERN in $NOTEBOOK_DIR"
    exit 0
fi

for notebook in $notebooks; do
    echo "Testing: $notebook"
    jupyter nbconvert --execute "$notebook" --to notebook --stdout > /dev/null
    echo "✓ $notebook passed"
done

echo "All notebooks passed!"
