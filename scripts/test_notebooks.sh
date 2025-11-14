#!/bin/bash
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
