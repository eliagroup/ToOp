#!/bin/bash

# List of packages
ITEMS=("interfaces_pkg" "grid_helpers_pkg" "contingency_analysis_pkg" "dc_solver_pkg" "importer_pkg" "topology_optimizer_pkg")

echo "Upgrading all packages to the latest versions..."
uv lock --upgrade

cd packages
# Iterate and publish
for item in "${ITEMS[@]}"; do
    echo "Updating: $item"
    cd "$item"
    uv lock --upgrade
    cd ..
done
cd ..

echo "✅ All packages were upgraded."
