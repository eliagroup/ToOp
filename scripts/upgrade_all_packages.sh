#!/bin/bash
# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0


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
