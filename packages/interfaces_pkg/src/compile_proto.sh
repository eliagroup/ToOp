#!/usr/bin/env bash
# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

# Regenerate the committed *_pb2.py / *_pb2.pyi files from the .proto schemas. Run it after editing
# a .proto, from anywhere:
#
#     bash packages/interfaces_pkg/src/compile_proto.sh
#
# protoc comes from the grpcio-tools wheel in the root project's opt-in "proto" dependency group, so
# its version is pinned by uv.lock. --mypy_out comes from mypy-protobuf, a dependency of
# toop_engine_interfaces.
#
# The gencode version written into *_pb2.py is the minimum protobuf runtime that can load it. Keep
# the "protobuf>=" floor in packages/interfaces_pkg/pyproject.toml at or below the major version the
# root uv.lock resolves, and raise it when regenerating with a newer major.
#
# The output directory is excluded from pre-commit, so this script adds the license header that CI's
# nwa check expects and formats the generated Python with ruff itself.

set -euo pipefail

# -I=. and the find below are relative to this directory, so anchor to it regardless of the caller.
cd "$(dirname "${BASH_SOURCE[0]}")"
repo_root="$(cd ../../.. && pwd)"

uv_run() {
  uv run --project "$repo_root" --group proto "$@"
}

# A read loop instead of mapfile: macOS still ships bash 3.2.
protos=()
while IFS= read -r proto; do
  protos+=("$proto")
done < <(find toop_engine_interfaces/ -name "*.proto" | sort)
if [ "${#protos[@]}" -eq 0 ]; then
  echo "No .proto files found under $(pwd)/toop_engine_interfaces" >&2
  exit 1
fi

uv_run python -m grpc_tools.protoc \
  -I=. \
  --python_out=. \
  --mypy_out=. \
  "${protos[@]}"

license_header='# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0
'

generated=()
for proto in "${protos[@]}"; do
  pb2="${proto%.proto}_pb2.py"
  { printf '%s\n' "$license_header"; cat "$pb2"; } > "$pb2.tmp"
  mv "$pb2.tmp" "$pb2"
  generated+=("$pb2")
done

uv_run ruff format --config "$repo_root/ruff.toml" "${generated[@]}"
