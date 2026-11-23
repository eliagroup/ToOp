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
# The gencode version written into *_pb2.py is the minimum protobuf runtime that can load it, down to
# the patch release: an older runtime raises VersionError on import. toop_engine_interfaces supports
# protobuf 5, so the generator has to target protobuf 5 even though the workspace lock resolves a
# newer runtime. It therefore runs in its own pinned environment (uvx), not in the project one:
#
#   * grpcio-tools 1.71.x bundles protoc 29 and writes gencode 5.29.0. Releases from 1.72 on write
#     6.x or newer gencode, which protobuf 5 cannot load.
#   * mypy-protobuf 3.7.0 is the last release whose plugin runs on a protobuf 5 runtime. 4.x fails
#     with "Runtime version cannot be older than the linked gencode version".
#
# Raise these pins only together with the "protobuf>=" floor in packages/interfaces_pkg/pyproject.toml,
# which must stay at or above the gencode version this script prints.
#
# The output directory is excluded from pre-commit, so this script adds the license header that CI's
# nwa check expects and formats the generated Python with ruff itself.

set -euo pipefail

# -I=. and the find below are relative to this directory, so anchor to it regardless of the caller.
cd "$(dirname "${BASH_SOURCE[0]}")"
repo_root="$(cd ../../.. && pwd)"

grpcio_tools_version="1.71.2"
mypy_protobuf_version="3.7.0"

# A read loop instead of mapfile: macOS still ships bash 3.2.
protos=()
while IFS= read -r proto; do
  protos+=("$proto")
done < <(find toop_engine_interfaces/ -name "*.proto" | sort)
if [ "${#protos[@]}" -eq 0 ]; then
  echo "No .proto files found under $(pwd)/toop_engine_interfaces" >&2
  exit 1
fi

uvx --from "grpcio-tools==${grpcio_tools_version}" --with "mypy-protobuf==${mypy_protobuf_version}" \
  python -m grpc_tools.protoc \
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

uv run --project "$repo_root" ruff format --config "$repo_root/ruff.toml" "${generated[@]}"

gencode="$(sed -n 's/^# Protobuf Python Version: //p' "${generated[0]}")"
echo "Generated with protobuf gencode ${gencode}." \
  "packages/interfaces_pkg/pyproject.toml must require protobuf>=${gencode}."
