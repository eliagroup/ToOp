#!/usr/bin/env bash
# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

# Regenerate the *_pb2.py / *_pb2.pyi files from the .proto schemas.
#
# The generated files are committed, so this only needs running after editing a .proto.
#
# protoc comes from the grpcio-tools wheel rather than a system binary, so the compiler version is
# pinned by uv.lock and this runs anywhere the project environment does -- host, container or dev
# container -- with no image-specific tooling. --mypy_out is provided by mypy-protobuf, a declared
# dependency of this package.
#
# Run from the repository root:
#     uv run bash packages/interfaces_pkg/src/compile_proto.sh

set -euo pipefail

# -I=. and the find below are relative to this directory, so anchor to it regardless of the caller.
cd "$(dirname "${BASH_SOURCE[0]}")"

python -m grpc_tools.protoc \
  -I=. \
  --python_out=. \
  --mypy_out=. \
  $(find toop_engine_interfaces/ -name "*.proto")
