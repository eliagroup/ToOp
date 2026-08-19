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
# Run from the repository root:
#     uv run bash packages/interfaces_pkg/src/compile_proto.sh
#
# protoc comes from the grpcio-tools wheel rather than a system binary, so the compiler version is
# pinned by uv.lock and this runs anywhere the project environment does. --mypy_out is provided by
# mypy-protobuf, a declared dependency of this package.
#
# Three things to know before regenerating:
#
#   * The committed output declares gencode 5.29.3, while the current toolchain emits 7.35.1. Older
#     gencode loads on a newer runtime, so the committed files work as they are -- but the reverse
#     does not, so regenerating raises the effective minimum protobuf runtime to 7.x while
#     interfaces_pkg still declares "protobuf>=5,<8". Tighten that lower bound in the same change if
#     you commit regenerated output. The serialized descriptor, field numbers and wire format are
#     unchanged; the .pyi additionally restyles its imports, which is mypy-protobuf's own output
#     changing rather than anything semantic.
#
#   * Do not pin grpcio-tools below 1.72 to reproduce the older gencode. Those releases cap protobuf
#     at <6, which drags the whole environment's runtime down, and mypy-protobuf ships gencode built
#     against 6.x -- so --mypy_out then fails with "Runtime version cannot be older than the linked
#     gencode version".
#
#   * Raw protoc output is neither formatted nor licensed, while the committed files are both. After
#     regenerating, re-add the MPL header (CI's nwa check enforces it) and run
#     `uv run pre-commit run --files <the regenerated files>` for ruff formatting and trailing
#     whitespace, otherwise the diff is dominated by style noise.

set -euo pipefail

# -I=. and the find below are relative to this directory, so anchor to it regardless of the caller.
cd "$(dirname "${BASH_SOURCE[0]}")"

python -m grpc_tools.protoc \
  -I=. \
  --python_out=. \
  --mypy_out=. \
  $(find toop_engine_interfaces/ -name "*.proto")
