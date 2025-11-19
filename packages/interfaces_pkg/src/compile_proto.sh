uv run protoc \
  -I=. \
  --python_out=. \
  --mypy_out=. \
  $(find toop_engine_interfaces/ -name "*.proto")
