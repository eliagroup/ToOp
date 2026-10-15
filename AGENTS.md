# ToOp Agent Context

ToOp is a GPU-accelerated topology optimization engine for electrical transmission grids. It runs N-1
contingency analysis and searches for substation reconfigurations that reduce line overloads, using a
fast DC stage (JAX on GPU) followed by AC validation.

This file holds only what an agent cannot cheaply read from the code. Human contributors follow
[`docs/contribution_guide.md`](docs/contribution_guide.md); architecture lives in
[`docs/architecture/`](docs/architecture/README.md).

## Project Overview

- **Purpose:** Find remedial topology actions that clear N-1 overloads, validated with full AC analysis.
- **Main Technologies:**
  - **Language:** Python 3.11–3.14 (`requires-python = ">=3.11,<3.15"`), dependencies managed with `uv`
  - **Numerics:** JAX (CPU or GPU), Equinox, jaxtyping, jax-dataclasses
  - **Grid backends:** pandapower and pypowsybl
  - **Messaging:** Kafka (`confluent_kafka`) with Pydantic message types; Ray for some workers
  - **Logging:** structlog
  - **Testing:** pytest (with xdist), coverage, beartype runtime checks
  - **Linting/Formatting:** Ruff, pre-commit, ty
  - **Docs:** MkDocs with mkdocstrings, LikeC4 diagrams
- **Architecture:** Monorepo with six packages in `packages/`, each with `src/toop_engine_<name>/`, `tests/`,
  `pyproject.toml` and `README.md`. They are editable cross-dependencies installed by `uv`.
  - `interfaces_pkg`: Shared contracts: `BackendInterface`, `Nminus1Definition`, action set, loadflow results,
    Kafka messages, `folder_structure.py` (file names and `NETWORK_MASK_NAMES`).
  - `importer_pkg`: Imports UCTE / CGMES / XIIDM grids. Entry: `pypowsybl_import.preprocessing.convert_file()`.
  - `dc_solver_pkg`: JAX DC loadflow with PTDF / LODF / BSDF matrices. Entry: `jax/topology_looper.py::run_solver()`.
  - `contingency_analysis_pkg`: AC N-1 analysis via pandapower or pypowsybl. Entry:
    `ac_loadflow_service.get_ac_loadflow_results()`.
  - `topology_optimizer_pkg`: DC Map-Elites search and AC validation, both run as Kafka workers.
  - `grid_helpers_pkg`: Backend-specific grid utilities (pandapower, powsybl, network graph).
- **Data flow:** grid file → importer (preprocessed folder) → DC optimizer (candidate repertoire) → AC validator → results.

## Building and Running

- **Install Dependencies:** `uv sync --all-groups` (the `syncing-monorepo-dependencies` skill covers extras).
- **Dev Environment:** `.devcontainer/` installs dependencies automatically.
- **Compile Check:** `uv run python -m compileall -q packages`
- **Build Docs:** `uv run mkdocs build -d ./site`; preview with `uv run mkdocs serve`.

## Testing and Quality

- **Test Commands:**
  - **All tests:** `uv run pytest`; parallel: `uv run pytest -n auto --dist loadgroup`
  - **One package:** `uv run pytest packages/<package_name>_pkg/tests`
  - **One test:** `uv run pytest packages/<package_name>_pkg/tests/<file>.py::<test_name>` (see `running-single-test`)
  - **Notebooks:** `uv run pytest -n auto --dist loadgroup notebooks/tests`
  - Kafka, Ray and some importer tests need Docker. Set `JAX_PLATFORMS=cpu` when no GPU is available.
- **Full Validation:** `uv run pre-commit run --all-files` (CI runs it with `SKIP=no-commit-to-branch`). For docs
  changes also run `uv run mkdocs build -d ./site` and `./scripts/check_architecture_doc_links.sh ./site`.
- **Individual Checks:**
  - **Lint and format:** Ruff (`ruff.toml`, line length 125, NumPy docstrings). See `linting-and-formatting-packages`.
  - **Types:** `ty`. See `python-type-checking` before adding any suppression.
  - **Coverage:** 90% minimum (`.coveragerc`), aim for 100%.
- **Workflow:** Run the narrowest test first (one test, then the package). Run the full suite only at the end
  of a change. Do not weaken, skip or delete a test to get green; report the failure instead.

## Development Conventions

- **Contributions:** Follow [`docs/contribution_guide.md`](docs/contribution_guide.md). A human must review,
  understand and approve every change and open every PR. Do not open PRs autonomously.
- **Branches:** Trunk-based. Branch from `main` as `<type>/<topic>` (e.g. `fix/busbar-outage-limits`). Never commit to `main`.
- **Commit Messages:** Conventional Commits, with DCO sign-off (`git commit -s`). PR titles are validated by CI.
  Do not add AI attribution trailers such as `Co-Authored-By`.
- **Naming:**
  - `snake_case` for variables and functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
  - `n_` prefix for dimensions (`n_branch`, not `num_branches`); `_mask` for boolean arrays; `_ids` for string identifiers.
  - `branch` for lines and transformers; `bus` or `node` for electrical buses; `sub` or `station` for substations;
    `from_node` / `to_node` for connectivity.
  - Names say what the thing is in domain terms. No `_helper`, `_v2`, `step1`, `res`, `df`, `data`.
- **Type Hints:** Complete on every function: parameters and return type.
- **Docstrings:** NumPy style with `Parameters`, `Returns` and `Raises` where they apply. Document dataclass and
  Pydantic fields with a string directly below the field. Explain intent, units and array shapes; do not restate the signature.
- **Comments:** Explain *why*, not *what*. Remove dead and commented-out code.
- **Logging:** `structlog` (see `using-structlog-logger`). No `print()` (Ruff `T20`).
- **Pandera:** Do not rely on `coerce=True`. Normalise dtypes explicitly in pandas code before validation.
- **Backends:** A new grid format implements every `BackendInterface` method and passes the contract in `test_backend.py`.
- **Messages:** Kafka payloads are Pydantic models in `interfaces_pkg/messages/`, with serialization tests.

### JAX

- Annotate every array with jaxtyping, leading space included: `Float[Array, " n_branch n_bus"]`.
  See `enforcing-jax-typing-shapes`.
- Traced data structures are `eqx.Module`. Non-traced fields use `eqx.field(static=True)`.
- Never put batch-varying data into `StaticInformation`; it is part of the compiled configuration and changing it
  forces recompilation. See `separating-static-vs-dynamic-information`.
- Keep traced functions pure. Use `jax.debug.print()`, not `print()`, and `jax.vmap` instead of Python loops.
  Avoid Python control flow on traced values; use `jax.lax.cond` or `jax.lax.fori_loop`.
- Large matrices are persisted as HDF5 via `h5py`. Paths are defined in `interfaces_pkg/folder_structure.py`.
- Validate static inputs with `validate_static_information()` after `preprocess()` and `convert_to_jax()`.

## Testing Conventions

- Tests mirror the `src/` layout under `packages/<package_name>_pkg/tests/`.
- Use fixtures from the package `conftest.py`. Prefer the smallest grid that shows the behaviour (case14 before Oberrhein).
- Test names state a behaviour (`test_<unit>_<does_what>_<when>`), test one thing and assert concrete values.
- Session-scoped `_name` fixtures build expensive data once; the function-scoped `name` fixture returns a copy.
- Beartype runtime checks are on in tests (`ENABLE_BEARTYPE=true` via `pytest_env`).
- Kafka tests use `@pytest.mark.xdist_group("kafka")`. Slow tests carry `@pytest.mark.timeout(...)`.
- Pandapower and pypowsybl results must agree within tolerance; see `test_example_grids.py::test_case57_backends_match`.
- For GPU-related tests, use `jax.devices('cpu')` or `JAX_PLATFORMS=cpu`.

## Documentation

- Documentation is in `docs/` (MkDocs; API references via `mkdocstrings`). Deployed by `.github/workflows/docs.yaml`.
- Architecture is LikeC4 in `docs/architecture/`. Read `docs/architecture/README.md` first; it maps each view to the question it answers.
- If you change what crosses package boundaries (Kafka messages, stored files, parameters, worker responsibilities),
  update the matching `docs/architecture/model/*.c4` file in the same change.
- Update docstrings and `docs/` pages when behaviour changes. Suggest documentation updates when code changes make
  existing pages obsolete, and keep additions in the style and size of the neighbouring pages.

## Citation

Academic work based on [arXiv:2501.17529](https://arxiv.org/abs/2501.17529).
