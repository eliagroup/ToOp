# ToOp Project Context

ToOp is a GPU-accelerated topology optimization engine for electrical transmission grids. It runs N-1
contingency analysis and searches for substation reconfigurations that reduce overloads, using a fast DC stage
(JAX on GPU) followed by AC validation.

This file holds only what you cannot cheaply read from the code. Longer material lives in `docs/`, procedures
live in skills (`.agents/skills/`). Stay inside this repository.

## Project Overview

- **Purpose:** Find topology actions (busbar splits, switching) that relieve N-1 overloads in transmission grids,
  fast enough for operational planning.
- **Main Technologies:**
  - **Hardware:** NVIDIA GPU for the DC stage; everything also runs on CPU (`JAX_PLATFORMS=cpu`).
  - **Runtime:** Python `>=3.11,<3.15`, managed with `uv` (workspace with editable cross-dependencies).
  - **Numerics:** JAX + Equinox + jaxtyping (DC solver), Polars / pandas + Pandera (tabular results).
  - **Grid backends:** pypowsybl and pandapower (UCTE, CGMES, XIIDM import and AC loadflow).
  - **Orchestration:** Kafka workers (DC optimizer, AC validator), Ray for parallel AC analysis.
  - **Testing:** pytest with xdist, beartype runtime checks.
  - **Linting/Formatting/Typecheck:** Ruff (`ruff.toml`), ty, pre-commit.
- **Architecture:** Monorepo of six packages in `packages/`, each with `src/toop_engine_<name>/` and `tests/`.
  Data flow: grid file → importer (preprocessed folder) → DC optimizer (repertoire of candidates)
  → AC validator → results.
  - `interfaces_pkg`: Shared contracts — `BackendInterface`, `Nminus1Definition`, asset topology, action set,
    loadflow results, Kafka messages, `folder_structure.py` (file names, `NETWORK_MASK_NAMES`).
  - `importer_pkg`: Grid import via pypowsybl or pandapower. Entry: `pypowsybl_import.preprocessing.convert_file()`.
  - `dc_solver_pkg`: JAX DC loadflow with PTDF / LODF / BSDF. Entry: `jax.topology_looper.run_solver()`.
    Has its own `packages/dc_solver_pkg/AGENTS.md`.
  - `contingency_analysis_pkg`: AC N-1 analysis returning `LoadflowResultsPolars`.
    Entry: `ac_loadflow_service.get_ac_loadflow_results()`.
  - `topology_optimizer_pkg`: DC Map-Elites search and AC validation as Kafka workers.
    Entries: `dc.worker.optimizer.initialize_optimization()`, `ac.worker.optimization_loop()`.
  - `grid_helpers_pkg`: Backend-specific grid utilities (pandapower, powsybl, network graph).
- **Architecture docs:** LikeC4 diagrams-as-code in `docs/architecture/`. Read `docs/architecture/README.md` first;
  it maps each view to the question it answers. **If you change what crosses package boundaries (Kafka messages,
  stored files, parameters, worker responsibilities), update the matching `docs/architecture/model/*.c4` file in
  the same change.**

## Building and Running

- **Install Dependencies:** `uv sync --all-groups --all-extras --frozen` (skill `syncing-monorepo-dependencies`).
- **Run Python:** always through `uv run …` so the workspace environment is used.
- **Build Docs:** MkDocs Material (`mkdocs.yml`), sources in `docs/`.

## Testing and Quality

- **Test Commands** (narrowest first):
  - **Single test:** `uv run pytest packages/<pkg>/tests/<path>::<test>` (skill `running-single-test`).
  - **Package:** `uv run pytest packages/<pkg>/tests` (skill `running-package-tests`).
  - **Full suite:** `uv run pytest -n auto --dist loadgroup` (skill `running-full-tests-xdist`). Slow; run it only
    at the end of a change that spans packages.
  - > **NOTE:** Kafka, Ray and some importer tests need Docker. Without Docker, skip them locally and rely on CI.
- **Lint and Format:** `uv run ruff check --fix <path>` and `uv run ruff format <path>`
  (skill `linting-and-formatting-packages`).
- **Type Check:** `uv run ty check <path>` (skill `python-type-checking`).
- **Full Validation:** `SKIP=no-commit-to-branch uv run pre-commit run --all-files` (skill
  `running-precommit-all-files`). Run once before a PR, not while iterating.
- **Coverage Gate:** 90% (`.coveragerc`).

## Development Conventions

- **Working Loop:** Find the existing test module for the code you touch (tests mirror `src/`), make a test fail
  first, make it pass, run ruff on the touched package, then update docstrings, `docs/`, the glossary and the `.c4`
  model where behaviour or terms changed. Full rules: `docs/contribution_guide.md`.
- **Domain Language:** Use the terms in `docs/glossary.md` (pointer: `CONTEXT.md`). Do not invent synonyms; if you
  introduce or rename a domain term, update the glossary in the same change.
- **Naming:** Every new name says *what* it is, in domain terms. No `_helper`, `_v2`, `step1`, `res`, `df`, `data`,
  or single letters outside formulas.
- **Typing and Docstrings:** Complete type hints on every function. NumPy-style docstrings with `Parameters` /
  `Returns` / `Raises`; document dataclass and Pydantic attributes with a string directly below the field. Explain
  intent, units and array shapes rather than restating the signature.
- **Comments:** Explain *why*, not *what*. No dead or commented-out code.
- **Logging:** `structlog` only (skill `using-structlog-logger`); no `print` (ruff `T20`).
- **Pandera:** Do not rely on `coerce=True`; normalise dtypes explicitly before validation so code stays correct
  with Pandera disabled.
- **Ruff Scope:** Line length 125. `tests/` is excluded from linting, so apply the conventions there by hand.
- **Commits:** Conventional Commits with DCO sign-off (`git commit -s`, skill `committing-changes`). The squashed PR
  title is validated by commitizen in CI.
- **Branches:** Trunk-based; branch from `main`, never commit to `main` directly.
- **Pull Requests:** Per the contribution guide's AI policy, a human reviews and opens every PR. Do not open PRs
  autonomously; prepare the body with skill `preparing-pr`.

## Testing Conventions

- **Never Weaken Tests:** Do not weaken an assertion, skip, or delete a test to get green; report the failure instead.
- **Test Names:** `test_<unit>_<does_what>_<when>`; one behaviour per test, asserting concrete values.
- **Fixtures:** Reuse fixtures from the package `conftest.py`. A session-scoped `_name` fixture builds expensive data
  once; the function-scoped `name` fixture returns a copy. Prefer the smallest grid (case14 before Oberrhein).
- **Environment:** Set `JAX_PLATFORMS=cpu` without a GPU. Beartype runtime checks are on in tests
  (`ENABLE_BEARTYPE=true` via `pytest_env`).
- **Markers:** Kafka tests share `@pytest.mark.xdist_group("kafka")`; slow tests carry `@pytest.mark.timeout(...)`.

## JAX Conventions

- **Shapes:** Annotate every array with jaxtyping, leading space included: `Float[Array, " n_branches n_bus"]`
  (skill `enforcing-jax-typing-shapes`).
- **Static vs Dynamic:** Traced structures are `eqx.Module`; non-traced fields use `eqx.field(static=True)`. Never put
  batch-varying data into `StaticInformation` — that forces recompilation (skill
  `separating-static-vs-dynamic-information`).
- **Debugging:** Use `jax.debug.print()` inside traced code.

## Documentation

- Documentation lives in `docs/` and is published with MkDocs.
- Update the matching `docs/` page, the glossary and the `.c4` model in the same change when behaviour, terms or
  package boundaries change.
- Suggest documentation updates when a change leaves existing documentation obsolete or incomplete.

## Citation

Academic work based on [arXiv:2501.17529](https://arxiv.org/abs/2501.17529).
