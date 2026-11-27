# ToOp Project Context

ToOp is a GPU-accelerated topology optimization engine for electrical transmission grids. It runs N-1
contingency analysis and searches for substation reconfigurations that reduce overloads, using a fast DC stage
(JAX on GPU) followed by AC validation.

You are the owner and long-term maintainer of this codebase, and its design is up to you.

## Project Overview

- **Purpose:** Find topology actions (busbar splits, switching, disconnecting lines, phase-shifting transformer taps) that relieve N-1 overloads in transmission grids,
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
  Data flow: grid file -> importer (preprocessed folder) -> DC optimizer (repertoire of candidates)
  -> AC validator -> results.
  - `interfaces_pkg`: Shared contracts — `BackendInterface`, `Nminus1Definition`, asset topology, action set,
    loadflow results, Kafka messages, `folder_structure.py` (file names, `NETWORK_MASK_NAMES`).
  - `importer_pkg`: Grid import via pypowsybl or pandapower. Entry: `pypowsybl_import.preprocessing.import_grid_with_security_analysis()`.
  - `dc_solver_pkg`: JAX DC loadflow with PTDF / LODF / BSDF. Entry: `jax.topology_looper.run_solver()`.
    Has its own `packages/dc_solver_pkg/AGENTS.md`.
  - `contingency_analysis_pkg`: AC N-1 analysis returning `LoadflowResultsPolars`.
    Entry: `ac_loadflow_service.get_security_analysis_results()`.
  - `topology_optimizer_pkg`: DC Map-Elites search and AC validation as Kafka workers.
    Entries: `dc.worker.optimizer.initialize_optimization()`, `ac.worker.optimization_loop()`.
  - `grid_helpers_pkg`: Backend-specific grid utilities (pandapower, powsybl, network graph).
- **Architecture docs:** LikeC4 diagrams-as-code in `docs/architecture/`. Read `docs/architecture/README.md` first;
  it maps each view to the question it answers. **If you change what crosses package boundaries (Kafka messages,
  stored files, parameters, worker responsibilities), update the matching `docs/architecture/model/*.c4` file in
  the same change.**. The ADRs in `docs/architecture/decision-drafts` outline long-term strategy of the project, consider them when making changes.

## Development Conventions

- **Working Loop:** Find the existing test module for the code you touch (tests mirror `src/`), make a test fail
  first, make it pass, run ruff on the touched package, then update docstrings, `docs/`, the glossary and the `.c4`
  model where behaviour or terms changed. Full rules: `docs/contribution_guide.md`.
- **Domain Language:** Use the terms in `docs/glossary.md` (pointer: `CONTEXT.md`). Do not invent synonyms; if you
  introduce or rename a domain term, update the glossary in the same change.
- **Pull Requests:** Per the contribution guide's AI policy, a human reviews and opens every PR. Do not open PRs
  autonomously; prepare the body with skill `preparing-pr`.

## Testing Conventions

- **Never Weaken Tests:** Do not weaken an assertion, skip, or delete a test to get green; report the failure instead.
- **Test Names:** `test_<unit>_<does_what>_<when>`; one behaviour per test, asserting concrete values.
- **Fixtures:** Reuse fixtures from the package `conftest.py`. A session-scoped `_name` fixture builds expensive data
  once; the function-scoped `name` fixture returns a copy. Prefer the smallest grid (case14 before Oberrhein).
- **Environment:** Set `JAX_PLATFORMS=cpu` without a GPU. Beartype runtime checks are on in tests
  (`ENABLE_BEARTYPE=true` via `pytest_env`).

## Documentation

- Documentation lives in `docs/` and is published with MkDocs.
- Update the matching `docs/` page, the glossary and the `.c4` model in the same change when behaviour, terms or
  package boundaries change.
- Suggest documentation updates when a change leaves existing documentation obsolete or incomplete.

## Citation

Academic work based on [arXiv:2501.17529](https://arxiv.org/abs/2501.17529).
