# ToOp Agent Instructions

ToOp is a GPU-accelerated topology optimization engine for electrical transmission grids.
It runs N-1 contingency analysis and searches for substation reconfigurations that reduce overloads,
using a fast DC stage (JAX on GPU) followed by AC validation.

This file holds only what you cannot cheaply read from the code. Longer material lives in `docs/`
and procedures live in skills (see the end of this file). Stay inside this repository.

## Packages

Six packages in `packages/`, each with `src/toop_engine_<name>/`, `tests/`, `pyproject.toml`, `README.md`.
They are installed as editable cross-dependencies via `uv`.

| Package | Purpose | Entry point |
| --- | --- | --- |
| `interfaces_pkg` | Shared contracts: `BackendInterface`, `Nminus1Definition`, asset topology, action set, loadflow results, Kafka messages, `folder_structure.py` (file names, `NETWORK_MASK_NAMES`) | — |
| `importer_pkg` | UCTE / CGMES / XIIDM import via pypowsybl or pandapower | `pypowsybl_import.preprocessing.convert_file()` |
| `dc_solver_pkg` | JAX DC loadflow with PTDF / LODF / BSDF | `jax.topology_looper.run_solver()` — see `packages/dc_solver_pkg/AGENTS.md` |
| `contingency_analysis_pkg` | AC N-1 analysis via pandapower or pypowsybl, returns `LoadflowResultsPolars` | `ac_loadflow_service.get_ac_loadflow_results()` |
| `topology_optimizer_pkg` | DC Map-Elites search + AC validation, both as Kafka workers | `dc.worker.optimizer.initialize_optimization()`, `ac.worker.optimization_loop()` |
| `grid_helpers_pkg` | Backend-specific grid utilities (pandapower, powsybl, network graph) | — |

## Architecture

The architecture is maintained as LikeC4 diagrams-as-code in `docs/architecture/`.
Read `docs/architecture/README.md` first: it maps each view to the question it answers.
Data flow in one line: grid file → importer (preprocessed folder) → DC optimizer (repertoire of candidates) → AC validator → results.

**If you change what crosses package boundaries (Kafka messages, stored files, parameters, worker responsibilities),
update the matching `docs/architecture/model/*.c4` file in the same change.**

## Domain language

Use the terms defined in [`docs/glossary.md`](docs/glossary.md) (pointer: `CONTEXT.md`).
Do not invent synonyms. If you introduce or rename a domain term, update the glossary in the same change.

## Working loop and definition of done

Full rules with examples: [`docs/contribution_guide.md`](docs/contribution_guide.md) → *Writing tests* and *Naming & documentation*.

1. Find the existing test module for the code you touch (tests mirror `src/`). Write or adjust a test that fails first.
2. Run the narrowest thing: one test → the test module → the package. Use the full suite only at the end, if at all.
3. Make it pass. Never weaken an assertion, skip, or delete a test to get green — report the failure instead.
4. Run ruff on the touched package.
5. Update NumPy docstrings, `docs/` pages, the glossary and the `.c4` model where behaviour or terms changed.
6. Self-review before you finish:
   - Every new name says *what* it is, in domain terms. No `_helper`, `_v2`, `step1`, `res`, `df`, `data`, single letters outside formulas.
   - Every test name states a behaviour (`test_<unit>_<does_what>_<when>`), tests one thing, and asserts concrete values.
   - Tests reuse existing fixtures from the package `conftest.py`; prefer the smallest grid (case14 before Oberrhein).
   - No dead code, no commented-out code, no leftover debug output.

## Testing facts

- Commands: `uv run pytest packages/<pkg>/tests/<path>::<test>`; full suite `uv run pytest -n auto --dist loadgroup`.
- Set `JAX_PLATFORMS=cpu` when no GPU is available. Beartype runtime checks are on in tests (`ENABLE_BEARTYPE=true` via `pytest_env`).
- Kafka, Ray and some importer tests need Docker. Kafka tests share `@pytest.mark.xdist_group("kafka")`. Slow tests carry `@pytest.mark.timeout(...)`.
- Fixture pattern: a session-scoped `_name` fixture builds expensive data once; the function-scoped `name` fixture returns a copy.
- Coverage gate: 90% (`.coveragerc`).

## Code conventions that tooling does not fully enforce

- Complete type hints on every function. NumPy-style docstrings (`ruff.toml`: `convention = "numpy"`) with
  `Parameters` / `Returns` / `Raises`; dataclass and Pydantic attributes are documented with a string directly below the field.
  Explain intent, units and array shapes — do not restate the signature.
- Comments explain *why*, not *what*.
- Logging: `structlog` (skill `using-structlog-logger`). No `print` (ruff `T20`).
- Pandera: do not rely on `coerce=True`; normalise dtypes explicitly before validation so code stays correct with Pandera disabled.
- Ruff: line length 125; `tests/` is excluded from linting, so apply the conventions there by hand.

## JAX (summary)

- Annotate every array with jaxtyping, leading space included: `Float[Array, " n_branches n_bus"]`.
- Traced structures are `eqx.Module`; non-traced fields use `eqx.field(static=True)`.
- Never put batch-varying data into `StaticInformation` (it holds `SolverConfig` as a static field) — that forces recompilation.
- Use `jax.debug.print()` inside traced code.
- Details: `packages/dc_solver_pkg/AGENTS.md`, skills `enforcing-jax-typing-shapes` and `separating-static-vs-dynamic-information`.

## Git and PRs

- Conventional Commits with DCO sign-off (`git commit -s`). The squashed PR title is validated by commitizen in CI (`validate_pr_title.yaml`).
- Trunk-based: branch from `main`, never commit to `main` directly.
- Per the contribution guide's AI policy, a human reviews and opens every PR. Do not open PRs autonomously.

## Skills

Skills live in `.agents/skills/` (separate repository, linked as `.claude/skills`). Prefer them over ad-hoc commands.

| Skill | Use for |
| --- | --- |
| `syncing-monorepo-dependencies` | `uv sync` all groups and extras |
| `activating-python-env` | Activate the virtual environment |
| `running-single-test` / `running-package-tests` / `running-full-tests-xdist` | Test runs, narrowest first |
| `linting-and-formatting-packages` | Ruff check and format |
| `running-precommit-all-files` | All pre-commit hooks |
| `enforcing-jax-typing-shapes` | jaxtyping shape conventions |
| `separating-static-vs-dynamic-information` | Static vs dynamic JAX data |
| `using-structlog-logger` | Logger setup in modules and tests |
| `create-commit` | Signed conventional commit |
| `pr-review` | Reviewing a PR |
| `grill-me` / `grill-with-docs` | Stress-testing a plan or design |

## Citation

Academic work based on [arXiv:2501.17529](https://arxiv.org/abs/2501.17529).
