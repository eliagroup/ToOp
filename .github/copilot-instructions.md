# ToOp Coding Agent Instructions

## Agent Persona
You are an expert Python researcher with 10+ years of software development experience. You have deep expertise in scientific computing, JAX, GPU acceleration, and electrical grid optimization.

## Available Tools
Use these tools strategically throughout your work:
- **#websearch**: Search the web for technical documentation, research papers, or domain-specific information
- **#think**: Break down complex problems and plan your approach before implementation
- **#manage_todo_list**: Track multi-step tasks and ensure systematic progress (use frequently for complex work)

## Project Overview
ToOp is a GPU-accelerated topology optimization engine for electrical transmission grids. It performs N-1 contingency analysis and finds optimal substation reconfigurations to reduce line overloads using a two-stage DC/AC optimization approach with JAX on GPU.

## Architecture

### Monorepo Structure
Six independent packages in `packages/`:
- **`interfaces_pkg`**: Shared abstractions (`BackendInterface`, `Nminus1Definition`, message protocols)
- **`importer_pkg`**: Grid import from UCTE/CGMES/XIIDM via pandapower/pypowsybl backends
- **`dc_solver_pkg`**: GPU-accelerated DC loadflow using JAX with PTDF/LODF/BSDF matrices
- **`contingency_analysis_pkg`**: N-1 AC validation via pandapower/pypowsybl
- **`topology_optimizer_pkg`**: Two-stage optimizer (DC Map-Elites + AC validation)
- **`grid_helpers_pkg`**: Utilities for grid operations

Each package has `src/`, `tests/`, `pyproject.toml`, and `README.md`. Use `uv` for dependency management with cross-package editable installs.

### Key Data Flow
1. **Import**: UCTE/CGMES → `importer_pkg.convert_file()` → preprocessed files in `data/{timestamp}/`
2. **Preprocess**: Grid file → `BackendInterface` → `NetworkData` → `StaticInformation` (HDF5)
3. **DC Solve**: `StaticInformation` + topology actions → JAX batch computation → sparsified results
4. **AC Validate**: DC candidates → full AC N-1 analysis → validated topologies

### Critical Patterns

**Backend Abstraction**: `BackendInterface` (abstract) ← `PandaPowerBackend`/`PowsyblBackend` (concrete). Always implement all interface methods; validation happens in tests like `test_backend.py`.

**JAX Pytrees**: Use `@pytree_dataclass` from `jax_dataclasses` for all JAX-traced data structures. Standard `@dataclass` for non-traced data. Example:
```python
from jax_dataclasses import pytree_dataclass
from jaxtyping import Array, Int, Float, Bool
@pytree_dataclass
class TopoVectBranchComputations:
    topology_branch_vec: Int[Array, "batch n_branch"]
```

**JAX Type Annotations (CRITICAL - Follow Religiously)**:
All JAX arrays MUST use `jaxtyping` annotations with shape specifications. The leading space in dimension strings is intentional and required:
```python
from jaxtyping import Array, Bool, Float, Int

# Correct patterns:
ptdf: Float[Array, " n_branches n_bus"]           # 2D: branches × buses
flows: Float[Array, " batch_size n_timesteps n_branch"]  # 3D batched
mask: Bool[Array, " n_sub_relevant"]              # 1D boolean
indices: Int[Array, " batch n_disconnections"]    # 2D integer
scalar: Int[Array, " "]                           # Scalar (0D array)

# Common dimension names (use consistently):
# - n_branch/n_branches: number of branches (lines/transformers)
# - n_bus: number of buses/nodes in grid
# - n_sub/n_sub_relevant: number of (relevant) substations
# - n_timesteps: temporal dimension
# - n_failures/n_contingencies: N-1 contingency cases
# - batch_size/batch_size_bsdf: batch dimension
# - max_branch_per_sub: maximum branches per substation
# - max_inj_per_sub: maximum injections per substation
# - n_injections: number of injection points

# Always annotate function signatures:
def compute_flows(
    ptdf: Float[Array, " n_branches n_bus"],
    injections: Float[Array, " n_timesteps n_bus"],
) -> Float[Array, " n_timesteps n_branches"]:
    """Compute branch flows from PTDF and injections."""
    return jnp.einsum("ij,tj->ti", ptdf, injections)
```

**JAX Debugging**: Use `jax.debug.print()` NEVER `print()` inside traced functions. Regular `print()` only executes during compilation.

**Static vs Dynamic Info**:
- `StaticInformation` (wrapped in `SolverConfig`): Grid structure that triggers JAX recompilation when changed. Contains dimensions, masks, configurations.
- `DynamicInformation` (pytree dataclass): Batch-varying data passed as function arguments. Contains PTDF matrices, injections, limits.
- **Never** put batch-varying data in `StaticInformation` - it forces expensive recompilation.
- Use `jax_dataclasses.Static` wrapper for pytree fields that shouldn't be traced:
```python
from jax_dataclasses import Static, pytree_dataclass
@pytree_dataclass
class MyData:
    traced_array: Float[Array, " n"]      # JAX will trace this
    static_config: Static[int]            # JAX won't trace this
```

**HDF5 Persistence**: Large matrices stored in `.hdf5` via `h5py`. See `dc_solver_pkg/jax/inputs.py` for save/load patterns. Paths in `interfaces_pkg/folder_structure.py`.

**Kafka Messaging**: Inter-service communication uses `confluent_kafka` with structured Pydantic message types in `interfaces_pkg/messages/`. Workers consume commands and produce heartbeats.

## Development Workflow

### Setup & Testing
```bash
# Dev container auto-installs dependencies via uv
uv run pytest packages/{package_name}_pkg/tests  # Single package
uv run pytest -n auto --dist loadgroup           # Parallel, all packages
```

**Coverage requirement**: 90%+ (aim for 100%). Tests must use fixtures from `conftest.py`. For GPU tests, use `jax.devices('cpu')` or set `JAX_PLATFORMS=cpu`.

### Code Quality
- **Ruff**: Line length 125, enforces annotations (`ANN`), docstrings (`D`), no prints (`T20`)
- **Type Hints**: ALL functions must have complete type annotations (parameters + return type)
- **Docstrings**: Required for all public functions using Google style with parameter type documentation
- **Commitizen**: Conventional commits required (`feat:`, `fix:`, `docs:`, etc.)
- **Pre-commit hooks**: Auto-run ruff, commitizen validation
- Ignore patterns in `ruff.toml`: `S101` (assert in tests), `F722` (JAX typing with spaces)

**Naming Conventions**:
- **Variables**: `snake_case` for all variables and functions
- **Classes**: `PascalCase` for classes
- **Constants**: `UPPER_SNAKE_CASE` for module-level constants
- **Grid Elements**:
  - `branch` for lines/transformers (not "line" when referring to both)
  - `node` or `bus` for electrical buses (use consistently within module)
  - `sub` or `station` for substations
  - `injection` for generators/loads
  - `from_node`, `to_node` for branch connectivity (not "source"/"target")
- **Dimensions**: Use `n_` prefix consistently (e.g., `n_branch`, not `num_branches` or `branch_count`)
- **Masks**: Boolean arrays end in `_mask` (e.g., `monitored_branch_mask`)
- **IDs**: String identifiers end in `_ids` (e.g., `branch_ids`, `node_ids`)

### Branching & Releases
- **Trunk-based development**: Feature branches → `main`
- Releases via GitHub Actions create tags only (no package publishing)
- Version from git tags using `uv-dynamic-versioning`

## Package-Specific Guidance

### `dc_solver_pkg`
Entry: `run_solver()` in `jax/topology_looper.py`. Key modes:
- **Symmetric**: 1 injection per branch topology (fast path)
- **Bruteforce**: Try all injections for each branch topology (pass `injection=None`)
- **Metric-first**: Custom aggregation via `AggregateMetricProtocol`/`AggregateOutputProtocol`

Preprocessing: `preprocess()` → `NetworkData` → `convert_to_jax()` → `StaticInformation`. Always validate with `validate_static_information()`.

### `importer_pkg`
Main entry: `pypowsybl_import.preprocessing.convert_file()`. Creates:
- `grid.xiidm` / `grid.json` (pandapower)
- `action_set.json` (switching possibilities)
- `nminus1_definition.json` (contingencies)
- `static_information.hdf5` (solver input)

Masks in `NETWORK_MASK_NAMES` define which branches/buses are relevant/monitored/disconnectable.

### `topology_optimizer_pkg`
DC stage: `dc.worker.optimizer.initialize_optimization()` uses Map-Elites repertoire for quality-diversity search.
AC stage: `ac.worker.optimization_loop()` validates with full AC. Evolution operators: pull, reconnection, coupler_closing.

### `contingency_analysis_pkg`
Entry: `ac_loadflow_service.get_ac_loadflow_results()` auto-selects pandapower/pypowsybl based on grid. Returns Polars or Pandas DataFrames.

## Common Tasks

**Add grid format support**: Implement `BackendInterface` in new file, add to `dc_solver_pkg/preprocess/`. Test against `test_backend.py` interface contract.

**Change JAX computation**: Edit `dc_solver_pkg/jax/compute_batch.py`. Ensure pytree-compatible types. Test with `test_compute_batch.py`.

**Add message type**: Create Pydantic model in `interfaces_pkg/messages/`, add serialization tests like `test_loadflow_commands.py`.

**Update docs**: MkDocs in `docs/`, use `mkdocstrings` for API refs. Deploy via `.github/workflows/mkdocs.yaml`.

## Debugging Tips
- **JAX traces**: Use `jax.debug.print()`, not `print()`. Prints only execute during trace/compile with regular `print()`
- **JAX errors**: "ConcretizationTypeError" means you're trying to use array values in Python control flow. Use `jax.lax.cond`, `jax.lax.fori_loop`, or `jax.vmap` instead
- **Shape mismatches**: Always verify array shapes match type annotations. Use `chex.assert_shape()` in tests
- **GPU memory**: Monitor with `nvidia-smi`. Reduce `batch_size_bsdf` or `batch_size_injection` if OOM
- **Ray persistence**: Set `CLEANUP_RAY_AFTER_TESTS=true` in tests to avoid resource leaks
- **Backend consistency**: Pandapower vs Pypowsybl results should match within tolerance; see `test_example_grids.py::test_case57_backends_match`
- **N-2 analysis**: Only for selected "safe" branches (not bridges); see `find_n_minus_2_safe_branches()`
- **Pytest markers**: Use `@pytest.mark.xdist_group("kafka")` for tests that can't run in parallel

## JAX Best Practices
- **Functional purity**: JAX functions must be pure (no side effects, same input → same output)
- **Batch operations**: Use `jax.vmap` for vectorization, not Python loops
- **JIT compilation**: Functions decorated with `@jax.jit` are compiled. Avoid Python control flow inside
- **Automatic differentiation**: Use `jax.grad` or `jax.value_and_grad` for gradients
- **Device placement**: Use `jax.device_put()` to explicitly move data to GPU when needed
- **Memory**: JAX keeps intermediate values for autodiff. Use `jax.checkpoint` for long computation graphs

## Citation
Academic work based on [arXiv:2501.17529](https://arxiv.org/abs/2501.17529).
