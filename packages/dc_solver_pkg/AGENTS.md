# dc_solver_pkg Agent Instructions

Read the root `AGENTS.md` first; this file adds what is specific to the JAX DC solver.
Background reading: `docs/dc_solver/` (quickstart, preprocessing, busbar outage, switching distance).

## Pipeline

1. `preprocess/`: a `BackendInterface` implementation (`pandapower/pandapower_backend.py`, `powsybl/powsybl_backend.py`)
   feeds `preprocess.preprocess()`, which produces `NetworkData` (`preprocess/network_data.py`).
2. `preprocess/convert_to_jax.py::convert_to_jax()` turns `NetworkData` into `StaticInformation`.
   Always check the result with `jax/inputs.py::validate_static_information()`.
3. `jax/inputs.py` saves and loads `StaticInformation` as HDF5 (`save_static_information`, `load_static_information`).
4. `jax/topology_looper.py::run_solver()` evaluates batches of topologies. It picks the symmetric path
   (one injection combination per branch topology) or the injection-bruteforce path from the `injections` argument,
   and aggregates with the metric configured in `SolverConfig` (`AggregateMetricProtocol` / `AggregateOutputProtocol` in `jax/types.py`).

## Static vs dynamic data

- `StaticInformation` (`jax/types.py`) bundles `DynamicInformation` (traced arrays: PTDF, injections, limits, …)
  with `SolverConfig` (a plain `@dataclass` held as `eqx.field(static=True)`).
- Anything in `SolverConfig` or another static field is part of the compilation key. Changing it recompiles.
  Never put batch- or timestep-varying data there.
- Traced structures are `eqx.Module`. Static fields use `eqx.field(static=True)`.
  `jax_dataclasses` is only used for `replace`.
- Skill: `separating-static-vs-dynamic-information`.

## Shape annotations

Every array in a signature or `eqx.Module` field gets a jaxtyping annotation. The leading space is required
(`F722` is ignored in ruff for this reason):

```python
from jaxtyping import Array, Bool, Float, Int

ptdf: Float[Array, " n_branches n_bus"]
flows: Float[Array, " batch_size n_timesteps n_branches"]
relevant_mask: Bool[Array, " n_sub_relevant"]
disconnections: Int[Array, " n_topologies n_disconnections"]
scalar: Float[Array, " "]
```

Use the dimension names that already dominate the code, and reuse them exactly:

| Name | Meaning |
| --- | --- |
| `n_branches` | all branches (lines, transformers); `n_branches_monitored` for the monitored subset |
| `n_bus` | buses / nodes in the PTDF |
| `n_timesteps` | time dimension |
| `n_failures` | N-1 cases (outages) evaluated by the solver |
| `n_sub_relevant` | relevant substations (prefer this over the older `n_relevant_subs`) |
| `max_branch_per_sub`, `max_inj_per_sub` | padding sizes per substation |
| `n_topologies`, `batch_size` | batch dimensions |
| `n_splits`, `n_disconnections`, `n_busbars`, `n_couplers` | per-topology action sizes |

Skill: `enforcing-jax-typing-shapes`.

## Writing traced code

- Use `jax.debug.print()` inside traced functions; `print()` only runs while tracing.
- No Python `if`/`for` on array values inside `jit` (it raises `ConcretizationTypeError`).
  Use `jnp.where`, `jax.lax.cond`, `jax.lax.fori_loop` or `jax.vmap`.
- If you hit GPU memory limits, reduce `batch_size_bsdf` / `batch_size_injection` before changing the algorithm.

## Tests

- Fixtures live in `tests/conftest.py`. Start with `jax_inputs` (case14) and only move to `jax_inputs_oberrhein`
  or other large grids when the behaviour needs them.
- `tests/numpy_reference.py` is a NumPy reference implementation of the solver. Compare new JAX paths against it.
- Backend changes must keep `packages/interfaces_pkg/tests/test_backend.py` and
  `tests/test_example_grids.py::test_case57_backends_match` green.
