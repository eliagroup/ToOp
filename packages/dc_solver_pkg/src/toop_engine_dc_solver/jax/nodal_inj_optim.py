import jax.numpy as jnp
from toop_engine_dc_solver.jax.types import DynamicInformation, NodalInjStartOptions, SolverConfig, TopologyResults
from jaxtyping import Float, Array, Int
from jax_dataclasses import pytree_dataclass
from toop_engine_dc_solver.jax.types import NodalInjOptimResults


def make_start_options(
    old_res: NodalInjOptimResults,
) -> NodalInjStartOptions:
    """Create start options for nodal injection optimization from previous results."""
    return NodalInjStartOptions(
        previous_results=old_res,
        precision_percent=jnp.array(1.0), # TODO
    )


def nodal_inj_optimization(
    n_0: Float[Array, " batch_size n_timesteps n_branches"],
    nodal_injections: Float[Array, " batch_size n_timesteps n_buses"],
    topo_res: TopologyResults,
    start_options: NodalInjStartOptions,
    dynamic_information: DynamicInformation,
    solver_config: SolverConfig,
) -> tuple[
    Float[Array, " batch_size n_timesteps n_branches"],
    Float[Array, " batch_size n_timesteps n_outages n_branches_monitored"],
    NodalInjOptimResults
]:
    """Optimize PST settings to reduce overloads in the base case.
    """
    raise NotImplementedError("Nodal injection optimization is not yet implemented.")



