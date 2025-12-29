# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Contains nodal injection optimization routines.

Nodal injection optimization includes PST Optimization routines.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float
from toop_engine_dc_solver.jax.types import (
    DynamicInformation,
    NodalInjOptimResults,
    NodalInjStartOptions,
    SolverConfig,
    TopologyResults,
)


def make_start_options(
    old_res: NodalInjOptimResults,
) -> NodalInjStartOptions:
    """Create start options for nodal injection optimization from previous results."""
    return NodalInjStartOptions(
        previous_results=old_res,
        precision_percent=jnp.array(1.0),  # TODO
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
    NodalInjOptimResults,
]:
    """Optimize PST settings to reduce overloads in the base case."""
    raise NotImplementedError("Nodal injection optimization is not yet implemented.")
