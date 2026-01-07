# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax
import jax.numpy as jnp
import pytest
from toop_engine_dc_solver.jax.compute_batch import compute_bsdf_lodf_static_flows
from toop_engine_dc_solver.jax.nodal_inj_optim import nodal_inj_optimization
from toop_engine_dc_solver.jax.types import (
    InjectionComputations,
    NodalInjOptimResults,
    NodalInjStartOptions,
    StaticInformation,
    TopoVectBranchComputations,
)


def test_compare_energy_overload_with_nodal_inj_optim(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
):
    topologies, _, static_information = jax_inputs
    solver_config = static_information.solver_config
    dynamic_information = static_information.dynamic_information

    batch_size_bsdf = static_information.solver_config.batch_size_bsdf
    batch_size_injection = static_information.solver_config.batch_size_injection
    # max_inj_per_sub = static_information.dynamic_information.max_inj_per_sub

    # this test relies on this
    assert batch_size_bsdf <= batch_size_injection

    topology_res = compute_bsdf_lodf_static_flows(topologies, None, dynamic_information, solver_config)

    pst_previous_results = NodalInjOptimResults(
        jax.random.uniform(
            jax.random.PRNGKey(32453423423),
            (
                batch_size_bsdf,
                static_information.dynamic_information.n_timesteps,
                static_information.dynamic_information.n_controllable_pst,
            ),
        )
    )
    solver_config = static_information.solver_config
    nodal_inj_start_options = NodalInjStartOptions(
        previous_results=pst_previous_results, precision_percent=jnp.array(solver_config.precision_percent)
    )

    with pytest.raises(NotImplementedError):
        n_0, n_1, nodal_inj_optim = nodal_inj_optimization(
            n_0=static_information.dynamic_information.unsplit_flow,
            nodal_injections=static_information.dynamic_information.nodal_injections,
            topo_res=topology_res,
            start_options=nodal_inj_start_options,
            dynamic_information=static_information.dynamic_information,
            solver_config=static_information.solver_config,
        )
