# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax.numpy as jnp
import numpy as np
from tests.numpy_reference import calc_lodf as calc_lodf_ref
from toop_engine_dc_solver.jax.lodf import calc_lodf, calc_lodf_matrix, zero_lodf_matrix
from toop_engine_dc_solver.jax.types import (
    InjectionComputations,
    StaticInformation,
    TopoVectBranchComputations,
)
from toop_engine_dc_solver.preprocess.network_data import NetworkData


def test_calc_lodf(
    case14_network_data: NetworkData,
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    dynamic_information = jax_inputs[2].dynamic_information
    jax_inputs[2].solver_config
    nd = case14_network_data

    branch_to_outage = 0

    lodf_ref = calc_lodf_ref(
        ptdf=nd.ptdf,
        from_node=nd.from_nodes,
        to_node=nd.to_nodes,
        branch_to_outage=branch_to_outage,
    )

    lodf, success = calc_lodf(
        branch_to_outage=jnp.array(branch_to_outage),
        ptdf=dynamic_information.ptdf,
        from_node=dynamic_information.from_node,
        to_node=dynamic_information.to_node,
        branches_monitored=None,
    )

    assert jnp.all(success)
    assert jnp.allclose(lodf, lodf_ref)
    assert lodf.shape == lodf_ref.shape

    lodf_matrix, success = calc_lodf_matrix(
        branches_to_outage=dynamic_information.branches_to_fail,
        ptdf=dynamic_information.ptdf,
        from_node=dynamic_information.from_node,
        to_node=dynamic_information.to_node,
        branches_monitored=dynamic_information.branches_monitored,
    )
    assert jnp.all(success)

    assert np.allclose(lodf_matrix[0, :], lodf_ref)


def test_zero_lodf_matrix(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    dynamic_information = jax_inputs[2].dynamic_information

    lodf_matrix, success = calc_lodf_matrix(
        branches_to_outage=dynamic_information.branches_to_fail,
        ptdf=dynamic_information.ptdf,
        from_node=dynamic_information.from_node,
        to_node=dynamic_information.to_node,
        branches_monitored=dynamic_information.branches_monitored,
    )
    assert jnp.all(success)

    zero_branches = np.concatenate([dynamic_information.branches_to_fail[0:3], np.array([900, 901])])
    zeroed_lodf_matrix, zeroed_success = zero_lodf_matrix(
        lodf_matrix=lodf_matrix,
        success=success,
        branches_to_zero=zero_branches,
        branches_to_outage=dynamic_information.branches_to_fail,
    )

    assert jnp.all(zeroed_lodf_matrix[0:3, :] == 0)
    assert jnp.array_equal(zeroed_lodf_matrix[3:, :], lodf_matrix[3:, :])
    assert jnp.all(zeroed_success)
