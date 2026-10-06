# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax.numpy as jnp
import numpy as np
from tests.numpy_reference import contingency_analysis as contingency_analysis_ref
from toop_engine_dc_solver.jax.contingency_analysis import (
    BatchedContingencyAnalysisParams,
    UnBatchedContingencyAnalysisParams,
    calc_injection_outage,
    calc_injection_outages,
    calc_n_1_matrix,
    contingency_analysis_matrix,
)
from toop_engine_dc_solver.jax.lodf import calc_lodf_matrix
from toop_engine_dc_solver.jax.types import (
    InjectionComputations,
    StaticInformation,
    TopoVectBranchComputations,
)
from toop_engine_dc_solver.preprocess.network_data import NetworkData


def test_n_1_analysis(
    case14_network_data: NetworkData,
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    dynamic_information = jax_inputs[2].dynamic_information
    solver_config = jax_inputs[2].solver_config
    nd = case14_network_data

    n_0_flow = dynamic_information.unsplit_flow
    n_0_flow_monitors = n_0_flow[:, dynamic_information.branches_monitored]

    lodf_matrix, success = calc_lodf_matrix(
        branches_to_outage=dynamic_information.branches_to_fail,
        ptdf=dynamic_information.ptdf,
        from_node=dynamic_information.from_node,
        to_node=dynamic_information.to_node,
        branches_monitored=dynamic_information.branches_monitored,
    )
    assert jnp.all(success)

    # Calculate the n-1 matrix
    n_1_flow = calc_n_1_matrix(
        lodf=lodf_matrix,
        branches_to_outage=dynamic_information.branches_to_fail,
        n_0_flow=n_0_flow,
        n_0_flow_monitors=n_0_flow_monitors,
    )
    batched_params = BatchedContingencyAnalysisParams(
        lodf=lodf_matrix,
        ptdf=dynamic_information.ptdf,
        modf=[],
        nodal_injections=dynamic_information.nodal_injections,
        n_0_flow=n_0_flow,
        injection_outage_node=jnp.array([], dtype=int),
    )
    full_n_1 = contingency_analysis_matrix(
        batched_params=batched_params,
        unbatched_params=UnBatchedContingencyAnalysisParams(
            branches_to_fail=dynamic_information.branches_to_fail,
            injection_outage_deltap=jnp.zeros((1, 0), dtype=float),
            branches_monitored=dynamic_information.branches_monitored,
            enable_bb_outages=jax_inputs[2].solver_config.enable_bb_outages,
        ),
    )

    n_1_flow_ref, success = contingency_analysis_ref(
        ptdf=nd.ptdf,
        from_node=nd.from_nodes,
        to_node=nd.to_nodes,
        branches_to_outage=np.flatnonzero(nd.outaged_branch_mask),
        nodal_injections=nd.nodal_injection[0],
    )
    assert np.all(success)

    assert jnp.allclose(n_1_flow[0], n_1_flow_ref)
    assert jnp.allclose(full_n_1[0], n_1_flow_ref)


def test_calc_injection_outage(
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    dynamic_information = jax_inputs[2].dynamic_information
    ptdf = dynamic_information.ptdf
    nodal_injections = dynamic_information.nodal_injections

    pre_flows = jnp.einsum("ij,tj->ti", ptdf, nodal_injections)

    delta_p_list = []
    post_flows_list = []
    changed_node_list = []
    for _ in range(100):
        changed_node = np.random.randint(0, nodal_injections.shape[1])
        delta_p = -nodal_injections[:, changed_node]
        post_flows = jnp.einsum("ij,tj->ti", ptdf, nodal_injections.at[:, changed_node].set(0))

        changed_node_list.append(changed_node)
        delta_p_list.append(delta_p)
        post_flows_list.append(post_flows)

        post_flow_computed = calc_injection_outage(
            ptdf=ptdf,
            n_0_flow=pre_flows,
            delta_p=delta_p,
            outage_node=changed_node,
            branches_monitored=dynamic_information.branches_monitored,
        )

        assert post_flow_computed.shape == post_flows.shape
        assert jnp.allclose(post_flows, post_flow_computed)

    delta_p = jnp.stack(delta_p_list, axis=1)
    changed_node = jnp.array(changed_node_list)
    post_flows = jnp.stack(post_flows_list, axis=1)

    assert post_flows.shape == (1, 100, ptdf.shape[0])

    # Also test the vmapped version
    post_flows_computed = calc_injection_outages(
        ptdf=ptdf,
        n_0_flow=pre_flows,
        injection_outage_deltap=delta_p,
        injection_outage_node=changed_node,
        branches_monitored=dynamic_information.branches_monitored,
    )
    assert post_flows_computed.shape == post_flows.shape
    assert jnp.allclose(post_flows, post_flows_computed)
