# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import jax.numpy as jnp
import numpy as np
from tests.numpy_reference import calc_bsdf as calc_bsdf_ref
from tests.numpy_reference import compute_bus_splits as compute_bus_splits_ref
from tests.numpy_reference import get_bsdf_branch_indices as get_bsdf_branch_indices_ref
from toop_engine_dc_solver.jax.bsdf import (
    calc_bsdf,
    compute_bus_splits,
    get_bus_data,
    get_bus_data_other,
    update_from_to_node,
)
from toop_engine_dc_solver.jax.topology_computations import convert_single_branch_topo_vect
from toop_engine_dc_solver.jax.types import (
    InjectionComputations,
    StaticInformation,
    TopoVectBranchComputations,
)
from toop_engine_dc_solver.preprocess.network_data import NetworkData


def test_get_bus_data(
    case14_network_data: NetworkData,
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    dynamic_information = jax_inputs[2].dynamic_information
    solver_config = jax_inputs[2].solver_config
    station_to_split = 2
    assignment = np.array([True, True, False, False], dtype=bool)

    (
        brh_from_a_ref,
        brh_to_a_ref,
        brh_from_b_ref,
        brh_to_b_ref,
        node_from_a_ref,
        node_to_a_ref,
    ) = get_bsdf_branch_indices_ref(
        assignment,
        case14_network_data.branches_at_nodes[station_to_split],
        case14_network_data.branch_direction[station_to_split],
        case14_network_data.from_nodes,
        case14_network_data.to_nodes,
    )

    assert len(brh_from_a_ref) + len(brh_to_a_ref) == 2
    assert len(brh_from_b_ref) + len(brh_to_b_ref) == 2

    assignment_padded = np.zeros(solver_config.max_branch_per_sub, dtype=bool)
    assignment_padded[: len(assignment)] = assignment

    assert np.array_equal(
        dynamic_information.tot_stat[station_to_split][0:4],
        case14_network_data.branches_at_nodes[station_to_split],
    )
    assert np.array_equal(
        dynamic_information.from_stat_bool[station_to_split][0:4],
        case14_network_data.branch_direction[station_to_split],
    )
    assert np.array_equal(dynamic_information.to_node, case14_network_data.to_nodes)
    assert np.array_equal(dynamic_information.from_node, case14_network_data.from_nodes)

    brh_to_bus_a, brh_from_bus_a = get_bus_data(
        jnp.array(assignment_padded),
        dynamic_information.tot_stat[station_to_split],
        dynamic_information.from_stat_bool[station_to_split],
        return_bus_b=False,
        n_branches=dynamic_information.n_branches,
    )
    brh_to_other, brh_from_other = get_bus_data_other(
        brh_to_bus_a,
        brh_from_bus_a,
        dynamic_information.to_node,
        dynamic_information.from_node,
    )

    int_max = jnp.iinfo(brh_to_bus_a.dtype).max
    assert set(brh_from_a_ref) == set(brh_from_bus_a[brh_from_bus_a != int_max].tolist())
    assert set(brh_to_a_ref) == set(brh_to_bus_a[brh_to_bus_a != int_max].tolist())
    assert set(node_from_a_ref) == set(brh_from_other[brh_from_other != int_max].tolist())
    assert set(node_to_a_ref) == set(brh_to_other[brh_to_other != int_max].tolist())


def test_calc_bsdf(
    case14_network_data: NetworkData,
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    dynamic_information = jax_inputs[2].dynamic_information
    solver_config = jax_inputs[2].solver_config
    nd = case14_network_data
    station_to_split = 2
    assignment = np.array([True, True, False, False], dtype=bool)

    bsdf_ref, ptdf_ref, from_node_ref, to_node_ref = calc_bsdf_ref(
        switched_node=station_to_split,
        assignment=assignment,
        is_slack=nd.relevant_nodes[station_to_split] == nd.slack,
        bus_a_ptdf=nd.relevant_nodes[station_to_split],
        bus_b_ptdf=nd.n_original_nodes + station_to_split,
        to_node=nd.to_nodes,
        from_node=nd.from_nodes,
        susceptance=nd.susceptances,
        ptdf=nd.ptdf,
        branches_at_nodes=nd.branches_at_nodes,
        branch_direction=nd.branch_direction,
    )

    padded_assignment = np.zeros(solver_config.max_branch_per_sub, dtype=bool)
    padded_assignment[: len(assignment)] = assignment

    bsdf, ptdf_th_sw, success = calc_bsdf(
        substation_topology=jnp.array(padded_assignment),
        ptdf=dynamic_information.ptdf,
        i_stat=solver_config.rel_stat_map.val[station_to_split],
        i_stat_rel=jnp.array(station_to_split),
        tot_stat=dynamic_information.tot_stat[station_to_split],
        from_stat_bool=dynamic_information.from_stat_bool[station_to_split],
        to_node=dynamic_information.to_node,
        from_node=dynamic_information.from_node,
        susceptance=dynamic_information.susceptance,
        slack=jnp.array(solver_config.slack),
        n_stat=jnp.array(solver_config.n_stat),
    )
    assert success

    assert np.allclose(bsdf, bsdf_ref)
    ptdf = dynamic_information.ptdf + jnp.outer(bsdf, ptdf_th_sw)
    assert np.allclose(ptdf, ptdf_ref)

    to_node, from_node = update_from_to_node(
        substation_topology=jnp.array(padded_assignment),
        tot_stat=dynamic_information.tot_stat[station_to_split],
        from_stat_bool=dynamic_information.from_stat_bool[station_to_split],
        i_stat_rel_id=jnp.array(station_to_split),
        to_node=dynamic_information.to_node,
        from_node=dynamic_information.from_node,
        n_stat=jnp.array(solver_config.n_stat),
    )

    assert np.array_equal(to_node, to_node_ref)
    assert np.array_equal(from_node, from_node_ref)


def test_compute_bus_split(
    case14_network_data: NetworkData,
    jax_inputs: tuple[TopoVectBranchComputations, InjectionComputations, StaticInformation],
) -> None:
    dynamic_information = jax_inputs[2].dynamic_information
    solver_config = jax_inputs[2].solver_config
    nd = case14_network_data

    computation = jax_inputs[0][40]

    np_topo_vect = convert_single_branch_topo_vect(
        computation.topologies, computation.sub_ids, solver_config.branches_per_sub
    )

    ptdf_ref, from_node_ref, to_node_ref = compute_bus_splits_ref(
        topo_vect=np_topo_vect,
        relevant_nodes=nd.relevant_nodes,
        slack=nd.slack,
        n_stat=nd.n_original_nodes,
        ptdf=nd.ptdf,
        susceptance=nd.susceptances,
        from_node=nd.from_nodes,
        to_node=nd.to_nodes,
        branches_at_nodes=nd.branches_at_nodes,
        branch_direction=nd.branch_direction,
    )

    bsdf = compute_bus_splits(
        topologies=computation.topologies,
        sub_ids=computation.sub_ids,
        ptdf=dynamic_information.ptdf,
        from_node=dynamic_information.from_node,
        to_node=dynamic_information.to_node,
        tot_stat=dynamic_information.tot_stat,
        from_stat_bool=dynamic_information.from_stat_bool,
        susceptance=dynamic_information.susceptance,
        rel_stat_map=solver_config.rel_stat_map,
        slack=solver_config.slack,
        n_stat=solver_config.n_stat,
    )

    assert np.allclose(bsdf.ptdf, ptdf_ref)
    assert np.array_equal(bsdf.from_node, from_node_ref)
    assert np.array_equal(bsdf.to_node, to_node_ref)
