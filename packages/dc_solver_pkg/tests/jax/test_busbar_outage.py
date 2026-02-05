# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import copy
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pypowsybl as pp
import pytest
from jaxtyping import Array, Bool, Float, Int
from tests.deprecated.assignment import realise_bus_split_single_station
from toop_engine_dc_solver.jax.bsdf import compute_bus_splits
from toop_engine_dc_solver.jax.busbar_outage import (
    filter_already_outaged_branches_single_outage,
    get_busbar_outage_penalty,
    get_busbar_outage_penalty_batched,
    perform_non_rel_bb_outages,
    perform_outage_single_busbar,
    perform_rel_bb_outage_batched,
    perform_rel_bb_outage_for_unsplit_grid,
    perform_rel_bb_outage_single_topo,
    remove_articulation_nodes_from_bb_outage,
)
from toop_engine_dc_solver.jax.compute_batch import compute_bsdf_lodf_static_flows, compute_injections
from toop_engine_dc_solver.jax.cross_coupler_flow import compute_cross_coupler_flows
from toop_engine_dc_solver.jax.disconnections import apply_disconnections
from toop_engine_dc_solver.jax.injections import get_injection_vector
from toop_engine_dc_solver.jax.topology_computations import (
    convert_action_set_index_to_topo,
    pad_action_with_unsplit_action_indices,
)
from toop_engine_dc_solver.jax.types import (
    ActionIndexComputations,
    ActionSet,
    RelBBOutageData,
    StaticInformation,
    int_max,
)
from toop_engine_dc_solver.postprocess.postprocess_powsybl import apply_topology
from toop_engine_dc_solver.preprocess.convert_to_jax import convert_to_jax, get_bb_outage_baseline_analysis
from toop_engine_dc_solver.preprocess.network_data import extract_action_set, map_branch_injection_ids
from toop_engine_dc_solver.preprocess.preprocess import NetworkData
from toop_engine_dc_solver.preprocess.preprocess_bb_outage import (
    extract_busbar_outage_data,
    get_busbar_branches_map,
    get_busbar_index,
    get_connected_assets,
    get_rel_non_rel_sub_bb_maps,
    get_relevant_stations,
    get_total_injection_along_stub_branch,
    logger,
    preprocess_bb_outages,
)
from toop_engine_interfaces.asset_topology import Station


# FIxme: Look deeply into this
def validate_power_flow_in_stub_branch(
    network_data: NetworkData,
    branch_index: int,
    lfs: Float[Array, "n_subs n_timesteps n_branches"],
    node_index: int,
    delta_p: Float[Array, " n_timesteps"],
    p_0: Float[Array, " n_timesteps"],
) -> None:
    assert network_data.bridging_branch_mask[branch_index]

    p_stub = get_total_injection_along_stub_branch(branch_index, node_index, network_data)
    # p_bb + p_obb = p_0
    # delta_p = p_stub + p_bb
    # p_obb = p_0 - p_bb = p_0 - (delta_p - p_stub)
    p_obb = p_0 - delta_p + p_stub
    jnp.isclose(jnp.abs(lfs[0][branch_index]), jnp.abs(p_stub - p_obb))
    # assert jnp.isclose(
    #     jnp.abs(lfs[0][branch_index]), jnp.abs(p_stub-p_obb)
    # ), f"Power flow in skeleton branch {branch_index} should be equal to the total injection along the stub branch."


def test_perform_outage_single_busbar(
    jax_inputs_oberrhein, network_data_preprocessed: NetworkData, oberrhein_outage_station_busbars_map
) -> None:
    _, static_information = jax_inputs_oberrhein
    dynamic_information = static_information.dynamic_information

    _, non_rel_station_busbars_map = get_rel_non_rel_sub_bb_maps(
        network_data_preprocessed, oberrhein_outage_station_busbars_map
    )

    connected_branches_to_outage = dynamic_information.non_rel_bb_outage_data.branch_outages
    injection_deltap_to_outage = dynamic_information.non_rel_bb_outage_data.deltap
    node_index_busbars = dynamic_information.non_rel_bb_outage_data.nodal_indices

    # lfs_original = jnp.einsum("bn, tn -> tb", dynamic_information.ptdf, dynamic_information.nodal_injections)
    # lfs_original = lfs_original[:, dynamic_information.branches_monitored]

    connected_branches_data = {}
    for node_index in node_index_busbars:
        station_id = network_data_preprocessed.node_ids[node_index]
        for station in network_data_preprocessed.asset_topology.stations:
            if station_id == station.grid_model_id:
                for index, busbar in enumerate(station.busbars):
                    connected_assets = get_connected_assets(station, index)
                    connected_branches = [asset.grid_model_id for asset in connected_assets if asset.is_branch()]
                    connected_branches = [
                        network_data_preprocessed.branch_ids.index(branch_id) for branch_id in connected_branches
                    ]
                    connected_branches_data[busbar.grid_model_id] = connected_branches

    n_0_flows = jnp.einsum("ij,tj -> ti", dynamic_information.ptdf, dynamic_information.nodal_injections)

    # bubsars_to_be_outaged = [busbar for busbars in list(non_rel_station_busbars_map.values()) for busbar in busbars]
    sorted_busbars_to_be_outaged = []
    for station in network_data_preprocessed.asset_topology.stations:
        if station.grid_model_id in non_rel_station_busbars_map:
            sorted_busbars_to_be_outaged += non_rel_station_busbars_map[station.grid_model_id]

    for outage_index in range(len(connected_branches_to_outage)):
        lfs, success = perform_outage_single_busbar(
            connected_branches_to_outage[outage_index],
            injection_deltap_to_outage[outage_index],
            node_index_busbars[outage_index],
            dynamic_information.ptdf,
            dynamic_information.nodal_injections,
            dynamic_information.from_node,
            dynamic_information.to_node,
            n_0_flows,
            jnp.arange(dynamic_information.ptdf.shape[0]),
            # dynamic_information.branches_monitored
        )

        updated_nodal_injection = dynamic_information.nodal_injections.at[:, node_index_busbars[outage_index]].add(
            -1 * injection_deltap_to_outage[outage_index]
        )
        assert success, "The outage should be successful"

        # Test Case 2: Assert that the load flows are zero for the disconnected branches
        zero_flow_indices = jnp.where(jnp.isclose(lfs[0], 0.0))[0]
        assert all(
            branch in zero_flow_indices
            for branch in connected_branches_to_outage[outage_index]
            if branch >= 0 and branch <= len(network_data_preprocessed.branch_ids)
        ), "All elements of connected_branches_to_outage[i] should be present in zero_flow_indices"

        # Test case 3: assert that all the branches of the outaged busbar has 0 flows. The ones that are disconnected along with the stub and skeleton branches
        node_index = node_index_busbars[outage_index]

        busbar_id = sorted_busbars_to_be_outaged[outage_index]
        connected_branches = connected_branches_data[busbar_id]
        n_skeleton_branches = 0
        for branch_index in connected_branches:
            if branch_index not in connected_branches_to_outage[outage_index]:
                # The branch_indices which are not in connected_branches_to_outage should be
                # either stub branches or skeleton branch. Each busbar should not have more than
                # one skeleton branch
                if network_data_preprocessed.bridging_branch_mask[branch_index]:
                    # This is a stub branch
                    # FIXME: This validation is not working properly. Need to dig deeper into this.
                    validate_power_flow_in_stub_branch(
                        network_data_preprocessed,
                        branch_index,
                        lfs,
                        node_index.tolist(),
                        injection_deltap_to_outage[outage_index],
                        dynamic_information.nodal_injections[:, node_index_busbars[outage_index]],
                    )
                else:
                    # This is a skeleton branch
                    n_skeleton_branches += 1
                    assert jnp.isclose(jnp.abs(lfs[:, branch_index]), jnp.abs(updated_nodal_injection[:, node_index])), (
                        "The load flow in the skeleton branch should be zero"
                    )
        assert n_skeleton_branches <= 1, "There should be at most one skeleton branch per busbar"


def test_perform_outage_single_busbar_with_disconnections(
    jax_inputs_oberrhein, network_data_preprocessed: NetworkData, oberrhein_outage_station_busbars_map
) -> None:
    _, static_information = jax_inputs_oberrhein
    dynamic_information = static_information.dynamic_information

    _, non_rel_station_busbars_map = get_rel_non_rel_sub_bb_maps(
        network_data_preprocessed, oberrhein_outage_station_busbars_map
    )

    connected_branches_to_outage = dynamic_information.non_rel_bb_outage_data.branch_outages
    injection_deltap_to_outage = dynamic_information.non_rel_bb_outage_data.deltap
    node_index_busbars = dynamic_information.non_rel_bb_outage_data.nodal_indices

    lfs_original = jnp.einsum("bn, tn -> tb", dynamic_information.ptdf, dynamic_information.nodal_injections)
    lfs_original = lfs_original[:, dynamic_information.branches_monitored]

    connected_branches_data = {}
    for node_index in node_index_busbars:
        station_id = network_data_preprocessed.node_ids[node_index]
        for station in network_data_preprocessed.asset_topology.stations:
            if station_id == station.grid_model_id:
                for index, busbar in enumerate(station.busbars):
                    connected_assets = get_connected_assets(station, index)
                    connected_branches = [asset.grid_model_id for asset in connected_assets if asset.is_branch()]
                    connected_branches = [
                        network_data_preprocessed.branch_ids.index(branch_id) for branch_id in connected_branches
                    ]
                    connected_branches_data[busbar.grid_model_id] = connected_branches

    n_0_flows = jnp.einsum("ij,tj -> ti", dynamic_information.ptdf, dynamic_information.nodal_injections)

    # bubsars_to_be_outaged = [busbar for busbars in list(non_rel_station_busbars_map.values()) for busbar in busbars]
    sorted_busbars_to_be_outaged = []
    for station in network_data_preprocessed.asset_topology.stations:
        if station.grid_model_id in non_rel_station_busbars_map:
            sorted_busbars_to_be_outaged += non_rel_station_busbars_map[station.grid_model_id]

    # Case 1: if all branches are outaged from 8%%bus (124, 16). In connected_branches_to_outage, only 124 is to be outaged
    # and branch 16 is left as a skeleton branch. This is similar to case when branch 16 is outaged by the
    # disconnection action before the busbar outage.
    branches_to_outage = connected_branches_to_outage[0]  # .at[1].set(16)

    lfs, success = perform_outage_single_busbar(
        branches_to_outage,
        injection_deltap_to_outage[0],
        node_index_busbars[0],
        dynamic_information.ptdf,
        dynamic_information.nodal_injections,
        dynamic_information.from_node,
        dynamic_information.to_node,
        n_0_flows,
        jnp.arange(dynamic_information.ptdf.shape[0]),
        # dynamic_information.branches_monitored
    )

    assert success, "The outage should be successful"


def test_perform_non_rel_bb_outages(
    jax_inputs_oberrhein, network_data_preprocessed: NetworkData, oberrhein_outage_station_busbars_map
) -> None:
    _, static_information = jax_inputs_oberrhein
    dynamic_information = static_information.dynamic_information
    _, non_rel_station_busbars_map = get_rel_non_rel_sub_bb_maps(
        network_data_preprocessed, oberrhein_outage_station_busbars_map
    )
    n_bb_outages = len([busbar for busbars in non_rel_station_busbars_map.values() for busbar in busbars])
    n_timesteps = dynamic_information.nodal_injections.shape[0]
    lfs_outage, success = perform_non_rel_bb_outages(
        n_0_flows=dynamic_information.unsplit_flow,
        ptdf=dynamic_information.ptdf,
        from_node=dynamic_information.from_node,
        to_node=dynamic_information.to_node,
        nodal_injections=dynamic_information.nodal_injections,
        non_rel_bb_outage_data=dynamic_information.non_rel_bb_outage_data,
        branches_monitored=dynamic_information.branches_monitored,
    )
    assert lfs_outage.shape == (
        n_bb_outages,
        n_timesteps,
        dynamic_information.branches_monitored.shape[0],
    ), "Shape of lfs_outage is incorrect"


def test_perform_rel_bb_outage_single_topo_with_no_inj_reassignments(
    jax_inputs_oberrhein: tuple[ActionIndexComputations, StaticInformation],
    network_data_preprocessed: NetworkData,
) -> None:
    topo_indices, static_information = jax_inputs_oberrhein
    di = static_information.dynamic_information
    topo_indices = topo_indices[0]

    # test = pad_out_action_indices_to_all_subs(topo_indices.action, di.action_set)
    # branch_actions = di.action_set.branch_actions.at[test].get(mode="fill", fill_value=False)

    # These indices are neccessarily sorted in the order of rel_subs
    updated_topo_indices = jax.jit(pad_action_with_unsplit_action_indices)(di.action_set, topo_indices.action)
    # updated_topo_indices = update_action_indices_with_defaults(di.action_set, topo_indices.action)

    branch_actions = di.action_set.branch_actions[updated_topo_indices]
    branch_actions = np.array(branch_actions)
    affected_sub_ids = di.action_set.substation_correspondence[updated_topo_indices]

    busbar_data = extract_busbar_metadata_after_split(network_data_preprocessed, static_information, branch_actions)
    splitted_ptdf, from_node, to_node, input_nodal_injections, _ = compute_splits_and_injections(
        static_information, branch_actions, updated_topo_indices, affected_sub_ids
    )
    n_0_flows = jnp.einsum("ij,tj -> ti", splitted_ptdf, input_nodal_injections)

    lfs, success = perform_rel_bb_outage_single_topo(
        n_0_flows=n_0_flows,
        action_set=di.action_set,
        action_indices=updated_topo_indices,
        ptdf=splitted_ptdf,
        nodal_injections=input_nodal_injections,
        from_nodes=from_node,
        to_nodes=to_node,
        branches_monitored=di.branches_monitored,
    )

    lfs = correct_power_flow_directions(lfs, network_data_preprocessed)
    rel_busbar_branches_flows = [
        np.array(lfs_instance)[:, out_branches]
        for lfs_instance, out_branches in zip(lfs, busbar_data["busbar_branches_map"].values())
    ]
    delta_p = jnp.concatenate(di.action_set.rel_bb_outage_data.deltap_set[updated_topo_indices], axis=0)
    nodal_indices = jnp.concatenate(di.action_set.rel_bb_outage_data.nodal_indices[updated_topo_indices])
    unsplit_mask = di.action_set.unsplit_action_mask[updated_topo_indices]
    for outage_index in range(len(lfs)):
        if jnp.isnan(lfs[outage_index]).any() or jnp.all(lfs[outage_index] == 0.0):
            continue

        nodal_index_bb = nodal_indices[outage_index]
        if nodal_index_bb in static_information.solver_config.rel_stat_map.val:
            # This busbar belongs to bus_A
            rel_sub_index = jnp.argmax(static_information.solver_config.rel_stat_map.val == nodal_index_bb)
        else:
            # Thsi busbar belongs the bus_B. Here, we calculate the nodal_index of busbar_A
            rel_sub_index = nodal_index_bb - static_information.solver_config.n_stat
        is_unsplit = unsplit_mask[rel_sub_index]
        updated_nodal_injection = input_nodal_injections.at[:, nodal_indices[outage_index]].add(-1 * delta_p[outage_index])

        if is_unsplit:
            continue
            other_bb_index = -1 * (len(static_information.solver_config.rel_stat_map.val) - rel_sub_index)
            power_at_node = updated_nodal_injection[:, nodal_index_bb] + updated_nodal_injection[:, other_bb_index]
        power_at_node = updated_nodal_injection[:, nodal_index_bb]
        connected_branches_flow = rel_busbar_branches_flows[outage_index]

        assert jnp.allclose(abs(connected_branches_flow.sum(axis=1)), abs(power_at_node), atol=1e-03), (
            "Kirchoff's current law should be satisfied"
        )

    validate_zero_flows(lfs, success, busbar_data["busbar_br_outage_map"])


def get_busbar_injection_map(station: Station, network: NetworkData) -> dict[str, Float[np.ndarray, " n_timestep"]]:
    busbar_injection_map = {}
    for i, bb in enumerate(station.busbars):
        connected_assets = get_connected_assets(station, i)
        injection: Float[np.ndarray, " n_timestep"] = np.zeros(network.nodal_injection.shape[0], float)

        for asset in connected_assets:
            if not asset.is_branch() and asset.in_service:
                if asset.grid_model_id not in network.injection_ids:
                    logger.warning(f"Asset {asset.grid_model_id} is not a valid injection. Might have been removed.")
                    continue
                injection_index = network.injection_ids.index(asset.grid_model_id)
                injection += network.mw_injections[:, injection_index]
        busbar_injection_map[bb.grid_model_id] = injection

    return busbar_injection_map


def extract_busbar_metadata_after_split(network_data_preprocessed, static_information: StaticInformation, branch_actions):
    rel_stations = get_relevant_stations(network_data_preprocessed)
    branch_ids_mapped, _ = map_branch_injection_ids(network_data_preprocessed)

    busbar_branches_map = {}
    busbar_deltap_map = {}
    busbar_br_outage_map = {}
    busbar_injection_map = {}
    for sub_index in range(len(rel_stations)):
        branch_action = branch_actions[sub_index]
        modified_station_br, _, _ = realise_bus_split_single_station(
            branch_ids_local=branch_ids_mapped[sub_index],
            branch_topology_local=branch_action[: static_information.solver_config.branches_per_sub.val[sub_index]],
            injection_ids_local=[],
            injection_topology_local=np.array([], dtype=bool),
            station=rel_stations[sub_index],
        )
        branch_action_combi_index = np.argmax(
            np.all(
                network_data_preprocessed.branch_action_set[sub_index]
                == branch_action[: static_information.solver_config.branches_per_sub.val[sub_index]],
                axis=1,
            )
        )
        for bb in modified_station_br.busbars:
            outage_data = extract_busbar_outage_data(
                modified_station_br, bb.grid_model_id, network_data_preprocessed, {}, branch_action_combi_index
            )
            busbar_deltap_map[bb.grid_model_id] = (outage_data.node_index, outage_data.nodal_injection)
            busbar_br_outage_map[bb.grid_model_id] = outage_data.branch_indices
        busbar_branches_map.update(get_busbar_branches_map(modified_station_br, network_data_preprocessed))
        busbar_injection_map.update(get_busbar_injection_map(modified_station_br, network_data_preprocessed))

    return {
        "busbar_branches_map": busbar_branches_map,
        "busbar_deltap_map": busbar_deltap_map,
        "busbar_br_outage_map": busbar_br_outage_map,
        "busbar_injection_map": busbar_injection_map,
    }


def compute_splits_and_injections(
    static_information: StaticInformation, branch_actions, branch_action_indices, affected_sub_ids, disconnections=None
):
    di = static_information.dynamic_information
    solver_config = static_information.solver_config
    bsdf_results = compute_bus_splits(
        ptdf=di.ptdf,
        from_node=di.from_node,
        to_node=di.to_node,
        tot_stat=di.tot_stat,
        from_stat_bool=di.from_stat_bool,
        susceptance=di.susceptance,
        rel_stat_map=solver_config.rel_stat_map,
        slack=solver_config.slack,
        n_stat=solver_config.n_stat,
        topologies=jnp.array(branch_actions),
        sub_ids=affected_sub_ids,
    )
    splitted_ptdf = bsdf_results.ptdf
    assert jnp.all(bsdf_results.success)

    action_set = di.action_set
    injection_action_all_relevant_subs = action_set[branch_action_indices].inj_actions
    sub_ids = action_set[branch_action_indices].substation_correspondence

    input_nodal_injections = get_injection_vector(
        injection_assignment=injection_action_all_relevant_subs,
        sub_ids=sub_ids,
        relevant_injections=di.relevant_injections,
        nodal_injections=di.nodal_injections,
        n_stat=jnp.array(solver_config.n_stat),
        rel_stat_map=jnp.array(solver_config.rel_stat_map.val),
    )
    from_nodes = bsdf_results.from_node
    to_nodes = bsdf_results.to_node
    success_disonnections = None
    if disconnections is not None:
        disc_res = apply_disconnections(
            ptdf=splitted_ptdf,
            from_node=bsdf_results.from_node,
            to_node=bsdf_results.to_node,
            disconnections=disconnections,
            guarantee_unique=True,
        )
        splitted_ptdf, success_disonnections = disc_res.ptdf, disc_res.success
        from_nodes = disc_res.from_node
        to_nodes = disc_res.to_node
        # from_nodes, to_nodes = update_from_to_nodes_after_disconnections(from_nodes,
        #  to_nodes,
        #  disconnections)

    return splitted_ptdf, from_nodes, to_nodes, input_nodal_injections, success_disonnections


def validate_zero_flows(lfs_list, success_list, busbar_br_outage_map):
    for lfs_instance, out_branches, success in zip(lfs_list, busbar_br_outage_map.values(), success_list):
        if not success:
            continue
        n_zero_flows = jnp.sum(jnp.isclose(lfs_instance[:, out_branches], 0.0))
        # The skeleton branches can have non zero flows.
        assert n_zero_flows == len(out_branches) or n_zero_flows == len(out_branches) - 1, "zero flow validation is false"


def correct_power_flow_directions(lfs, network_data_preprocessed: NetworkData):
    branch_dir = network_data_preprocessed.branch_direction
    branches_at_nodes = network_data_preprocessed.branches_at_nodes
    lfs_corrrected_directions = []
    for lfs_instance in lfs:
        for local_branches, local_branches_dir in zip(branches_at_nodes, branch_dir):
            power_flow_line = lfs_instance.at[:, local_branches].multiply(np.where(local_branches_dir == 1, -1, 1))
        lfs_corrrected_directions.append(power_flow_line)
    return lfs_corrrected_directions


def test_compare_loadflows_non_rel_bb_outage_powsybl(
    test_grid_folder_path: Path,
    network_data_test_grid: NetworkData,
    outage_map_test_grid: dict[str, list[str]],
):
    net = pp.network.load(test_grid_folder_path / "grid.xiidm")
    # outage_map = outage_map_test_grid
    outage_map = network_data_test_grid.busbar_outage_map

    rel_bb_outage_map, non_rel_bb_outage_map = get_rel_non_rel_sub_bb_maps(network_data_test_grid, outage_map)
    network_data = replace(network_data_test_grid, busbar_outage_map=outage_map)
    network_data = preprocess_bb_outages(network_data)
    static_information = convert_to_jax(
        network_data,
        enable_bb_outage=True,
    )

    asset_topology = network_data.simplified_asset_topology
    dynamic_information = static_information.dynamic_information
    lfs_non_rel, success = perform_non_rel_bb_outages(
        n_0_flows=dynamic_information.unsplit_flow,
        ptdf=dynamic_information.ptdf,
        from_node=dynamic_information.from_node,
        to_node=dynamic_information.to_node,
        nodal_injections=dynamic_information.nodal_injections,
        non_rel_bb_outage_data=dynamic_information.non_rel_bb_outage_data,
        branches_monitored=jnp.arange(dynamic_information.ptdf.shape[0]),
    )
    lfs_index = 0
    for station in asset_topology.stations:
        if station.grid_model_id not in non_rel_bb_outage_map:
            continue
        for bb in station.busbars:
            copy_net = copy.deepcopy(net)
            if bb.grid_model_id not in non_rel_bb_outage_map[station.grid_model_id]:
                continue
            bb_index = get_busbar_index(station, bb.grid_model_id)
            connected_assets = get_connected_assets(station, bb_index)
            connected_asset_ids = [asset.grid_model_id for asset in connected_assets]
            connected_branch_indices = [
                network_data.branch_ids.index(asset.grid_model_id) for asset in connected_assets if asset.is_branch()
            ]
            copy_net.remove_elements(connected_asset_ids)
            config = pp.loadflow.Parameters(
                distributed_slack=False,
            )
            pp.loadflow.run_dc(copy_net, parameters=config)

            # Get the stub branches connected to this busbar
            stub_branch_indices = []
            for asset in connected_assets:
                if asset.is_branch():
                    br_index = network_data.branch_ids.index(asset.grid_model_id)
                    if network_data.bridging_branch_mask[br_index]:
                        stub_branch_indices.append(br_index)

            # All the loadflows should match except the stub branch and the disconnected branches where the load flow should be zero.
            # The stub branch lf will not match
            for index, branch_model_id in enumerate(network_data.branch_ids):
                # If index in connected_branch, then the load flow should be zero

                if index in stub_branch_indices:
                    assert branch_model_id not in copy_net.get_lines()["p2"]
                elif index in connected_branch_indices:
                    assert jnp.isclose(lfs_non_rel[lfs_index][0][index], 0.0)
                else:
                    lf_match = jnp.isclose(copy_net.get_lines()["p2"][branch_model_id], lfs_non_rel[lfs_index][0][index])
                    assert lf_match, f"Load flow mismatch for branch {branch_model_id}"
            lfs_index += 1


def test_compare_loadflows_rel_bb_outage(
    test_grid_folder_path: Path,
    network_data_test_grid: NetworkData,
    outage_map_test_grid: dict[str, list[str]],
    jax_inputs_test_grid: tuple[ActionIndexComputations, StaticInformation],
):
    net = pp.network.load(test_grid_folder_path / "grid.xiidm")
    nd_action_set = extract_action_set(network_data_test_grid)

    topo_indices, static_information = jax_inputs_test_grid
    topo_indices = topo_indices[0]
    di = static_information.dynamic_information

    # These indices are neccessarily sorted in the order of rel_subs
    updated_topo_indices = jax.jit(pad_action_with_unsplit_action_indices)(di.action_set, topo_indices.action)
    rel_bb_outage_map, _ = get_rel_non_rel_sub_bb_maps(network_data_test_grid, outage_map_test_grid)

    branch_actions = di.action_set.branch_actions[updated_topo_indices]
    affected_sub_ids = di.action_set.substation_correspondence[updated_topo_indices]

    splitted_ptdf, from_node, to_node, input_nodal_injections, _ = compute_splits_and_injections(
        static_information, branch_actions, topo_indices.action, affected_sub_ids
    )
    n_0_flows = jnp.einsum("ij,tj -> ti", splitted_ptdf, input_nodal_injections)
    lfs_rel_list, success = perform_rel_bb_outage_single_topo(
        n_0_flows=n_0_flows,
        action_set=di.action_set,
        action_indices=updated_topo_indices,
        ptdf=splitted_ptdf,
        nodal_injections=input_nodal_injections,
        from_nodes=from_node,
        to_nodes=to_node,
        branches_monitored=jnp.arange(di.ptdf.shape[0]),
        disconnections=None,
    )

    asset_topology = network_data_test_grid.asset_topology

    # Apply branch_action on the powsybl network
    lfs_index = 0
    net, _ = apply_topology(net, actions=topo_indices.action.tolist(), action_set=nd_action_set)

    for station in asset_topology.stations:
        if station.grid_model_id not in rel_bb_outage_map:
            continue

        rel_sub_index = np.argmax(
            network_data_test_grid.relevant_nodes == (np.argmax(network_data_test_grid.node_ids == station.grid_model_id))
        )
        modified_station = network_data_test_grid.realised_stations[rel_sub_index][updated_topo_indices[rel_sub_index]]

        for bb in station.busbars:
            if bb.grid_model_id not in rel_bb_outage_map[modified_station.grid_model_id]:
                continue
            copy_net = copy.deepcopy(net)
            bb_index = get_busbar_index(modified_station, bb.grid_model_id)
            connected_assets = get_connected_assets(modified_station, bb_index)
            connected_asset_ids = [asset.grid_model_id for asset in connected_assets]
            connected_branch_indices = [
                network_data_test_grid.branch_ids.index(asset.grid_model_id)
                for asset in connected_assets
                if asset.is_branch()
            ]
            copy_net.remove_elements(connected_asset_ids)
            config = pp.loadflow.Parameters(
                distributed_slack=False,
            )
            pp.loadflow.run_dc(copy_net, parameters=config)

            # Get the stub branches connected to this busbar
            stub_branch_indices = []
            for asset in connected_assets:
                if asset.is_branch():
                    br_index = network_data_test_grid.branch_ids.index(asset.grid_model_id)
                    if network_data_test_grid.bridging_branch_mask[br_index]:
                        stub_branch_indices.append(br_index)

            # The stub branch and skeleton branch lf may not match. Also, the loadflow in skeleton branch and stub branch
            # will not neccessarily be 0

            # lfs_dict = {
            #     branch_model_id: lfs_rel_list[lfs_index][0][network_data_test_grid.branch_ids.index(branch_model_id)]
            #     for branch_model_id in network_data_test_grid.branch_ids
            # }
            n_skeleton_branch = 0
            for index, branch_model_id in enumerate(network_data_test_grid.branch_ids):
                # If index in connected_branch, then the load flow should be zero
                if index in stub_branch_indices:
                    assert branch_model_id not in copy_net.get_lines()["p2"]
                elif index in connected_branch_indices:
                    lf_is_0 = jnp.isclose(lfs_rel_list[lfs_index][0][index], 0.0)
                    if not lf_is_0 and n_skeleton_branch < 1:
                        n_skeleton_branch += 1
                        assert n_skeleton_branch <= 1, "There should be at most one skeleton branch per busbar"
                    else:
                        assert lf_is_0, f"Load flow for branch {branch_model_id} should be zero"
                else:
                    lf_match = jnp.isclose(copy_net.get_lines()["p2"][branch_model_id], lfs_rel_list[lfs_index][0][index])
                    assert lf_match, f"Load flow mismatch for branch {branch_model_id}"
            lfs_index += 1


def create_dummy_rel_bb_outage_data(
    n_br_combis: int, n_max_bb_to_outage_per_sub: int, max_branches_per_sub: int, n_timesteps: int, seed: int
) -> RelBBOutageData:
    """Creates a dummy RelBBOutageData object with the given dimensions filled with random data"""
    return RelBBOutageData(
        branch_outage_set=jax.random.randint(
            jax.random.PRNGKey(seed),
            (n_br_combis, n_max_bb_to_outage_per_sub, max_branches_per_sub),
            0,
            100,
            dtype=jnp.int64,
        ),
        deltap_set=jax.random.uniform(
            jax.random.PRNGKey(seed + 1), (n_br_combis, n_max_bb_to_outage_per_sub, n_timesteps), dtype=float
        ),
        nodal_indices=jax.random.randint(
            jax.random.PRNGKey(seed + 2), (n_br_combis, n_max_bb_to_outage_per_sub), 0, 100, dtype=int
        ),
        articulation_node_mask=jax.random.bernoulli(
            jax.random.PRNGKey(seed + 3), 0.5, (n_br_combis, n_max_bb_to_outage_per_sub)
        ),
    )


def test_remove_critical_busbars_from_outage():
    rel_outage_data = create_dummy_rel_bb_outage_data(
        n_br_combis=10,
        n_max_bb_to_outage_per_sub=4,
        max_branches_per_sub=5,
        n_timesteps=1,
        seed=42,
    )
    branch_action_indices = jnp.array([3, 7, 8, 10])
    branch_outages, nodal_indices, deltap_outages = remove_articulation_nodes_from_bb_outage(
        rel_outage_data, branch_action_indices
    )
    assert rel_outage_data.branch_outage_set[branch_action_indices].shape == branch_outages.shape, (
        "Shape of branch_outages is incorrect"
    )
    assert jnp.all(
        jnp.isclose(
            ~jnp.all(rel_outage_data.branch_outage_set[branch_action_indices] == branch_outages, axis=2),
            rel_outage_data.articulation_node_mask[branch_action_indices],
        )
    ), "Branch outage data is incorrect"
    assert jnp.all(
        jnp.isclose(
            ~jnp.all(rel_outage_data.deltap_set[branch_action_indices] == deltap_outages, axis=2),
            rel_outage_data.articulation_node_mask[branch_action_indices],
        )
    ), "Deltap data is incorrect"
    assert jnp.all(
        jnp.isclose(
            ~(rel_outage_data.nodal_indices[branch_action_indices] == nodal_indices),
            rel_outage_data.articulation_node_mask[branch_action_indices],
        )
    ), "Nodal indices are incorrect"


def test_perform_rel_bb_outage_for_unsplit_grid(
    jax_inputs_oberrhein: tuple[ActionIndexComputations, StaticInformation],
):
    _, static_information = jax_inputs_oberrhein
    di = static_information.dynamic_information

    action_indices = pad_action_with_unsplit_action_indices(
        di.action_set,
        jnp.full((1,), int_max(), dtype=int),
    )

    injection_action_all_relevant_subs = di.action_set[action_indices].inj_actions
    sub_ids = di.action_set[action_indices].substation_correspondence
    input_nodal_injections = get_injection_vector(
        injection_assignment=injection_action_all_relevant_subs,
        sub_ids=sub_ids,
        relevant_injections=di.relevant_injections,
        nodal_injections=di.nodal_injections,
        n_stat=jnp.array(static_information.solver_config.n_stat),
        rel_stat_map=jnp.array(static_information.solver_config.rel_stat_map.val),
    )
    n_0_flows = di.unsplit_flow
    lfs, success = perform_rel_bb_outage_for_unsplit_grid(
        n_0_flows, di.ptdf, input_nodal_injections, di.from_node, di.to_node, di.action_set, di.branches_monitored
    )

    assert jnp.all(success), "The outage should be successful"


def test_perform_rel_bb_outage_batched(
    jax_inputs_oberrhein: tuple[ActionIndexComputations, StaticInformation], network_data_preprocessed: NetworkData
) -> None:
    batch_topo_indices, static_information = jax_inputs_oberrhein
    di = static_information.dynamic_information

    bitvector_topology = convert_action_set_index_to_topo(batch_topo_indices, di.action_set)
    topo_res = compute_bsdf_lodf_static_flows(bitvector_topology, None, di, static_information.solver_config)
    sub_ids = jnp.where(
        bitvector_topology.topologies.any(axis=-1),
        bitvector_topology.sub_ids,
        int_max(),
    )
    injections = di.action_set.inj_actions.at[batch_topo_indices.action].get(mode="fill", fill_value=False)
    n_0_flows, cross_coupler_flows = jax.vmap(
        compute_cross_coupler_flows,
        in_axes=(0, 0, 0, 0, None, None, None, None),
    )(
        topo_res.bsdf,
        bitvector_topology.topologies,
        sub_ids,
        injections,
        di.relevant_injections,
        di.unsplit_flow,
        di.tot_stat,
        di.from_stat_bool,
    )
    nodal_injections = compute_injections(
        injections=injections,
        sub_ids=sub_ids,
        dynamic_information=di,
        solver_config=static_information.solver_config,
    )

    padded_action_indices = jax.vmap(pad_action_with_unsplit_action_indices, in_axes=(None, 0))(
        di.action_set, batch_topo_indices.action
    )
    lfs, success = perform_rel_bb_outage_batched(
        n_0_flows,
        padded_action_indices,
        topo_res.ptdf,
        nodal_injections,
        topo_res.from_node,
        topo_res.to_node,
        di.action_set,
        di.branches_monitored,
    )

    branch_actions = di.action_set.branch_actions[padded_action_indices]
    for batch_index in range(batch_topo_indices.action.shape[0]):
        busbar_data = extract_busbar_metadata_after_split(
            network_data_preprocessed, static_information, branch_actions[batch_index]
        )
        validate_zero_flows(lfs[batch_index], success[batch_index], busbar_data["busbar_br_outage_map"])


def test_get_busbar_outage_penalty_batched(
    jax_inputs_oberrhein: tuple[ActionIndexComputations, StaticInformation], network_data_preprocessed: NetworkData
) -> None:
    batch_topo_indices, static_information = jax_inputs_oberrhein
    di = static_information.dynamic_information

    bitvector_topology = convert_action_set_index_to_topo(batch_topo_indices, di.action_set)
    topo_res = compute_bsdf_lodf_static_flows(bitvector_topology, None, di, static_information.solver_config)
    sub_ids = jnp.where(
        bitvector_topology.topologies.any(axis=-1),
        bitvector_topology.sub_ids,
        int_max(),
    )
    injections = di.action_set.inj_actions.at[batch_topo_indices.action].get(mode="fill", fill_value=False)
    n_0_flows, cross_coupler_flows = jax.vmap(
        compute_cross_coupler_flows,
        in_axes=(0, 0, 0, 0, None, None, None, None),
    )(
        topo_res.bsdf,
        bitvector_topology.topologies,
        sub_ids,
        injections,
        di.relevant_injections,
        di.unsplit_flow,
        di.tot_stat,
        di.from_stat_bool,
    )
    nodal_injections = compute_injections(
        injections=injections,
        sub_ids=sub_ids,
        dynamic_information=di,
        solver_config=static_information.solver_config,
    )

    padded_action_indices = jax.vmap(pad_action_with_unsplit_action_indices, in_axes=(None, 0))(
        di.action_set, batch_topo_indices.action
    )

    unsplit_bb_outage_analysis = get_bb_outage_baseline_analysis(
        di,
        1000.0,
    )

    penalty, overloads, failures = get_busbar_outage_penalty_batched(
        n_0_flows,
        padded_action_indices,
        topo_res.ptdf,
        nodal_injections,
        topo_res.from_node,
        topo_res.to_node,
        di.action_set,
        di.branches_monitored,
        unsplit_bb_outage_analysis,
    )

    assert penalty.shape == (static_information.solver_config.batch_size_bsdf,)


def test_busbar_outage_penalty(jax_inputs_oberrhein: tuple[ActionIndexComputations, StaticInformation]) -> None:
    _, static_information = jax_inputs_oberrhein
    di = static_information.dynamic_information

    lfs, success = perform_rel_bb_outage_for_unsplit_grid(
        di.unsplit_flow,
        di.ptdf,
        di.nodal_injections,
        di.from_node,
        di.to_node,
        di.action_set,
        di.branches_monitored,
    )

    unsplit_bb_outage_analysis = get_bb_outage_baseline_analysis(
        di,
        1000.0,
    )

    penalty, overload, failures = get_busbar_outage_penalty(
        unsplit_bb_outage_analysis,
        lfs,
        success,
    )

    assert penalty == 0.0, "Penalty should be 0.0 when the grid is unsplit"


def test_filter_already_outaged_branches_single_outage():
    # Test case 1: No branches are already outaged
    branch_outages = jnp.array([1, 2, 3, 4])
    disconnections = jnp.array([5, 6])
    filtered_branches = filter_already_outaged_branches_single_outage(branch_outages, disconnections)
    assert jnp.array_equal(filtered_branches, branch_outages), "Branches should remain unchanged when no overlap exists."

    # Test case 2: Some branches are already outaged
    branch_outages = jnp.array([1, 2, 3, 4, int_max()])
    disconnections = jnp.array([2, 4])
    filtered_branches = filter_already_outaged_branches_single_outage(branch_outages, disconnections)
    expected_result = jnp.array([1, 3, int_max(), int_max(), int_max()])
    assert jnp.array_equal(filtered_branches, expected_result), "Branches already outaged should be set to int_max()."

    # Test case 3: All branches are already outaged
    branch_outages = jnp.array([1, 2, 3, 4])
    disconnections = jnp.array([1, 2, 3, 4])
    filtered_branches = filter_already_outaged_branches_single_outage(branch_outages, disconnections)
    expected_result = jnp.array([int_max(), int_max(), int_max(), int_max()])
    assert jnp.array_equal(filtered_branches, expected_result), "All branches should be set to int_max()."

    # Test case 4: Empty branch_outages
    branch_outages = jnp.array([])
    disconnections = jnp.array([1, 2])
    filtered_branches = filter_already_outaged_branches_single_outage(branch_outages, disconnections)
    assert filtered_branches.size == 0, "Filtered branches should be empty when branch_outages is empty."

    # Test case 5: Empty disconnections
    branch_outages = jnp.array([1, 2, 3, 4])
    disconnections = jnp.array([])
    filtered_branches = filter_already_outaged_branches_single_outage(branch_outages, disconnections)
    assert jnp.array_equal(filtered_branches, branch_outages), (
        "Branches should remain unchanged when disconnections are empty."
    )


def perform_rel_bb_outage_single_topo_unjaxed(
    action_indices: Int[Array, " n_rel_subs"],
    ptdf: Float[Array, " n_branches n_nodes"],
    nodal_injections: Float[Array, " n_timesteps n_nodes"],
    from_nodes: Int[Array, " n_branches"],
    to_nodes: Int[Array, " n_branches"],
    action_set: ActionSet,
    branches_monitored: Int[Array, " n_branches_monitored"],
    disconnections: Int[Array, "n_disconnections"] = None,
) -> tuple[Float[Array, " n_bb_outages n_timesteps n_branches_monitored"], Bool[Array, " n_bb_outages"]]:
    """Perform a relevant busbar outage for a single topology.

    This function calculates the impact of a relevant busbar outage on the power grid
    by updating nodal injections and loadlfows.

    Parameters
    ----------
    action_indices : Int[Array, " n_rel_subs"]
        Indices of the actions to be performed for the relative busbar outage.
    ptdf : Float[Array, " n_branches n_nodes"]
        Power Transfer Distribution Factor (PTDF) matrix.
    nodal_injections : Float[Array, " n_timesteps n_nodes"]
        Nodal injection values for each timestep and node.
    from_nodes : Int[Array, " n_branches"]
        Array of "from" nodes for each branch.
    to_nodes : Int[Array, " n_branches"]
        Array of "to" nodes for each branch.
    action_set : ActionSet
        ActionSet object containing information about branch actions and relative busbar outage data.
    branches_monitored : Int[Array, " n_branches_monitored"]
        Indices of branches to be monitored during the outage.
    disconnections : Int[Array, "n_disconnections"], optional
        Array of disconnection actions which were performed before the busbar outage as part of
        topological actions. This is used to prevent double outage of branches that are already outaged
        by the disconnection action. If not provided, defaults to None.

    Returns
    -------
    lfs_list : Float[Array, " n_bb_outages n_timesteps n_branches_monitored"]
        Array of load flow solutions for each busbar outages, timestep and branch.
    success : list[Bool[Array, " "]]
        Array indicating the success or failure of the outage calculations for each busbar outage.

    Raises
    ------
    AssertionError
        If the branch outage set is None or if there is a mismatch between the branch action set
        and the branch outage set.
    """
    branch_action_set = action_set.branch_actions

    assert action_set.rel_bb_outage_data.branch_outage_set is not None, (
        "Branch outage set is None in dynamic information. Perform the outage calculation first."
    )
    assert branch_action_set.shape[0] == action_set.rel_bb_outage_data.branch_outage_set.shape[0], (
        "Mismatch in branch action set and branch outage set."
    )

    branch_outages, nodal_indices_outages, deltap_outages = remove_articulation_nodes_from_bb_outage(
        action_set.rel_bb_outage_data, action_indices
    )
    # Note: branch_indices with value -1 or int_max are automatically ignored in the  build_modf_matrix  function
    branch_outages: Int[Array, " n_rel_subs*max_n_physical_bb_per_sub max_branches_per_sub"] = jnp.concatenate(
        branch_outages, axis=0
    )
    deltap_outages: Float[Array, " n_rel_subs*max_n_physical_bb_per_sub n_timesteps"] = jnp.concatenate(
        deltap_outages, axis=0
    )
    nodal_indices_outages: Int[Array, " n_rel_subs*max_n_physical_bb_per_sub "] = jnp.concatenate(
        nodal_indices_outages, axis=0
    )

    # Handle disconnection actions: Prevent double outage of branches
    # that are already outaged by the disconnection action.
    branch_outages = jax.vmap(filter_already_outaged_branches_single_outage, in_axes=(0, None))(
        branch_outages, disconnections
    )
    n_0_flows = jnp.einsum("ij,tj -> ti", ptdf, nodal_injections)

    lfs_list = []
    sucess_list = []
    for i in range(branch_outages.shape[0]):
        lfs, success = perform_outage_single_busbar(
            branch_outages[i],
            deltap_outages[i],
            nodal_indices_outages[i],
            ptdf,
            nodal_injections,
            from_nodes,
            to_nodes,
            n_0_flows,
            branches_monitored,
        )
        lfs_list.append(lfs)
        sucess_list.append(success)
    lfs_list = jnp.stack(lfs_list)
    sucess_list = jnp.stack(sucess_list)

    return lfs_list, sucess_list


@pytest.mark.skip(reason="bb outages need a general rework")
def test_perform_rel_bb_outage_with_disconnections(
    jax_inputs_oberrhein: tuple[ActionIndexComputations, StaticInformation],
    network_data_preprocessed: NetworkData,
) -> None:
    topo_indices, static_information = jax_inputs_oberrhein
    di = static_information.dynamic_information
    topo_indices = topo_indices[0]

    # These indices are neccessarily sorted in the order of rel_subs
    updated_topo_indices = jax.jit(pad_action_with_unsplit_action_indices)(di.action_set, topo_indices.action)
    # updated_topo_indices = update_action_indices_with_defaults(di.action_set, topo_indices.action)

    branch_actions = di.action_set.branch_actions[updated_topo_indices]
    branch_actions = np.array(branch_actions)
    affected_sub_ids = di.action_set.substation_correspondence[updated_topo_indices]

    busbar_data = extract_busbar_metadata_after_split(network_data_preprocessed, static_information, branch_actions)

    # Simulate disconnections
    # 18 - skeleton branch outage at 71%%bus
    # 185 - duplicate branch outage at 71%%bus
    # 11 - double outage at 157%%bus (This is the only branch that can be outaged)
    # 181 - duplicate outage at 165%%bus
    # 182 - skeleton branch at 165%%bus
    # 36 - skeleton branch at 348%%bus

    disconnections = jnp.array(
        [
            18,
            185,
            11,
            181,
            36,
        ]
    )  # Example disconnections

    splitted_ptdf, from_node, to_node, input_nodal_injections, success_disonnections = compute_splits_and_injections(
        static_information, branch_actions, updated_topo_indices, affected_sub_ids, disconnections=disconnections
    )
    assert jnp.all(success_disonnections)
    n_0_flows = jnp.einsum("ij,tj -> ti", splitted_ptdf, input_nodal_injections)

    lfs, success = perform_rel_bb_outage_single_topo(
        n_0_flows=n_0_flows,
        action_set=di.action_set,
        action_indices=updated_topo_indices,
        ptdf=splitted_ptdf,
        nodal_injections=input_nodal_injections,
        from_nodes=from_node,
        to_nodes=to_node,
        branches_monitored=di.branches_monitored,
        disconnections=disconnections,
    )
    assert jnp.all(success)

    disconnected_branches_flows = [lfs_instance[:, disconnections] for lfs_instance in lfs]

    zero_flows_successful_disconnections = jnp.all(
        ~jnp.logical_xor(~jnp.array(disconnected_branches_flows[0].squeeze(0), dtype=bool), jnp.array(success_disonnections))
    )
    assert zero_flows_successful_disconnections, "The disconnected branches should have zero flows"
    validate_zero_flows(lfs, success, busbar_data["busbar_br_outage_map"])
