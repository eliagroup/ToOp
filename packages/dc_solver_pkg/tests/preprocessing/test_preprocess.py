# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandapower as pp
import pytest
from beartype.typing import Optional, get_args
from fsspec.implementations.dirfs import DirFileSystem
from pandapower.pypower.makePTDF import makePTDF
from toop_engine_dc_solver.jax.inputs import (
    load_static_information,
    save_static_information,
    validate_static_information,
)
from toop_engine_dc_solver.preprocess.convert_to_jax import convert_to_jax
from toop_engine_dc_solver.preprocess.helpers.find_bridges import (
    find_n_minus_2_safe_branches,
)
from toop_engine_dc_solver.preprocess.helpers.injection_topology import (
    get_mw_injections_at_nodes,
    identify_inactive_injections,
)
from toop_engine_dc_solver.preprocess.helpers.node_grouping import (
    get_num_elements_per_node,
    group_by_node,
)
from toop_engine_dc_solver.preprocess.helpers.relevant_branches import (
    get_relevant_branches,
)
from toop_engine_dc_solver.preprocess.network_data import validate_network_data
from toop_engine_dc_solver.preprocess.pandapower.pandapower_backend import PandaPowerBackend
from toop_engine_dc_solver.preprocess.preprocess import (
    NetworkData,
    add_bus_b_columns_to_ptdf,
    add_nodal_injections_to_network_data,
    combine_phaseshift_and_injection,
    compute_branch_topology_info,
    compute_bridging_branches,
    compute_electrical_actions,
    compute_injection_actions,
    compute_injection_topology_info,
    compute_psdf_if_not_given,
    compute_ptdf_if_not_given,
    compute_separation_set_for_stations,
    convert_multi_outages,
    exclude_bridges_from_outage_masks,
    filter_disconnectable_branches_nminus2,
    filter_inactive_injections,
    filter_relevant_nodes_branch_count,
    filter_relevant_nodes_no_asset_station,
    preprocess,
    process_injection_outages,
    reduce_branch_dimension,
    reduce_node_dimension,
    simplify_asset_topology,
)
from toop_engine_dc_solver.preprocess.preprocess_bb_outage import get_busbar_map_adjacent_branches
from toop_engine_dc_solver.preprocess.preprocess_station_realisations import enumerate_station_realisations
from toop_engine_grid_helpers.pandapower.pandapower_helpers import (
    get_pandapower_branch_loadflow_results_sequence,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import table_ids
from toop_engine_interfaces.asset_topology import Station
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_heartbeat import PreprocessStage


def test_compute_ptdf_if_not_given(data_folder: str, network_data: NetworkData) -> None:
    grid_file_path = Path(data_folder) / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    net = pp.from_json(grid_file_path)
    pp.rundcpp(net)
    expected_ptdf = makePTDF(
        net._ppc["baseMVA"],
        net._ppc["internal"]["bus"],
        net._ppc["internal"]["branch"],
        using_sparse_solver=True,
    )

    assert network_data.ptdf is None
    network_data = compute_ptdf_if_not_given(network_data)
    assert network_data.ptdf is not None

    assert np.isnan(expected_ptdf).sum() == 0
    assert np.isnan(network_data.ptdf).sum() == 0
    assert network_data.ptdf.shape == (
        len(network_data.branch_ids),
        len(network_data.node_ids),
    )
    assert np.allclose(network_data.ptdf, expected_ptdf, rtol=1e-3)


def test_compute_ptdf_if_not_given_does_not_override(network_data: NetworkData) -> None:
    assert network_data.ptdf is None
    existing_ptdf = np.array([[1, 2, 3], [2, 3, 4]], dtype=float)
    network_data = replace(network_data, ptdf=existing_ptdf)
    network_data = compute_ptdf_if_not_given(network_data)
    assert np.allclose(network_data.ptdf, existing_ptdf)


def test_compute_psdf_if_not_given_returns_correct_shape_and_types(
    network_data_with_ptdf: NetworkData,
) -> None:
    assert network_data_with_ptdf.psdf is None
    network_data_with_ptdf = compute_psdf_if_not_given(network_data_with_ptdf)
    assert np.isnan(network_data_with_ptdf.psdf).sum() == 0
    assert network_data_with_ptdf.psdf.shape == (
        network_data_with_ptdf.ptdf.shape[0],
        network_data_with_ptdf.phase_shift_mask.sum(),
    )


def test_compute_psdf_if_not_given_does_not_override(
    network_data_with_ptdf: NetworkData,
) -> None:
    assert network_data_with_ptdf.psdf is None
    existing_psdf = np.array([[1, 2, 3], [2, 3, 4]], dtype=float)
    network_data_with_ptdf = replace(network_data_with_ptdf, psdf=existing_psdf)
    network_data_with_ptdf = compute_psdf_if_not_given(network_data_with_ptdf)
    assert np.allclose(network_data_with_ptdf.psdf, existing_psdf)


def test_add_nodal_injections_to_network_data(data_folder: str, network_data: NetworkData) -> None:
    grid_file_path = Path(data_folder) / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    net = pp.from_json(grid_file_path)
    pp.rundcpp(net)
    slack = net.ext_grid.iloc[0]["bus"].item() if len(net.ext_grid) else net.gen.loc[net.gen.slack].bus.values[0]
    slack = net.bus.index.get_loc(slack)
    network_data = add_nodal_injections_to_network_data(network_data)
    # We are only interestend in the first timestep in this test case
    nodal_injections = network_data.nodal_injection[0]
    # The value of the slack does not matter for us as the slack column in the ptdf is zero.
    nodal_injections_without_slack = np.delete(nodal_injections, [slack])
    # Get nodal injections from pandapower
    pp_nodal_injections = np.concatenate([net.res_bus.p_mw.values, [0] * (len(net.trafo3w) + len(net.xward))])
    if len(net.dcline) > 0:
        # Pandapower creates gens from dc lines. When summing up injections for res_bus it adds the gen-powers and subtracts the dcline powers leading to a net-zero nodal injection for the dcline
        np.add.at(
            pp_nodal_injections,
            net.dcline.from_bus.values.astype(int),
            net.res_dcline.p_from_mw.values,
        )
        np.add.at(
            pp_nodal_injections,
            net.dcline.to_bus.values.astype(int),
            net.res_dcline.p_to_mw.values,
        )
    pp_nodal_injections_without_slack = np.delete(pp_nodal_injections, [slack])
    assert np.allclose(nodal_injections_without_slack, pp_nodal_injections_without_slack)

    # Get nodal injections from pypower
    ppci_nodal_injections = net._ppc["internal"]["bus"][:, 2] + net._ppc["internal"]["bus"][:, 4]  # (load + shunt)
    np.add.at(
        ppci_nodal_injections,
        net._ppc["internal"]["gen"][:, 0].astype(int),
        -1 * net._ppc["internal"]["gen"][:, 1],
    )
    ppci_nodal_injections_without_slack = np.delete(ppci_nodal_injections, [slack])
    assert np.allclose(nodal_injections_without_slack, ppci_nodal_injections_without_slack)

    assert len(network_data.node_ids) == len(nodal_injections)
    assert len(network_data.node_names) == len(nodal_injections)
    assert len(network_data.node_types) == len(nodal_injections)
    assert len(network_data.relevant_node_mask) == len(nodal_injections)


def test_compute_bridging_branches(data_folder: str, network_data: NetworkData) -> None:
    np.random.seed(0)
    num_test_branches = 100
    grid_file_path = Path(data_folder) / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    net = pp.from_json(grid_file_path)
    pp.rundcpp(net)
    network_data = compute_bridging_branches(network_data)
    assert len(net._isolated_buses) == 0
    test_branches = np.random.rand(len(network_data.bridging_branch_mask)) < (
        num_test_branches / len(network_data.bridging_branch_mask)
    )
    branch_types = np.array(network_data.branch_types)[test_branches]
    branch_ids = np.array(table_ids(network_data.branch_ids))[test_branches]
    branch_bridge_status = network_data.bridging_branch_mask[test_branches]
    # Find bridging branches by taking elements out of service and see, if that leads to any isolated buses
    for branch_type, pp_id, expected_result in zip(branch_types, branch_ids, branch_bridge_status):
        if branch_type not in ["line", "trafo"]:
            continue
        outage_net = deepcopy(net)
        outage_net[branch_type].loc[pp_id, "in_service"] = False
        pp.rundcpp(outage_net)
        branch_is_bridge = len(outage_net._isolated_buses) > 0
        assert branch_is_bridge == expected_result


def test_combine_phaseshift_and_injection_shapes(
    network_data_filled: NetworkData,
) -> None:
    assert network_data_filled.ptdf is not None
    assert network_data_filled.psdf is not None
    assert network_data_filled.nodal_injection is not None
    network_data = deepcopy(network_data_filled)
    network_data = combine_phaseshift_and_injection(network_data)
    n_timestep = 1
    # Make sure everything has the correct dimensions
    new_node_length = network_data_filled.phase_shift_mask.sum() + len(network_data_filled.node_ids)
    assert network_data.ptdf.shape == (
        len(network_data_filled.branch_ids),
        new_node_length,
    )

    # Nodal stuff should be lenght of ptdf-columns and equal to phaseshifters + actual nodes
    assert network_data.nodal_injection.shape == (n_timestep, new_node_length)
    assert len(network_data.node_ids) == (new_node_length)
    assert len(network_data.node_names) == (new_node_length)
    assert len(network_data.node_types) == (new_node_length)
    assert network_data.relevant_node_mask.shape == (new_node_length,)

    # injections should be of length phaseshift + actual injections
    new_inj_length = network_data_filled.phase_shift_mask.sum() + len(network_data_filled.injection_ids)
    assert network_data.injection_nodes.shape == (new_inj_length,)
    assert network_data.mw_injections.shape == (n_timestep, new_inj_length)
    assert len(network_data.injection_ids) == (new_inj_length)
    assert len(network_data.injection_names) == (new_inj_length)

    # Branches should stay the same
    branch_length = len(network_data_filled.branch_ids)
    assert len(network_data.branch_ids) == (branch_length)
    assert len(network_data.branch_names) == (branch_length)
    assert len(network_data.branch_types) == (branch_length)
    assert network_data.monitored_branch_mask.shape == (branch_length,)
    assert network_data.outaged_branch_mask.shape == (branch_length,)
    assert network_data.max_mw_flows.shape == (n_timestep, branch_length)
    assert network_data.from_nodes.shape == (branch_length,)
    assert network_data.to_nodes.shape == (branch_length,)
    assert network_data.susceptances.shape == (branch_length,)
    assert network_data.bridging_branch_mask.shape == (branch_length,)
    assert network_data.controllable_phase_shift_mask.shape == (branch_length,)
    assert network_data.controllable_pst_node_mask.shape == (new_node_length,)


def test_combine_phaseshift_and_injection_logic(
    network_data_filled: NetworkData,
) -> None:
    assert network_data_filled.ptdf is not None
    assert network_data_filled.psdf is not None
    assert network_data_filled.nodal_injection is not None
    number_of_phase_shifters = network_data_filled.phase_shift_mask.sum()
    network_data = deepcopy(network_data_filled)
    network_data = combine_phaseshift_and_injection(network_data)
    # test if loadflow is the same
    loadflow_before = (
        network_data_filled.psdf @ (network_data_filled.shift_angles[0, network_data_filled.phase_shift_mask])
        + network_data_filled.ptdf @ network_data_filled.nodal_injection[0]
    )
    loadflow_after = network_data.ptdf @ network_data.nodal_injection[0]
    assert np.allclose(loadflow_before, loadflow_after)

    # Relevant nodes, injection_nodes and from_nodes and to_nodes in ptdf the same
    assert np.allclose(
        network_data_filled.ptdf[:, network_data_filled.relevant_node_mask],
        network_data.ptdf[:, network_data.relevant_node_mask],
    )
    assert np.allclose(
        network_data_filled.ptdf[:, network_data_filled.from_nodes],
        network_data.ptdf[:, network_data.from_nodes],
    )
    assert np.allclose(
        network_data_filled.ptdf[:, network_data_filled.to_nodes],
        network_data.ptdf[:, network_data.to_nodes],
    )
    # For injection nodes we compare PST and Injection separately
    assert np.allclose(
        network_data_filled.ptdf[:, network_data_filled.injection_nodes],
        network_data.ptdf[:, network_data.injection_nodes[:-number_of_phase_shifters]],
    )
    assert np.allclose(
        network_data_filled.psdf,
        network_data.ptdf[:, network_data.injection_nodes[-number_of_phase_shifters:]],
    )

    # Slack column still zero
    assert network_data.ptdf[:, network_data.slack].sum() == 0

    # From and to Nodes in PTDF the same
    assert np.allclose(
        network_data_filled.ptdf[:, network_data_filled.relevant_node_mask],
        network_data.ptdf[:, network_data.relevant_node_mask],
    )

    # Nodal reporting stuff is the same  if you remove phaseshifters in front
    assert network_data.node_ids[number_of_phase_shifters:] == network_data_filled.node_ids
    assert network_data.node_names[number_of_phase_shifters:] == network_data_filled.node_names
    assert network_data.node_types[number_of_phase_shifters:] == network_data_filled.node_types

    # For the phaseshifters we want the branch info of the phaseshifter in the nodal report
    assert np.array_equal(
        np.array(network_data.node_names[:number_of_phase_shifters]),
        np.array(network_data_filled.branch_names)[network_data_filled.phase_shift_mask],
    )
    assert np.array_equal(
        np.array(network_data.node_ids[:number_of_phase_shifters]),
        np.array(network_data_filled.branch_ids)[network_data_filled.phase_shift_mask],
    )
    assert network_data.node_types[:number_of_phase_shifters] == (["PSTNode"] * number_of_phase_shifters)

    # The phaseshifters are assumed to be in the end of the injection array
    assert network_data.injection_ids[:-number_of_phase_shifters] == network_data_filled.injection_ids
    assert network_data.injection_names[:-number_of_phase_shifters] == network_data_filled.injection_names
    assert np.array_equal(
        network_data.mw_injections[:, :-number_of_phase_shifters],
        network_data_filled.mw_injections,
    )

    assert np.array_equal(
        np.array(network_data.injection_names[-number_of_phase_shifters:]),
        np.array(network_data_filled.branch_names)[network_data_filled.phase_shift_mask],
    )
    assert np.array_equal(
        np.array(network_data.injection_ids[-number_of_phase_shifters:]),
        np.array(network_data_filled.branch_ids)[network_data_filled.phase_shift_mask],
    )
    # TODO why the -1*
    assert (
        np.all(
            network_data.mw_injections[:, -number_of_phase_shifters:]
            == network_data_filled.shift_angles[:, network_data_filled.phase_shift_mask]
        )
        * -1
    )

    assert network_data.controllable_pst_node_mask is not None
    assert network_data.controllable_pst_node_mask.sum() == network_data_filled.controllable_phase_shift_mask.sum()
    assert not np.any(network_data.controllable_pst_node_mask[number_of_phase_shifters:])
    controllable_pst_idx = np.flatnonzero(network_data.controllable_pst_node_mask)
    controllable_pst_idx_2 = np.flatnonzero(
        network_data_filled.controllable_phase_shift_mask[network_data_filled.phase_shift_mask]
    )
    assert np.array_equal(controllable_pst_idx, controllable_pst_idx_2)

    assert all(network_data.node_types[i] == "PSTNode" for i in controllable_pst_idx)


def test_add_bus_b_columns_to_ptdf_adds_correct_columns(
    network_data_with_ptdf: NetworkData,
) -> None:
    assert network_data_with_ptdf.ptdf is not None
    assert network_data_with_ptdf.ptdf_is_extended is False
    network_data = deepcopy(network_data_with_ptdf)
    network_data = add_bus_b_columns_to_ptdf(network_data)

    num_relevant_buses = network_data_with_ptdf.relevant_node_mask.sum()
    assert num_relevant_buses != 0

    # Make sure the right amount of columns was appended
    assert network_data.ptdf.shape[1] == network_data_with_ptdf.ptdf.shape[1] + num_relevant_buses
    # Make sure no branches were added
    assert network_data.ptdf.shape[0] == network_data_with_ptdf.ptdf.shape[0]

    # Make sure the added columns are identical to the original columns
    added_columns = network_data.ptdf[:, -num_relevant_buses:]
    original_columns = network_data_with_ptdf.ptdf[:, network_data_with_ptdf.relevant_node_mask]
    assert np.allclose(added_columns, original_columns)

    assert network_data.ptdf_is_extended is True
    assert network_data.relevant_node_mask.shape == (network_data.ptdf.shape[1],)
    assert np.sum(network_data.relevant_node_mask) == num_relevant_buses
    assert len(network_data.relevant_node_mask) == network_data.ptdf.shape[1]
    assert len(network_data.node_ids) == network_data.ptdf.shape[1]
    assert len(network_data.node_names) == network_data.ptdf.shape[1]
    assert len(network_data.node_types) == network_data.ptdf.shape[1]


def test_add_bus_b_columns_to_ptdf_raises_error_when_extended_again(
    network_data_with_ptdf: NetworkData,
) -> None:
    assert network_data_with_ptdf.ptdf is not None
    assert network_data_with_ptdf.ptdf_is_extended is False
    network_data = deepcopy(network_data_with_ptdf)
    network_data = add_bus_b_columns_to_ptdf(network_data)
    with pytest.raises(AssertionError):
        add_bus_b_columns_to_ptdf(network_data)


def test_compute_branch_topology_info(network_data: NetworkData) -> None:
    network_data = compute_branch_topology_info(network_data)
    relevant_node_ids = np.flatnonzero(network_data.relevant_node_mask)
    for node_entry, node_id in enumerate(relevant_node_ids):
        branches_at_node = network_data.branches_at_nodes[node_entry]
        num_branches_per_node = network_data.num_branches_per_node[node_entry]
        branch_direction = network_data.branch_direction[node_entry]

        expected_from_branches = np.flatnonzero(network_data.from_nodes == node_id)
        assert np.allclose(expected_from_branches, branches_at_node[branch_direction])

        expected_to_branches = np.flatnonzero(network_data.to_nodes == node_id)
        assert np.allclose(expected_to_branches, branches_at_node[~branch_direction])

        amount_of_branches = expected_from_branches.size + expected_to_branches.size
        assert amount_of_branches == num_branches_per_node


def test_compute_injection_topology_info(network_data: NetworkData) -> None:
    network_data_with_injection = compute_injection_topology_info(network_data)
    assert network_data_with_injection.active_injections is not None
    assert network_data_with_injection.num_injections_per_node is not None
    assert network_data_with_injection.active_injections is not None
    relevant_node_ids = np.flatnonzero(network_data.relevant_node_mask)
    injection_idx_at_node = group_by_node(network_data.injection_nodes, relevant_node_ids)
    for result, expectation in zip(network_data_with_injection.injection_idx_at_nodes, injection_idx_at_node):
        assert np.array_equal(result, expectation)

    num_injections_per_nodes = get_num_elements_per_node(injection_idx_at_node)

    assert np.array_equal(network_data_with_injection.num_injections_per_node, num_injections_per_nodes)

    mw_injections_at_nodes = get_mw_injections_at_nodes(injection_idx_at_node, network_data.mw_injections)
    active_injections = identify_inactive_injections(mw_injections_at_nodes)
    for result, expectation in zip(network_data_with_injection.active_injections, active_injections):
        assert np.array_equal(result, expectation)


def test_reduce_branch_dimension(
    network_data: NetworkData,
) -> None:
    network_data = add_nodal_injections_to_network_data(network_data)
    network_data = compute_ptdf_if_not_given(network_data)
    network_data = compute_psdf_if_not_given(network_data)
    network_data = compute_bridging_branches(network_data)
    network_data = reduce_node_dimension(network_data)
    network_data = combine_phaseshift_and_injection(network_data)
    reduced_branches = get_relevant_branches(
        from_node=network_data.from_nodes,
        to_node=network_data.to_nodes,
        relevant_node_mask=network_data.relevant_node_mask,
        monitored_branch_mask=network_data.monitored_branch_mask,
        outaged_branch_mask=network_data.outaged_branch_mask,
        multi_outage_mask=network_data.multi_outage_branch_mask,
        busbar_outage_branch_mask=get_busbar_map_adjacent_branches(network_data),
    )

    network_data_reduced = reduce_branch_dimension(network_data)
    assert np.array_equal(network_data_reduced.ptdf, network_data.ptdf[reduced_branches])
    assert np.array_equal(network_data_reduced.psdf, network_data.psdf[reduced_branches])
    assert np.array_equal(
        np.array(network_data_reduced.branch_ids),
        np.array(network_data.branch_ids)[reduced_branches],
    )
    assert np.array_equal(
        np.array(network_data_reduced.branch_names),
        np.array(network_data.branch_names)[reduced_branches],
    )
    assert np.array_equal(
        np.array(network_data_reduced.branch_types),
        np.array(network_data.branch_types)[reduced_branches],
    )
    assert np.array_equal(
        network_data_reduced.monitored_branch_mask,
        network_data.monitored_branch_mask[reduced_branches],
    )
    assert np.array_equal(
        network_data_reduced.outaged_branch_mask,
        network_data.outaged_branch_mask[reduced_branches],
    )
    assert np.array_equal(
        network_data_reduced.disconnectable_branch_mask,
        network_data.disconnectable_branch_mask[reduced_branches],
    )
    assert np.array_equal(
        network_data_reduced.ac_dc_mismatch,
        network_data.ac_dc_mismatch[:, reduced_branches],
    )
    assert np.array_equal(
        network_data_reduced.max_mw_flows,
        network_data.max_mw_flows[:, reduced_branches],
    )
    assert np.array_equal(
        network_data_reduced.from_nodes,
        network_data.from_nodes[reduced_branches],
    )
    assert np.array_equal(network_data_reduced.to_nodes, network_data.to_nodes[reduced_branches])
    assert np.array_equal(
        network_data_reduced.susceptances,
        network_data.susceptances[reduced_branches],
    )
    assert np.array_equal(
        network_data_reduced.bridging_branch_mask,
        network_data.bridging_branch_mask[reduced_branches],
    )
    assert np.array_equal(
        network_data_reduced.multi_outage_branch_mask,
        network_data.multi_outage_branch_mask[:, reduced_branches],
    )
    assert np.array_equal(
        network_data_reduced.controllable_phase_shift_mask,
        network_data.controllable_phase_shift_mask[reduced_branches],
    )


def test_reduce_branch_dimension_after_topology_info(
    network_data_filled: NetworkData,
) -> None:
    network_data = compute_branch_topology_info(network_data_filled)
    with pytest.raises(AssertionError):
        reduce_branch_dimension(network_data)


def test_filter_disconnectable_branches_nminus2(
    network_data_filled: NetworkData,
) -> None:
    network_data_new = filter_disconnectable_branches_nminus2(network_data_filled)
    assert network_data_new.disconnectable_branch_mask.shape == network_data_filled.disconnectable_branch_mask.shape
    assert np.sum(network_data_new.disconnectable_branch_mask) <= np.sum(network_data_filled.disconnectable_branch_mask)
    assert np.all(network_data_new.disconnectable_branch_mask <= network_data_filled.disconnectable_branch_mask)

    disc_branch = np.flatnonzero(network_data_new.disconnectable_branch_mask)
    assert len(disc_branch) > 0
    mask = find_n_minus_2_safe_branches(
        network_data_new.from_nodes,
        network_data_new.to_nodes,
        len(network_data_new.branch_ids),
        len(network_data_new.node_ids),
        disc_branch,
    )

    assert np.all(mask)
    network_data_parallel = filter_disconnectable_branches_nminus2(network_data_filled, n_processes=2)
    assert np.array_equal(network_data_new.disconnectable_branch_mask, network_data_parallel.disconnectable_branch_mask)


def test_compute_electrical_actions(network_data_filled: NetworkData) -> None:
    network_data = filter_relevant_nodes_branch_count(network_data_filled)
    network_data = compute_branch_topology_info(network_data)
    network_data = filter_inactive_injections(network_data)
    network_data = compute_injection_topology_info(network_data)
    network_data = simplify_asset_topology(network_data)
    network_data = compute_separation_set_for_stations(network_data)
    network_data = compute_electrical_actions(network_data)
    network_data = enumerate_station_realisations(network_data)
    network_data = compute_injection_actions(network_data)

    assert network_data.branch_action_set is not None
    assert len(network_data.branch_action_set) == sum(network_data.relevant_node_mask)
    assert network_data.injection_action_set is not None
    assert len(network_data.injection_action_set) == sum(network_data.relevant_node_mask)
    assert network_data.branch_action_set_switching_distance is not None
    assert len(network_data.branch_action_set_switching_distance) == sum(network_data.relevant_node_mask)
    for sub, (inj_set, branch_set, sw_dist) in enumerate(
        zip(
            network_data.injection_action_set,
            network_data.branch_action_set,
            network_data.branch_action_set_switching_distance,
        )
    ):
        assert inj_set.shape[0] == branch_set.shape[0]
        assert sw_dist.shape[0] == branch_set.shape[0]
        assert inj_set.shape[1] == network_data.num_injections_per_node[sub]
        assert branch_set.shape[1] == network_data.num_branches_per_node[sub]
        assert inj_set.ndim == 2
        assert sw_dist.ndim == 1
        assert branch_set.ndim == 2


def test_enumerate_station_realisations(
    network_data_filled: NetworkData,
) -> None:
    network_data = filter_relevant_nodes_branch_count(network_data_filled)
    network_data = compute_branch_topology_info(network_data)
    network_data = filter_inactive_injections(network_data)
    network_data = compute_injection_topology_info(network_data)
    network_data = simplify_asset_topology(network_data)
    network_data = compute_separation_set_for_stations(network_data)
    network_data = compute_electrical_actions(network_data)
    network_data = enumerate_station_realisations(network_data)
    assert network_data.realised_stations is not None
    assert len(network_data.realised_stations) == network_data.relevant_node_mask.sum()

    for stations, br_act in zip(network_data.realised_stations, network_data.branch_action_set, strict=True):
        assert br_act.shape[0] == len(stations)
        for station in stations:
            Station.model_validate(station)


def test_enumerate_station_realisations_no_coupler(
    network_data_filled: NetworkData,
) -> None:
    stations = [station.model_copy(update={"couplers": []}) for station in network_data_filled.asset_topology.stations]
    network_data = replace(
        network_data_filled,
        asset_topology=network_data_filled.asset_topology.model_copy(update={"stations": stations}),
    )
    network_data = filter_relevant_nodes_branch_count(network_data)
    network_data = compute_branch_topology_info(network_data)
    network_data = filter_inactive_injections(network_data)
    network_data = compute_injection_topology_info(network_data)
    with pytest.raises(ValueError):
        network_data = simplify_asset_topology(network_data)


def test_simplify_asset_topology(
    network_data_filled: NetworkData,
) -> None:
    network_data = compute_branch_topology_info(network_data_filled)
    network_data = compute_injection_topology_info(network_data)
    network_data = simplify_asset_topology(network_data)
    assert network_data.simplified_asset_topology is not None
    assert len(network_data.simplified_asset_topology.stations) == network_data.relevant_node_mask.sum()
    for rel_node_index, (rel_node_id, station) in enumerate(
        zip(network_data.relevant_nodes, network_data.simplified_asset_topology.stations, strict=True)
    ):
        assert station.grid_model_id == network_data.node_ids[rel_node_id]
        Station.model_validate(station)
        branch_ids = [network_data.branch_ids[i] for i in network_data.branches_at_nodes[rel_node_index]]
        inj_ids = [network_data.injection_ids[i] for i in network_data.injection_idx_at_nodes[rel_node_index]]
        asset_ids = [a.grid_model_id for a in station.assets]
        assert branch_ids == asset_ids[: len(branch_ids)]
        assert inj_ids == asset_ids[len(branch_ids) :]


def test_exclude_bridges_from_outage_masks(
    network_data_filled: NetworkData,
) -> None:
    network_data = exclude_bridges_from_outage_masks(network_data_filled)
    assert np.array_equal(
        network_data.outaged_branch_mask,
        network_data_filled.outaged_branch_mask & ~network_data_filled.bridging_branch_mask,
    )
    assert np.array_equal(
        network_data.multi_outage_branch_mask,
        network_data_filled.multi_outage_branch_mask & ~network_data_filled.bridging_branch_mask,
    )


def test_convert_multi_outages(network_data_filled: NetworkData) -> None:
    network_data = convert_multi_outages(network_data_filled)
    assert network_data.multi_outage_branch_mask.shape == network_data_filled.multi_outage_branch_mask.shape
    assert network_data.multi_outage_node_mask.shape == network_data_filled.multi_outage_node_mask.shape
    assert np.sum(network_data.multi_outage_branch_mask) == np.sum(network_data_filled.multi_outage_branch_mask)
    assert np.sum(network_data.multi_outage_node_mask) == np.sum(network_data_filled.multi_outage_node_mask)

    index = 0
    for branch_indices in network_data.split_multi_outage_branches:
        assert not np.any(branch_indices == -1)
        for outage in branch_indices:
            assert np.all(network_data.multi_outage_branch_mask[index, outage])
            assert np.sum(network_data.multi_outage_branch_mask[index]) == outage.shape[0] + 1
            index += 1

    index = 0
    for node_indices in network_data.split_multi_outage_nodes:
        for outage in node_indices:
            assert np.all(network_data.multi_outage_node_mask[index, outage[outage != -1]])
            assert np.sum(network_data.multi_outage_node_mask[index]) == np.sum(outage != -1)
            index += 1


def test_filter_inactive_injections(network_data_filled: NetworkData) -> None:
    network_data = filter_inactive_injections(network_data_filled)
    assert np.all(np.any(np.abs(network_data.mw_injections) > 0, axis=0))

    n_inj = network_data.mw_injections.shape[1]
    assert n_inj == len(network_data.injection_ids)
    assert n_inj == len(network_data.injection_names)
    assert n_inj == len(network_data.injection_types)
    assert n_inj == len(network_data.outaged_injection_mask)
    assert n_inj <= len(network_data_filled.injection_ids)


def test_convert_multi_outages_no_outages(network_data_filled: NetworkData) -> None:
    network_data = replace(
        network_data_filled,
        multi_outage_branch_mask=np.zeros_like(network_data_filled.multi_outage_branch_mask),
        multi_outage_node_mask=np.zeros_like(network_data_filled.multi_outage_node_mask),
    )

    assert network_data.split_multi_outage_branches is None
    assert network_data.split_multi_outage_nodes is None

    network_data = convert_multi_outages(network_data)
    assert network_data.split_multi_outage_branches == []
    assert network_data.split_multi_outage_nodes == []

    # Should be the same as with size zero
    network_data = replace(
        network_data_filled,
        multi_outage_branch_mask=np.zeros((0, network_data_filled.multi_outage_branch_mask.shape[1]), dtype=bool),
        multi_outage_node_mask=np.zeros((0, network_data_filled.multi_outage_node_mask.shape[1]), dtype=bool),
    )

    assert network_data.split_multi_outage_branches is None
    assert network_data.split_multi_outage_nodes is None

    network_data = convert_multi_outages(network_data)
    assert network_data.split_multi_outage_branches == []
    assert network_data.split_multi_outage_nodes == []

    # When we leave the nodes in place, we should get one empty branch outage
    network_data = replace(
        network_data_filled,
        multi_outage_branch_mask=np.zeros_like(network_data_filled.multi_outage_branch_mask),
    )

    network_data = convert_multi_outages(network_data)
    assert len(network_data.split_multi_outage_branches) == 1
    assert network_data.split_multi_outage_branches[0].size == 0
    assert network_data.split_multi_outage_branches[0].shape[0] == network_data.split_multi_outage_nodes[0].shape[0]
    assert network_data.split_multi_outage_nodes[0].size > 0

    # When we leave the branches in place, we should get one empty node outage
    network_data = replace(
        network_data_filled,
        multi_outage_node_mask=np.zeros_like(network_data_filled.multi_outage_node_mask),
    )
    network_data = convert_multi_outages(network_data)
    assert len(network_data.split_multi_outage_nodes) == len(network_data.split_multi_outage_branches)
    assert network_data.split_multi_outage_nodes[0].size == 0
    assert network_data.split_multi_outage_nodes[0].shape[0] == network_data.split_multi_outage_branches[0].shape[0]
    assert network_data.split_multi_outage_branches[0].size > 0


def test_process_injection_outages(network_data: NetworkData) -> None:
    np.random.seed(1)
    network_data = replace(
        network_data,
        outaged_injection_mask=np.random.randint(0, 2, network_data.mw_injections.shape[1]).astype(bool),
    )
    network_data = compute_injection_topology_info(network_data)

    n_inj_out_rel = np.sum(
        network_data.relevant_node_mask[network_data.injection_nodes[network_data.outaged_injection_mask]]
    )
    n_inj_out_nonrel = np.sum(
        ~network_data.relevant_node_mask[network_data.injection_nodes[network_data.outaged_injection_mask]]
    )

    assert n_inj_out_rel + n_inj_out_nonrel == np.sum(network_data.outaged_injection_mask)

    network_data_new = process_injection_outages(network_data)
    assert network_data_new.nonrel_io_deltap.shape == (
        network_data.mw_injections.shape[0],  # n_timestep
        n_inj_out_nonrel,
    )
    assert network_data_new.nonrel_io_node.shape == (n_inj_out_nonrel,)
    assert network_data_new.nonrel_io_global_inj_index.shape == (n_inj_out_nonrel,)
    assert network_data_new.rel_io_local_inj_index.shape == (n_inj_out_rel,)
    assert network_data_new.rel_io_sub.shape == (n_inj_out_rel,)
    assert network_data_new.rel_io_global_inj_index.shape == (n_inj_out_rel,)

    for sub, idx in zip(network_data_new.rel_io_sub, network_data_new.rel_io_local_inj_index):
        local_inj = network_data_new.injection_idx_at_nodes[sub]
        assert idx < len(local_inj)
        assert network_data.outaged_injection_mask[local_inj[idx]]


def test_filter_relevant_nodes_branch_count(network_data: NetworkData) -> None:
    old_relevant_nodes = network_data.relevant_node_mask
    old_indices = np.flatnonzero(old_relevant_nodes)
    old_cross_coupler_limits = {
        network_data.node_ids[old_indices[i]]: limit for (i, limit) in enumerate(network_data.cross_coupler_limits)
    }

    network_data = compute_bridging_branches(network_data)
    network_data = filter_relevant_nodes_branch_count(network_data)
    assert np.all(old_relevant_nodes[network_data.relevant_node_mask])
    for node_id in np.flatnonzero(network_data.relevant_node_mask):
        assert np.sum(network_data.from_nodes == node_id) + np.sum(network_data.to_nodes == node_id) >= 4

    new_indices = np.flatnonzero(network_data.relevant_node_mask)
    new_cross_coupler_limits = {
        network_data.node_ids[new_indices[i]]: limit for (i, limit) in enumerate(network_data.cross_coupler_limits)
    }

    for key in new_cross_coupler_limits.keys():
        assert old_cross_coupler_limits[key] == new_cross_coupler_limits[key]


def test_filter_relevant_nodes_no_asset_station(network_data: NetworkData) -> None:
    # Remove a few stations from the asset topology
    network_data = replace(
        network_data,
        asset_topology=network_data.asset_topology.model_copy(
            update={
                "stations": network_data.asset_topology.stations[:2]  # Keep only the first two stations
            }
        ),
    )
    # Compute the relevant nodes based on the new asset topology
    network_data = filter_relevant_nodes_no_asset_station(network_data)
    assert network_data.relevant_node_mask.sum() == 2  # Only the first two nodes should remain relevant


def test_reduce_node_dimension(network_data_filled):
    network_data_reduced = reduce_node_dimension(network_data_filled)
    # Check for consistent shapes
    assert network_data_reduced.ptdf.shape[1] == network_data_reduced.relevant_node_mask.shape[0]
    assert network_data_reduced.ptdf.shape[1] == network_data_reduced.nodal_injection.shape[1]
    assert network_data_reduced.ptdf.shape[1] == network_data_reduced.multi_outage_node_mask.shape[1]
    assert network_data_reduced.ptdf.shape[1] == len(network_data_reduced.node_ids)
    assert network_data_reduced.ptdf.shape[1] == len(network_data_reduced.node_names)
    assert network_data_reduced.ptdf.shape[1] == len(network_data_reduced.node_types)

    # Check loadflows match
    lf_old = network_data_filled.ptdf @ network_data_filled.nodal_injection[0]
    lf_new = network_data_reduced.ptdf @ network_data_reduced.nodal_injection[0]
    assert np.allclose(lf_old, lf_new)

    # Check relevant nodes
    new_rel_nodes = network_data_reduced.relevant_node_mask
    old_rel_nodes = network_data_filled.relevant_node_mask
    assert np.all(network_data_reduced.ptdf[:, new_rel_nodes] == network_data_filled.ptdf[:, old_rel_nodes])
    assert np.all(
        network_data_reduced.nodal_injection[:, new_rel_nodes] == network_data_filled.nodal_injection[:, old_rel_nodes]
    )
    assert np.all(
        np.array(network_data_reduced.node_ids)[new_rel_nodes] == np.array(network_data_filled.node_ids)[old_rel_nodes]
    )
    assert np.all(
        np.array(network_data_reduced.node_names)[new_rel_nodes] == np.array(network_data_filled.node_names)[old_rel_nodes]
    )
    assert np.all(
        np.array(network_data_reduced.node_types)[new_rel_nodes] == np.array(network_data_filled.node_types)[old_rel_nodes]
    )

    # Check that all links to nodes are updated
    assert network_data_reduced.ptdf.shape[1] >= network_data_reduced.injection_nodes.max()
    assert network_data_reduced.ptdf.shape[1] >= network_data_reduced.from_nodes.max()
    assert network_data_reduced.ptdf.shape[1] >= network_data_reduced.to_nodes.max()
    assert network_data_reduced.ptdf.shape[1] >= network_data_reduced.slack
    assert network_data_reduced.ptdf.shape[1] >= new_rel_nodes.shape[0]

    # Check last node
    assert network_data_reduced.node_ids[-1] == "REDUCED_NODE"
    assert network_data_reduced.node_names[-1] == "REDUCED_NODE"
    assert network_data_reduced.node_types[-1] == "REDUCED_NODE"
    assert new_rel_nodes[-1] == False
    assert network_data_reduced.nodal_injection[0, -1] == 1.0
    assert network_data_reduced.multi_outage_node_mask[:, -1].sum() == 0
    reduced_node_ids = np.array(network_data_reduced.node_ids)
    old_node_ids = np.array(network_data_filled.node_ids)
    matching_ids = reduced_node_ids[network_data_reduced.from_nodes] == old_node_ids[network_data_filled.from_nodes]
    assert np.all(reduced_node_ids[network_data_reduced.from_nodes][~matching_ids] == "REDUCED_NODE")
    assert matching_ids.sum() != 0

    reduced_node_names = np.array(network_data_reduced.node_names)
    old_node_names = np.array(network_data_filled.node_names)
    matching_names = reduced_node_names[network_data_reduced.from_nodes] == old_node_names[network_data_filled.from_nodes]
    assert np.all(reduced_node_names[network_data_reduced.from_nodes][~matching_names] == "REDUCED_NODE")
    assert matching_names.sum() != 0

    reduced_node_types = np.array(network_data_reduced.node_types)
    old_node_types = np.array(network_data_filled.node_types)
    matching_types = reduced_node_types[network_data_reduced.from_nodes] == old_node_types[network_data_filled.from_nodes]
    assert np.all(reduced_node_types[network_data_reduced.from_nodes][~matching_types] == "REDUCED_NODE")
    assert matching_types.sum() != 0


def test_preprocess(data_folder: str, tmp_path: str) -> None:
    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir)
    network_data = preprocess(backend)
    validate_network_data(network_data)

    static_information = convert_to_jax(network_data)
    save_static_information(Path(tmp_path) / "test_static_information.hdf5", static_information)
    assert os.path.exists(Path(tmp_path) / "test_static_information.hdf5")

    static_information = load_static_information(Path(tmp_path) / "test_static_information.hdf5")
    validate_static_information(static_information)

    for rel_node_index, (rel_node_id, station) in enumerate(
        zip(network_data.relevant_nodes, network_data.simplified_asset_topology.stations, strict=True)
    ):
        assert station.grid_model_id == network_data.node_ids[rel_node_id]
        Station.model_validate(station)
        branch_ids = [network_data.branch_ids[i] for i in network_data.branches_at_nodes[rel_node_index]]
        inj_ids = [network_data.injection_ids[i] for i in network_data.injection_idx_at_nodes[rel_node_index]]
        asset_ids = [a.grid_model_id for a in station.assets]
        assert branch_ids == asset_ids[: len(branch_ids)]
        assert inj_ids == asset_ids[len(branch_ids) :]


def test_loadflows_match(data_folder: str) -> None:
    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir)
    network_data = preprocess(backend)

    lf = network_data.ptdf @ network_data.nodal_injection[0]

    pp.rundcpp(backend.net)
    lf_ref = get_pandapower_branch_loadflow_results_sequence(
        backend.net, network_data.branch_types, table_ids(network_data.branch_ids), measurement="active"
    )

    assert np.allclose(lf, lf_ref, rtol=1e-3, atol=1e-3)
    assert lf.shape == lf_ref.shape

    pp.runpp(backend.net)
    lf_ref_ac = get_pandapower_branch_loadflow_results_sequence(
        backend.net, network_data.branch_types, table_ids(network_data.branch_ids), measurement="active"
    )
    lf_ac = lf + network_data.ac_dc_mismatch[0]

    assert np.allclose(lf_ac, lf_ref_ac, rtol=1e-3, atol=1e-3)


def test_preprocess_case30(case30_data_folder: str) -> None:
    filesystem_dir = DirFileSystem(str(case30_data_folder))
    backend = PandaPowerBackend(filesystem_dir)
    network_data = preprocess(backend)
    n_nodes = len(network_data.node_ids)
    n_branch = len(network_data.branch_ids)

    assert network_data.controllable_phase_shift_mask.shape == (n_branch,)
    assert not np.any(network_data.controllable_phase_shift_mask & ~network_data.phase_shift_mask)
    assert network_data.controllable_pst_node_mask.shape == (n_nodes,)
    assert np.sum(network_data.controllable_phase_shift_mask) == np.sum(network_data.controllable_pst_node_mask)
    assert len(network_data.phase_shift_taps) == network_data.controllable_phase_shift_mask.sum()

    assert network_data.controllable_phase_shift_mask.sum() == 3


def test_preprocess_logging(data_folder: str) -> None:
    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir)

    logs = []

    def log_function(stage: PreprocessStage, message: Optional[str]) -> None:
        logs.append((stage, message))
        assert stage in get_args(PreprocessStage)

    preprocess(backend, logging_fn=log_function)
    assert logs
    assert logs[0][0] == "preprocess_started"
    assert logs[-1][0] == "preprocess_done"
