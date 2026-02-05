# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from pathlib import Path

import numpy as np
import pandapower as pp
from pandapower.pypower.idx_brch import PF
from pandapower.pypower.idx_bus import GS, PD
from toop_engine_dc_solver.preprocess.helpers.ptdf import get_susceptance_matrices
from toop_engine_dc_solver.preprocess.network_data import NetworkData
from toop_engine_dc_solver.preprocess.preprocess import (
    combine_phaseshift_and_injection,
)
from toop_engine_grid_helpers.pandapower.pandapower_helpers import (
    get_pandapower_loadflow_results_in_ppc,
)
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS


def test_pandapower_vs_ppci_loadflow(loaded_net: pp.pandapowerNet) -> None:
    pp_branch_loads = get_pandapower_loadflow_results_in_ppc(loaded_net)[loaded_net._ppc["internal"]["branch_is"]]
    ppci_loads = loaded_net._ppc["internal"]["branch"][:, PF].real
    assert np.allclose(abs(pp_branch_loads), abs(ppci_loads))


def test_nodal_injection_with_nodal_susceptance_matrix(
    loaded_net: pp.pandapowerNet, network_data_filled: NetworkData
) -> None:
    expected_nodal_injection = np.concatenate(
        [
            loaded_net.res_bus.p_mw.values,
            np.zeros(len(loaded_net.trafo3w) + len(loaded_net.xward), dtype=float),
        ]
    )
    if len(loaded_net.dcline) > 0:
        # Pandapower creates gens from dc lines. When summing up injections for res_bus it adds the gen-powers and subtracts the dcline powers leading to a net-zero nodal injection for the dcline
        np.add.at(
            expected_nodal_injection,
            loaded_net.dcline.from_bus.values.astype(int),
            loaded_net.res_dcline.p_from_mw.values,
        )
        np.add.at(
            expected_nodal_injection,
            loaded_net.dcline.to_bus.values.astype(int),
            loaded_net.res_dcline.p_to_mw.values,
        )
    num_branches = len(network_data_filled.branch_ids)
    num_nodes = len(network_data_filled.node_ids)
    b_bus, _ = get_susceptance_matrices(
        network_data_filled.from_nodes,
        network_data_filled.to_nodes,
        network_data_filled.susceptances,
        num_branches,
        num_nodes,
    )
    voltage_angle = loaded_net._ppc["internal"]["V"].real
    pst_correction = loaded_net._ppc["internal"]["Pbusinj"].real
    nodal_injection = -1 * (b_bus @ voltage_angle + pst_correction)

    # Exclude slack
    expected_nodal_injection = np.delete(expected_nodal_injection, [network_data_filled.slack])
    nodal_injection = np.delete(nodal_injection, [network_data_filled.slack])

    # Debug
    # diff = nodal_injection - expected_nodal_injection
    # big_diff = np.where(abs(diff) >= 1)[0]
    # network_data_filled.susceptances[big_diff]
    # b_bus[big_diff].argmax()
    # from pandapower.toolbox import get_connected_elements_dict
    # elements = get_connected_elements_dict(loaded_net, buses=big_diff[0])
    assert np.allclose(nodal_injection.real, expected_nodal_injection, atol=1e-6)


def test_powerflow_with_branch_susceptance_matrix(loaded_net: pp.pandapowerNet, network_data_filled: NetworkData) -> None:
    expected_powerflow = get_pandapower_loadflow_results_in_ppc(loaded_net)[loaded_net._ppc["internal"]["branch_is"]]
    # susceptance_matrix
    num_branches = len(network_data_filled.branch_ids)
    num_nodes = len(network_data_filled.node_ids)
    _, b_branch = get_susceptance_matrices(
        network_data_filled.from_nodes,
        network_data_filled.to_nodes,
        network_data_filled.susceptances,
        num_branches,
        num_nodes,
    )

    voltage_angle = loaded_net._ppc["internal"]["V"]
    powershift_from_phaseshift = loaded_net._ppc["internal"]["Pfinj"]
    powerflow = b_branch @ voltage_angle + powershift_from_phaseshift
    assert np.allclose(np.abs(powerflow), np.abs(expected_powerflow))


def test_voltage_angle_with_inverted_nodal_susceptance(
    loaded_net: pp.pandapowerNet, network_data_filled: NetworkData
) -> None:
    expected_voltage_angle = loaded_net._ppc["internal"]["V"]
    # susceptance matrix
    num_branches = len(network_data_filled.branch_ids)
    num_nodes = len(network_data_filled.node_ids)
    b_bus, _ = get_susceptance_matrices(
        network_data_filled.from_nodes,
        network_data_filled.to_nodes,
        network_data_filled.susceptances,
        num_branches,
        num_nodes,
    )

    # slack info
    slack = network_data_filled.slack
    non_slack = np.ones(num_nodes, dtype=bool)
    non_slack[slack] = False

    # Invert b_bus
    b_bus_inv_no_slack = b_bus.todense().real[np.ix_(non_slack, non_slack)].I
    b_bus_inv = np.insert(b_bus_inv_no_slack, slack, 0.0, axis=1)
    b_bus_inv = np.insert(b_bus_inv, slack, 0.0, axis=0)

    # Nodal injection with phaseshift correction
    nodal_injections = network_data_filled.nodal_injection[0]  # Only first timestep
    phaseshift_correction = loaded_net._ppc["internal"]["Pbusinj"].real
    nodal_injections_adjusted_for_pst = -nodal_injections - phaseshift_correction
    voltage_angle = b_bus_inv @ nodal_injections_adjusted_for_pst.T
    assert np.allclose(voltage_angle, expected_voltage_angle)


def test_ptdf_computation_from_susceptance_matrices(
    network_data_filled: NetworkData,
) -> None:
    # susceptance matrix
    num_branches = len(network_data_filled.branch_ids)
    num_nodes = len(network_data_filled.node_ids)
    b_bus, b_branch = get_susceptance_matrices(
        network_data_filled.from_nodes,
        network_data_filled.to_nodes,
        network_data_filled.susceptances,
        num_branches,
        num_nodes,
    )
    # slack info
    slack = network_data_filled.slack
    non_slack = np.ones(num_nodes, dtype=bool)
    non_slack[slack] = False

    # Invert b_bus
    b_bus_inv_no_slack = b_bus.todense().real[np.ix_(non_slack, non_slack)].I
    b_bus_inv = np.insert(b_bus_inv_no_slack, slack, 0.0, axis=1)
    b_bus_inv = np.insert(b_bus_inv, slack, 0.0, axis=0)
    ptdf = b_branch @ b_bus_inv

    expected_ptdf = network_data_filled.ptdf
    assert np.allclose(ptdf, expected_ptdf, atol=1e-6)


def test_psdf_replaces_phaseshift_influence(loaded_net: pp.pandapowerNet, network_data_filled: NetworkData) -> None:
    powerflow_phaseshift_correction = loaded_net._ppc["internal"]["Pfinj"]
    injection_phaseshift_correction = loaded_net._ppc["internal"]["Pbusinj"]
    nodal_injection = network_data_filled.nodal_injection[0]  # Only one timestep
    ptdf = network_data_filled.ptdf

    expected_powerflow = get_pandapower_loadflow_results_in_ppc(loaded_net)[loaded_net._ppc["internal"]["branch_is"]]
    powerflow = ptdf @ nodal_injection.T + ptdf @ injection_phaseshift_correction - powerflow_phaseshift_correction

    assert np.allclose(np.abs(expected_powerflow), np.abs(powerflow.real), atol=1e-6)

    psdf = network_data_filled.psdf
    phaseshift_values = network_data_filled.shift_angles[
        0, network_data_filled.phase_shift_mask
    ]  # Only one timestep. only phaseshifter

    expected_phaseshift_influence = +ptdf @ injection_phaseshift_correction - powerflow_phaseshift_correction
    phase_shift_influence = psdf @ phaseshift_values.T
    # Debug
    # diff = expected_phaseshift_influence - phase_shift_influence
    # big_diff = np.where(abs(diff) >= 40)[0]
    # np.array(network_data_filled.branch_types)[big_diff]
    # np.array(network_data_filled.branch_ids)[big_diff]
    # network_data_filled.susceptances[big_diff]
    # b_bus[big_diff].argmax()
    # from pandapower.toolbox import get_connected_elements_dict
    # elements = get_connected_elements_dict(loaded_net, buses=big_diff[0])
    assert np.allclose(expected_phaseshift_influence.real, phase_shift_influence)


def test_pandapower_vs_ppci_nodal_injections(data_folder: Path, network_data: NetworkData) -> None:
    grid_file_path = data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    net = pp.from_json(grid_file_path)
    pp.rundcpp(net)
    pp_nodal_injections = np.concatenate(
        [
            net.res_bus.p_mw.values,
            np.zeros(len(net.trafo3w) + len(net.xward), dtype=float),
        ]
    )
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
    # Remove out of service branches, since they are not in ppci
    ppci_injection_load = net._ppc["internal"]["bus"][:, PD]
    ppci_injection_shunt = net._ppc["internal"]["bus"][:, GS]
    # ppci innjections without gen
    ppci_injection = ppci_injection_load + ppci_injection_shunt / network_data.base_mva
    # Add generator injection
    np.add.at(
        ppci_injection,
        net._ppc["internal"]["gen"][:, 0].astype(int),
        -1 * net._ppc["internal"]["gen"][:, 1],
    )
    # Exclude slack
    pp_nodal_injections = np.delete(pp_nodal_injections, [network_data.slack])
    ppci_injection = np.delete(ppci_injection, [network_data.slack])
    assert np.allclose(pp_nodal_injections, ppci_injection.real)


def test_combined_ptdfpsdf_returns_correct_powerflow(
    loaded_net: pp.pandapowerNet,
    network_data_filled: NetworkData,
) -> None:
    network_data = combine_phaseshift_and_injection(network_data_filled)
    shift_nodal_injection = network_data.nodal_injection[0]  # Only one timestep
    psdf_ptdf = network_data.ptdf
    powerflow = psdf_ptdf @ shift_nodal_injection

    expected_powerflow = get_pandapower_loadflow_results_in_ppc(loaded_net)[loaded_net._ppc["internal"]["branch_is"]]

    assert np.allclose(np.abs(expected_powerflow), np.abs(powerflow.real), atol=1e-6)
