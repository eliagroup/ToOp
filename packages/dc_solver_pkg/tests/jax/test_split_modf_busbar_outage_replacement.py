# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Demonstrate split-MODF as a physical busbar-outage replacement."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from toop_engine_dc_solver.jax.multi_outages import compute_multi_outage
from toop_engine_dc_solver.jax.split_modf_main_grid import compute_split_modf_main_grid
from toop_engine_dc_solver.postprocess.postprocess_powsybl import PowsyblRunner
from toop_engine_dc_solver.preprocess.network_data import NetworkData, extract_action_set, load_lf_params
from toop_engine_dc_solver.preprocess.preprocess_bb_outage import get_busbar_index
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.loadflow_result_helpers_polars import extract_solver_matrices_polars
from toop_engine_interfaces.nminus1_definition import MonitoredElement, Nminus1Definition, load_nminus1_definition

TOL = 1e-9


def _run_busbar_reference(
    data_folder: Path,
    network_data: NetworkData,
    busbar_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the physical busbar contingency and return ToOp-sign flows."""
    full_definition = load_nminus1_definition(data_folder / PREPROCESSING_PATHS["nminus1_definition_file_path"])
    busbar_contingency = next(contingency for contingency in full_definition.contingencies if contingency.id == busbar_id)
    monitored_elements = [
        MonitoredElement(id=branch_id, name=branch_name, type=branch_type, kind="branch")
        for branch_id, branch_name, branch_type in zip(
            network_data.branch_ids,
            network_data.branch_names,
            network_data.branch_types,
            strict=True,
        )
    ]
    definition = Nminus1Definition(
        monitored_elements=monitored_elements,
        contingencies=[full_definition.contingencies[0], busbar_contingency],
    )

    runner = PowsyblRunner(lf_params=load_lf_params(data_folder / PREPROCESSING_PATHS["loadflow_parameters_file_path"]))
    runner.load_base_grid(data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    runner.store_action_set(extract_action_set(network_data))
    runner.store_nminus1_definition(definition)
    results = runner.run_dc_loadflow([], [])
    n_0_flow, n_1_flow, success = extract_solver_matrices_polars(results, definition, 0)

    assert success.shape == (1,)
    assert success[0]
    return -n_0_flow, -n_1_flow[0]


def _get_physical_busbar_outage_inputs(
    network_data: NetworkData,
    busbar_id: str,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Get all directly disconnected branches and injections for a busbar."""
    assert network_data.asset_topology is not None
    assert network_data.mw_injections is not None

    station = next(
        station
        for station in network_data.asset_topology.stations
        if any(busbar.grid_model_id == busbar_id for busbar in station.busbars)
    )
    busbar_index = get_busbar_index(station, busbar_id)
    branch_assets = station.get_connected_assets(busbar_index, asset_scope="branch")
    injection_assets = station.get_connected_assets(busbar_index, asset_scope="injection")

    branch_index_by_id = {branch_id: index for index, branch_id in enumerate(network_data.branch_ids)}
    injection_index_by_id = {injection_id: index for index, injection_id in enumerate(network_data.injection_ids)}
    branch_outages = np.asarray(
        [branch_index_by_id[asset.grid_model_id] for asset in branch_assets],
        dtype=np.int32,
    )
    injection_indices = [
        injection_index_by_id[asset.grid_model_id]
        for asset in injection_assets
        if asset.grid_model_id in injection_index_by_id
    ]
    injection_to_remove = network_data.mw_injections[:, injection_indices].sum(axis=1)

    # These requested busbars belong to non-relevant unsplit stations, so their
    # bus-branch bus maps directly to the solver node. Legacy skeleton-branch
    # and detached-subtree preprocessing is intentionally not used.
    bus_branch_bus_id = station.busbars[busbar_index].bus_branch_bus_id
    node_index = network_data.node_ids.index(bus_branch_bus_id)
    return branch_outages, int(node_index), injection_to_remove


@pytest.mark.parametrize(
    ("busbar_id", "ordinary_modf_succeeds"),
    [
        pytest.param("BBS5_1", False, id="BBS5_1-islanding"),
        pytest.param("BBS4_2", True, id="BBS4_2-regular"),
        pytest.param("BBS4_1", True, id="BBS4_1-islanding"),
    ],
)
def test_split_modf_replaces_busbar_outage_path(
    test_grid_folder_path: Path,
    network_data_test_grid: NetworkData,
    busbar_id: str,
    ordinary_modf_succeeds: bool,
) -> None:
    """Match a physical busbar outage without skeleton-branch retry logic."""
    network_data = network_data_test_grid
    assert network_data.ptdf is not None
    assert network_data.nodal_injection is not None

    branch_outages, node_index, injection_to_remove = _get_physical_busbar_outage_inputs(
        network_data,
        busbar_id,
    )
    assert branch_outages.size > 0
    reference_n_0_flow, reference_n_1_flow = _run_busbar_reference(
        test_grid_folder_path,
        network_data,
        busbar_id,
    )
    assert reference_n_0_flow.shape == (len(network_data.branch_ids),)
    assert reference_n_1_flow.shape == (len(network_data.branch_ids),)

    physical_nodal_injections = network_data.nodal_injection.copy()
    physical_nodal_injections[:, np.asarray(network_data.node_types) == "PSTNode"] = 0.0
    physical_nodal_injections[:, node_index] -= injection_to_remove

    # Preserve any additive base-flow contribution while removing only the
    # injections physically connected to the outaged busbar.
    injection_delta = np.zeros_like(physical_nodal_injections)
    injection_delta[:, node_index] = -injection_to_remove
    n_0_flow_after_injection_outage = reference_n_0_flow + injection_delta @ network_data.ptdf.T
    branches_monitored = jnp.arange(len(network_data.branch_ids), dtype=jnp.int32)

    ordinary_flow, ordinary_success = compute_multi_outage(
        ptdf=jnp.asarray(network_data.ptdf),
        from_node=jnp.asarray(network_data.from_nodes),
        to_node=jnp.asarray(network_data.to_nodes),
        n_0_flow=jnp.asarray(n_0_flow_after_injection_outage),
        multi_outages=jnp.asarray(branch_outages),
        branches_monitored=branches_monitored,
    )
    assert bool(ordinary_success) is ordinary_modf_succeeds

    split_flow, main_bus, main_monitored, _main_imbalance, rank_loss = compute_split_modf_main_grid(
        ptdf=jnp.asarray(network_data.ptdf),
        from_node=jnp.asarray(network_data.from_nodes),
        to_node=jnp.asarray(network_data.to_nodes),
        nodal_injections=jnp.asarray(physical_nodal_injections),
        n_0_flow=jnp.asarray(n_0_flow_after_injection_outage),
        outages=jnp.asarray(branch_outages[None, :]),
        branches_monitored=branches_monitored,
        slack_bus=jnp.asarray(network_data.slack, dtype=jnp.int32),
    )

    split_flow = np.asarray(split_flow[0, 0])
    main_monitored = np.asarray(main_monitored[0])
    reference_n_1_flow = reference_n_1_flow.copy()
    reference_n_1_flow[branch_outages] = 0.0

    if ordinary_modf_succeeds:
        assert int(rank_loss[0]) == 0
        assert np.all(main_monitored)
        np.testing.assert_allclose(split_flow, np.asarray(ordinary_flow[0]), atol=TOL, rtol=0.0)
    else:
        assert not np.all(np.asarray(main_bus[0]))
        assert int(rank_loss[0]) > 0
    np.testing.assert_allclose(
        split_flow[main_monitored],
        reference_n_1_flow[main_monitored],
        atol=TOL,
        rtol=0.0,
    )
    assert np.all(np.isnan(split_flow[~main_monitored]))
