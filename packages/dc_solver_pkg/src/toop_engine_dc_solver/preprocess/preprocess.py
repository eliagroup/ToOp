# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Preprocess module for the DC solver

Provides high-level routines for converting data from a backend (Pandapower, PowerFactory, ...)
to the format required by the DC solver. The output is two-fold, a network_data object that contains
descriptive information and is required in the postprocessing to apply topologies to the original
network, and a static_information object that only contains the information necessary to run the
DC solver. The static_information is a jax dataclass and will reside on GPU memory, the
network_data is not needed for running the solver itself.
"""

from dataclasses import replace

import numpy as np
import structlog
from beartype.typing import Callable, Optional
from jaxtyping import Bool, Int
from toop_engine_dc_solver.preprocess.action_set import (
    determine_injection_topology,
    enumerate_branch_actions,
)
from toop_engine_dc_solver.preprocess.helpers.branch_topology import (
    get_branch_direction,
    zip_branch_lists,
)
from toop_engine_dc_solver.preprocess.helpers.find_bridges import (
    find_bridges,
    find_n_minus_2_safe_branches,
)
from toop_engine_dc_solver.preprocess.helpers.injection_topology import (
    compute_nodal_injection,
    get_mw_injections_at_nodes,
)
from toop_engine_dc_solver.preprocess.helpers.node_grouping import (
    convert_boolean_mask_to_index_array,
    get_num_elements_per_node,
    group_by_node,
)
from toop_engine_dc_solver.preprocess.helpers.psdf import compute_psdf
from toop_engine_dc_solver.preprocess.helpers.ptdf import (
    compute_ptdf,
    get_extended_nodal_injections,
    get_extended_ptdf,
)
from toop_engine_dc_solver.preprocess.helpers.reduce_node_dimension import (
    get_significant_nodes,
    reduce_ptdf_and_nodal_injections,
    update_ids_linking_to_nodes,
)
from toop_engine_dc_solver.preprocess.helpers.relevant_branches import (
    get_relevant_branches,
)
from toop_engine_dc_solver.preprocess.network_data import (
    NetworkData,
    assert_network_data,
    extract_network_data_from_interface,
)
from toop_engine_dc_solver.preprocess.preprocess_bb_outage import get_busbar_map_adjacent_branches, preprocess_bb_outages
from toop_engine_dc_solver.preprocess.preprocess_station_realisations import (
    enumerate_station_realisations,
)
from toop_engine_dc_solver.preprocess.preprocess_switching import (
    OptimalSeparationSetInfo,
    make_optimal_separation_set,
    prepare_for_separation_set,
)
from toop_engine_interfaces.asset_topology.asset_topology import RuntimeAssetTopology
from toop_engine_interfaces.asset_topology.materialized_topology import MaterializedAssetConnection, MaterializedStation
from toop_engine_interfaces.backend import BackendInterface
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters, ReassignmentLimits
from toop_engine_interfaces.messages.preprocess.preprocess_heartbeat import (
    PreprocessStage,
    empty_status_update_fn,
)

logger = structlog.get_logger(__name__)


def _get_runtime_asset_ids_per_bus_id(station: MaterializedStation) -> dict[str, set[str]]:
    """Collect unique in-service runtime asset ids per current bus-branch bus id.

    Parameters
    ----------
    station : MaterializedStation
        Runtime station whose currently energized bus groups should be evaluated.

    Returns
    -------
    dict[str, set[str]]
        Mapping from non-empty bus-branch bus id to unique connected in-service
        branch and injection asset ids.
    """
    asset_ids_per_bus_id: dict[str, set[str]] = {}

    for busbar_index, busbar in enumerate(station.busbars):
        bus_id = busbar.bus_branch_bus_id
        if not busbar.in_service or bus_id in {None, ""}:
            continue

        connected_asset_ids = asset_ids_per_bus_id.setdefault(bus_id, set())
        connected_asset_ids.update(
            f"branch:{asset.grid_model_id}" for asset in station.get_connected_assets(busbar_index, asset_scope="branch")
        )
        connected_asset_ids.update(
            f"injection:{asset.grid_model_id}"
            for asset in station.get_connected_assets(busbar_index, asset_scope="injection")
        )

    return asset_ids_per_bus_id


def _get_effective_station_bus_components(
    station: MaterializedStation,
    pst_branch_ids: set[str],
) -> list[set[str]]:
    """Collapse runtime bus ids that are internally tied together by PST branches.

    Parameters
    ----------
    station : MaterializedStation
        Runtime station to analyze.
    pst_branch_ids : set[str]
        Grid-model branch ids that represent phase-shifting transformers.

    Returns
    -------
    list[set[str]]
        Connected components of active runtime bus ids after merging bus ids that
        share a PST branch inside the station.
    """
    active_bus_ids = {
        busbar.bus_branch_bus_id
        for busbar in station.busbars
        if busbar.in_service and busbar.bus_branch_bus_id not in {None, ""}
    }
    if not active_bus_ids:
        return []

    neighbors = _get_pst_neighbors_by_bus_id(station=station, active_bus_ids=active_bus_ids, pst_branch_ids=pst_branch_ids)
    return _get_connected_bus_components(neighbors)


def _get_pst_neighbors_by_bus_id(
    station: MaterializedStation,
    active_bus_ids: set[str],
    pst_branch_ids: set[str],
) -> dict[str, set[str]]:
    """Build runtime bus adjacency after collapsing PST-linked internal buses."""
    neighbors = {bus_id: {bus_id} for bus_id in active_bus_ids}
    pst_bus_ids_by_asset_id: dict[str, set[str]] = {}
    for busbar_index, busbar in enumerate(station.busbars):
        bus_id = busbar.bus_branch_bus_id
        if not busbar.in_service or bus_id not in active_bus_ids:
            continue
        for asset in station.get_connected_assets(busbar_index, asset_scope="branch"):
            if asset.grid_model_id in pst_branch_ids:
                pst_bus_ids_by_asset_id.setdefault(asset.grid_model_id, set()).add(bus_id)

    for connected_bus_ids in pst_bus_ids_by_asset_id.values():
        for bus_id in connected_bus_ids:
            neighbors[bus_id].update(connected_bus_ids)
    return neighbors


def _get_connected_bus_components(neighbors: dict[str, set[str]]) -> list[set[str]]:
    """Return connected components of the provided bus adjacency map."""
    components: list[set[str]] = []
    remaining_bus_ids = set(neighbors)
    while remaining_bus_ids:
        start_bus_id = remaining_bus_ids.pop()
        component = {start_bus_id}
        pending_bus_ids = [start_bus_id]
        while pending_bus_ids:
            current_bus_id = pending_bus_ids.pop()
            for neighbor_bus_id in neighbors[current_bus_id]:
                if neighbor_bus_id in remaining_bus_ids:
                    remaining_bus_ids.remove(neighbor_bus_id)
                    component.add(neighbor_bus_id)
                    pending_bus_ids.append(neighbor_bus_id)
        components.append(component)
    return components


def _get_materially_split_station_bus_ids(station: MaterializedStation, pst_branch_ids: set[str]) -> set[str]:
    """Return bus ids only for runtime splits that affect more than a singleton island.

    Parameters
    ----------
    station : MaterializedStation
        Runtime station to classify.
    pst_branch_ids : set[str]
        Grid-model branch ids that represent phase-shifting transformers.

    Returns
    -------
    set[str]
        The active bus-branch bus ids when at least two current bus groups each carry
        more than one connected in-service asset. Otherwise an empty set.
    """
    asset_ids_per_bus_id = _get_runtime_asset_ids_per_bus_id(station)
    effective_components = _get_effective_station_bus_components(station, pst_branch_ids)
    if len(effective_components) <= 1:
        return set()

    effective_component_asset_counts = [
        len(set().union(*(asset_ids_per_bus_id.get(bus_id, set()) for bus_id in component)))
        for component in effective_components
    ]
    sorted_asset_counts = sorted(effective_component_asset_counts, reverse=True)
    if len(sorted_asset_counts) < 2 or sorted_asset_counts[1] <= 1:
        return set()

    return set().union(*effective_components)


def _get_local_busbar_keep_indices(station: MaterializedStation, node_id: str | None) -> list[int]:
    """Return busbar indices for the requested electrical node slice."""
    if node_id is None or node_id not in station.bus_branch_bus_ids or len(station.bus_branch_bus_ids) <= 1:
        return list(range(len(station.busbars)))

    return [index for index, busbar in enumerate(station.busbars) if busbar.bus_branch_bus_id == node_id]


def _select_locally_switched_asset_indices(
    asset_connections: list[MaterializedAssetConnection],
    allowed_asset_ids: set[str],
    switching_rows: np.ndarray,
) -> list[int]:
    """Keep the first locally switched occurrence of each requested asset id."""
    selected_indices: list[int] = []
    seen_asset_ids: set[str] = set()

    for index, connection in enumerate(asset_connections):
        asset_id = connection.asset.grid_model_id
        if asset_id not in allowed_asset_ids or asset_id in seen_asset_ids:
            continue
        if not np.any(switching_rows[:, index]):
            continue
        selected_indices.append(index)
        seen_asset_ids.add(asset_id)

    return selected_indices


def _filter_asset_bay_to_kept_busbars(
    asset_connection: MaterializedAssetConnection,
    kept_busbar_grid_model_ids: set[str],
) -> MaterializedAssetConnection:
    """Drop asset-bay SR references to busbars that were removed from the slice."""
    if asset_connection.asset_bay is None:
        return asset_connection

    return asset_connection.model_copy(
        update={
            "asset_bay": asset_connection.asset_bay.model_copy(
                update={
                    "sr_switch_grid_model_id": {
                        busbar_id: switch_id
                        for busbar_id, switch_id in asset_connection.asset_bay.sr_switch_grid_model_id.items()
                        if busbar_id in kept_busbar_grid_model_ids
                    }
                }
            )
        }
    )


def _slice_optional_matrix(
    matrix: np.ndarray | None,
    row_indices: list[int],
    column_indices: list[int],
) -> np.ndarray | None:
    """Slice an optional switching/connectivity matrix if present."""
    if matrix is None:
        return None
    return matrix[np.ix_(row_indices, column_indices)]


def _project_station_to_local_assets(
    station: MaterializedStation,
    branch_ids: list[str],
    injection_ids: list[str],
    node_id: str | None = None,
) -> MaterializedStation:
    """Restrict a runtime station to the assets represented by one relevant node slice.

    Parameters
    ----------
    station : MaterializedStation
        Runtime station to project.
    branch_ids : list[str]
        Branch ids present in the relevant-node slice.
    injection_ids : list[str]
        Injection ids present in the relevant-node slice.
    node_id : str | None, optional
        Runtime electrical bus id of the relevant-node slice. When it matches one
        of multiple bus groups inside the runtime station, only that active bus
        group is kept.

    Returns
    -------
    MaterializedStation
        Runtime station with switching-table columns restricted to the requested assets.
    """
    busbar_keep_indices = _get_local_busbar_keep_indices(station, node_id)
    kept_busbars = [station.busbars[index] for index in busbar_keep_indices]
    kept_busbar_int_ids = {busbar.int_id for busbar in kept_busbars}
    kept_busbar_grid_model_ids = {busbar.grid_model_id for busbar in kept_busbars}

    branch_switching_rows = station.branch_switching_table[busbar_keep_indices, :]
    injection_switching_rows = station.injection_switching_table[busbar_keep_indices, :]
    branch_keep_indices = _select_locally_switched_asset_indices(
        asset_connections=station.branch_connections,
        allowed_asset_ids=set(branch_ids),
        switching_rows=branch_switching_rows,
    )
    injection_keep_indices = _select_locally_switched_asset_indices(
        asset_connections=station.injection_connections,
        allowed_asset_ids=set(injection_ids),
        switching_rows=injection_switching_rows,
    )

    return station.model_copy(
        update={
            "bus_branch_bus_ids": [node_id],
            "busbars": kept_busbars,
            "couplers": [
                coupler
                for coupler in station.couplers
                if coupler.busbar_from_id in kept_busbar_int_ids and coupler.busbar_to_id in kept_busbar_int_ids
            ],
            "branch_connections": [
                _filter_asset_bay_to_kept_busbars(station.branch_connections[index], kept_busbar_grid_model_ids)
                for index in branch_keep_indices
            ],
            "injection_connections": [
                _filter_asset_bay_to_kept_busbars(station.injection_connections[index], kept_busbar_grid_model_ids)
                for index in injection_keep_indices
            ],
            "branch_switching_table": station.branch_switching_table[np.ix_(busbar_keep_indices, branch_keep_indices)],
            "injection_switching_table": station.injection_switching_table[
                np.ix_(busbar_keep_indices, injection_keep_indices)
            ],
            "branch_connectivity": _slice_optional_matrix(
                station.branch_connectivity,
                busbar_keep_indices,
                branch_keep_indices,
            ),
            "injection_connectivity": _slice_optional_matrix(
                station.injection_connectivity,
                busbar_keep_indices,
                injection_keep_indices,
            ),
        }
    )


def disable_busbar_outage_contingencies(network_data: NetworkData) -> NetworkData:
    """Clear busbar-outage configuration from network data.

    The importer may provide a busbar outage map unconditionally, but when preprocessing-side
    busbar outages are disabled we must not reconstruct bus contingencies from it later on.
    """
    return replace(network_data, busbar_outage_map=None)


def compute_ptdf_if_not_given(network_data: NetworkData) -> NetworkData:
    """Compute the PTDF if not given.

    Parameters
    ----------
    network_data : NetworkData
        The network data to compute the PTDF for

    Returns
    -------
    NetworkData
        The network data with the PTDF computed
    """
    if network_data.ptdf is None:
        network_data = replace(
            network_data,
            ptdf=compute_ptdf(
                network_data.from_nodes,
                network_data.to_nodes,
                network_data.susceptances,
                network_data.slack,
            ),
        )

    return network_data


def compute_psdf_if_not_given(network_data: NetworkData) -> NetworkData:
    """Compute the PSDF if not given.

    Parameters
    ----------
    network_data : NetworkData
        The network data to compute the PTDF for

    Returns
    -------
    NetworkData
        The network data with the PSDF computed
    """
    assert network_data.ptdf is not None, "PSDF computation not possible without PTDF. Please compute first"
    if network_data.psdf is None:
        network_data = replace(
            network_data,
            psdf=compute_psdf(
                network_data.ptdf,
                network_data.from_nodes,
                network_data.to_nodes,
                network_data.susceptances,
                network_data.phase_shift_mask,
                network_data.base_mva,
            ),
        )

    return network_data


def filter_relevant_nodes_branch_count(network_data: NetworkData) -> NetworkData:
    """Filter the relevant nodes to only include nodes with at least 4 non-bridge branches connected.

    Parameters
    ----------
    network_data : NetworkData
        The network data to filter

    Returns
    -------
    NetworkData
        The network data with an adjusted relevant node mask
    """
    assert network_data.bridging_branch_mask is not None, "Bridges have to be computed before filtering relevant nodes"

    relevant_node_indices = np.flatnonzero(network_data.relevant_node_mask)
    n_connections = np.array(
        list(
            map(
                lambda node_idx: (
                    np.sum((network_data.from_nodes == node_idx) & ~network_data.bridging_branch_mask)
                    + np.sum((network_data.to_nodes == node_idx) & ~network_data.bridging_branch_mask)
                ),
                relevant_node_indices,
            )
        )
    )
    keep_condition = n_connections >= 4

    removed_relevant_nodes = np.array(network_data.node_ids)[relevant_node_indices[~keep_condition]]
    if len(removed_relevant_nodes) > 0:
        logger.info(
            f"Removed {len(removed_relevant_nodes)} relevant nodes, "
            "since they had less than 4 non-bridge branches connected.",
            removed_relevant_node_ids=np.array2string(removed_relevant_nodes),
        )

    return remove_relevant_subs(network_data, keep_mask=keep_condition)


def filter_relevant_nodes_no_asset_station(network_data: NetworkData) -> NetworkData:
    """Filter the relevant node masks to include only those for which an asset topology is available.

    Parameters
    ----------
    network_data : NetworkData
        The network data to filter

    Returns
    -------
    NetworkData
        The network data with the relevant node mask adjusted to only include nodes with an asset topology
    """
    relevant_node_ids = np.array(network_data.node_ids)[np.flatnonzero(network_data.relevant_node_mask)]
    assert network_data.asset_topology is not None, "Missing runtime asset-topology stations"
    station_bus_ids = network_data.electrical_bus_to_station.keys()

    keep_mask = np.isin(relevant_node_ids, np.array(sorted(station_bus_ids), dtype=object))

    for node_id in relevant_node_ids[~keep_mask]:
        logger.warning(f"Removed relevant node {node_id}, since no asset topology is available for it")

    return remove_relevant_subs(network_data, keep_mask=keep_mask)


def filter_relevant_split_asset_stations(network_data: NetworkData) -> NetworkData:
    """Filter out relevant nodes whose runtime station view is materially split.

    Parameters
    ----------
    network_data : NetworkData
        The network data carrying relevant-node masks and runtime-enriched asset-topology stations.

    Returns
    -------
    NetworkData
        Network data with relevant nodes removed when their runtime station spans multiple
        non-trivial active bus groups.
    """
    relevant_node_ids = np.array(network_data.node_ids)[np.flatnonzero(network_data.relevant_node_mask)]
    assert network_data.asset_topology is not None, "Missing runtime asset-topology stations"
    pst_branch_ids = {network_data.branch_ids[index] for index in np.flatnonzero(network_data.controllable_phase_shift_mask)}
    split_station_bus_ids = {
        bus_id
        for station in network_data.asset_topology.stations
        for bus_id in _get_materially_split_station_bus_ids(station, pst_branch_ids)
    }
    if not split_station_bus_ids:
        return network_data

    keep_mask = ~np.isin(relevant_node_ids, np.array(sorted(split_station_bus_ids), dtype=object))
    for node_id in relevant_node_ids[~keep_mask]:
        logger.warning(f"Removed relevant node {node_id}, since its runtime asset topology station is split")

    return remove_relevant_subs(network_data, keep_mask=keep_mask)


def _switching_table_has_double_connections(switching_table: np.ndarray) -> bool:
    """Return whether a switching table contains assets connected to multiple busbars."""
    return bool(switching_table.size > 0 and np.any(np.sum(switching_table, axis=0) > 1))


def filter_relevant_nodes_no_double_connections(network_data: NetworkData) -> NetworkData:
    """Filter relevant nodes whose asset-topology station contains double connections.

    Parameters
    ----------
    network_data : NetworkData
        The network data to filter.

    Returns
    -------
    NetworkData
        The network data with relevant nodes removed when their station has double connections.
    """
    assert network_data.asset_topology is not None, "Asset topology has to be passed in"
    relevant_node_ids = np.array(network_data.node_ids)[np.flatnonzero(network_data.relevant_node_mask)]
    station_ids_without_double_connections = np.array(
        [
            bus_id
            for bus_id, station in network_data.electrical_bus_to_station.items()
            if not _switching_table_has_double_connections(station.asset_switching_table)
        ]
    )
    keep_mask = np.isin(relevant_node_ids, station_ids_without_double_connections)

    for node_id in relevant_node_ids[~keep_mask]:
        logger.warning(f"Removed relevant node {node_id}, since its asset topology station has double connections")

    return remove_relevant_subs(network_data, keep_mask=keep_mask)


def compute_bridging_branches(network_data: NetworkData) -> NetworkData:
    """Identify branches whose outages would lead to islanding of the network (like bridges to islands).

    Parameters
    ----------
    network_data : NetworkData
        The network data including nodes and branches to analyze

    Returns
    -------
    NetworkData
        The network data with the PSDF computed
    """
    from_node = network_data.from_nodes
    to_node = network_data.to_nodes

    number_of_branches = len(network_data.branch_ids)
    number_of_busses = len(network_data.node_ids)
    branch_is_bridge = find_bridges(from_node, to_node, number_of_branches, number_of_busses)

    return replace(network_data, bridging_branch_mask=branch_is_bridge)


def add_nodal_injections_to_network_data(network_data: NetworkData) -> NetworkData:
    """Compute the nodal injection for all nodes in the network data.

    Parameters
    ----------
    network_data : NetworkData
        The network data including nodes and injections to analyze

    Returns
    -------
    NetworkData
        The network data with the nodal injections computed including bus B
    """
    injection_power = network_data.mw_injections
    injection_nodes = network_data.injection_nodes

    number_of_nodes = len(network_data.node_ids)

    nodal_injection = compute_nodal_injection(injection_power, injection_nodes, number_of_nodes)
    return replace(network_data, nodal_injection=nodal_injection)


def combine_phaseshift_and_injection(network_data: NetworkData) -> NetworkData:
    """Add PSDF to PTDF columns and shifts in degree to nodal injections and corresponding masks.

    Description
    -----------
    The PSDF needs exactly the same updates for line outages and bus splits. Therefore, the PSDF can be assumed to be
    a column of the PTDF and the angle_shift vector to be a nodal injection. That is exactly how we implemented
    the PSDF. We add masks and node/branch info for the phase shifters, too.

    Parameters
    ----------
    network_data : NetworkData
        The network data including nodal injections, phaseshifts, psdf, and ptdf

    Returns
    -------
    NetworkData
        The network data with the phase shift data added on top of the injection data,
        including updated masks on node/branch level
    """
    # Add PSDF into PTDF as new columns in the front
    assert network_data.ptdf is not None, "The PTDF has to be computed first!"
    assert network_data.psdf is not None, "The PSDF has to be computed first!"
    assert network_data.nodal_injection is not None, "The nodal injections have to be computed first"
    ptdf = np.concatenate([network_data.psdf, network_data.ptdf], axis=1)

    # Gather phase_shifter data for update
    phase_shift_mask = network_data.phase_shift_mask
    phase_shift_indices = np.flatnonzero(phase_shift_mask)
    number_of_phase_shifters = phase_shift_indices.shape[0]

    # We want to find out for each controllable PST which injection it is connected to
    controllable_psts = np.flatnonzero(network_data.controllable_phase_shift_mask[phase_shift_mask])
    controllable_pst_node_mask = np.zeros((number_of_phase_shifters + len(network_data.node_ids),), dtype=bool)
    controllable_pst_node_mask[controllable_psts] = True

    # Add nodal injections to the phase shifters
    phase_shift_names = [network_data.branch_names[i] for i in phase_shift_indices]
    phase_shift_ids = [network_data.branch_ids[i] for i in phase_shift_indices]
    phase_shifter_type = ["PST"] * number_of_phase_shifters
    phase_shifter_node_type = ["PSTNode"] * number_of_phase_shifters
    phase_shifter_angle_shift = network_data.shift_angles[:, phase_shift_mask]

    # Update nodal injections to include phase shift degrees
    nodal_injection = np.concatenate([phase_shifter_angle_shift, network_data.nodal_injection], axis=1)

    # Update the node related information. (The first few nodes will contain the phase_shifter information.)
    node_names = phase_shift_names + network_data.node_names
    node_ids = phase_shift_ids + network_data.node_ids
    node_types = phase_shifter_node_type + network_data.node_types

    slack = network_data.slack + number_of_phase_shifters
    relevant_node_mask = np.concatenate(
        [
            np.zeros(number_of_phase_shifters, dtype=bool),
            network_data.relevant_node_mask,
        ]
    )
    multi_outage_node_mask = np.concatenate(
        [
            np.zeros(
                (
                    network_data.multi_outage_node_mask.shape[0],
                    number_of_phase_shifters,
                ),
                dtype=bool,
            ),
            network_data.multi_outage_node_mask,
        ],
        axis=1,
    )
    from_nodes = network_data.from_nodes + number_of_phase_shifters
    to_nodes = network_data.to_nodes + number_of_phase_shifters

    # Update injection data
    injection_nodes = np.concatenate(
        [
            network_data.injection_nodes + number_of_phase_shifters,
            np.arange(number_of_phase_shifters, dtype=int),
        ]
    )
    injection_names = network_data.injection_names + phase_shift_names
    injection_ids = network_data.injection_ids + phase_shift_ids
    injection_types = network_data.injection_types + phase_shifter_type
    injection_outages = np.concatenate(
        [
            network_data.outaged_injection_mask,
            np.zeros(number_of_phase_shifters, dtype=bool),
        ]
    )

    mw_injections = np.concatenate([network_data.mw_injections, phase_shifter_angle_shift], axis=1)

    return replace(
        network_data,
        ptdf=ptdf,
        nodal_injection=nodal_injection,
        node_ids=node_ids,
        node_names=node_names,
        node_types=node_types,
        slack=slack,
        relevant_node_mask=relevant_node_mask,
        from_nodes=from_nodes,
        to_nodes=to_nodes,
        injection_nodes=injection_nodes,
        injection_names=injection_names,
        injection_ids=injection_ids,
        injection_types=injection_types,
        outaged_injection_mask=injection_outages,
        mw_injections=mw_injections,
        multi_outage_node_mask=multi_outage_node_mask,
        controllable_pst_node_mask=controllable_pst_node_mask,
    )


def add_bus_b_columns_to_ptdf(network_data: NetworkData) -> NetworkData:
    """Add new columns for split busses to the PTDF.

    The nodes to split are identified by the relevant node mask

    The new columns represent busbar B of the relevant substation, i.e. the busbar that has no
    connections in unsplit state. It also extends all masks (multi-outage, relevant nodes) to
    include the new busbar.

    Parameters
    ----------
    network_data : NetworkData
        The network data including an unextended ptdf and relevant node mask

    Returns
    -------
    NetworkData
        The network data with the extended ptdf
    """
    # Build extended PTDF
    assert network_data.ptdf_is_extended is False, "PTDF was already extended. Extending it again would lead to issues"

    rel_node_indices = np.flatnonzero(network_data.relevant_node_mask)
    n_rel_nodes = len(rel_node_indices)

    # Extend relevant node mask with zeros as by convention, relevant node mask can only point to
    # busbar A
    # Extend multi-outage node mask with zeros too, as no multi-outages can be defined for relevant
    # nodes at this point in time (TODO revisit this once busbar outages for relevant nodes are
    # implemented)
    return replace(
        network_data,
        ptdf=get_extended_ptdf(network_data.ptdf, network_data.relevant_node_mask),
        nodal_injection=get_extended_nodal_injections(network_data.nodal_injection, network_data.relevant_node_mask),
        relevant_node_mask=np.concatenate(
            [
                network_data.relevant_node_mask,
                np.zeros(n_rel_nodes, dtype=bool),
            ]
        ),
        multi_outage_node_mask=np.concatenate(
            [
                network_data.multi_outage_node_mask,
                np.zeros(
                    (network_data.multi_outage_node_mask.shape[0], n_rel_nodes),
                    dtype=bool,
                ),
            ],
            axis=1,
        ),
        controllable_pst_node_mask=np.concatenate(
            [
                network_data.controllable_pst_node_mask,
                np.zeros(n_rel_nodes, dtype=bool),
            ]
        )
        if network_data.controllable_pst_node_mask is not None
        else None,
        node_ids=network_data.node_ids + [network_data.node_ids[i] for i in rel_node_indices],
        node_names=network_data.node_names + [f"{network_data.node_names[i]}_bus_b" for i in rel_node_indices],
        node_types=network_data.node_types + ["BUS_B"] * n_rel_nodes,
        ptdf_is_extended=True,
    )


def filter_inactive_injections(network_data: NetworkData) -> NetworkData:
    """Filter out all inactive injections from the network data.

    Parameters
    ----------
    network_data : NetworkData
        The network data to filter

    Returns
    -------
    NetworkData
        The network data with inactive injections removed
    """
    assert network_data.injection_idx_at_nodes is None, "Please filter injections before computing topology info"

    active_injections = np.any(network_data.mw_injections != 0, axis=0)
    active_injections_idx = np.flatnonzero(active_injections)
    return replace(
        network_data,
        injection_nodes=network_data.injection_nodes[active_injections_idx],
        mw_injections=network_data.mw_injections[:, active_injections_idx],
        outaged_injection_mask=network_data.outaged_injection_mask[active_injections_idx],
        injection_ids=[network_data.injection_ids[i] for i in active_injections_idx],
        injection_names=[network_data.injection_names[i] for i in active_injections_idx],
        injection_types=[network_data.injection_types[i] for i in active_injections_idx],
    )


def compute_injection_topology_info(network_data: NetworkData) -> NetworkData:
    """Compute the injection topology info at each relevant node.

    This includes grouping the injections at each relevant node

    Parameters
    ----------
    network_data : NetworkData
        The network data including basic injection info

    Returns
    -------
    NetworkData
        The network data with topological enhanced injection info
    """
    relevant_node_idx = np.flatnonzero(network_data.relevant_node_mask)
    injection_idx_at_node = group_by_node(network_data.injection_nodes, relevant_node_idx)
    num_injections_per_node = get_num_elements_per_node(injection_idx_at_node)
    mw_injections_at_node = get_mw_injections_at_nodes(injection_idx_at_node, network_data.mw_injections)
    # active_injections = identify_inactive_injections(mw_injections_at_node)
    active_injections = [np.ones(mw_injections.shape[1], dtype=bool) for mw_injections in mw_injections_at_node]

    return replace(
        network_data,
        injection_idx_at_nodes=injection_idx_at_node,
        num_injections_per_node=num_injections_per_node,
        active_injections=active_injections,
    )


def compute_branch_topology_info(network_data: NetworkData) -> NetworkData:
    """Compute the branch info at each relevant node.

    This includes grouping the branches at each relevant node
    indication the direction of each branch at each relevant node
    counting the amount of branches for each relevant node

    Parameters
    ----------
    network_data : NetworkData
        The network data containing from_node and to_node and relevant nodes

    Returns
    -------
    NetworkData
        The network data with additional branch topology informations
    """
    relevant_node_idx = np.flatnonzero(network_data.relevant_node_mask)
    branches_from_nodes = group_by_node(network_data.from_nodes, relevant_node_idx)
    branches_to_nodes = group_by_node(network_data.to_nodes, relevant_node_idx)

    branches_at_nodes = zip_branch_lists(branches_from_nodes, branches_to_nodes)

    return replace(
        network_data,
        branches_at_nodes=branches_at_nodes,
        branch_direction=get_branch_direction(branches_at_nodes, branches_from_nodes),
        num_branches_per_node=get_num_elements_per_node(branches_at_nodes),
    )


def reduce_branch_dimension(network_data: NetworkData) -> NetworkData:
    """Reduce the branch dimension of the network data.

    Only a subset of the branches are relevant for the computation, we can safely
    discard the rest.

    Parameters
    ----------
    network_data : NetworkData
        The network data with unnecessary branches

    Returns
    -------
    NetworkData
        The network data with the unnecessary branches removed
    """
    assert network_data.branches_at_nodes is None, (
        "Branches at nodes have to be computed after reducing the branch dimension"
    )
    assert network_data.branch_direction is None, "Branch direction has to be computed after reducing the branch dimension"
    assert network_data.num_branches_per_node is None, (
        "Branches per nodes have to be computed after reducing the branch dimension"
    )

    assert network_data.bridging_branch_mask is not None, "Bridges have to be computed before reducing the branch dimension"

    relevant_branches = get_relevant_branches(
        from_node=network_data.from_nodes,
        to_node=network_data.to_nodes,
        relevant_node_mask=network_data.relevant_node_mask,
        monitored_branch_mask=network_data.monitored_branch_mask,
        outaged_branch_mask=network_data.outaged_branch_mask,
        multi_outage_mask=network_data.multi_outage_branch_mask,
        busbar_outage_branch_mask=get_busbar_map_adjacent_branches(network_data),
        controllable_phase_shift_mask=network_data.controllable_phase_shift_mask,
    )

    pst_branches = np.flatnonzero(network_data.controllable_phase_shift_mask)
    kept_pst_branches = np.isin(pst_branches, relevant_branches)
    relevant_phase_shift_taps = list(
        [taps for taps, keep in zip(network_data.phase_shift_taps, kept_pst_branches, strict=True) if keep]
    )
    relevant_phase_shift_susceptance_taps = list(
        [
            susceptance_taps
            for susceptance_taps, keep in zip(network_data.phase_shift_susceptance_taps, kept_pst_branches, strict=True)
            if keep
        ]
    )
    relevant_phase_shift_starting_tap_idx = network_data.phase_shift_starting_tap_idx[kept_pst_branches]
    relevant_phase_shift_low_tap = network_data.phase_shift_low_tap[kept_pst_branches]
    relevant_parallel_pst_group_mask = None
    relevant_parallel_pst_group_ids = None
    if network_data.parallel_pst_group_mask is not None:
        relevant_parallel_pst_group_mask = network_data.parallel_pst_group_mask[:, kept_pst_branches]
        kept_group_rows = np.any(relevant_parallel_pst_group_mask, axis=1)
        relevant_parallel_pst_group_mask = relevant_parallel_pst_group_mask[kept_group_rows]
        if network_data.parallel_pst_group_ids is not None:
            relevant_parallel_pst_group_ids = [
                group_id for group_id, keep in zip(network_data.parallel_pst_group_ids, kept_group_rows, strict=True) if keep
            ]
    # PST branches carry a node injection as well, so we need to adjust the injection indices
    pst_node_indices = np.flatnonzero(network_data.controllable_pst_node_mask)
    # Assert that the number of PST branches and nodes is the same
    assert len(pst_branches) == len(pst_node_indices), (
        "Number of PST branches and PST nodes do not match. Please check the controllable PST masks."
    )
    if np.any(kept_pst_branches):
        # WARNING: This assumes that PSTs are ordered the same way in both masks
        kept_pst_nodes_indices = pst_node_indices[kept_pst_branches]
        # Adapt the controllable PST node mask
        kept_controllable_pst_node_mask = np.zeros(network_data.controllable_pst_node_mask.shape, dtype=bool)
        kept_controllable_pst_node_mask[kept_pst_nodes_indices] = True
    else:
        kept_controllable_pst_node_mask = np.zeros(network_data.controllable_pst_node_mask.shape, dtype=bool)

    return replace(
        network_data,
        ptdf=network_data.ptdf[relevant_branches, :],
        psdf=network_data.psdf[relevant_branches, :],
        ac_dc_mismatch=network_data.ac_dc_mismatch[:, relevant_branches],
        max_mw_flows=network_data.max_mw_flows[:, relevant_branches],
        max_mw_flows_n_1=network_data.max_mw_flows_n_1[:, relevant_branches],
        overload_weights=network_data.overload_weights[relevant_branches],
        n0_n1_max_diff_factors=network_data.n0_n1_max_diff_factors[relevant_branches],
        susceptances=network_data.susceptances[relevant_branches],
        from_nodes=network_data.from_nodes[relevant_branches],
        to_nodes=network_data.to_nodes[relevant_branches],
        shift_angles=network_data.shift_angles[:, relevant_branches],
        phase_shift_mask=network_data.phase_shift_mask[relevant_branches],
        controllable_phase_shift_mask=network_data.controllable_phase_shift_mask[relevant_branches],
        phase_shift_taps=relevant_phase_shift_taps,
        phase_shift_susceptance_taps=relevant_phase_shift_susceptance_taps,
        phase_shift_starting_tap_idx=relevant_phase_shift_starting_tap_idx,
        phase_shift_low_tap=relevant_phase_shift_low_tap,
        parallel_pst_group_mask=relevant_parallel_pst_group_mask,
        parallel_pst_group_ids=relevant_parallel_pst_group_ids,
        controllable_pst_node_mask=kept_controllable_pst_node_mask,
        monitored_branch_mask=network_data.monitored_branch_mask[relevant_branches],
        disconnectable_branch_mask=network_data.disconnectable_branch_mask[relevant_branches],
        outaged_branch_mask=network_data.outaged_branch_mask[relevant_branches],
        multi_outage_branch_mask=network_data.multi_outage_branch_mask[:, relevant_branches],
        branch_ids=[network_data.branch_ids[i] for i in relevant_branches],
        branch_names=[network_data.branch_names[i] for i in relevant_branches],
        branch_types=[network_data.branch_types[i] for i in relevant_branches],
        bridging_branch_mask=(
            network_data.bridging_branch_mask[relevant_branches] if network_data.bridging_branch_mask is not None else None
        ),
    )


def filter_disconnectable_branches_nminus2(network_data: NetworkData, n_processes: int = 1) -> NetworkData:
    """Filter the disconnectable branch mask to only include N-2 safe branches"""
    disconnectable_branches = np.flatnonzero(network_data.disconnectable_branch_mask)
    outage_cases = np.flatnonzero(network_data.outaged_branch_mask)
    n_minus_2_safe = find_n_minus_2_safe_branches(
        from_node=network_data.from_nodes,
        to_node=network_data.to_nodes,
        number_of_branches=len(network_data.branch_ids),
        number_of_nodes=len(network_data.node_ids),
        cases_to_check=disconnectable_branches,
        outage_cases=outage_cases,
        n_processes=n_processes,
    )
    disconnectable_branches = disconnectable_branches[n_minus_2_safe]

    n_minus_2_safe_mask = np.zeros_like(network_data.disconnectable_branch_mask)
    n_minus_2_safe_mask[disconnectable_branches] = True
    return replace(
        network_data,
        disconnectable_branch_mask=n_minus_2_safe_mask,
    )


def exclude_bridges_from_outage_masks(network_data: NetworkData) -> NetworkData:
    """Exclude bridges from the outage masks.

    Exclude bridges whose disconnection would lead to islanding from n-1 and disconnection-masks,
    since this would lead to 0-division anyway

    Parameters
    ----------
    network_data : NetworkData
        The network data with the bridging branch mask

    Returns
    -------
    NetworkData
        The network data with the briding branches removed from n-1 and disconnection-masks
    """
    assert network_data.bridging_branch_mask is not None, "Please compute bridges first!"
    excluded_outaged_branch_ids = np.array(network_data.branch_ids)[
        network_data.outaged_branch_mask & network_data.bridging_branch_mask
    ].tolist()
    if excluded_outaged_branch_ids:
        logger.info(
            "Excluded branches from mask",
            mask_name="outaged_branch_mask",
            reason="bridging_branch",
            n_excluded=len(excluded_outaged_branch_ids),
            excluded_branch_ids=excluded_outaged_branch_ids,
        )
    return replace(
        network_data,
        outaged_branch_mask=network_data.outaged_branch_mask & ~network_data.bridging_branch_mask,
        multi_outage_branch_mask=network_data.multi_outage_branch_mask & ~network_data.bridging_branch_mask,
        disconnectable_branch_mask=network_data.disconnectable_branch_mask & ~network_data.bridging_branch_mask,
    )


def convert_multi_outages(network_data: NetworkData) -> NetworkData:
    """Convert the multi-outage masks to a list of indices

    Furthermore, remove one of the branches from the mask to avoid islanding.
    Sort them by the amount of branches involved in the outage so that the backend can
    efficiently batch them

    Parameters
    ----------
    network_data : NetworkData
        The network data with the multi-outage masks

    Returns
    -------
    NetworkData
        The network data with the multi-outage masks converted to indices
    """
    # Make sure no outaged node is in relevant nodes
    # This is currently not supported
    assert not np.any(network_data.multi_outage_node_mask & network_data.relevant_node_mask[None, :])

    if not np.any(network_data.multi_outage_branch_mask) and not np.any(network_data.multi_outage_node_mask):
        return replace(network_data, split_multi_outage_branches=[], split_multi_outage_nodes=[])

    n_outaged_branches = np.sum(network_data.multi_outage_branch_mask, axis=1)
    sorted_indices = np.argsort(n_outaged_branches)

    # Reorder the multi-outage masks
    multi_outage_branch_mask = network_data.multi_outage_branch_mask[sorted_indices]
    multi_outage_node_mask = network_data.multi_outage_node_mask[sorted_indices]
    multi_outage_names = [network_data.multi_outage_names[i] for i in sorted_indices]
    multi_outage_ids = [network_data.multi_outage_ids[i] for i in sorted_indices]
    multi_outage_types = [network_data.multi_outage_types[i] for i in sorted_indices]
    n_outaged_branches = n_outaged_branches[sorted_indices]

    # Split the multi outage masks so that masks with the same number of branches are in one list
    split_indices = np.flatnonzero(np.diff(n_outaged_branches)) + 1
    multi_outage_branch_mask_split = np.split(multi_outage_branch_mask, split_indices, axis=0)
    multi_outage_node_mask_split = np.split(multi_outage_node_mask, split_indices, axis=0)

    # Convert the split list from boolean masks to indices for each outage
    branch_res = [convert_boolean_mask_to_index_array(mask) for mask in multi_outage_branch_mask_split]
    node_res = [convert_boolean_mask_to_index_array(mask) for mask in multi_outage_node_mask_split]

    # Furthermore, remove the first branch from the outage to avoid islanding
    # TODO find a more canonical way how to avoid islanding in trafo3w/busbar outages
    trafo_busbar_outage = np.array([elem_type in ["trafo3w", "bus"] for elem_type in multi_outage_types])
    trafo_busbar_outage = np.split(trafo_busbar_outage, split_indices)

    def _zero_out_first_branch(
        indices: Int[np.ndarray, " n_outages n_outaged_branches"],
        is_trafo_bus: Bool[np.ndarray, " n_outages"],
    ) -> Int[np.ndarray, " n_outages n_outaged_branches"]:
        """Set the first branch of the outage to -1, if it is a trafo or busbar outage.

        Parameters
        ----------
        indices : Int[np.ndarray, " n_outages n_outaged_branches"]
            The indices of the branches in the outage
        is_trafo_bus : Bool[np.ndarray, " n_outages"]
            The boolean mask indicating if the outage is a trafo or busbar outage

        Returns
        -------
        Int[np.ndarray, " n_outages n_outaged_branches"]
            The indices of the branches in the outage with the first branch set to -1
        """
        if indices.size == 0:
            return indices
        indices[is_trafo_bus, 0] = -1
        if np.all(indices[:, 0] == -1):
            indices = indices[:, 1:]
        return indices

    branch_res = [
        _zero_out_first_branch(out, is_trafo_bus) for out, is_trafo_bus in zip(branch_res, trafo_busbar_outage, strict=True)
    ]

    return replace(
        network_data,
        multi_outage_branch_mask=multi_outage_branch_mask,
        multi_outage_node_mask=multi_outage_node_mask,
        split_multi_outage_branches=branch_res,
        split_multi_outage_nodes=node_res,
        multi_outage_names=multi_outage_names,
        multi_outage_ids=multi_outage_ids,
        multi_outage_types=multi_outage_types,
    )


def extract_relevant_sub_injection_outages(
    injection_idx_at_nodes: list[Int[np.ndarray, " n_injections_at_node"]],
    injection_outage_mask: Bool[np.ndarray, " n_injection"],
) -> tuple[
    Int[np.ndarray, " n_rel_inj_failures"],
    Int[np.ndarray, " n_rel_inj_failures"],
    Int[np.ndarray, " n_rel_inj_failures"],
]:
    """Find the inj outages at relevant subs and return their indices

    Parameters
    ----------
    injection_idx_at_nodes : list[Int[np.ndarray, " n_injections_at_node"]]
        The injection indices at each relevant sub
    injection_outage_mask : Bool[np.ndarray, " n_injection"]
        The mask of the injection outages

    Returns
    -------
    Int[np.ndarray, " n_rel_inj_failures"]
        The indices of the substation for each relevant sub injection outage
    Int[np.ndarray, " n_rel_inj_failures"]
        The indices of the failed injection inside the substation
    Int[np.ndarray, " n_rel_inj_failures"]
        The indices of the failed injections globally into the injection array
    """
    rel_inj_failures_idx = []
    rel_inj_failures_sub = []
    rel_outage_indices = []
    for i, injections_at_node in enumerate(injection_idx_at_nodes):
        is_outaged = injection_outage_mask[injections_at_node]
        rel_inj_failures_sub.extend([i] * is_outaged.sum())
        rel_inj_failures_idx.extend(np.flatnonzero(is_outaged).tolist())
        rel_outage_indices.extend(injections_at_node[is_outaged].tolist())

    return (
        np.array(rel_inj_failures_sub, dtype=int),
        np.array(rel_inj_failures_idx, dtype=int),
        np.array(rel_outage_indices, dtype=int),
    )


def process_injection_outages(network_data: NetworkData) -> NetworkData:
    """Convert the injection outage mask into index and delta-p values"""
    assert network_data.injection_idx_at_nodes is not None, "Please compute injection topology info first"

    # Relevant outages need to be processed at runtime
    rel_inj_failures_sub, rel_inj_failures_idx, rel_inj_failures_global = extract_relevant_sub_injection_outages(
        network_data.injection_idx_at_nodes,
        network_data.outaged_injection_mask,
    )

    # The non-relevant outages are directly passed on
    outaged_injection_mask = np.copy(network_data.outaged_injection_mask)
    if rel_inj_failures_global.size > 0:
        outaged_injection_mask[rel_inj_failures_global] = False
    injection_idx = np.flatnonzero(outaged_injection_mask)
    injection_outage_node = network_data.injection_nodes[injection_idx]
    injection_outage_deltap = -network_data.mw_injections[:, injection_idx]

    return replace(
        network_data,
        nonrel_io_deltap=injection_outage_deltap,
        nonrel_io_node=injection_outage_node,
        nonrel_io_global_inj_index=injection_idx,
        rel_io_sub=rel_inj_failures_sub,
        rel_io_local_inj_index=rel_inj_failures_idx,
        rel_io_global_inj_index=rel_inj_failures_global,
    )


def compute_electrical_actions(
    network_data: NetworkData,
    exclude_bridge_lookup_splits: bool = True,
    exclude_bsdf_lodf_splits: bool = False,
    bsdf_lodf_batch_size: int = 8,
    clip_to_n_actions: int = 2**23,
    reassignment_limits: Optional[ReassignmentLimits] = None,
) -> NetworkData:
    """Compute the electrical branch actions for the grid and update the network data accordingly

    Takes some additional parameters for the branch action computation
    Injection actions are not handled here as they are just pulled out of the asset topology, meaning they can only be
    added after the realization.

    Parameters
    ----------
    network_data : NetworkData
        The network data to compute the branch actions for
    exclude_bridge_lookup_splits : bool, optional
        Exclude actions that isolate a non-bridge branch on a busbar. Should only be False if you
        plan to do some post-processing on the branch actions.
    exclude_bsdf_lodf_splits : bool, optional
        Exclude actions that fail after applying both the bsdf and lodf. Setting this to true will increase
        the preprocessing time but will reduce the number of actions slightly.
    bsdf_lodf_batch_size : int, optional
        The batch size for the bsdf and lodf computation, if enabled.
    clip_to_n_actions : int, optional
        Clip the number of actions to this number. Avoids blowing up for large substations, as the
        number of actions is exponential in the number of branches.
    reassignment_limits : Optional[ReassignmentLimits], optional
        Settings to limit the amount of reassignment during the electrical reconfiguration.

    Returns
    -------
    NetworkData
        The network data with the branch actions computed
    """
    assert network_data.ptdf_is_extended is False, "Please filter relevant nodes first, before extending the ptdf"
    assert network_data.separation_sets_info is not None, "Please compute separation sets before computing branch actions"
    branch_actions = enumerate_branch_actions(
        network_data=network_data,
        exclude_isolations=True,
        exclude_bridge_lookup_splits=exclude_bridge_lookup_splits,
        exclude_bsdf_lodf_splits=exclude_bsdf_lodf_splits,
        bsdf_lodf_batch_size=bsdf_lodf_batch_size,
        clip_to_n_actions=clip_to_n_actions,
        reassignment_limits=reassignment_limits,
    )
    network_data = replace(
        network_data,
        branch_action_set=branch_actions,
    )
    return network_data


def remove_relevant_subs(
    network_data: NetworkData, keep_mask: Bool[np.ndarray, " n_rel_nodes_before_filter"]
) -> NetworkData:
    """Remove relevant subs from the network data according to a keep mask

    Parameters
    ----------
    network_data : NetworkData
        The network data to filter
    keep_mask : Bool[np.ndarray, " n_rel_nodes_before_filter"]
        The mask to keep the relevant subs, with as many entries as there were relevant subs before filtering.
        The number of true entries will determine the number of relevant subs after filtering.

    Returns
    -------
    NetworkData
        The network data with the relevant subs removed
    """
    if not np.any(keep_mask):
        raise ValueError(f"No relevant nodes out of previously {len(keep_mask)} left after filtering.")
    if np.all(keep_mask):
        return network_data

    original_relevant_nodes = np.flatnonzero(network_data.relevant_node_mask)
    relevant_nodes = original_relevant_nodes[keep_mask]

    irrelevant_node_ids = np.array(network_data.node_ids)[original_relevant_nodes[~keep_mask]]
    logger.info(
        f"Removed {len(irrelevant_node_ids)} from relevant nodes "
        "since they had no branch_actions. "
        "Check irrelevant_node_ids log attribute for details.",
        irrelevant_node_ids=np.array2string(irrelevant_node_ids),
    )
    relevant_node_mask = np.zeros_like(network_data.relevant_node_mask, dtype=bool)
    relevant_node_mask[relevant_nodes] = True

    # Remove from all attributes that are relevant-node specific
    branches_at_nodes = (
        [x for x, has_action in zip(network_data.branches_at_nodes, keep_mask, strict=True) if has_action]
        if network_data.branches_at_nodes is not None
        else None
    )
    branch_direction = (
        [x for x, has_action in zip(network_data.branch_direction, keep_mask, strict=True) if has_action]
        if network_data.branch_direction is not None
        else None
    )
    num_branches_per_node = (
        network_data.num_branches_per_node[keep_mask] if network_data.num_branches_per_node is not None else None
    )
    injection_idx_at_nodes = (
        [x for x, has_action in zip(network_data.injection_idx_at_nodes, keep_mask, strict=True) if has_action]
        if network_data.injection_idx_at_nodes is not None
        else None
    )
    num_injections_per_node = (
        network_data.num_injections_per_node[keep_mask] if network_data.num_injections_per_node is not None else None
    )
    active_injections = (
        [x for x, has_action in zip(network_data.active_injections, keep_mask, strict=True) if has_action]
        if network_data.active_injections is not None
        else None
    )
    cross_coupler_limits = (
        network_data.cross_coupler_limits[keep_mask] if network_data.cross_coupler_limits is not None else None
    )
    realised_stations = (
        [x for x, has_action in zip(network_data.realised_stations, keep_mask, strict=True) if has_action]
        if network_data.realised_stations is not None
        else None
    )
    simplified_asset_topology = (
        network_data.simplified_asset_topology.model_copy(
            update={
                "stations": [
                    station
                    for station, has_action in zip(network_data.simplified_asset_topology.stations, keep_mask, strict=True)
                    if has_action
                ]
            }
        )
        if network_data.simplified_asset_topology is not None
        else None
    )
    busbar_a_mappings = (
        [x for x, has_action in zip(network_data.busbar_a_mappings, keep_mask, strict=True) if has_action]
        if network_data.busbar_a_mappings is not None
        else None
    )
    branch_action_set_switching_distance = (
        [x for x, has_action in zip(network_data.branch_action_set_switching_distance, keep_mask, strict=True) if has_action]
        if network_data.branch_action_set_switching_distance is not None
        else None
    )
    injection_action_set = (
        [x for x, has_action in zip(network_data.injection_action_set, keep_mask, strict=True) if has_action]
        if network_data.injection_action_set is not None
        else None
    )
    branch_action_set = (
        [action for action, has_action in zip(network_data.branch_action_set, keep_mask, strict=True) if has_action]
        if network_data.branch_action_set is not None
        else None
    )

    return replace(
        network_data,
        relevant_node_mask=relevant_node_mask,
        branches_at_nodes=branches_at_nodes,
        branch_direction=branch_direction,
        num_branches_per_node=num_branches_per_node,
        injection_idx_at_nodes=injection_idx_at_nodes,
        num_injections_per_node=num_injections_per_node,
        active_injections=active_injections,
        cross_coupler_limits=cross_coupler_limits,
        branch_action_set=branch_action_set,
        realised_stations=realised_stations,
        simplified_asset_topology=simplified_asset_topology,
        busbar_a_mappings=busbar_a_mappings,
        branch_action_set_switching_distance=branch_action_set_switching_distance,
        injection_action_set=injection_action_set,
    )


def remove_relevant_subs_without_actions(network_data: NetworkData) -> NetworkData:
    """Filter out relevant subs which are left without branch actions after the action set generation

    Parameters
    ----------
    network_data : NetworkData
        The network data of the grid with all relevant subs that were used for the action set
        generation


    Returns
    -------
    NetworkData
        The network data with only the relevant subs that have actions
    """
    actions = network_data.branch_action_set

    assert network_data.rel_io_sub is None, "Call this before processing injections"

    keep_mask = np.array([action.shape[0] > 1 and np.any(action) for action in actions])
    if network_data.controllable_phase_shift_mask is not None and network_data.branches_at_nodes is not None:
        keep_mask = keep_mask | np.array(
            [
                bool(np.any(network_data.controllable_phase_shift_mask[branches_at_node]))
                for branches_at_node in network_data.branches_at_nodes
            ],
            dtype=bool,
        )

    # Remove from relevant node mask
    return remove_relevant_subs(network_data, keep_mask)


def compute_injection_actions(network_data: NetworkData) -> NetworkData:
    """Compute injection actions and materialize them into realized stations.

    Parameters
    ----------
    network_data : NetworkData
        The network data to compute the injection actions for.

    Returns
    -------
    NetworkData
        The network data with computed injection actions and updated realized stations.
    """
    assert network_data.branch_action_set is not None, "Branch action set is not available."
    assert network_data.realised_stations is not None, "Realised stations are not available."
    assert network_data.busbar_a_mappings is not None, "Busbar A mappings are not available."

    injection_actions = determine_injection_topology(network_data)
    realised_stations_with_injections = []
    for realised_stations, local_injection_actions, local_busbar_a_mappings in zip(
        network_data.realised_stations,
        injection_actions,
        network_data.busbar_a_mappings,
        strict=True,
    ):
        local_realised_stations = []
        for station, injection_action, busbar_a_mapping in zip(
            realised_stations,
            local_injection_actions,
            local_busbar_a_mappings,
            strict=True,
        ):
            if station.injection_switching_table.shape[1] == 0:
                local_realised_stations.append(station)
                continue

            busbar_a_indices = set(int(index) for index in busbar_a_mapping)
            busbar_b_indices = [index for index in range(len(station.busbars)) if index not in busbar_a_indices]
            updated_injection_switching_table = np.zeros_like(station.injection_switching_table, dtype=bool)

            for injection_idx, injection_on_bus_b in enumerate(injection_action.tolist()):
                target_busbar_indices = busbar_b_indices if injection_on_bus_b else sorted(busbar_a_indices)
                current_busbar_indices = np.flatnonzero(station.injection_switching_table[:, injection_idx]).tolist()

                if not target_busbar_indices:
                    updated_injection_switching_table[:, injection_idx] = station.injection_switching_table[:, injection_idx]
                    continue

                target_busbar_index = next(
                    (index for index in current_busbar_indices if index in target_busbar_indices),
                    target_busbar_indices[0],
                )
                updated_injection_switching_table[target_busbar_index, injection_idx] = True

            local_realised_stations.append(
                station.model_copy(update={"injection_switching_table": updated_injection_switching_table})
            )

        realised_stations_with_injections.append(local_realised_stations)

    return replace(
        network_data,
        injection_action_set=injection_actions,
        realised_stations=realised_stations_with_injections,
    )


def add_missing_asset_topo_info(network_data: NetworkData) -> NetworkData:
    """Validate that productive runtime station data is available.

    Parameters
    ----------
    network_data : NetworkData
        The network data whose runtime station information should be validated.

    Returns
    -------
    NetworkData
        The unchanged network data when runtime station data is available.

    Raises
    ------
    ValueError
        If runtime asset-topology stations are missing.
    """
    assert network_data.asset_topology is not None, (
        "Missing runtime asset-topology stations for asset topology preprocessing. "
        "Preprocessing requires backend-enriched runtime stations."
    )
    return network_data


def reduce_node_dimension(network_data: NetworkData) -> NetworkData:
    """Reduce the node dimension by removing nodes that are not relevant for the computation

    This should  happen before extending the ptdf, since otherwise the last columns are not the B-busses.

    Parameters
    ----------
    network_data : NetworkData
        The network data to reduce the node dimension for.
        Includes computed ptdf and nodal_injection

    Returns
    -------
    NetworkData:
        The network data without irrelevant nodes in all fields relating to nodes.
        All irrelevant nodes are grouped into a single node at the end of the various arrays
    """
    assert network_data.ptdf is not None, "The PTDF has to be computed before reducing the node dimension"
    assert network_data.psdf is not None, "The PSDF has to be computed before reducing the node dimension"
    assert network_data.nodal_injection is not None, (
        "Nodal Injections have to be computed before reducing the node dimension"
    )
    assert network_data.ptdf_is_extended is False, (
        "This step adds new columns at the end of the PTDF. Please extend the ptdf after reducing the node dimension."
    )
    assert network_data.split_multi_outage_nodes is None

    relevant_branches = get_relevant_branches(
        from_node=network_data.from_nodes,
        to_node=network_data.to_nodes,
        relevant_node_mask=network_data.relevant_node_mask,
        monitored_branch_mask=network_data.monitored_branch_mask,
        outaged_branch_mask=network_data.outaged_branch_mask,
        multi_outage_mask=network_data.multi_outage_branch_mask,
        busbar_outage_branch_mask=get_busbar_map_adjacent_branches(network_data),
        controllable_phase_shift_mask=network_data.controllable_phase_shift_mask,
    )
    significant_nodes = get_significant_nodes(
        network_data.relevant_node_mask,
        network_data.multi_outage_node_mask,
        relevant_branches,
        network_data.from_nodes,
        network_data.to_nodes,
        network_data.slack,
    )
    if network_data.busbar_outage_map is not None and network_data.asset_topology is not None:
        busbar_outage_station_ids = set(network_data.busbar_outage_map.keys())
        for station in network_data.asset_topology.stations:
            if station.bus_group_id in busbar_outage_station_ids:
                electrical_busses = set(station.bus_branch_bus_ids)
                significant_nodes |= np.array([node_id in electrical_busses for node_id in network_data.node_ids])
    significant_node_ids = np.flatnonzero(significant_nodes)
    ptdf, nodal_injection = reduce_ptdf_and_nodal_injections(
        network_data.ptdf, network_data.nodal_injection, significant_nodes
    )
    index_of_last_column = ptdf.shape[1] - 1
    from_nodes, to_nodes, injection_nodes, slack = update_ids_linking_to_nodes(
        network_data.from_nodes,
        network_data.to_nodes,
        network_data.injection_nodes,
        network_data.slack,
        significant_node_ids,
        index_of_last_column,
    )
    n_timesteps = nodal_injection.shape[0]
    return replace(
        network_data,
        ptdf=ptdf,
        nodal_injection=nodal_injection,
        from_nodes=from_nodes,
        to_nodes=to_nodes,
        injection_nodes=injection_nodes,
        slack=slack,
        node_ids=[network_data.node_ids[i] for i in significant_node_ids] + ["REDUCED_NODE"] * n_timesteps,
        node_names=[network_data.node_names[i] for i in significant_node_ids] + ["REDUCED_NODE"] * n_timesteps,
        node_types=[network_data.node_types[i] for i in significant_node_ids] + ["REDUCED_NODE"] * n_timesteps,
        relevant_node_mask=np.r_[network_data.relevant_node_mask[significant_nodes], [False] * n_timesteps],
        multi_outage_node_mask=np.c_[
            network_data.multi_outage_node_mask[:, significant_nodes],
            np.zeros((network_data.multi_outage_node_mask.shape[0], n_timesteps), dtype=bool),
        ],
    )


def simplify_asset_topology(network_data: NetworkData, close_couplers: bool = False) -> NetworkData:
    """Simplify the asset topology for easier handling in the preprocessing routines.

    Does the following to every station:
    - Close all open couplers if close_couplers is True (if False, the couplers are removed)
    - Order the assets in the station according to the branch_ids and injection_ids. The convention is to put the branches
    first and then the injections.
    - Remove out-of-service assets
    - Remove duplicate couplers
    - Remove disconnected busbars
    - Select an arbitraty bus for multi-connected assets without a coupler

    Parameters
    ----------
    network_data : NetworkData
        The network data to simplify the asset topology for
    close_couplers : bool, optional
        Whether to close the couplers or not, by default False. If False, the couplers are removed from the topology.

    Returns
    -------
    NetworkData
        The network data with the simplified asset topology
    """
    node_ids = [network_data.node_ids[i] for i in network_data.relevant_nodes]
    assert network_data.asset_topology is not None, (
        "Missing runtime asset-topology stations for asset topology simplification. "
        "Preprocessing requires backend-enriched runtime stations."
    )
    runtime_stations_by_node_id: dict[str | None, MaterializedStation] = network_data.electrical_bus_to_station

    not_found = [node_id for node_id in node_ids if node_id not in runtime_stations_by_node_id]
    if not_found:
        raise ValueError(f"Some stations were not found in the asset topology: {not_found}")

    runtime_stations = [runtime_stations_by_node_id[node_id] for node_id in node_ids]

    stations = []
    keep_mask = []
    for _node_index, branches_at_sub, inj_at_sub, station in zip(
        network_data.relevant_nodes,
        network_data.branches_at_nodes,
        network_data.injection_idx_at_nodes,
        runtime_stations,
        strict=True,
    ):
        node_id = network_data.node_ids[_node_index]
        branch_ids_local = [network_data.branch_ids[i] for i in branches_at_sub]
        injection_ids_local = [network_data.injection_ids[i] for i in inj_at_sub]
        has_controllable_pst = bool(np.any(network_data.controllable_phase_shift_mask[branches_at_sub]))
        electrical_bus_station = _project_station_to_local_assets(
            station=station,
            branch_ids=branch_ids_local,
            injection_ids=injection_ids_local,
            node_id=node_id,
        )
        try:
            simplified_station, _problems = prepare_for_separation_set(
                station=electrical_bus_station,
                branch_ids=branch_ids_local,
                injection_ids=injection_ids_local,
                close_couplers=close_couplers,
            )
            if (busbar_outage_map := network_data.busbar_outage_map) is not None:
                simplified_busbar_ids = {busbar.grid_model_id for busbar in simplified_station.busbars}
                busbar_outage_map[simplified_station.bus_group_id] = list(simplified_busbar_ids)
            keep_mask.append(True)
        except ValueError as e:
            logger.warning(
                f"Station {electrical_bus_station.bus_group_id}/{electrical_bus_station.name} could not be simplified "
                f"because {e}, removing it from the relevant nodes."
            )
            simplified_station = electrical_bus_station
            busbar_outage_map = network_data.busbar_outage_map
            keep_mask.append(has_controllable_pst)

        stations.append(simplified_station)

    network_data = replace(
        network_data,
        simplified_asset_topology=RuntimeAssetTopology(
            stations=stations,
            circuit_groups=network_data.asset_topology.circuit_groups if network_data.asset_topology is not None else None,
        ),
        busbar_outage_map=busbar_outage_map,
    )
    return remove_relevant_subs(network_data, np.array(keep_mask, dtype=bool))


def compute_separation_set_for_stations(
    network_data: NetworkData,
    clip_hamming_distance: int = 0,
    clip_at_size: int = 100,
) -> NetworkData:
    """Compute the optimal separation set for all stations in the network data

    Parameters
    ----------
    network_data : NetworkData
        The network data to compute the separation set for
    clip_hamming_distance : int, optional
        The maximum hamming distance to consider for the separation set, by default 0
    clip_at_size : int, optional
        The maximum size of the separation set to consider, by default 100

    Returns
    -------
    NetworkData
        The network data with the separation set computed
    """
    actions = 0
    separation_sets_info: list[OptimalSeparationSetInfo] = []
    assert network_data.simplified_asset_topology is not None, (
        "Missing simplified asset-topology stations for separation set preprocessing."
    )
    for station in network_data.simplified_asset_topology.stations:
        separation_set_info = make_optimal_separation_set(station, clip_hamming_distance, clip_at_size)
        separation_sets_info.append(separation_set_info)
        actions += separation_set_info.separation_set.shape[0]
    return replace(
        network_data,
        separation_sets_info=separation_sets_info,
    )


def preprocess(  # noqa: PLR0915
    interface: BackendInterface,
    logging_fn: Optional[Callable[[PreprocessStage, Optional[str]], None]] = None,
    parameters: Optional[PreprocessParameters] = None,
) -> NetworkData:
    """Run the preprocessing pipeline, pulling data from the interface

    Parameters
    ----------
    interface : BackendInterface
        The interface to pull data from
    logging_fn : Callable[[PreprocessStage, Optional[str]], None], optional
        A function to log the progress of the preprocessing, if not given will just log to stdout
    parameters : PreprocessParameters, optional
        The parameters to use for the preprocessing, if not given will use the default parameters
        (see PreprocessParameters for more information)

    Returns
    -------
    NetworkData
        A populated and preprocessed NetworkData object that can be used to extract to jax
    """
    if logging_fn is None:
        logging_fn = empty_status_update_fn
    if parameters is None:
        parameters = PreprocessParameters()

    logging_fn("preprocess_started", None)

    logging_fn("extract_network_data_from_interface", None)
    network_data = extract_network_data_from_interface(interface)

    logging_fn("compute_bridging_branches", None)
    network_data = compute_bridging_branches(network_data)

    logging_fn("filter_relevant_nodes", None)
    network_data = filter_relevant_nodes_branch_count(network_data)
    network_data = filter_relevant_nodes_no_asset_station(network_data)
    network_data = filter_relevant_split_asset_stations(network_data)
    network_data = filter_relevant_nodes_no_double_connections(network_data)

    logging_fn("assert_network_data", None)
    assert_network_data(network_data)

    logging_fn("compute_ptdf_if_not_given", None)
    network_data = compute_ptdf_if_not_given(network_data)

    logging_fn("add_nodal_injections_to_network_data", None)
    network_data = add_nodal_injections_to_network_data(network_data)

    logging_fn("compute_psdf_if_not_given", None)
    network_data = compute_psdf_if_not_given(network_data)

    logging_fn("reduce_node_dimension", None)
    network_data = reduce_node_dimension(network_data)

    logging_fn("combine_phaseshift_and_injection", None)
    network_data = combine_phaseshift_and_injection(network_data)

    logging_fn("exclude_bridges_from_outage_masks", None)
    network_data = exclude_bridges_from_outage_masks(network_data)

    logging_fn("reduce_branch_dimension", None)
    network_data = reduce_branch_dimension(network_data)

    logging_fn("filter_disconnectable_branches_nminus2", None)
    network_data = filter_disconnectable_branches_nminus2(
        network_data, n_processes=parameters.filter_disconnectable_branches_processes
    )

    logging_fn("compute_branch_topology_info", None)
    network_data = compute_branch_topology_info(network_data)

    logging_fn("filter_inactive_injections", None)
    network_data = filter_inactive_injections(network_data)

    logging_fn("compute_injection_topology_info", None)
    network_data = compute_injection_topology_info(network_data)

    logging_fn("convert_multi_outages", None)
    network_data = convert_multi_outages(network_data)

    logging_fn("add_missing_asset_topo_info", None)
    network_data = add_missing_asset_topo_info(network_data)

    logging_fn("simplify_asset_topology", None)
    network_data = simplify_asset_topology(network_data, close_couplers=parameters.asset_topo_close_couplers)

    logging_fn("compute_separation_set", None)
    network_data = compute_separation_set_for_stations(
        network_data,
        clip_hamming_distance=parameters.separation_set_clip_hamming_distance,
        clip_at_size=parameters.separation_set_clip_at_size,
    )

    logging_fn("compute_electrical_actions", None)
    network_data = compute_electrical_actions(
        network_data,
        exclude_bridge_lookup_splits=parameters.action_set_filter_bridge_lookup,
        exclude_bsdf_lodf_splits=parameters.action_set_filter_bsdf_lodf,
        bsdf_lodf_batch_size=parameters.action_set_filter_bsdf_lodf_batch_size,
        clip_to_n_actions=parameters.action_set_clip,
        reassignment_limits=parameters.electrical_reassignment_limits,
    )

    logging_fn("enumerate_station_realizations", None)
    network_data = enumerate_station_realisations(
        network_data, choice_heuristic=parameters.realise_station_busbar_choice_heuristic
    )

    logging_fn("remove_relevant_subs_without_actions", None)
    network_data = remove_relevant_subs_without_actions(network_data)

    logging_fn("enumerate_injection_actions", None)
    network_data = compute_injection_actions(network_data)

    logging_fn("process_injection_outages", None)
    network_data = process_injection_outages(network_data)

    logging_fn("add_bus_b_columns_to_ptdf", None)
    network_data = add_bus_b_columns_to_ptdf(network_data)
    if parameters.preprocess_bb_outages:
        logging_fn("preprocess_bb_outage", None)
        network_data = preprocess_bb_outages(network_data)
    else:
        logging_fn("preprocess_bb_outage", "BB-Outages disabled, skipping preprocessing step")
        network_data = disable_busbar_outage_contingencies(network_data)

    logging_fn("preprocess_done", None)
    return network_data
