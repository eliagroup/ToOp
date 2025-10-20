"""Contains functions to enumerate branch and injection outages for unsplit busbar outages in the given network."""

from dataclasses import replace

import logbook
import networkx as nx
import numpy as np
from beartype.typing import Optional, Union
from jaxtyping import Array, Bool, Float, Int
from toop_engine_dc_solver.preprocess.network_data import (
    NetworkData,
    OutageData,
    get_relevant_stations,
)
from toop_engine_interfaces.asset_topology import Station, SwitchableAsset
from toop_engine_interfaces.asset_topology_helpers import find_station_by_id, get_connected_assets

logger = logbook.Logger(__name__)


def get_total_injection_along_stub_branch(
    stub_branch_index: int, current_nodal_index: int, network_data: NetworkData
) -> Float[np.ndarray, " n_timestep"]:
    """Calculate the total injection along a stub branch in the network.

    This function computes the total power injection at the busbar connected to a specified stub branch
    for all time steps. It traverses the network starting from the node connected to the stub branch
    and sums the injections at each node it visits.

    Note: The injection at the current node is not included in the sum.

    Parameters
    ----------
    stub_branch_index : int
        The index of the stub branch in the network.
    current_nodal_index : int
        The nodal index of the current busbar in the network.
    network_data : NetworkData
        An object containing the network data, including nodal injections, from_nodes, and to_nodes.

    Returns
    -------
    Float[np.ndarray, " n_timestep "]
        A numpy array containing the total injection at the busbar connected to the stub branch for all time steps.
    """
    from_node = network_data.from_nodes[stub_branch_index]
    to_node = network_data.to_nodes[stub_branch_index]

    if from_node == current_nodal_index:
        other_node = to_node
    else:
        other_node = from_node

    # Sum the injections at the current node for all time steps
    total_injection = np.zeros((network_data.nodal_injection.shape[0],), float)
    nodes_to_visit = [other_node]
    visited_nodes = set()
    visited_nodes.add(current_nodal_index)
    while nodes_to_visit:
        node = nodes_to_visit.pop()
        if node in visited_nodes:
            continue
        visited_nodes.add(node)

        # Sum the injections at the current node
        total_injection += network_data.nodal_injection[:, node]

        # Find all branches connected to this node
        connected_branches = np.unique(np.where((network_data.from_nodes == node) | (network_data.to_nodes == node))[0])

        for branch in connected_branches:
            from_node = network_data.from_nodes[branch]
            to_node = network_data.to_nodes[branch]
            next_node = to_node if from_node == node else from_node

            if next_node not in visited_nodes:
                nodes_to_visit.append(next_node)

    return total_injection


def get_busbar_index(station: Station, busbar_id: str) -> int:
    """Get the index of a busbar within a station by its grid_model_id of busbar (busbar_id).

    Parameters
    ----------
    station : Station
        The station object containing busbars.
    busbar_id : str
        The grid_model_id of the busbar to find.

    Returns
    -------
    int
        The index of the busbar within the station's busbars list.

    Raises
    ------
    ValueError
        If the busbar with the specified ID is not found in the station.
    """
    for index, busbar in enumerate(station.busbars):
        if busbar_id == busbar.grid_model_id:
            return index
    raise ValueError(f"Busbar {busbar_id} not found in station {station.grid_model_id}")


def extract_outage_index_injection_from_asset(
    asset: SwitchableAsset,
    network: NetworkData,
    nodal_index_for_busbar: int,
    stub_power_map: Optional[dict[str, Float[np.ndarray, " n_timestep"]]],
) -> tuple[Optional[int], Float[np.ndarray, " n_timestep"]]:
    """Extract the outage index and nodal injection from a switchable asset.

    Processes a switchable asset connected to the busbar and returns the index of the brannch to be outaged
    (None if the asset is an injection), nodal injection to be outaged (if the asset is an injection,
    [0]*n_timestep otherwise) and the index of the node to which the injection/branch is connected.

    Parameters
    ----------
    asset : SwitchableAsset
        The asset for which the outage index and nodal injection are to be extracted.
    network : NetworkData
        The network data containing information about branches, injections, and nodal injections.
    nodal_index_for_busbar :  int
        Nodal index of the busbar.
    stub_power_map : dict[str, Float[np.ndarray, " n_timestep"]]
        A dictionary mapping stub branch indices to their total injection values. This is used to avoid recalculating
        the total injection along a stub branch if it has already been calculated.

    Returns
    -------
    Optional[int]
        The index of the branch to be taken out of service, or None if the asset is an injection.
    Float[np.ndarray, " n_timestep"]
        The nodal injection values corresponding to the outage, with shape (n_timestep,).
    """
    nodal_injection_to_outage: Float[np.ndarray, " n_timestep"] = np.zeros(network.nodal_injection.shape[0], float)
    branch_index_to_outage = None

    if asset.in_service:
        if asset.is_branch():
            # Branch is a line or a trafo
            try:
                branch_index = network.branch_ids.index(asset.grid_model_id)
            except ValueError as e:
                raise ValueError(f"Branch {asset.grid_model_id} not found in network data.") from e
            if not network.bridging_branch_mask[branch_index]:
                branch_index_to_outage = branch_index
            else:
                # the branch is a stub branch and can't be removed.
                key = str(branch_index) + "-" + str(nodal_index_for_busbar)
                if stub_power_map is not None and key in stub_power_map:
                    # If the branch is a stub branch, we need to get the total injection along the stub branch.
                    # Check if the stub_power_map already has the key.
                    nodal_injection_to_outage += stub_power_map[key]
                else:
                    # If it's a cache miss, perform the calculation and store it in the stub_power_map.
                    nodal_injection_to_outage += get_total_injection_along_stub_branch(
                        branch_index, nodal_index_for_busbar, network
                    )
                    stub_power_map[key] = nodal_injection_to_outage
        else:
            # Branch is an injection
            try:
                injection_index = network.injection_ids.index(asset.grid_model_id)
                nodal_injection_to_outage += network.mw_injections[:, injection_index]
            except ValueError:
                logger.warning(
                    f"Asset {asset.grid_model_id} is not a valid injection. Might have been removed.",
                )

    return branch_index_to_outage, nodal_injection_to_outage


def get_busbar_map_adjacent_branches(network_data: NetworkData) -> Bool[np.ndarray, " n_branch"]:
    """Get a boolean mask indicating which branches are connected to a station that has a busbar outage.

    This does not check if the branches are actually part of the outage, just if they are at the station

    Parameters
    ----------
    network_data : NetworkData
        The network data object containing asset topology and busbar outage information.

    Returns
    -------
    Bool[np.ndarray, " n_branch"]
        A boolean mask indicating which branches are connected to a station with a busbar outage.
        The mask is of the same length as the number of branches in the network data. True if a branch
        connects to a station with a busbar outage, False otherwise.
        Will be all False if no busbar_outage_map is defined.
    """
    bb_outage_asset_indices = set()
    busbar_outage_branch_mask = np.zeros(len(network_data.branch_ids), dtype=bool)
    if network_data.busbar_outage_map is not None:
        # Gather all branches connected to a station with a busbar outage
        for station_id in network_data.busbar_outage_map.keys():
            # Find the asset topo station id
            station = find_station_by_id(network_data.asset_topology.stations, station_id)
            for asset in station.assets:
                bb_outage_asset_indices.add(asset.grid_model_id)

        busbar_outage_branch_mask = np.array([(id in bb_outage_asset_indices) for id in network_data.branch_ids])
    return busbar_outage_branch_mask


def get_busbar_branches_map(station: Station, network_data: NetworkData) -> dict[str, list[int]]:
    """Map each busbar in a station to the list of branch indices connected to it.

    These branch_indices index into network_data.branch_ids.

    Parameters
    ----------
    station : Station
        The station object containing busbars and their associated data.
    network_data : NetworkData
        The network data object containing branch IDs and other network-related information.

    Returns
    -------
    dict[str, list[int]]
        A dictionary where the keys are the grid model IDs of the busbars, and the values
        are lists of indices of the branches connected to each busbar. Only branches that
        are in service are included in the mapping.

    Notes
    -----
    - The function filters out assets that are not branches or are not in service.
    - The branch indices are determined based on their position in the `network_data.branch_ids` list.
    """
    busbar_branches_map = {}
    for bb_index, bb in enumerate(station.busbars):
        connected_assets = get_connected_assets(station, bb_index)
        connected_branches = [asset.grid_model_id for asset in connected_assets if asset.is_branch() and asset.in_service]
        connected_branches = [network_data.branch_ids.index(branch_id) for branch_id in connected_branches]
        busbar_branches_map[bb.grid_model_id] = connected_branches
    return busbar_branches_map


def get_phy_bb_nodal_index(
    busbar_index: Union[int, np.int64],
    possible_node_indices: tuple[Union[int, np.int64], Union[int, np.int64]],
    network_data: NetworkData,
    sub_index: Union[int, np.int64],
    branch_action_combi_index: Union[int, Int[Array, " "] | np.integer],
) -> Union[int, np.int64]:
    """Determine the nodal_index of the physical busbar represented by busbar_index.

    This is determined according to the branch action such that there is minimum
    switching required to realise the branch_action.

    Parameters
    ----------
    busbar_index: int
        The index of the physical busbar in the station. This indices into station.busbars
    possible_node_indices : tuple[Union[int, np.int64], Union[int, np.int64]]
        A tuple of possible nodal indices corresponding to the busbar. The length of the list is
        equal to two and contains the nodal indices corresponding to busbar_a and busbar_b.
    network_data : NetworkData
        The network data object.
    sub_index : int
        The index of the rel_station. This indexes into network_data.relevant_nodes
    branch_action_combi_index : int
        The index of the branch action for the station of which the busbar represented by
        busbar_index is part of.
        This indices into the 2nd dimension of network_data.branch_action_set

    Returns
    -------
    int
        The nodal index of the physical busbar corresponding to the branch_action_combi_index.
        The nodal index of the busbar would depend on the branch action combination. If the
        branch action combination is such that the physical busbar is grouped as busbar_a,
        then the nodal index of busbar_a is returned. Otherwise, the nodal index of busbar_b
        is returned.
    """
    local_busbar_a_mappings = network_data.busbar_a_mappings[sub_index]
    if busbar_index in local_busbar_a_mappings[branch_action_combi_index]:
        return possible_node_indices[0]
    return possible_node_indices[1]


def extract_busbar_outage_data(
    station: Station,
    busbar_id: str,
    network: NetworkData,
    stub_power_map: dict[str, Float[np.ndarray, " n_timestep"]],
    branch_action_combi_index: Optional[Union[Int[Array, " "] | int | np.integer]] = None,
) -> OutageData:
    """Extract data about the branch indices, nodal injection and index of the busbar that has to be outaged.

    Parameters
    ----------
    station : Station
        The station object containing busbars and assets.
    busbar_id : str
        The identifier of the busbar to be outaged.
    network : NetworkData
        The network data object containing nodal injections and node IDs.
    stub_power_map : dict[str, Float[np.ndarray, " n_timestep"]]
        A dictionary mapping stub branch indices to their total injection values. This is used to avoid recalculating
        the total injection along a stub branch if it has already been calculated.
    branch_action_combi_index : int, optional
        The index of the branch action combination. If None, the function assumes that the station is not a relevant
        substation. The branch_action_combi_index indices into the network_data.branch_action_set for the given station.
        This is required to calculate the nodal_index of any relevant physical busbar. If this method is used to
        calculate the outage data for non-relevant substations, then this parameter can be None.

    Returns
    -------
    OutageData
        A namedtuple containing:
        - branch_indices: A list of indices of the connected branches to be outaged.
        - nodal_injection: An array of nodal injections to be outaged.
        - node_index: The index of the node to be outaged.
    """
    busbar_index = get_busbar_index(station, busbar_id)

    assert station.busbars[busbar_index].grid_model_id == busbar_id, "Busbar index is not correct."
    assert station.busbars[busbar_index].in_service, f"Busbar {busbar_id} is not in service. Cannot outage it."

    connected_branches_to_outage = []
    connected_assets = get_connected_assets(station, busbar_index)
    node_indices_to_outage = np.where(np.array(network.node_ids) == station.grid_model_id)[0].tolist()

    # Determine the nodal_index of the physical busbar.
    node_index_to_outage = None
    node_indices_to_outage = tuple(sorted(node_indices_to_outage))

    if len(node_indices_to_outage) == 1:
        # The station is not a relevant substation. In this case, node_index_to_outage will be a list of length 1
        node_index_to_outage = node_indices_to_outage[0]
    else:
        # The station is a relevant substation. In this case, the nodal_index of the physical busbar would depend on the
        # branch action combination.
        rel_sub_index = np.argmax(network.relevant_nodes == network.node_ids.index(station.grid_model_id)).item()
        assert network.busbar_a_mappings is not None, "busbar_a_mappings is not defined."
        assert branch_action_combi_index is not None, "branch_action_combi_index is not defined."
        node_index_to_outage = get_phy_bb_nodal_index(
            busbar_index, node_indices_to_outage, network, rel_sub_index, branch_action_combi_index
        )

    nodal_injection_to_outage = np.zeros(network.nodal_injection.shape[0], float)
    for asset in connected_assets:
        branch_index, delta_p = extract_outage_index_injection_from_asset(
            asset,
            network,
            node_index_to_outage,
            stub_power_map=stub_power_map,
        )
        if branch_index is not None:
            connected_branches_to_outage.append(branch_index)
        nodal_injection_to_outage += delta_p

    return OutageData(
        branch_indices=list(set(connected_branches_to_outage)),
        nodal_injection=nodal_injection_to_outage,
        node_index=node_index_to_outage,
    )


def update_network_data_with_non_rel_bb_outages(
    network: NetworkData, outage_station_busbars_map: dict[str, list[str]]
) -> NetworkData:
    """Update the network_data with outage data of non-relevant busbars.

    Parameters
    ----------
    network : NetworkData
        The network data containing asset topology and outage information.
    outage_station_busbars_map : dict[str, list[str]]
        A dictionary mapping station grid model IDs to lists of busbar IDs to be outaged.
        If the mapping is empty. Return the original network data

    Returns
    -------
    NetworkData
        The network_data updated with non-rel busabar outage data. This includes the branch indices
        that have to be outaged, the nodal injection that has to be outaged and the nodal index
        of the busbar that has to be outaged corresponding to each non-rel busbar outage.
    """
    if len(outage_station_busbars_map) == 0:
        n_busbar_outages = 0
        return replace(
            network,
            non_rel_bb_outage_br_indices=[],
            non_rel_bb_outage_deltap=np.zeros((n_busbar_outages, network.nodal_injection.shape[0]), float),
            non_rel_bb_outage_nodal_indices=np.zeros((n_busbar_outages), int),
        )
    asset_topology = network.asset_topology

    branch_indices = []
    delta_p = []
    nodal_indices = []

    for station in asset_topology.stations:
        if station.grid_model_id in outage_station_busbars_map:
            for bb_id in outage_station_busbars_map[station.grid_model_id]:
                (branch_indices_to_outage, nodal_injection_to_outage, node_index_to_outage) = extract_busbar_outage_data(
                    station, bb_id, network, stub_power_map={}, branch_action_combi_index=None
                )
                branch_indices.append(list(set(branch_indices_to_outage)))
                delta_p.append(nodal_injection_to_outage)
                nodal_indices.append(node_index_to_outage)

    return replace(
        network,
        non_rel_bb_outage_br_indices=branch_indices,
        non_rel_bb_outage_deltap=np.array(delta_p),
        non_rel_bb_outage_nodal_indices=np.array(nodal_indices),
    )


def get_branch_injection_outages_for_rel_subs(
    network_data: NetworkData,
    rel_station_busbars_map: Optional[
        dict[
            str,
            list[str],
        ]
    ],
    ignore_injection_actions: bool = True,
) -> tuple[
    list[Optional[list[list[Optional[list[int]]]]]],
    list[Optional[list[list[Optional[Union[np.ndarray, list]]]]]],
    list[Optional[list[list[Optional[int]]]]],
]:
    """Get the branch and injection outages for split substations in the network.

    Parameters
    ----------
    network_data : NetworkData
        The network data containing asset topology and outage information.
    rel_station_busbars_map : Optional[dict[str, list[str]]]
        A dictionary mapping station grid model IDs to lists of busbar IDs to be outaged. If None, all busbars are
        considered.
    ignore_injection_actions : bool, optional
        If True, injection actions will be ignored when determining outages. Default is True.

    Returns
    -------
    list[Optional[list[list[Optional[list[int]]]]]]
        This correpsonds to the branch indices that have to be outaged for the relevant
        busbars due to branch_actions.

        The outer list is of length of n_relevant sub station. The next inner list is of length of maximum number of
        branch_action combinations for the substation. Each element of the list has a list of length equal
        to the number of busbars. The next inner list contains the list of branch indices that have to be outaged.
    list[Optional[list[list[Optional[np.ndarray]]]]]
        This corresponds to the delta p that has to be outaged for the relevant.

        The outer list is of length of n_relevant sub station. The next inner list is of length of maximum number of
        branch_action combinations for the substation. Each element of the list has a list of length equal
        to the number of busbars. The next numpy array if of length equal to the number of timesteps and contains the delta p
        for each timestep.
    list[Optional[list[list[Optional[int]]]]],
        The outer list is of length of n_relevant sub station. The next inner list is of length of maximum number of
        branch_action combinations for the substation. Each element of the list has a list of length equal
        to the number of busbars. The next inner list contains the nodal indices where the deltap have to be applied.

    Note
    ----
    If a particular rel_sub is not to be outaged then the first inner list will be empty. Likewise, if a particular busbar
    of a rel_sub is not to be outaged, then the corresponding entry in the most inner list will be None.

    """
    if not ignore_injection_actions:
        raise NotImplementedError("Injection actions are not supported yet.")

    outage_stations = list(rel_station_busbars_map.keys()) if rel_station_busbars_map is not None else None
    modified_stations_br = get_modified_stations(
        network_data=network_data,
        stations_to_outage=outage_stations,
    )
    busbars_to_outage = set(
        [bb for bbs in rel_station_busbars_map.values() for bb in bbs] if rel_station_busbars_map is not None else None
    )
    outage_data_branch_actions = get_all_rel_bb_outage_data(modified_stations_br, network_data, busbars_to_outage)
    outage_data_branch_indices = [
        [
            [outage_data.branch_indices if outage_data is not None else [] for outage_data in busbar_outages]
            for busbar_outages in station_outages
        ]
        for station_outages in outage_data_branch_actions
    ]

    outage_data_deltap = [
        [
            [outage_data.nodal_injection if outage_data is not None else [] for outage_data in busbar_outages]
            for busbar_outages in station_outages
        ]
        for station_outages in outage_data_branch_actions
    ]

    outage_data_nodal_index = [
        [
            [outage_data.node_index if outage_data is not None else None for outage_data in busbar_outages]
            for busbar_outages in station_outages
        ]
        for station_outages in outage_data_branch_actions
    ]

    return outage_data_branch_indices, outage_data_deltap, outage_data_nodal_index


def get_modified_stations(
    network_data: NetworkData,
    stations_to_outage: Optional[list[str]],
) -> list[Optional[list[Station]]]:
    """Get the modified stations after applying branch actions.

    Note: We don't consider injection actions here.

    Parameters
    ----------
    network_data : NetworkData
        The network data containing asset topology and outage information.
    stations_to_outage : Optional[list[str]]
        A list of station grid model IDs to consider for outage. If None, all relevant stations are considered.

    Returns
    -------
    list[Optional[list[Station]]]
        Modified stations for each relevant substation due to branch actions. The outer list is of length of n_relevant
        sub station. The next inner list equals the number of possible branch action combinations for the substation.
        Note that if the station is not to be outaged, then the inner list will be empty.
    """
    if stations_to_outage is None:
        # In this case, we are interested in calculating bb_outage data for all the busbars
        # of the relevant substations. Hence, we need to return the realised_stations
        return network_data.realised_stations

    # Modify the network_data.realised_stations in a way such that if the station is not to be outaged,
    # then the inner list will be empty
    realised_stations = []
    for station_combis in network_data.realised_stations:
        if stations_to_outage is not None and station_combis[0].grid_model_id not in stations_to_outage:
            realised_stations.append([])
        else:
            realised_stations.append(station_combis)
    return realised_stations


def get_all_rel_bb_outage_data(
    modified_stations: list[Optional[list[Station]]],
    network_data: NetworkData,
    busbars_to_outage: Optional[set[str]] = None,
) -> list[Optional[list[list[Optional[OutageData]]]]]:
    """Get all outage data for each relevant substation.

    Parameters
    ----------
    modified_stations : list[list[Station]]
        The outer list is of length of n_relevant sub station. Each relevant substation
        can have different combinations of branch actions. The inner list contains the modified
        Station as per the branch action combinations and is of length of maximum number of
        combinations for the substation.

        For example, if a substation has 5 branch_actions , then the inner list will have
        5 Station objects. These Station objects have the updated asset_switching_table as per the branch and injection
        actions.

    network_data : NetworkData
        The network data containing asset topology and outage information.

    busbars_to_outage : Optional[set[str]]
        A set of busbar grid model IDs to consider for outage. If None, all busbars are considered.

    Returns
    -------
    list[Optional[list[list[OutageData]]]]
        The outer list is of length of n_relevant sub station. The next inner list is of length of maximum number of
        branch_action combinations for the substation. Each element of the list has a list of length equal
        to the number of physical busbars. The next inner list contains an OutageData namedtuple with branch indices,
        nodal injection, and node index. If the station is not to be outaged, then the first inner list will be empty.
        Likewise, if a particular busbar of a station is not to be outaged, then the
        corresponding entry in the second inner list will be None.

    """
    all_outage_data = []
    for local_station_combis in modified_stations:
        station_outages = []
        stub_power_map = {}
        for branch_action_combi_index, station_combi in enumerate(local_station_combis):
            busbar_outages = []

            for busbar in station_combi.busbars:
                if busbars_to_outage is not None and busbar.grid_model_id not in busbars_to_outage:
                    busbar_outages.append(None)
                    continue
                if busbar.in_service:
                    outage_data = extract_busbar_outage_data(
                        station_combi,
                        busbar.grid_model_id,
                        network_data,
                        stub_power_map=stub_power_map,
                        branch_action_combi_index=branch_action_combi_index,
                    )
                    busbar_outages.append(outage_data)

            station_outages.append(busbar_outages)

        all_outage_data.append(station_outages)

    return all_outage_data


def update_network_data_with_rel_bb_outages(
    network_data: NetworkData, rel_station_busbars_map: dict[str, list[str]], ignore_injection_actions: bool = False
) -> NetworkData:
    """Update the network data with busbar outages for the relevant substations.

    Parameters
    ----------
    network_data : NetworkData
        The network data object containing the current state of the network.
    rel_station_busbars_map : dict[str, list[str]]
        A mapping of relevant station names to their corresponding busbars.
    ignore_injection_actions : bool, optional
        If True, injection actions will be ignored when determining outages. Default is False.

    Returns
    -------
    NetworkData
        A new instance of the network data with updated rel_bb_outage data.

    Notes
    -----
    - For relevant substations, whether a busbar is an articulation node depends on the branch action combinations.
    - articulation busbars are identified and stored to ensure they are not outaged.
    """
    (rel_bb_outage_br_indices, rel_bb_outage_deltap, rel_bb_outage_nodal_indices) = (
        get_branch_injection_outages_for_rel_subs(network_data, rel_station_busbars_map, ignore_injection_actions)
    )

    relevant_subs = get_relevant_stations(network_data=network_data)
    rel_bb_articulation_nodes = get_rel_articulation_nodes(relevant_subs, network_data.busbar_a_mappings)
    return replace(
        network_data,
        rel_bb_outage_br_indices=rel_bb_outage_br_indices,
        rel_bb_outage_deltap=rel_bb_outage_deltap,
        rel_bb_outage_nodal_indices=rel_bb_outage_nodal_indices,
        rel_bb_articulation_nodes=rel_bb_articulation_nodes,
    )


def get_rel_non_rel_sub_bb_maps(
    network_data: NetworkData, outage_station_busbars_map: dict[str, list[str]]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Separate outage_station_busbars map into relevant and non-relevant categories based on the network data.

    Parameters
    ----------
    network_data : NetworkData
        The network data containing node IDs and a mask indicating relevant nodes.
    outage_station_busbars_map : dict[str, list[str]]
        A mapping of station IDs to their respective busbars.

    Returns
    -------
    tuple[dict[str, list[str]], dict[str, list[str]]]
        A tuple containing two dictionaries:
        - The first dictionary maps relevant station IDs to their busbars.
        - The second dictionary maps non-relevant station IDs to their busbars.

    Notes
    -----
    A station is considered relevant if its ID is present in the `relevent_node_ids` list,
    which is derived from the `network_data.relevant_node_mask`.
    """
    non_rel_station_busbars_map = {}
    rel_station_busbars_map = {}
    relevent_node_ids = [node for node, mask in zip(network_data.node_ids, network_data.relevant_node_mask) if mask]
    for station_id, busbars in outage_station_busbars_map.items():
        if station_id in relevent_node_ids:
            rel_station_busbars_map[station_id] = busbars
        else:
            non_rel_station_busbars_map[station_id] = busbars

    return rel_station_busbars_map, non_rel_station_busbars_map


def get_articulation_nodes(
    nodes: list[Union[np.int64, int]], edges: list[tuple[Union[np.int64, int], Union[np.int64, int]]]
) -> list[Union[np.int64, int]]:
    """Identify articulation nodes in a graph.

    The articulation nodes are the nodes which, when deleted, would split the graph into two components.
    This function returns the list of articulation nodes for the given graph.

    Parameters
    ----------
    nodes : list[Union[np.int64, int]]
        List of nodes in the graph.
    edges : list[tuple[Union[np.int64, int], Union[np.int64, int]]]
        List of edges in the graph, where each edge is represented as a tuple of two nodes.

    Returns
    -------
    list[Union[np.int64, int]]
        List of articulation nodes that, when removed, would split the graph into two components.
    """
    if len(edges) <= 1:
        # ladder configuration is possible only with two or more edges
        return []
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return list(nx.articulation_points(graph))


def get_non_rel_articulation_nodes(
    non_rel_busbar_outage_map: dict[str, list[str]], network_data: NetworkData
) -> dict[str, list[str]]:
    """Filter out the busbars which serve as articulation node from the non-rel busbar outage map.

    An articulation busbar is the busbar which when outaged leads to splitting the station, we skip the
    outaging of such busbars. These busbars can also be seen as bridge busbars, and their entries are deleted
    from the non_rel_busbar_outage_map.

    Parameters
    ----------
    non_rel_busbar_outage_map : dict[str, list[str]]
        A dictionary mapping station grid_model_ids to lists of non-rel busbar grid_model_ids.
    network_data : NetworkData
        An object containing the network data, including asset topology.

    Returns
    -------
    dict[str, list[str]]
        The updated non_rel_busbar_outage_map with those busbars removed which serve as articulation node.
    , list[str]],network_data: NetworkData) -> dict[str, list[str]]:
    """
    asset_topology = network_data.simplified_asset_topology
    for station in asset_topology.stations:
        if station.grid_model_id in non_rel_busbar_outage_map:
            busbar_intid_index_mapper = {busbar.int_id: index for index, busbar in enumerate(station.busbars)}
            nodes = [busbar_index for busbar_index in range(len(station.busbars))]
            edges = [
                (
                    busbar_intid_index_mapper[coupler.busbar_from_id],
                    busbar_intid_index_mapper[coupler.busbar_to_id],
                )
                for coupler in station.couplers
                if not coupler.open
            ]
            articulation_nodes = get_articulation_nodes(nodes, edges)
            articulation_busbar_ids = [station.busbars[node].grid_model_id for node in articulation_nodes]
            # remove the busbar_ids which are articulation
            if articulation_busbar_ids:
                non_rel_busbar_outage_map[station.grid_model_id] = [
                    bb for bb in non_rel_busbar_outage_map[station.grid_model_id] if bb not in articulation_busbar_ids
                ]

    return non_rel_busbar_outage_map


def get_rel_articulation_nodes(
    rel_stations: list[Station], busbar_a_mappings: list[list[list[int]]]
) -> list[list[list[int]]]:
    """Determine the busbars that serve as articulation nodes for relevant substations in the network.

    These busbars are like bridge busbars which, when outaged, would split the station into two components.
    For relevant substations, the function considers the branch action combinations to determine these bridge busbars.

    These indices will be converted into masks in convert_to_jax module to filter out the busbars
    which cause the station to split into two components due to their outage

    Parameters
    ----------
    rel_stations : list[Station]
        A list of relevant substations in the network.
    busbar_a_mappings : list[list[list[int]]]
        A list of busbar_a mappings for each relevant substation. The outer list is of length equal to the number
        of relevant substations. The next inner list is of length equal to the number of branch_action combinations
        for the particular station. The next inner list contains the list of busbar_a indices for the particular
        branch_action combination.
        The busbar_a indices are the indices of the busbars in the station's busbars list that are mapped as
        electrical busbar_A.

    Returns
    -------
    all_station_articulation_nodes : list[list[list[int]]]
        A list of articulation busbar combinations for each relevant substation.
        The outer list is of length equal to the number of relevant substations.
        The next inner list is of length equal to the number of branch_action combinations
        for the particular station. The next inner list contains the list of articulation busbar indices
        for the particular branch_action combination.

    """
    all_station_articulation_nodes = []
    for rel_sub_index, station in enumerate(rel_stations):
        busbar_intid_index_mapper = {busbar.int_id: index for index, busbar in enumerate(station.busbars)}
        coupler_setpoints = [
            (
                busbar_intid_index_mapper[c.busbar_from_id],
                busbar_intid_index_mapper[c.busbar_to_id],
            )
            for c in station.couplers
            if not c.open
        ]
        local_bus_a_mappings = busbar_a_mappings[rel_sub_index]
        local_articulation_node_combis = []
        cache = {}
        # Iterate through the possible busbar_a and busbar_b mappings
        for bus_a_mapping in local_bus_a_mappings:
            bus_b_mapping = list(set(range(len(station.busbars))) - set(bus_a_mapping))

            cache_key_a = tuple(sorted(bus_a_mapping))
            cache_key_b = tuple(sorted(bus_b_mapping))
            if cache_key_a in cache:
                articulation_nodes_a = cache[cache_key_a]
            else:
                edges_a = [
                    (coupler_setpoint[0], coupler_setpoint[1])
                    for coupler_setpoint in coupler_setpoints
                    if coupler_setpoint[0] not in bus_b_mapping and coupler_setpoint[1] not in bus_b_mapping
                ]
                articulation_nodes_a = get_articulation_nodes(bus_a_mapping, edges_a)
                cache[cache_key_a] = articulation_nodes_a
            if cache_key_b in cache:
                articulation_nodes_b = cache[cache_key_b]
            else:
                edges_b = [
                    (coupler_setpoint[0], coupler_setpoint[1])
                    for coupler_setpoint in coupler_setpoints
                    if coupler_setpoint[0] not in bus_a_mapping and coupler_setpoint[1] not in bus_a_mapping
                ]
                articulation_nodes_b = get_articulation_nodes(bus_b_mapping, edges_b)
                cache[cache_key_b] = articulation_nodes_b

            # Combine the articulation nodes for bus_a and bus_b
            articulation_nodes = list(set(articulation_nodes_a + articulation_nodes_b))
            local_articulation_node_combis.append(articulation_nodes)
        all_station_articulation_nodes.append(local_articulation_node_combis)

    return all_station_articulation_nodes


def add_default_bb_outage_map(network_data: NetworkData) -> NetworkData:
    """Add a default busbar outage map to the network data if it is not already present.

    By default, all relevant substations will be included in the busbar outage map.

    Parameters
    ----------
    network_data : NetworkData
        The network data object with relevant substations

    Returns
    -------
    NetworkData
        A new instance of the network data with the busbar outage map added.
    """
    if network_data.busbar_outage_map is not None:
        return network_data

    busbar_outage_map = {}
    rel_subs = get_relevant_stations(network_data)
    for sub in rel_subs:
        busbar_outage_map[sub.grid_model_id] = [bb.grid_model_id for bb in sub.busbars]

    return replace(
        network_data,
        busbar_outage_map=busbar_outage_map,
    )


def preprocess_bb_outages(
    network_data: NetworkData,
    ignore_injection_actions: bool = True,
) -> NetworkData:
    """Preprocess busbar outages in the network data.

    This function updates the network data with busbar outage data of busbars from relevant and non-relevant substations.
    It separates the busbar outages into relevant and non-relevant categories, processes the non-relevant
    busbar outages, and updates the network data with the relevant busbar outages.

    Parameters
    ----------
    network_data : NetworkData
        The network data object containing the current state of the network.
    ignore_injection_actions : bool, optional
        If True, injection actions will be ignored when determining outages. Default is True.

    Returns
    -------
    NetworkData
        A new instance of the network data with updated busbar outage data.

    Raises
    ------
    AssertionError
        If the network data contains injection actions and ignore_injection_actions is False.
    NotImplementedError
        If injection actions are not supported yet.
    """
    network_data = add_default_bb_outage_map(network_data)

    rel_station_busbars_map, non_rel_station_busbars_map = get_rel_non_rel_sub_bb_maps(
        network_data, network_data.busbar_outage_map
    )
    if not ignore_injection_actions:
        assert network_data.injection_action_set is None, "Injection actions are not supported yet."
        raise NotImplementedError("Injection actions are not supported yet.")

    # For non-rel stations, remove the busbar from the station-busbar map if the busbar is articulation node
    non_rel_station_busbars_map = get_non_rel_articulation_nodes(non_rel_station_busbars_map, network_data)
    network_data = update_network_data_with_non_rel_bb_outages(network_data, non_rel_station_busbars_map)

    network_data = update_network_data_with_rel_bb_outages(network_data, rel_station_busbars_map, ignore_injection_actions)

    return network_data
