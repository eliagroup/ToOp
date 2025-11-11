"""The network data class that holds the necessary information about the grid."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from beartype.typing import NamedTuple, Optional, Sequence, Union
from jaxtyping import Bool, Float, Int
from toop_engine_interfaces.asset_topology import Station, Topology
from toop_engine_interfaces.backend import BackendInterface
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, LoadflowParameters, Nminus1Definition
from toop_engine_interfaces.stored_action_set import ActionSet, PSTRange


class OutageData(NamedTuple):
    """The outage data class that holds the necessary information about the grid."""

    branch_indices: list[int]
    """The indices of the branches that are outaged"""

    nodal_injection: Float[np.ndarray, " n_timesteps"]
    """The injected netto power at each node in the grid for all timesteps"""

    node_index: int
    """The index of the node that is outaged"""


@dataclass(frozen=True)
class NetworkData:
    """The NetworkData class holds all the information about the grid.

    This is a central class in the DC solver. It holds all the information about the grid
    that is needed run a loadflow calculation. The network data is initially populated from the
    backend interface with "raw" node-branch data.
    Further information is then going to be extracted through the individual pre-processing steps.
    """

    ptdf: Optional[Float[np.ndarray, " n_branch n_node"]]
    """The PTDF matrix of the grid"""

    psdf: Optional[Float[np.ndarray, " n_branch n_phase_shifters"]]
    """The PSDF matrix of the grid"""

    slack: int
    """The index of the slack node"""

    relevant_node_mask: Bool[np.ndarray, " n_node"]
    """The relevant nodes of the grid, as boolean mask"""

    ac_dc_mismatch: Float[np.ndarray, " n_timestep n_branch"]
    """The AC-DC mismatch for each branch and timestep."""

    max_mw_flows: Float[np.ndarray, " n_timestep n_branch"]
    """The maximum flow per branch"""

    max_mw_flows_n_1: Float[np.ndarray, " n_timestep n_branch"]
    """The maximum flow per branch in the N-1 case"""

    overload_weights: Float[np.ndarray, " n_branch"]
    """The weights of the branches for the overload calculation"""

    n0_n1_max_diff_factors: Float[np.ndarray, " n_branch"]
    """The factors for the N-0 to N-1 max flow difference"""

    cross_coupler_limits: Float[np.ndarray, " n_relevant_nodes"]
    """The cross-coupler limits for the relevant nodes"""

    susceptances: Float[np.ndarray, " n_branch"]
    """The susceptances of the branches"""

    from_nodes: Int[np.ndarray, " n_branch"]
    """The from nodes of the branches"""

    to_nodes: Int[np.ndarray, " n_branch"]
    """The to nodes of the branches"""

    shift_angles: Float[np.ndarray, " n_timestep n_branch"]
    """The shift angles of the branches (only relevant for trafos)"""

    phase_shift_mask: Bool[np.ndarray, " n_branch"]
    """Which branch can have a phase shift"""

    controllable_phase_shift_mask: Bool[np.ndarray, " n_branch"]
    """Which branch is a phase shifter that is controllable. This must be a subset of phase_shift_mask"""

    phase_shift_taps: list[Float[np.ndarray, " n_tap_positions"]]
    """The shift angles of the controllable PSTs. The outer list has as many entries as there are controllable PSTs (see
    controllable_phase_shift_mask). The inner np array has as many entries as there are taps for the given PST with each
    value representing the angle shift for the given tap position. The taps are ordered smallest to largest angle shift."""

    monitored_branch_mask: Bool[np.ndarray, " n_branch"]
    """Which branch is monitored"""

    disconnectable_branch_mask: Bool[np.ndarray, " n_branch"]
    """Which branch can be disconnected"""

    outaged_branch_mask: Bool[np.ndarray, " n_branch"]
    """Which branch should be outaged as part of the N-1 computation"""

    outaged_injection_mask: Bool[np.ndarray, " n_injection"]
    """Which injection should be outaged as part of the N-1 computation"""

    multi_outage_branch_mask: Bool[np.ndarray, " n_multi_outages n_branch"]
    """Which sets of branches should be outaged as part of the multi-outage computation"""

    multi_outage_node_mask: Bool[np.ndarray, " n_multi_outages n_node"]
    """Which sets of nodes should be outaged as part of the multi-outage computation"""

    injection_nodes: Int[np.ndarray, " n_injection"]
    """The node index that the injection injects onto"""

    mw_injections: Float[np.ndarray, " n_timestep n_injection"]
    """The MW injections of the injections"""

    base_mva: float
    """The base MVA of the grid"""

    node_ids: Union[Sequence[str], Sequence[int]]
    """The ids of the nodes in the original modelling system, length N_node"""

    branch_ids: Union[Sequence[str], Sequence[int]]
    """The ids of the branches in the original modelling system, length N_branch"""

    injection_ids: Union[Sequence[str], Sequence[int]]
    """The ids of the injections in the original modelling system, length N_injection"""

    multi_outage_ids: Union[Sequence[str], Sequence[int]]
    """The ids of the multi-outages in the original modelling system, length N_multi_outages"""

    node_names: Sequence[str]
    """The names of the nodes in the original modelling system, length N_node"""

    branch_names: Sequence[str]
    """The names of the branches in the original modelling system, length N_branch"""

    injection_names: Sequence[str]
    """The names of the injections in the original modelling system, length N_injection"""

    multi_outage_names: Sequence[str]
    """The names of the multi-outages in the original modelling system, length N_multi_outages"""

    branch_types: Sequence[str]
    """The types of the branches in the original modelling system, length N_branch"""

    node_types: Sequence[str]
    """The types of the nodes in the original modelling system, length N_node"""

    injection_types: Sequence[str]
    """The types of the injections in the original modelling system, length N_injection"""

    multi_outage_types: Sequence[str]
    """The types of the multi-outages in the original modelling system, length N_multi_outages."""

    metadata: dict
    """The metadata of the network, if any"""

    bridging_branch_mask: Optional[Bool[np.ndarray, " n_branch"]] = None
    """Mask of branches that would lead to islanding if outaged"""

    nodal_injection: Optional[Float[np.ndarray, " n_timestep n_node"]] = None
    """The injected netto power at each node in the grid for all timesteps"""

    ptdf_is_extended: bool = False
    """Flag to show if PTDF was already extended"""

    branches_at_nodes: Optional[list[Int[np.ndarray, " n_branches_at_node"]]] = None
    """list of arrays containing all branch indices entering or leaving the relevant nodes.
    The branch index points into the list of all branches, length N_relevant_nodes"""

    branch_direction: Optional[list[Bool[np.ndarray, " n_branches_at_node"]]] = None
    """A boolean Array indicating whether the according branch in the branches_at_node
    list is leaving (True) or entering (False) the given node, length N_relevant_nodes"""

    num_branches_per_node: Optional[Int[np.ndarray, " n_relevant_nodes"]] = None
    """An int array indicating how many branches are at each relevant node"""

    injection_idx_at_nodes: Optional[list[Int[np.ndarray, " n_injections_at_node"]]] = None
    """list of length relevant nodes with arrays containing all injections connected at the given
    relevant node, length N_relevant_nodes"""

    num_injections_per_node: Optional[Int[np.ndarray, " n_relevant_nodes"]] = None
    """An int array indicating how many injections are at each relevant node"""

    active_injections: Optional[list[Bool[np.ndarray, " n_injections_at_node"]]] = None
    """A list of the length of relevant nodes.
    Contains Boolean Arrays depicting if injection is non-zero / active in any timesteps"""

    split_multi_outage_branches: Optional[list[Int[np.ndarray, " n_multi_outages n_splits"]]] = None
    """The indices of the branches that are outaged in the multi-outage cases, sorted by
    the amount of branches involved in the outage and represented as integers"""

    split_multi_outage_nodes: Optional[list[Int[np.ndarray, " n_multi_outages n_splits"]]] = None
    """The indices of the nodes that are outaged in the multi-outage cases, sorted by
    the amount of branches involved in the outage and represented as integers"""

    nonrel_io_deltap: Optional[Float[np.ndarray, " n_timesteps n_injection_outages"]] = None
    """The delta p for every injection outage at non-relevant substations.
    Data for relevant and non-relevant substation is stored separately as it's handled separately in the solver.
    Will be computed during process_injection_outages"""

    nonrel_io_node: Optional[Int[np.ndarray, " n_injection_outages"]] = None
    """The node index of the injection outage at non-relevant substations, indexing into all nodes in the PTDF.
    Data for relevant and non-relevant substation is stored separately as it's handled separately in the solver.
    Will be computed during process_injection_outages"""

    nonrel_io_global_inj_index: Optional[Int[np.ndarray, " n_injection_outages"]] = None
    """The injection that this injection outage refers to, pointing into all injections."""

    rel_io_sub: Optional[Int[np.ndarray, " n_relevant_injection_outages"]] = None
    """The relevant substation of the injection outage at relevant substations, pointing only into relevant substation.
    Relevant injection outages might move around due to injection actions, so they are stored in a different format.
    Will be computed during process_injection_outages"""

    rel_io_local_inj_index: Optional[Int[np.ndarray, " n_relevant_injection_outages"]] = None
    """The index of the injection that is outaged at relevant substations, indexing into local injections at the station.
    Will be computed during process_injection_outages"""

    rel_io_global_inj_index: Optional[Int[np.ndarray, " n_relevant_injection_outages"]] = None
    """The injection that this injection outage refers to, pointing into all injections."""

    asset_topology: Optional[Topology] = None
    """The asset topology of the pre-optimization grid."""

    simplified_asset_topology: Optional[Topology] = None
    """The asset topology in a simplified version, containing only optimization-relevant stations and assets."""

    branch_action_set: Optional[list[Bool[np.ndarray, " n_local_actions n_branches_at_node"]]] = None
    """If computed, the branch action set for the grid. This is a list of length relevant nodes with
    an array of branch configurations for each action in the action set of that station. The configurations don't need to be
    unique in case multiple injection combinations are to be evaluated with each branch action"""

    injection_action_set: Optional[list[Bool[np.ndarray, " n_local_actions n_injections_at_node"]]] = None
    """If computed, the injection action set for the grid. This is a list of length relevant nodes with
    an array of injection combinations that are to be chosen alongside the corresponding branch action. The entries are
    usually not unique, as potentially the same injection configuration could be evaluated with different branch
    configurations"""

    branch_action_set_switching_distance: Optional[list[Int[np.ndarray, " n_local_actions"]]] = None
    """If computed, for every element in the branch action set this stores the amount of switching steps
    required to reach that configuration. The outer list is of length equal to the number of relevant nodes."""

    non_rel_bb_outage_br_indices: Optional[list[list[int]]] = None
    """The indices of the branches that are outaged in the busbar-outage cases, represented as integers.
    The length of the outer list equals the number of busbar outages, the inner list contains the indices
    of the branches that are outaged. This will be computed during busbar-outage cases."""

    non_rel_bb_outage_deltap: Optional[Float[np.ndarray, " n_busbar_outages n_timesteps"]] = None
    """The delta p for every injection outage at the time of busbar outage. The length of the outer list equals
    the number of busbar outages, the inner list contains the delta p for each timestep. Will be computed during
    busbar-outage cases"""

    non_rel_bb_outage_nodal_indices: Optional[Int[np.ndarray, " n_busbar_outages"]] = None
    """The node index of the the busbar that will be outaged . Will be computed during busbar-outage cases"""

    rel_bb_outage_br_indices: Optional[list[list[list[list[int]]]]] = None
    """
    This correpsonds to the branch indices that have to be outaged for the relevant
    busbars due to branch_actions.

    The outer list is of length of n_relevant sub station. The next inner list is of length of maximum number of
    branch_action combinations for the substation. Each element of the list has a list of length equal
    to the number of busbars. The next inner list contains the list of branch indices that have to be outaged.
    """
    rel_bb_outage_deltap: Optional[list[list[list[np.ndarray]]]] = None
    """
    This correpsonds to the change in nodal injection (delata_p) that have to be outaged for the relevant busbars.

    The outer list is of length of n_relevant sub station. The next inner list is of length of maximum number of
    branch_actions combinations for the substation if injection_actions are ignored. Each element of the list
    has a list of length equal to the number of physical busbars. The next inner numpy array if of length equal
    to the number of timesteps and contains the delta_p for each timestep.
    """
    rel_bb_outage_nodal_indices: Optional[list[list[list[int]]]] = None
    """
    This correpsonds to the nodal indices where the deltap have to be applied for the relevant busbars due to
    injection_actions.

    The outer list is of length of n_relevant sub station. The next inner list is of length of maximum number of
    branch_action combinations for the substation. Each element of the list has a list of length equal
    to the number of busbars to be outaged. Corresponding to each busbar is an integer representing the
    nodal index of the busbar.
    """

    controllable_pst_node_mask: Optional[Bool[np.ndarray, " n_node"]] = None
    """The mask over nodes that are a controllable phase shifter. When adding the PSDF matrix, bogus
    nodes will be included. The ones that refer to a controllable PST will be mentioned in this mask."""

    realised_stations: Optional[list[list[Station]]] = None
    """The realised stations for each relevant node depending on the branch_actions. The outer list
    is of length equal to the number of relevant nodes. The inner list if of length equal to the number
    of branch actions feasible for the given node. Each station is a simplified station."""

    busbar_a_mappings: Optional[list[list[list[int]]]] = None
    """ The indices of the physical busbars that are mapped to busbar A in each relevant station. This mapping
    is chosen such that minimum switching of assets is required to implement the branch_actions. The outer list
    is of length equal to the number of relevant nodes. The next inner list is of length equal to the number
    of branch actions feasible for the given node. The next inner list stores the indices
    of the physical busbars that are mapped to busbar A."""

    rel_bb_articulation_nodes: Optional[list[list[list[int]]]] = None
    """The nodal_indices of the busbars that create an articulation node inside a station.
    The outer list is of length equal to the number of relevant
    substation. The next inner list is of length equal to the number of branch actions for the sub. The next
    inner list stores the indices of the busbars that are marked as articulation nodes for the given branch action. Such
    busbars when outaged split the station into two.
    For ex, if bus_a: 1 - 2 - 3 ; bus_b: 4 | P.S.: In the ex. '-' means the two busbars are connected via a coupler.
    Here 2 is a bridge busbar as if it is outaged, bus_a will be split into 1 and 3.
    """

    busbar_outage_map: Optional[dict[str, list[str]]] = None
    """
    The information about busbars that have to be outaged.

    The key of the dict is the station's grid_model_id and the value is a list of grid_model_ids
    of the busbars that have to be outaged. If is None then, all the physical
    busbars of the relevant stations will be outaged."""

    @property
    def relevant_nodes(self) -> Int[np.ndarray, " n_relevant_nodes"]:
        """Get relevant nodes of the grid, as indices into all nodes"""
        return np.flatnonzero(self.relevant_node_mask)

    @property
    def n_original_nodes(self) -> int:
        """Get the number of nodes before extending the ptdf"""
        if self.ptdf_is_extended:
            return self.ptdf.shape[1] - len(self.relevant_nodes)
        return self.ptdf.shape[1]

    @property
    def contingency_ids(self) -> list[str]:
        """Get the contingency ids as per the outage masks"""
        branch_outage_ids = np.array(self.branch_ids)[self.outaged_branch_mask]
        injection_outage_ids = np.array(self.injection_ids)[self.outaged_injection_mask]
        # Concatenate branch_outage_ids, injection_outage_ids, and self.multi_outage_ids
        return np.concatenate([branch_outage_ids, injection_outage_ids, np.array(self.multi_outage_ids)]).tolist()


def extract_network_data_from_interface(interface: BackendInterface) -> NetworkData:
    """Extract the network data from the interface.

    Parameters
    ----------
    interface : BackendInterface
        The interface to extract the network data from

    Returns
    -------
    NetworkData
        The extracted network data
    """

    def fillna(a: np.ndarray, b: Union[np.ndarray, float]) -> np.ndarray:
        return np.where(np.isnan(a), b, a)

    return NetworkData(
        ptdf=interface.get_ptdf(),
        psdf=interface.get_psdf(),
        slack=interface.get_slack(),
        relevant_node_mask=interface.get_relevant_node_mask(),
        ac_dc_mismatch=interface.get_ac_dc_mismatch(),
        max_mw_flows=interface.get_max_mw_flows(),
        max_mw_flows_n_1=fillna(interface.get_max_mw_flows_n_1(), interface.get_max_mw_flows()),
        overload_weights=interface.get_overload_weights(),
        n0_n1_max_diff_factors=fillna(interface.get_n0_n1_max_diff_factors(), -1.0),
        cross_coupler_limits=interface.get_cross_coupler_limits()[interface.get_relevant_node_mask()],
        susceptances=interface.get_susceptances(),
        from_nodes=interface.get_from_nodes(),
        to_nodes=interface.get_to_nodes(),
        shift_angles=interface.get_shift_angles(),
        phase_shift_mask=interface.get_phase_shift_mask(),
        monitored_branch_mask=interface.get_monitored_branch_mask(),
        disconnectable_branch_mask=interface.get_disconnectable_branch_mask(),
        outaged_branch_mask=interface.get_outaged_branch_mask(),
        outaged_injection_mask=interface.get_outaged_injection_mask(),
        multi_outage_branch_mask=interface.get_multi_outage_branches(),
        multi_outage_node_mask=interface.get_multi_outage_nodes(),
        injection_nodes=interface.get_injection_nodes(),
        mw_injections=interface.get_mw_injections(),
        base_mva=interface.get_base_mva(),
        node_ids=interface.get_node_ids(),
        branch_ids=interface.get_branch_ids(),
        injection_ids=interface.get_injection_ids(),
        multi_outage_ids=interface.get_multi_outage_ids(),
        node_names=interface.get_node_names(),
        branch_names=interface.get_branch_names(),
        injection_names=interface.get_injection_names(),
        multi_outage_names=interface.get_multi_outage_names(),
        branch_types=interface.get_branch_types(),
        node_types=interface.get_node_types(),
        injection_types=interface.get_injection_types(),
        multi_outage_types=interface.get_multi_outage_types(),
        metadata=interface.get_metadata(),
        asset_topology=interface.get_asset_topology(),
        controllable_phase_shift_mask=interface.get_controllable_phase_shift_mask(),
        phase_shift_taps=interface.get_phase_shift_taps(),
        busbar_outage_map=interface.get_busbar_outage_map(),
    )


def save_network_data(filename: Union[str, Path], network_data: NetworkData) -> None:
    """Save the network data to a file.

    Parameters
    ----------
    filename : Union[str, Path]
        The filename to save the network data to
    network_data : NetworkData
        The network data to save

    """
    with open(filename, "wb") as file:
        pickle.dump(network_data, file)


def load_network_data(filename: Union[str, Path]) -> NetworkData:
    """Load the network data from a file.

    Parameters
    ----------
    filename : Union[str, Path]
        The filename to load the network data from

    Returns
    -------
    NetworkData
        The loaded network data
    """
    with open(filename, "rb") as file:
        return pickle.load(file)


def assert_network_data(network_data: NetworkData) -> None:
    """Check network data for inconsistencies.

    Find some obvious flaws in the network data and raises
    an exception if any are found.

    Parameters
    ----------
    network_data : NetworkData
        The network data to check, can be any state (before or after preprocessing)
    """
    assert np.all(network_data.from_nodes != network_data.to_nodes), "There are branches with the same from and to node"
    assert np.all(np.abs(network_data.susceptances) > 1e-10), "There are branches with susceptances close to zero"
    assert np.all(network_data.max_mw_flows > 0), "There are branches with zero/negative max flow"
    assert np.all(network_data.max_mw_flows_n_1 > 0), "There are branches with zero/negative max flow in the N-1 case"
    assert np.all(np.abs(network_data.shift_angles[:, network_data.phase_shift_mask]) <= 360), (
        "There are branches with shift angles above 360 degrees"
    )
    # We currently can't split the slack node - something in the BSDF doesn't work properly...
    assert network_data.relevant_node_mask[network_data.slack].item() is False


def get_monitored_node_indices(
    network_data: NetworkData,
    include_relevant_nodes: bool = True,
    include_monitored_branch_end: bool = True,
) -> Int[np.ndarray, " n_monitored_nodes"]:
    """Get the indices of the monitored nodes.

    A busbar is monitored if
    - it is a relevant node (if include_relevant_nodes is True)
    - it is the from/to node of a monitored branch (if include_monitored_branch_end is True)

    Parameters
    ----------
    network_data : NetworkData
        The network data
    include_relevant_nodes : bool
        Whether to include the relevant nodes as monitored nodes
    include_monitored_branch_end : bool
        Whether to include the from/to ends of a monitored branch as monitored nodes

    Returns
    -------
    Int[np.ndarray, " n_monitored_nodes"]
        The indices of the monitored nodes, sorted for reproducibility
    """
    retval = np.zeros(0, dtype=int)
    if include_relevant_nodes:
        retval = np.append(retval, np.flatnonzero(network_data.relevant_node_mask))
    if include_monitored_branch_end:
        retval = np.append(retval, network_data.from_nodes[network_data.monitored_branch_mask])
        retval = np.append(retval, network_data.to_nodes[network_data.monitored_branch_mask])
    retval = np.unique(retval)
    return retval


def get_monitored_node_ids(
    network_data: NetworkData,
    include_relevant_nodes: bool = True,
    include_monitored_branch_end: bool = True,
) -> list[str]:
    """Get the ids of the monitored nodes.

    A busbar is monitored if
    - it is a relevant node (if include_relevant_nodes is True)
    - it is the from/to node of a monitored branch (if include_monitored_branch_end is True)

    Parameters
    ----------
    network_data : NetworkData
        The network data
    include_relevant_nodes : bool
        Whether to include the relevant nodes as monitored nodes
    include_monitored_branch_end : bool
        Whether to include the from/to ends of a monitored branch as monitored nodes

    Returns
    -------
    list[str]
        The ids of the monitored nodes, sorted for reproducibility
    """
    node_ids_np = np.array(network_data.node_ids)
    retval = node_ids_np[get_monitored_node_indices(network_data, include_relevant_nodes, include_monitored_branch_end)]
    retval.sort()
    return retval.tolist()


def extract_branch_ids(network_data: NetworkData) -> tuple[list[str], list[str]]:
    """Extract monitored and outaged branch ids from a network data object.

    Parameters
    ----------
    network_data : NetworkData
        The network data object to extract the branch ids from

    Returns
    -------
    list[str]
        A list of monitored branch ids
    list[str]
        A list of outaged branch ids
    """
    monitored_branches = [
        id for (id, monitored) in zip(network_data.branch_ids, network_data.monitored_branch_mask, strict=True) if monitored
    ]
    outaged_branches = [
        id for (id, outaged) in zip(network_data.branch_ids, network_data.outaged_branch_mask, strict=True) if outaged
    ]
    return monitored_branches, outaged_branches


# ruff: noqa: PLR0915
def validate_network_data(network_data: NetworkData) -> None:
    """Run some validation on the preprocessed network data.

    Parameters
    ----------
    network_data : NetworkData
        The network data to validate

    Raises
    ------
    AssertionError
        If the network data is invalid

    Returns
    -------
    None
    """
    assert network_data.ptdf is not None
    assert network_data.psdf is not None
    assert network_data.ptdf_is_extended
    assert network_data.branches_at_nodes is not None

    # Check node B values
    node_b_mask = np.array(network_data.node_types) == "BUS_B"
    assert node_b_mask.sum() == network_data.relevant_node_mask.sum()
    node_a_columns = network_data.ptdf[:, network_data.relevant_node_mask]
    node_b_columns = network_data.ptdf[:, node_b_mask]
    assert node_a_columns.shape == node_b_columns.shape
    assert np.allclose(node_a_columns, node_b_columns)

    # Check correct shapes of everything
    n_nodes = len(network_data.node_ids)
    n_branch = len(network_data.branch_ids)
    n_injections = len(network_data.injection_ids)
    n_timestep = network_data.max_mw_flows.shape[0]
    n_multi_outage = len(network_data.multi_outage_ids)
    n_rel_subs = sum(network_data.relevant_node_mask)
    n_rel_inj_out = len(network_data.rel_io_global_inj_index)
    n_nonrel_inj_out = len(network_data.nonrel_io_global_inj_index)

    assert network_data.ptdf.shape == (n_branch, n_nodes)
    assert network_data.psdf.shape[0] == n_branch
    assert network_data.slack > 0 and network_data.slack < n_nodes
    assert network_data.relevant_node_mask.shape == (n_nodes,)
    assert network_data.max_mw_flows.shape == (n_timestep, n_branch)
    assert network_data.max_mw_flows_n_1.shape == (n_timestep, n_branch)
    assert network_data.overload_weights.shape == (n_branch,)
    assert network_data.n0_n1_max_diff_factors.shape == (n_branch,)
    assert network_data.cross_coupler_limits.shape == (sum(network_data.relevant_node_mask),)
    assert network_data.susceptances.shape == (n_branch,)
    assert network_data.from_nodes.shape == (n_branch,)
    assert network_data.to_nodes.shape == (n_branch,)
    assert network_data.shift_angles.shape == (n_timestep, n_branch)
    assert network_data.phase_shift_mask.shape == (n_branch,)
    assert network_data.controllable_phase_shift_mask.shape == (n_branch,)
    assert not np.any(network_data.controllable_phase_shift_mask & ~network_data.phase_shift_mask)
    assert network_data.controllable_pst_node_mask.shape == (n_nodes,)
    assert np.sum(network_data.controllable_phase_shift_mask) == np.sum(network_data.controllable_pst_node_mask)
    assert len(network_data.phase_shift_taps) == network_data.controllable_phase_shift_mask.sum()
    assert all(len(tap) > 0 for tap in network_data.phase_shift_taps)
    assert network_data.monitored_branch_mask.shape == (n_branch,)
    assert network_data.disconnectable_branch_mask.shape == (n_branch,)
    assert network_data.outaged_branch_mask.shape == (n_branch,)
    assert network_data.multi_outage_branch_mask.shape == (n_multi_outage, n_branch)
    assert network_data.multi_outage_node_mask.shape == (n_multi_outage, n_nodes)
    assert network_data.injection_nodes.shape == (n_injections,)
    assert network_data.mw_injections.shape == (n_timestep, n_injections)
    assert network_data.ac_dc_mismatch.shape == (n_timestep, n_branch)

    assert len(network_data.node_names) == n_nodes
    assert len(network_data.branch_names) == n_branch
    assert len(network_data.injection_names) == n_injections
    assert len(network_data.multi_outage_names) == n_multi_outage
    assert len(network_data.node_types) == n_nodes
    assert len(network_data.branch_types) == n_branch
    assert len(network_data.injection_types) == n_injections
    assert len(network_data.multi_outage_types) == n_multi_outage
    assert len(network_data.node_ids) == n_nodes
    assert len(network_data.branch_ids) == n_branch
    assert len(network_data.injection_ids) == n_injections
    assert len(network_data.multi_outage_ids) == n_multi_outage

    assert network_data.bridging_branch_mask.shape == (n_branch,)
    assert network_data.nodal_injection.shape == (n_timestep, n_nodes)
    assert len(network_data.branches_at_nodes) == n_rel_subs
    assert len(network_data.branch_direction) == n_rel_subs
    assert network_data.num_branches_per_node.shape == (n_rel_subs,)
    assert len(network_data.injection_idx_at_nodes) == n_rel_subs
    assert len(network_data.rel_io_local_inj_index) == n_rel_inj_out
    assert len(network_data.rel_io_sub) == n_rel_inj_out
    assert network_data.nonrel_io_node.shape == (n_nonrel_inj_out,)
    assert network_data.nonrel_io_deltap.shape == (n_timestep, n_nonrel_inj_out)
    assert network_data.num_injections_per_node.shape == (n_rel_subs,)
    assert len(network_data.active_injections) == n_rel_subs
    assert sum(len(mo) for mo in network_data.split_multi_outage_branches) == n_multi_outage
    assert sum(len(mo) for mo in network_data.split_multi_outage_nodes) == n_multi_outage

    for branch_act, inj_act, sw_dist in zip(
        network_data.branch_action_set,
        network_data.injection_action_set,
        network_data.branch_action_set_switching_distance,
        strict=True,
    ):
        assert len(branch_act) == len(inj_act) == len(sw_dist)

    assert len(network_data.simplified_asset_topology.stations) == n_rel_subs
    assert len(network_data.realised_stations) == n_rel_subs
    for realizations in network_data.realised_stations:
        for realized_station in realizations:
            Station.model_validate(realized_station)
    assert len(network_data.busbar_a_mappings) == n_rel_subs
    assert len(network_data.branch_action_set_switching_distance) == n_rel_subs


def get_relevant_stations(network_data: NetworkData) -> list[Station]:
    """
    Get the relevant asset-topology stations from the network data.

    Parameters
    ----------
    network_data : NetworkData
        The network data containing asset topology.

    Returns
    -------
    list[Station]
        A list of relevant stations.
    """
    relevant_node_ids = [
        node for node, mask in zip(network_data.node_ids, network_data.relevant_node_mask, strict=True) if mask
    ]

    def find_station(stations: list[Station], grid_model_id: str, fallback: Optional[Station] = None) -> Station:
        for station in stations:
            if station.grid_model_id == grid_model_id:
                return station
        if fallback is not None:
            return fallback
        raise ValueError(f"Could not find station with grid_model_id {grid_model_id}")

    return [find_station(network_data.simplified_asset_topology.stations, node_id) for node_id in relevant_node_ids]


def map_branch_injection_ids(
    network_data: NetworkData,
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Map the branch and injection IDs for each relevant station.

    Parameters
    ----------
    network_data : NetworkData
        The network data containing branch IDs, injection IDs, and node information.

    Returns
    -------
    list[list[str]]
        Mapped branch IDs for each relevant station.
    list[list[str]]
        Mapped injection IDs for each relevant station.
    """
    branch_ids = network_data.branch_ids
    injection_ids = network_data.injection_ids
    branch_ids_mapped = [
        [branch_ids[branch_id] for branch_id in branch_ids_at_node] for branch_ids_at_node in network_data.branches_at_nodes
    ]
    injection_ids_mapped = [
        [injection_ids[injection_id] for injection_id in injection_ids_at_node]
        for injection_ids_at_node in network_data.injection_idx_at_nodes
    ]
    return branch_ids_mapped, injection_ids_mapped


def extract_action_set(network_data: NetworkData) -> ActionSet:
    """Extract an action set from a filled network data

    This will read the realized stations as saved in the network data

    Parameters
    ----------
    network_data : NetworkData
        The network data to extract the action set from.

    Returns
    -------
    ActionSet
        The action set extracted from the network data.
    """
    assert network_data.realised_stations is not None, "No realised stations in network data"
    assert network_data.asset_topology is not None, "No asset topology in network data"

    # Flatten the realised stations as they are currently stored in per-station batches, i.e.
    # every batch holds only changes for one station. However in the action set we store it flattened.
    local_actions = [station for batch in network_data.realised_stations for station in batch]

    disconnectable_branches = [
        GridElement(id=branch_id, type=branch_type, name=branch_name, kind="branch")
        for (branch_id, branch_type, branch_name, disconnectable) in zip(
            network_data.branch_ids,
            network_data.branch_types,
            network_data.branch_names,
            network_data.disconnectable_branch_mask,
            strict=True,
        )
        if disconnectable
    ]

    controllable_pst_indices = np.flatnonzero(network_data.controllable_phase_shift_mask)
    pst_ranges = [
        PSTRange(
            id=network_data.branch_ids[index],
            type=network_data.branch_types[index],
            name=network_data.branch_names[index],
            kind="branch",
            shift_steps=taps.tolist(),
        )
        for (index, taps) in zip(controllable_pst_indices, network_data.phase_shift_taps, strict=True)
    ]

    return ActionSet(
        starting_topology=network_data.asset_topology,
        local_actions=local_actions,
        disconnectable_branches=disconnectable_branches,
        pst_ranges=pst_ranges,
        hvdc_ranges=[],  # Not implemented yet
        global_actions=[],
        connectable_branches=[],  # Not implemented yet
    )


def extract_nminus1_definition(network_data: NetworkData) -> Nminus1Definition:
    """Extract an N-1 definition from a filled network data

    This will read the monitored and outaged elements in the same order as the jax code processes it, i.e. an N-1 analysis
    should be directly comparable to the jax results

    Parameters
    ----------
    network_data : NetworkData
        The network data to extract the N-1 definition from.

    Returns
    -------
    Nminus1Definition
        The N-1 definition extracted from the network data.
    """
    monitored_branches = [
        GridElement(id=branch_id, name=branch_name, type=branch_type, kind="branch")
        for (branch_id, branch_type, branch_name, monitored) in zip(
            network_data.branch_ids,
            network_data.branch_types,
            network_data.branch_names,
            network_data.monitored_branch_mask,
            strict=True,
        )
        if monitored
    ]

    asset_topology = (
        network_data.simplified_asset_topology if network_data.simplified_asset_topology else network_data.asset_topology
    )
    monitored_nodes = [
        GridElement(id=busbar.grid_model_id, name=busbar.name or "", type=busbar.type, kind="bus")
        for station in asset_topology.stations
        for busbar in station.busbars
    ]

    monitored_switches = [
        GridElement(id=switch.grid_model_id, name=switch.name or "", type=switch.type, kind="switch")
        for station in asset_topology.stations
        for switch in station.couplers
    ]

    basecase_contingency = [Contingency(elements=[], id="BASECASE")]

    branch_contingencies = [
        Contingency(
            elements=[GridElement(id=branch_id, name=branch_name or "", type=branch_type, kind="branch")],
            id=branch_id,
            name=branch_name,
        )
        for (branch_id, branch_type, branch_name, outage) in zip(
            network_data.branch_ids,
            network_data.branch_types,
            network_data.branch_names,
            network_data.outaged_branch_mask,
            strict=True,
        )
        if outage
    ]

    multi_contingencies = []
    for branch_mask, node_mask, outage_id, outage_name in zip(  # noqa: B007
        network_data.multi_outage_branch_mask,
        network_data.multi_outage_node_mask,
        network_data.multi_outage_ids,
        network_data.multi_outage_names,
        strict=True,
    ):
        elements = [
            GridElement(id=branch_id, type=branch_type, name=branch_name or "", kind="branch")
            for (branch_id, branch_type, branch_name, outage) in zip(
                network_data.branch_ids, network_data.branch_types, network_data.branch_names, branch_mask, strict=True
            )
            if outage
        ]
        # This does not make sense right now as multi-outages will never have node outages.
        # TODO refactor multi-outages and change this.
        # elements += [
        #     GridElement(id=node_id, type=node_type, kind="node")
        #     for (node_id, node_type, outage) in zip(network_data.node_ids, network_data.node_types, node_mask)
        #     if outage
        # ]
        multi_contingencies.append(Contingency(elements=elements, id=outage_id, name=outage_name))

    nonrel_inj_contingencies = [
        Contingency(
            elements=[
                GridElement(
                    id=network_data.injection_ids[index],
                    type=network_data.injection_types[index],
                    name=network_data.injection_names[index],
                    kind="injection",
                )
            ],
            id=network_data.injection_ids[index],
            name=network_data.injection_names[index],
        )
        for index in network_data.nonrel_io_global_inj_index
    ]
    rel_inj_contingencies = [
        Contingency(
            elements=[
                GridElement(
                    id=network_data.injection_ids[index],
                    type=network_data.injection_types[index],
                    name=network_data.injection_names[index],
                    kind="injection",
                )
            ],
            id=network_data.injection_ids[index],
            name=network_data.injection_names[index],
        )
        for index in network_data.rel_io_global_inj_index
    ]

    loadflow_parameters = LoadflowParameters(distributed_slack=network_data.metadata.get("distributed_slack", True))

    return Nminus1Definition(
        monitored_elements=monitored_branches + monitored_nodes + monitored_switches,
        contingencies=basecase_contingency
        + branch_contingencies
        + multi_contingencies
        + nonrel_inj_contingencies
        + rel_inj_contingencies,
        loadflow_parameters=loadflow_parameters,
    )
