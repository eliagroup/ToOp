"""The postprocessing module has the main functionality of porting back a topology to a pandapower network."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandapower as pp
from beartype.typing import Iterable, Literal, Optional
from jaxtyping import Bool, Float
from overrides import overrides
from toop_engine_contingency_analysis.pandapower import run_contingency_analysis_pandapower
from toop_engine_dc_solver.postprocess.abstract_runner import (
    AbstractLoadflowRunner,
)
from toop_engine_dc_solver.postprocess.apply_asset_topo_pandapower import apply_station
from toop_engine_dc_solver.preprocess.network_data import NetworkData, extract_action_set, extract_nminus1_definition
from toop_engine_grid_helpers.pandapower.pandapower_helpers import (
    get_element_table,
    get_pandapower_branch_loadflow_results_sequence,
    get_pandapower_loadflow_results_injection,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import (
    parse_globally_unique_id,
)
from toop_engine_interfaces.asset_topology import RealizedStation, RealizedTopology
from toop_engine_interfaces.asset_topology_helpers import accumulate_diffs, electrical_components
from toop_engine_interfaces.loadflow_result_helpers_polars import extract_solver_matrices_polars
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Contingency, Nminus1Definition
from toop_engine_interfaces.stored_action_set import ActionSet


@dataclass
class ProcessedContingency:
    """A processed outage case"""

    contingency_name: str
    """The name of the contingency in the N-1 definition"""

    elements: Iterable[tuple[str, int]]
    """The elements in the outage, as an iterable of tuples of the pandapower table and index in that table.
    For the N-0 case, pass an empty iterable. For most outages, the iterable will contain only one element. Only for
    multi-outages will the iterable contain more than one element."""


def apply_topology(
    net: pp.pandapowerNet, actions: list[int], action_set: ActionSet
) -> tuple[pp.pandapowerNet, RealizedTopology]:
    """Apply actions to the pandapower network

    Parameters
    ----------
    net : pp.pandapowerNet
        The network to apply the actions to
    actions : list[int]
        The actions to apply, as a list of action indices in the action set.
    action_set : ActionSet
        The action set in asset topology format

    Returns
    -------
    pp.pandapowerNet
        A copy of the network with the actions applied
    RealizedTopology
        The realized topology after applying the actions, containing switching diffs.
    """
    net = deepcopy(net)

    realized_stations: list[RealizedStation] = []
    for action in actions:
        # Apply the action to the network
        if action >= len(action_set.local_actions):
            raise ValueError(f"Action {action} is out of bounds for the action set")
        _diff, realized_station = apply_station(net, action_set.local_actions[action])
        realized_stations.append(realized_station)

    coupler_diff, reassignment_diff, disconnection_diff = accumulate_diffs(realized_stations)
    realized_topology = RealizedTopology(
        topology=action_set.starting_topology.model_copy(
            update={
                "stations": [s.station for s in realized_stations],
            }
        ),
        coupler_diff=coupler_diff,
        reassignment_diff=reassignment_diff,
        disconnection_diff=disconnection_diff,
    )

    return net, realized_topology


def apply_disconnections(net: pp.pandapowerNet, disconnections: list[int], action_set: ActionSet) -> pp.pandapowerNet:
    """Apply disconnections to the pandapower network

    Parameters
    ----------
    net : pp.pandapowerNet
        The network to apply the disconnections to
    disconnections : list[int]
        The disconnections to apply, as a list of indices into action_set.disconnectable_branches
    action_set : ActionSet
        The action set with disconnectable branches

    Returns
    -------
    pp.pandapowerNet
        A copy of the network with the disconnections applied
    """
    net = deepcopy(net)

    for disconnection in disconnections:
        if disconnection >= len(action_set.disconnectable_branches):
            raise ValueError(f"Disconnection {disconnection} is out of bounds for the action set")
        elem = action_set.disconnectable_branches[disconnection]

        pp_id, asset_type = parse_globally_unique_id(elem.id)
        pp_table = get_element_table(asset_type)
        net[pp_table].loc[int(pp_id), "in_service"] = False

    return net


def compute_cross_coupler_flows(
    net: pp.pandapowerNet,
    actions: list[int],
    action_set: ActionSet,
    method: Literal["ac", "dc"] = "dc",
) -> tuple[
    Float[np.ndarray, " n_splits"],
    Float[np.ndarray, " n_splits"],
    Bool[np.ndarray, " n_splits"],
]:
    """Compute the cross-coupler flows for every coupler

    Computes the cross-coupler flows for the n-th coupler after opening all previous n-1 couplers in
    the topology list.

    Parameters
    ----------
    net : pp.pandapowerNet
        The network to compute the cross-coupler flows for, will be copied before any changes are
        made
    actions : list[int]
        The topology to compute the cross-coupler flows for. The order of the splits matters.
    action_set : ActionSet
        The action set to use for the topology. This is used to get the split nodes and the
        disconnections. The action set is not modified.
    method : Literal["ac", "dc"], optional
        The method to use for the loadflow, by default "dc"

    Returns
    -------
    Float[np.ndarray, " n_splits"]
        The active power flow on the couplers, in MW
    Float[np.ndarray, " n_splits"]
        The reactive power flow on the couplers, in MVar (will be all zeros in DC)
    Bool[np.ndarray, " n_splits"]
        Whether the load-flow computation was successful
    """
    net = deepcopy(net)
    cross_coupler_flows_p = []
    cross_coupler_flows_q = []
    success = []

    for split in actions:
        try:
            if method == "dc":
                pp.rundcpp(net)
            else:
                pp.runpp(net)
        except pp.LoadflowNotConverged:
            cross_coupler_flows_p.append(np.nan)
            cross_coupler_flows_q.append(np.nan)
            success.append(False)
            continue

        # Aggregate the power on busbar A across all branches and injections
        p_on_a = 0
        q_on_a = 0

        station = action_set.local_actions[split]
        # Find the two sides of the split
        components = electrical_components(station, min_num_assets=2)
        assert len(components) == 2, "The split station must have exactly two sides"
        busbars_a = list(components[0])

        for index, asset in enumerate(station.assets):
            asset_id, asset_type = parse_globally_unique_id(asset.grid_model_id)
            asset_id = int(asset_id)

            if station.asset_switching_table[busbars_a, index].any():
                # The asset is on busbar A, include it
                if asset.is_branch() is True:
                    if asset.branch_end is None:
                        raise ValueError("Branch end must be set in asset topo")
                    from_end = asset.branch_end in ("from", "hv")
                    p_on_a += get_pandapower_branch_loadflow_results_sequence(
                        net, types=[asset_type], ids=[asset_id], measurement="active", from_end=from_end, adjust_signs=False
                    ).item()

                    q_on_a += np.nan_to_num(
                        get_pandapower_branch_loadflow_results_sequence(
                            net,
                            types=[asset_type],
                            ids=[asset_id],
                            measurement="reactive",
                            from_end=from_end,
                            adjust_signs=False,
                        )
                    ).item()
                elif asset.is_branch() is False:
                    p_on_a += get_pandapower_loadflow_results_injection(
                        net,
                        types=[asset_type],
                        ids=[asset_id],
                        reactive=False,
                    ).item()

                    q_on_a += np.nan_to_num(
                        get_pandapower_loadflow_results_injection(
                            net,
                            types=[asset_type],
                            ids=[asset_id],
                            reactive=True,
                        )
                    ).item()
                else:
                    raise ValueError("Asset type must be set in asset topo to determine branch/injection")

        cross_coupler_flows_p.append(p_on_a)
        cross_coupler_flows_q.append(q_on_a)
        success.append(True)

        net, _ = apply_topology(net, [split], action_set)

    return (
        np.array(cross_coupler_flows_p, dtype=float),
        np.array(cross_coupler_flows_q, dtype=float),
        np.array(success, dtype=bool),
    )


class PandapowerRunner(AbstractLoadflowRunner):
    """Implements the AbstractLoadflowRunner interface through pandapower."""

    def __init__(
        self,
        n_processes: int = 1,
        batch_size: Optional[int] = None,
    ) -> None:
        """Create a new PandapowerRunner

        Parameters
        ----------
        n_processes : int, optional
            The number of workers to use for parallelization, by default 1 (no parallelization)
        batch_size: Optional[int], optional
            The size of the batches to split the outages into, by default None (automatic)
        """
        self.n_processes = n_processes
        self.batch_size = batch_size
        self.net: Optional[pp.pandapowerNet] = None
        self.nminus1_definition: Optional[Nminus1Definition] = None
        self.action_set: Optional[ActionSet] = None
        self.last_action_info: Optional[RealizedTopology] = None

    @overrides
    def load_base_grid(self, grid_path: Path) -> None:
        """Load the base grid from a file

        Parameters
        ----------
        grid_path : Path
            The path to the grid file
        """
        self.replace_grid(pp.from_json(grid_path))

    def replace_grid(self, net: pp.pandapowerNet) -> None:
        """Replace the base grid with a new one

        Parameters
        ----------
        net : pp.pandapowerNet
            The new base grid
        """
        self.net = net

    @overrides
    def store_nminus1_definition(self, nminus1_definition: Nminus1Definition) -> None:
        """Store the N-1 definition in the loadflow runner.

        This extracts monitored branches and buses
        """
        self.nminus1_definition = nminus1_definition

    @overrides
    def get_nminus1_definition(self) -> Nminus1Definition:
        """Get the N-1 definition from the runner.

        Returns
        -------
        Nminus1Definition
            The N-1 definition stored in the runner.
        """
        assert self.nminus1_definition is not None, "N-1 definition must be set before getting it"
        return self.nminus1_definition

    @overrides
    def store_action_set(self, action_set: ActionSet) -> None:
        """Store the action set in the loadflow runner."""
        self.action_set = action_set

    @overrides
    def run_dc_n_0(
        self,
        actions: list[int],
        disconnections: list[int],
    ) -> LoadflowResultsPolars:
        """Run only the N-0 analysis, no N-1

        Parameters
        ----------
        actions : list[int]
            The list of actions to be applied. This is a list of indices into the action set local_actions list.
        disconnections : list[int]
            The list of disconnections to be applied. This is a list of indices into the action set
            disconnectable_branches list.

        Returns
        -------
        LoadflowResultsPolars
            The loadflow results with exactly one case, the BASECASE only.
        """
        assert self.net is not None, "Base grid must be loaded before running loadflow"

        net, self.last_action_info = apply_topology(self.net, actions, self.action_set)
        net = apply_disconnections(net, disconnections, self.action_set)

        # Run a "N-1" loadflow with only the BASECASE outage
        nminus1_definition = self.nminus1_definition.model_copy(
            update={"contingencies": [Contingency(elements=[], id="BASECASE")]}
        )
        return run_contingency_analysis_pandapower(
            net=net, n_minus_1_definition=nminus1_definition, job_id="", timestep=0, method="dc", polars=True
        )

    @overrides
    def run_dc_loadflow(self, actions: list[int], disconnections: list[int]) -> LoadflowResultsPolars:
        """Run the DC loadflow on the grid.

        Parameters
        ----------
        actions : list[int]
            The list of actions to be applied. This is a list of indices into the action set local_actions list.
        disconnections : list[int]
            The list of disconnections to be applied. This is a list of indices into the action set
            disconnectable_branches list.

        Returns
        -------
        LoadflowResultsPolars
            The loadflow results with a full N-1 analysis.
        """
        return self.run_loadflow(actions, disconnections, "dc")

    @overrides
    def run_ac_loadflow(self, actions: list[int], disconnections: list[int]) -> LoadflowResultsPolars:
        """Run the AC loadflow on the grid.

        Parameters
        ----------
        actions : list[int]
            The list of actions to be applied. This is a list of indices into the action set local_actions list.
        disconnections : list[int]
            The list of disconnections to be applied. This is a list of indices into the action set
            disconnectable_branches list.

        Returns
        -------
        LoadflowResultsPolars
            The loadflow results with a full N-1 analysis
        """
        return self.run_loadflow(actions, disconnections, "ac")

    def run_loadflow(
        self,
        actions: list[int],
        disconnections: list[int],
        method: Literal["ac", "dc"],
    ) -> LoadflowResultsPolars:
        """Run the AC/DC loadflow on the grid

        Parameters
        ----------
        actions : list[int]
            The list of actions to be applied. This is a list of indices into the action set local_actions list.
        disconnections : list[int]
            The list of disconnections to be applied. This is a list of indices into the action set
            disconnectable_branches list.
        method : Literal["ac", "dc"]
            The method to use for the loadflow

        Returns
        -------
        LoadflowResultsPolars
            The results of the loadflow
        """
        net, self.last_action_info = apply_topology(self.net, actions, self.action_set)
        net = apply_disconnections(net, disconnections, self.action_set)

        return run_contingency_analysis_pandapower(
            net=net,
            n_minus_1_definition=self.nminus1_definition,
            job_id="",
            timestep=0,
            method=method,
            n_processes=self.n_processes,
            polars=True,
        )

    def get_last_action_info(self) -> Optional[RealizedTopology]:
        """Get the last action info, which is the realized topology after the last loadflow run.

        Returns
        -------
        Optional[RealizedTopology]
            The realized topology after the last loadflow run, or None if no loadflow has been run yet.
        """
        return self.last_action_info


def compute_n_1_dc(
    net: pp.pandapowerNet,
    network_data: NetworkData,
    n_processes: int = 1,
) -> tuple[
    Float[np.ndarray, " n_failures n_branches_monitored"],
    Bool[np.ndarray, " n_failures"],
]:
    """Compute the n-1 loadflows for the given network

    This is a shorthand wrapper around the pandapower runner to compute the n-1 loadflows for
    a single timestep in a given network

    Parameters
    ----------
    net : pp.pandapowerNet
        The network to compute the n-1 loadflows for
    network_data : NetworkData
        The network data to take branch masks and outages from
    n_processes : int, optional
        The number of workers to use for parallelization, by default 1

    Returns
    -------
    Float[np.ndarray, " n_failures n_branches_monitored"]
        The n-1 loadflows for the given network in MW
    Bool[np.ndarray, " n_failures"]
        Whether the loadflow was successful for the given outage (or a split was detected)
    """
    runner = PandapowerRunner(n_processes=n_processes)
    runner.replace_grid(net)
    runner.store_action_set(extract_action_set(network_data))
    nminus1_def = extract_nminus1_definition(network_data)
    runner.store_nminus1_definition(nminus1_def)
    res = runner.run_dc_loadflow([], [])
    _n0, n_1, success = extract_solver_matrices_polars(
        loadflow_results=res,
        nminus1_definition=nminus1_def,
        timestep=0,
    )

    return n_1, success
