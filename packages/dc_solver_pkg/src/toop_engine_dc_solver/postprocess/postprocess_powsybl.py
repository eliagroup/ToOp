"""Postprocessing for powsybl based networks.

Provides postprocessing routines for powsybl based networks, including routines for reassigning
busbars of lines, transformers, loads and generators (because it's not supported in powsybl yet) and
routines that help parsing a topology optimization result
"""

from copy import deepcopy
from io import BytesIO
from pathlib import Path

import numpy as np
import pypowsybl
from beartype.typing import Literal, Optional
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from jaxtyping import Bool, Float
from overrides import overrides
from pypowsybl.network import Network
from toop_engine_contingency_analysis.pypowsybl import (
    run_contingency_analysis_powsybl,
)
from toop_engine_dc_solver.postprocess.abstract_runner import AbstractLoadflowRunner, AdditionalActionInfo
from toop_engine_dc_solver.postprocess.apply_asset_topo_powsybl import (
    apply_node_breaker_topology,
    apply_topology_bus_branch,
    is_node_breaker_grid,
)
from toop_engine_grid_helpers.powsybl.loadflow_parameters import (
    DISTRIBUTED_SLACK,
    SINGLE_SLACK,
)
from toop_engine_grid_helpers.powsybl.powsybl_helpers import (
    extract_single_branch_loadflow_result,
    extract_single_injection_loadflow_result,
)
from toop_engine_interfaces.asset_topology_helpers import electrical_components
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Contingency, Nminus1Definition
from toop_engine_interfaces.stored_action_set import ActionSet


def apply_topology(net: Network, actions: list[int], action_set: ActionSet) -> tuple[Network, AdditionalActionInfo]:
    """Apply actions to a powsybl network

    Parameters
    ----------
    net : Network
        The powsybl network to modify
    actions : list[int]
        A list of actions to apply to the network. This is a list of indices into the action set
        local_actions list.
    action_set : ActionSet
        The action set to use for the actions in the form of asset topologies

    Returns
    -------
    Network
        The modified powsybl network as a copy
    AdditionalActionInfo
        Additional information about the action, either a DataFrame of switch updates or a RealizedTopology
    """
    net = deepcopy(net)

    if not len(actions):
        return net

    stations = [action_set.local_actions[action] for action in actions]
    changed_stations_topo = action_set.starting_topology.model_copy(update={"stations": stations})

    if is_node_breaker_grid(net, stations[0].grid_model_id):
        additional_info = apply_node_breaker_topology(net, changed_stations_topo)
    else:
        additional_info = apply_topology_bus_branch(net, changed_stations_topo)

    return net, additional_info


def apply_disconnections(net: Network, disconnections: list[int], action_set: ActionSet) -> Network:
    """Apply static disconnections to a powsybl network.

    Works by removing the elements from the network.

    Parameters
    ----------
    net : Network
        The powsybl network to modify
    disconnections : list[int]
        A list of disconnections to apply to the network. This is a list of indices into the action set
        disconnectable_branches list.
    action_set : ActionSet
        The action set to use for the disconnections

    Returns
    -------
    Network
        The modified powsybl network as a copy

    Raises
    ------
    RuntimeError
        If an element could not be disconnected


    """
    net = deepcopy(net)
    for disconnection in disconnections:
        assert disconnection < len(action_set.disconnectable_branches), "Disconnection index out of range"
        elem = action_set.disconnectable_branches[disconnection]
        if net.disconnect(elem.id) is False:
            raise RuntimeError(f"Failed to disconnect {elem}")
    return net


def compute_cross_coupler_flows(
    net: Network,
    actions: list[int],
    action_set: ActionSet,
    method: Literal["ac", "dc"] = "dc",
    distributed_slack: bool = True,
) -> tuple[
    Float[np.ndarray, " n_splits"],
    Float[np.ndarray, " n_splits"],
    Bool[np.ndarray, " n_splits"],
]:
    """Compute the cross-coupler flows in DC or AC.

    Parameters
    ----------
    net : Network
        The powsybl network to compute the cross-coupler flows for, without any topology applied.
    actions : list[int]
        The list of split substations represented by their action index of length n_splits. The order matters in this case,
        as the return values will be in the same order when opening one switch after the other.
    action_set : ActionSet
        The action set to use for the actions in the form of asset topologies
    method : Literal["ac", "dc"], optional
        Whether to compute the AC or DC power flow, by default "dc"
    distributed_slack : bool, optional
        Whether to use the power values of the generators with distributed slack for the N0-computation, by default True

    Returns
    -------
    Float[np.ndarray, " n_splits"]
        The absolute N-0 active power flow on the cross-coupler branches, in MW
    Float[np.ndarray, " n_splits"]
        The absolute N-0 reactive power flow on the cross-coupler branches, in MVar. Will be all
        zero for DC power flow.
    Bool[np.ndarray, " n_splits"]
        Whether the power flow was successful for each split
    """
    net = deepcopy(net)

    active_power = []
    reactive_power = []
    success = []

    for split in actions:
        if method == "ac":
            res = pypowsybl.loadflow.run_ac(net, DISTRIBUTED_SLACK if distributed_slack else SINGLE_SLACK)
        else:
            res = pypowsybl.loadflow.run_dc(net, DISTRIBUTED_SLACK if distributed_slack else SINGLE_SLACK)
        if res[0].status != pypowsybl.loadflow.ComponentStatus.CONVERGED:
            active_power.append(np.nan)
            reactive_power.append(np.nan)
            success.append(False)
            continue

        station = action_set.local_actions[split]
        # Find the two sides of the split
        components = electrical_components(station, min_num_assets=2)
        assert len(components) == 2, "The split station must have exactly two sides"
        busbars_a = list(components[0])

        branch_res = net.get_branches(attributes=["p1", "q1", "p2", "q2"])
        injection_res = net.get_injections(attributes=["p", "q"])

        p_sum = 0.0
        q_sum = 0.0
        for index, asset in enumerate(station.assets):
            if station.asset_switching_table[busbars_a, index].any():
                # The asset is on busbar A, include it
                if asset.is_branch() is True:
                    if asset.branch_end is None:
                        raise ValueError("Branch end is None")
                    from_end = asset.branch_end in ("from", "hv")
                    p, q = extract_single_branch_loadflow_result(branch_res, asset.grid_model_id, from_end)
                    p_sum += p
                    q_sum += q
                elif asset.is_branch() is False:
                    p, q = extract_single_injection_loadflow_result(
                        injection_res,
                        asset.grid_model_id,
                    )
                    p_sum += p
                    q_sum += q

        active_power.append(p_sum)
        reactive_power.append(q_sum)
        success.append(True)

        net, _ = apply_topology(net, [split], action_set)

    return np.array(active_power, dtype=float), np.array(reactive_power, dtype=float), np.array(success, dtype=bool)


class PowsyblRunner(AbstractLoadflowRunner):
    """A runner for powsybl based networks which implements the AbstractLoadflowRunner interface.

    It can run in parallel using ray's remote mechanic.
    """

    def __init__(self, n_processes: int = 1, batch_size: Optional[int] = None) -> None:
        """Initialize the runner

        Parameters
        ----------
        n_processes : int, optional
            The number of processes to use for parallel computation, by default 1
        batch_size : Optional[int], optional
            The batch size to use for parallel computation, by default None (auto-determined)
        """
        self.n_processes = n_processes
        self.batch_size = batch_size

        self.net = None
        self.action_set: Optional[ActionSet] = None
        self.nminus1_definition: Optional[Nminus1Definition] = None
        self.last_action_info: Optional[AdditionalActionInfo] = None

    @overrides
    def load_base_grid_fs(self, filesystem: AbstractFileSystem, grid_path: Path) -> None:
        """Load the base grid into the loadflow runner, loading from a file system.

        Parameters
        ----------
        filesystem : AbstractFileSystem
            The file system to use to load the grid.
        grid_path : Path
            The path to the grid file
        """
        with filesystem.open(str(grid_path), "rb") as f:
            binary_buffer = BytesIO(f.read())
            self.replace_grid(pypowsybl.network.load_from_binary_buffer(binary_buffer))

    @overrides
    def load_base_grid(self, grid_path: Path) -> None:
        """Load the base grid into the loadflow runner.

        Parameters
        ----------
        grid_path : Path
            The path to the grid file
        """
        return self.load_base_grid_fs(LocalFileSystem(), grid_path)

    def replace_grid(self, net: Network) -> None:
        """Apply a base grid to the runner, if you don't want to load it from a file

        Parameters
        ----------
        net : Network
            The powsybl network to use as the base grid
        """
        self.net = net

    @overrides
    def store_nminus1_definition(self, nminus1_definition: Nminus1Definition) -> None:
        """Store the N-1 definition in the runner."""
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
        """Store the action set in the runner"""
        self.action_set = action_set

    @overrides
    def run_dc_n_0(
        self,
        actions: list[int],
        disconnections: list[int],
    ) -> LoadflowResultsPolars:
        """Run a single N-0 analysis.

        Note: Currently this does not support multi-timesteps

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

        net = self.net
        self.last_action_info = None
        if len(actions):
            net, self.last_action_info = apply_topology(net, actions, self.action_set)
        if len(disconnections):
            net = apply_disconnections(net, disconnections, self.action_set)

        # Run a "N-1" loadflow with only the BASECASE outage
        nminus1_definition = self.nminus1_definition.model_copy(
            update={"contingencies": [Contingency(elements=[], id="BASECASE")]}
        )
        return run_contingency_analysis_powsybl(
            net=net, n_minus_1_definition=nminus1_definition, job_id="", timestep=0, method="dc", polars=True
        )

    @overrides
    def run_dc_loadflow(
        self,
        actions: list[int],
        disconnections: list[int],
    ) -> LoadflowResultsPolars:
        """Run the DC loadflow on the grid.

        Implements run_dc_loadflow from the abstract_runner interface

        Note that this will currently always return n_timesteps=1, as multi-timesteps haven't been
        implemented yet

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
        return self.run_loadflow_single_timestep(actions, disconnections, method="dc")

    @overrides
    def run_ac_loadflow(
        self,
        actions: list[int],
        disconnections: list[int],
    ) -> LoadflowResultsPolars:
        """Run the AC loadflow on the grid.

        Implements run_ac_loadflow from the abstract runner interface.

        Note that this will currently always return n_timesteps=1, as multi-timesteps haven't been
        implemented yet

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
        return self.run_loadflow_single_timestep(actions, disconnections, method="ac")

    def run_loadflow_single_timestep(
        self,
        actions: list[int],
        disconnections: list[int],
        method: Literal["ac", "dc"] = "dc",
    ) -> LoadflowResultsPolars:
        """Run the loadflow for a single timestep with either ac or dc method.

        Parameters
        ----------
        actions : list[int]
            The list of actions to be applied. This is a list of indices into the action set local_actions list.
        disconnections : list[int]
            The list of disconnections to be applied. This is a list of indices into the action set
            disconnectable_branches list.
        method : Literal["ac", "dc"], optional
            The method to use for the loadflow, by default "dc"

        Returns
        -------
        LoadflowResultsPolars
            The results of the loadflow computation
        """
        assert self.net is not None, "Base grid must be loaded before running loadflow"

        net = self.net
        self.last_action_info = None
        if len(actions):
            net, self.last_action_info = apply_topology(net, actions, self.action_set)
        if len(disconnections):
            net = apply_disconnections(net, disconnections, self.action_set)

        return run_contingency_analysis_powsybl(
            net=net,
            n_minus_1_definition=self.nminus1_definition,
            job_id="",
            timestep=0,
            method=method,
            polars=True,
            n_processes=self.n_processes,
        )

    @overrides
    def get_last_action_info(self) -> Optional[AdditionalActionInfo]:
        """Get the additional action info from the last run.

        Returns
        -------
        Optional[AdditionalActionInfo]
            The additional action info, which is either a DataFrame of switch updates or a RealizedTopology.
            If no action was run yet, None is returned.
        """
        return self.last_action_info
