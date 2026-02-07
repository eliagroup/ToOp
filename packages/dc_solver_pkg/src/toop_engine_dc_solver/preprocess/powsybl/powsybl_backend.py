# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provides a powsybl backend for loading powsybl based grids into the DC solver"""

import functools
from pathlib import Path

import logbook
import numpy as np
import pandas as pd
import pandera.typing as pat
import pypowsybl as pp
from beartype.typing import Optional, Sequence, Union
from fsspec import AbstractFileSystem
from jaxtyping import Bool, Float, Int
from toop_engine_dc_solver.preprocess.powsybl.powsybl_helpers import (
    BranchModel,
    get_lines,
    get_network_as_pu,
    get_p_max,
    get_tie_lines,
    get_trafos,
)
from toop_engine_grid_helpers.powsybl.loadflow_parameters import DISTRIBUTED_SLACK, SINGLE_SLACK
from toop_engine_grid_helpers.powsybl.powsybl_helpers import load_powsybl_from_fs
from toop_engine_interfaces.asset_topology import Topology
from toop_engine_interfaces.backend import BackendInterface
from toop_engine_interfaces.filesystem_helper import load_numpy_filesystem, load_pydantic_model_fs
from toop_engine_interfaces.folder_structure import (
    NETWORK_MASK_NAMES,
    PREPROCESSING_PATHS,
)

logger = logbook.Logger(__name__)


class PowsyblBackend(BackendInterface):
    """Interface for a net using powsybl

    This assumes
    - single slack bus
    - no trafo3ws
    - no HVDC lines
    - no batteries
    - no shunt compensators with active power

    These constraints should be met when loading from ucte.
    Furthermore, it expects a similar file structure as the pandapower backend with:
    - grid.xiidm (the gridfile)
    - relevant_subs.npy (a boolean mask of relevant nodes)
    - line_for_reward.npy (a boolean mask of lines that are relevant for the reward)
    - line_for_nminus1.npy (a boolean mask of lines that are relevant for n-1)
    - line_overload_weight.npy (a float mask of weights for the overload)
    - line_disconnectable.npy (a boolean mask of lines that can be disconnected)
    - trafo_for_reward.npy (a boolean mask of transformers that are relevant for the reward)
    - trafo_for_nminus1.npy (a boolean mask of transformers that are relevant for n-1)
    - trafo_n0_n1_max_diff_factor.npy (if a trafo shall be limited in its N-0 to N-1 difference and
      by how much)
    - trafo_overload_weight.npy (a float mask of weights for the overload)
    - trafo_disconnectable.npy (a boolean mask of transformers that can be disconnected)
    - tie_line_for_reward.npy (a boolean mask of tie lines that are relevant for the reward)
    - tie_line_for_nminus1.npy (a boolean mask of tie lines that are relevant for n-1)
    - tie_line_overload_weight.npy (a float mask of weights for the overload)
    - tie_line_disconnectable.npy (a boolean mask of tie lines that can be disconnected)

    Currently, the backend doesn't accept chronics, i.e. only a single timestep.
    """

    def __init__(self, data_folder_dirfs: AbstractFileSystem, distributed_slack: bool = True) -> None:
        """Initiate the powsybl model by a given AbstractFileSystem.

        Parameters
        ----------
        data_folder_dirfs : AbstractFileSystem
            A filesystem which is assumed to be a dirfs pointing to the root for this import job. I.e. the folder structure
            as defined in toop_engine_interfaces.folder_structure is expected to start at root in this filesystem.
        distributed_slack: bool
            Use distributed_slack to initialize the backend.
        """
        super().__init__()
        self.data_folder_dirfs = data_folder_dirfs
        net = load_powsybl_from_fs(
            filesystem=data_folder_dirfs,
            file_path=Path(PREPROCESSING_PATHS["grid_file_path_powsybl"]),
        )

        self.distributed_slack = distributed_slack
        lf_params = DISTRIBUTED_SLACK if distributed_slack else SINGLE_SLACK

        ac_results = pp.loadflow.run_ac(net, lf_params)
        if ac_results[0].status != pp.loadflow.ComponentStatus.CONVERGED:
            logger.warning("AC loadflow did not converge, can't compute ac-dc-mismatch")
            self.ac_p_values = None
        else:
            self.ac_p_values = net.get_branches(attributes=["p1"])["p1"]

        dc_results = pp.loadflow.run_dc(net, lf_params)
        self.slack_id = dc_results[0].reference_bus_id
        self.net = net
        self.net_pu = get_network_as_pu(net)

        assert dc_results[0].status == pp.loadflow.ComponentStatus.CONVERGED, "DC loadflow did not converge"
        assert not self.net.get_shunt_compensators()["p"].any(), "Shunt compensators are not supported yet"
        assert self.net.get_3_windings_transformers().empty, "3 winding transformers are not supported yet"

    @functools.lru_cache
    def _get_nodes(self) -> pd.DataFrame:
        """Add an integer index and a slack column to the result of get_buses().

        This makes sure all nodes are connected to the slack bus and have an integer id.

        TODO add x-nodes for trafo3ws
        """
        nodes = self.net.get_buses(attributes=["name", "connected_component", "synchronous_component"])
        n_nodes = len(nodes)
        nodes["relevant"] = self._get_mask(NETWORK_MASK_NAMES["relevant_subs"], False, n_nodes)
        nodes["coupler_limit"] = self._get_mask(NETWORK_MASK_NAMES["cross_coupler_limits"], 0.0, n_nodes)

        # Filter to only the first connected component
        nodes = nodes[(nodes["connected_component"] == 0) & (nodes["synchronous_component"] == 0)]

        nodes["int_id"] = np.arange(len(nodes))
        return nodes

    @functools.lru_cache
    def _get_branches(self) -> pd.DataFrame:
        """Merge information into the branches list

        This gathers informations from lines, trafos and tie lines into a unified dataframe.
        It also only displays branches which are connected to the nodes from _get_nodes().

        Doesn't know about 3 winding transformers yet
        """
        nodes = self._get_nodes()

        branches = self.net.get_branches()
        # Ignore disconnected branches
        branches = branches[branches["connected1"] & branches["connected2"]]
        # Ignore branches where the nodes have been masked out (usually due to being a separate connected component)
        branches = branches[branches["bus1_id"].isin(nodes.index) & branches["bus2_id"].isin(nodes.index)]
        branches["from_index"] = nodes.loc[branches["bus1_id"].values, "int_id"].values
        branches["to_index"] = nodes.loc[branches["bus2_id"].values, "int_id"].values

        branches = pd.merge(
            left=branches,
            right=pd.concat([self._get_lines(), self._get_trafos(), self._get_tie_lines()]),
            left_index=True,
            right_index=True,
            how="left",
        )
        branches[["p_max_mw", "p_max_mw_n_1"]] = get_p_max(self.net)
        return branches

    @functools.lru_cache
    def _get_injections(self) -> pd.DataFrame:
        """Merge information from generators, loads and dangling lines into the injections dataframe."""
        injections = pd.concat(
            [
                self._get_generators(),
                self._get_loads(),
                self._get_dangling_lines(),
                self._get_battery(),
                self._get_hvdc_lcc(),
                self._get_hvdc_vsc(),
            ]
        )

        return injections

    def _get_mask(
        self, mask_filename: str, default_value: Union[bool, float], default_shape: int
    ) -> Bool[np.ndarray, " n_masked_element"] | Float[np.ndarray, " n_masked_element"]:
        """Load a given mask or return a default mask.

        Parameters
        ----------
        mask_filename: str
            The filename of the mask to load
        default_value: Union[bool, float]
            The default value to set, if the mask file cant be loaded
        default_shape: np._ShapeType
            The shape of the returned default mask

        Returns
        -------
        Bool[np.ndarray, " n_masked_element"]
            A mask for the chosen element either with the values in the file or the default value
        """
        try:
            return load_numpy_filesystem(
                filesystem=self.data_folder_dirfs, file_path=str(self._get_masks_path() / mask_filename)
            )
        except FileNotFoundError:
            return np.full(default_shape, default_value)

    @functools.lru_cache
    def _get_lines(self) -> pat.DataFrame[BranchModel]:
        """Add N-1 and observation masks to the lines"""
        lines = get_lines(self.net, self.net_pu)
        if lines.empty:
            return lines

        n_lines = len(lines)
        # Add N-1 and observation masks
        lines["for_reward"] = self._get_mask(NETWORK_MASK_NAMES["line_for_reward"], False, n_lines)
        lines["for_nminus1"] = self._get_mask(NETWORK_MASK_NAMES["line_for_nminus1"], False, n_lines)
        lines["overload_weight"] = self._get_mask(NETWORK_MASK_NAMES["line_overload_weight"], 1.0, n_lines)
        lines["disconnectable"] = self._get_mask(NETWORK_MASK_NAMES["line_disconnectable"], False, n_lines)
        lines.sort_values("name", inplace=True)

        return lines

    @functools.lru_cache
    def _get_trafos(self) -> pat.DataFrame[BranchModel]:
        """Ddd N-1 and observation masks to the transformers

        also corrects the x and r values for phase and ratio tap changers according to the math in
        https://www.powsybl.org/pages/documentation/grid/model/#transformers
        """
        trafos = get_trafos(self.net, self.net_pu)
        if trafos.empty:
            return trafos

        n_trafos = len(trafos)

        # Add N-1 and observation masks
        trafos["for_reward"] = self._get_mask(NETWORK_MASK_NAMES["trafo_for_reward"], False, n_trafos)
        trafos["for_nminus1"] = self._get_mask(NETWORK_MASK_NAMES["trafo_for_nminus1"], False, n_trafos)
        trafos["overload_weight"] = self._get_mask(NETWORK_MASK_NAMES["trafo_overload_weight"], 1.0, n_trafos)
        trafos["disconnectable"] = self._get_mask(NETWORK_MASK_NAMES["trafo_disconnectable"], False, n_trafos)
        trafos["n0_n1_max_diff_factor"] = self._get_mask(NETWORK_MASK_NAMES["trafo_n0_n1_max_diff_factor"], -1.0, n_trafos)
        trafos["pst_controllable"] = (
            self._get_mask(NETWORK_MASK_NAMES["trafo_pst_controllable"], False, n_trafos) & trafos["has_pst_tap"]
        )

        trafos.sort_values("name", inplace=True)

        return trafos

    @functools.lru_cache
    def _get_tie_lines(self) -> pat.DataFrame[BranchModel]:
        """Merge the information from dangling lines into the tie lines dataframe."""
        tie_lines = get_tie_lines(self.net, self.net_pu)
        if tie_lines.empty:
            return tie_lines

        n_tie_lines = len(tie_lines)
        tie_lines["for_reward"] = self._get_mask(NETWORK_MASK_NAMES["tie_line_for_reward"], False, n_tie_lines)
        tie_lines["for_nminus1"] = self._get_mask(NETWORK_MASK_NAMES["tie_line_for_nminus1"], False, n_tie_lines)
        tie_lines["overload_weight"] = np.ones(n_tie_lines)
        tie_lines["disconnectable"] = np.zeros(n_tie_lines, dtype=bool)

        tie_lines.sort_values("name", inplace=True)

        return tie_lines

    @functools.lru_cache
    def _get_generators(self) -> pd.DataFrame:
        """Get all generators that are connected to a node in _get_nodes()"""
        nodes = self._get_nodes()

        gens = self.net.get_generators()

        gens["for_nminus1"] = self._get_mask(NETWORK_MASK_NAMES["generator_for_nminus1"], False, len(gens))

        gens = gens[gens["bus_id"].isin(nodes.index) & (gens["bus_id"] != self.slack_id)]
        gens["bus_id_int"] = nodes.loc[gens["bus_id"], "int_id"].values
        gens["type"] = "GENERATOR"

        return gens

    @functools.lru_cache
    def _get_battery(self) -> pd.DataFrame:
        """Get all batteries that are connected to a node in _get_nodes()"""
        nodes = self._get_nodes()

        batteries = self.net.get_batteries()

        # TODO: create battery mask
        # batteries["for_nminus1"] = self._get_mask(NETWORK_MASK_NAMES["battery_for_nminus1"], False, len(batteries))
        batteries["for_nminus1"] = False

        batteries = batteries[batteries["bus_id"].isin(nodes.index) & (batteries["bus_id"] != self.slack_id)]
        batteries["bus_id_int"] = nodes.loc[batteries["bus_id"], "int_id"].values
        batteries["type"] = "GENERATOR"
        batteries.loc[batteries["p"] > 0, "type"] = "LOAD"

        return batteries

    @functools.lru_cache
    def _get_hvdc_lcc(self) -> pd.DataFrame:
        """Get all lcc converter stations that are connected to a node in _get_nodes()"""
        nodes = self._get_nodes()

        lcc = self.net.get_lcc_converter_stations()

        # TODO: create lcc and vsc mask
        # lcc["for_nminus1"] = self._get_mask(NETWORK_MASK_NAMES["lcc_for_nminus1"], False, len(lcc))
        lcc["for_nminus1"] = False

        lcc = lcc[lcc["bus_id"].isin(nodes.index) & (lcc["bus_id"] != self.slack_id)]
        lcc["bus_id_int"] = nodes.loc[lcc["bus_id"], "int_id"].values
        lcc["type"] = "GENERATOR"
        lcc.loc[lcc["p"] > 0, "type"] = "LOAD"

        return lcc

    @functools.lru_cache
    def _get_hvdc_vsc(self) -> pd.DataFrame:
        """Get all vsc converter stations that are connected to a node in _get_nodes()"""
        nodes = self._get_nodes()

        vsc = self.net.get_vsc_converter_stations()

        # TODO: create vsc mask
        # vsc["for_nminus1"] = self._get_mask(NETWORK_MASK_NAMES["vsc_for_nminus1"], False, len(vsc))
        vsc["for_nminus1"] = False

        vsc = vsc[vsc["bus_id"].isin(nodes.index) & (vsc["bus_id"] != self.slack_id)]
        vsc["bus_id_int"] = nodes.loc[vsc["bus_id"], "int_id"].values
        vsc["type"] = "GENERATOR"
        vsc.loc[vsc["p"] > 0, "type"] = "LOAD"

        return vsc

    @functools.lru_cache
    def _get_loads(self) -> pd.DataFrame:
        """Get all loads that are connected to a node in _get_nodes()"""
        nodes = self._get_nodes()

        loads = self.net.get_loads()

        loads["for_nminus1"] = self._get_mask(NETWORK_MASK_NAMES["load_for_nminus1"], False, len(loads))

        loads = loads[loads["bus_id"].isin(nodes.index) & (loads["bus_id"] != self.slack_id)]
        loads["bus_id_int"] = nodes.loc[loads["bus_id"], "int_id"].values
        loads["type"] = "LOAD"

        return loads

    @functools.lru_cache
    def _get_dangling_lines(self) -> pd.DataFrame:
        """Get dangling lines from the grid.

        Get all dangling lines that are connected to a node in _get_nodes() and are not
        part of a tie line. These are injections in powsybl
        """
        nodes = self._get_nodes()
        dangling = self.net.get_dangling_lines()

        dangling["for_nminus1"] = self._get_mask(NETWORK_MASK_NAMES["dangling_line_for_nminus1"], False, len(dangling))

        dangling.drop(self.net.get_tie_lines()["dangling_line1_id"].values, inplace=True)
        dangling.drop(self.net.get_tie_lines()["dangling_line2_id"].values, inplace=True)
        dangling = dangling[dangling["bus_id"].isin(nodes.index) & (dangling["bus_id"] != self.slack_id)]
        dangling["bus_id_int"] = nodes.loc[dangling["bus_id"], "int_id"].values
        dangling["type"] = "DANGLING_LINE"

        return dangling

    def _get_masks_path(self) -> Path:
        return Path(PREPROCESSING_PATHS["masks_path"])

    def _get_logs_path(self) -> Path:
        return Path(PREPROCESSING_PATHS["logs_path"])

    def get_slack(self) -> int:
        """Get the index of the slack node"""
        return int(self._get_nodes().loc[self.slack_id, "int_id"])

    def get_susceptances(self) -> Float[np.ndarray, " n_branch"]:
        """Get the susceptances of the branches"""
        return 1 / self._get_branches()["x"].values

    def get_from_nodes(self) -> Int[np.ndarray, " n_branch"]:
        """Get the integer indices of the from nodes"""
        return self._get_branches()["from_index"].values

    def get_to_nodes(self) -> Int[np.ndarray, " n_branch"]:
        """Get the integer indices of the to nodes"""
        return self._get_branches()["to_index"].values

    def get_ac_dc_mismatch(self) -> Float[np.ndarray, " n_timestep n_branch"]:
        """Return the mismatch between the computed AC and DC power flows."""
        if self.ac_p_values is None:
            return np.zeros((1, len(self._get_branches())), dtype=float)
        merged = pd.merge(
            left=self._get_branches(),
            right=self.ac_p_values.rename("ac_p1"),
            left_index=True,
            right_index=True,
            how="left",
        )
        # Since powsybl has a different sign convention for the power flow, we need to invert the sign
        diff = -(merged["ac_p1"] - merged["p1"])
        diff.fillna(0.0, inplace=True)
        return np.expand_dims(diff.values, axis=0)

    def get_max_mw_flows(self) -> Float[np.ndarray, " n_timestep n_branch"]:
        """Get the maximum power flows in MW per branch"""
        return np.expand_dims(self._get_branches()["p_max_mw"].values, axis=0)

    def get_max_mw_flows_n_1(self) -> Float[np.ndarray, " n_timestep n_branch"]:
        """Get the maximum power flows in MW per branch for N-1"""
        return np.expand_dims(self._get_branches()["p_max_mw_n_1"].values, axis=0)

    def get_overload_weights(self) -> Float[np.ndarray, " n_branch"]:
        """Get the overload weights for each branch"""
        return self._get_branches()["overload_weight"].values

    def get_n0_n1_max_diff_factors(self) -> Float[np.ndarray, " n_branch"]:
        """Get the N0-N1 max diff factors for each branch"""
        return self._get_branches()["n0_n1_max_diff_factor"].values

    def get_shift_angles(self) -> Float[np.ndarray, " n_timestep n_branch"]:
        """Get the shift angles in degrees per branch"""
        # TODO find out where this minus comes from...
        return -np.expand_dims(self._get_branches()["alpha"].fillna(0.0).values, axis=0)

    def get_phase_shift_mask(self) -> Bool[np.ndarray, " n_branch"]:
        """Get a mask of branches that can have a phase shift"""
        return self._get_branches()["has_pst_tap"].values

    def get_controllable_phase_shift_mask(self) -> Bool[np.ndarray, " n_branch"]:
        """Get a mask of controllable psts"""
        return self._get_branches()["pst_controllable"].values

    def get_phase_shift_taps(self) -> list[Float[np.ndarray, " n_controllable_psts"]]:
        """Get a list of taps for each pst"""
        shift_taps = []
        steps = self.net.get_phase_tap_changer_steps(attributes=["alpha"])
        for pst_id in self._get_branches()[self._get_branches()["pst_controllable"]].index:
            taps = np.squeeze(steps.loc[pst_id].values)
            shift_taps.append(np.sort(taps))
        return shift_taps

    def get_phase_shift_starting_taps(self) -> Int[np.ndarray, " n_controllable_psts"]:
        """Get the starting setpoint of each controllable PST as an integer index into the tap values"""
        psts = self._get_branches()[self._get_branches()["pst_controllable"]].index
        tap_changers = self.net.get_phase_tap_changers().loc[psts]
        return tap_changers["tap"].values.astype(int) - tap_changers["low_tap"].values.astype(int)

    def get_relevant_node_mask(self) -> Bool[np.ndarray, " n_node"]:
        """Get a mask of relevant nodes"""
        return self._get_nodes()["relevant"].values

    def get_cross_coupler_limits(self) -> Float[np.ndarray, " n_node"]:
        """Get the cross coupler limits for each node"""
        return self._get_nodes()["coupler_limit"].values

    def get_monitored_branch_mask(self) -> Bool[np.ndarray, " n_branch"]:
        """Get a mask of branches that are monitored"""
        return self._get_branches()["for_reward"].values.astype(bool)

    def get_branches_in_maintenance(
        self,
    ) -> Bool[np.ndarray, " n_timestep n_branch"]:
        """Get a mask of branches that are in maintenance, currently always empty"""
        return np.zeros((1, len(self._get_branches())), dtype=bool)

    def get_disconnectable_branch_mask(self) -> Bool[np.ndarray, " n_branch"]:
        """Get a mask of branches that can be disconnected"""
        return self._get_branches()["disconnectable"].values.astype(bool)

    def get_outaged_branch_mask(self) -> Bool[np.ndarray, " n_branch"]:
        """Get a mask of branches that are part of the N-1 definition"""
        return self._get_branches()["for_nminus1"].values.astype(bool)

    def get_multi_outage_branches(
        self,
    ) -> Bool[np.ndarray, " n_multi_outages n_branch"]:
        """Get a mask of branches that are part of the multi-outage definition, currently always empty."""
        return np.zeros((0, len(self._get_branches())), dtype=bool)

    def get_multi_outage_nodes(
        self,
    ) -> Bool[np.ndarray, " n_multi_outages n_node"]:
        """Get a mask of nodes that are part of the multi-outage definition, currently always empty."""
        return np.zeros((0, len(self._get_nodes())), dtype=bool)

    def get_injection_nodes(self) -> Int[np.ndarray, " n_injection"]:
        """Get the integer busbar indices of the injections"""
        return self._get_injections()["bus_id_int"].values

    def get_mw_injections(self) -> Float[np.ndarray, " n_timestep n_injection"]:
        """Get the MW active power of the injections"""
        return np.expand_dims(self._get_injections()["p"].values, axis=0)

    def get_outaged_injection_mask(self) -> Bool[np.ndarray, " n_injection"]:
        """Get a mask of injections that are part of the N-1 definition"""
        return self._get_injections()["for_nminus1"].values.astype(bool)

    def get_base_mva(self) -> float:
        """Get the base MVA of the grid to compensate the psdf as susceptances are in per unit"""
        return float(self.net.nominal_apparent_power)

    def get_node_ids(self) -> Sequence[str]:
        """Node ids are powsybl indices"""
        return self._get_nodes().index.to_list()

    def get_branch_ids(self) -> Sequence[str]:
        """Branch ids are powsybl indices"""
        return self._get_branches().index.to_list()

    def get_injection_ids(self) -> Sequence[str]:
        """Injection ids are powsybl indices"""
        return self._get_injections().index.to_list()

    def get_multi_outage_ids(self) -> Sequence[str]:
        """Currently empty as no multi outages are implemented"""  # noqa: D401
        return []

    def get_node_names(self) -> Sequence[str]:
        """Node names are pulled from powsybl and roughly match their original names"""
        return self._get_nodes()["name"].to_list()

    def get_branch_names(self) -> Sequence[str]:
        """Branch names are in the format of "from - to" voltage levels"""
        return self._get_branches()["name"].to_list()

    def get_injection_names(self) -> Sequence[str]:
        """Injection names are powsybl names"""
        return self._get_injections()["name"].to_list()

    def get_multi_outage_names(self) -> Sequence[str]:
        """Currently empty as no multi outages are implemented"""  # noqa: D401
        return []

    def get_node_types(self) -> Sequence[str]:
        """We only have busbars, so we can return a constant BUS for every node"""
        return ["BUS"] * len(self._get_nodes())

    def get_branch_types(self) -> Sequence[str]:
        """Branch types can be LINE, TWO_WINDINGS_TRANSFORMER or TIE_LINE"""
        return self._get_branches()["type"].to_list()

    def get_injection_types(self) -> Sequence[str]:
        """Injection types can be GENERATOR, LOAD or DANGLING_LINE"""
        return self._get_injections()["type"].to_list()

    def get_multi_outage_types(self) -> Sequence[str]:
        """Currently empty as no multi outages are implemented"""  # noqa: D401
        return []

    def get_asset_topology(self) -> Optional[Topology]:
        """Get the asset topology if it exists"""
        if self.data_folder_dirfs.exists(PREPROCESSING_PATHS["asset_topology_file_path"]):
            return load_pydantic_model_fs(
                filesystem=self.data_folder_dirfs,
                file_path=PREPROCESSING_PATHS["asset_topology_file_path"],
                model_class=Topology,
            )
        return None

    def get_metadata(self) -> dict:
        """Get the path to the data_folder, masks_folder and the start datetime of the case"""
        return {
            "masks_folder": self._get_masks_path(),
            "start_datetime": str(self.net.case_date),
            "distributed_slack": self.distributed_slack,
        }
