# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Provides a powsybl backend for loading powsybl based grids into the DC solver"""

import functools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.typing as pat
import pypowsybl as pp
import structlog
from beartype.typing import Optional, Sequence, Union
from fsspec import AbstractFileSystem
from jaxtyping import Bool, Float, Int
from toop_engine_dc_solver.preprocess.parallel_pst_groups import build_2d_pst_group_mask_and_labels
from toop_engine_dc_solver.preprocess.powsybl.powsybl_helpers import (
    BranchModel,
    get_lines,
    get_network_as_pu,
    get_p_max,
    get_tie_lines,
    get_trafos,
)
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import (
    materialize_runtime_bus_groups_from_network_state,
)
from toop_engine_grid_helpers.powsybl.powsybl_helpers import load_powsybl_from_fs, sort_powsybl_element_frame_by_id
from toop_engine_interfaces.asset_topology.asset_topology import MasterAssetTopology
from toop_engine_interfaces.asset_topology.runtime_topology import RuntimeAssetTopology, RuntimeBusGroup
from toop_engine_interfaces.backend import BackendInterface
from toop_engine_interfaces.filesystem_helper import load_numpy_filesystem, load_pydantic_model_fs
from toop_engine_interfaces.folder_structure import (
    NETWORK_MASK_NAMES,
    PREPROCESSING_PATHS,
)
from toop_engine_interfaces.nminus1_definition import Contingency, Nminus1Definition

logger = structlog.get_logger(__name__)

INJECTION_COLUMNS = ["name", "p", "bus_id_int", "for_nminus1", "type"]


def _station_ids(stations: Sequence[RuntimeBusGroup]) -> list[str]:
    """Return bus-group ids in order for coverage checks and logging."""
    return [station.bus_group_id for station in stations]


def _runtime_stations_preserve_master_asset_topology_connectivity(
    master_data: MasterAssetTopology,
    runtime_stations: Sequence[RuntimeBusGroup],
) -> tuple[bool, list[str]]:
    """Check whether runtime bus groups preserve canonical connectivity tables from master data."""
    runtime_bus_groups_by_id = {bus_group.bus_group_id: bus_group for bus_group in runtime_stations}
    narrowed_station_ids: list[str] = []
    for station in master_data.bus_groups:
        runtime_bus_group = runtime_bus_groups_by_id.get(station.bus_group_id)
        if runtime_bus_group is None:
            continue
        if station.branch_connectivity is not None and not np.array_equal(
            np.asarray(runtime_bus_group.branch_connectivity, dtype=bool),
            np.asarray(station.branch_connectivity, dtype=bool),
        ):
            narrowed_station_ids.append(station.bus_group_id)
            continue
        if station.injection_connectivity is not None and not np.array_equal(
            np.asarray(runtime_bus_group.injection_connectivity, dtype=bool),
            np.asarray(station.injection_connectivity, dtype=bool),
        ):
            narrowed_station_ids.append(station.bus_group_id)
    return not narrowed_station_ids, narrowed_station_ids


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

    def __init__(
        self,
        data_folder_dirfs: AbstractFileSystem,
        lf_params: Optional[pp.loadflow.Parameters] = None,
        fail_on_non_convergence: bool = True,
    ) -> None:
        """Initiate the powsybl model by a given AbstractFileSystem.

        Parameters
        ----------
        data_folder_dirfs : AbstractFileSystem
            A filesystem which is assumed to be a dirfs pointing to the root for this import job. I.e. the folder structure
            as defined in toop_engine_interfaces.folder_structure is expected to start at root in this filesystem.
        lf_params: Optional[pp.loadflow.Parameters]
            The loadflow parameters to use for the initial loadflow calculation. If None, the default parameters are used.
        fail_on_non_convergence: bool
            Whether to raise an error if the initial loadflow does not converge.
            If False, a warning is logged instead and the backend is initialized with the dc loadflow results
        """
        super().__init__()
        self.data_folder_dirfs = data_folder_dirfs
        net = load_powsybl_from_fs(
            filesystem=data_folder_dirfs,
            file_path=Path(PREPROCESSING_PATHS["grid_file_path_powsybl"]),
        )

        if lf_params is None:
            lf_params = CGMES_DISTRIBUTED_SLACK
        self.lf_params = lf_params
        ac_results, *_ = pp.loadflow.run_ac(net, lf_params)
        if ac_results.status != pp.loadflow.ComponentStatus.CONVERGED:
            message = "Initial AC loadflow did not converge"
            if fail_on_non_convergence:
                raise RuntimeError(message)
            logger.warning(message)
            self.ac_p_values = None
        else:
            self.ac_p_values = net.get_branches(attributes=["p1"])["p1"]

        dc_results = pp.loadflow.run_dc(net, lf_params)
        self.slack_id = net.get_extension("slackTerminal").iloc[0].bus_id
        self.net = net
        self.net_pu = get_network_as_pu(net)
        dc_definition_path = PREPROCESSING_PATHS["dc_nminus1_definition_file_path"]
        self.uses_dc_definition = data_folder_dirfs.exists(dc_definition_path)
        nminus1_definition_path = dc_definition_path
        if not data_folder_dirfs.exists(nminus1_definition_path):
            nminus1_definition_path = PREPROCESSING_PATHS["nminus1_definition_file_path"]
        if data_folder_dirfs.exists(nminus1_definition_path):
            self.nminus1_definition = load_pydantic_model_fs(
                filesystem=data_folder_dirfs,
                file_path=nminus1_definition_path,
                model_class=Nminus1Definition,
            )
        else:
            self.nminus1_definition = Nminus1Definition(contingencies=[], monitored_elements=[], id_type="powsybl")

        assert dc_results[0].status == pp.loadflow.ComponentStatus.CONVERGED, "DC loadflow did not converge"
        assert not self.net.get_shunt_compensators()["p"].any(), "Shunt compensators are not supported yet"
        assert self.net.get_3_windings_transformers().empty, "3 winding transformers are not supported yet"
        self._report_unsupported_definition_elements()

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

        branches = self.net.get_branches(attributes=["connected1", "connected2", "bus1_id", "bus2_id", "p1", "type"])
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
                self._get_boundary_lines(),
                self._get_battery(),
                self._get_hvdc_lcc(),
                self._get_hvdc_vsc(),
            ]
        )

        return injections

    def _get_mask(
        self, mask_filename: str, default_value: Union[bool, float, int], default_shape: int
    ) -> (
        Bool[np.ndarray, " n_masked_element"] | Float[np.ndarray, " n_masked_element"] | Int[np.ndarray, " n_masked_element"]
    ):
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

    def _get_definition_mask(self, element_ids: pd.Index, kind: str, fallback_mask_name: str) -> np.ndarray:
        """Return outage eligibility from the DC definition when available."""
        if not self.uses_dc_definition:
            return self._get_mask(fallback_mask_name, False, len(element_ids))
        definition_ids = {
            element.id
            for contingency in self.nminus1_definition.contingencies
            for element in contingency.elements
            if element.kind == kind
            and (self.nminus1_definition.source_schema != "complex" or contingency.is_single_outage())
        }
        return np.asarray(element_ids.isin(definition_ids), dtype=bool)

    def _report_unsupported_definition_elements(self) -> None:
        """Report definition elements that cannot participate in DC computation."""
        supported_ids = set(self.get_branch_ids()) | set(self.get_injection_ids())
        for contingency in self.nminus1_definition.contingencies:
            supported_elements = [element for element in contingency.elements if element.id in supported_ids]
            for element in contingency.elements:
                if element.id not in supported_ids:
                    logger.warning(
                        "dc_contingency_element_unsupported",
                        contingency_id=contingency.id,
                        element_id=element.id,
                        element_type=element.type,
                    )
            if contingency.elements and not supported_elements:
                logger.warning("dc_contingency_projection_empty", contingency_id=contingency.id)

    @functools.lru_cache
    def _get_lines(self) -> pat.DataFrame[BranchModel]:
        """Add N-1 and observation masks to the lines"""
        lines = get_lines(self.net, self.net_pu)
        if lines.empty:
            return lines

        n_lines = len(lines)
        # Add N-1 and observation masks
        lines["for_reward"] = self._get_mask(NETWORK_MASK_NAMES["line_for_reward"], False, n_lines)
        lines["for_nminus1"] = self._get_definition_mask(lines.index, "branch", "line_for_nminus1")
        lines["overload_weight"] = self._get_mask(NETWORK_MASK_NAMES["line_overload_weight"], 1.0, n_lines)
        lines["disconnectable"] = self._get_mask(NETWORK_MASK_NAMES["line_disconnectable"], False, n_lines)
        lines["controllable"] = np.zeros(n_lines, dtype=bool)
        lines.sort_index(inplace=True)

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
        trafos = sort_powsybl_element_frame_by_id(trafos)

        n_trafos = len(trafos)

        # Add N-1 and observation masks
        trafos["for_reward"] = self._get_mask(NETWORK_MASK_NAMES["trafo_for_reward"], False, n_trafos)
        trafos["for_nminus1"] = self._get_definition_mask(trafos.index, "branch", "trafo_for_nminus1")
        trafos["overload_weight"] = self._get_mask(NETWORK_MASK_NAMES["trafo_overload_weight"], 1.0, n_trafos)
        trafos["disconnectable"] = self._get_mask(NETWORK_MASK_NAMES["trafo_disconnectable"], False, n_trafos)
        trafos["controllable"] = self._get_mask(NETWORK_MASK_NAMES["trafo_controllable"], False, n_trafos)
        trafos["n0_n1_max_diff_factor"] = self._get_mask(NETWORK_MASK_NAMES["trafo_n0_n1_max_diff_factor"], -1.0, n_trafos)
        trafos["has_pst_tap"] = trafos["has_pst_tap"].to_numpy(dtype=bool)
        trafos["pst_linear"] = trafos["pst_linear"].to_numpy(dtype=bool)

        return trafos

    @functools.lru_cache
    def _get_tie_lines(self) -> pat.DataFrame[BranchModel]:
        """Merge the information from dangling lines into the tie lines dataframe."""
        tie_lines = get_tie_lines(self.net, self.net_pu)
        if tie_lines.empty:
            return tie_lines

        n_tie_lines = len(tie_lines)
        tie_lines["for_reward"] = self._get_mask(NETWORK_MASK_NAMES["tie_line_for_reward"], False, n_tie_lines)
        tie_lines["for_nminus1"] = self._get_definition_mask(tie_lines.index, "branch", "tie_line_for_nminus1")
        tie_lines["overload_weight"] = np.ones(n_tie_lines)
        tie_lines["disconnectable"] = np.zeros(n_tie_lines, dtype=bool)
        tie_lines["controllable"] = np.zeros(n_tie_lines, dtype=bool)
        tie_lines.sort_index(inplace=True)

        return tie_lines

    @functools.lru_cache
    def _get_generators(self) -> pd.DataFrame:
        """Get all generators that are connected to a node in _get_nodes()"""
        nodes = self._get_nodes()

        gens = self.net.get_generators()

        gens["for_nminus1"] = self._get_definition_mask(gens.index, "injection", "generator_for_nminus1")

        gens = gens[gens["bus_id"].isin(nodes.index) & (gens["bus_id"] != self.slack_id)]
        gens["bus_id_int"] = nodes.loc[gens["bus_id"], "int_id"].values
        gens["type"] = "GENERATOR"

        return gens[INJECTION_COLUMNS]

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

        return batteries[INJECTION_COLUMNS]

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

        return lcc[INJECTION_COLUMNS]

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

        return vsc[INJECTION_COLUMNS]

    @functools.lru_cache
    def _get_loads(self) -> pd.DataFrame:
        """Get all loads that are connected to a node in _get_nodes()"""
        nodes = self._get_nodes()

        loads = self.net.get_loads()

        loads["for_nminus1"] = self._get_definition_mask(loads.index, "injection", "load_for_nminus1")

        loads = loads[loads["bus_id"].isin(nodes.index) & (loads["bus_id"] != self.slack_id)]
        loads["bus_id_int"] = nodes.loc[loads["bus_id"], "int_id"].values
        loads["type"] = "LOAD"

        return loads[INJECTION_COLUMNS]

    @functools.lru_cache
    def _get_boundary_lines(self) -> pd.DataFrame:
        """Get boundary lines from the grid.

        Get all boundary lines that are connected to a node in _get_nodes() and are not
        part of a tie line. These are injections in powsybl
        """
        nodes = self._get_nodes()
        boundary_lines = self.net.get_boundary_lines()

        boundary_lines["for_nminus1"] = self._get_definition_mask(
            boundary_lines.index, "injection", "boundary_line_for_nminus1"
        )

        boundary_lines.drop(self.net.get_tie_lines()["boundary_line1_id"].values, inplace=True)
        boundary_lines.drop(self.net.get_tie_lines()["boundary_line2_id"].values, inplace=True)
        boundary_lines = boundary_lines[
            boundary_lines["bus_id"].isin(nodes.index) & (boundary_lines["bus_id"] != self.slack_id)
        ]
        boundary_lines["bus_id_int"] = nodes.loc[boundary_lines["bus_id"], "int_id"].values
        boundary_lines["type"] = "BOUNDARY_LINE"

        return boundary_lines[INJECTION_COLUMNS]

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

    def get_basecase_dc_branch_flows(self) -> Float[np.ndarray, " n_timestep n_branch"]:
        """Return base-case DC flows in the solver branch orientation."""
        # Powsybl's p1 convention is opposite to the solver's from-node to to-node orientation.
        return -np.expand_dims(self._get_branches()["p1"].values, axis=0)

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
        """Get a mask of controllable PSTs"""
        return self._get_branches()["controllable"].astype(bool).values & self.get_phase_shift_mask()

    def get_phase_shift_linearity(self) -> Bool[np.ndarray, " n_controllable_psts"]:
        """Get the linearity of the phase shift for each controllable PST.

        i.e. whether the shift angle is linear to the tap position
        """
        return self._get_branches()[self.get_controllable_phase_shift_mask()]["pst_linear"].values

    def get_phase_shift_taps(self) -> list[Float[np.ndarray, " n_controllable_psts"]]:
        """Get a list of taps for each controllable PST"""
        shift_taps = []
        steps = self.net.get_phase_tap_changer_steps(attributes=["alpha"])

        for pst_id in self._get_branches()[self.get_controllable_phase_shift_mask()].index:
            taps_df = steps.loc[pst_id].sort_index()
            taps = -taps_df["alpha"].to_numpy()
            shift_taps.append(taps)
        return shift_taps

    def get_phase_shift_susceptance_taps(self) -> list[Float[np.ndarray, " n_controllable_psts"]]:
        """Get the effective branch susceptance for each controllable PST tap."""
        controllable_branches = self._get_branches()[self.get_controllable_phase_shift_mask()]
        if controllable_branches.empty:
            return []

        tap_steps = self.net.get_phase_tap_changer_steps(attributes=["x", "rho"])
        tap_changers = self.net.get_phase_tap_changers().loc[controllable_branches.index]
        susceptance_taps: list[np.ndarray] = []
        for pst_id in controllable_branches.index:
            steps_df = tap_steps.loc[pst_id].sort_index()
            current_tap = int(tap_changers.at[pst_id, "tap"])
            current_step = steps_df.loc[current_tap]
            current_step_x = float(current_step["x"])
            current_step_rho = float(current_step["rho"])
            current_effective_x = float(controllable_branches.at[pst_id, "x"])

            # x / rho: transformer tap ratios are used in DC susceptance calculations
            # equal pypowsybls to dc_use_transformer_ratio = True
            current_step_factor = (1.0 + current_step_x / 100.0) / current_step_rho
            # This can happen for intentionally constructed or malformed tap tables where the
            # step definition cancels out the normalized reactance at the active tap.
            if np.isclose(current_step_factor, 0.0):
                effective_x_taps = np.full(steps_df.shape[0], current_effective_x, dtype=float)
            else:
                reactance_reference = current_effective_x / current_step_factor
                effective_x_taps = (
                    reactance_reference
                    * (1.0 + steps_df["x"].to_numpy(dtype=float) / 100.0)
                    / steps_df["rho"].to_numpy(dtype=float)
                )

            susceptance_taps.append(1.0 / effective_x_taps)

        return susceptance_taps

    def get_phase_shift_starting_taps(self) -> Int[np.ndarray, " n_controllable_psts"]:
        """Get the starting setpoint of each controllable PST as an integer index into the tap values"""
        psts = self._get_branches()[self.get_controllable_phase_shift_mask()].index
        tap_changers = self.net.get_phase_tap_changers().loc[psts]
        return tap_changers["tap"].values.astype(int) - tap_changers["low_tap"].values.astype(int)

    def get_phase_shift_low_taps(self) -> Int[np.ndarray, " n_controllable_psts"]:
        """Get the lowest tap position in the original grid model

        This is needed so taps as integer indices into tap values
        can be converted back to the original tap positions by tap + low_tap
        """
        psts = self._get_branches()[self.get_controllable_phase_shift_mask()].index
        tap_changers = self.net.get_phase_tap_changers().loc[psts]
        return tap_changers["low_tap"].values.astype(int)

    @functools.lru_cache
    def _get_parallel_pst_groups(self) -> tuple[Bool[np.ndarray, " n_parallel_pst_groups n_controllable_pst"], list[str]]:
        """Get parallel PST grouping metadata aligned with controllable PST arrays.

        The parallel PSTs and their group labels are identified during importing and stored per PST (branch):
          1. BranchModel.``pst_linear``
          2. BranchModel.``pst_group``
        Use the masks to create a 2-d boolean array with rows as parallel PST groups and columns as controllable PSTs, where
        True indicates that a PST belongs to a group. The order of the columns is aligned with the order of controllable PSTs
        in get_controllable_phase_shift_mask(), so that the resulting 2-d array can be used as a mask consumed downstream.
        """
        controllable_branches = self._get_branches()[self.get_controllable_phase_shift_mask()]
        group_labels = controllable_branches["pst_group"].to_numpy(dtype=int)
        return build_2d_pst_group_mask_and_labels(
            group_labels=group_labels,
            pst_id_list=self.get_controllable_phase_shift_ids(),
        )

    def get_parallel_pst_group_mask(self) -> Optional[Bool[np.ndarray, " n_parallel_pst_groups n_controllable_pst"]]:
        """Get the parallel PST groups aligned with the controllable PST arrays."""
        return self._get_parallel_pst_groups()[0]

    def get_parallel_pst_group_ids(self) -> Optional[list[str]]:
        """Get the parallel PST group ids aligned with the group mask rows.

        The group ids are derived from the branch names of the first PST (first-seen order) in the group.
        """
        return self._get_parallel_pst_groups()[1]

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
        """Get a mask of branches that are part of the multi-outage definition."""
        branch_indices = {branch_id: index for index, branch_id in enumerate(self.get_branch_ids())}
        masks = []
        for contingency in self._get_dc_multi_outage_contingencies():
            mask = np.zeros(len(branch_indices), dtype=bool)
            for element in contingency.elements:
                if element.kind == "branch" and element.id in branch_indices:
                    mask[branch_indices[element.id]] = True
            masks.append(mask)
        return np.asarray(masks, dtype=bool).reshape((-1, len(branch_indices)))

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
        """Get IDs of contingencies containing multiple elements."""
        return [contingency.id for contingency in self._get_dc_multi_outage_contingencies()]

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
        """Get names of contingencies containing multiple elements."""
        return [contingency.name for contingency in self._get_dc_multi_outage_contingencies()]

    def get_node_types(self) -> Sequence[str]:
        """We only have busbars, so we can return a constant BUS for every node"""
        return ["BUS"] * len(self._get_nodes())

    def get_branch_types(self) -> Sequence[str]:
        """Branch types can be LINE, TWO_WINDINGS_TRANSFORMER or TIE_LINE"""
        return self._get_branches()["type"].to_list()

    def get_injection_types(self) -> Sequence[str]:
        """Injection types can be GENERATOR, LOAD or BOUNDARY_LINE"""
        return self._get_injections()["type"].to_list()

    def get_multi_outage_types(self) -> Sequence[str]:
        """Get types of contingencies containing multiple elements."""
        return ["CONTINGENCY"] * len(self.get_multi_outage_ids())

    def _get_dc_multi_outage_contingencies(self) -> list[Contingency]:
        """Return multi-outages that contain at least one DC branch."""
        branch_ids = set(self.get_branch_ids())
        return [
            contingency
            for contingency in self.nminus1_definition.contingencies
            if contingency.is_multi_outage()
            and any(element.kind == "branch" and element.id in branch_ids for element in contingency.elements)
        ]

    @functools.lru_cache
    def get_master_asset_topology(self) -> Optional[MasterAssetTopology]:
        """Get canonical asset-topology master data if it exists."""
        if self.data_folder_dirfs.exists(PREPROCESSING_PATHS["asset_topology_master_data_file_path"]):
            return load_pydantic_model_fs(
                filesystem=self.data_folder_dirfs,
                file_path=PREPROCESSING_PATHS["asset_topology_master_data_file_path"],
                model_class=MasterAssetTopology,
            )
        return None

    @functools.lru_cache
    def get_runtime_asset_topology(self) -> Optional[RuntimeAssetTopology]:
        """Get live runtime-enriched topology payloads from canonical master data and the current powsybl net."""
        master_data = self.get_master_asset_topology()
        if master_data is None:
            return None

        runtime_stations = materialize_runtime_bus_groups_from_network_state(network=self.net, master_data=master_data)
        expected_station_ids = [station.bus_group_id for station in master_data.bus_groups]
        runtime_station_ids = _station_ids(runtime_stations)
        missing_station_ids = [station_id for station_id in expected_station_ids if station_id not in runtime_station_ids]
        if missing_station_ids:
            logger.warning(
                "Direct powsybl station materialization did not cover all canonical stations",
                station_ids=missing_station_ids,
            )

        preserves_connectivity, narrowed_station_ids = _runtime_stations_preserve_master_asset_topology_connectivity(
            master_data=master_data,
            runtime_stations=runtime_stations,
        )
        if not preserves_connectivity:
            raise ValueError(
                "Direct powsybl station materialization narrowed canonical connectivity for stations: "
                + ", ".join(narrowed_station_ids)
            )
        return RuntimeAssetTopology(bus_groups=runtime_stations, circuit_groups=master_data.circuit_groups)

    def get_busbar_outage_map(self) -> Optional[dict[str, Sequence[str]]]:
        """Get busbar outages grouped by station id.

        This maps the bus_group_id of each station to a list of busbar grid_model_ids that are part of the N-1 definition.

        Returns
        -------
        Optional[dict[str, Sequence[str]]]
            A dictionary mapping station bus_group_ids to lists of busbar grid_model_ids that are part
            of the N-1 definition. If no busbar outage mask is found, returns None.
        """
        mask_path = self._get_masks_path() / NETWORK_MASK_NAMES["busbar_for_nminus1"]
        if not self.data_folder_dirfs.exists(str(mask_path)):
            return None

        busbar_sections = self.net.get_busbar_sections(attributes=["bus_id"])
        busbar_for_nminus1 = load_numpy_filesystem(filesystem=self.data_folder_dirfs, file_path=str(mask_path))
        selected_busbars = busbar_sections[busbar_for_nminus1]

        outage_map: dict[str, list[str]] = defaultdict(list)
        for station in self.get_runtime_asset_topology().bus_groups:
            busbars = [
                str(busbar.grid_model_id) for busbar in station.busbars if busbar.grid_model_id in selected_busbars.index
            ]
            if busbars:
                outage_map[station.bus_group_id] = [
                    str(busbar.grid_model_id) for busbar in station.busbars if busbar.grid_model_id in selected_busbars.index
                ]
        return outage_map

    def get_metadata(self) -> dict:
        """Get the path to the data_folder, masks_folder and the start datetime of the case"""
        return {
            "masks_folder": self._get_masks_path(),
            "start_datetime": str(self.net.case_date),
        }
