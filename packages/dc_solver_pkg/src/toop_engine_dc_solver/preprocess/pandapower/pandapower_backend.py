"""Implementation of the Backend-Interface to extract pandapower data."""

import os
from pathlib import Path

import logbook
import numpy as np
import pandapower as pp
from beartype.typing import Iterable, Optional, Union
from jaxtyping import Bool, Float, Int
from pandapower.pypower.idx_brch import F_BUS, SHIFT, T_BUS
from pandapower.pypower.makeBdc import calc_b_from_branch
from toop_engine_grid_helpers.pandapower.pandapower_helpers import (
    get_dc_bus_voltage,
    get_pandapower_branch_loadflow_results_sequence,
    get_phaseshift_mask,
    get_shunt_real_power,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import (
    get_globally_unique_id,
    table_ids,
)
from toop_engine_grid_helpers.pandapower.pandapower_tasks import (
    get_max_line_flow,
    get_max_trafo3w_flow,
    get_max_trafo_flow,
    get_trafo3w_ppc_branch_idx,
    get_trafo3w_ppc_node_idx,
)
from toop_engine_interfaces.asset_topology import Topology
from toop_engine_interfaces.asset_topology_helpers import load_asset_topology
from toop_engine_interfaces.backend import BackendInterface
from toop_engine_interfaces.folder_structure import (
    CHRONICS_FILE_NAMES,
    NETWORK_MASK_NAMES,
    PREPROCESSING_PATHS,
)

logger = logbook.Logger(__name__)


def convert_to_string_list(data: Iterable) -> list[str]:
    """Convert an iterable of anything into a string list, replacing None with an empty string"""
    return [str(d) if d is not None else "" for d in data]


class PandaPowerBackend(BackendInterface):
    """Interface for a Pandapower backend.

    This assumes a DC loadflow and the following conversion steps to have happened:
        distributed slack -> single slack
        unsupplied busses -> removed
        no scaling injections
        closed switches are fused

    Branches in Pandapower are split into
        lines
        trafos
        3-winding trafos (resulting in 3 branches)
        impedances
    Nodes are referencing
        busses (in service and not islanded)
        auxiliary busses (one per 3-winding trafo or xward)
    Injections are split into
        gens
        loads
    """

    INJECTION_TYPE_MAPPING = (
        ("gen", ""),
        ("sgen", ""),
        ("load", ""),
        ("shunt", ""),
        ("dcline", "_from"),
        ("dcline", "_to"),
        ("ward", "_load"),
        ("ward", "_shunt"),
        ("xward", "_load"),
        ("xward", "_gen"),
        ("xward", "_shunt"),
    )

    def __init__(
        self,
        data_path: Union[Path, str],
        chronics_id: Optional[int] = None,
        chronics_slice: Optional[slice] = None,
    ) -> None:
        """Initiate the pandapower model by a given Path.

        The folder at the path should contain:
        grid.json (The grid model)
        relevant_subs.npy (Mask of relevant busbars in the pandapower grid model)
        for branch_types line, trafo, trafo3w:
        - {branch_type}_for_nminus1.npy (Mask of all branches to outage split per pandapower branch type)
        - {branch_type}_for_reward.npy (Mask of all branches to monitor split per pandapower branch type)
        - All missing masks will be assumed as not monitored/outaged
        optionally a timestep subdirectory chronics/000X including
        - load_p.npy
        - gen_p.npy (previously prod_p in the grid2op naming convention)
        - sgen_p.npy (optional)
        - dcline_p.npy (optional)
        - Timestep info for other injections will be filled with constant value from net
        """
        super().__init__()
        data_path = Path(data_path)
        self.data_path = data_path
        self.chronics_id = chronics_id
        self.chronics_slice = chronics_slice
        if chronics_id is not None:
            chronics_path = self._get_chronics_path()
            self.load_p = np.load(chronics_path / CHRONICS_FILE_NAMES["load_p"])
            self.gen_p = np.load(chronics_path / CHRONICS_FILE_NAMES["gen_p"])
            self.sgen_p = (
                np.load(chronics_path / CHRONICS_FILE_NAMES["sgen_p"])
                if os.path.exists(chronics_path / CHRONICS_FILE_NAMES["sgen_p"])
                else None
            )
            self.dcline_p = (
                np.load(chronics_path / CHRONICS_FILE_NAMES["dcline_p"])
                if os.path.exists(chronics_path / CHRONICS_FILE_NAMES["dcline_p"])
                else None
            )

            if chronics_slice is not None:
                self.load_p = self.load_p[chronics_slice]
                self.gen_p = self.gen_p[chronics_slice]
                self.sgen_p = self.sgen_p[chronics_slice] if self.sgen_p is not None else None
                self.dcline_p = self.dcline_p[chronics_slice] if self.dcline_p is not None else None

        grid_file_path = data_path / PREPROCESSING_PATHS["grid_file_path_pandapower"]
        self.net: pp.pandapowerNet = pp.from_json(grid_file_path)
        self.ppci = pp.converter.to_ppc(self.net, init="flat", calculate_voltage_angles=True)
        # # assert len(pp.topology.unsupplied_buses(net)) == 0
        # assert len(self.net.shunt) == 0
        # assert len(self.net.dcline) == 0
        assert (len(self.net.ext_grid) + self.net.gen.slack.sum()) == 1
        assert len(self.net._isolated_buses) == 0
        assert np.all(self.net.load.scaling == 1.0)
        assert np.all(self.net.load.const_z_percent == 0.0)
        assert np.all(self.net.load.const_i_percent == 0.0)
        assert np.all(self.net.gen.scaling == 1.0)
        assert np.all(self.net.sgen.scaling == 1.0)

        # assert len(self.net.xward) == 0
        # assert np.all(self.net.sgen.p_mw == 0)

    @property
    def _n_timesteps(self) -> int:
        if self.chronics_id is not None:
            return len(self.load_p)
        return 1

    def _get_ppc_bus_lookup(self) -> Int[np.ndarray, " n_node"]:
        """Get the lookup from pandapower bus index to ppc bus index"""
        lookup = self.net["_pd2ppc_lookups"]["bus"]
        lookup = lookup[lookup != -1]
        return lookup

    def _get_masks_path(self) -> Path:
        return self.data_path / PREPROCESSING_PATHS["masks_path"]

    def _get_logs_path(self) -> Path:
        return self.data_path / PREPROCESSING_PATHS["logs_path"]

    def _get_chronics_path(self) -> Path:
        return self.data_path / PREPROCESSING_PATHS["chronics_path"] / f"{self.chronics_id:04d}"

    def get_slack(self) -> int:
        """Get index of the slack node

        Note that the solver does not support distributed slack nodes, if you have a
        distributed slack, replace all but one slack node by their injections or create
        a virtual slack node that is connected with same-impendance lines to the other
        slack nodes.

        Returns
        -------
        int
            The index of the slack node
        """
        if any(self.net.gen.slack):
            slack_bus = self.net.gen[self.net.gen.slack]["bus"].values[0]
        else:
            slack_bus = self.net.ext_grid.iloc[0]["bus"]

        # Translate the slack bus id to an index, in case the bus index is not continuous
        slack_bus = self.net.bus.index.get_loc(slack_bus)

        return int(slack_bus)

    def get_relevant_node_mask(self) -> Bool[np.ndarray, " n_node"]:
        """Get the relevant nodes mask

        This refers to the node A (the node that is present in the un-extended PTDF) of the
        relevant substations. The relevant nodes are those that can be split later on
        in the solver.

        Returns
        -------
        Bool[np.ndarray, " n_node"]:
            The mask over nodes, indicating if they are relevant (splittable)
        """
        try:
            pp_bus_mask = np.load(self._get_masks_path() / NETWORK_MASK_NAMES["relevant_subs"])
        except FileNotFoundError:
            pp_bus_mask = np.zeros(len(self.net.bus), dtype=bool)
        aux_bus_mask = np.zeros(len(self.net.trafo3w) + len(self.net.xward), dtype=bool)
        ppc_bus_mask = np.concatenate([pp_bus_mask, aux_bus_mask])
        ppci_relevant_subs = ppc_bus_mask[self._get_ppc_bus_lookup()]
        return ppci_relevant_subs

    def get_cross_coupler_limits(self) -> Float[np.ndarray, " n_node"]:
        """Get cross-coupler limits of the buses

        Returns
        -------
        Float[np.ndarray, " n_node"]:
            The cross-coupler limits of the buses
        """
        try:
            pp_bus_mask = np.load(self._get_masks_path() / NETWORK_MASK_NAMES["cross_coupler_limits"])
        except FileNotFoundError:
            pp_bus_mask = np.zeros(len(self.net.bus), dtype=float)
        aux_bus_mask = np.zeros(len(self.net.trafo3w) + len(self.net.xward), dtype=float)
        ppc_bus_mask = np.concatenate([pp_bus_mask, aux_bus_mask])
        ppci_relevant_subs = ppc_bus_mask[self._get_ppc_bus_lookup()]
        return ppci_relevant_subs

    def get_ac_dc_mismatch(self) -> Float[np.ndarray, " n_timestep n_branch"]:
        """Get the AC-DC mismatch for each branch and timestep."""
        try:
            pp.runpp(self.net)
        except pp.LoadflowNotConverged:
            return np.zeros((self._n_timesteps, len(self.net._ppc["internal"]["branch_is"])), dtype=float)
        ac_flows = get_pandapower_branch_loadflow_results_sequence(
            self.net, self.get_branch_types(), table_ids(self.get_branch_ids()), measurement="active"
        )
        pp.rundcpp(self.net)
        dc_flows = get_pandapower_branch_loadflow_results_sequence(
            self.net, self.get_branch_types(), table_ids(self.get_branch_ids()), measurement="active"
        )
        # TODO rerun the computation for each timestep
        return np.expand_dims(ac_flows - dc_flows, axis=0).repeat(self._n_timesteps, axis=0)

    def get_max_mw_flows(self) -> Float[np.ndarray, " n_timestep n_branch"]:
        """Get maximum flow per branch.

        The timestep dimension is added to represent temperature-dependent capacity
        limits. If the capacity limits are not temperature-dependent, the same value
        should be returned for all timesteps.

        Returns
        -------
        Float[np.ndarray, " n_timestep n_branch"]
            The maximum flow per branch and per timestep
        """
        # Max Line Load
        max_line_flow = get_max_line_flow(self.net)

        # Max Trafo2w Load
        max_trafo_flow = get_max_trafo_flow(self.net)

        # Max Trafo3w Flow
        max_trafo3w_flow = get_max_trafo3w_flow(self.net)

        # Max Impedance Flow
        max_impedance_flow = np.full(len(self.net.impedance), 9999999.0)

        # Max xward Flow
        max_xward_flow = np.full(len(self.net.xward), self.get_base_mva())
        max_mw_flow = np.concatenate(
            [
                max_line_flow,
                max_trafo_flow,
                max_trafo3w_flow,
                max_impedance_flow,
                max_xward_flow,
            ]
        )

        max_mw_flow = np.nan_to_num(max_mw_flow, nan=np.inf)

        return np.tile(
            max_mw_flow[self.net._ppc["internal"]["branch_is"]][None, :],
            (self._n_timesteps, 1),
        )

    def get_susceptances(self) -> Float[np.ndarray, " n_branch"]:
        """Get susceptances of the branches

        Returns
        -------
        Float[np.ndarray, " n_branch"]
            The susceptances of the branches
        """
        susceptance = calc_b_from_branch(self.ppci["branch"], len(self.ppci["branch"])).real.astype(float)
        return susceptance

    def get_from_nodes(self) -> Int[np.ndarray, " n_branch"]:
        """Get "from" nodes of the branches

        Returns
        -------
        Int[np.ndarray, " n_branch"]:
            The from nodes of the branches
        """
        from_node = self.ppci["branch"][:, F_BUS].real.astype(int)
        return from_node

    def get_to_nodes(self) -> Int[np.ndarray, " n_branch"]:
        """Get "to" nodes of the branches

        Returns
        -------
        Int[np.ndarray, " n_branch"]:
            The to nodes of the branches
        """
        to_node = self.ppci["branch"][:, T_BUS].real.astype(int)
        return to_node

    def get_shift_angles(self) -> Float[np.ndarray, " n_timestep n_branch"]:
        """Get shift angles of the branches in degree

        The timestep dimension is added to represent time-varying phase shift angles.
        If the phase shift angles are not time-varying, the same value should be returned
        for all timesteps.

        Returns
        -------
        Float[np.ndarray, " n_timestep n_branch"]
            The shift angles of the branches
        """
        shift_angle = self.ppci["branch"][:, SHIFT]
        return np.tile(shift_angle[None], (self._n_timesteps, 1))

    def get_phase_shift_mask(self) -> Bool[np.ndarray, " n_branch"]:
        """Get mask of phase shifters

        True means a branch is a phase shifter, i.e. can have shift_degree != 0
        False means it is not a phase shifter

        Returns
        -------
        Bool[np.ndarray, " n_branch"]
            The mask of phase shifters
        """
        mask, _, _ = get_phaseshift_mask(self.net)
        assert mask.shape[0] == self.ppci["branch"].shape[0]

        return mask

    def _get_controllable_phase_shift_data(
        self,
    ) -> tuple[Bool[np.ndarray, " n_branch"], list[Float[np.ndarray, " n_tap_positions"]]]:
        """Get both the mask and shift taps of the controllable phase shifters

        Returns
        -------
        Bool[np.ndarray, " n_branch"]
            The mask of controllable phase shifters
        list[Float[np.ndarray, " n_tap_positions"]]
            The shift taps of the controllable phase shifters
        """
        try:
            controllable_pst_mask = np.load(self._get_masks_path() / NETWORK_MASK_NAMES["trafo_pst_controllable"])
        except FileNotFoundError:
            controllable_pst_mask = np.zeros(len(self.net.trafo), dtype=bool)

        controllable_pst_mask_extended = np.zeros(self.ppci["branch"].shape[0], dtype=bool)

        if self.net.trafo.empty:
            return controllable_pst_mask_extended, []
        trafo_start, trafo_end = self.net._pd2ppc_lookups["branch"]["trafo"]
        controllable_pst_mask_extended[trafo_start:trafo_end] = controllable_pst_mask

        mask, tech_controllable, shift_taps = get_phaseshift_mask(self.net)
        # A trafo is controllable if it is technically a controllable phase shifter and in the controllable mask
        controllable = tech_controllable & controllable_pst_mask_extended

        # Filter the shift taps to only include the ones also in the mask
        shift_taps = [
            taps
            for (taps, index) in zip(shift_taps, np.flatnonzero(tech_controllable), strict=True)
            if controllable_pst_mask_extended[index]
        ]

        assert not np.any(controllable & ~mask)
        assert controllable.shape[0] == self.ppci["branch"].shape[0]

        return controllable, shift_taps

    def get_controllable_phase_shift_mask(self) -> Bool[np.ndarray, " n_branch"]:
        """Get the controllable phase shifters.

        A trafo is controllable if it is technically a controllable phase shifter and in the controllable mask

        Returns
        -------
        Bool[np.ndarray, " n_branch"]
            The mask of controllable phase shifters
        """
        mask, _taps = self._get_controllable_phase_shift_data()
        return mask

    def get_phase_shift_taps(self) -> list[Float[np.ndarray, " n_tap_positions"]]:
        """Return the tap positions of the phase shifters

        Returns
        -------
        list[Float[np.ndarray, " n_tap_positions"]]
            The tap positions of the phase shifters
        """
        _mask, taps = self._get_controllable_phase_shift_data()
        return taps

    def get_monitored_branch_mask(self) -> Bool[np.ndarray, " n_branch"]:
        """Get mask of monitored branches for the reward calculation

        True means a branch is monitored, False means it is not monitored

        Returns
        -------
        Bool[np.ndarray, " n_branch"]
            The mask of monitored branches
        """
        ppc_branch_inservice = self.net._ppc["internal"]["branch_is"]
        ppc_branch_mask = np.zeros(ppc_branch_inservice.shape, dtype=bool)
        for branch_type in ["line", "trafo", "trafo3w"]:
            try:
                branch_type_mask = np.load(self._get_masks_path() / NETWORK_MASK_NAMES[f"{branch_type}_for_reward"])
                (
                    branch_start_index,
                    branch_end_index,
                ) = self.net._pd2ppc_lookups["branch"][branch_type]
                if branch_type == "trafo3w":
                    # 3-winding trafos are modeled as 3 branches per trafo3w under the hood
                    # but the initial mask is based on pandapower indices
                    branch_type_mask = np.concatenate([branch_type_mask, branch_type_mask, branch_type_mask])
                ppc_branch_mask[branch_start_index:branch_end_index] = branch_type_mask
            except FileNotFoundError:
                logger.info(
                    f"No file '{branch_type}_for_reward.npy' in given grid path '{self._get_masks_path()}'. "
                    f"In this case, {branch_type}s are not taken into account for the reward"
                )
                continue
            except KeyError:
                # Thrown if there is no branch_type in the grid
                continue
        return ppc_branch_mask[ppc_branch_inservice]

    def get_branches_in_maintenance(
        self,
    ) -> Bool[np.ndarray, " n_timestep n_branch"]:
        """Get the mask of branches that are in maintenance

        This is currently not implemented and returns a mask of False
        """
        ppc_branch_inservice = self.net._ppc["internal"]["branch_is"]
        n_branch = np.sum(ppc_branch_inservice)
        return np.zeros((self._n_timesteps, n_branch), dtype=bool)

    def get_disconnectable_branch_mask(self) -> Bool[np.ndarray, " n_branch"]:
        """Get the mask of disconnectable branches

        By default, no branches are disconnectable
        """
        ppc_branch_inservice = self.net._ppc["internal"]["branch_is"]
        ppc_branch_mask = np.zeros(ppc_branch_inservice.shape, dtype=bool)
        for branch_type in ["line", "trafo", "trafo3w"]:
            try:
                branch_type_mask = np.load(self._get_masks_path() / NETWORK_MASK_NAMES[f"{branch_type}_disconnectable"])
                (
                    branch_start_index,
                    branch_end_index,
                ) = self.net._pd2ppc_lookups["branch"][branch_type]
                if branch_type == "trafo3w":
                    # 3-winding trafos are modeled as 3 branches per trafo3w under the hood
                    # but the initial mask is based on pandapower indices
                    branch_type_mask = np.concatenate([branch_type_mask, branch_type_mask, branch_type_mask])
                ppc_branch_mask[branch_start_index:branch_end_index] = branch_type_mask
            except FileNotFoundError:
                logger.info(
                    f"No file '{branch_type}_disconnectable.npy' in given grid path '{self._get_masks_path()}'. "
                    f"In this case, {branch_type}s are not taken into account for the disconnectable branches"
                )
                continue
            except KeyError:
                # Thrown if there is no branch_type in the grid
                continue
        return ppc_branch_mask[ppc_branch_inservice]

    def get_outaged_branch_mask(self) -> Bool[np.ndarray, " n_branch"]:
        """Get mask of outaged branches for the N-1 computation

        True means a branch is outaged, False means it is not outaged
        3winding trafos are handled in multi outages, since they are modeled as three branches

        Returns
        -------
        Bool[np.ndarray, " n_branch"]
            The mask of outaged branches
        """
        ppc_branch_inservice = self.net._ppc["internal"]["branch_is"]
        ppc_branch_mask = np.zeros(ppc_branch_inservice.shape, dtype=bool)
        for branch_type in ["line", "trafo"]:
            if self.net[branch_type].empty:
                continue
            try:
                branch_type_mask = np.load(self._get_masks_path() / NETWORK_MASK_NAMES[f"{branch_type}_for_nminus1"])
                (
                    branch_start_index,
                    branch_end_index,
                ) = self.net._pd2ppc_lookups["branch"][branch_type]
                ppc_branch_mask[branch_start_index:branch_end_index] = branch_type_mask
            except FileNotFoundError:
                logger.info(
                    f"No file '{branch_type}_for_nminus1.npy' in given grid path '{self._get_masks_path()}'. "
                    f"In this case, {branch_type}s are not taken into account in the N-1 analysis"
                )
                continue
        return ppc_branch_mask[ppc_branch_inservice]

    def get_trafo3w_multioutage(
        self,
    ) -> tuple[
        Bool[np.ndarray, " n_trafo3w_outages n_branch"],
        Bool[np.ndarray, " n_trafo3w_outages n_node"],
        list[str],
        list[int],
    ]:
        """Get mask of outaged branches, busbars and names for trafo3ws to outage

        True means a branch is outaged, False means it is not outaged.

        Returns
        -------
        Bool[np.ndarray, " n_trafo3w_outages n_branch"]
            The mask of outaged branches for every trafo3w-outage
        Bool[np.ndarray, " n_trafo3w_outages n_node"]
            The mask of outaged nodes for every trafo3w-outage
        list[str]
            The names of the trafo3w multi-outages
        list[int]
            The pandapower IDs of the trafo3w multi-outages
        """
        # Get multi outages relating to trafo3ws
        try:
            trafo3w_outage_mask = np.load(self._get_masks_path() / NETWORK_MASK_NAMES["trafo3w_for_nminus1"])
        except FileNotFoundError:
            logger.info(
                f"No file 'trafo3w_for_nminus1.npy' in given grid path '{self._get_masks_path()}'. "
                "In this case, trafo3ws are not taken into account in the multi outage N-1 analysis"
            )
            return (
                np.empty((0, self.ppci["branch"].shape[0]), dtype=bool),
                np.empty((0, self.ppci["bus"].shape[0]), dtype=bool),
                [],
                [],
            )

        trafo3w_outage_names = convert_to_string_list(self.net.trafo3w.name[trafo3w_outage_mask])
        trafo3w_outage_id = self.net.trafo3w.index[trafo3w_outage_mask].tolist()

        ppc_branch_inservice = self.net._ppc["internal"]["branch_is"]
        outaged_trafo3w_idx = np.flatnonzero(trafo3w_outage_mask)
        ppc_branch_indices = get_trafo3w_ppc_branch_idx(self.net, outaged_trafo3w_idx)
        branch_outages = np.zeros((len(outaged_trafo3w_idx), ppc_branch_inservice.shape[0]), dtype=bool)
        branch_outages[np.arange(len(outaged_trafo3w_idx)), ppc_branch_indices] = True

        ppc_node_indices = get_trafo3w_ppc_node_idx(self.ppci, ppc_branch_indices)
        bus_outages = np.zeros((len(outaged_trafo3w_idx), self.ppci["bus"].shape[0]), dtype=bool)
        bus_outages[np.arange(len(outaged_trafo3w_idx)), ppc_node_indices] = True

        # Filter down to only ppci-branches
        return (
            branch_outages[:, ppc_branch_inservice],
            bus_outages,
            trafo3w_outage_names,
            trafo3w_outage_id,
        )

    def get_busbar_multioutage(
        self,
    ) -> tuple[
        Bool[np.ndarray, " n_busbar_outages n_branch"],
        Bool[np.ndarray, " n_busbar_outages n_node"],
        list[str],
        list[int],
    ]:
        """Get mask of nodes for which nodal injections will be zeroed

        Returns
        -------
        Bool[np.ndarray, " n_busbar_outages n_branch"],
            The mask of outaged branches for every busbar-outage
        Bool[np.ndarray, " n_busbar_outages n_node"],
            The mask of outaged nodes for every busbar-outage
        list[str]
            The names of the busbar multi-outages
        list[int]
            The pandapower IDs of the busbar multi-outages
        """
        try:
            busbar_mask = np.load(self._get_masks_path() / NETWORK_MASK_NAMES["busbar_for_nminus1"])
        except FileNotFoundError:
            logger.info(
                f"No file 'busbar_for_nminus1.npy' in given grid path '{self._get_masks_path()}'. "
                "In this case, busbars are not taken into account in the multi outage N-1 analysis"
            )
            return (
                np.empty((0, self.ppci["branch"].shape[0]), dtype=bool),
                np.empty((0, self.ppci["bus"].shape[0]), dtype=bool),
                [],
                [],
            )
        busbar_outage_names = convert_to_string_list(self.net.bus.name[busbar_mask])
        busbar_outage_id = self.net.bus.index[busbar_mask].tolist()

        busbar_pp_idx = np.flatnonzero(busbar_mask)
        # Translate busses to ppci_format
        busbar_ppci_idx = self.net._pd2ppc_lookups["bus"][busbar_pp_idx]
        busbar_outages = np.zeros((len(busbar_ppci_idx), self.ppci["bus"].shape[0]), dtype=bool)
        busbar_outages[np.arange(len(busbar_ppci_idx)), busbar_ppci_idx] = True
        # Find branches going from or to these bus_idx
        branch_bus_columns = self.ppci["branch"][:, [F_BUS, T_BUS]]
        branch_outages = np.array([(branch_bus_columns == bus_id).any(axis=1) for bus_id in busbar_ppci_idx])
        return branch_outages, busbar_outages, busbar_outage_names, busbar_outage_id

    def get_multi_outage_branches(
        self,
    ) -> Bool[np.ndarray, " n_multi_outages n_branch"]:
        """Get mask of outaged branches for potential multi-outages

        True means a branch is outaged, False means it is not outaged.

        Returns
        -------
        Bool[np.ndarray, " n_multi_outages n_branch"]
            The mask of outaged branches for every multi-outage
        """
        trafo3w_outages, _, _, _ = self.get_trafo3w_multioutage()
        busbar_outages, _, _, _ = self.get_busbar_multioutage()
        return np.concatenate([trafo3w_outages, busbar_outages])

    def get_multi_outage_nodes(
        self,
    ) -> Bool[np.ndarray, " n_multi_outages n_node"]:
        """Get mask of outaged nodes for potential multi-outages

        True means a node is outaged, False means it is not outaged.

        Returns
        -------
        Bool[np.ndarray, " n_multi_outages n_node"]
            The mask of outaged branches for every multi-outage
        """
        _, trafo3w_outages, _, _ = self.get_trafo3w_multioutage()
        _, busbar_outages, _, _ = self.get_busbar_multioutage()
        return np.concatenate([trafo3w_outages, busbar_outages])

    def get_injection_nodes(self) -> Int[np.ndarray, " n_injection"]:
        """Get node index of the injections

        Returns
        -------
        Int[np.ndarray, " n_injection"]
            The node index that the injection injects onto
        """
        injection_node = np.concatenate(
            [
                self.net.gen.bus,
                self.net.sgen.bus,
                self.net.load.bus,
                self.net.shunt.bus,
                self.net.dcline.from_bus,  # from side
                self.net.dcline.to_bus,  # to side
                self.net.ward.bus,  # load part
                self.net.ward.bus,  # shunt part
                self.net.xward.bus,  # load part
                self.net._pd2ppc_lookups["aux"].get("xward", np.array([], dtype=int)),  # gen part
                self.net.xward.bus,  # shunt part
            ]
        )
        injection_node = injection_node[self.get_injection_status()]
        return self.net["_pd2ppc_lookups"]["bus"][injection_node]

    def get_mw_injections(self) -> Float[np.ndarray, " n_timestep n_injection"]:
        """Get MW injections of the injections.

        The timestep dimension is added to represent time-varying injections.
        If the injections are not time-varying, the same value should be returned
        for all timesteps.

        Returns
        -------
        Float[np.ndarray, " n_timestep n_injection"]
            The MW injections of the injections
        """
        injections_mw = self.get_mw_injections_from_net()
        if self.chronics_id is not None:
            injection_types = np.array(self.get_injection_types())
            injections_mw[:, injection_types == "load"] = self.load_p[:, self.net.load.in_service]
            injections_mw[:, injection_types == "gen"] = -self.gen_p[:, self.net.gen.in_service]
            if self.sgen_p is not None:
                injections_mw[:, injection_types == "sgen"] = -self.sgen_p[:, self.net.sgen.in_service]
            if self.dcline_p is not None:
                injections_mw[:, injection_types == "dcline_from"] = self.dcline_p[:, self.net.dcline.in_service]

                # The dc line losses have to be included for the to-side. They are assumed to be constant across timesteps
                loss_percent = self.net.dcline.loss_percent.values[self.net.dcline.in_service.values]
                loss_mw = self.net.dcline.loss_mw.values[self.net.dcline.in_service.values]
                injections_mw[:, injection_types == "dcline_to"] = -1 * (
                    self.dcline_p[:, self.net.dcline.in_service] * (1 - loss_percent[:, None] / 100) - loss_mw[:, None]
                )
        return injections_mw

    def get_injection_status(self) -> Bool[np.ndarray, " n_injection"]:
        """Get injection status in the grid.

        Refering to in service/out of service.
        Includes auxiliary injections from wards, xwards and dclines

        Returns
        -------
        Bool[np.ndarray, " n_injection"]
            The status of each injection element, assumed to be constant over all timesteps
        """
        injection_status = np.concatenate(
            [self.net[injection_type].in_service for injection_type, _ in self.INJECTION_TYPE_MAPPING]
        )
        return injection_status

    def get_outaged_injection_mask(self) -> Bool[np.ndarray, " n_injection"]:
        """Get the injections that are part of the N-1 definition

        Currently only supports failing generators
        """
        try:
            gen_mask = np.load(self._get_masks_path() / NETWORK_MASK_NAMES["generator_for_nminus1"])
        except FileNotFoundError:
            gen_mask = np.zeros(len(self.net.gen), dtype=bool)
        gen_mask = gen_mask[self.net.gen.in_service]

        try:
            sgen_mask = np.load(self._get_masks_path() / NETWORK_MASK_NAMES["sgen_for_nminus1"])
        except FileNotFoundError:
            sgen_mask = np.zeros(len(self.net.sgen), dtype=bool)
        sgen_mask = sgen_mask[self.net.sgen.in_service]

        load_mask = np.zeros(sum(self.net.load.in_service), dtype=bool)
        shunt_mask = np.zeros(sum(self.net.shunt.in_service), dtype=bool)
        dcline_mask = np.zeros(2 * sum(self.net.dcline.in_service), dtype=bool)
        ward_mask = np.zeros(2 * sum(self.net.ward.in_service), dtype=bool)
        xward_mask = np.zeros(3 * sum(self.net.xward.in_service), dtype=bool)

        return np.concatenate(
            [
                gen_mask,
                sgen_mask,
                load_mask,
                shunt_mask,
                dcline_mask,
                ward_mask,
                xward_mask,
            ]
        )

    def get_mw_injections_from_net(
        self,
    ) -> Float[np.ndarray, " n_timestep n_injection"]:
        """Get MW injections stored directly in the network dataframes

        The timestep dimension is added to represent time-varying injections.
        But their is only ever one timestep stored in the grid

        Returns
        -------
        Float[np.ndarray, " n_timestep n_injection"]
            The MW injections of the injections
        """
        dc_bus_voltage = get_dc_bus_voltage(self.net)

        mw_injection_array = np.concatenate(
            [
                -1 * self.net.gen.p_mw,
                -1 * self.net.sgen.p_mw,
                self.net.load.p_mw,
                get_shunt_real_power(
                    dc_bus_voltage.loc[self.net.shunt.bus.values].values,
                    self.net.shunt.p_mw.values,
                    self.net.shunt.vn_kv.values,
                    self.net.shunt.step.values,
                ),
                -1 * -self.net.dcline.p_mw,  # from side
                -1
                * (
                    self.net.dcline.p_mw * (1 - self.net.dcline.loss_percent / 100) - self.net.dcline.loss_mw
                ),  # to side including loss
                self.net.ward.ps_mw,  # load part
                get_shunt_real_power(
                    dc_bus_voltage.loc[self.net.ward.bus.values].values,
                    self.net.ward.pz_mw.values,
                ),  # shunt part of ward
                self.net.xward.ps_mw,  # load part
                np.full(len(self.net.xward), 0),  # gen part
                get_shunt_real_power(
                    dc_bus_voltage.loc[self.net.xward.bus.values].values,
                    self.net.xward.pz_mw.values,
                ),  # shunt part of xward
            ]
        )
        inservice_injections = mw_injection_array[self.get_injection_status()]
        return np.tile(inservice_injections, (self._n_timesteps, 1))

    def get_base_mva(self) -> float:
        """Get baseMVA of the grid

        Returns
        -------
        float
            The base MVA of the grid
        """
        return float(self.ppci["baseMVA"])

    ################################
    # Reporting functions
    def get_node_ids_internal(self) -> list[int]:
        """Get the node IDs as in the pandapower table index.

        This is not necessarily globally unique, hence get_node_ids uses this to produce globally unique ids.

        Returns
        -------
        list[int]
            The node IDs
        """
        bus_idx = self.net.bus.index.values
        aux_bus_idx = np.arange(len(self.net.trafo3w) + len(self.net.xward))
        ppc_bus_idx = np.concatenate([bus_idx, aux_bus_idx])
        return ppc_bus_idx[self._get_ppc_bus_lookup()].tolist()

    def get_node_ids(self) -> list[str]:
        """Get globally unique node IDs.

        You can use parse_globally_unique_id to retrieve table id, table name and element name from
        this string.

        Returns
        -------
        list[str]
            The globally unique node IDs
        """
        node_ids = self.get_node_ids_internal()
        node_types = self.get_node_types()

        return [get_globally_unique_id(node_id, node_type) for node_id, node_type in zip(node_ids, node_types)]

    def get_branch_ids_internal(self) -> list[int]:
        """Get ids of the branches

        Returns
        -------
        list[int]
            The ids of the branches
        """
        ppc_branch_ids = np.concatenate(
            [
                self.net.line.index,
                self.net.trafo.index,
                self.net.trafo3w.index,  # hv
                self.net.trafo3w.index,  # mv
                self.net.trafo3w.index,  # lv
                self.net.impedance.index,
                self.net.xward.index,
            ]
        )
        return ppc_branch_ids[self.net._ppc["internal"]["branch_is"]].tolist()

    def get_branch_ids(self) -> list[str]:
        """Get globally unique ids of the branches

        Returns
        -------
        list[str]
            The globally unique ids of the branches
        """
        branch_ids = self.get_branch_ids_internal()
        branch_types = self.get_branch_types()

        return [get_globally_unique_id(branch_id, branch_type) for branch_id, branch_type in zip(branch_ids, branch_types)]

    def get_injection_ids_internal(self) -> list[int]:
        """Get ids of the injections

        Returns
        -------
        list[int]
            The ids of the injections
        """
        injection_id_array = np.concatenate(
            [self.net[injection_type].index.values for injection_type, _ in self.INJECTION_TYPE_MAPPING]
        )

        return injection_id_array[self.get_injection_status()].tolist()

    def get_injection_ids(self) -> list[str]:
        """Get globally unique ids of the injections

        Returns
        -------
        list[str]
            The globally unique ids of the injections
        """
        injection_ids = self.get_injection_ids_internal()
        injection_types = self.get_injection_types()

        return [
            get_globally_unique_id(injection_id, injection_type)
            for injection_id, injection_type in zip(injection_ids, injection_types)
        ]

    def get_node_names(self) -> list[str]:
        """Get names of the nodes

        Returns
        -------
        list[str]
            The names of the nodes
        """
        ppc_busses = np.concatenate(
            [
                self.net.bus.name.values,
                self.net.trafo3w.name.values,
                self.net.xward.name.values,
            ]
        )
        bus_name_array = ppc_busses[self._get_ppc_bus_lookup()]
        return convert_to_string_list(bus_name_array)

    def get_branch_names(self) -> list[str]:
        """Get names of the branches

        Returns
        -------
        list[str]
            The names of the branches
        """
        ppc_branch_names = np.concatenate(
            [
                self.net.line.name,
                self.net.trafo.name,
                self.net.trafo3w.name + "_hv",
                self.net.trafo3w.name + "_mv",
                self.net.trafo3w.name + "_lv",
                self.net.impedance.name,
                self.net.xward.name,
            ]
        )

        return convert_to_string_list(ppc_branch_names[self.net._ppc["internal"]["branch_is"]])

    def get_injection_names(self) -> list[str]:
        """Get names of the injections

        Returns
        -------
        list[str]
            The names of the injections
        """
        injection_names_array = np.concatenate(
            [self.net[injection_type].name.astype(str) + suffix for injection_type, suffix in self.INJECTION_TYPE_MAPPING]
        )
        return convert_to_string_list(injection_names_array[self.get_injection_status()])

    def get_branch_types(self) -> list[str]:
        """Get type of the branches

        Returns
        -------
        list[str]
            The type of the branches
        """
        ppc_branch_types = np.concatenate(
            [
                np.full(len(self.net.line), "line"),
                np.full(len(self.net.trafo), "trafo"),
                np.full(len(self.net.trafo3w), "trafo3w_hv"),
                np.full(len(self.net.trafo3w), "trafo3w_mv"),
                np.full(len(self.net.trafo3w), "trafo3w_lv"),
                np.full(len(self.net.impedance), "impedance"),
                np.full(len(self.net.xward), "xward"),
            ]
        )
        return convert_to_string_list(ppc_branch_types[self.net._ppc["internal"]["branch_is"]])

    def get_node_types(self) -> list[str]:
        """Get type of the nodes

        Returns
        -------
        list[str]
            The type of the nodes
        """
        ppc_busses = np.concatenate(
            [
                np.full(len(self.net.bus), "bus"),
                np.full(len(self.net.trafo3w), "trafo3w_aux_bus"),
                np.full(len(self.net.xward), "xward_aux_bus"),
            ]
        )
        bus_type_array = ppc_busses[self._get_ppc_bus_lookup()]
        return convert_to_string_list(bus_type_array)

    def get_injection_types(self) -> list[str]:
        """Get type of the injections

        Returns
        -------
        list[str]
            The type of the injections
        """
        injection_types_array = np.concatenate(
            [
                np.full(
                    np.sum(self.net[injection_type].in_service),
                    f"{injection_type}{suffix}",
                )
                for injection_type, suffix in self.INJECTION_TYPE_MAPPING
            ]
        )
        return convert_to_string_list(injection_types_array)

    def get_multi_outage_names(self) -> list[str]:
        """Get names of the multi-outages

        Returns
        -------
        list[str]
            The names of the multi-outages
        """
        _, _, trafo_names, _ = self.get_trafo3w_multioutage()
        _, _, busbar_names, _ = self.get_busbar_multioutage()
        return trafo_names + busbar_names

    def get_multi_outage_ids_internal(self) -> list[int]:
        """Get ids of the multi-outages

        Returns
        -------
        list[int]
            The ids of the multi-outages
        """
        _, _, _, trafo_ids = self.get_trafo3w_multioutage()
        _, _, _, busbar_ids = self.get_busbar_multioutage()
        return trafo_ids + busbar_ids

    def get_multi_outage_ids(self) -> list[str]:
        """Get globally unique ids of the multi-outages

        Returns
        -------
        list[str]
            The globally unique ids of the multi-outages
        """
        multi_outage_ids = self.get_multi_outage_ids_internal()
        multi_outage_types = self.get_multi_outage_types()

        return [
            get_globally_unique_id(multi_outage_id, multi_outage_type)
            for multi_outage_id, multi_outage_type in zip(multi_outage_ids, multi_outage_types)
        ]

    def get_multi_outage_types(self) -> list[str]:
        """Get types of the multi-outages

        Returns
        -------
        list[str]
            The types of the multi-outages
        """
        _, trafo_outages, _, _ = self.get_trafo3w_multioutage()
        _, busbar_outages, _, _ = self.get_busbar_multioutage()
        return ["trafo3w"] * len(trafo_outages) + ["bus"] * len(busbar_outages)

    def get_asset_topology(self) -> Optional[Topology]:
        """Get asset topology of the grid if it exists"""
        if (self.data_path / PREPROCESSING_PATHS["asset_topology_file_path"]).exists():
            return load_asset_topology(self.data_path / PREPROCESSING_PATHS["asset_topology_file_path"])
        return None

    def get_metadata(self) -> dict:
        """Get metadata of the grid

        Returns
        -------
        dict
            The metadata of the grid
        """
        start_datetime = None
        chronics_path = None

        if self.chronics_id is not None:
            if os.path.exists(self._get_logs_path() / "start_datetime.info"):
                chronics_path = self._get_chronics_path()
                with open(
                    self._get_logs_path() / "start_datetime.info",
                    "r",
                    encoding="utf-8",
                ) as file:
                    start_datetime = file.read()

        return {
            "data_path": self.data_path,
            "chronics_id": self.chronics_id,
            "chronics_slice": self.chronics_slice,
            "chronics_path": chronics_path,
            "start_datetime": start_datetime,
            "max_voltage_fluctuation": 0.15,
        }
