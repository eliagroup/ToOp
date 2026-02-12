# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""A set of routines to convert a network data object into a static information object.

Includes the high-level load_grid function which is performing the entire preprocessing.
"""

from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import logbook
import numpy as np
from beartype.typing import Callable, Literal, Optional
from fsspec import AbstractFileSystem
from jaxtyping import Bool, Float, Int
from pypowsybl.loadflow import Parameters as LoadflowParameters
from toop_engine_dc_solver.jax.aggregate_results import (
    aggregate_to_metric,
    compute_double_limits,
    compute_n0_n1_max_diff,
    get_overload_energy_n_1_matrix,
)
from toop_engine_dc_solver.jax.busbar_outage import perform_rel_bb_outage_for_unsplit_grid
from toop_engine_dc_solver.jax.compute_batch import compute_symmetric_batch
from toop_engine_dc_solver.jax.cross_coupler_flow import get_unsplit_flows
from toop_engine_dc_solver.jax.inputs import (
    convert_from_stat_bool,
    convert_tot_stat,
    save_static_information_fs,
    validate_static_information,
)
from toop_engine_dc_solver.jax.nminus2_outage import unsplit_n_2_analysis
from toop_engine_dc_solver.jax.topology_computations import default_topology
from toop_engine_dc_solver.jax.types import (
    BBOutageBaselineAnalysis,
    BranchLimits,
    DynamicInformation,
    MetricType,
    NodalInjectionInformation,
    NodalInjOptimResults,
    NodalInjStartOptions,
    NonRelBBOutageData,
    RelBBOutageData,
    SolverConfig,
    StaticInformation,
    int_max,
)
from toop_engine_dc_solver.jax.utils import HashableArrayWrapper
from toop_engine_dc_solver.postprocess.write_aux_data import write_aux_data_fs
from toop_engine_dc_solver.preprocess.action_set import (
    pad_out_action_set,
)
from toop_engine_dc_solver.preprocess.network_data import NetworkData, save_network_data_fs
from toop_engine_dc_solver.preprocess.pandapower.pandapower_backend import PandaPowerBackend
from toop_engine_dc_solver.preprocess.powsybl.powsybl_backend import PowsyblBackend
from toop_engine_dc_solver.preprocess.preprocess import preprocess
from toop_engine_grid_helpers.powsybl.loadflow_parameters import DISTRIBUTED_SLACK
from toop_engine_interfaces.filesystem_helper import save_pydantic_model_fs
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters
from toop_engine_interfaces.messages.preprocess.preprocess_heartbeat import (
    PreprocessStage,
    empty_status_update_fn,
)
from toop_engine_interfaces.messages.preprocess.preprocess_results import StaticInformationStats

logger = logbook.Logger(__name__)


def convert_relevant_injections(
    injection_idx_at_nodes: list[Int[np.ndarray, " n_injections_at_node"]],
    mw_injections: Float[np.ndarray, " n_timestep n_injection"],
) -> Float[np.ndarray, " n_timesteps n_sub_relevant max_inj_per_sub"]:
    """Convert the relevant injections from network data format to padded jax format

    Parameters
    ----------
    injection_idx_at_nodes : list[Int[np.ndarray, " n_injections_at_node"]]
        The injection indices at each relevant sub
    mw_injections : Float[np.ndarray, " n_timestep n_injection"]
        The injection values

    Returns
    -------
    Float[np.ndarray, " n_timesteps n_sub_relevant max_inj_per_sub"]
        The padded relevant_injections
    """
    max_inj_per_sub = max(len(x) for x in injection_idx_at_nodes)
    n_timesteps = mw_injections.shape[0]
    n_sub_relevant = len(injection_idx_at_nodes)
    relevant_injections = np.zeros((n_timesteps, n_sub_relevant, max_inj_per_sub))
    for i, injections_at_node in enumerate(injection_idx_at_nodes):
        relevant_injections[:, i, : len(injections_at_node)] = mw_injections[:, injections_at_node]

    return relevant_injections


def convert_to_jax(  # noqa: PLR0913
    network_data: NetworkData,
    number_most_affected_n_0: int = 30,
    number_most_affected: int = 30,
    number_max_out_in_most_affected: int = 5,
    batch_size_bsdf: int = 16,
    batch_size_injection: int = 16,
    buffer_size_injection: int = 512,
    limit_n_subs: Optional[int] = None,
    aggregation_metric: Literal[
        "max_flow_n_0",
        "median_flow_n_0",
        "overload_energy_n_0",
        "underload_energy_n_0",
        "max_flow_n_1",
        "median_flow_n_1",
        "overload_energy_n_1",
        "underload_energy_n_1",
    ] = "overload_energy_n_1",
    distributed: bool = False,
    enable_n_2: bool = False,
    n_2_more_splits_penalty: float = 1000.0,
    enable_bb_outage: bool = False,
    bb_outage_as_nminus1: bool = True,
    bb_outage_more_splits_penalty: float = 2000.0,
    clip_bb_outage_penalty: bool = False,
    ac_dc_interpolation: float = 0.0,
    logging_fn: Optional[Callable[[PreprocessStage, Optional[str]], None]] = None,
) -> StaticInformation:
    """Convert the finalized network data into static info for the solver

    Parameters
    ----------
    network_data : NetworkData
        The network data to convert
    number_most_affected_n_0 : int, optional
        How many top-k N-0 cases to store
    number_most_affected : int, optional
        How many top-k N-1 cases to store
    number_max_out_in_most_affected : int, optional
        How many of the overall top N-1 cases can be from one outage
    batch_size_bsdf : int, optional
        The batch size for the BSDF computation
    batch_size_injection : int, optional
        The batch size for the injection computation
    buffer_size_injection : int, optional
        The buffer size for the injection computation
    limit_n_subs : int, optional
        How many splits can maximally be expected
    aggregation_metric : Literal[
            "max_flow_n_0",
            "median_flow_n_0",
            "overload_energy_n_0",
            "underload_energy_n_0",
            "max_flow_n_1",
            "median_flow_n_1",
            "overload_energy_n_1",
            "underload_energy_n_1",
        ], optional
        The metric to use for the aggregation, see aggregate_to_metric for more details
    distributed: bool, optional
        Whether to use the distributed version of the solver
    enable_n_2: bool, optional
        Whether to enable the N-2 analysis feature
    n_2_more_splits_penalty: float, optional
        How to penalize additional splits in N-2 that were not there in the unsplit grid. Will be
        added to the overload energy penalty.
    enable_bb_outage: bool, optional
        Whether to enable the busbar outage feature
    bb_outage_as_nminus1: bool, optional
        Whether to compute the busbar outage as N-1 or not. If False, the busbar outage
        compared of split grid is compared with the unsplit grid
    bb_outage_more_splits_penalty: float, optional
        How to penalize additional splits in busbar outage that were not there in the unsplit grid. Will be
        added to the overload energy penalty.
    clip_bb_outage_penalty: bool, optional
        Whether to clip the lower bound of busbar outage penalty to 0.
    ac_dc_interpolation: float, optional
        The interpolation factor for AC/DC mismatch, by default 0.0 (full DC).
    logging_fn: Callable, optional
        A function to call to signal progress in the preprocessing pipeline. Takes a stage and an
        optional message as parameters, by default None

    Returns
    -------
    StaticInformation
        The StaticInformation required by the solver

    """
    if logging_fn is None:
        logging_fn = empty_status_update_fn
    logging_fn("convert_to_jax_started", None)

    if not jax.config.read("jax_enable_x64"):
        logger.warning("jax_enable_x64 is set to False. This means the grid data will be converted to float32")

    n_relevant_nodes = len(network_data.branches_at_nodes)
    n_original_nodes = int(network_data.ptdf.shape[1] - n_relevant_nodes)

    # Convert (more complex) arrays to jax format
    logging_fn("convert_tot_stat", f"Converting {len(network_data.branches_at_nodes)} nodes")
    branches_at_nodes = convert_tot_stat(network_data.branches_at_nodes)
    branch_direction = convert_from_stat_bool(network_data.branch_direction)

    logging_fn("convert_relevant_inj", None)
    relevant_injections = jnp.array(
        convert_relevant_injections(
            network_data.injection_idx_at_nodes,
            network_data.mw_injections,
        )
    )

    # Convert masks to idx in jax format
    logging_fn("convert_masks", None)
    branches_to_fail = jnp.flatnonzero(network_data.outaged_branch_mask)
    disconnectable_branches = jnp.flatnonzero(network_data.disconnectable_branch_mask)
    branches_monitored = jnp.flatnonzero(network_data.monitored_branch_mask)

    rel_stat_map = HashableArrayWrapper(np.flatnonzero(network_data.relevant_node_mask))
    max_mw_flows = jnp.array(network_data.max_mw_flows[0, branches_monitored])
    max_mw_flows_n_1 = jnp.array(network_data.max_mw_flows_n_1[0, branches_monitored])
    overload_weights = jnp.array(network_data.overload_weights[branches_monitored])
    n0_n1_max_diff_factors = jnp.array(network_data.n0_n1_max_diff_factors[branches_monitored])
    susceptance = jnp.array(network_data.susceptances)
    shift_degree_min = jnp.array([min(taps) for taps in network_data.phase_shift_taps])
    shift_degree_max = jnp.array([max(taps) for taps in network_data.phase_shift_taps])

    pst_n_taps = jnp.array([len(taps) for taps in network_data.phase_shift_taps])
    max_pst_n_taps = int(jnp.max(pst_n_taps) if pst_n_taps.size > 0 else 0)
    pst_tap_values = jnp.array(
        [
            jnp.pad(jnp.array(taps), (0, max_pst_n_taps - len(taps)), "constant", constant_values=jnp.nan)
            for taps in network_data.phase_shift_taps
        ]
    )

    logging_fn("pad_out_branch_actions", None)
    assert network_data.branch_action_set is not None, "Please compute branch action set first!"
    assert network_data.branch_action_set_switching_distance is not None, "Please compute switching distance first!"
    assert network_data.injection_action_set is not None, "Please compute injection action set first!"

    action_set = pad_out_action_set(
        branch_actions=network_data.branch_action_set,
        injection_actions=network_data.injection_action_set,
        reassignment_distance=network_data.branch_action_set_switching_distance,
    )

    if enable_bb_outage:
        logging_fn("convert_rel_bb_outage_data", None)
        action_set = replace(action_set, rel_bb_outage_data=convert_rel_bb_outage_data(network_data))

    logging_fn("create_static_information", None)
    static_information = StaticInformation(
        dynamic_information=DynamicInformation(
            # Network Data arguments
            from_node=jnp.array(network_data.from_nodes, dtype=int),
            to_node=jnp.array(network_data.to_nodes, dtype=int),
            ptdf=jnp.array(network_data.ptdf),
            generators_per_sub=jnp.array(network_data.num_injections_per_node, dtype=int),
            branch_limits=BranchLimits(
                max_mw_flow=max_mw_flows,
                max_mw_flow_n_1=(max_mw_flows_n_1 if not jnp.allclose(max_mw_flows, max_mw_flows_n_1) else None),
                overload_weight=(overload_weights if jnp.any(overload_weights != 1) else None),
                # Store the factors first, extract_static_information will convert that to absolute
                # values.
                n0_n1_max_diff=n0_n1_max_diff_factors,
                coupler_limits=jnp.array(network_data.cross_coupler_limits),
            ),
            tot_stat=branches_at_nodes,
            from_stat_bool=branch_direction,
            susceptance=susceptance,
            relevant_injections=relevant_injections,
            nodal_injections=jnp.array(network_data.nodal_injection, dtype=float),
            branches_to_fail=branches_to_fail,
            disconnectable_branches=disconnectable_branches,
            # Solver arguments
            action_set=action_set,
            multi_outage_branches=[jnp.array(x, dtype=int) for x in network_data.split_multi_outage_branches],
            multi_outage_nodes=[jnp.array(x, dtype=int) for x in network_data.split_multi_outage_nodes],
            nonrel_injection_outage_deltap=jnp.array(network_data.nonrel_io_deltap, dtype=float),
            nonrel_injection_outage_node=jnp.array(network_data.nonrel_io_node, dtype=int),
            relevant_injection_outage_idx=jnp.array(network_data.rel_io_local_inj_index, dtype=int),
            relevant_injection_outage_sub=jnp.array(network_data.rel_io_sub, dtype=int),
            unsplit_flow=get_unsplit_flows(
                ptdf=network_data.ptdf,
                nodal_injections=network_data.nodal_injection,
                ac_dc_mismatch=network_data.ac_dc_mismatch,
                ac_dc_interpolation=ac_dc_interpolation,
            ),
            branches_monitored=branches_monitored,
            n2_baseline_analysis=None,
            non_rel_bb_outage_data=convert_non_rel_bb_outage(network_data) if enable_bb_outage else None,
            bb_outage_baseline_analysis=None,
            nodal_injection_information=NodalInjectionInformation(
                controllable_pst_indices=jnp.flatnonzero(network_data.controllable_pst_node_mask),
                shift_degree_min=shift_degree_min,
                shift_degree_max=shift_degree_max,
                pst_n_taps=pst_n_taps,
                pst_tap_values=pst_tap_values,
                starting_tap_idx=jnp.array(network_data.phase_shift_starting_tap_idx, dtype=int),
                grid_model_low_tap=jnp.array(network_data.phase_shift_low_tap, dtype=int),
            )
            if network_data.controllable_pst_node_mask.any()
            else None,
        ),
        solver_config=SolverConfig(
            branches_per_sub=HashableArrayWrapper(network_data.num_branches_per_node),
            slack=int(network_data.slack),
            n_stat=n_original_nodes,
            rel_stat_map=rel_stat_map,
            number_most_affected_n_0=number_most_affected_n_0,
            number_most_affected=number_most_affected,
            number_max_out_in_most_affected=number_max_out_in_most_affected,
            batch_size_bsdf=batch_size_bsdf,
            batch_size_injection=batch_size_injection,
            buffer_size_injection=buffer_size_injection,
            limit_n_subs=limit_n_subs,
            aggregation_metric=aggregation_metric,
            distributed=distributed,
            enable_bb_outages=enable_bb_outage,
            bb_outage_as_nminus1=bb_outage_as_nminus1,
            clip_bb_outage_penalty=clip_bb_outage_penalty,
            contingency_ids=network_data.contingency_ids,
        ),
    )

    if enable_n_2:
        logging_fn("unsplit_n2_analysis", None)
        n_2_baseline = unsplit_n_2_analysis(
            dynamic_information=static_information.dynamic_information,
            more_splits_penalty=n_2_more_splits_penalty,
        )
        static_information = replace(
            static_information,
            dynamic_information=replace(
                static_information.dynamic_information,
                n2_baseline_analysis=n_2_baseline,
            ),
        )
    if enable_bb_outage and not bb_outage_as_nminus1:
        # A comparision of the overload energy of the unsplit grid with the overload energy of the split grid
        # after busbar outages is required. Therefore, we need to store the baseline loadflows after busbar outages
        # of unsplit grid.
        logging_fn("bb_outage_baseline_analysis", None)
        static_information = replace(
            static_information,
            dynamic_information=replace(
                static_information.dynamic_information,
                bb_outage_baseline_analysis=get_bb_outage_baseline_analysis(
                    di=static_information.dynamic_information,
                    more_splits_penalty=bb_outage_more_splits_penalty,
                ),
            ),
        )

    return static_information


def get_bb_outage_baseline_analysis(di: DynamicInformation, more_splits_penalty: float) -> BBOutageBaselineAnalysis:
    """Get the baseline loadflows after busbar outages of unsplit grid.

    Parameters
    ----------
    di : DynamicInformation
        The dynamic information dataclass
    more_splits_penalty : Float[Array, " "]
        A scalar value to scale the difference between the success counts of the unsplit grid
        and the split grid.

    Returns
    -------
    BBOutageBaselineAnalysis
        The baseline loadflows after busbar outages of unsplit grid
    """
    lfs, success = perform_rel_bb_outage_for_unsplit_grid(
        di.unsplit_flow, di.ptdf, di.nodal_injections, di.from_node, di.to_node, di.action_set, di.branches_monitored
    )

    if not jnp.all(success):
        logger.warning(f"Baseline calculation for bb outage not successful: {jnp.sum(success)}/{len(success)} successful")

    overload = get_overload_energy_n_1_matrix(
        n_1_matrix=jnp.transpose(lfs, (1, 0, 2)),
        max_mw_flow=di.branch_limits.max_mw_flow,
        overload_weight=di.branch_limits.overload_weight,
        aggregate_strategy="nanmax",
    )
    return BBOutageBaselineAnalysis(
        overload=overload,
        success_count=jnp.sum(success),
        more_splits_penalty=jnp.array(more_splits_penalty),
        overload_weight=di.branch_limits.overload_weight,
        max_mw_flow=di.branch_limits.max_mw_flow,
    )


def convert_non_rel_bb_outage(
    network_data: NetworkData,
) -> NonRelBBOutageData:
    """Convert non-relevant busbar outage data into padded JAX arrays.

    Parameters
    ----------
    network_data : NetworkData
        The network data containing information about non-reliable busbar outages,
        including delta power, branch indices, and nodal indices.

    Returns
    -------
    NonRelBBOutageData
        A structured data object containing padded and converted non-relevant busbar outage data.

    Notes
    -----
    - The branch indices are padded to the maximum number of branches per substation
      using a placeholder value (`int_max`).
    """
    outage_deltap = jnp.array(network_data.non_rel_bb_outage_deltap)
    outage_branches = network_data.non_rel_bb_outage_br_indices
    max_branches_per_sub = max(len(branches) for branches in network_data.branches_at_nodes)
    padded_outage_branches = np.full((outage_deltap.shape[0], max_branches_per_sub), int_max(), dtype=int)

    for i, branches in enumerate(outage_branches):
        padded_outage_branches[i, : len(branches)] = branches

    return NonRelBBOutageData(
        branch_outages=jnp.array(padded_outage_branches),
        deltap=outage_deltap,
        nodal_indices=jnp.array(network_data.non_rel_bb_outage_nodal_indices),
    )


# TODO: refactor due to C901
def convert_rel_bb_outage_data(  # noqa: C901
    network_data: NetworkData,
) -> RelBBOutageData:
    """Convert busbar rel_bb_outage data to a structured format suitable for JAX operations.

    Parameters
    ----------
    network_data : NetworkData
        The network data object containing additional information.

    Returns
    -------
    RelBBOutageData
        A structured data object containing padded and converted outage and action data.
    """
    branch_action_combis_all_rel_subs = network_data.branch_action_set
    actions_per_sub = [sub.shape[0] for sub in branch_action_combis_all_rel_subs]
    cum_sum_actions_per_sub = np.cumsum(actions_per_sub)
    rel_bb_outage_br_indices = network_data.rel_bb_outage_br_indices
    rel_bb_outage_deltap = network_data.rel_bb_outage_deltap
    rel_bb_outage_nodal_indices = network_data.rel_bb_outage_nodal_indices
    n_timesteps = network_data.nodal_injection.shape[0]

    # Determine dimensions for padding of branch_outage_set
    n_actions = sum(actions_per_sub)  # Total number of combinations
    n_max_bb_to_outage_per_sub = max(len(combi) for sub in rel_bb_outage_br_indices for combi in sub)
    max_branches_per_sub = max(len(branches) for branches in network_data.branches_at_nodes)

    # Initialize the padded array with a sentinel value (int_max)
    max_val = int_max()
    padded_branch_outage_set = max_val * np.ones((n_actions, n_max_bb_to_outage_per_sub, max_branches_per_sub), dtype=int)
    padded_delta_p_set = np.zeros((n_actions, n_max_bb_to_outage_per_sub, n_timesteps), dtype=float)
    padded_nodal_index_set = max_val * np.ones((n_actions, n_max_bb_to_outage_per_sub), dtype=int)
    padded_articulation_node_mask = np.zeros((n_actions, n_max_bb_to_outage_per_sub), dtype=bool)

    def fill_padded_array(
        padded_array: np.ndarray,
        data: list[list],
        fill_fn: Callable[[np.ndarray, int, list], np.ndarray],
    ) -> jnp.ndarray:
        """Fill a padded array with data using a specified fill function.

        This function iterates over a nested data structure and applies a
        fill function to populate a padded array with the provided data. This
        is a generic method that is used to fill bb_outage data for rel_subs.

        padded_array : np.ndarray
            The array to be filled with data. This array is expected to have
            sufficient space to accommodate the data.
        data : list of lists
            A nested list where each sublist contains the data to be inserted
            into the padded array.
        fill_fn : Callable[[np.ndarray, int, list], np.ndarray]
            A function that takes three arguments: the padded array, an index,
            and a data element. It is responsible for inserting the data element
            into the padded array at the specified index.

        Returns
        -------
        jax.numpy.ndarray
            A JAX array containing the filled data.

        Notes
        -----
        - The function assumes that `cum_sum_actions_per_sub` is defined in the
          scope where this function is used. This variable is expected to
          contain cumulative sums of the lengths of sublists in `data`.
        - If a sublist in `data` is empty, it is skipped during the filling
          process.
        """
        for sub_idx, sub_combis in enumerate(data):
            if len(sub_combis) == 0:
                continue
            start_combi_idx = 0 if sub_idx == 0 else cum_sum_actions_per_sub[sub_idx - 1]
            for combi_idx, combi in enumerate(sub_combis):
                padded_array = fill_fn(padded_array, int(start_combi_idx + combi_idx), combi)
        return jnp.array(padded_array)

    def fill_branch_outage_set(
        padded_array: Int[np.ndarray, " n_actions n_max_bb_to_outage_per_sub max_branches_per_sub"],
        action_idx: int,
        branch_indices_all_bbs: list[list[int]],
    ) -> Int[np.ndarray, " n_actions n_max_bb_to_outage_per_sub max_branches_per_sub"]:
        """Populate a padded array with branch_indices to be outaged corresponding to busbar_outages.

        padded_array : Int[np.ndarray, " n_actions n_max_bb_to_outage_per_sub max_branches_per_sub"]
            A pre-allocated array to be filled with branch outage data.
        action_idx : int
            The index where the branch_outage_data corresponding to a particular branch_action should be
            stored in the padded array. This index indexes into the first dimension of the
            dynamic_information.action_set
        branch_indices_all_bbs : list[list[int]]
            The outer list is of length equal to the number of
            physical busbars that have to be outaged in the station for which
            this method is called. The inner lists contain the indices of the
            branches that are outaged in the respective combination.

        Returns
        -------
        Int[np.ndarray, " n_actions n_max_bb_to_outage_per_sub max_branches_per_sub"]
            The padded array with the branch outage combinations filled in.
        """
        for bb_idx, branch_indices in enumerate(branch_indices_all_bbs):
            padded_array[action_idx, bb_idx, : len(branch_indices)] = branch_indices
        return padded_array

    def fill_delta_p_set(
        padded_array: Float[np.ndarray, " n_actions n_max_bb_to_outage_per_sub n_timesteps"],
        action_idx: int,
        deltap_all_bbs: list[Float[np.ndarray, " n_max_bb_to_outage_per_sub"]],
    ) -> Float[np.ndarray, " n_actions n_max_bb_to_outage_per_sub n_timesteps"]:
        """Fill a padded array with delta P values for a specific action_idx.

        Parameters
        ----------
        padded_array : Float[np.ndarray, shape=(n_actions, n_max_bb_to_outage_per_sub, n_timesteps)]
            A 3D array preallocated with padding to store delta P values.
        action_idx : int
            The index where the delta_p data corresponding to a particular branch_action should be
            stored in the padded array. This index indexes into the first dimension of the
            dynamic_information.action_set
        deltap_all_bbs : list[list[float]]
            The outer list is of length equal to the maximum number of bbs in the station.
            The next innter list is of length equal to the number of timestamps.

        Returns
        -------
        Float[np.ndarray, shape=(n_actions, n_max_bb_to_outage_per_sub, n_timesteps)]
            The updated padded array with delta P values filled.
        """
        for bb_idx, deltap in enumerate(deltap_all_bbs):
            padded_array[action_idx, bb_idx, : len(deltap)] = deltap
        return padded_array

    def fill_nodal_index_set(
        padded_array: Int[np.ndarray, " n_actions n_max_bb_to_outage_per_sub"], action_idx: int, nodal_idx_all_bbs: list[int]
    ) -> Int[np.ndarray, " n_actions n_max_bb_to_outage_per_sub"]:
        """Update a padded array with nodal indices for a specific action_idx.

        Parameters
        ----------
        padded_array : Int[np.ndarray, "n_actions n_max_bb_to_outage_per_sub"]
            A boolean array with dimensions `(n_actions, n_max_bb_to_outage_per_sub)`
            that will be updated with nodal indices.
        action_idx : int
            The index where the nodal_index of the busbar corresponding to a particular branch_action should be
            stored in the padded array. This index indexes into the first dimension of the
            dynamic_information.action_set
        nodal_idx_all_bbs : list[int]
            Stores the nodal indices of the busbars that are outaged in the respective combination.
            The outer list is of length equal to the maximum number of bbs in the station.

        Returns
        -------
        Int[np.ndarray, "n_actions n_max_bb_to_outage_per_sub"]
            The updated padded array with nodal indices filled for the specified global combination index.
        """
        for bb_idx, nodal_index in enumerate(nodal_idx_all_bbs):
            if nodal_index is not None:
                padded_array[action_idx, bb_idx] = nodal_index
        return padded_array

    def fill_articulation_node_mask(
        padded_array: Bool[np.ndarray, "n_actions n_max_bb_to_outage_per_sub"], action_idx: int, articulation_bbs: list[int]
    ) -> Bool[np.ndarray, "n_actions n_max_bb_to_outage_per_sub"]:
        """Update a padded boolean array to mark articulation nodes for a specific action represented by the action_idx.

        Parameters
        ----------
        padded_array : Bool[np.ndarray, "n_actions n_max_bb_to_outage_per_sub"]
            A 2D boolean array where each row corresponds to an action set, and each column
            represents a potential articulation node. This array is updated in-place.
        action_idx : int
            The index of the action.
        articulation_bbs : list[int]
            A list of busbar indices representing the articulation nodes to be marked as `True`.

        Returns
        -------
        Bool[np.ndarray, "n_actions n_max_bb_to_outage_per_sub"]
            The updated padded boolean array with the specified articulation nodes marked
            as `True`.
        """
        padded_array[action_idx, articulation_bbs] = True
        return padded_array

    padded_branch_outage_set = fill_padded_array(padded_branch_outage_set, rel_bb_outage_br_indices, fill_branch_outage_set)
    padded_delta_p_set = fill_padded_array(padded_delta_p_set, rel_bb_outage_deltap, fill_delta_p_set)
    padded_nodal_index_set = fill_padded_array(padded_nodal_index_set, rel_bb_outage_nodal_indices, fill_nodal_index_set)
    padded_articulation_node_mask = fill_padded_array(
        padded_articulation_node_mask, network_data.rel_bb_articulation_nodes, fill_articulation_node_mask
    )

    return RelBBOutageData(
        branch_outage_set=padded_branch_outage_set,
        deltap_set=padded_delta_p_set,
        nodal_indices=padded_nodal_index_set,
        articulation_node_mask=padded_articulation_node_mask,
    )


def load_grid(
    data_folder_dirfs: AbstractFileSystem,
    chronics_id: Optional[int] = None,
    timesteps: Optional[slice] = None,
    pandapower: bool = False,
    parameters: Optional[PreprocessParameters] = None,
    status_update_fn: Optional[Callable[[PreprocessStage, Optional[str]], None]] = None,
    lf_params: Optional[LoadflowParameters | dict] = None,
) -> tuple[StaticInformationStats, StaticInformation, NetworkData]:
    """Load the grid and preprocess it

    Parameters
    ----------
    data_folder_dirfs : AbstractFileSystem
        A filesystem which is assumed to be a dirfs pointing to the root for this import job. I.e. the folder structure
        as defined in toop_engine_interfaces.folder_structure is expected to start at root in this filesystem.
    chronics_id : Optional[int], optional
        The chronics id to use, by default None. Note that powsybl does not support chronics yet
    timesteps : Optional[slice], optional
        The timesteps to use, by default None. Note that powsybl does not support timesteps yet
    pandapower : bool, optional
        Whether to use pandapower as backend, by default False
    parameters : Optional[PreprocessParameters], optional
        The parameters to use for the preprocess and convert_to_jax functions. If None, the default
        parameters are used.
    status_update_fn : Optional[Callable[[PreprocessStage, Optional[str]], None]], optional
        A function to call to signal progress in the preprocessing pipeline. Takes a stage and an
        optional message as parameters, by default None
    lf_params : Optional[LoadflowParameters], optional
        The loadflow parameters to use for the initial loadflow calculation. If None, the default parameters are used.

    Returns
    -------
    StaticInformationStats
        Some information about the grid
    StaticInformation
        The populated static information dataclass for the solver
    NetworkData
        The network data object containing additional information
    """
    jax.clear_caches()
    if status_update_fn is None:
        status_update_fn = empty_status_update_fn
    if parameters is None:
        parameters = PreprocessParameters()
    if lf_params is None and not pandapower:
        lf_params = DISTRIBUTED_SLACK
    elif lf_params is None and pandapower:
        lf_params = {}

    if pandapower:
        status_update_fn("load_grid_into_loadflow_solver_backend", "load into PandaPower backend")
        backend = PandaPowerBackend(data_folder_dirfs=data_folder_dirfs, chronics_id=chronics_id, chronics_slice=timesteps)
    else:
        status_update_fn("load_grid_into_loadflow_solver_backend", "load into Powsybl backend")
        backend = PowsyblBackend(
            data_folder_dirfs=data_folder_dirfs,
            lf_params=lf_params,
            fail_on_non_convergence=parameters.fail_on_non_convergence,
        )
    network_data = preprocess(backend, logging_fn=status_update_fn, parameters=parameters)
    static_information = convert_to_jax(
        network_data=network_data,
        logging_fn=status_update_fn,
        enable_n_2=parameters.enable_n_2,
        n_2_more_splits_penalty=parameters.n_2_more_splits_penalty,
        enable_bb_outage=parameters.enable_bb_outage,
        bb_outage_as_nminus1=parameters.bb_outage_as_nminus1,
        bb_outage_more_splits_penalty=parameters.bb_outage_more_splits_penalty,
        ac_dc_interpolation=parameters.ac_dc_interpolation,
    )

    validate_static_information(static_information)
    status_update_fn("compute_base_loadflows", "compute_base_loadflows")
    static_information, (overload_n0, overload_n1) = run_initial_loadflow(
        static_information,
        lower_limit_n_0=parameters.double_limit_n0,
        lower_limit_n_1=parameters.double_limit_n1,
    )

    info = extract_static_information_stats(
        static_information,
        overload_n0,
        overload_n1,
        network_data.metadata.get("start_datetime", ""),
    )

    status_update_fn("save_artifacts", "Saving to filesystem")
    data_folder_dirfs.makedirs(Path(PREPROCESSING_PATHS["static_information_file_path"]).parent.as_posix(), exist_ok=True)
    save_static_information_fs(
        PREPROCESSING_PATHS["static_information_file_path"], static_information, filesystem=data_folder_dirfs
    )

    data_folder_dirfs.makedirs(Path(PREPROCESSING_PATHS["network_data_file_path"]).parent.as_posix(), exist_ok=True)
    save_network_data_fs(
        filesystem=data_folder_dirfs, filename=PREPROCESSING_PATHS["network_data_file_path"], network_data=network_data
    )
    write_aux_data_fs(filesystem=data_folder_dirfs, network_data=network_data)

    save_pydantic_model_fs(
        filesystem=data_folder_dirfs,
        file_path=PREPROCESSING_PATHS["static_information_stats_file_path"],
        pydantic_model=info,
    )

    return info, static_information, network_data


def extract_static_information_stats(
    static_information: StaticInformation,
    overload_n0: Optional[float] = None,
    overload_n1: Optional[float] = None,
    time: Optional[str] = None,
) -> StaticInformationStats:
    """Extract some stats about the static information dataclass

    Parameters
    ----------
    static_information : StaticInformation
        The static information dataclass
    overload_n0 : Optional[float]
        The overload energy of the unsplit grid, use run_initial_loadflow to determine
    overload_n1 : Optional[float]
        The overload energy of the unsplit grid, use run_initial_loadflow to determine
    time : str
        The timestep, this is stored in network_data

    Returns
    -------
    StaticInformationStats
        The extracted stats
    """
    di = static_information.dynamic_information
    config = static_information.solver_config

    return StaticInformationStats(
        time=time,
        fp_dtype=str(di.ptdf.dtype),
        has_double_limits=di.branch_limits.max_mw_flow_n_1_limited is not None,
        n_branches=static_information.n_branches,
        n_nodes=static_information.n_nodes,
        n_branch_outages=static_information.n_outages,
        n_multi_outages=static_information.n_multi_outages,
        n_injection_outages=static_information.n_inj_failures,
        n_busbar_outages=di.n_bb_outages,
        n_controllable_psts=di.n_controllable_pst,
        n_nminus1_cases=di.n_nminus1_cases,
        n_monitored_branches=static_information.n_branches_monitored,
        n_timesteps=static_information.n_timesteps,
        n_relevant_subs=static_information.n_sub_relevant,
        n_disc_branches=di.n_disconnectable_branches,
        overload_energy_n0=overload_n0 or 0.0,
        overload_energy_n1=overload_n1 or 0.0,
        n_actions=len(di.action_set.branch_actions),
        max_station_branch_degree=config.branches_per_sub.val.max().item(),
        max_station_injection_degree=di.generators_per_sub.max().item(),
        mean_station_branch_degree=config.branches_per_sub.val.mean().item(),
        mean_station_injection_degree=di.generators_per_sub.mean().item(),
        reassignable_branch_assets=config.branches_per_sub.val.sum().item(),
        reassignable_injection_assets=di.generators_per_sub.sum().item(),
        max_reassignment_distance=di.action_set.reassignment_distance.max().item(),
    )


def run_initial_loadflow(
    static_information: StaticInformation,
    lower_limit_n_0: Optional[float] = 0.9,
    lower_limit_n_1: Optional[float] = 0.9,
    metrics: tuple[MetricType, ...] = ("overload_energy_n_0", "overload_energy_n_1"),
) -> tuple[StaticInformation, tuple[float, ...]]:
    """Run one initial loadflow computation with the unsplit grid

    Parameters
    ----------
    static_information : StaticInformation
        The static information dataclass
    lower_limit_n_0 : Optional[float], optional
        The lower limit for the n-0 branch limits, by default 0.9. If None, no lower limit is
        computed
    lower_limit_n_1 : Optional[float], optional
        The lower limit for the n-1 branch limits, by default 0.9. If None, no lower limit is
        computed
    metrics : tuple[MetricType], optional
        The metric to use for aggregation, by default "overload_energy_n_1/n_0". If you pass
        multiple metrics, all of them will be computed and returned

    Returns
    -------
    StaticInformation
        The updated static information dataclass with the branch limits computed
    tuple[float]
        The aggregated metrics for the unsplit grid
    """
    orig_batch_size = static_information.solver_config.batch_size_bsdf
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=1),
    )

    topo = default_topology(static_information.solver_config)

    # Prepare starting options for nodal injection optimization if enabled
    nodal_inj_start_options = None
    if static_information.dynamic_information.nodal_injection_information is not None:
        nodal_inj_start_options = NodalInjStartOptions(
            previous_results=NodalInjOptimResults(
                pst_tap_idx=static_information.dynamic_information.nodal_injection_information.starting_tap_idx[
                    None, None, :
                ]
            ),
            precision_percent=0.0,
        )

    lf_res, success = compute_symmetric_batch(
        topology_batch=topo,
        disconnection_batch=None,
        injections=None,
        nodal_inj_start_options=nodal_inj_start_options,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
    )
    assert jnp.all(success)
    n_0 = lf_res.n_0_matrix
    n_1 = lf_res.n_1_matrix

    static_information = replace(
        static_information,
        dynamic_information=replace(
            static_information.dynamic_information,
            branch_limits=replace(
                static_information.dynamic_information.branch_limits,
                max_mw_flow_limited=(
                    compute_double_limits(
                        n_0[0, :, None, :],
                        static_information.dynamic_information.branch_limits.max_mw_flow,
                        lower_limit=lower_limit_n_0,
                    )
                    if lower_limit_n_0 is not None
                    else None
                ),
                max_mw_flow_n_1_limited=(
                    compute_double_limits(
                        n_1[0],
                        (
                            static_information.dynamic_information.branch_limits.max_mw_flow_n_1
                            if static_information.dynamic_information.branch_limits.max_mw_flow_n_1 is not None
                            else static_information.dynamic_information.branch_limits.max_mw_flow
                        ),
                        lower_limit=lower_limit_n_1,
                    )
                    if lower_limit_n_1 is not None
                    else None
                ),
                n0_n1_max_diff=compute_n0_n1_max_diff(
                    n_0[0],
                    n_1[0],
                    static_information.dynamic_information.branch_limits.n0_n1_max_diff,
                ),
            ),
        ),
        solver_config=replace(
            static_information.solver_config,
            batch_size_bsdf=orig_batch_size,
        ),
    )

    agg = partial(
        aggregate_to_metric,
        lf_res=lf_res[0],
        branch_limits=static_information.dynamic_information.branch_limits,
        n_relevant_subs=static_information.n_sub_relevant,
        reassignment_distance=static_information.dynamic_information.action_set.reassignment_distance,
    )
    aggregate = tuple(float(agg(metric=metric)) for metric in metrics)

    return static_information, aggregate
