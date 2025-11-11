"""Provides utilities for handling static information and topology computations.

Instead of storing the topology computations in a tree, we omit this optimization and
use a plain large array of topology computations. If we were to use a tree, we would have to
store a copy of the ptdf matrix at every node in the tree, which would become very large very
quickly. Instead, we assume that the number of bus splits that are typically combined is rather
limited (up to 10) and the bsdf computation is quite fast, so recomputing it is not a problem.
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import io
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
from beartype.typing import BinaryIO, Iterator, Optional
from jax import numpy as jnp  # pylint: disable=no-name-in-module
from jaxtyping import Array, Bool, Int
from toop_engine_dc_solver.jax.types import (
    ActionSet,
    BBOutageBaselineAnalysis,
    BranchLimits,
    DynamicInformation,
    N2BaselineAnalysis,
    NonRelBBOutageData,
    RelBBOutageData,
    SolverConfig,
    StaticInformation,
)
from toop_engine_dc_solver.jax.utils import HashableArrayWrapper


def convert_tot_stat(
    tot_stat: list[Int[np.ndarray, " n_branches_at_sub"]],
) -> Int[Array, " n_sub_relevant max_branch_per_sub"]:
    """Convert the tot_stat arrays from the numpy code to the format used in the jax code.

    Parameters
    ----------
    tot_stat : list[np.ndarray]
        The tot_stat array from the numpy code

    Returns
    -------
    Int[Array, " n_sub_relevant max_branch_per_sub"]
        The converted tot_stat array
    """
    # Find out int max, int could be both 32 or 64 bits
    int_max = jnp.iinfo(jnp.array([1], dtype=int).dtype).max

    c_l = np.array([len(x) for x in tot_stat])

    # Pad tot_stat and from_stat_bool with int_max to max_branch_per_sub
    tot_stat_jax = jnp.full(
        (c_l.shape[0], jnp.max(c_l)),
        fill_value=int_max,
        dtype=int,
    )
    for sub_id in range(c_l.shape[0]):
        tot_stat_jax = tot_stat_jax.at[sub_id, : c_l[sub_id]].set(tot_stat[sub_id])

    return tot_stat_jax


def convert_from_stat_bool(
    from_stat_bool: list[Bool[np.ndarray, " n_branches_at_sub"]],
) -> Bool[Array, " n_sub_relevant max_branch_per_sub"]:
    """Convert the from_stat_bool array from the numpy code to the format used in the jax code

    Parameters
    ----------
    from_stat_bool : list[np.ndarray]
        The from_stat_bool array from the numpy code

    Returns
    -------
    Bool[Array, " n_sub_relevant max_branch_per_sub"]
        The converted from_stat_bool array
    """
    c_l = np.array([len(x) for x in from_stat_bool])

    # Pad from_stat_bool with int_max to max_branch_per_sub
    from_stat_bool_jax = jnp.zeros((c_l.shape[0], jnp.max(c_l)), dtype=bool)
    for sub_id in range(c_l.shape[0]):
        from_stat_bool_jax = from_stat_bool_jax.at[sub_id, : c_l[sub_id]].set(from_stat_bool[sub_id])

    return from_stat_bool_jax


# ruff: noqa: PLR0915
def validate_static_information(
    static_information: StaticInformation,
    n_branch: Optional[int] = None,
    n_bus: Optional[int] = None,
    n_sub_relevant: Optional[int] = None,
    max_branch_per_sub: Optional[int] = None,
    n_timesteps: Optional[int] = None,
) -> None:
    """Validate that the static information has plausible shape and values.

    Parameters
    ----------
    static_information : StaticInformation
        The static information to validate
    n_branch : int
        The number of branches expected in the network. If none, uses ptdf.shape[0]
    n_bus : int
        The number of buses expected in the network. If none, uses ptdf.shape[1]
    n_sub_relevant : int
        The number of relevant substations expected in the network. If none, uses
        len(branches_per_sub)
    max_branch_per_sub : int
        The maximum number of branches per substation expected in the network
        If None, uses max(branches_per_sub)
    n_timesteps : int
        The number of timesteps expected in the network, if None, uses nodal_injections.shape[0]

    Raises
    ------
    AssertionError
        If the static information has unexpected shape or values
    """
    di = static_information.dynamic_information
    sc = static_information.solver_config

    n_branch = di.ptdf.shape[0] if n_branch is None else n_branch
    n_branch_monitored = static_information.n_branches_monitored
    n_bus = di.ptdf.shape[1] if n_bus is None else n_bus
    n_sub_relevant = sc.branches_per_sub.val.shape[0] if n_sub_relevant is None else n_sub_relevant
    max_branch_per_sub = np.max(sc.branches_per_sub.val) if max_branch_per_sub is None else max_branch_per_sub
    max_inj_per_sub = np.max(di.generators_per_sub)
    n_timesteps = di.nodal_injections.shape[0] if n_timesteps is None else n_timesteps
    n_actions = di.n_actions

    assert di.ptdf.shape == (n_branch, n_bus)
    bus_a_columns = di.ptdf[:, sc.rel_stat_map.val]
    bus_b_columns = di.ptdf[:, sc.n_stat :]
    assert np.array_equal(bus_a_columns, bus_b_columns)
    assert di.from_node.shape == (n_branch,)
    assert jnp.all(di.from_node >= 0)
    assert jnp.all(di.from_node < n_bus)
    assert di.to_node.shape == (n_branch,)
    assert jnp.all(di.to_node >= 0)
    assert jnp.all(di.to_node < n_bus)
    assert sc.branches_per_sub.shape == (n_sub_relevant,)
    assert hash(sc.branches_per_sub) is not None
    assert di.generators_per_sub.shape == (n_sub_relevant,)
    assert di.branch_limits.max_mw_flow.shape == (n_branch_monitored,)
    assert jnp.all(di.branch_limits.max_mw_flow > 0)
    assert di.branch_limits.max_mw_flow_n_1 is None or di.branch_limits.max_mw_flow_n_1.shape == (n_branch_monitored,)
    assert di.branch_limits.overload_weight is None or di.branch_limits.overload_weight.shape == (n_branch_monitored,)
    assert di.branch_limits.max_mw_flow_limited is None or di.branch_limits.max_mw_flow_limited.shape == (
        n_branch_monitored,
    )
    assert di.branch_limits.max_mw_flow_n_1_limited is None or di.branch_limits.max_mw_flow_n_1_limited.shape == (
        n_branch_monitored,
    )
    assert di.branch_limits.n0_n1_max_diff is None or di.branch_limits.n0_n1_max_diff.shape == (n_branch_monitored,)
    assert di.branches_monitored.shape[0] <= n_branch
    assert di.branches_monitored.shape == (n_branch_monitored,)
    assert di.branches_monitored.dtype in [
        jnp.int32,
        jnp.int64,
    ]
    assert di.disconnectable_branches.shape[0] <= n_branch
    assert jnp.all(di.disconnectable_branches >= 0)
    assert jnp.all(di.disconnectable_branches < n_branch)
    assert di.branches_to_fail.shape[0] <= n_branch
    assert jnp.all(di.branches_to_fail >= 0)
    assert jnp.all(di.branches_to_fail < n_branch)
    assert jnp.all(di.nonrel_injection_outage_node >= 0)
    assert jnp.all(di.nonrel_injection_outage_node < n_bus)
    assert di.nonrel_injection_outage_deltap.shape[0] == n_timesteps
    assert di.tot_stat.shape == (
        n_sub_relevant,
        max_branch_per_sub,
    )
    assert di.from_stat_bool.shape == (
        n_sub_relevant,
        max_branch_per_sub,
    )
    assert sc.rel_stat_map.shape == (n_sub_relevant,)
    assert jnp.all(sc.rel_stat_map.val >= 0)
    assert jnp.all(sc.rel_stat_map.val < n_bus)
    assert hash(sc.rel_stat_map) is not None

    assert di.relevant_injections.shape == (
        n_timesteps,
        n_sub_relevant,
        max_inj_per_sub,
    )
    assert jnp.all(jnp.isfinite(di.relevant_injections))

    assert di.nodal_injections.shape == (
        n_timesteps,
        n_bus,
    )
    assert jnp.all(jnp.isfinite(di.nodal_injections))

    # The padded values in tot_stat must be all false in from_stat_bool
    assert not jnp.any(
        jnp.where(
            di.tot_stat == jnp.iinfo(di.tot_stat.dtype).max,
            di.from_stat_bool,
            False,
        )
    )
    assert jnp.any(di.from_stat_bool)

    assert sc.slack >= 0 and sc.slack < n_bus
    assert isinstance(sc.slack, int)
    assert sc.n_stat + sc.rel_stat_map.shape[0] == n_bus
    assert isinstance(sc.n_stat, int)

    assert di.susceptance.shape == (n_branch,)
    assert jnp.all(di.susceptance != 0)

    assert di.branches_to_fail.shape[0] <= n_branch

    assert di.action_set is not None
    assert di.action_set.n_actions_per_sub.shape[0] == n_sub_relevant
    assert di.action_set.branch_actions.shape == (n_actions, max_branch_per_sub)
    assert len(di.action_set.branch_actions.shape) == 2
    assert di.action_set.branch_actions.dtype == bool
    assert di.action_set.n_actions_per_sub.dtype in [
        jnp.int32,
        jnp.int64,
    ]
    assert di.action_set.substation_correspondence.dtype in [
        jnp.int32,
        jnp.int64,
    ]
    assert di.action_set.substation_correspondence.shape[0] == jnp.sum(di.action_set.n_actions_per_sub)
    assert di.action_set.reassignment_distance.shape == (n_actions,)
    assert di.action_set.reassignment_distance.dtype in [
        jnp.int32,
        jnp.int64,
    ]
    assert jnp.all(di.action_set.reassignment_distance >= 0)
    assert di.action_set.inj_actions.shape == (n_actions, max_inj_per_sub)

    assert jnp.all(di.action_set.reassignment_distance >= 0)
    assert len(di.multi_outage_branches) == len(di.multi_outage_nodes)
    for branch_arr, node_arr in zip(di.multi_outage_branches, di.multi_outage_nodes, strict=True):
        assert branch_arr.shape[0] == node_arr.shape[0]
        assert len(branch_arr.shape) == 2
        assert len(node_arr.shape) == 2
        assert branch_arr.dtype in [jnp.int32, jnp.int64]
        assert node_arr.dtype in [jnp.int32, jnp.int64]
        assert jnp.all(branch_arr >= 0)
        assert jnp.all(branch_arr < n_branch)

    assert sc.batch_size_bsdf > 0
    assert sc.batch_size_injection > 0
    assert sc.buffer_size_injection is None or sc.buffer_size_injection > 0
    assert sc.limit_n_subs is None or sc.limit_n_subs > 0
    assert sc.number_max_out_in_most_affected is None or sc.number_max_out_in_most_affected > 0
    assert sc.number_most_affected is None or sc.number_most_affected > 0
    assert sc.number_most_affected_n_0 is None or sc.number_most_affected_n_0 > 0
    assert di.unsplit_flow is not None
    assert di.unsplit_flow.shape == (
        n_timesteps,
        n_branch,
    )

    if di.n2_baseline_analysis is not None:
        baseline = di.n2_baseline_analysis
        n_l1_outages = baseline.l1_branches.shape[0]
        assert baseline.l1_branches.shape == (n_l1_outages,)
        assert baseline.tot_stat_blacklisted.shape[0] == n_sub_relevant
        assert baseline.tot_stat_blacklisted.shape[1] <= max_branch_per_sub
        assert baseline.n_2_overloads.shape == (n_l1_outages,)
        assert baseline.n_2_success_count.shape == (n_l1_outages,)
        assert baseline.more_splits_penalty.shape == ()
        assert baseline.max_mw_flow.shape == (n_branch_monitored,)
        if baseline.overload_weight is not None:
            assert baseline.overload_weight.shape == (n_branch_monitored,)

    assert jnp.all(di.controllable_pst_indices >= 0)
    assert jnp.all(di.controllable_pst_indices < n_bus)
    assert di.controllable_pst_indices.shape == di.shift_degree_min.shape
    assert di.controllable_pst_indices.shape == di.shift_degree_max.shape
    assert jnp.isfinite(di.shift_degree_min).all()
    assert jnp.isfinite(di.shift_degree_max).all()
    # assert jnp.all(di.shift_degree_min < di.shift_degree_max) # not used for now, needs a preprocessing step


def save_static_information(filename: str | Path, static_information: StaticInformation) -> None:
    """Save the static information to a hdf5 file.

    Parameters
    ----------
    filename : str
        The filename to save to
    static_information : StaticInformation
        The static information to save
    """
    with open(filename, "wb") as file:
        _save_static_information(file, static_information)


# ruff: noqa: PLR0915, PLR0912, C901
def _save_static_information(binaryio: BinaryIO, static_information: StaticInformation) -> None:
    """Save the static information to a hdf5 file, given an open file-like object.

    Parameters
    ----------
    binaryio : BinaryIO
        The open file-like object to save to
    static_information : StaticInformation
        The static information to save
    """
    dynamic_information = static_information.dynamic_information
    solver_config = static_information.solver_config

    with h5py.File(binaryio, mode="w") as file:
        file.create_dataset("ptdf", data=dynamic_information.ptdf)
        file.create_dataset("from_node", data=dynamic_information.from_node)
        file.create_dataset("to_node", data=dynamic_information.to_node)
        file.create_dataset(
            "branches_per_sub",
            data=solver_config.branches_per_sub.val,
        )
        file.create_dataset(
            "generators_per_sub",
            data=dynamic_information.generators_per_sub,
        )
        file.create_dataset(
            "max_mw_flow",
            data=dynamic_information.branch_limits.max_mw_flow,
        )
        if dynamic_information.branch_limits.max_mw_flow_n_1 is not None:
            file.create_dataset(
                "max_mw_flow_n_1",
                data=dynamic_information.branch_limits.max_mw_flow_n_1,
            )
        if dynamic_information.branch_limits.overload_weight is not None:
            file.create_dataset(
                "overload_weight",
                data=dynamic_information.branch_limits.overload_weight,
            )
        if dynamic_information.branch_limits.max_mw_flow_limited is not None:
            file.create_dataset(
                "max_mw_flow_limited",
                data=dynamic_information.branch_limits.max_mw_flow_limited,
            )
        if dynamic_information.branch_limits.max_mw_flow_n_1_limited is not None:
            file.create_dataset(
                "max_mw_flow_n_1_limited",
                data=dynamic_information.branch_limits.max_mw_flow_n_1_limited,
            )
        if dynamic_information.branch_limits.n0_n1_max_diff is not None:
            file.create_dataset(
                "n0_n1_max_diff",
                data=dynamic_information.branch_limits.n0_n1_max_diff,
            )
        if dynamic_information.branch_limits.coupler_limits is not None:
            file.create_dataset(
                "coupler_limits",
                data=dynamic_information.branch_limits.coupler_limits,
            )
        file.create_dataset(
            "branches_monitored",
            data=dynamic_information.branches_monitored,
        )
        file.create_dataset("tot_stat", data=dynamic_information.tot_stat)
        file.create_dataset("from_stat_bool", data=dynamic_information.from_stat_bool)
        file.create_dataset("slack", data=solver_config.slack)
        file.create_dataset("n_stat", data=solver_config.n_stat)
        file.create_dataset("rel_stat_map", data=solver_config.rel_stat_map.val)
        file.attrs["enable_bb_outages"] = solver_config.enable_bb_outages
        file.attrs["bb_outage_as_nminus1"] = solver_config.bb_outage_as_nminus1
        file.attrs["clip_bb_outage_penalty"] = solver_config.clip_bb_outage_penalty
        file.create_dataset("susceptance", data=dynamic_information.susceptance)
        file.create_dataset("relevant_injections", data=dynamic_information.relevant_injections)
        file.create_dataset(
            "nodal_injections",
            data=dynamic_information.nodal_injections,
        )
        file.create_dataset(
            "branches_to_fail",
            data=dynamic_information.branches_to_fail,
        )
        file.create_dataset(
            "disconnectable_branches",
            data=dynamic_information.disconnectable_branches,
        )
        file.create_dataset(
            "nonrel_injection_outage_deltap",
            data=dynamic_information.nonrel_injection_outage_deltap,
        )
        file.create_dataset(
            "nonrel_injection_outage_node",
            data=dynamic_information.nonrel_injection_outage_node,
        )
        file.create_dataset(
            "relevant_injection_outage_sub",
            data=dynamic_information.relevant_injection_outage_sub,
        )
        file.create_dataset(
            "relevant_injection_outage_idx",
            data=dynamic_information.relevant_injection_outage_idx,
        )

        file.create_dataset("unsplit_flow", data=dynamic_information.unsplit_flow)
        file.attrs["number_most_affected"] = solver_config.number_most_affected
        file.attrs["number_most_affected_n_0"] = solver_config.number_most_affected_n_0
        file.attrs["number_max_out_in_most_affected"] = solver_config.number_max_out_in_most_affected
        file.attrs["batch_size_bsdf"] = solver_config.batch_size_bsdf
        file.attrs["batch_size_injection"] = solver_config.batch_size_injection
        file.attrs["buffer_size_injection"] = solver_config.buffer_size_injection
        if solver_config.limit_n_subs is not None:
            file.attrs["limit_n_subs"] = solver_config.limit_n_subs
        file.attrs["aggregation_metric"] = solver_config.aggregation_metric
        file.attrs["distributed"] = solver_config.distributed
        file.attrs["contingency_ids"] = solver_config.contingency_ids

        file.create_dataset(
            "action_set_branch_actions",
            data=dynamic_information.action_set.branch_actions,
        )
        file.create_dataset(
            "action_set_inj_actions",
            data=dynamic_information.action_set.inj_actions,
        )
        file.create_dataset(
            "action_set_n_actions_per_sub",
            data=dynamic_information.action_set.n_actions_per_sub,
        )
        file.create_dataset(
            "action_set_substation_correspondence",
            data=dynamic_information.action_set.substation_correspondence,
        )
        file.create_dataset(
            "action_set_unsplit_action_mask",
            data=dynamic_information.action_set.unsplit_action_mask,
        )
        file.create_dataset(
            "action_set_reassignment_distance",
            data=dynamic_information.action_set.reassignment_distance,
        )

        if dynamic_information.n2_baseline_analysis is not None:
            baseline = dynamic_information.n2_baseline_analysis
            file.create_dataset(
                "n_2_l1_branches",
                data=baseline.l1_branches,
            )
            file.create_dataset(
                "n_2_tot_stat_blacklisted",
                data=baseline.tot_stat_blacklisted,
            )
            file.create_dataset(
                "n_2_overloads",
                data=baseline.n_2_overloads,
            )
            file.create_dataset(
                "n_2_success_count",
                data=baseline.n_2_success_count,
            )
            file.attrs["n_2_more_splits_penalty"] = baseline.more_splits_penalty.item()
            file.create_dataset(
                "n_2_max_mw_flow",
                data=baseline.max_mw_flow,
            )
            if baseline.overload_weight is not None:
                file.create_dataset(
                    "n_2_overload_weight",
                    data=baseline.overload_weight,
                )
        file.create_dataset(
            "controllable_pst_indices",
            data=dynamic_information.controllable_pst_indices,
        )
        file.create_dataset(
            "shift_degree_min",
            data=dynamic_information.shift_degree_min,
        )
        file.create_dataset(
            "shift_degree_max",
            data=dynamic_information.shift_degree_max,
        )

        for idx, (branches, nodes) in enumerate(
            zip(
                dynamic_information.multi_outage_branches,
                dynamic_information.multi_outage_nodes,
                strict=True,
            )
        ):
            file.create_dataset(
                f"multi_outage_branches_{idx}",
                data=branches,
            )
            file.create_dataset(
                f"multi_outage_nodes_{idx}",
                data=nodes,
            )

        file.attrs["version"] = version("toop-engine-dc-solver")
        file.attrs["format"] = "jax"

        if dynamic_information.non_rel_bb_outage_data is not None:
            file.create_dataset(
                "non_rel_bb_outage_data_branch_outages",
                data=dynamic_information.non_rel_bb_outage_data.branch_outages,
            )
            file.create_dataset(
                "non_rel_bb_outage_data_nodal_indices",
                data=dynamic_information.non_rel_bb_outage_data.nodal_indices,
            )
            file.create_dataset(
                "non_rel_bb_outage_data_deltap",
                data=dynamic_information.non_rel_bb_outage_data.deltap,
            )
        if dynamic_information.action_set.rel_bb_outage_data is not None:
            file.create_dataset(
                "action_set_rel_bb_outage_data_branch_outage_set",
                data=dynamic_information.action_set.rel_bb_outage_data.branch_outage_set,
            )
            file.create_dataset(
                "action_set_rel_bb_outage_data_nodal_indices",
                data=dynamic_information.action_set.rel_bb_outage_data.nodal_indices,
            )
            file.create_dataset(
                "action_set_rel_bb_outage_data_deltap_set",
                data=dynamic_information.action_set.rel_bb_outage_data.deltap_set,
            )
            file.create_dataset(
                "action_set_rel_bb_outage_data_critical_node_mask",
                data=dynamic_information.action_set.rel_bb_outage_data.articulation_node_mask,
            )
        if dynamic_information.bb_outage_baseline_analysis is not None:
            file.create_dataset(
                "bb_outage_baseline_analysis_overload",
                data=dynamic_information.bb_outage_baseline_analysis.overload,
            )
            file.create_dataset(
                "bb_outage_baseline_analysis_success_count",
                data=dynamic_information.bb_outage_baseline_analysis.success_count,
            )
            if dynamic_information.bb_outage_baseline_analysis.overload_weight is not None:
                file.create_dataset(
                    "bb_outage_baseline_analysis_overload_weight",
                    data=dynamic_information.bb_outage_baseline_analysis.overload_weight,
                )

            file.create_dataset(
                "bb_outage_baseline_analysis_max_mw_flow",
                data=dynamic_information.bb_outage_baseline_analysis.max_mw_flow,
            )
            file.create_dataset(
                "bb_outage_baseline_more_splits_penalty",
                data=dynamic_information.bb_outage_baseline_analysis.more_splits_penalty,
            )


def load_static_information(filename: str | Path) -> StaticInformation:
    """Load the static information from a hdf5 file in jax format

    Parameters
    ----------
    filename : str
        The filename to load from

    Returns
    -------
    StaticInformation
        The loaded static information
    """
    with open(filename, "rb") as file:
        return _load_static_information(file)


def _load_static_information(binaryio: BinaryIO) -> StaticInformation:
    """Load the static information from a hdf5 file in jax format, given an open file

    Parameters
    ----------
    binaryio : BinaryIO
        The open file to load from

    Returns
    -------
    StaticInformation
        The loaded static information
    """

    def _load_multi_outage_branch(
        file: h5py.File,
    ) -> Iterator[Int[np.ndarray, " n_branches"]]:
        idx = 0
        while f"multi_outage_branches_{idx}" in file:
            yield jnp.array(file[f"multi_outage_branches_{idx}"][:])
            idx += 1

    def _load_multi_outage_node(
        file: h5py.File,
    ) -> Iterator[Int[np.ndarray, " n_nodes"]]:
        idx = 0
        while f"multi_outage_nodes_{idx}" in file:
            yield jnp.array(file[f"multi_outage_nodes_{idx}"][:])
            idx += 1

    with h5py.File(binaryio, mode="r") as file:
        n2_baseline_analysis_present = (
            "n_2_l1_branches" in file
            and "n_2_tot_stat_blacklisted" in file
            and "n_2_overloads" in file
            and "n_2_success_count" in file
            and "n_2_more_splits_penalty" in file.attrs
            and "n_2_max_mw_flow" in file
        )

        rel_bb_outage_data_present = (
            "action_set_rel_bb_outage_data_branch_outage_set" in file
            and "action_set_rel_bb_outage_data_nodal_indices" in file
            and "action_set_rel_bb_outage_data_deltap_set" in file
            and "action_set_rel_bb_outage_data_critical_node_mask" in file
        )

        non_rel_bb_outage_data_present = (
            "non_rel_bb_outage_data_branch_outages" in file
            and "non_rel_bb_outage_data_nodal_indices" in file
            and "non_rel_bb_outage_data_deltap" in file
        )

        bb_outage_baseline_analysis_present = (
            "bb_outage_baseline_analysis_overload" in file and "bb_outage_baseline_analysis_success_count" in file
        )

        return StaticInformation(
            dynamic_information=DynamicInformation(
                ptdf=jnp.array(file["ptdf"][:]),
                from_node=jnp.array(file["from_node"][:]),
                to_node=jnp.array(file["to_node"][:]),
                generators_per_sub=jnp.array(file["generators_per_sub"][:]),
                branch_limits=BranchLimits(
                    max_mw_flow=jnp.array(file["max_mw_flow"][:]),
                    max_mw_flow_n_1=(jnp.array(file["max_mw_flow_n_1"][:]) if "max_mw_flow_n_1" in file else None),
                    overload_weight=(jnp.array(file["overload_weight"][:]) if "overload_weight" in file else None),
                    max_mw_flow_limited=(
                        jnp.array(file["max_mw_flow_limited"][:]) if "max_mw_flow_limited" in file else None
                    ),
                    max_mw_flow_n_1_limited=(
                        jnp.array(file["max_mw_flow_n_1_limited"][:]) if "max_mw_flow_n_1_limited" in file else None
                    ),
                    n0_n1_max_diff=(jnp.array(file["n0_n1_max_diff"][:]) if "n0_n1_max_diff" in file else None),
                    coupler_limits=(jnp.array(file["coupler_limits"][:]) if "coupler_limits" in file else None),
                ),
                tot_stat=jnp.array(file["tot_stat"][:]),
                from_stat_bool=jnp.array(file["from_stat_bool"][:]),
                susceptance=jnp.array(file["susceptance"][:]),
                relevant_injections=jnp.array(file["relevant_injections"][:]),
                nodal_injections=jnp.array(file["nodal_injections"][:]),
                branches_to_fail=jnp.array(file["branches_to_fail"][:]),
                disconnectable_branches=jnp.array(file["disconnectable_branches"][:]),
                action_set=(
                    ActionSet(
                        branch_actions=jnp.array(file["action_set_branch_actions"][:]),
                        inj_actions=jnp.array(file["action_set_inj_actions"][:]),
                        n_actions_per_sub=jnp.array(file["action_set_n_actions_per_sub"][:]),
                        substation_correspondence=jnp.array(file["action_set_substation_correspondence"][:]),
                        unsplit_action_mask=jnp.array(file["action_set_unsplit_action_mask"][:]),
                        reassignment_distance=jnp.array(file["action_set_reassignment_distance"][:]),
                        rel_bb_outage_data=RelBBOutageData(
                            branch_outage_set=jnp.array(file["action_set_rel_bb_outage_data_branch_outage_set"][:]),
                            nodal_indices=jnp.array(file["action_set_rel_bb_outage_data_nodal_indices"][:]),
                            deltap_set=jnp.array(file["action_set_rel_bb_outage_data_deltap_set"][:]),
                            articulation_node_mask=jnp.array(file["action_set_rel_bb_outage_data_critical_node_mask"][:]),
                        )
                        if rel_bb_outage_data_present
                        else None,
                    )
                ),
                multi_outage_branches=list(_load_multi_outage_branch(file)),
                multi_outage_nodes=list(_load_multi_outage_node(file)),
                nonrel_injection_outage_deltap=jnp.array(file["nonrel_injection_outage_deltap"][:]),
                nonrel_injection_outage_node=jnp.array(file["nonrel_injection_outage_node"][:]),
                relevant_injection_outage_sub=jnp.array(file["relevant_injection_outage_sub"][:]),
                relevant_injection_outage_idx=jnp.array(file["relevant_injection_outage_idx"][:]),
                unsplit_flow=jnp.array(file["unsplit_flow"][:]),
                branches_monitored=jnp.array(file["branches_monitored"][:]),
                n2_baseline_analysis=(
                    N2BaselineAnalysis(
                        l1_branches=jnp.array(file["n_2_l1_branches"][:]),
                        tot_stat_blacklisted=jnp.array(file["n_2_tot_stat_blacklisted"][:]),
                        n_2_overloads=jnp.array(file["n_2_overloads"][:]),
                        n_2_success_count=jnp.array(file["n_2_success_count"][:]),
                        more_splits_penalty=jnp.array(float(file.attrs["n_2_more_splits_penalty"])),
                        max_mw_flow=jnp.array(file["n_2_max_mw_flow"][:]),
                        overload_weight=(
                            jnp.array(file["n_2_overload_weight"][:]) if "n_2_overload_weight" in file else None
                        ),
                    )
                    if n2_baseline_analysis_present
                    else None
                ),
                controllable_pst_indices=jnp.array(file["controllable_pst_indices"][:]),
                shift_degree_min=jnp.array(file["shift_degree_min"][:]),
                shift_degree_max=jnp.array(file["shift_degree_max"][:]),
                non_rel_bb_outage_data=NonRelBBOutageData(
                    branch_outages=jnp.array(file["non_rel_bb_outage_data_branch_outages"][:]),
                    nodal_indices=jnp.array(file["non_rel_bb_outage_data_nodal_indices"][:]),
                    deltap=jnp.array(file["non_rel_bb_outage_data_deltap"][:]),
                )
                if non_rel_bb_outage_data_present
                else None,
                bb_outage_baseline_analysis=(
                    BBOutageBaselineAnalysis(
                        overload=jnp.array(file["bb_outage_baseline_analysis_overload"]),
                        success_count=jnp.array(file["bb_outage_baseline_analysis_success_count"]),
                        more_splits_penalty=jnp.array(file["bb_outage_baseline_more_splits_penalty"]),
                        overload_weight=float(file.attrs["bb_outage_baseline_analysis_overload_weight"][:])
                        if "bb_outage_baseline_analysis_overload_weight" in file.attrs
                        else None,
                        max_mw_flow=jnp.array(file["bb_outage_baseline_analysis_max_mw_flow"][:]),
                    )
                    if bb_outage_baseline_analysis_present
                    else None
                ),
            ),
            solver_config=SolverConfig(
                branches_per_sub=HashableArrayWrapper(file["branches_per_sub"][:]),
                slack=int(file["slack"][()]),
                n_stat=int(file["n_stat"][()]),
                rel_stat_map=HashableArrayWrapper(file["rel_stat_map"][:]),
                number_most_affected=int(file.attrs["number_most_affected"]),
                number_most_affected_n_0=int(file.attrs["number_most_affected_n_0"]),
                number_max_out_in_most_affected=int(file.attrs["number_max_out_in_most_affected"]),
                batch_size_bsdf=int(file.attrs["batch_size_bsdf"]),
                batch_size_injection=int(file.attrs["batch_size_injection"]),
                buffer_size_injection=int(file.attrs["buffer_size_injection"]),
                limit_n_subs=file.attrs.get("limit_n_subs", None),
                aggregation_metric=file.attrs["aggregation_metric"],
                distributed=bool(file.attrs["distributed"]),
                enable_bb_outages=bool(file.attrs.get("enable_bb_outages", False)),
                bb_outage_as_nminus1=bool(file.attrs.get("bb_outage_as_nminus1", True)),
                clip_bb_outage_penalty=bool(file.attrs.get("clip_bb_outage_penalty", False)),
                contingency_ids=list(file.attrs.get("contingency_ids", [])),
            ),
        )


def serialize_static_information(static_information: StaticInformation) -> bytes:
    """Serialize the static information to a bytestring.

    Parameters
    ----------
    static_information : StaticInformation
        The static information to serialize

    Returns
    -------
    bytes
        The serialized static information
    """
    bytes_io = io.BytesIO()
    _save_static_information(bytes_io, static_information)
    return bytes_io.getvalue()


def deserialize_static_information(serialized: bytes) -> StaticInformation:
    """Deserialize the static information from a bytestring.

    Parameters
    ----------
    serialized : bytes
        The serialized static information

    Returns
    -------
    StaticInformation
        The deserialized static information
    """
    bytes_io = io.BytesIO(serialized)
    return _load_static_information(bytes_io)
