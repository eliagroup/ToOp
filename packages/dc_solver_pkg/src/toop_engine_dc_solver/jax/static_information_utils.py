# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helpers for runtime static-information updates and busbar-outage setup."""

import structlog
from beartype.typing import Iterable
from jax import numpy as jnp
from jax_dataclasses import replace
from toop_engine_dc_solver.jax.aggregate_results import get_overload_energy_n_1_matrix
from toop_engine_dc_solver.jax.busbar_outage import perform_rel_bb_outage_for_unsplit_grid
from toop_engine_dc_solver.jax.types import BBOutageBaselineAnalysis, DynamicInformation, SolverConfig, StaticInformation

logger = structlog.get_logger(__name__)


def get_bb_outage_baseline_analysis(di: DynamicInformation, more_splits_penalty: float) -> BBOutageBaselineAnalysis:
    """Get the baseline loadflows after busbar outages of unsplit grid.

    Parameters
    ----------
    di : DynamicInformation
        The dynamic information dataclass
    more_splits_penalty : float
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


def update_static_information(
    static_informations: tuple[StaticInformation, ...],
    batch_size: int,
    enable_nodal_inj_optim: bool,
    enable_parallel_pst_group_optim: bool,
    enable_bb_outage: bool,
    bb_outage_as_nminus1: bool,
    clip_bb_outage_penalty: bool,
    bb_outage_more_islands_penalty: float,
) -> tuple[StaticInformation, ...]:
    """Perform any necessary preprocessing on the static information.

    This mainly applies updated optimization parameters such as batch size, busbar settings, etc.

    Parameters
    ----------
    static_informations : tuple[StaticInformation, ...]
        The list of static informations to preprocess.
    batch_size : int
        The batch size to use, will replace the batch size in the solver config.
    enable_nodal_inj_optim : bool
        Whether to enable the nodal injection optimization.
    enable_parallel_pst_group_optim : bool
        Whether to enable parallel PST group optimization.
    enable_bb_outage : bool
        Whether the optimizer should include busbar outage effects.
    bb_outage_as_nminus1 : bool
        Whether busbar outages are handled as additional N-1 cases.
    clip_bb_outage_penalty : bool
        Whether busbar outage penalties are clipped at 0.
    bb_outage_more_islands_penalty : float
        The islanding penalty stored in the busbar outage baseline.

    Returns
    -------
    tuple[StaticInformation, ...]
        The updated static informations.
    """
    updated_pairs = []
    for static_information in static_informations:
        solver_config, dynamic_information = update_single_pair_branch_limit_information(
            solver_config=static_information.solver_config,
            dynamic_information=static_information.dynamic_information,
            batch_size=batch_size,
            enable_nodal_inj_optim=enable_nodal_inj_optim,
            enable_parallel_pst_group_optim=enable_parallel_pst_group_optim,
        )
        solver_config, dynamic_information = update_single_pair_bb_outage_information(
            solver_config=solver_config,
            dynamic_information=dynamic_information,
            enable_bb_outage=enable_bb_outage,
            bb_outage_as_nminus1=bb_outage_as_nminus1,
            clip_bb_outage_penalty=clip_bb_outage_penalty,
            bb_outage_more_islands_penalty=bb_outage_more_islands_penalty,
        )
        updated_pairs.append((solver_config, dynamic_information))

    static_informations = [
        replace(
            static_information,
            solver_config=solver_config,
            dynamic_information=dynamic_information,
        )
        for static_information, (solver_config, dynamic_information) in zip(static_informations, updated_pairs, strict=True)
    ]

    return tuple(static_informations)


def update_single_pair_branch_limit_information(
    solver_config: SolverConfig,
    dynamic_information: DynamicInformation,
    batch_size: int,
    enable_nodal_inj_optim: bool,
    enable_parallel_pst_group_optim: bool,
) -> tuple[SolverConfig, DynamicInformation]:
    """Normalize branch-limit and nodal-injection data for one timestep."""
    updated_solver_config = replace(
        solver_config,
        batch_size_bsdf=batch_size,
        batch_size_injection=batch_size,
        enable_parallel_pst_group_optim=enable_parallel_pst_group_optim if enable_nodal_inj_optim else False,
    )
    nodal_injection_info = None
    if dynamic_information.nodal_injection_information is not None:
        if not enable_nodal_inj_optim:
            nodal_injection_info = replace(dynamic_information.nodal_injection_information, parallel_pst_group_mask=None)
        else:
            nodal_injection_info = replace(
                dynamic_information.nodal_injection_information,
                parallel_pst_group_mask=dynamic_information.nodal_injection_information.parallel_pst_group_mask
                if enable_parallel_pst_group_optim
                else None,
            )

    updated_dynamic_information = replace(
        dynamic_information,
        branch_limits=replace(
            dynamic_information.branch_limits,
            max_mw_flow_limited=(
                dynamic_information.branch_limits.max_mw_flow
                if dynamic_information.branch_limits.max_mw_flow_limited is None
                else dynamic_information.branch_limits.max_mw_flow_limited
            ),
            n0_n1_max_diff=(
                jnp.zeros_like(dynamic_information.branch_limits.max_mw_flow)
                if dynamic_information.branch_limits.n0_n1_max_diff is None
                else dynamic_information.branch_limits.n0_n1_max_diff
            ),
        ),
        nodal_injection_information=nodal_injection_info,
    )
    return updated_solver_config, updated_dynamic_information


def update_single_pair_bb_outage_information(
    solver_config: SolverConfig,
    dynamic_information: DynamicInformation,
    enable_bb_outage: bool,
    bb_outage_as_nminus1: bool,
    clip_bb_outage_penalty: bool,
    bb_outage_more_islands_penalty: float,
) -> tuple[SolverConfig, DynamicInformation]:
    """Apply runtime busbar-outage configuration for one timestep."""
    has_rel_bb_outage_data = dynamic_information.action_set.rel_bb_outage_data is not None
    has_monitored_branches = dynamic_information.branches_monitored.size > 0
    has_stored_bb_outage_baseline = dynamic_information.bb_outage_baseline_analysis is not None
    has_non_rel_bb_outage_data = dynamic_information.non_rel_bb_outage_data is not None

    has_bb_outage_data = has_stored_bb_outage_baseline or has_non_rel_bb_outage_data or has_rel_bb_outage_data
    should_enable_bb_outage = enable_bb_outage and has_bb_outage_data
    needs_penalty_baseline = (
        should_enable_bb_outage and not bb_outage_as_nminus1 and has_rel_bb_outage_data and has_monitored_branches
    )
    should_keep_or_create_baseline = (
        should_enable_bb_outage and not bb_outage_as_nminus1 and (has_stored_bb_outage_baseline or needs_penalty_baseline)
    )

    base_nminus1_cases = (
        dynamic_information.n_outages + dynamic_information.n_multi_outages + dynamic_information.n_inj_failures
    )
    expected_nminus1_cases = (
        base_nminus1_cases + dynamic_information.n_bb_outages
        if should_enable_bb_outage and bb_outage_as_nminus1
        else base_nminus1_cases
    )
    stored_contingency_ids = list(solver_config.contingency_ids)
    contingency_ids = stored_contingency_ids[:base_nminus1_cases]
    if len(contingency_ids) < base_nminus1_cases:
        contingency_ids.extend(f"nminus1_case_{i}" for i in range(len(contingency_ids), base_nminus1_cases))

    if should_enable_bb_outage and bb_outage_as_nminus1:
        stored_bb_outage_ids = stored_contingency_ids[
            base_nminus1_cases : base_nminus1_cases + dynamic_information.n_bb_outages
        ]
        if len(stored_bb_outage_ids) >= dynamic_information.n_bb_outages:
            contingency_ids.extend(stored_bb_outage_ids)
        else:
            contingency_ids.extend(f"bb_outage_case_{i}" for i in range(dynamic_information.n_bb_outages))

    contingency_ids = contingency_ids[:expected_nminus1_cases]

    updated_solver_config = replace(
        solver_config,
        enable_bb_outages=should_enable_bb_outage,
        bb_outage_as_nminus1=bb_outage_as_nminus1,
        clip_bb_outage_penalty=clip_bb_outage_penalty,
        contingency_ids=contingency_ids,
    )
    updated_action_set = (
        dynamic_information.action_set
        if should_enable_bb_outage
        else replace(dynamic_information.action_set, rel_bb_outage_data=None)
    )
    updated_dynamic_information = replace(
        dynamic_information,
        action_set=updated_action_set,
        bb_outage_baseline_analysis=(
            replace(
                dynamic_information.bb_outage_baseline_analysis
                if dynamic_information.bb_outage_baseline_analysis is not None
                else get_bb_outage_baseline_analysis(dynamic_information, bb_outage_more_islands_penalty),
                more_splits_penalty=jnp.array(bb_outage_more_islands_penalty),
            )
            if should_keep_or_create_baseline
            else None
        ),
        non_rel_bb_outage_data=(dynamic_information.non_rel_bb_outage_data if should_enable_bb_outage else None),
    )
    return updated_solver_config, updated_dynamic_information


def verify_static_information(
    static_informations: Iterable[StaticInformation],
    max_num_disconnections: int,
    enable_nodal_inj_optim: bool,
    enable_parallel_pst_group_optim: bool = False,
) -> None:
    """Verify that static information objects are compatible with the optimization run."""
    first_static_information = next(iter(static_informations))

    assert all(
        [
            jnp.array_equal(
                static_information.solver_config.branches_per_sub.val,
                first_static_information.solver_config.branches_per_sub.val,
            )
            for static_information in static_informations
        ]
    )
    first_set = first_static_information.dynamic_information.action_set
    if first_set is not None:
        assert all(
            static_information.dynamic_information.action_set is not None for static_information in static_informations
        )
        assert all(
            [(static_information.dynamic_information.action_set == first_set) for static_information in static_informations]
        ), "All static informations must have the same branch actions"
        assert all(
            [
                jnp.array_equal(
                    static_information.dynamic_information.action_set.n_actions_per_sub,
                    first_set.n_actions_per_sub,
                )
                for static_information in static_informations
            ]
        ), "All static informations must have the same number of branch actions"

    assert first_static_information.dynamic_information.disconnectable_branches.shape[0] >= max_num_disconnections, (
        "Not enough disconnectable branches for the maximum number of disconnections, "
        + f"got {first_static_information.dynamic_information.disconnectable_branches.shape[0]} and {max_num_disconnections}"
    )
    if first_static_information.dynamic_information.disconnectable_branches.shape[0] > 0:
        assert all(
            [
                jnp.array_equal(
                    first_static_information.dynamic_information.disconnectable_branches,
                    static_information.dynamic_information.disconnectable_branches,
                )
                for static_information in static_informations
            ]
        ), "All static informations must have the same disconnectable branches"
    if enable_nodal_inj_optim:
        assert first_static_information.dynamic_information.nodal_injection_information is not None, (
            "Nodal injection opt. is enabled, but the first static information does not contain nodal injection info. "
            "For nodal injection optimization, we require at least one controllable PST in the nodal injection info. "
        )
        assert all(
            [
                static_information.dynamic_information.nodal_injection_information is not None
                and static_information.dynamic_information.nodal_injection_information.controllable_pst_indices.shape[0] > 0
                for static_information in static_informations
            ]
        ), (
            "Nodal injection optimization is enabled, but some static/nodal injection info does not contain controll. PSTs. "
            "This requires at least one controllable PST in the nodal injection information. "
            "Disable nodal injection optimization or provide correct static information. "
        )
    if enable_parallel_pst_group_optim:
        assert first_static_information.dynamic_information.nodal_injection_information is not None, (
            "Parallel PST group optimization requires nodal injection information with controllable PSTs."
        )
        assert (
            first_static_information.dynamic_information.nodal_injection_information.parallel_pst_group_mask is not None
        ), (
            "Parallel PST group optimization is enabled, but the first static information lacks a parallel_pst_group_mask. "
            "This requires a parallel_pst_group_mask in the nodal injection information. "
            "Disable parallel PST group optimization or provide correct static information. "
        )
