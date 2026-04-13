# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Module for comparing the postprocessing loadflow results in DC to the solver results."""

from functools import partial

import jax.numpy as jnp
import numpy as np
from beartype.typing import Optional
from jaxtyping import ArrayLike, Bool, Float
from pydantic import BaseModel
from pypowsybl.network import Network
from toop_engine_dc_solver.jax import (
    run_solver_symmetric,
)
from toop_engine_dc_solver.jax.compute_batch import compute_bsdf_lodf_static_flows
from toop_engine_dc_solver.jax.topology_computations import convert_action_set_index_to_topo
from toop_engine_dc_solver.jax.types import ActionIndexComputations, DynamicInformation, SolverConfig, StaticInformation
from toop_engine_dc_solver.postprocess.postprocess_powsybl import get_islanding_contingency_ids
from toop_engine_dc_solver.preprocess.helpers.find_bridges import find_bridges
from toop_engine_interfaces.loadflow_result_helpers_polars import extract_solver_matrices_polars
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Contingency, Nminus1Definition


class LoadflowValidationParameters(BaseModel):
    """Parameters for validating loadflow results."""

    atol: float = 1e-5
    """Absolute tolerance for the comparison."""
    rtol: float = 1e-5
    """Relative tolerance for the comparison."""
    equal_nan: bool = False
    """Whether to consider NaN values as equal."""

    compare_signs: bool = False
    """Whether to compare the signs of the results."""


def get_islanding_contingencies_solver(
    static_information: StaticInformation,
    actions: list[int],
    disconnections: list[int] | None,
    contingencies: list[Contingency],
) -> set[str]:
    """Get contingency ids that lead to islanding in the active topology.

    This is relevant for the validation, as such contingencies should be expected to lead to a loadflow failure

    Parameters
    ----------
    static_information : StaticInformation
        The static information of the problem, containing the mapping from contingencies to branches.
    actions : list[int]
        The actions that are taken in the grid, as indices into the action set.
    disconnections : list[int] | None
        The disconnections as indices into the disconnectable branches set. Can be None if no disconnections are taken.
    contingencies : list[Contingency]
        The list of contingencies to check against.

    Returns
    -------
    set[str]
        The set of contingency ids that lead to islanding in the active topology.
    """
    dynamic_information = static_information.dynamic_information
    solver_branch_contingency_ids = static_information.solver_config.contingency_ids[: dynamic_information.n_outages]
    bridge_contingency_ids = {
        contingency.id
        for contingency in contingencies
        if contingency.is_single_outage() and contingency.id not in solver_branch_contingency_ids
    }

    if len(actions) == 0 and (disconnections is None or len(disconnections) == 0):
        from_node = np.asarray(dynamic_information.from_node)
        to_node = np.asarray(dynamic_information.to_node)
        valid_branch_mask = (
            (from_node >= 0)
            & (to_node >= 0)
            & (from_node < dynamic_information.n_nodes)
            & (to_node < dynamic_information.n_nodes)
        )
        bridge_mask = np.zeros(dynamic_information.n_branches, dtype=bool)
        if np.any(valid_branch_mask):
            bridge_mask[np.asarray(valid_branch_mask)] = find_bridges(
                from_node=from_node[valid_branch_mask],
                to_node=to_node[valid_branch_mask],
                number_of_branches=int(np.sum(valid_branch_mask)),
                number_of_nodes=dynamic_information.n_nodes,
            )
    else:
        topology = ActionIndexComputations(
            action=jnp.array([actions], dtype=int),
            pad_mask=jnp.array([True]),
        )
        bitvector_topology = convert_action_set_index_to_topo(topology, dynamic_information.action_set)
        disconnection_array = None
        if disconnections is not None and len(disconnections) > 0:
            disconnection_array = dynamic_information.disconnectable_branches.at[jnp.array(disconnections, dtype=int)].get(
                mode="fill"
            )[None]

        topo_res = compute_bsdf_lodf_static_flows(
            topology_batch=bitvector_topology,
            disconnection_batch=disconnection_array,
            dynamic_information=dynamic_information,
            solver_config=static_information.solver_config,
        )

        from_node = np.asarray(topo_res.from_node[0])
        to_node = np.asarray(topo_res.to_node[0])
        valid_branch_mask = (
            (from_node >= 0)
            & (to_node >= 0)
            & (from_node < dynamic_information.n_nodes)
            & (to_node < dynamic_information.n_nodes)
        )
        bridge_mask = np.zeros(dynamic_information.n_branches, dtype=bool)
        if np.any(valid_branch_mask):
            bridge_mask[np.asarray(valid_branch_mask)] = find_bridges(
                from_node=from_node[valid_branch_mask],
                to_node=to_node[valid_branch_mask],
                number_of_branches=int(np.sum(valid_branch_mask)),
                number_of_nodes=dynamic_information.n_nodes,
            )
    branches_to_fail = np.asarray(dynamic_information.branches_to_fail)

    contingencies_by_id = {contingency.id: contingency for contingency in contingencies}
    for contingency_id, branch_index in zip(solver_branch_contingency_ids, branches_to_fail, strict=True):
        if bridge_mask[int(branch_index)]:
            contingency = contingencies_by_id.get(contingency_id)
            if contingency is not None:
                bridge_contingency_ids.add(contingency.id)
    return bridge_contingency_ids


def _get_inactive_branch_contingency_ids(
    static_information: StaticInformation,
    actions: list[int],
    disconnections: list[int] | None,
    contingencies: list[Contingency],
) -> set[str]:
    """Get branch contingency ids that are already inactive in the active topology.

    This is relevant for the validation, as such contingencies should not be expected to lead to a loadflow failure,
    even if they are not marked as "islanding" contingencies.

    The function works by checking which branches are inactive in the active topology
    (either due to actions or disconnections) and then checking which contingencies correspond to those branches.

    Parameters
    ----------
    static_information : StaticInformation
        The static information of the problem, containing the mapping from contingencies to branches.
    actions : list[int]
        The actions that are taken in the grid, as indices into the action set.
    disconnections : list[int] | None
        The disconnections as indices into the disconnectable branches set. Can be None if no disconnections are taken.
    contingencies : list[Contingency]
        The list of contingencies to check against.

    Returns
    -------
    set[str]
        The set of contingency ids that correspond to branches that are already inactive in the active topology.
    """
    if len(actions) == 0 and (disconnections is None or len(disconnections) == 0):
        return set()

    dynamic_information = static_information.dynamic_information
    solver_branch_contingency_ids = static_information.solver_config.contingency_ids[: dynamic_information.n_outages]

    topology = ActionIndexComputations(
        action=jnp.array([actions], dtype=int),
        pad_mask=jnp.array([True]),
    )
    bitvector_topology = convert_action_set_index_to_topo(topology, dynamic_information.action_set)
    disconnection_array = None
    if disconnections is not None and len(disconnections) > 0:
        disconnection_array = dynamic_information.disconnectable_branches.at[jnp.array(disconnections, dtype=int)].get(
            mode="fill"
        )[None]

    topo_res = compute_bsdf_lodf_static_flows(
        topology_batch=bitvector_topology,
        disconnection_batch=disconnection_array,
        dynamic_information=dynamic_information,
        solver_config=static_information.solver_config,
    )

    from_node = np.asarray(topo_res.from_node[0])
    to_node = np.asarray(topo_res.to_node[0])
    inactive_branch_mask = (
        (from_node < 0)
        | (to_node < 0)
        | (from_node >= dynamic_information.n_nodes)
        | (to_node >= dynamic_information.n_nodes)
    )

    contingencies_by_id = {contingency.id: contingency for contingency in contingencies}
    inactive_contingency_ids: set[str] = set()
    for contingency_id, branch_index in zip(
        solver_branch_contingency_ids, np.asarray(dynamic_information.branches_to_fail), strict=True
    ):
        if inactive_branch_mask[int(branch_index)] and contingency_id in contingencies_by_id:
            inactive_contingency_ids.add(contingency_id)
    return inactive_contingency_ids


def validate_loadflow_results(
    static_information: StaticInformation,
    nminus1_definition: Nminus1Definition,
    loadflows: LoadflowResultsPolars,
    actions: list[int],
    active_topology_network: Network,
    disconnections: list[int] | None,
    timestep: int = 0,
    validation_parameters: Optional[LoadflowValidationParameters] = None,
) -> None:
    """Validate the loadflow results of a single topology against the solver results

    Parameters
    ----------
    static_information : StaticInformation
        The same static_information that was used to optimize the grid.
    nminus1_definition : Nminus1Definition
        The nminus1_definition that was used to compute the loadflows.
    loadflows : MultiTimestepLoadflowResults
        The DC loadflow results from the solver
    actions : list[int]
        The actions that were taken in the grid
    active_topology_network : Network
        The active topology as a powsybl Network object, used to determine which branches are inactive in powsybl
    disconnections : list[int] | None
        The disconnections as indices into the disconnectable branches set
    timestep : int, optional
        The timestep to validate, by default 0
    validation_parameters: Optional[LoadflowValidationParameters] = None,
        The paramters for validation. Used for np.allclose and to determine whether to compare signs.
        Defaults to LoadflowValidationParameters()

    Raises
    ------
    AssertionError
        If the loadflow results are not consistent
    """
    if validation_parameters is None:
        validation_parameters = LoadflowValidationParameters()

    dynamic_information = static_information.dynamic_information
    solver_config = static_information.solver_config

    n_0, n_1, success = extract_solver_matrices_polars(
        loadflow_results=loadflows,
        nminus1_definition=nminus1_definition,
        timestep=timestep,
    )

    n_0_solver, n_1_solver, success_solver = get_solver_results(
        actions, disconnections, timestep, dynamic_information, solver_config
    )
    if not validation_parameters.compare_signs:
        n_0 = np.abs(n_0)
        n_1 = np.abs(n_1)
        n_0_solver = np.abs(n_0_solver)
        n_1_solver = np.abs(n_1_solver)

    messages = []

    allclose = partial(
        np.allclose,
        atol=validation_parameters.atol,
        rtol=validation_parameters.rtol,
        equal_nan=validation_parameters.equal_nan,
    )

    if not allclose(n_0, n_0_solver):
        error = np.abs(n_0 - n_0_solver)
        messages.append(f"N-0 does not match, mean error: {error.mean()}, max error: {error.max()}")

    case_contingencies = [contingency for contingency in nminus1_definition.contingencies if not contingency.is_basecase()]

    assert_shapes(n_0, n_1, success, n_0_solver, n_1_solver, success_solver, case_contingencies)

    islanding_contingency_ids_solver = get_islanding_contingencies_solver(
        static_information=static_information,
        actions=actions,
        disconnections=disconnections,
        contingencies=case_contingencies,
    )
    islanding_contingency_ids_powsybl = get_islanding_contingency_ids(
        net=active_topology_network,
        nminus1_definition=nminus1_definition,
    )
    # Check happy case
    both_converged = success & success_solver
    if not allclose(n_1[both_converged, :], n_1_solver[both_converged, :]):
        error = np.abs(n_1[both_converged, :] - n_1_solver[both_converged:])
        high_diff_cases = case_contingencies[error > validation_parameters.atol]
        messages.append(
            f"N-1 for cases: {[contingency.id for contingency in high_diff_cases]} does not match, "
            f"mean error: {error.mean()}, max error: {error.max()}"
        )

    contingency_leads_to_solver_islanding = np.array(
        [contingency.id in islanding_contingency_ids_solver for contingency in case_contingencies]
    )
    contingency_leads_to_powsybl_islanding = np.array(
        [contingency.id in islanding_contingency_ids_powsybl for contingency in case_contingencies]
    )

    # Check neiter converged
    neither_converged = ~success & ~success_solver
    neither_islanded = ~contingency_leads_to_solver_islanding & ~contingency_leads_to_powsybl_islanding
    if any(neither_converged & neither_islanded):
        messages.append(
            f"N-1 for cases: "
            f"{[contingency.id for contingency in case_contingencies[neither_converged & neither_islanded]]} "
            f"failed, but there is no islanding."
        )

    # Check only powsybl converged, since they have smarter functionalities for islanding
    only_powsybl_converged = success & ~success_solver
    if any(only_powsybl_converged & neither_islanded):
        messages.append(
            f"N-1 for cases: "
            f"{[contingency.id for contingency in case_contingencies[only_powsybl_converged & neither_islanded]]} "
            f"failed only in solver, but there is no islanding."
        )
    only_solver_converged = ~success & success_solver
    if any(only_solver_converged):
        messages.append(
            f"N-1 for cases: "
            f"{[contingency.id for contingency in case_contingencies[only_solver_converged]]} "
            f"succeeded only in solver. This should not happen. Please have a look."
        )

    assert len(messages) == 0, "\n".join(messages)


def assert_shapes(
    n_0: Float[ArrayLike, " n_branches"],
    n_1: Float[ArrayLike, " n_cases n_branches"],
    success: Bool[ArrayLike, " n_cases"],
    n_0_solver: Float[ArrayLike, " n_branches"],
    n_1_solver: Float[ArrayLike, " n_cases n_branches"],
    success_solver: Bool[ArrayLike, " n_cases"],
    case_contingencies: list[Contingency],
) -> None:
    """Assert that the shapes of the loadflow results and solver results are consistent.

    Parameters
    ----------
    n_0 : Float[ArrayLike, " n_branches"]
        The N-0 results from the loadflow, shape (n_branches,)
    n_1 : Float[ArrayLike, " n_cases n_branches"]
        The N-1 results from the loadflow, shape (n_cases, n_branches)
    success : Bool[ArrayLike, " n_cases"]
        The success flags from the loadflow, length n_cases
    n_0_solver : Float[ArrayLike, " n_branches"]
        The N-0 results from the solver, shape (n_branches,)
    n_1_solver : Float[ArrayLike, " n_cases n_branches"]
        The N-1 results from the solver, shape (n_cases, n_branches)
    success_solver : Bool[ArrayLike, " n_cases"]
        The success flags from the solver, length n_cases
    case_contingencies : list[Contingency]
        The list of contingencies that are considered as cases (i.e. all contingencies except the base case)

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the shapes are not consistent
    """
    assert n_0.shape == n_0_solver.shape, (
        f"Shape mismatch between solver and loadflow results: {n_0.shape} vs {n_0_solver.shape}"
    )
    assert n_1.shape == n_1_solver.shape, (
        f"Shape mismatch between solver and loadflow results: {n_1.shape} vs {n_1_solver.shape}"
    )

    assert len(case_contingencies) == n_1.shape[0], (
        f"Number of cases in nminus1_definition ({len(case_contingencies)}) does not match n_1 shape {n_1.shape}"
    )
    assert len(case_contingencies) == len(success), (
        f"Number of loadflow success entries ({len(success)}) does not match number of contingencies"
    )
    assert len(case_contingencies) == len(success_solver), (
        f"Number of solver success entries ({len(success_solver)}) does not match number of contingencies"
    )


def _get_inactive_branches_powsybl(active_topology_network: Network) -> set[str]:
    """Get the set of branch ids that are inactive in the active topology according to powsybl.

    Parameters
    ----------
    active_topology_network : Network
        The active topology as a powsybl Network object.

    Returns
    -------
    set[str]
        The set of branch ids that are inactive in the active topology according to powsybl
    """
    branches = active_topology_network.get_branches(attributes=["connected1", "connected2"])
    branches = branches[~branches.connected1 | ~branches.connected2]
    out_of_service_branches_powsybl = set(branches.index)
    return out_of_service_branches_powsybl


def get_solver_results(
    actions: list[int],
    disconnections: list[int] | None,
    timestep: int,
    dynamic_information: DynamicInformation,
    solver_config: SolverConfig,
) -> tuple[Float[ArrayLike, " n_branches"], Float[ArrayLike, " n_cases n_branches"], Bool[ArrayLike, " n_cases"]]:
    """Get the solver results for the given actions and disconnections, for the specified timestep.

    Parameters
    ----------
    actions : list[int]
        The actions that are taken in the grid, as indices into the action set.
    disconnections : list[int] | None
        The disconnections as indices into the disconnectable branches set. Can be None if no dis
    connections are taken.
    timestep : int
        The timestep to get the results for.
    dynamic_information : DynamicInformation
        The dynamic information of the problem, containing the action set and disconnectable branches.
    solver_config : SolverConfig
        The solver configuration, containing the contingency ids and other solver parameters.

    Returns
    -------
    tuple[Float[ArrayLike, " n_branches"], Float[ArrayLike, " n_cases n_branches"], Bool[ArrayLike, " n_cases"]]
        The n_0, n_1 and success results from the solver for the given actions and disconnections.
    """
    if disconnections is not None and len(disconnections) > 0:
        disconnections = jnp.array(disconnections)
    else:
        disconnections = None
    (n_0_solver, n_1_solver), success_solver = run_solver_symmetric(
        topologies=ActionIndexComputations(
            action=jnp.array([actions], dtype=int),
            pad_mask=jnp.array([True]),
        ),
        disconnections=disconnections[None] if disconnections is not None else None,
        injections=None,
        dynamic_information=dynamic_information,
        solver_config=solver_config,
        aggregate_output_fn=lambda lf_res: (lf_res.n_0_matrix, lf_res.n_1_matrix),
    )

    # Remove the batch and timestep dimensions
    n_0_solver = n_0_solver[0, timestep]
    n_1_solver = n_1_solver[0, timestep]
    success_solver = success_solver[0]
    return n_0_solver, n_1_solver, success_solver
