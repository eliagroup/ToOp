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
from pydantic import BaseModel
from toop_engine_dc_solver.jax import (
    run_solver_symmetric,
)
from toop_engine_dc_solver.jax.types import ActionIndexComputations, StaticInformation
from toop_engine_interfaces.loadflow_result_helpers_polars import extract_solver_matrices_polars
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition


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


def validate_loadflow_results(
    static_information: StaticInformation,
    nminus1_definition: Nminus1Definition,
    loadflows: LoadflowResultsPolars,
    actions: list[int],
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

    if disconnections is not None and len(disconnections) > 0:
        disconnections = jnp.array(disconnections)
    else:
        disconnections = None

    (n_0_solver, n_1_solver), success_solver = run_solver_symmetric(
        topologies=ActionIndexComputations(
            action=jnp.array([actions]),
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
    assert jnp.all(success), "Solver did not converge for all topologies"
    assert all(success), "Loadflow did not converge"

    assert n_0.shape == n_0_solver.shape, (
        f"Shape mismatch between solver and loadflow results: {n_0.shape} vs {n_0_solver.shape}"
    )
    assert n_1.shape == n_1_solver.shape, (
        f"Shape mismatch between solver and loadflow results: {n_1.shape} vs {n_1_solver.shape}"
    )
    if not validation_parameters.compare_signs:
        n_0 = np.abs(n_0)
        n_1 = np.abs(n_1)
        n_0_solver = np.abs(n_0_solver)
        n_1_solver = np.abs(n_1_solver)

    allclose = partial(
        np.allclose,
        atol=validation_parameters.atol,
        rtol=validation_parameters.rtol,
        equal_nan=validation_parameters.equal_nan,
    )
    messages = []

    if not allclose(n_0, n_0_solver):
        error = np.abs(n_0 - n_0_solver)
        messages.append(f"N-0 does not match, mean error: {error.mean()}, max error: {error.max()}")

    case_ids = [contingency.id for contingency in nminus1_definition.contingencies if not contingency.is_basecase()]
    assert len(case_ids) == n_1.shape[0], (
        f"Number of cases in nminus1_definition ({len(case_ids)}) does not match n_1 shape {n_1.shape}"
    )

    for i, (case, converged) in enumerate(zip(case_ids, success, strict=True)):
        if converged and not allclose(n_1[i, :], n_1_solver[i, :]):
            error = np.abs(n_1[i, :] - n_1_solver[i, :])
            messages.append(f"N-1 for case {case} does not match, mean error: {error.mean()}, max error: {error.max()}")

    assert len(messages) == 0, "\n".join(messages)
