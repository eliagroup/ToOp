# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Loadflow result helpers for polars LazyFrames or DataFrames.

Holds functions to work with the loadflow results interfaces.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
from fsspec import AbstractFileSystem
from jaxtyping import Bool, Float
from toop_engine_interfaces.loadflow_results import (
    BranchSide,
    ConvergenceStatus,
)
from toop_engine_interfaces.loadflow_results_polars import (
    BranchResultSchemaPolars,
    ConvergedSchemaPolars,
    LoadflowResultsPolars,
    NodeResultSchemaPolars,
    RegulatingElementResultSchemaPolars,
    VADiffResultSchemaPolars,
)
from toop_engine_interfaces.messages.lf_service.stored_loadflow_reference import StoredLoadflowReference
from toop_engine_interfaces.nminus1_definition import GridElement, Nminus1Definition


def save_loadflow_results_polars(
    fs: AbstractFileSystem, file_path: str | Path, loadflows: LoadflowResultsPolars
) -> StoredLoadflowReference:
    """Save loadflow results to a file in hdf5 format.

    Parameters
    ----------
    fs : AbstractFileSystem
        The filesystem to use to save the results. This can be a local filesystem or an object store like S3 or Azure, using
        the fsspec library. For writing to local disk, you should use the DirFilesystem to inject a base path like this:
        ```python
        from fsspec.implementations.local import DirFileSystem
        fs = DirFileSystem(base_path="/path/to/base")
        ```
        Similarly, buckets can be used with the appropriate fsspec filesystem like adbs
    file_path: str | Path
        The file path where to save the results to. This is relative to the base name or bucket defined in the storage
    loadflows : LoadflowResultsPolars
        The loadflow results to save.

    Returns
    -------
    StoredLoadflowReference
        A reference to the stored loadflow results.
    """
    file_path = str(file_path)
    fs.makedirs(file_path, exist_ok=True)
    metadata = {
        "job_id": loadflows.job_id,
        "warnings": loadflows.warnings,
        "additional_information": loadflows.additional_information,
    }
    with fs.open(file_path + "/metadata.json", "w") as f:
        json.dump(metadata, f)

    with fs.open(file_path + "/branch_results.parquet", "wb") as f:
        loadflows.branch_results.sink_parquet(f)
    with fs.open(file_path + "/node_results.parquet", "wb") as f:
        loadflows.node_results.sink_parquet(f)
    with fs.open(file_path + "/regulating_element_results.parquet", "wb") as f:
        loadflows.regulating_element_results.sink_parquet(f)
    with fs.open(file_path + "/converged.parquet", "wb") as f:
        loadflows.converged.sink_parquet(f)
    with fs.open(file_path + "/va_diff_results.parquet", "wb") as f:
        loadflows.va_diff_results.sink_parquet(f)

    return StoredLoadflowReference(
        relative_path=str(file_path),
    )


def load_loadflow_results_polars(
    fs: AbstractFileSystem, reference: StoredLoadflowReference, validate: bool = True
) -> LoadflowResultsPolars:
    """Load loadflow results from a StoredLoadflowReference.

    Parameters
    ----------
    fs: AbstractFileSystem
        The filesystem to use to load the results. This can be a local filesystem or an object store like S3 or Azure, using
        the fsspec library.
    reference: StoredLoadflowReference
        The reference to the stored loadflow results.
    validate: bool
        Whether to validate the loaded results against the schemas defined in the interfaces.
        For dataframes with a lot of data this can take a few seconds.

    Returns
    -------
    LoadflowResults
        The loaded loadflow results.
    """
    file_path = str(reference.relative_path)
    with fs.open(file_path + "/metadata.json", "r") as f:
        metadata = json.load(f)
    job_id = metadata["job_id"]
    warnings = metadata["warnings"]
    additional_information = metadata["additional_information"]

    with fs.open(file_path + "/branch_results.parquet", "rb") as f:
        branch_results = pl.scan_parquet(f)
    with fs.open(file_path + "/node_results.parquet", "rb") as f:
        node_results = pl.scan_parquet(f)
    with fs.open(file_path + "/regulating_element_results.parquet", "rb") as f:
        regulating_element_results = pl.scan_parquet(f)
    with fs.open(file_path + "/converged.parquet", "rb") as f:
        converged = pl.scan_parquet(f)
    with fs.open(file_path + "/va_diff_results.parquet", "rb") as f:
        va_diff_results = pl.scan_parquet(f)

    if validate:
        return LoadflowResultsPolars(
            job_id=job_id,
            branch_results=BranchResultSchemaPolars.validate(branch_results),
            node_results=NodeResultSchemaPolars.validate(node_results),
            regulating_element_results=RegulatingElementResultSchemaPolars.validate(regulating_element_results),
            converged=ConvergedSchemaPolars.validate(converged),
            va_diff_results=VADiffResultSchemaPolars.validate(va_diff_results),
            warnings=warnings,
            additional_information=additional_information,
        )

    return LoadflowResultsPolars.model_construct(
        job_id=job_id,
        branch_results=branch_results,
        node_results=node_results,
        regulating_element_results=regulating_element_results,
        converged=converged,
        va_diff_results=va_diff_results,
        warnings=warnings,
        additional_information=additional_information,
    )


def concatenate_loadflow_results_polars(
    loadflow_results_list: list[LoadflowResultsPolars],
) -> LoadflowResultsPolars:
    """Concatenate the results of the loadflow results.

    Parameters
    ----------
    loadflow_results_list : list
        The list of loadflow results to concatenate

    Returns
    -------
    LoadflowResultsPolars
        The concatenated loadflow results
    """
    assert len(loadflow_results_list) > 0, "The list of loadflow results must not be empty"
    assert all(loadflow_results_list[0].job_id == res.job_id for res in loadflow_results_list), (
        "All loadflow results must have the same job_id"
    )
    # make sure None values are not included in the concatenation
    branch_results_list = [res.branch_results for res in loadflow_results_list if res.branch_results is not None]
    node_results_list = [res.node_results for res in loadflow_results_list if res.node_results is not None]
    regulating_element_results_list = [
        res.regulating_element_results for res in loadflow_results_list if res.regulating_element_results is not None
    ]
    converged_list = [res.converged for res in loadflow_results_list if res.converged is not None]
    va_diff_results_list = [res.va_diff_results for res in loadflow_results_list if res.va_diff_results is not None]

    branch_results = pl.concat(branch_results_list, how="vertical")
    node_results = pl.concat(node_results_list, how="vertical")
    regulating_element_results = pl.concat(regulating_element_results_list, how="vertical")
    converged = pl.concat(converged_list, how="vertical")
    va_diff_results = pl.concat(va_diff_results_list, how="vertical")
    warnings = [warning for lf_results in loadflow_results_list for warning in lf_results.warnings]
    additional_information = [
        additional_information
        for lf_results in loadflow_results_list
        for additional_information in lf_results.additional_information
    ]
    return LoadflowResultsPolars(
        job_id=loadflow_results_list[0].job_id,
        branch_results=branch_results,
        node_results=node_results,
        regulating_element_results=regulating_element_results,
        converged=converged,
        va_diff_results=va_diff_results,
        warnings=warnings,
        additional_information=additional_information,
    )


def select_timestep_polars(loadflow_results: LoadflowResultsPolars, timestep: int) -> LoadflowResultsPolars:
    """Select a single timestep from the loadflow results.

    Parameters
    ----------
    loadflow_results : LoadflowResultsPolars
        The loadflow results to select the timestep from.
    timestep : int
        The timestep to select.

    Returns
    -------
    LoadflowResultsPolars
        The loadflow results for the selected timestep.
    """
    return LoadflowResultsPolars(
        job_id=loadflow_results.job_id,
        branch_results=loadflow_results.branch_results.filter(pl.col("timestep") == timestep),
        node_results=loadflow_results.node_results.filter(pl.col("timestep") == timestep),
        regulating_element_results=loadflow_results.regulating_element_results.filter(pl.col("timestep") == timestep),
        converged=loadflow_results.converged.filter(pl.col("timestep") == timestep),
        va_diff_results=loadflow_results.va_diff_results.filter(pl.col("timestep") == timestep),
        warnings=loadflow_results.warnings,
        additional_information=loadflow_results.additional_information,
    )


def extract_branch_results_polars(
    branch_results: BranchResultSchemaPolars,
    timestep: int,
    contingencies: list[str],
    monitored_branches: list[GridElement],
    basecase: str,
) -> tuple[Float[np.ndarray, " n_branches_monitored"], Float[np.ndarray, " n_contingencies n_branches_monitored"]]:
    """Extract the branch results for a specific timestep.

    Parameters
    ----------
    branch_results: BranchResultSchema,
        The branch results dataframe to extract the branch results from.
    timestep : int
        The selected timestep to pull from the loadflow results.
    basecase : str
        The basecase contingency id to use for the N-0 results.
    contingencies : list[str]
        The list of contingencies to extract the results for.
    monitored_branches : list[GridElement]
        The list of monitored branches to extract the results for.
        buses switches etc should not be included here, only branches.

    Returns
    -------
    Float[np.ndarray, " n_contingencies n_branches_monitored"]
        The branch results with the following:
        - shape (n_contingencies, n_branches_monitored)
        - only the p values of the monitored branches at the from-end
        - For three winding transformers, the p values are split into three rows for each side (hv, mv, lv).
    """
    assert basecase not in contingencies, "Basecase contingency should not be in the list of N-k contingencies"
    n_monitored_branches = len(monitored_branches)
    n_contingencies = len(contingencies)
    if (n_monitored_branches == 0) or (n_contingencies == 0 and basecase is None):
        # If there are no monitored branches, return empty arrays
        return np.full(n_monitored_branches, dtype=float), np.full((n_contingencies, n_monitored_branches), dtype=float)
    # Get the branch results for the given job_id and timestep
    three_winding_side_dict = {
        "trafo3w_hv": BranchSide.ONE.value,
        "trafo3w_mv": BranchSide.TWO.value,
        "trafo3w_lv": BranchSide.THREE.value,
    }
    all_cases = [basecase, *contingencies]

    normal_branches_ids = [elem.id for elem in monitored_branches if elem.type not in three_winding_side_dict]

    timestep_df = branch_results.select(pl.col("timestep").unique())
    contingency_df = timestep_df.join(pl.LazyFrame({"contingency": all_cases}), how="cross")

    normal_branch_df = contingency_df.join(pl.LazyFrame({"element": normal_branches_ids}), how="cross").with_columns(
        side=BranchSide.ONE.value
    )

    trafo3w_dfs = []
    for trafo_type, side in three_winding_side_dict.items():
        three_winding_branches = [element.id for element in monitored_branches if element.type == trafo_type]
        trafo_side_df = contingency_df.join(pl.LazyFrame({"element": three_winding_branches}), how="cross").with_columns(
            side=side
        )
        trafo3w_dfs.append(trafo_side_df)
    all_branches_df = pl.concat([normal_branch_df, *trafo3w_dfs], how="vertical")

    merge_columns = ["timestep", "contingency", "element", "side"]
    all_p_results = (
        all_branches_df.join(
            branch_results.filter(pl.col("timestep") == timestep).select([*merge_columns, "p"]), on=merge_columns, how="left"
        )
        .fill_null(0.0)
        .fill_nan(0.0)
    )

    n_0_results = all_p_results.filter(pl.col("contingency") == basecase).collect()
    n_0_vector = n_0_results["p"].to_numpy()

    sort_by = [
        pl.col("contingency").cast(pl.Enum(contingencies)),
        pl.col("element").cast(pl.Enum([elem.id for elem in monitored_branches])),
    ]
    is_not_basecase = pl.col("contingency") != basecase
    n_1_results = all_p_results.filter(is_not_basecase).sort(sort_by).collect()
    n_1_array = n_1_results["p"].to_numpy().reshape(n_contingencies, n_monitored_branches)
    return n_0_vector, n_1_array


def extract_node_matrices_polars(
    node_results: NodeResultSchemaPolars,
    timestep: int,
    contingencies: list[str],
    monitored_nodes: list[GridElement],
    basecase: str = "BASECASE",
) -> tuple[
    Float[np.ndarray, " n_nodes_monitored"],
    Float[np.ndarray, "  n_nodes_monitored"],
    Float[np.ndarray, " n_contingencies n_nodes_monitored"],
    Float[np.ndarray, " n_contingencies n_nodes_monitored"],
]:
    """Extract the node results for a specific timestep.

    Parameters
    ----------
    node_results: NodeResultSchemaPolars,
        The node results polars dataframe to extract the node results from.
    timestep : int
        The selected timestep to pull from the loadflow results.
    basecase : str
        The basecase contingency id to use for the N-0 results.
    contingencies : list[str]
        The list of contingencies to extract the results for.
    monitored_nodes : list[GridElement]
        The list of monitored nodes to extract the results for.
        buses switches etc should not be included here, only nodes.

    Returns
    -------
    vm_n0 : Float[np.ndarray, " n_nodes_monitored"]
        The voltage magnitude results for the basecase contingency at the monitored nodes.
    va_n0 : Float[np.ndarray, " n_nodes_monitored"]
        The voltage angle results for the basecase contingency at the monitored nodes.
    vm_n1 : Float[np.ndarray, " n_contingencies n_nodes_monitored"]
        The voltage magnitude results for the contingencies at the monitored nodes.
    va_n1 : Float[np.ndarray, " n_contingencies n_nodes_monitored"]
        The voltage angle results for the contingencies at the monitored nodes.
    """
    assert basecase not in contingencies, "Basecase contingency should not be in the list of N-k contingencies"
    n_contingencies = len(contingencies)
    n_monitored_nodes = len(monitored_nodes)
    if (n_monitored_nodes == 0) or (n_contingencies == 0 and basecase is None):
        # If there are no monitored nodes, return empty arrays
        return (
            np.full(n_monitored_nodes, dtype=float),
            np.full(n_monitored_nodes, dtype=float),
            np.full((n_contingencies, n_monitored_nodes), dtype=float),
            np.full((n_contingencies, n_monitored_nodes), dtype=float),
        )

    # Get the node results for the given job_id and timestep
    contingency_df = pl.LazyFrame({"contingency": [basecase, *contingencies]})
    all_cases_df = contingency_df.join(pl.LazyFrame({"element": [elem.id for elem in monitored_nodes]}), how="cross")
    node_results = all_cases_df.join(
        node_results.filter(pl.col("timestep") == timestep).select(["contingency", "element", "vm", "va"]),
        on=["contingency", "element"],
        how="left",
    )
    v_n0 = node_results.filter(pl.col("contingency") == basecase).select(["vm", "va"]).collect()
    vm_n0 = v_n0["vm"].to_numpy()
    va_n0 = v_n0["va"].to_numpy()

    v_n1 = node_results.filter(pl.col("contingency") != basecase).select(["vm", "va"]).collect()
    vm_n1 = v_n1["vm"].to_numpy()
    va_n1 = v_n1["va"].to_numpy()
    # reshape the results to have the contingencies as first dimension
    vm_n1_reshaped = vm_n1.reshape(len(contingencies), len(monitored_nodes))
    va_n1_reshaped = va_n1.reshape(len(contingencies), len(monitored_nodes))
    return vm_n0, va_n0, vm_n1_reshaped, va_n1_reshaped


def extract_solver_matrices_polars(
    loadflow_results: LoadflowResultsPolars,
    nminus1_definition: Nminus1Definition,
    timestep: int,
) -> tuple[
    Float[np.ndarray, " n_branches_monitored"],
    Float[np.ndarray, " n_solver_contingencies n_branches_monitored"],
    Bool[np.ndarray, " n_solver_contingencies"],
]:
    """Extract the N-0 and N-1 matrices in a similar format to the DC solver.

    Parameters
    ----------
    loadflow_results : LoadflowResults
        The loadflow results to extract the matrices from.
    nminus1_definition : Nminus1Definition
        The N-1 definition to use for the contingencies and monitored elements.
    timestep : int
        The selected timestep to pull from the loadflow results.

    Returns
    -------
    Float[np.ndarray, " n_branches_monitored"]
        The N-0 matrix
    Float[np.ndarray, " n_solver_contingencies n_branches_monitored"]
        The N-1 matrix
    Bool[np.ndarray, " n_solver_contingencies"]
        The convergence status of the contingencies in the N-1 matrix
        True if converged or not calculated, False if not converged.
    """
    basecase = next((cont for cont in nminus1_definition.contingencies if cont.is_basecase()), None)
    assert basecase is not None, "No basecase contingency found in the N-1 definition."
    contingency_order = [cont.id for cont in nminus1_definition.contingencies if not cont.is_basecase()]

    # Only consider the selected timestep
    timestep_filter = pl.col("timestep") == timestep
    # For n-1 results, only consider non-basecase contingencies
    not_basecase_filter = pl.col("contingency") != basecase.id
    # A contingency is considered successful if it converged or if no calculation was performed
    # (e.g. for disconnected elements)
    is_success = pl.col("status").is_in([ConvergenceStatus.CONVERGED.value, ConvergenceStatus.NO_CALCULATION.value])

    filtered_converged = loadflow_results.converged.filter(timestep_filter & not_basecase_filter)
    sorted_converged = filtered_converged.sort(pl.col("contingency").cast(pl.Enum(contingency_order)))
    status_converged = sorted_converged.select(is_success).collect()

    success = status_converged["status"].to_numpy()

    branch_elements = [elem for elem in nminus1_definition.monitored_elements if elem.kind == "branch"]
    n_0_vector, n1_matrix = extract_branch_results_polars(
        branch_results=loadflow_results.branch_results,
        timestep=timestep,
        contingencies=contingency_order,
        monitored_branches=branch_elements,
        basecase=basecase.id,
    )

    return n_0_vector, n1_matrix, success
