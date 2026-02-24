# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Loadflow result helpers. Holds functions to work with the loadflow results interfaces."""

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
import pandera.typing as pat
import polars as pl
from beartype.typing import Optional, Union
from fsspec import AbstractFileSystem
from jaxtyping import Bool, Float
from toop_engine_interfaces.loadflow_results import (
    BranchResultSchema,
    BranchSide,
    ConvergedSchema,
    ConvergenceStatus,
    LoadflowResults,
    NodeResultSchema,
    RegulatingElementResultSchema,
    VADiffResultSchema,
)
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.messages.lf_service.stored_loadflow_reference import StoredLoadflowReference
from toop_engine_interfaces.nminus1_definition import GridElement, Nminus1Definition


def save_loadflow_results(
    fs: AbstractFileSystem, file_path: str | Path, loadflows: LoadflowResults
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
    loadflows : LoadflowResults
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
        loadflows.branch_results.to_parquet(f)
    with fs.open(file_path + "/node_results.parquet", "wb") as f:
        loadflows.node_results.to_parquet(f)
    with fs.open(file_path + "/regulating_element_results.parquet", "wb") as f:
        loadflows.regulating_element_results.to_parquet(f)
    with fs.open(file_path + "/converged.parquet", "wb") as f:
        loadflows.converged.to_parquet(f)
    with fs.open(file_path + "/va_diff_results.parquet", "wb") as f:
        loadflows.va_diff_results.to_parquet(f)

    return StoredLoadflowReference(
        relative_path=str(file_path),
    )


def load_loadflow_results(
    fs: AbstractFileSystem, reference: StoredLoadflowReference, validate: bool = True
) -> LoadflowResults:
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
        branch_results = pd.read_parquet(f)
    with fs.open(file_path + "/node_results.parquet", "rb") as f:
        node_results = pd.read_parquet(f)
    with fs.open(file_path + "/regulating_element_results.parquet", "rb") as f:
        regulating_element_results = pd.read_parquet(f)
    with fs.open(file_path + "/converged.parquet", "rb") as f:
        converged = pd.read_parquet(f)
    with fs.open(file_path + "/va_diff_results.parquet", "rb") as f:
        va_diff_results = pd.read_parquet(f)

    if validate:
        return LoadflowResults(
            job_id=job_id,
            branch_results=BranchResultSchema.validate(branch_results),
            node_results=NodeResultSchema.validate(node_results),
            regulating_element_results=RegulatingElementResultSchema.validate(regulating_element_results),
            converged=ConvergedSchema.validate(converged),
            va_diff_results=VADiffResultSchema.validate(va_diff_results),
            warnings=warnings,
            additional_information=additional_information,
        )
    return LoadflowResults.model_construct(
        job_id=job_id,
        branch_results=branch_results,
        node_results=node_results,
        regulating_element_results=regulating_element_results,
        converged=converged,
        va_diff_results=va_diff_results,
        warnings=warnings,
        additional_information=additional_information,
    )


def concatenate_loadflow_results(
    loadflow_results_list: list[LoadflowResults],
) -> LoadflowResults:
    """Concatenate the results of the loadflow results.

    Parameters
    ----------
    loadflow_results_list : list
        The list of loadflow results to concatenate

    Returns
    -------
    LoadflowResults
        The concatenated loadflow results
    """
    assert len(loadflow_results_list) > 0, "The list of loadflow results must not be empty"
    assert all(loadflow_results_list[0].job_id == res.job_id for res in loadflow_results_list), (
        "All loadflow results must have the same job_id"
    )
    branch_results = pd.concat([res.branch_results for res in loadflow_results_list], axis=0)
    node_results = pd.concat([res.node_results for res in loadflow_results_list], axis=0)
    regulating_element_results = pd.concat([res.regulating_element_results for res in loadflow_results_list], axis=0)
    converged = pd.concat([res.converged for res in loadflow_results_list], axis=0)
    va_diff_results = pd.concat([res.va_diff_results for res in loadflow_results_list], axis=0)
    warnings = [warning for lf_results in loadflow_results_list for warning in lf_results.warnings]
    additional_information = [
        additional_information
        for lf_results in loadflow_results_list
        for additional_information in lf_results.additional_information
    ]
    return LoadflowResults(
        job_id=loadflow_results_list[0].job_id,
        branch_results=branch_results,
        node_results=node_results,
        regulating_element_results=regulating_element_results,
        converged=converged,
        va_diff_results=va_diff_results,
        warnings=warnings,
        additional_information=additional_information,
    )


@pa.check_types
def get_failed_branch_results(
    timestep: int, failed_outages: list[str], monitored_2_end_branches: list[str], monitored_3_end_branches: list[str]
) -> pat.DataFrame[BranchResultSchema]:
    """Get the failed branch results.

    Parameters
    ----------
    timestep : int
        The timestep of the results
    failed_outages : list
        The list of failed outages
    monitored_2_end_branches : list
        The list of monitored 2 end branches. i.e. most branches
    monitored_3_end_branches : list
        The list of monitored 3 end branches. i.e. 3 winding transformers

    Returns
    -------
    pat.DataFrame[BranchResultSchema]
        The failed branch results
    """
    # With two Sides
    failed_branch_results = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [[timestep], failed_outages, monitored_2_end_branches, [BranchSide.ONE.value, BranchSide.TWO.value]],
            names=["timestep", "contingency", "element", "side"],
        )
    ).assign(p=np.nan, q=np.nan, i=np.nan, loading=np.nan)
    # Add results for non convergent contingencies
    failed_trafo3w_results = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [
                [timestep],
                failed_outages,
                monitored_3_end_branches,
                [BranchSide.ONE.value, BranchSide.TWO.value, BranchSide.THREE.value],
            ],
            names=["timestep", "contingency", "element", "side"],
        )
    ).assign(p=np.nan, q=np.nan, i=np.nan, loading=np.nan)
    converted_branch_results = pd.concat([failed_branch_results, failed_trafo3w_results], axis=0)
    # add empty element_name and contingency_name columns to match the schema
    converted_branch_results["element_name"] = ""
    converted_branch_results["contingency_name"] = ""
    return converted_branch_results


@pa.check_types
def get_failed_node_results(
    timestep: int, failed_outages: list[str], monitored_nodes: list[str]
) -> pat.DataFrame[NodeResultSchema]:
    """Get the failed node results.

    Parameters
    ----------
    timestep : int
        The timestep of the results
    failed_outages : list
        The list of failed outages
    monitored_nodes : list
        The list of monitored nodes

    Returns
    -------
    pat.DataFrame[NodeResultSchema]
        The failed node results
    """
    failed_node_results = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [[timestep], failed_outages, monitored_nodes],
            names=["timestep", "contingency", "element"],
        )
    ).assign(
        vm=np.nan,
        va=np.nan,
        vm_loading=np.nan,
        p=np.nan,
        q=np.nan,
        basecase_deviation=np.nan,
        element_name="",
        contingency_name="",
    )
    # fill in empty columns to match the schema
    failed_node_results["p"] = np.nan
    failed_node_results["q"] = np.nan
    failed_node_results["basecase_deviation"] = np.nan
    failed_node_results["element_name"] = ""
    failed_node_results["contingency_name"] = ""
    return failed_node_results


def extract_branch_results(
    branch_results: BranchResultSchema,
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
        "trafo3w_hv": [BranchSide.ONE.value],
        "trafo3w_mv": [BranchSide.TWO.value],
        "trafo3w_lv": [BranchSide.THREE.value],
    }
    all_cases = [basecase, *contingencies]
    normal_branches_ids = [elem.id for elem in monitored_branches if elem.type not in three_winding_side_dict]
    normal_branch_multi_idx = [*product([timestep], all_cases, normal_branches_ids, [BranchSide.ONE.value])]
    trafo3w_idx = []
    for trafo_type, sides in three_winding_side_dict.items():
        three_winding_branches = [element.id for element in monitored_branches if element.type == trafo_type]
        multi_idx = product([timestep], all_cases, three_winding_branches, sides)
        trafo3w_idx.extend(multi_idx)

    multi_index = pd.MultiIndex.from_tuples(
        [*normal_branch_multi_idx, *trafo3w_idx],
        names=["timestep", "contingency", "element", "side"],
    )

    # Drop timestep and side, since we do not need them anymore
    p_results = branch_results.reindex(multi_index, fill_value=0.0).droplevel(["side", "timestep"])["p"]

    # bring into correct order
    monitored_branches_order = [elem.id for elem in monitored_branches]

    n_0_vector = p_results.fillna(0.0).loc[basecase, monitored_branches_order].values

    n_1_index = pd.MultiIndex.from_product([contingencies, monitored_branches_order], names=["contingency", "element"])
    n_1_results = p_results.reindex(n_1_index, fill_value=0.0)
    n_1_array = n_1_results.fillna(0.0).values.reshape(len(contingencies), len(monitored_branches_order))
    return n_0_vector, n_1_array


def extract_node_matrices(
    node_results: NodeResultSchema,
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
    node_results: NodeResultSchema,
        The node results dataframe to extract the node results from.
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
    node_results = node_results.xs(timestep, level="timestep")
    # Reindex to ensure all contingencies and monitored nodes are present, fill missing with 0
    product_index = pd.MultiIndex.from_product(
        [[basecase, *contingencies], [elem.id for elem in monitored_nodes]],
        names=["contingency", "element"],
    )
    node_results = node_results.reindex(product_index, fill_value=np.nan)
    vm_n0 = node_results.loc[basecase, "vm"].values
    va_n0 = node_results.loc[basecase, "va"].values

    vm_n1 = node_results.loc[contingencies, :]["vm"].values
    va_n1 = node_results.loc[contingencies, :]["va"].values
    # reshape the results to have the contingencies as first dimension
    vm_n1 = vm_n1.reshape(len(contingencies), len(monitored_nodes))
    va_n1 = va_n1.reshape(len(contingencies), len(monitored_nodes))
    return vm_n0, va_n0, vm_n1, va_n1


def extract_solver_matrices(
    loadflow_results: LoadflowResults,
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

    success = (
        (
            loadflow_results.converged["status"]
            .loc[timestep]
            .isin([ConvergenceStatus.CONVERGED.value, ConvergenceStatus.NO_CALCULATION.value])
        )
        .reindex(contingency_order, fill_value=False)
        .values
    )
    branch_elements = [elem for elem in nminus1_definition.monitored_elements if elem.kind == "branch"]
    n_0_vector, n1_matrix = extract_branch_results(
        branch_results=loadflow_results.branch_results,
        timestep=timestep,
        contingencies=contingency_order,
        monitored_branches=branch_elements,
        basecase=basecase.id,
    )

    return n_0_vector, n1_matrix, success


def select_timestep(loadflow_results: LoadflowResults, timestep: int) -> LoadflowResults:
    """Select a specific timestep from the loadflow results.

    Parameters
    ----------
    loadflow_results : LoadflowResults
        The loadflow results to select the timestep from.
    timestep : int
        The timestep to select.

    Returns
    -------
    LoadflowResults
        The loadflow results for the selected timestep.
    """

    def safe_xs(df: pd.DataFrame) -> pd.DataFrame:
        """Safely select a timestep from a DataFrame."""
        try:
            return df.xs(timestep, level="timestep", drop_level=False)
        except KeyError:
            return df.iloc[0:0]

    return LoadflowResults(
        job_id=loadflow_results.job_id,
        warnings=loadflow_results.warnings,
        additional_information=loadflow_results.additional_information,
        branch_results=safe_xs(loadflow_results.branch_results),
        node_results=safe_xs(loadflow_results.node_results),
        regulating_element_results=safe_xs(loadflow_results.regulating_element_results),
        converged=safe_xs(loadflow_results.converged),
        va_diff_results=safe_xs(loadflow_results.va_diff_results),
    )


def convert_polars_loadflow_results_to_pandas(
    loadflow_results_polars: LoadflowResultsPolars,
) -> LoadflowResults:
    """Convert the LoadflowResultsPolars class to LoadflowResults class.

    Parameters
    ----------
    loadflow_results_polars : LoadflowResultsPolars
        The loadflow results in polars format.

    Returns
    -------
    LoadflowResults
        The loadflow results in pandas format.
    """

    def polars_to_pandas(df: Optional[Union[pl.DataFrame, pl.LazyFrame]]) -> Optional[pd.DataFrame]:
        """Convert a polars DataFrame or LazyFrame to a pandas DataFrame.

        Parameters
        ----------
        df : Optional[Union[pl.DataFrame, pl.LazyFrame]]
            The polars DataFrame or LazyFrame to convert.

        Returns
        -------
        Optional[pd.DataFrame]
            The pandas DataFrame or None if the input was None.
        """
        if df is None:
            return None
        if hasattr(df, "collect"):
            df = df.collect()
        pdf = df.to_pandas()
        # Set multi-index if possible
        index_cols = []
        for col in ["timestep", "contingency", "element", "side"]:
            if col in pdf.columns:
                index_cols.append(col)
        if index_cols:
            pdf = pdf.set_index(index_cols)
        return pdf

    return LoadflowResults(
        job_id=loadflow_results_polars.job_id,
        branch_results=polars_to_pandas(loadflow_results_polars.branch_results),
        node_results=polars_to_pandas(loadflow_results_polars.node_results),
        regulating_element_results=polars_to_pandas(loadflow_results_polars.regulating_element_results),
        converged=polars_to_pandas(loadflow_results_polars.converged),
        va_diff_results=polars_to_pandas(loadflow_results_polars.va_diff_results),
        warnings=loadflow_results_polars.warnings,
        additional_information=loadflow_results_polars.additional_information,
    )


def convert_pandas_loadflow_results_to_polars(loadflow_results: LoadflowResults) -> LoadflowResultsPolars:
    """Convert the LoadflowResults class to LoadflowResultsPolars class.

    Parameters
    ----------
    loadflow_results : LoadflowResults
        The loadflow results in pandas format.

    Returns
    -------
    LoadflowResultsPolars
        The loadflow results in polars format.
    """

    def pandas_to_polars(df: Optional[pd.DataFrame], lazy: bool) -> Optional[pl.DataFrame]:
        """Convert a pandas DataFrame to a polars DataFrame.

        Parameters
        ----------
        df : Optional[pd.DataFrame]
            The pandas DataFrame to convert.
        lazy : bool
            Whether to return a LazyFrame or a DataFrame.

        Returns
        -------
        Optional[pl.DataFrame]
            The polars DataFrame or None if the input was None.
        """
        if df is None:
            return None
        if isinstance(df, pd.DataFrame):
            df = pl.from_pandas(df, include_index=True, nan_to_null=False)
        if lazy:
            df = df.lazy()  # Assume it's a pandas DataFrame
        return df  # Assume it's already a polars DataFrame

    return LoadflowResultsPolars(
        job_id=loadflow_results.job_id,
        branch_results=pandas_to_polars(loadflow_results.branch_results, lazy=True),
        node_results=pandas_to_polars(loadflow_results.node_results, lazy=True),
        regulating_element_results=pandas_to_polars(loadflow_results.regulating_element_results, lazy=True),
        converged=pandas_to_polars(loadflow_results.converged, lazy=True),
        va_diff_results=pandas_to_polars(loadflow_results.va_diff_results, lazy=True),
        warnings=loadflow_results.warnings,
        additional_information=loadflow_results.additional_information,
        lazy=True,
    )
