# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from copy import deepcopy

import pandera.typing as pat
from beartype.typing import Optional
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_results import (
    BranchResultSchema,
    ConvergedSchema,
    LoadflowResults,
    NodeResultSchema,
    RegulatingElementResultSchema,
    RegulatingElementType,
    VADiffResultSchema,
)


def add_contingency_to_branch_results(
    branch_results: pat.DataFrame[BranchResultSchema], timestep: int, contingency_id: str, n_elements: int
):
    """Add dummy contingency results to branch results DataFrame."""
    for i in range(n_elements):
        branch_results.loc[(timestep, contingency_id, f"element_{i}", 1), ["i", "p", "q", "loading", "element_name"]] = [
            i * 0.1,
            i * 0.2,
            i * 0.3,
            i * 0.4,
            f"branch_name_{i}",
        ]
        branch_results.loc[(timestep, contingency_id, f"element_{i}", 2), ["i", "p", "q", "loading", "element_name"]] = [
            i * 0.1,
            i * 0.2,
            i * 0.3,
            i * 0.4,
            f"branch_name_{i}",
        ]


def add_contingency_to_node_results(
    node_results: pat.DataFrame[NodeResultSchema], timestep: int, contingency_id: str, n_elements: int
):
    """Add dummy contingency results to node results DataFrame."""
    for i in range(n_elements):
        node_results.loc[(timestep, contingency_id, f"node_{i}"), ["vm", "va", "vm_loading", "element_name"]] = [
            i * 1.01,
            i * 1.1,
            1.0,
            f"node_name_{i}",
        ]


def add_contingency_to_regulating_element_results(
    regulating_element_results: pat.DataFrame[RegulatingElementResultSchema],
    timestep: int,
    contingency_id: str,
    n_elements: int,
):
    """Add dummy contingency results to regulating element results DataFrame."""
    for i in range(n_elements):
        regulating_element_results.loc[
            (timestep, contingency_id, f"regulating_element_{i}"), ["value", "element_name", "regulating_element_type"]
        ] = [i * 1.01, f"regulating_element_name_{i}", RegulatingElementType.SLACK_P.value]


def add_contingency_to_va_diff_results(
    va_diff_results: pat.DataFrame[VADiffResultSchema], timestep: int, contingency_id: str, n_elements: int
):
    """Add dummy contingency results to va diff results DataFrame."""
    for i in range(n_elements):
        va_diff_results.loc[(timestep, contingency_id, f"switch_{i}"), ["va_diff", "element_name"]] = [
            i * 0.05,
            f"va_diff_name_{i}",
        ]


def get_loadflow_results_example(
    job_id: str = "", timestep: int = 0, size: int = 5, contingencies: Optional[list[str]] = None
) -> LoadflowResults:
    """Create an example LoadflowResults object with dummy data."""
    branch_results = get_empty_dataframe_from_model(BranchResultSchema)
    node_results = get_empty_dataframe_from_model(NodeResultSchema)
    regulating_element_results = get_empty_dataframe_from_model(RegulatingElementResultSchema)
    va_diff_results = get_empty_dataframe_from_model(VADiffResultSchema)
    converged = get_empty_dataframe_from_model(ConvergedSchema)
    if contingencies is None:
        contingencies = ["contingency"]
    for i, contingency in enumerate(contingencies):
        add_contingency_to_branch_results(branch_results, timestep, contingency, size)
        add_contingency_to_node_results(node_results, timestep, contingency, size)
        add_contingency_to_regulating_element_results(regulating_element_results, timestep, contingency, size)
        add_contingency_to_va_diff_results(va_diff_results, timestep, contingency, size)
        if size > 0:
            converged.loc[(timestep, contingency), "status"] = "CONVERGED" if i % 2 == 0 else "FAILED"

    return LoadflowResults(
        job_id=job_id,
        branch_results=branch_results,
        node_results=node_results,
        regulating_element_results=regulating_element_results,
        va_diff_results=va_diff_results,
        converged=converged,
        additional_information=["This is generated test data"],
        warnings=["warning1"],
    )


def test_loadflow_results_equality_identical():
    job_id = "job1"
    lfr1 = get_loadflow_results_example(job_id=job_id, timestep=0, size=5)
    lfr2 = deepcopy(lfr1)
    assert lfr1 == lfr2


def test_loadflow_results_equality_floats_rounded():
    job_id = "job1"
    lfr1 = get_loadflow_results_example(job_id=job_id, timestep=0, size=5)
    lfr2 = deepcopy(lfr1)
    lfr1.branch_results.i += 0.0000001
    assert lfr1 == lfr2


def test_loadflow_results_equality_different_job_id():
    lfr1 = get_loadflow_results_example(job_id="job1", timestep=0, size=5)
    lfr2 = deepcopy(lfr1)
    lfr2.job_id = "job2"
    assert lfr1 != lfr2


def test_loadflow_results_equality_different_branch_results():
    lfr1 = get_loadflow_results_example(job_id="job1", timestep=0, size=5)
    lfr2 = deepcopy(lfr1)
    lfr2.branch_results.i += 10
    assert lfr1 != lfr2


def test_loadflow_results_equality_different_node_results():
    lfr1 = get_loadflow_results_example(job_id="job1", timestep=0, size=5)
    lfr2 = deepcopy(lfr1)
    lfr2.node_results.vm += 10
    assert lfr1 != lfr2


def test_loadflow_results_equality_different_regulating_results():
    lfr1 = get_loadflow_results_example(job_id="job1", timestep=0, size=5)
    lfr2 = deepcopy(lfr1)
    lfr2.regulating_element_results.value += 10
    assert lfr1 != lfr2


def test_loadflow_results_equality_different_converged_results():
    lfr1 = get_loadflow_results_example(job_id="job1", timestep=0, size=5)
    lfr2 = deepcopy(lfr1)
    lfr2.converged.status = "SKIPPED"
    assert lfr1 != lfr2


def test_loadflow_results_equality_different_va_diff_results():
    lfr1 = get_loadflow_results_example(job_id="job1", timestep=0, size=5)
    lfr2 = deepcopy(lfr1)
    lfr2.va_diff_results.va_diff += 10
    assert lfr1 != lfr2


def test_loadflow_results_equality_type_check():
    lfr = get_loadflow_results_example(job_id="job1", timestep=0, size=5)
    assert lfr != "not a LoadflowResults"


def test_loadflow_results_equality_empty():
    lfr1 = get_loadflow_results_example(job_id="job1", timestep=0, size=0)
    lfr2 = get_loadflow_results_example(job_id="job1", timestep=0, size=0)
    assert lfr1 == lfr2


def test_loadflow_results_equality_ifferent_index():
    lfr1 = get_loadflow_results_example(job_id="job1", timestep=0, size=5)
    lfr2 = deepcopy(lfr1)
    lfr2.branch_results.index = lfr2.branch_results.index.set_levels(lfr2.branch_results.index.levels[0] + 1, level=0)
    assert lfr1 != lfr2
