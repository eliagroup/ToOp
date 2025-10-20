from copy import deepcopy

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


def get_loadflow_results_example(job_id: str = "", timestep: int = 0, size: int = 5) -> LoadflowResults:
    """Create an example LoadflowResults object with dummy data."""
    branch_results = get_empty_dataframe_from_model(BranchResultSchema)
    node_results = get_empty_dataframe_from_model(NodeResultSchema)
    regulating_element_results = get_empty_dataframe_from_model(RegulatingElementResultSchema)
    va_diff_results = get_empty_dataframe_from_model(VADiffResultSchema)
    converged = get_empty_dataframe_from_model(ConvergedSchema)
    for i in range(size):
        branch_results.loc[(timestep, "contingency", f"element_{i}", 1), ["i", "p", "q", "loading", "element_name"]] = [
            i * 0.1,
            i * 0.2,
            i * 0.3,
            i * 0.4,
            f"branch_name_{i}",
        ]
        branch_results.loc[(timestep, "contingency", f"element_{i}", 2), ["i", "p", "q", "loading", "element_name"]] = [
            i * 0.1,
            i * 0.2,
            i * 0.3,
            i * 0.4,
            f"branch_name_{i}",
        ]
        node_results.loc[(timestep, "contingency", f"node_{i}"), ["vm", "va", "vm_loading", "element_name"]] = [
            i * 1.01,
            i * 1.1,
            1.0,
            f"node_name_{i}",
        ]
        regulating_element_results.loc[
            (timestep, "contingency", f"regulating_element_{i}"), ["value", "element_name", "regulating_element_type"]
        ] = [i * 1.01, f"regulating_element_name_{i}", RegulatingElementType.SLACK_P.value]
        va_diff_results.loc[(timestep, "contingency", f"va_diff_{i}"), ["va_diff", "element_name"]] = [
            i * 0.05,
            f"va_diff_name_{i}",
        ]
        converged.loc[(timestep, "contingency"), ["status"]] = ["CONVERGED" if i % 2 == 0 else "FAILED"]

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
