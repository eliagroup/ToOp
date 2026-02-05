# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from test_loadflow_results_new import get_loadflow_results_example
from toop_engine_interfaces.loadflow_result_helpers import (
    concatenate_loadflow_results,
    extract_branch_results,
    extract_solver_matrices,
    get_failed_branch_results,
    get_failed_node_results,
    load_loadflow_results,
    save_loadflow_results,
    select_timestep,
)
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, Nminus1Definition


def test_save_and_load_loadflow_results(tmp_path):
    loadflow_results = get_loadflow_results_example(job_id="test", timestep=0, size=5)

    fs = DirFileSystem(tmp_path)
    ref = save_loadflow_results(fs, "test_loadflow_results", loadflow_results)
    loadflow_results_loaded = load_loadflow_results(fs, ref)
    assert loadflow_results_loaded == loadflow_results


def test_save_and_load_loadflow_results_no_validate(tmp_path):
    loadflow_results = get_loadflow_results_example(job_id="test", timestep=0, size=5)

    fs = DirFileSystem(tmp_path)
    ref = save_loadflow_results(fs, "test_loadflow_results", loadflow_results)
    loadflow_results_loaded = load_loadflow_results(fs, ref, validate=False)
    assert loadflow_results == loadflow_results_loaded, "Loadflow results should be equal even when validate is False"


def test_extract_branch_results():
    contingencies = ["BASECASE", "contingency_1", "contingency_2", "contingency_3"]
    res = get_loadflow_results_example(job_id="test_job", timestep=0, size=50, contingencies=contingencies)
    contingencies = res.branch_results.reset_index()["contingency"].unique().tolist()
    monitored_branches = res.branch_results.reset_index()["element"].unique().tolist()
    monitored_elements = [GridElement(id=elem, name=elem, kind="branch", type="line") for elem in monitored_branches]
    _, matrix = extract_branch_results(
        branch_results=res.branch_results,
        basecase="BASECASE",
        contingencies=contingencies[1:],
        monitored_branches=monitored_elements,
        timestep=0,
    )
    assert matrix.shape == (len(contingencies) - 1, len(monitored_branches))
    assert matrix.dtype == float
    assert np.all(np.isfinite(matrix))

    _, matrix_2 = extract_branch_results(
        branch_results=res.branch_results,
        basecase="BASECASE",
        contingencies=[cont for cont in reversed(contingencies[1:])],
        monitored_branches=monitored_elements,
        timestep=0,
    )
    assert matrix_2.shape == (len(contingencies) - 1, len(monitored_branches))
    assert matrix_2.dtype == float
    assert np.all(np.isfinite(matrix_2))

    assert np.all(matrix == matrix_2[::-1, :])  # Check that the order of contingencies does not change the result


def test_select_timestep():
    loadflow_results_0 = get_loadflow_results_example(timestep=0, size=10)
    loadflow_results_1 = get_loadflow_results_example(timestep=1, size=10)
    loadflow_results = concatenate_loadflow_results([loadflow_results_0, loadflow_results_1])

    timesteps = loadflow_results.branch_results.reset_index()["timestep"].unique()
    assert len(timesteps) > 1

    loadflow_results_new = select_timestep(loadflow_results, timesteps[0])
    timesteps_new = loadflow_results_new.branch_results.reset_index()["timestep"].unique()
    assert len(timesteps_new) == 1
    assert timesteps_new[0] == timesteps[0]


def test_select_timestep_empty():
    loadflow_results = get_loadflow_results_example(job_id="test_job", timestep=0, size=0)

    timesteps = loadflow_results.branch_results.reset_index()["timestep"].unique()
    assert len(timesteps) == 0

    loadflow_results_new = select_timestep(loadflow_results, 0)
    assert loadflow_results_new.branch_results.empty
    assert loadflow_results_new.node_results.empty
    assert loadflow_results_new.regulating_element_results.empty
    assert loadflow_results_new.converged.empty
    assert loadflow_results_new.va_diff_results.empty


def test_extract_solver_matrices():
    contingencies = ["BASECASE", "contingency1", "contingency2"]
    loadflow_results = get_loadflow_results_example(job_id="test_job", timestep=0, size=5, contingencies=contingencies)
    monitored_elements = loadflow_results.branch_results.reset_index()["element"].unique().tolist()

    n1_contingencies = [
        Contingency(id=cont, elements=[GridElement(id=cont, name=cont, kind="branch", type="line")])
        for cont in contingencies[1:]
    ]
    n1_contingencies.insert(0, Contingency(id="BASECASE", elements=[]))
    n1_monitored_elements = [GridElement(id=elem, name=elem, kind="branch", type="line") for elem in monitored_elements]
    nminus1_def = Nminus1Definition(
        monitored_elements=n1_monitored_elements,
        contingencies=n1_contingencies,
    )

    n_0, n_1, success = extract_solver_matrices(
        loadflow_results=loadflow_results,
        nminus1_definition=nminus1_def,
        timestep=0,
    )
    assert n_0.shape == (len(nminus1_def.monitored_elements),)
    assert n_1.shape == (len(nminus1_def.contingencies) - 1, len(nminus1_def.monitored_elements))
    assert success.shape == (len(nminus1_def.contingencies) - 1,)
    assert n_0.dtype == float
    assert n_1.dtype == float
    assert success.dtype == bool
    assert np.all(np.isfinite(n_0))
    assert np.all(np.isfinite(n_1))
    assert np.any(success)
    contingencies = [contingency for contingency in nminus1_def.contingencies if not contingency.is_basecase()]
    nminus1_def.contingencies = contingencies
    with pytest.raises(AssertionError):
        extract_solver_matrices(
            loadflow_results=loadflow_results,
            nminus1_definition=nminus1_def,
            timestep=0,
        )


def test_concatenate_loadflow_results():
    res_1 = get_loadflow_results_example(job_id="test_job", timestep=0, size=5, contingencies=["BASECASE", "contingency1"])
    res_2 = get_loadflow_results_example(job_id="test_job", timestep=0, size=5, contingencies=["contingency2"])

    res = concatenate_loadflow_results([res_1, res_2])

    assert len(res.additional_information) == len(res_1.additional_information) + len(res_2.additional_information)
    assert len(res.warnings) == len(res_1.warnings) + len(res_2.warnings)
    assert len(res.branch_results) == len(res_1.branch_results) + len(res_2.branch_results)
    assert len(res.node_results) == len(res_1.node_results) + len(res_2.node_results)
    assert len(res.regulating_element_results) == len(res_1.regulating_element_results) + len(
        res_2.regulating_element_results
    )
    assert len(res.va_diff_results) == len(res_1.va_diff_results) + len(res_2.va_diff_results)
    assert len(res.converged) == len(res_1.converged) + len(res_2.converged)

    assert res.node_results.loc[res_1.node_results.index].equals(res_1.node_results)
    assert res.node_results.loc[res_2.node_results.index].equals(res_2.node_results)
    assert res.branch_results.loc[res_1.branch_results.index].equals(res_1.branch_results)
    assert res.branch_results.loc[res_2.branch_results.index].equals(res_2.branch_results)
    assert res.regulating_element_results.loc[res_1.regulating_element_results.index].equals(
        res_1.regulating_element_results
    )
    assert res.regulating_element_results.loc[res_2.regulating_element_results.index].equals(
        res_2.regulating_element_results
    )
    assert res.va_diff_results.loc[res_1.va_diff_results.index].equals(res_1.va_diff_results)
    assert res.va_diff_results.loc[res_2.va_diff_results.index].equals(res_2.va_diff_results)
    assert res.converged.loc[res_1.converged.index].equals(res_1.converged)
    assert res.converged.loc[res_2.converged.index].equals(res_2.converged)

    other_job_res = res_2.model_copy(update={"job_id": "other_test_job"})
    with pytest.raises(AssertionError):
        res = concatenate_loadflow_results([res_1, other_job_res])

    with pytest.raises(AssertionError):
        concatenate_loadflow_results([])


def test_get_failed_branch_results():
    timestep = 0
    failed_outages = ["branch1", "branch2"]
    monitored_2_end_branches = ["branch1", "branch2"]
    monitored_3_end_branches = ["branch3", "branch4"]
    res = get_failed_branch_results(
        timestep=timestep,
        failed_outages=failed_outages,
        monitored_2_end_branches=monitored_2_end_branches,
        monitored_3_end_branches=monitored_3_end_branches,
    )
    expected_length = 1 * len(failed_outages) * (2 * len(monitored_2_end_branches) + 3 * len(monitored_3_end_branches))
    assert len(res) == expected_length
    assert all(res.index.get_level_values("timestep") == timestep), "Wrong timestep in results"
    assert all(res.index.get_level_values("contingency").isin(failed_outages)), "Wrong contingency in results"
    assert all(res.index.get_level_values("element").isin(monitored_2_end_branches + monitored_3_end_branches)), (
        "Wrong element in results"
    )
    assert all(res.index.get_level_values("side").isin([1, 2, 3])), "Wrong side in results"
    assert all(res["loading"].isna()), "Loading should be NaN for failed branches"
    assert all(res["p"].isna()), "Active power should be NaN for failed branches"
    assert all(res["q"].isna()), "Reactive power should be NaN for failed branches"
    assert all(res["i"].isna()), "Current should be NaN for failed branches"

    res = get_failed_branch_results(
        timestep=timestep,
        failed_outages=[],
        monitored_2_end_branches=monitored_2_end_branches,
        monitored_3_end_branches=monitored_3_end_branches,
    )
    assert len(res) == 0, "Result should be empty when no failed outages are provided"

    res = get_failed_branch_results(
        timestep=timestep, failed_outages=failed_outages, monitored_2_end_branches=[], monitored_3_end_branches=[]
    )
    assert len(res) == 0, "Result should be empty when no monitored branches are provided"

    res = get_failed_branch_results(
        timestep=timestep,
        failed_outages=failed_outages,
        monitored_2_end_branches=monitored_2_end_branches,
        monitored_3_end_branches=[],
    )
    assert len(res) == 2 * len(failed_outages) * len(monitored_2_end_branches), "Result should contain only 2-end branches"

    res = get_failed_branch_results(
        timestep=timestep,
        failed_outages=failed_outages,
        monitored_2_end_branches=[],
        monitored_3_end_branches=monitored_3_end_branches,
    )
    assert len(res) == 3 * len(failed_outages) * len(monitored_3_end_branches), "Result should contain only 3-end branches"


def test_get_failed_node_results():
    timestep = 0
    failed_outages = ["branch_1", "branch_2"]
    monitored_nodes = ["node1", "node2", "node3"]
    res = get_failed_node_results(timestep=timestep, failed_outages=failed_outages, monitored_nodes=monitored_nodes)
    expected_length = 1 * len(failed_outages) * len(monitored_nodes)  # timestep * n_contingencies * n_monitored_nodes
    assert len(res) == expected_length
    assert all(res.index.get_level_values("timestep") == timestep), "Wrong timestep in results"
    assert all(res.index.get_level_values("contingency").isin(failed_outages)), "Wrong contingency in results"
    assert all(res.index.get_level_values("element").isin(monitored_nodes)), "Wrong element in results"
    assert all(res["vm"].isna()), "Voltage Magnitude should be NaN for failed nodes"
    assert all(res["va"].isna()), "Voltage Angle should be NaN for failed nodes"
    res = get_failed_node_results(timestep=timestep, failed_outages=[], monitored_nodes=monitored_nodes)
    assert len(res) == 0, "Result should be empty when no failed outages are provided"

    res = get_failed_node_results(timestep=timestep, failed_outages=failed_outages, monitored_nodes=[])
    assert len(res) == 0, "Result should be empty when no monitored nodes are provided"
