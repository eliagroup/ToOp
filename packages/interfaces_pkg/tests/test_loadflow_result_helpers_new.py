import numpy as np
import pandapower
import pandas as pd
import pypowsybl
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from polars.testing import assert_frame_equal
from test_loadflow_results_new import get_loadflow_results_example
from toop_engine_contingency_analysis.pandapower import (
    run_contingency_analysis_pandapower,
)
from toop_engine_contingency_analysis.pypowsybl import (
    run_contingency_analysis_powsybl,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_interfaces.loadflow_result_helpers import (
    concatenate_loadflow_results,
    convert_pandas_loadflow_results_to_polars,
    convert_polars_loadflow_results_to_pandas,
    extract_branch_results,
    extract_solver_matrices,
    get_failed_branch_results,
    get_failed_node_results,
    load_loadflow_results,
    save_loadflow_results,
    select_timestep,
)
from toop_engine_interfaces.loadflow_results import BranchSide
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
    net = pypowsybl.network.create_ieee14()
    nminus1_def = Nminus1Definition(
        monitored_elements=[
            GridElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in list(net.get_branches().iterrows())[:10]
        ],
        contingencies=[
            Contingency(id=index, elements=[GridElement(id=index, name=row.name, kind="branch", type=row.type)])
            for index, row in list(net.get_branches().iterrows())[3:7]
        ],
    )

    res = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def, job_id="test_job", timestep=0, method="dc"
    )
    _, matrix = extract_branch_results(
        branch_results=res.branch_results,
        basecase="BASECASE",
        contingencies=[contingency.id for contingency in nminus1_def.contingencies if not contingency.is_basecase()],
        monitored_branches=[element for element in nminus1_def.monitored_elements if element.kind == "branch"],
        timestep=0,
    )
    assert matrix.shape == (len(nminus1_def.contingencies), len(nminus1_def.monitored_elements))
    assert matrix.dtype == float
    assert np.all(np.isfinite(matrix))

    _, matrix_2 = extract_branch_results(
        branch_results=res.branch_results,
        basecase="BASECASE",
        contingencies=[cont.id for cont in reversed(nminus1_def.contingencies)],
        monitored_branches=[element for element in nminus1_def.monitored_elements if element.kind == "branch"],
        timestep=0,
    )
    assert matrix_2.shape == (len(nminus1_def.contingencies), len(nminus1_def.monitored_elements))
    assert matrix_2.dtype == float
    assert np.all(np.isfinite(matrix_2))

    assert np.all(matrix == matrix_2[::-1, :])  # Check that the order of contingencies does not change the result


def test_extract_branch_results_disconnected():
    net = pypowsybl.network.create_ieee14()
    net.disconnect(net.get_branches().index[0])
    nminus1_def = Nminus1Definition(
        monitored_elements=[
            GridElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[
            Contingency(id=index, elements=[GridElement(id=index, name=row.name, kind="branch", type=row.type)])
            for index, row in net.get_branches().iterrows()
        ],
    )

    res = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def, job_id="test_job", timestep=0, method="dc"
    )
    _, matrix = extract_branch_results(
        branch_results=res.branch_results,
        basecase="BASECASE",
        contingencies=[contingency.id for contingency in nminus1_def.contingencies if not contingency.is_basecase()],
        monitored_branches=[element for element in nminus1_def.monitored_elements if element.kind == "branch"],
        timestep=0,
    )
    assert matrix.shape == (len(nminus1_def.contingencies), len(nminus1_def.monitored_elements))
    assert matrix.dtype == float
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix[0, :] == 0.0)  # Check that the disconnected branch has a loading of 0.0 in all contingencies
    assert np.all(matrix[:, 0] == 0.0)  # Check that the first monitored branch has a loading of 0.0 in all contingencies


def test_extract_branch_results_pandapower():
    net = pandapower.networks.case14()
    nminus1_def = Nminus1Definition(
        monitored_elements=[
            GridElement(id=get_globally_unique_id(index, "line"), name=str(row.name), kind="branch", type="line")
            for index, row in net.line.iterrows()
        ],
        contingencies=[
            Contingency(
                id=str(index),
                elements=[
                    GridElement(id=get_globally_unique_id(index, "line"), name=str(row.name), kind="branch", type="line")
                ],
            )
            for index, row in net.line.iterrows()
        ],
    )

    res = run_contingency_analysis_pandapower(
        net=net, n_minus_1_definition=nminus1_def, job_id="test_job", timestep=0, method="dc"
    )
    _, matrix = extract_branch_results(
        branch_results=res.branch_results,
        basecase="BASECASE",
        contingencies=[contingency.id for contingency in nminus1_def.contingencies if not contingency.is_basecase()],
        monitored_branches=[element for element in nminus1_def.monitored_elements if element.kind == "branch"],
        timestep=0,
    )
    assert matrix.shape == (len(nminus1_def.contingencies), len(nminus1_def.monitored_elements))
    assert matrix.dtype == float
    assert np.all(np.isfinite(matrix))


def test_extract_branch_results_pandapower_disconnected():
    net = pandapower.networks.case14()
    net.line.loc[0, "in_service"] = False  # Disconnect the first line
    nminus1_def = Nminus1Definition(
        monitored_elements=[
            GridElement(id=get_globally_unique_id(index, "line"), name=str(row.name), kind="branch", type="line")
            for index, row in net.line.iterrows()
        ],
        contingencies=[
            Contingency(
                id=str(index),
                elements=[
                    GridElement(id=get_globally_unique_id(index, "line"), name=str(row.name), kind="branch", type="line")
                ],
            )
            for index, row in net.line.iterrows()
        ],
    )

    res = run_contingency_analysis_pandapower(
        net=net, n_minus_1_definition=nminus1_def, job_id="test_job", timestep=0, method="dc"
    )
    contingencies = [contingency.id for contingency in nminus1_def.contingencies if not contingency.is_basecase()]
    _, matrix = extract_branch_results(
        branch_results=res.branch_results,
        basecase="BASECASE",
        contingencies=[contingency.id for contingency in nminus1_def.contingencies if not contingency.is_basecase()],
        monitored_branches=[element for element in nminus1_def.monitored_elements if element.kind == "branch"],
        timestep=0,
    )
    assert matrix.shape == (len(nminus1_def.contingencies), len(nminus1_def.monitored_elements))
    assert matrix.dtype == float
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix[0, :] == 0.0)  # Check that the disconnected branch has a loading of 0.0 in all contingencies
    assert np.all(matrix[:, 0] == 0.0)  # Check that the first monitored branch has a loading of 0.0 in all contingencies


def test_extract_solver_matrices():
    net = pypowsybl.network.create_ieee14()
    nminus1_def = Nminus1Definition(
        monitored_elements=[
            GridElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[Contingency(id="BASECASE", elements=[])]
        + [
            Contingency(id=index, elements=[GridElement(id=index, name=row.name, kind="branch", type=row.type)])
            for index, row in net.get_branches().iterrows()
        ],
    )

    res = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def, job_id="test_job", timestep=0, method="dc"
    )
    n_0, n_1, success = extract_solver_matrices(
        loadflow_results=res,
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
    assert np.all(success)
    contingencies = [contingency for contingency in nminus1_def.contingencies if not contingency.is_basecase()]
    nminus1_def.contingencies = contingencies
    with pytest.raises(AssertionError):
        extract_solver_matrices(
            loadflow_results=res,
            nminus1_definition=nminus1_def,
            timestep=0,
        )


def test_extract_solver_matrices_disconnected():
    net = pypowsybl.network.create_ieee14()
    net.disconnect(net.get_branches().index[0])
    nminus1_def = Nminus1Definition(
        monitored_elements=[
            GridElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[Contingency(id="BASECASE", elements=[])]
        + [
            Contingency(id=index, elements=[GridElement(id=index, name=row.name, kind="branch", type=row.type)])
            for index, row in net.get_branches().iterrows()
        ],
    )

    res = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def, job_id="test_job", timestep=0, method="dc"
    )
    n_0, n_1, success = extract_solver_matrices(
        loadflow_results=res,
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
    assert success[0].item() is True, "Outages that are disconnected should be considered successful"
    assert np.all(success[1:])
    assert n_0[0] == 0.0  # Check that the disconnected branch has a loading of 0.0 in the base case
    assert np.all(n_1[:, 0] == 0.0)  # Check that the first monitored branch has a loading of 0.0 in all contingencies
    assert np.all(n_1[0, :] == 0.0)  # Check that the first contingency has a loading of 0.0 in all monitored branches


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


def test_concatenate_loadflow_results():
    net = pypowsybl.network.create_ieee14()
    nminus1_def_1 = Nminus1Definition(
        monitored_elements=[
            GridElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[Contingency(id="BASECASE", elements=[])],
    )

    nminus1_def_2 = Nminus1Definition(
        monitored_elements=[
            GridElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[
            Contingency(id=index, elements=[GridElement(id=index, name=row.name, kind="branch", type=row.type)])
            for index, row in net.get_branches().iterrows()
        ],
    )

    res_1 = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def_1, job_id="same_test_job", timestep=0, method="dc"
    )
    res_2 = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def_2, job_id="same_test_job", timestep=0, method="dc"
    )

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


class DummyBranchResult(pd.DataFrame):
    # Allow attribute access for DataFrame columns
    @property
    def _constructor(self):
        return DummyBranchResult


def make_branch_results(timestep, contingencies, branches, values=None):
    # Create a MultiIndex DataFrame for branch_results with "p" column
    idx = pd.MultiIndex.from_product(
        [[timestep], contingencies, branches, [BranchSide.ONE.value]], names=["timestep", "contingency", "element", "side"]
    )
    if values is None:
        p = np.arange(len(idx), dtype=float)
    else:
        p = np.array(values, dtype=float)
    df = DummyBranchResult({"p": p}, index=idx)
    return df


def test_convert_polars_loadflow_results_to_pandas():
    net = pypowsybl.network.create_ieee14()
    nminus1_def_1 = Nminus1Definition(
        monitored_elements=[
            GridElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[Contingency(id="BASECASE", elements=[])],
    )

    nminus1_def_2 = Nminus1Definition(
        monitored_elements=[
            GridElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[
            Contingency(id=index, elements=[GridElement(id=index, name=row.name, kind="branch", type=row.type)])
            for index, row in net.get_branches().iterrows()
        ],
    )

    loadflow_data_polars = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def_1, job_id="same_test_job", timestep=0, method="dc", polars=True
    )
    loadflow_data_pandas = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def_1, job_id="same_test_job", timestep=0, method="dc", polars=False
    )

    loadflow_data_pandas_2 = convert_polars_loadflow_results_to_pandas(loadflow_data_polars)

    loadflow_data_polars_2 = convert_pandas_loadflow_results_to_polars(loadflow_data_pandas_2)

    kw_args_testing = {
        "check_row_order": False,
        "check_column_order": False,
        "check_dtypes": True,
        "check_exact": False,
        "abs_tol": 1e-9,
    }

    # this is for debugging purposes
    assert loadflow_data_polars.job_id == loadflow_data_polars_2.job_id
    assert_frame_equal(loadflow_data_polars.branch_results, loadflow_data_polars_2.branch_results, **kw_args_testing)
    assert_frame_equal(loadflow_data_polars.node_results, loadflow_data_polars_2.node_results, **kw_args_testing)
    assert_frame_equal(
        loadflow_data_polars.regulating_element_results,
        loadflow_data_polars_2.regulating_element_results,
        **kw_args_testing,
    )
    assert_frame_equal(loadflow_data_polars.va_diff_results, loadflow_data_polars_2.va_diff_results, **kw_args_testing)
    assert_frame_equal(loadflow_data_polars.converged, loadflow_data_polars_2.converged, **kw_args_testing)
    assert loadflow_data_polars.additional_information == loadflow_data_polars_2.additional_information
    assert loadflow_data_polars.warnings == loadflow_data_polars_2.warnings

    # this is the actual test
    assert loadflow_data_polars.__eq__(loadflow_data_polars_2)

    assert loadflow_data_pandas.__eq__(loadflow_data_pandas_2)
