import numpy as np
import pypowsybl
import pytest
from polars.testing import assert_frame_equal
from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_contingency_analysis.pypowsybl import (
    get_full_nminus1_definition_powsybl,
    run_powsybl_analysis,
    translate_nminus1_for_powsybl,
)
from toop_engine_contingency_analysis.pypowsybl.contingency_analysis_powsybl import run_contingency_analysis_powsybl
from toop_engine_interfaces.loadflow_result_helpers import (
    convert_pandas_loadflow_results_to_polars,
    convert_polars_loadflow_results_to_pandas,
    extract_branch_results,
    extract_solver_matrices,
)
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, Nminus1Definition


def test_run_powsybl_analysis(powsybl_bus_breaker_net: pypowsybl.network.Network) -> None:
    nminus1_definition = get_full_nminus1_definition_powsybl(powsybl_bus_breaker_net)

    pow_n1_def = translate_nminus1_for_powsybl(nminus1_definition, powsybl_bus_breaker_net)
    result, basecase_name = run_powsybl_analysis(powsybl_bus_breaker_net, pow_n1_def, "dc")
    assert all(result.branch_results["q1"].isna())
    assert all(result.bus_results["v_mag"].isna())
    assert result is not None
    assert basecase_name == "BASECASE"

    result, basecase_name = run_powsybl_analysis(powsybl_bus_breaker_net, pow_n1_def, "ac")
    assert not all(result.branch_results["q1"].isna())
    assert not all(result.bus_results["v_mag"].isna())
    assert result is not None
    assert basecase_name == "BASECASE"


@pytest.mark.parametrize("powsybl_net", ["powsybl_bus_breaker_net", "powsybl_node_breaker_net"])
def test_run_ac_contingency_analysis_powsybl(powsybl_net: str, request, init_ray) -> None:
    net = request.getfixturevalue(powsybl_net)
    nminus1_definition = get_full_nminus1_definition_powsybl(net)

    lf_result_sequential_polars = get_ac_loadflow_results(net, nminus1_definition, job_id="test_job", n_processes=1)
    assert len(lf_result_sequential_polars.branch_results.collect()) > 0
    assert len(lf_result_sequential_polars.node_results.collect()) > 0
    assert len(lf_result_sequential_polars.va_diff_results.collect()) > 0
    assert len(lf_result_sequential_polars.regulating_element_results.collect()) > 0
    assert len(lf_result_sequential_polars.converged.collect()) > 0
    # Run the loadflow in parallel
    lf_result_parallel_polars = get_ac_loadflow_results(net, nminus1_definition, job_id="test_job", n_processes=2)

    assert lf_result_sequential_polars is not None
    assert lf_result_parallel_polars is not None

    assert len(lf_result_sequential_polars.branch_results.collect()) == len(
        lf_result_parallel_polars.branch_results.collect()
    )
    assert len(lf_result_sequential_polars.node_results.collect()) == len(lf_result_parallel_polars.node_results.collect())
    assert len(lf_result_sequential_polars.va_diff_results.collect()) == len(
        lf_result_parallel_polars.va_diff_results.collect()
    )
    assert len(lf_result_sequential_polars.regulating_element_results.collect()) == len(
        lf_result_parallel_polars.regulating_element_results.collect()
    )
    assert len(lf_result_sequential_polars.converged.collect()) == len(lf_result_parallel_polars.converged.collect())


@pytest.mark.parametrize("powsybl_net", ["powsybl_bus_breaker_net", "powsybl_node_breaker_net"])
def test_contingency_analysis_ray_vs_powsybl(powsybl_net: str, request):
    net = request.getfixturevalue(powsybl_net)
    nminus1_definition = get_full_nminus1_definition_powsybl(net)

    lf_parallel_ray_polars = get_ac_loadflow_results(
        net, nminus1_definition, job_id="test_job", n_processes=2, batch_size=10
    )

    lf_parallel_native_polars = get_ac_loadflow_results(
        net,
        nminus1_definition,
        job_id="test_job",
        timestep=0,
        n_processes=2,
    )
    assert lf_parallel_ray_polars == lf_parallel_native_polars


def test_run_contingency_analysis_powsybl_not_converging_basecase() -> None:
    net = pypowsybl.network.create_ieee14()
    loads = net.get_loads(attributes=["q0"])
    # make the ac loadflow fail
    loads["q0"] = loads["q0"] * 10
    net.update_loads(loads)

    nminus1_def_1 = Nminus1Definition(
        monitored_elements=[
            GridElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[Contingency(id="BASECASE", elements=[])],
    )

    result = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def_1, job_id="same_test_job", timestep=0, method="ac", polars=True
    )
    assert len(result.converged.collect()) == 1
    assert result.converged.collect()["contingency"][0] == "BASECASE"
    assert result.converged.collect()["status"][0] == "MAX_ITERATION_REACHED"


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
