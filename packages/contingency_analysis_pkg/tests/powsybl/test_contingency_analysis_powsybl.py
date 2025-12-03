import pypowsybl
import pytest
from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_contingency_analysis.pypowsybl import (
    get_full_nminus1_definition_powsybl,
    run_powsybl_analysis,
    translate_nminus1_for_powsybl,
)
from toop_engine_contingency_analysis.pypowsybl.contingency_analysis_powsybl import run_contingency_analysis_powsybl
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


def test_run_contingency_analysis_powsybl_converging_basecase() -> None:
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
