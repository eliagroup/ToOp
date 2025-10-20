import pypowsybl
import pytest
from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_contingency_analysis.pypowsybl import (
    get_full_nminus1_definition_powsybl,
    run_powsybl_analysis,
    translate_nminus1_for_powsybl,
)


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
