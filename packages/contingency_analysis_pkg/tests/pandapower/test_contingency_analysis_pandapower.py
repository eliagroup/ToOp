import pandapower as pp
import pytest
from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_contingency_analysis.pandapower import get_full_nminus1_definition_pandapower


def test_run_ac_contingency_analysis_pandapower(pandapower_net: pp.pandapowerNet, init_ray) -> None:
    nminus1_definition = get_full_nminus1_definition_pandapower(pandapower_net)

    lf_result_sequential_polars = get_ac_loadflow_results(
        pandapower_net, nminus1_definition, job_id="test_job", n_processes=1
    )
    assert lf_result_sequential_polars is not None


@pytest.mark.timeout(300)
@pytest.mark.skip(reason="Does not work on CI")
def test_run_ac_contingency_analysis_pandapower_mt(pandapower_net: pp.pandapowerNet, init_ray) -> None:
    nminus1_definition = get_full_nminus1_definition_pandapower(pandapower_net)

    lf_result_parallel_polars = get_ac_loadflow_results(pandapower_net, nminus1_definition, job_id="test_job", n_processes=2)
    assert lf_result_parallel_polars is not None
