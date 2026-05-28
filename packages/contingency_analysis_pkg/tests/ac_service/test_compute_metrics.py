# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pandapower
import polars as pl
import pypowsybl
from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_contingency_analysis.ac_loadflow_service.compute_metrics import (
    compute_max_load,
    compute_metrics,
    compute_overload_energy,
    count_critical_branches,
    get_worst_k_contingencies_ac,
)
from toop_engine_contingency_analysis.pandapower import get_full_nminus1_definition_pandapower
from toop_engine_contingency_analysis.pypowsybl import (
    get_full_nminus1_definition_powsybl,
    run_contingency_analysis_powsybl,
)
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars


def test_compute_metrics_pandapower(pandapower_net: pandapower.pandapowerNet):
    nminus1_definition = get_full_nminus1_definition_pandapower(pandapower_net)
    base_case = next((cont for cont in nminus1_definition.contingencies if cont.is_basecase()), None)
    assert base_case is not None, "Base case (N-0) not found in n-1 definition."

    lf_res = get_ac_loadflow_results(
        net=pandapower_net,
        n_minus_1_definition=nminus1_definition,
        job_id="test_job",
        timestep=0,
    )

    metrics = compute_metrics(lf_res, base_case_id=base_case.id)

    assert isinstance(metrics, dict)
    assert "max_flow_n_0" in metrics
    assert "overload_energy_n_0" in metrics
    assert "max_flow_n_1" in metrics
    assert "overload_energy_n_1" in metrics
    # assert "max_va_diff_n_0" in metrics
    # assert "max_va_diff_n_1" in metrics
    assert "overload_current_n_0" in metrics
    assert "overload_current_n_1" in metrics
    assert "critical_branch_count_n_1" in metrics
    assert "critical_branch_count_n_0" in metrics

    assert metrics["max_flow_n_0"] <= metrics["max_flow_n_1"]
    assert metrics["overload_energy_n_0"] <= metrics["overload_energy_n_1"]
    # assert metrics["max_va_diff_n_0"] <= metrics["max_va_diff_n_1"]
    assert metrics["overload_current_n_0"] <= metrics["overload_current_n_1"]

    metrics_2 = compute_metrics(lf_res)
    assert "max_flow_n_1" in metrics_2
    assert "overload_energy_n_1" in metrics_2
    # assert "max_va_diff_n_1" in metrics_2
    assert "overload_current_n_1" in metrics_2
    assert np.isclose(metrics_2["max_flow_n_1"], metrics["max_flow_n_1"])
    # assert np.isclose(metrics_2["max_va_diff_n_1"], metrics["max_va_diff_n_1"])
    assert np.isclose(metrics_2["overload_current_n_1"], metrics["overload_current_n_1"])
    assert np.isclose(metrics_2["overload_energy_n_1"], metrics["overload_energy_n_1"])
    assert "max_flow_n_0" not in metrics_2


def test_compute_metrics_powsybl(powsybl_bus_breaker_net: pypowsybl.network.Network) -> None:
    nminus1_definition = get_full_nminus1_definition_powsybl(powsybl_bus_breaker_net)

    base_case = next((cont for cont in nminus1_definition.contingencies if cont.is_basecase()), None)
    assert base_case is not None, "Base case (N-0) not found in n-1 definition."

    lf_res = run_contingency_analysis_powsybl(
        net=powsybl_bus_breaker_net,
        n_minus_1_definition=nminus1_definition,
        job_id="test_job",
        timestep=0,
        method="ac",
        polars=True,
    )
    metrics = compute_metrics(lf_res, base_case_id=base_case.id)

    assert isinstance(metrics, dict)
    assert "max_flow_n_0" in metrics
    assert "overload_energy_n_0" in metrics
    assert "max_flow_n_1" in metrics
    assert "overload_energy_n_1" in metrics
    assert "max_va_diff_n_0" in metrics
    assert "max_va_diff_n_1" in metrics
    assert "overload_current_n_0" in metrics
    assert "overload_current_n_1" in metrics
    assert "critical_branch_count_n_1" in metrics
    assert "critical_branch_count_n_0" in metrics

    assert metrics["max_flow_n_0"] <= metrics["max_flow_n_1"]
    assert metrics["overload_energy_n_0"] <= metrics["overload_energy_n_1"]
    assert metrics["max_va_diff_n_0"] <= metrics["max_va_diff_n_1"]
    assert metrics["overload_current_n_0"] <= metrics["overload_current_n_1"]

    metrics_2 = compute_metrics(lf_res)
    assert "max_flow_n_1" in metrics_2
    assert "overload_energy_n_1" in metrics_2
    assert "max_va_diff_n_1" in metrics_2
    assert "overload_current_n_1" in metrics_2
    assert np.isclose(metrics_2["max_flow_n_1"], metrics["max_flow_n_1"])
    assert np.isclose(metrics_2["max_va_diff_n_1"], metrics["max_va_diff_n_1"])
    assert np.isclose(metrics_2["overload_current_n_1"], metrics["overload_current_n_1"])
    assert np.isclose(metrics_2["overload_energy_n_1"], metrics["overload_energy_n_1"])
    assert "max_flow_n_0" not in metrics_2


def test_get_worst_k_contingencies_basic(branch_results_df_fast_failing_polars):
    df = branch_results_df_fast_failing_polars
    contingencies, overloads = get_worst_k_contingencies_ac(df, k=2, field="p")
    # Should return 2 contingencies per timestep, and overloads per timestep
    assert len(contingencies) == 2 and len(contingencies[0]) == 2
    assert len(overloads) == 2
    # For timestep 0, cont2 and cont3 should have higher overloads than cont1
    assert "cont2" in contingencies[0] and "cont3" in contingencies[0]
    # For timestep 1, cont2 and cont3 should have higher overloads than cont1
    assert "cont2" in contingencies[1] and "cont3" in contingencies[1]
    # Overloads should be positive and in decreasing order
    assert overloads[0] >= 0 and overloads[1] >= 0


def test_get_worst_k_contingencies_ac_returns_expected_shape(branch_results_df_fast_failing_polars):
    df = branch_results_df_fast_failing_polars
    contingencies, overloads = get_worst_k_contingencies_ac(df, k=2, field="p")
    # There are 2 timesteps in the test data
    assert isinstance(contingencies, list)
    assert isinstance(overloads, list)
    assert len(contingencies) == 2
    assert len(overloads) == 2
    # Each timestep should have k contingencies
    assert all(len(c) == 2 for c in contingencies)


def test_get_worst_k_contingencies_ac_k_greater_than_available(branch_results_df_fast_failing_polars):
    df = branch_results_df_fast_failing_polars
    # There are 3 contingencies per timestep, so k=5 should return all
    contingencies, overloads = get_worst_k_contingencies_ac(df, k=5, field="p")

    overload_n_minus_1 = compute_overload_energy(df, field="p")
    assert all(len(c) == 3 for c in contingencies)
    for o in overloads:
        assert overload_n_minus_1 == o


def test_compute_metrics_excludes_basecase_from_n_1_when_base_case_id_is_given(
    branch_results_df_fast_failing_polars: pl.LazyFrame,
) -> None:
    # Choose strongly overloaded basecase values so inclusion in N-1 metrics is easy to detect.
    basecase_power = 500.0
    basecase_current = 50.0
    basecase_loading_factor = 10.0
    basecase_branch_results = (
        branch_results_df_fast_failing_polars.filter(pl.col("timestep") == 0)
        .filter(pl.col("contingency") == "cont1")
        .limit(2)
        .with_columns(
            contingency=pl.lit("BASECASE"),
            p=pl.lit(basecase_power),
            i=pl.lit(basecase_current),
            loading=pl.lit(basecase_loading_factor),
        )
    )
    branch_results = pl.concat([branch_results_df_fast_failing_polars, basecase_branch_results], how="vertical")

    va_diff_results = pl.DataFrame(
        {
            "timestep": [0, 0],
            "contingency": ["BASECASE", "cont1"],
            "element": ["branch1", "branch1"],
            "va_diff": [9.0, 1.0],
        }
    ).lazy()

    loadflow_results = LoadflowResultsPolars(
        job_id="test_job",
        branch_results=branch_results,
        va_diff_results=va_diff_results,
        node_results=pl.LazyFrame(),
        regulating_element_results=pl.LazyFrame(),
        converged=pl.LazyFrame(),
    )

    metrics_without_basecase_filter = compute_metrics(loadflow_results)
    metrics_with_basecase_filter = compute_metrics(loadflow_results, base_case_id="BASECASE")
    n_1_only_branch_results = branch_results.filter(pl.col("contingency") != "BASECASE")

    expected_overload_energy_n_1 = compute_overload_energy(n_1_only_branch_results, field="p")
    expected_overload_current_n_1 = compute_overload_energy(n_1_only_branch_results, field="i")
    expected_max_flow_n_1 = compute_max_load(n_1_only_branch_results)
    expected_critical_branch_count_n_1 = count_critical_branches(n_1_only_branch_results)

    assert np.isclose(metrics_with_basecase_filter["overload_energy_n_1"], expected_overload_energy_n_1)
    assert np.isclose(metrics_with_basecase_filter["overload_current_n_1"], expected_overload_current_n_1)
    assert np.isclose(metrics_with_basecase_filter["max_flow_n_1"], expected_max_flow_n_1)
    assert metrics_with_basecase_filter["critical_branch_count_n_1"] == expected_critical_branch_count_n_1
    assert metrics_without_basecase_filter["overload_energy_n_1"] > metrics_with_basecase_filter["overload_energy_n_1"]
    expected_basecase_power_overload_per_branch = basecase_power - abs(basecase_power / basecase_loading_factor)
    assert np.isclose(metrics_with_basecase_filter["overload_energy_n_0"], 2 * expected_basecase_power_overload_per_branch)
    assert np.isclose(metrics_with_basecase_filter["max_va_diff_n_1"], 1.0)
