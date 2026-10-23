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
import pytest
from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_contingency_analysis.compute_metrics import (
    _compute_branch_limit_metrics,
    compute_max_load,
    compute_metrics,
    compute_overload_energy,
    count_critical_branches,
    count_critical_va_diff_cases,
    count_voltage_jumps,
    get_worst_k_contingencies_ac,
)
from toop_engine_contingency_analysis.pandapower import get_full_nminus1_definition_pandapower
from toop_engine_contingency_analysis.pypowsybl import (
    get_full_nminus1_definition_powsybl,
    run_contingency_analysis_powsybl,
)
from toop_engine_grid_helpers.powsybl.loadflow_parameters import SINGLE_SLACK
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
    assert "max_flow_permanent_n_0" in metrics
    assert "overload_energy_permanent_n_0" in metrics
    assert "max_flow_permanent_n_1" in metrics
    assert "overload_energy_permanent_n_1" in metrics
    # assert "max_va_diff_n_0" in metrics
    # assert "max_va_diff_n_1" in metrics
    assert "critical_branch_count_permanent_n_1" in metrics
    assert "critical_branch_count_permanent_n_0" in metrics
    assert "critical_va_diff_count_n_0" in metrics
    assert "critical_va_diff_count_n_1" in metrics
    assert "voltage_jump_count_n_1" in metrics

    assert metrics["max_flow_permanent_n_0"] <= metrics["max_flow_permanent_n_1"]
    assert metrics["overload_energy_permanent_n_0"] <= metrics["overload_energy_permanent_n_1"]
    # assert metrics["max_va_diff_n_0"] <= metrics["max_va_diff_n_1"]

    metrics_2 = compute_metrics(lf_res)
    assert "max_flow_permanent_n_1" in metrics_2
    assert "overload_energy_permanent_n_1" in metrics_2
    # assert "max_va_diff_n_1" in metrics_2
    assert "critical_va_diff_count_n_1" in metrics_2
    assert "voltage_jump_count_n_1" in metrics_2
    assert np.isclose(metrics_2["max_flow_permanent_n_1"], metrics["max_flow_permanent_n_1"])
    # assert np.isclose(metrics_2["max_va_diff_n_1"], metrics["max_va_diff_n_1"])
    assert np.isclose(metrics_2["overload_energy_permanent_n_1"], metrics["overload_energy_permanent_n_1"])
    assert "max_flow_permanent_n_0" not in metrics_2


def test_compute_metrics_max_flow_displayed_permanent(pandapower_net: pandapower.pandapowerNet) -> None:
    nminus1_definition = get_full_nminus1_definition_pandapower(pandapower_net)
    base_case = nminus1_definition.base_case
    assert base_case is not None
    lf_res = get_ac_loadflow_results(
        net=pandapower_net, n_minus_1_definition=nminus1_definition, job_id="test_job", timestep=0
    )
    branch_ids = [element.id for element in nminus1_definition.monitored_elements if element.kind == "branch"]

    metrics = compute_metrics(lf_res, base_case_id=base_case.id)
    assert "max_flow_displayed_permanent_n_1" not in metrics

    all_displayed = compute_metrics(lf_res, base_case_id=base_case.id, displayed_branch_ids=branch_ids)
    for family in ("max_flow", "overload_energy", "overload_current"):
        for level in ("n_0", "n_1"):
            assert all_displayed[f"{family}_displayed_permanent_{level}"] == pytest.approx(
                metrics[f"{family}_permanent_{level}"]
            )

    one_displayed = compute_metrics(lf_res, base_case_id=base_case.id, displayed_branch_ids=branch_ids[:1])
    assert one_displayed["max_flow_displayed_permanent_n_1"] <= metrics["max_flow_permanent_n_1"]
    assert one_displayed["overload_energy_displayed_permanent_n_1"] <= metrics["overload_energy_permanent_n_1"]
    assert one_displayed["overload_current_displayed_permanent_n_1"] <= metrics["overload_current_permanent_n_1"]

    none_displayed = compute_metrics(lf_res, base_case_id=base_case.id, displayed_branch_ids=[])
    for family in ("max_flow", "overload_energy", "overload_current"):
        assert none_displayed[f"{family}_displayed_permanent_n_1"] is None


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
    assert "max_flow_artificial_limit_n_0" in metrics
    assert "overload_energy_artificial_limit_n_0" in metrics
    assert "max_flow_artificial_limit_n_1" in metrics
    assert "overload_energy_artificial_limit_n_1" in metrics
    assert "max_va_diff_n_0" in metrics
    assert "max_va_diff_n_1" in metrics
    assert "critical_branch_count_artificial_limit_n_1" in metrics
    assert "critical_branch_count_artificial_limit_n_0" in metrics
    assert "critical_va_diff_count_n_0" in metrics
    assert "critical_va_diff_count_n_1" in metrics
    assert "voltage_jump_count_n_1" in metrics

    assert metrics["max_flow_artificial_limit_n_0"] <= metrics["max_flow_artificial_limit_n_1"]
    assert metrics["overload_energy_artificial_limit_n_0"] <= metrics["overload_energy_artificial_limit_n_1"]
    assert metrics["max_va_diff_n_0"] <= metrics["max_va_diff_n_1"]

    metrics_2 = compute_metrics(lf_res)
    assert "max_flow_artificial_limit_n_1" in metrics_2
    assert "overload_energy_artificial_limit_n_1" in metrics_2
    assert "max_va_diff_n_1" in metrics_2
    assert "critical_va_diff_count_n_1" in metrics_2
    assert "voltage_jump_count_n_1" in metrics_2
    assert np.isclose(metrics_2["max_flow_artificial_limit_n_1"], metrics["max_flow_artificial_limit_n_1"])
    assert np.isclose(metrics_2["max_va_diff_n_1"], metrics["max_va_diff_n_1"])
    assert np.isclose(metrics_2["overload_energy_artificial_limit_n_1"], metrics["overload_energy_artificial_limit_n_1"])
    assert "max_flow_artificial_limit_n_0" not in metrics_2


@pytest.mark.parametrize("family", ["max_flow", "overload_energy", "overload_current", "critical_branch_count"])
def test_compute_metrics_limit_variants_powsybl(powsybl_bus_breaker_net: pypowsybl.network.Network, family: str) -> None:
    nminus1_definition = get_full_nminus1_definition_powsybl(powsybl_bus_breaker_net)
    lf_res = run_contingency_analysis_powsybl(
        net=powsybl_bus_breaker_net,
        n_minus_1_definition=nminus1_definition,
        job_id="test_job",
        timestep=0,
        method="ac",
        polars=True,
    )
    metrics = compute_metrics(lf_res, base_case_id="BASECASE")

    for case in ["n_0", "n_1"]:
        assert metrics[f"{family}_permanent_{case}"] is not None
        assert metrics[f"{family}_artificial_limit_{case}"] is not None

    # The example grid has artificial limits equal to the permanent limit. The N-1 limit of twice the permanent limit is
    # not used by the permanent metrics.
    if family == "max_flow":
        assert metrics["max_flow_permanent_n_1"] == pytest.approx(metrics["max_flow_artificial_limit_n_1"])
        assert metrics["max_flow_permanent_n_0"] == pytest.approx(metrics["max_flow_artificial_limit_n_0"])


def test_branch_limits_powsybl(powsybl_bus_breaker_net: pypowsybl.network.Network) -> None:
    nminus1_definition = get_full_nminus1_definition_powsybl(powsybl_bus_breaker_net)
    lf_res = run_contingency_analysis_powsybl(
        net=powsybl_bus_breaker_net,
        n_minus_1_definition=nminus1_definition,
        job_id="test_job",
        timestep=0,
        method="ac",
        polars=True,
    )
    limits = lf_res.branch_limits.collect()

    assert set(limits["limit_name"].unique()) == {
        "permanent_limit",
        "N-1",
        "artificial_limit_n_0",
        "artificial_limit_n_1",
    }
    # The limits are stored once per branch side and limit name, not per result row
    assert limits.height == limits.unique(subset=["timestep", "element", "side", "limit_name"]).height


def test_branch_loading_matches_powsybl_limit_violations(powsybl_bus_breaker_net: pypowsybl.network.Network) -> None:
    net = powsybl_bus_breaker_net
    nminus1_definition = get_full_nminus1_definition_powsybl(net)
    lf_res = run_contingency_analysis_powsybl(
        net=net,
        n_minus_1_definition=nminus1_definition,
        job_id="test_job",
        timestep=0,
        method="ac",
        polars=True,
        lf_params=SINGLE_SLACK,
    )
    analysis = pypowsybl.security.create_analysis()
    analysis.add_monitored_elements(branch_ids=net.get_branches().index.tolist())
    for branch_id in net.get_branches().index:
        analysis.add_single_element_contingency(branch_id, contingency_id=branch_id)
    violations = analysis.run_ac(net, parameters=SINGLE_SLACK).limit_violations.reset_index()
    violations = violations[violations["limit_type"] == "CURRENT"]
    assert len(violations) > 0, "The example grid should have limit violations after outages"

    # Powsybl calls the permanent limit "permanent" and reports the side as ONE/TWO
    violations = violations.assign(
        limit_name=violations["limit_name"].replace({"permanent": "permanent_limit"}),
        side=violations["side"].map({"ONE": 1, "TWO": 2}),
        contingency_id=violations["contingency_id"].replace({"": "BASECASE"}),
    )
    ours = (
        lf_res.branch_results.join(lf_res.branch_limits, on=["timestep", "element", "side"], how="inner")
        .with_columns(loading_of_limit=pl.col("i") / pl.col("limit"))
        .collect()
        .to_pandas()
        .set_index(["contingency", "element", "side", "limit_name"])
    )
    for violation in violations.itertuples():
        row = ours.loc[(violation.contingency_id, violation.subject_id, violation.side, violation.limit_name)]
        assert row["loading_of_limit"] > 1.0
        assert row["i"] == pytest.approx(violation.value, rel=1e-6)
        assert row["limit"] == pytest.approx(violation.limit, rel=1e-6)


def test_compute_metrics_limit_columns_missing(pandapower_net: pandapower.pandapowerNet) -> None:
    nminus1_definition = get_full_nminus1_definition_pandapower(pandapower_net)
    lf_res = get_ac_loadflow_results(
        net=pandapower_net, n_minus_1_definition=nminus1_definition, job_id="test_job", timestep=0
    )
    metrics = compute_metrics(lf_res, base_case_id="BASECASE")

    assert metrics["max_flow_permanent_n_1"] > 0
    assert metrics["max_flow_artificial_limit_n_1"] is None
    assert metrics["overload_energy_artificial_limit_n_0"] is None


def test_get_worst_k_contingencies_basic(branch_results_df_fast_failing_polars, convergence_results_df_fast_failing_polars):
    df = branch_results_df_fast_failing_polars
    n_non_converging = (
        convergence_results_df_fast_failing_polars.filter((pl.col("status") != "CONVERGED") & (pl.col("timestep") == 0))
        .collect()
        .shape[0]
    )
    k = 2
    contingencies, overloads = get_worst_k_contingencies_ac(df, convergence_results_df_fast_failing_polars, k=k, field="p")
    # Should return 2 contingencies per timestep, and overloads per timestep
    assert (
        len(contingencies) == 2
        and len(contingencies[0]) == k + n_non_converging
        and len(contingencies[1]) == k + n_non_converging
    )
    assert len(overloads) == 2
    # For timestep 0, cont2 and cont3 should have higher overloads than cont1
    assert "cont2" in contingencies[0] and "cont3" in contingencies[0]
    # For timestep 1, cont2 and cont3 should have higher overloads than cont1
    assert "cont2" in contingencies[1] and "cont3" in contingencies[1]
    # Overloads should be positive and in decreasing order
    assert overloads[0] >= 0 and overloads[1] >= 0
    assert "cont4" in contingencies[0] and "cont4" in contingencies[1]  # Non-converging contingency should be included


def test_get_worst_k_contingencies_ac_returns_expected_shape(
    branch_results_df_fast_failing_polars, convergence_results_df_fast_failing_polars
):
    df = branch_results_df_fast_failing_polars
    contingencies, overloads = get_worst_k_contingencies_ac(df, convergence_results_df_fast_failing_polars, k=2, field="p")
    # There are 2 timesteps in the test data
    assert isinstance(contingencies, list)
    assert isinstance(overloads, list)
    assert len(contingencies) == 2
    assert len(overloads) == 2
    # Each timestep should have k contingencies
    assert all(
        len(c)
        == 2
        + convergence_results_df_fast_failing_polars.filter((pl.col("status") != "CONVERGED") & (pl.col("timestep") == i))
        .collect()
        .shape[0]
        for i, c in enumerate(contingencies)
    )


def test_get_worst_k_contingencies_ac_k_greater_than_available(
    branch_results_df_fast_failing_polars, convergence_results_df_fast_failing_polars
):
    df = branch_results_df_fast_failing_polars
    # There are 3 contingencies per timestep, so k=5 should return all
    contingencies, overloads = get_worst_k_contingencies_ac(df, convergence_results_df_fast_failing_polars, k=5, field="p")

    overload_n_minus_1 = compute_overload_energy(df, field="p")
    assert all(len(c) == 4 for c in contingencies)
    for o in overloads:
        assert overload_n_minus_1 == o


def test_get_worst_k_contingencies_ac_can_disable_non_converging_append(
    branch_results_df_fast_failing_polars, convergence_results_df_fast_failing_polars
):
    df = branch_results_df_fast_failing_polars

    contingencies_with_non_converging, _ = get_worst_k_contingencies_ac(
        df,
        convergence_results_df_fast_failing_polars,
        k=2,
        field="p",
        include_non_converging_loadflows=True,
    )
    contingencies_without_non_converging, _ = get_worst_k_contingencies_ac(
        df,
        convergence_results_df_fast_failing_polars,
        k=2,
        field="p",
        include_non_converging_loadflows=False,
    )

    assert all("cont4" in contingencies for contingencies in contingencies_with_non_converging)
    assert all("cont4" not in contingencies for contingencies in contingencies_without_non_converging)
    assert all(len(contingencies) == 2 for contingencies in contingencies_without_non_converging)


def test_compute_metrics_excludes_basecase_from_n_1_when_base_case_id_is_given(
    branch_results_df_fast_failing_polars: pl.LazyFrame,
    convergence_results_df_fast_failing_polars: pl.LazyFrame,
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
            "timestep": [0, 0, 0, 0, 1],
            "contingency": ["BASECASE", "cont1", "cont1", "cont2", "cont3"],
            "element": ["branch1", "branch1", "branch2", "branch1", "branch1"],
            "va_diff": [9.0, 1.0, 7.0, 4.0, 8.0],
        }
    ).lazy()
    node_results = pl.DataFrame(
        {
            "timestep": [0, 0, 0, 0],
            "contingency": ["BASECASE", "BASECASE", "cont1", "cont1"],
            "element": ["node1", "node2", "node1", "node2"],
            "vm": [100.0, 100.0, 105.0, 105.1],
            "vm_basecase_deviation": [0.0, 0.0, 5.0, 5.1],
        }
    ).lazy()

    loadflow_results = LoadflowResultsPolars(
        job_id="test_job",
        branch_results=branch_results,
        va_diff_results=va_diff_results,
        node_results=node_results,
        regulating_element_results=pl.LazyFrame(),
        converged=convergence_results_df_fast_failing_polars,
    )

    metrics_without_basecase_filter = compute_metrics(loadflow_results)
    metrics_with_basecase_filter = compute_metrics(
        loadflow_results,
        base_case_id="BASECASE",
        critical_va_diff_threshold=5.0,
        critical_voltage_jump_threshold=5.0,
    )
    n_1_only_branch_results = branch_results.filter(pl.col("contingency") != "BASECASE")

    expected_overload_energy_n_1 = compute_overload_energy(n_1_only_branch_results, field="p")
    expected_overload_current_n_1 = compute_overload_energy(n_1_only_branch_results, field="i")
    expected_max_flow_n_1 = compute_max_load(n_1_only_branch_results)
    expected_critical_branch_count_n_1 = count_critical_branches(n_1_only_branch_results)
    expected_voltage_jump_count_n_1 = count_voltage_jumps(
        node_results,
        critical_voltage_jump_threshold=5.0,
    )
    expected_critical_va_diff_count_n_1 = count_critical_va_diff_cases(
        va_diff_results.filter(pl.col("contingency") != "BASECASE"), critical_threshold=5.0
    )

    assert np.isclose(metrics_with_basecase_filter["overload_energy_permanent_n_1"], expected_overload_energy_n_1)
    assert np.isclose(metrics_with_basecase_filter["overload_current_permanent_n_1"], expected_overload_current_n_1)
    assert np.isclose(metrics_with_basecase_filter["max_flow_permanent_n_1"], expected_max_flow_n_1)
    assert metrics_with_basecase_filter["critical_branch_count_permanent_n_1"] == expected_critical_branch_count_n_1
    assert metrics_with_basecase_filter["voltage_jump_count_n_1"] == expected_voltage_jump_count_n_1
    assert metrics_with_basecase_filter["critical_va_diff_count_n_1"] == 2
    assert metrics_with_basecase_filter["critical_va_diff_count_n_1"] == expected_critical_va_diff_count_n_1
    assert (
        metrics_without_basecase_filter["overload_energy_permanent_n_1"]
        > metrics_with_basecase_filter["overload_energy_permanent_n_1"]
    )
    expected_basecase_power_overload_per_branch = basecase_power - abs(basecase_power / basecase_loading_factor)
    assert np.isclose(
        metrics_with_basecase_filter["overload_energy_permanent_n_0"], 2 * expected_basecase_power_overload_per_branch
    )
    assert np.isclose(metrics_with_basecase_filter["max_va_diff_n_1"], 8.0)
    assert metrics_with_basecase_filter["critical_va_diff_count_n_0"] == 1
    assert metrics_with_basecase_filter["voltage_jump_count_n_1"] == 1


def test_count_voltage_jumps_returns_none_without_node_deviation_column() -> None:
    assert count_voltage_jumps(pl.LazyFrame()) is None


def test_count_critical_va_diff_cases_uses_configurable_threshold() -> None:
    va_diff_results = pl.DataFrame(
        {
            "timestep": [0, 0, 0, 1, 1],
            "contingency": ["cont1", "cont1", "cont2", "cont1", "cont2"],
            "element": ["branch1", "branch2", "branch1", "branch1", "branch1"],
            "va_diff": [-4.0, -6.0, 5.0, -9.0, 3.0],
        }
    ).lazy()

    assert count_critical_va_diff_cases(va_diff_results, critical_threshold=5.0) == 2
    assert count_critical_va_diff_cases(va_diff_results, critical_threshold=8.0) == 1


def test_compute_metrics_passes_configurable_va_diff_threshold(
    branch_results_df_fast_failing_polars: pl.LazyFrame,
) -> None:
    basecase_branch_results = (
        branch_results_df_fast_failing_polars.filter(pl.col("timestep") == 0)
        .filter(pl.col("contingency") == "cont1")
        .limit(1)
        .with_columns(contingency=pl.lit("BASECASE"))
    )
    loadflow_results = LoadflowResultsPolars(
        job_id="test_job",
        branch_results=pl.concat([branch_results_df_fast_failing_polars, basecase_branch_results], how="vertical"),
        va_diff_results=pl.DataFrame(
            {
                "timestep": [0, 0, 0],
                "contingency": ["BASECASE", "cont1", "cont2"],
                "element": ["branch1", "branch1", "branch1"],
                "va_diff": [9.0, 6.0, 4.0],
            }
        ).lazy(),
        node_results=pl.LazyFrame(),
        regulating_element_results=pl.LazyFrame(),
        converged=pl.LazyFrame(),
    )

    metrics = compute_metrics(loadflow_results, base_case_id="BASECASE", critical_va_diff_threshold=5.0)

    assert metrics["critical_va_diff_count_n_0"] == 1
    assert metrics["critical_va_diff_count_n_1"] == 1


def test_worst_k_overload_matches_full_run_for_artificial_limit() -> None:
    """The top-k overload of all overloaded cases must equal the full-run overload of the same limit scope."""
    permanent_limit, artificial_limit = 100.0, 50.0
    # (contingency, element, current): C1 and C3 only exceed the artificial limit, C2 exceeds both, C4 neither
    rows = [("C1", "L1", 80.0), ("C2", "L1", 120.0), ("C3", "L2", 60.0), ("C4", "L2", 10.0)]
    branch_results = pl.LazyFrame(
        {
            "timestep": [0] * len(rows),
            "contingency": [row[0] for row in rows],
            "element": [row[1] for row in rows],
            "side": [1] * len(rows),
            "i": [row[2] for row in rows],
            "p": [row[2] for row in rows],
            "q": [0.0] * len(rows),
            "loading": [row[2] / permanent_limit for row in rows],
            "element_name": [""] * len(rows),
            "contingency_name": [""] * len(rows),
        }
    )
    branch_limits = pl.LazyFrame(
        {
            "timestep": [0] * 4,
            "element": ["L1", "L2"] * 2,
            "side": [1] * 4,
            "limit_name": ["permanent_limit"] * 2 + ["artificial_limit_n_1"] * 2,
            "limit": [permanent_limit] * 2 + [artificial_limit] * 2,
        }
    )
    converged = pl.LazyFrame(
        {
            "timestep": [0] * 4,
            "contingency": [row[0] for row in rows],
            "status": ["CONVERGED"] * 4,
            "iteration_count": [1.0] * 4,
            "warnings": [""] * 4,
            "contingency_name": [""] * 4,
        }
    )

    full_run = _compute_branch_limit_metrics(branch_results, branch_limits, "n_1")
    overloaded_cases = ["C1", "C2", "C3"]
    worst_k_cases, worst_k_overload = get_worst_k_contingencies_ac(
        branch_results,
        converged,
        k=len(overloaded_cases),
        field="p",
        branch_limits=branch_limits,
        limit_scope="artificial_limit",
    )

    assert sorted(worst_k_cases[0]) == overloaded_cases
    assert worst_k_overload[0] == pytest.approx(full_run["overload_energy_artificial_limit_n_1"])

    permanent_cases, permanent_overload = get_worst_k_contingencies_ac(
        branch_results, converged, k=len(overloaded_cases), field="p", branch_limits=branch_limits, limit_scope="permanent"
    )
    assert permanent_cases[0] == ["C2"]
    assert permanent_overload[0] == pytest.approx(full_run["overload_energy_permanent_n_1"])
