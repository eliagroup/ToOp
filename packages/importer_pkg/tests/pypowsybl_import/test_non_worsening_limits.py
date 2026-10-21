# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pypowsybl
from pypowsybl.network.impl.network import Network
from toop_engine_contingency_analysis.pypowsybl import run_contingency_analysis_powsybl
from toop_engine_contingency_analysis.pypowsybl.powsybl_helpers import get_full_nminus1_definition_powsybl
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK
from toop_engine_importer.pypowsybl_import.preprocessing import (
    apply_artificial_dc_limits,
    apply_artificial_limits,
    compute_initial_metrics,
)
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.messages.preprocess.preprocess_commands import DoubleLimits
from toop_engine_interfaces.types import MetricType


def _get_ac_and_dc_metrics(
    network: Network, non_worsening: bool, double_limits: DoubleLimits = DoubleLimits()
) -> tuple[dict[MetricType, float], dict[MetricType, float]]:
    pypowsybl.loadflow.run_ac(network, CGMES_DISTRIBUTED_SLACK)
    full_definition = get_full_nminus1_definition_powsybl(network)
    nminus1_definition = full_definition.model_copy(
        update={
            "monitored_elements": [
                element.model_copy(
                    update={
                        "optimized": not non_worsening,
                        "non_worsening": non_worsening,
                    }
                )
                for element in full_definition.monitored_elements
                if element.kind == "branch"
            ]
        }
    )

    ac_results = run_contingency_analysis_powsybl(
        net=network,
        n_minus_1_definition=nminus1_definition,
        job_id="test",
        timestep=0,
        method="ac",
        polars=True,
        lf_params=CGMES_DISTRIBUTED_SLACK,
    )
    assert isinstance(ac_results, LoadflowResultsPolars)
    ac_loadflow = apply_artificial_limits(network, nminus1_definition, ac_results, double_limits)
    ac_metrics = compute_initial_metrics(ac_loadflow, nminus1_definition)
    dc_metrics = apply_artificial_dc_limits(
        network=network,
        nminus1_definition=nminus1_definition,
        double_limits=double_limits,
        lf_params=CGMES_DISTRIBUTED_SLACK,
        job_id="test",
        n_processes=1,
        batch_size=None,
    )
    return ac_metrics, dc_metrics


def test_all_non_worsening_has_no_critical_branches_in_ac_and_dc(complex_grid_network: Network) -> None:
    ac_metrics, dc_metrics = _get_ac_and_dc_metrics(
        complex_grid_network, non_worsening=True, double_limits=DoubleLimits(lower=0.4, upper=0.4)
    )

    for metrics in (ac_metrics, dc_metrics):
        for case in ("n_0", "n_1"):
            assert metrics[f"critical_branch_count_artificial_limit_{case}"] == 0


def test_without_non_worsening_branches_are_overloaded_in_ac_and_dc(complex_grid_network: Network) -> None:
    ac_metrics, dc_metrics = _get_ac_and_dc_metrics(
        complex_grid_network, non_worsening=False, double_limits=DoubleLimits(lower=0.4, upper=0.4)
    )

    for metrics in (ac_metrics, dc_metrics):
        assert metrics.get("critical_branch_count_artificial_limit_n_1", 0) > 0
        assert metrics.get("critical_branch_count_artificial_limit_n_0", 0) > 0
