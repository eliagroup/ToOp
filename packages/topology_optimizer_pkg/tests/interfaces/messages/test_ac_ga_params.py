# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACGAParameters
from toop_engine_topology_optimizer.interfaces.messages.commons import FilterStrategy


def test_acga_parameters_default():
    params = ACGAParameters()
    assert params.runtime_seconds == 180
    assert params.n_worst_contingencies == 20
    assert params.include_non_converging_loadflows_in_worst_k is True
    assert params.seed == 42
    assert params.worst_k_runner_processes == 1
    assert params.runner_processes == 1
    assert params.runner_processes == 1
    assert params.filter_strategy is None
    assert params.enable_ac_rejection is True
    assert params.reject_convergence_threshold == 1.0
    assert params.reject_overload_threshold == 0.95
    assert params.reject_critical_branch_threshold == 1.1
    assert params.reject_voltage_jump_threshold == 1.1
    assert params.reject_critical_va_diff_threshold == 1.1
    assert params.enable_critical_voltage_rejection is False
    assert params.critical_voltage_jump_percent == 5.0
    assert params.critical_va_diff_degree == 20.0
    # Probabilities sum to one


def test_acga_parameters_filter_strategy():
    filter_strat = FilterStrategy(
        filter_dominator_metrics_target=["switching_distance", "split_subs"],
        filter_dominator_metrics_observed=["switching_distance", "split_subs"],
        filter_discriminator_metric_distances={
            "split_subs": {0.0},
            "switching_distance": {-0.9, 0.9},
            "fitness": {-60, 60},
        },
        filter_discriminator_metric_multiplier={"split_subs": 1.0},
        filter_median_metric=["split_subs"],
    )
    params = ACGAParameters(
        runtime_seconds=60,
        n_worst_contingencies=5,
        seed=123,
        worst_k_runner_processes=2,
        runner_processes=3,
        full_analysis_batchsize=10,
        enable_ac_rejection=False,
        reject_convergence_threshold=0.6,
        reject_overload_threshold=0.9,
        reject_critical_branch_threshold=0.8,
        reject_voltage_jump_threshold=0.7,
        reject_critical_va_diff_threshold=0.85,
        enable_critical_voltage_rejection=True,
        critical_voltage_jump_percent=7.5,
        critical_va_diff_degree=12.0,
        filter_strategy=filter_strat,
    )
    assert params.runtime_seconds == 60
    model_dump = params.model_dump_json()

    model_loaded = ACGAParameters.model_validate_json(model_dump)
    assert model_loaded == params


def test_acga_parameters_accept_legacy_runner_processes_input() -> None:
    params = ACGAParameters(runner_processes=4)

    assert params.runner_processes == 4
