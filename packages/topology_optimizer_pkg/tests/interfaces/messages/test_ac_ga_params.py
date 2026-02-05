# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACGAParameters
from toop_engine_topology_optimizer.interfaces.messages.commons import DescriptorDef, FilterStrategy


def test_acga_parameters_default():
    params = ACGAParameters()
    assert params.runtime_seconds == 30
    assert params.pull_prob == 0.9
    assert params.reconnect_prob == 0.05
    assert params.close_coupler_prob == 0.05
    assert params.n_worst_contingencies == 20
    assert params.seed == 42
    assert params.timestep_processes == 1
    assert params.runner_processes == 1
    assert params.runner_batchsize is None
    assert params.filter_strategy is None
    assert params.enable_ac_rejection is True
    assert params.reject_convergence_threshold == 1.0
    assert params.reject_overload_threshold == 0.95
    assert params.reject_critical_branch_threshold == 1.1
    # Probabilities sum to one
    assert params.pull_prob + params.reconnect_prob + params.close_coupler_prob == 1.0


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
        timestep_processes=2,
        runner_processes=3,
        runner_batchsize=10,
        enable_ac_rejection=False,
        reject_convergence_threshold=0.6,
        reject_overload_threshold=0.9,
        reject_critical_branch_threshold=0.8,
        me_descriptors=(
            DescriptorDef(metric="split_subs", num_cells=2, range=(0, 5)),
            DescriptorDef(metric="switching_distance", num_cells=5, range=(0, 50)),
            DescriptorDef(metric="disconnected_branches", num_cells=2),
        ),
        filter_strategy=filter_strat,
    )
    assert params.pull_prob + params.reconnect_prob + params.close_coupler_prob == 1.0
    assert params.runtime_seconds == 60
    assert params.me_descriptors[0].metric == "split_subs"
    model_dump = params.model_dump_json()

    model_loaded = ACGAParameters.model_validate_json(model_dump)
    assert model_loaded == params
