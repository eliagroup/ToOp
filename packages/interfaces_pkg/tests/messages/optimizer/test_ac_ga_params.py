from toop_engine_interfaces.messages.optimiser_ac_dc_commons_factory import (
    create_descriptor_def as DescriptorDef,
)
from toop_engine_interfaces.messages.optimiser_ac_dc_commons_factory import (
    create_filter_distance_set as FilterDistanceSet,
)
from toop_engine_interfaces.messages.optimiser_ac_dc_commons_factory import (
    create_filter_strategy as FilterStrategy,
)
from toop_engine_interfaces.messages.optimiser_ac_params_factory import create_ac_ga_parameters as ACGAParameters


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
    assert params.runner_batchsize == 0
    assert not params.HasField("filter_strategy")
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
            "split_subs": FilterDistanceSet(distances=[0.0]),
            "switching_distance": FilterDistanceSet(distances=[-0.9, 0.9]),
            "fitness": FilterDistanceSet(distances=[-60.0, 60.0]),
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
        me_descriptors=[
            DescriptorDef(metric="split_subs", num_cells=2, range=(0.0, 5.0)),
            DescriptorDef(metric="switching_distance", num_cells=5, range=(0.0, 50.0)),
            DescriptorDef(metric="disconnected_branches", num_cells=2),
        ],
        filter_strategy=filter_strat,
    )
    assert params.pull_prob + params.reconnect_prob + params.close_coupler_prob == 1.0
    assert params.runtime_seconds == 60
    assert params.me_descriptors[0].metric == "split_subs"
