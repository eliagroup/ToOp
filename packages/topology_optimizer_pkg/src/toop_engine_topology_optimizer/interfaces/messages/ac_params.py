"""The parameters for the AC optimizer.

On AC, some subtelties are different to the DC optimization such as that the optimization is not
batched, and the parameters are slightly different.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, PositiveInt, confloat, model_validator
from toop_engine_interfaces.messages.lf_service.loadflow_results import StoredLoadflowReference
from toop_engine_topology_optimizer.interfaces.messages.commons import DescriptorDef, FilterStrategy


class ACGAParameters(BaseModel):
    """Parameters for the AC genetic algorithm"""

    runtime_seconds: PositiveInt = 30
    """The maximum runtime of the AC optimization in seconds"""

    pull_prob: confloat(ge=0.0, le=1.0) = 0.9
    """The probability of pulling a strategy from the DC repertoire"""

    me_descriptors: tuple[DescriptorDef, ...] = (
        DescriptorDef(metric="split_subs", num_cells=2, range=(0, 5)),
        DescriptorDef(metric="switching_distance", num_cells=5, range=(0, 50)),
        DescriptorDef(metric="disconnected_branches", num_cells=2),
    )
    """The descriptors for the aggregated map elites repertoire."""

    reconnect_prob: confloat(ge=0.0, le=1.0) = 0.05
    """The probability of reconnecting a disconnected branch in a strategy"""

    close_coupler_prob: confloat(ge=0.0, le=1.0) = 0.05
    """The probability of closing an opened coupler in a strategy"""

    n_worst_contingencies: PositiveInt = 20
    """How many worst contingencies to consider for the initial metrics, i.e. the top k contingencies
    that are used to compute the initial metrics. This is used to compute the top_k_overloads_n_1"""

    seed: int = 42
    """The seed for the random number generator"""

    timestep_processes: PositiveInt = 1
    """How many processes to spawn for computing the timesteps in parallel"""

    runner_processes: PositiveInt = 1
    """How many processes to spawn for computing the N-1 cases in each timestep in parallel. Note
    that this multiplies with timestep_processes and you might run out of memory if you set both
    too high"""

    runner_batchsize: Optional[PositiveInt] = None
    """Whether to batch the N-1 definition into smaller chunks, might conserve memory"""

    filter_strategy: Optional[FilterStrategy] = None
    """The filter strategy to use for the optimization, used to filter out strategies
    based on the discriminator, median or dominator filter."""

    enable_ac_rejection: bool = True
    """Whether to enable the AC rejection, i.e. no messages will be sent to the results topic in case of non-acceptance."""

    reject_convergence_threshold: float = 1.0
    """The rejection threshold for the convergence rate, i.e. the split case must have at most the same amount of
    non converging loadflows as the unsplit case or it will be rejected."""

    reject_overload_threshold: float = 0.95
    """The rejection threshold for the overload energy improvement, i.e. the split case must have at least 5% lower
    overload energy than the unsplit case or it will be rejected."""

    reject_critical_branch_threshold: float = 1.1
    """The rejection threshold for the critical branches increase, i.e. the split case must have less than 10% more
    critical branches than the unsplit case or it will be rejected."""

    early_stop_validation: bool = True
    """Whether to enable early stopping during the optimization process."""

    early_stopping_non_convergence_percentage_threshold: float = 0.1
    """The threshold for the early stopping criterion, i.e. if the percentage of non-converging cases is greater than
    this value, the ac validation will be stopped early."""

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> ACOptimizerParameters:
        """Ensure that the probabilities sum to one"""
        if self.pull_prob + self.reconnect_prob + self.close_coupler_prob != 1.0:
            raise ValueError("The probabilities must sum to one")
        return self


class ACOptimizerParameters(BaseModel):
    """The set of parameters that are used in the AC optimizer only"""

    initial_loadflow: Optional[StoredLoadflowReference] = None
    """If an initial AC loadflow was computed before the start of the optimization run, this can
    be passed and will be used e.g. to compute double limits. It will be sent back through the
    initial topology push."""

    ga_config: ACGAParameters = ACGAParameters()
    """The genetic algorithm configuration"""
