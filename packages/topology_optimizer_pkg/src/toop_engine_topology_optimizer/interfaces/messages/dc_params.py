# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""The DC optimizer is the GPU stage where massive amounts of topologies are being checked.

This holds the parameters to start the optimization. Some parameters can not be changed (mainly
the names of the kafka streams) and are included in the command line start parameters instead.
"""

from __future__ import annotations

from typing import Optional

from pydantic import (
    BaseModel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    confloat,
)
from pydantic.functional_validators import model_validator
from toop_engine_interfaces.types import MetricType
from toop_engine_topology_optimizer.interfaces.messages.commons import DescriptorDef, DoubleLimitsSetpoint


class BatchedMEParameters(BaseModel):
    """Parameters for starting the batched genetic algorithm (In this case Map-Elites)"""

    substation_split_prob: confloat(ge=0.0, le=1.0) = 0.1
    """The probability to split an unsplit substation. If not split, a reconfiguration is applied"""

    substation_unsplit_prob: confloat(ge=0.0, le=1.0) = 0.01
    """The probability to reset a split substation to the unsplit state"""

    disconnect_prob: confloat(ge=0.0, le=1.0) = 0.1
    """The probability to disconnect a new branch"""

    reconnect_prob: confloat(ge=0.0, le=1.0) = 0.1
    """The probability to reconnect a disconnected branch, will overwrite a possible disconnect"""

    pst_mutation_sigma: NonNegativeFloat = 0.0
    """The sigma to use for the normal distribution that mutates the PST taps. The mutation is applied by adding a random
    value drawn from this distribution to the current tap position. A value of 0.0 means no PST mutation."""

    enable_nodal_inj_optim: bool = False
    """Whether to enable the nodal injection optimization stage. This can optimize PSTs (currently) and soon HVDC and
    potentially even redispatch clusters. Using this will increase runtime."""

    n_subs_mutated_lambda: PositiveFloat = 2.0
    """The number of substations to mutate in a single iteration is drawn from a poisson with this
    lambda"""

    proportion_crossover: confloat(ge=0, le=1) = 0.1
    """The proportion of the first topology to take in the crossover"""

    crossover_mutation_ratio: confloat(ge=0, le=1) = 0.5
    """The ratio of crossovers to mutations"""

    target_metrics: tuple[tuple[MetricType, float], ...] = (("overload_energy_n_1", 1.0),)
    """The list of metrics to optimize for with their weights"""

    observed_metrics: tuple[MetricType, ...] = (
        "max_flow_n_0",
        "overload_energy_n_0",
        "overload_energy_limited_n_0",
        "max_flow_n_1",
        "overload_energy_n_1",
        "overload_energy_limited_n_1",
        "split_subs",
        "switching_distance",
    )
    """The observed metrics, i.e. which metrics are to be computed for logging purposes. The
    target_metrics and me_descriptors must be included in the observed metrics and will be added
    automatically by the validator if they are missing"""

    me_descriptors: tuple[DescriptorDef, ...] = (
        DescriptorDef(metric="split_subs", num_cells=5),
        DescriptorDef(metric="switching_distance", num_cells=45),
    )
    """The descriptors to use for MAP-Elites. This includes a metric that determines the cell index
    and a number of cells. If the metric exceeds the number of cells, it will be clipped to the
    largest cell index. Currently, this must be integer metrics"""

    runtime_seconds: PositiveFloat = 60
    """The runtime in seconds"""

    iterations_per_epoch: PositiveInt = 100
    """The number of iterations per epoch"""

    random_seed: int = 42
    """The random seed to use for reproducibility"""

    cell_depth: PositiveInt = 1
    """When applicable, each cell contains cell_depth unique topologies. Use 1 to retain the
    original map-elites behaviour"""

    plot: bool = False
    """Whether to plot the repertoire"""

    mutation_repetition: PositiveInt = 1
    """More chance to get unique mutations by mutating multiple copies of the repertoire"""

    n_worst_contingencies: PositiveInt = 20
    """The number of worst contingencies to consider in the scoring function.
    This is used to determine the worst cases for overloads."""

    @model_validator(mode="after")
    def infer_missing_observed_metrics(self) -> BatchedMEParameters:
        """Add potentially missing target and descriptor metrics to the observed metrics."""
        for metric, _ in self.target_metrics:
            if metric not in self.observed_metrics:
                self.observed_metrics += (metric,)
        for descriptor in self.me_descriptors:
            if descriptor.metric not in self.observed_metrics:
                self.observed_metrics += (descriptor.metric,)
        return self


class LoadflowSolverParameters(BaseModel):
    """Parameters for the loadflow solver."""

    max_num_splits: NonNegativeInt = 4
    """The maximum number of splits per topology"""

    max_num_disconnections: NonNegativeInt = 0
    """The maximum number of disconnections to apply per topology"""

    batch_size: PositiveInt = 8
    """The batch size for the genetic algorithm"""

    distributed: bool = False
    """Whether to run the genetic algorithm distributed over multiple devices"""

    cross_coupler_flow: bool = False
    """Whether to compute cross-coupler flows"""


class DCOptimizerParameters(BaseModel):
    """The set of parameters that are used in the DC optimizer only"""

    ga_config: BatchedMEParameters = BatchedMEParameters()
    """The configuration options for the genetic algorithm"""

    loadflow_solver_config: LoadflowSolverParameters = LoadflowSolverParameters()
    """The configuration options for the loadflow solver"""

    double_limits: Optional[DoubleLimitsSetpoint] = None
    """The double limits for the optimization, if they should be updated"""

    summary_frequency: PositiveInt = 10
    """The frequency to push back results, based on number of iterations.
    Default is after every 10 iterations."""

    check_command_frequency: PositiveInt = 10
    """The frequency to check for new commands, based on number of iterations. Should be a multiple
    of summary_frequency"""
