# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Some small definitions that are common to both AC and DC optimizations"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, TypeAlias, Union

from pydantic import (
    BaseModel,
    PositiveFloat,
    PositiveInt,
)
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.types import MetricType

Fitness: TypeAlias = Literal["fitness"]


class Framework(str, Enum):
    """The grid modelling framework in use with the data"""

    PYPOWSYBL = "pypowsybl"
    PANDAPOWER = "pandapower"


class OptimizerType(Enum):
    """The type of optimizer, currently ac or dc"""

    DC = "dc"
    AC = "ac"


class GridFile(BaseModel):
    """Holds information about the grid files to load

    This refers to a grid folder for a single timestep. By convention, for every timestep to be
    optimized, we have a GridFile object. The time_coupling field will hold information on how
    to couple these timesteps together.
    """

    framework: Framework
    """Which modelling software can read the grid file"""

    grid_folder: str
    """The path to the scenario grid folder. This is not a full filepath but a path relative to the
    folder where all grid folders are stored. The folder should satisfy the folder structure as
    defined in interfaces.folder_structure. Workers are responsible of
    prefixing the full path to the base folder."""

    timestep_correspondence: Optional[datetime] = None
    """The timestep correspondence to real world time if given"""

    coupling: Literal["loose", "tight", "none"] = "none"
    """How this timestep is coupled to the previous timestep.

    Timesteps can either be
    - loosely coupled, which means they share a PTDF matrix but can have different topologies.
    - tightly coupled, which means they share a PTDF matrix and the same topology
    - not coupled, which means they have different PTDF matrices and topologies
    Depending on the data, loose and tight coupling might not be available, in which case the
    optimization will throw a warning and fall back to not coupled.

    The first timestep will always be uncoupled as there is nothing it can couple to.

    TODO this is not yet used.
    """

    @property
    def static_information_file(self) -> Path:
        """The path to the static information hdf5 file"""
        return Path(self.grid_folder) / PREPROCESSING_PATHS["static_information_file_path"]

    @property
    def grid_file(self) -> Path:
        """The path to the grid file"""
        if self.framework == Framework.PYPOWSYBL:
            return Path(self.grid_folder) / PREPROCESSING_PATHS["grid_file_path_powsybl"]
        if self.framework == Framework.PANDAPOWER:
            return Path(self.grid_folder) / PREPROCESSING_PATHS["grid_file_path_pandapower"]
        raise NotImplementedError(f"Grid file for framework {self.framework} not implemented.")

    @property
    def network_data_file(self) -> Path:
        """The path to the network data file"""
        return Path(self.grid_folder) / PREPROCESSING_PATHS["network_data_file_path"]

    @property
    def action_set_file(self) -> Path:
        """The path to the stored action set file"""
        return Path(self.grid_folder) / PREPROCESSING_PATHS["action_set_file_path"]

    @property
    def nminus1_definition_file(self) -> Path:
        """The path to the n-1 definition file"""
        return Path(self.grid_folder) / PREPROCESSING_PATHS["nminus1_definition_file_path"]


class DescriptorDef(BaseModel):
    """A descriptor definition for the MAP-Elites algorithm, defining the dimension of the repertoire."""

    metric: MetricType
    """The metric to use for the descriptor. Currently this must be an integer metric"""

    num_cells: PositiveInt
    """The number of cells to use for the descriptor"""

    range: Optional[tuple[float, float]] = None
    """The range of the descriptor dimension defined through min and max value. If not given, 0..num_cells will be used."""


class DoubleLimitsSetpoint(BaseModel):
    """A setpoint for the double limits, consisting of a lower and upper limit.

    Will be passed to update_max_mw_flows_according_to_double_limits.
    """

    lower: PositiveFloat = 1.0
    """The new lower limit. 1.0 will leave it unchanged"""

    upper: PositiveFloat = 1.0
    """The new upper limit. 1.0 will leave it unchanged"""


class FilterStrategy(BaseModel):
    """A filter strategy to pull strategies from the DC repertoire."""

    filter_dominator_metrics_target: Optional[list[MetricType]] = None
    """Whether to use the dominator filter for pulling strategies from the DC repertoire
    A dominator is a metric entry with no other entry with a better fitness,
    in respect to the distance to the original topology.
    The distance in measured by the metric, assuming that lower values are better.
    The target metric is used to fix the discrete value for which the dominance is checked.
    Note: expects the target metrics to be discrete values, not continuous. E.g. split_subs, disconnected_branches, etc.
    Example:
    filter_dominator_metrics = ["switching_distance", "split_subs"]
    """

    filter_dominator_metrics_observed: Optional[list[MetricType]] = None
    """Whether to use the dominator filter for pulling strategies from the DC repertoire
    A dominator is a metric entry with no other entry with a better fitness,
    in respect to the distance to the original topology.
    The distance in measured by the metric, assuming that lower values are better.
    The observed metric is used to apply the dominator filter. It gives the option to choose more
    dimensions for the dominator filter and can be both discrete and continues values.
    Example:
    filter_dominator_metrics = ["switching_distance", "split_subs"]
    """

    filter_discriminator_metric_distances: Optional[dict[Union[MetricType, Fitness], set[float]]] = None
    """The distances for the metrics to use for the dominator filter.
    This is a dictionary where the key is the metric and the value is a set of distances.
    The distances are used to filter out strategies that are too far away from the original topology.
    Note: the fitness metric is a special case, where it is treated as a percentage.
    Example:
    filter_discriminator_metric_distances = {
            "split_subs": {0.,0.},
            "switching_distance": {-0.9, 0.9},
            "fitness": {-0.1, 0.1},
        }
    """

    filter_discriminator_metric_multiplier: dict[MetricType, float] = {"split_subs": 1.0}
    """A dictionary defining multiplier for the metric distances.
        The keys are metric names and the values are sets of distances.
        If None, defaults to an empty dictionary.
        Multiple values are added by:
        distance_multiplier = (
           `metric_multiplier[metric1]` * `discriminator_df[metric1]` +
           `metric_multiplier[metric2]` * `discriminator_df[metric2]` + ...
        )
        example: {"split_subs": 0.5}
                 The discriminator_df will be multiplied by the split_subs value.
                 In the case of the metric_distances values will be multiplied by the
                 split_subs value and the metric_multiplier.
                 -> metric_distances["switching_distance"] * 0.5 * split_subs_col (e.g. 4 splits)
                 -> the metric distance is increased by this metric multiplier."""

    filter_median_metric: Optional[list[MetricType]] = None
    """Whether to use the median filter for pulling strategies from the DC repertoire.
    Note: expects the target metrics to be discrete values, not continuous. E.g. split_subs, disconnected_branches, etc.
    Example:
    filter_median_metric = ["split_subs"]
    """
