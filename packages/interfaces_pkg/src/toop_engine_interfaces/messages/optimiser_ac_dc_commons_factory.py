"""A modukle containing factory functions for creating protobuf messages related to AC/DC grid optimization interfaces.

Classes
Framework : Enum
    Specifies the grid modelling framework in use (e.g., "pypowsybl", "pandapower").
OptimizerType : Enum
    Specifies the type of optimizer ("ac" or "dc").

Functions
---------
create_grid_file(framework, grid_folder, timestep_correspondence="", coupling="none") -> PbGridFile
    Creates a GridFile protobuf message with specified framework, grid folder, timestep correspondence, and coupling.
    Raises ValueError if coupling is invalid.

create_descriptor_def(metric, num_cells, range=None) -> PbDescriptorDef
    Creates a DescriptorDef protobuf message for a given metric, number of cells, and optional range.
    Raises ValueError if num_cells is not positive or range is invalid.

create_double_limits_setpoint(lower=1.0, upper=1.0) -> PbDoubleLimitsSetpoint
    Creates a DoubleLimitsSetpoint protobuf message with specified lower and upper limits.
    Raises ValueError if limits are non-positive or lower > upper.

create_filter_distance_set(distances) -> PbFilterDistanceSet
    Creates a FilterDistanceSet protobuf message from a list of distances.
    Raises ValueError if the list is empty.

create_filter_strategy(
    filter_dominator_metrics_target=None,
    filter_dominator_metrics_observed=None,
    filter_discriminator_metric_distances=None,
    filter_discriminator_metric_multiplier=None,
    filter_median_metric=None
) -> PbFilterStrategy
    Creates a FilterStrategy protobuf message with specified filtering parameters.
    Raises ValueError if any multiplier is negative or if keys mismatch between distances and multipliers.
"""

from enum import Enum
from typing import Dict, List, Optional

from toop_engine_interfaces.messages.protobuf_schema.optimizer.ac_dc_commons_pb2 import (
    DescriptorDef as PbDescriptorDef,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.ac_dc_commons_pb2 import (
    DoubleLimitsSetpoint as PbDoubleLimitsSetpoint,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.ac_dc_commons_pb2 import (
    FilterDistanceSet as PbFilterDistanceSet,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.ac_dc_commons_pb2 import (
    FilterStrategy as PbFilterStrategy,
)
from toop_engine_interfaces.messages.protobuf_schema.optimizer.ac_dc_commons_pb2 import (
    GridFile as PbGridFile,
)


class Framework(str, Enum):
    """The grid modelling framework in use with the data"""

    PYPOWSYBL = "pypowsybl"
    PANDAPOWER = "pandapower"


class OptimizerType(Enum):
    """The type of optimizer, currently ac or dc"""

    DC = "dc"
    AC = "ac"


def create_grid_file(
    framework: Framework,
    grid_folder: str,
    timestep_correspondence: str = "",
    coupling: str = "none",
) -> PbGridFile:
    """
    Create a GridFile message.

    Parameters
    ----------
    framework: Framework
        Which modelling software can read the grid file.
    grid_folder : str
        Path to the scenario grid folder (relative, not absolute).
        Should comply with the folder structure defined in `interfaces.folder_structure`.
    timestep_correspondence : str, optional
        The timestep correspondence to real-world time (ISO 8601 datetime string).
    coupling : str, optional
        How this timestep is coupled to the previous timestep.
        Possible values: {"loose", "tight", "none"}.
        The first timestep must always be "none".

    Returns
    -------
    PbGridFile
        A protobuf `GridFile` instance.

    Raises
    ------
    ValueError
        If coupling is not one of the allowed values.
    """
    valid_couplings = {"loose", "tight", "none"}
    if coupling not in valid_couplings:
        raise ValueError(f"Invalid coupling '{coupling}'. Must be one of {valid_couplings}.")

    return PbGridFile(
        string=framework,
        grid_folder=grid_folder,
        timestep_correspondence=timestep_correspondence,
        coupling=coupling,
    )


def create_descriptor_def(
    metric: str,
    num_cells: int,
    range: Optional[tuple[float, float]] = None,
) -> PbDescriptorDef:
    """
    Create a DescriptorDef message.

    Parameters
    ----------
    metric : str
        The metric to use for the descriptor. Must correspond to an integer-type metric.
    num_cells : int
        Number of cells to use for the descriptor.
    range : tuple of float, optional
        Range of the descriptor dimension defined by (min, max).
        If not given, defaults to (0.0, float(num_cells)).

    Returns
    -------
    PbDescriptorDef
        A protobuf `DescriptorDef` instance.

    Raises
    ------
    ValueError
        If num_cells is not positive or range has invalid length.
    """
    if num_cells <= 0:
        raise ValueError("num_cells must be a positive integer.")
    if range is not None and len(range) != 2:
        raise ValueError("range must contain exactly two values: (min, max).")

    if range is None:
        range = (0.0, float(num_cells))

    return PbDescriptorDef(metric=metric, num_cells=num_cells, range=list(range))


def create_double_limits_setpoint(lower: float = 1.0, upper: float = 1.0) -> PbDoubleLimitsSetpoint:
    """
    Create a DoubleLimitsSetpoint message.

    Parameters
    ----------
    lower : float, optional
        The new lower limit. Default is 1.0 (no change).
    upper : float, optional
        The new upper limit. Default is 1.0 (no change).

    Returns
    -------
    PbDoubleLimitsSetpoint
        A protobuf `DoubleLimitsSetpoint` instance.

    Raises
    ------
    ValueError
        If lower or upper are non-positive or if lower > upper.
    """
    if lower <= 0 or upper <= 0:
        raise ValueError("Both lower and upper must be positive.")
    if lower > upper:
        raise ValueError("lower cannot be greater than upper.")

    return PbDoubleLimitsSetpoint(lower=lower, upper=upper)


def create_filter_distance_set(distances: List[float]) -> PbFilterDistanceSet:
    """
    Create a FilterDistanceSet message.

    Parameters
    ----------
    distances : list of float
        Distances used for filtering.

    Returns
    -------
    PbFilterDistanceSet
        A protobuf `FilterDistanceSet` instance.

    Raises
    ------
    ValueError
        If distances list is empty.
    """
    if not distances:
        raise ValueError("distances cannot be empty.")

    return PbFilterDistanceSet(distances=distances)


def create_filter_strategy(
    filter_dominator_metrics_target: List[str] | None = None,
    filter_dominator_metrics_observed: List[str] | None = None,
    filter_discriminator_metric_distances: Dict[str, PbFilterDistanceSet] | None = None,
    filter_discriminator_metric_multiplier: Dict[str, float] | None = None,
    filter_median_metric: List[str] | None = None,
) -> PbFilterStrategy:
    """
    Create a FilterStrategy message.

    Parameters
    ----------
    filter_dominator_metrics_target : list of str, optional
        Target metrics for the dominator filter (e.g., ["split_subs", "disconnected_branches"]).
    filter_dominator_metrics_observed : list of str, optional
        Observed metrics for the dominator filter (may include continuous values).
    filter_discriminator_metric_distances : dict of str -> FilterDistanceSet, optional
        Distances for metrics used in the discriminator filter.
    filter_discriminator_metric_multiplier : dict of str -> float, optional
        Multiplier dictionary for metric distances.
    filter_median_metric : list of str, optional
        Metrics to use for the median filter.

    Returns
    -------
    PbFilterStrategy
        A protobuf `FilterStrategy` instance.

    Raises
    ------
    ValueError
        If any multiplier is negative or if provided keys mismatch between distances and multipliers.
    """
    filter_dominator_metrics_target = filter_dominator_metrics_target or []
    filter_dominator_metrics_observed = filter_dominator_metrics_observed or []
    filter_discriminator_metric_distances = filter_discriminator_metric_distances or {}
    filter_discriminator_metric_multiplier = filter_discriminator_metric_multiplier or {}
    filter_median_metric = filter_median_metric or []

    for key, value in filter_discriminator_metric_multiplier.items():
        if value < 0:
            raise ValueError(f"Multiplier for '{key}' must be non-negative.")

    return PbFilterStrategy(
        filter_dominator_metrics_target=filter_dominator_metrics_target,
        filter_dominator_metrics_observed=filter_dominator_metrics_observed,
        filter_discriminator_metric_distances=filter_discriminator_metric_distances,
        filter_discriminator_metric_multiplier=filter_discriminator_metric_multiplier,
        filter_median_metric=filter_median_metric,
    )
