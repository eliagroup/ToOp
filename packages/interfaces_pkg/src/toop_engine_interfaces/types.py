# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Define global type aliases.

Currently this holds only the MetricType, a Literal of all possible metrics.

The meaning of the metrics is documented in metrics.md in the docs folder.
"""

from enum import Enum

from beartype.typing import Literal, TypeAlias

MatrixMetric: TypeAlias = Literal[
    "max_flow_artificial_limit_n_0",
    "median_flow_artificial_limit_n_0",
    "overload_energy_artificial_limit_n_0",
    "underload_energy_artificial_limit_n_0",
    "exponential_overload_energy_artificial_limit_n_0",
    "critical_branch_count_artificial_limit_n_0",
    "max_flow_artificial_limit_n_1",
    "median_flow_artificial_limit_n_1",
    "overload_energy_artificial_limit_n_1",
    "underload_energy_artificial_limit_n_1",
    "exponential_overload_energy_artificial_limit_n_1",
    "critical_branch_count_artificial_limit_n_1",
    "top_k_overloads_artificial_limit_n_1",
    "cumulative_overload_artificial_limit_n_0",
    "cumulative_overload_artificial_limit_n_1",
    "transport_artificial_limit_n_0",
    "transport_artificial_limit_n_1",
]

OperationMetric: TypeAlias = Literal[
    "switching_distance",
    "split_subs",
    "disconnected_branches",
    "fitness",
    "pst_switching_distance",
    "pst_switching_distance_squared",
    "pst_activated",
]

OtherMetric: TypeAlias = Literal[
    "n0_n1_delta",
    "cross_coupler_flow",
    "bb_outage_penalty",
    "bb_outage_overload",
    "bb_outage_grid_splits",
    "max_va_diff_n_0",
    "max_va_diff_n_1",
    "critical_va_diff_count_n_0",
    "critical_va_diff_count_n_1",
    "max_flow_permanent_n_0",
    "max_flow_permanent_n_1",
    "max_flow_displayed_permanent_n_0",
    "max_flow_displayed_permanent_n_1",
    "overload_energy_permanent_n_0",
    "overload_energy_permanent_n_1",
    "overload_current_permanent_n_0",
    "overload_current_permanent_n_1",
    "critical_branch_count_permanent_n_0",
    "critical_branch_count_permanent_n_1",
    "overload_current_artificial_limit_n_0",
    "overload_current_artificial_limit_n_1",
    "voltage_jump_count_n_1",
    "non_converging_loadflows",
    # TODO: FIXME: remove fitness_dc when "Topology" is refactored and accepts different stages like "dc", "dc+" and "ac"
    "fitness_dc",
]

MetricType: TypeAlias = Literal[MatrixMetric, OperationMetric, OtherMetric]

# The scopes the AC metrics are computed against.
MetricLimitScope: TypeAlias = Literal["permanent", "artificial_limit"]

LimitCase: TypeAlias = Literal["n_0", "n_1"]


class OperationalLimit(str, Enum):
    """The names of the operational limits of the monitored branches in the powsybl network."""

    PERMANENT = "permanent_limit"
    DEFAULT_PERMANENT = "default_permanent_limit"
    DACF_PERMANENT_N_1 = "dacf_permanent_n_1"  # TODO: remove me
    LOADFLOW_BASED_N_0 = "loadflow_based_n_0"
    LOADFLOW_BASED_N_1 = "loadflow_based_n_1"
    ARTIFICIAL_N_0 = "artificial_limit_n_0"
    ARTIFICIAL_N_1 = "artificial_limit_n_1"
    ARTIFICIAL_DC_N_0 = "artificial_dc_limit_n_0"
    ARTIFICIAL_DC_N_1 = "artificial_dc_limit_n_1"

    @property
    def case(self) -> LimitCase:
        """The case the limit applies to, the permanent limit counts as N-0."""
        return "n_1" if self in _N_1_LIMITS else "n_0"

    @property
    def acceptable_duration(self) -> int:
        """The acceptable duration of the limits the importer creates.

        Powsybl keys temporary limits by their duration, so the DC limits are shifted to not overwrite the AC ones.
        """
        duration = 200 if self.case == "n_1" else 100
        return duration + 10 if self in _DC_LIMITS else duration


_N_1_LIMITS = (
    OperationalLimit.DACF_PERMANENT_N_1,
    OperationalLimit.LOADFLOW_BASED_N_1,
    OperationalLimit.ARTIFICIAL_N_1,
    OperationalLimit.ARTIFICIAL_DC_N_1,
)
_DC_LIMITS = (OperationalLimit.ARTIFICIAL_DC_N_0, OperationalLimit.ARTIFICIAL_DC_N_1)
