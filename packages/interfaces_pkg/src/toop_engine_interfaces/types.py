# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Define global type aliases.

Currently this holds only the MetricType, a Literal of all possible metrics.

The following metrics are defined:
TODO document the other metrics
- non_converging_loadflows: The number of loadflow cases (N-0 or N-1) that did not converge. In the DC optimizer, this
value is always 0 as either the entire loadflow converges or the solver returns a separate success flag which invalidates
all other returns. On AC this is not the case, so this metric is useful to track
"""

from beartype.typing import Literal, TypeAlias

MatrixMetric: TypeAlias = Literal[
    "max_flow_n_0",
    "median_flow_n_0",
    "overload_energy_n_0",
    "underload_energy_n_0",
    "overload_energy_limited_n_0",
    "exponential_overload_energy_n_0",
    "exponential_overload_energy_limited_n_0",
    "critical_branch_count_n_0",
    "critical_branch_count_limited_n_0",
    "max_flow_n_1",
    "median_flow_n_1",
    "overload_energy_n_1",
    "underload_energy_n_1",
    "overload_energy_limited_n_1",
    "exponential_overload_energy_n_1",
    "exponential_overload_energy_limited_n_1",
    "critical_branch_count_n_1",
    "critical_branch_count_limited_n_1",
    "top_k_overloads_n_1",
    "cumulative_overload_n_0",
    "cumulative_overload_n_1",
]

OperationMetric: TypeAlias = Literal[
    "switching_distance",
    "split_subs",
    "disconnected_branches",
]

OtherMetric: TypeAlias = Literal[
    "n0_n1_delta",
    "cross_coupler_flow",
    "n_2_penalty",
    "bb_outage_penalty",
    "bb_outage_overload",
    "bb_outage_grid_splits",
    "max_va_across_coupler",
    "max_va_diff_n_0",
    "max_va_diff_n_1",
    "overload_current_n_0",
    "overload_current_n_1",
    "non_converging_loadflows",
    # TODO: FIXME: remove fitness_dc when "Topology" is refactored and accepts different stages like "dc", "dc+" and "ac"
    "fitness_dc",
]

MetricType: TypeAlias = Literal[MatrixMetric, OperationMetric, OtherMetric]
