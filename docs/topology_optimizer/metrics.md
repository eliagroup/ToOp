# Metrics

There are multiple metrics that the optimizers can compute. We can separate these metrics into 3 classes:

* Matrix Metrics: All metrics that can be computed from the N-1 matrix
* Operation Metrics: Metrics that are dictated by operation and not a result of the loadflow computation
* Other Metrics: Various other metrics

Some metrics are exclusive to one stage (ac/dc) and some are feasible on both stages. Generally, Matrix Metrics and
Operation Metrics are feasible on both stages while for other metrics it depends.

All N-0 metrics operate on the base case (no contingency), while N-1 metrics consider all contingencies. Metrics with
`_limited` suffix use alternative (usually more conservative) flow limits from `max_mw_flow_limited` or `max_mw_flow_n_1_limited`.

## Matrix Metrics

- **max_flow_n_0** / **max_flow_n_1**: The maximum flow on any branch in fractions. For N-0 this is the base case, for N-1 this considers all contingencies. A value of 1.1 means the highest loaded branch is 10% above its limits.

- **median_flow_n_0** / **median_flow_n_1**: The median relative flow across all branches. For N-1, considers the worst contingency for each branch. Provides a measure of overall grid loading.

- **overload_energy_n_0** / **overload_energy_n_1**: Total amount of energy (in MW) that exceeds branch ratings, summed across all timesteps and overloaded branches. For N-1, considers the worst contingency for each timestep. High voltage branches dominate this metric as they carry more MW, so the optimizer prioritizes these. Can optionally apply per-branch weights via `overload_weight`.

- **underload_energy_n_0** / **underload_energy_n_1**: Total unused capacity (in MW) below branch ratings. Indicative of how much flow can still be pushed through the grid, but is no sensible optimization metric. Use for debugging purposes only.

- **overload_energy_limited_n_0** / **overload_energy_limited_n_1**: Same as `overload_energy` but uses `max_mw_flow_limited` (N-0) or `max_mw_flow_n_1_limited` (N-1, with fallback to `max_mw_flow_limited`) as the threshold. The limits are updated from the double limits, i.e. double limits are only mirrored in the limited versions.

- **exponential_overload_energy_n_0** / **exponential_overload_energy_n_1**: Exponentially weighted overload energy (default α=1.5) that more heavily penalizes severe overloads. Branches loaded at 150% contribute more than proportionally compared to those at 110%.

- **exponential_overload_energy_limited_n_0** / **exponential_overload_energy_limited_n_1**: Same as `exponential_overload_energy` but uses limited flow thresholds.

- **critical_branch_count_n_0** / **critical_branch_count_n_1**: The number of branches that are overloaded in the worst timestep. For N-1, counts branches overloaded in at least one contingency. Targeting this metric will incentivize the optimizer to concentrate the overload onto fewer lines, which might be easier to redispatch than many overloaded lines at the same time.

- **critical_branch_count_limited_n_0** / **critical_branch_count_limited_n_1**: Same as `critical_branch_count` but uses `max_mw_flow_limited` thresholds.

- **cumulative_overload_n_0** / **cumulative_overload_n_1**: Sum of relative overload percentages across all branches. Unlike overload energy, this metric treats all branches equally regardless of their capacity - a 10% overload on a small line contributes the same as on a large line.

- **top_k_overloads_n_1**: Returns information about the k worst N-1 contingencies including their relative overloads and identifiers. This is used internally for the early stopping feature and should not become an optimization metric. For debugging purposes, plotting might be interesting.

## Operation Metrics

These metrics measure operational aspects of the topology rather than electrical flows:

- **switching_distance**: Total number of reassignments required to reach the topology from the initial state, computed using `reassignment_distance` information. Measures the number of switching operations needed. Note that currently, the AC solver counts this slightly differently than the DC solver. This is a bug, not a feature.

- **split_subs**: Number of substations that have been split into multiple busbars. Higher values indicate more complex substation configurations.

- **disconnected_branches**: Number of branches (lines/transformers) that have been intentionally disconnected in the topology.

## Other Metrics

- **n0_n1_delta**: Penalty for exceeding the maximum allowed flow change between N-0 and N-1 cases. Computed as the sum of MW exceeding `n0_n1_max_diff` limits across all branches and timesteps. Useful for operational constraints that limit how much flow can shift during contingencies.

- **cross_coupler_flow**: Penalty for violating maximum cross-coupler flow limits in split substations, summed across all splits and timesteps. Enforces limits on power flow between busbars within a substation.

- **n_2_penalty**: Penalty value for N-2 contingency violations, if N-2 analysis is enabled. Higher values indicate more severe or more frequent N-2 issues.

- **bb_outage_penalty**: Penalty for busbar outage scenarios. Only relevant when busbar outage analysis is included.

- **bb_outage_overload**: Maximum overload observed during busbar outage scenarios.

- **bb_outage_grid_splits**: Number of grid splits (islands) created during busbar outage scenarios.

- **max_va_across_coupler**: Maximum voltage angle difference across couplers in split substations. Only applicable in AC analysis.

- **max_va_diff_n_0** / **max_va_diff_n_1**: Maximum voltage angle difference across any branch. AC analysis only.

- **overload_current_n_0** / **overload_current_n_1**: Total overload measured in amperes (current) rather than MW. AC analysis only.

- **non_converging_loadflows**: The number of loadflow cases (N-0 or N-1) that did not converge. In the DC optimizer, this value is always 0 as either the entire loadflow converges or the solver returns a separate success flag which invalidates all other returns. On AC this is not the case, so this metric is useful to track.

- **fitness_dc**: Legacy fitness metric for DC stage. TODO: Remove when Topology is refactored to accept different stages like "dc", "dc+" and "ac".
