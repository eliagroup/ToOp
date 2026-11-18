# Contingency analysis from the powsybl limit violation table

`run_contingency_analysis_powsybl_violation_table` in `toop_engine_contingency_analysis.pypowsybl_limit_table`
is the successor of `run_contingency_analysis_powsybl`. It runs the same powsybl security analysis, but it
reads the limit violation table instead of monitoring elements.

It does not replace the monitored element runner everywhere: the table only holds the elements beyond a limit, so
whatever needs the flows of every monitored branch for every contingency keeps `run_contingency_analysis_powsybl`.
That is the importer, which derives the artificial limits from the worst N-1 current of every branch, and the
validation of the DC solver against the full N-1 flow matrix.

## Limit violations

The violation table of powsybl is returned as it is, split in three fields of `LoadflowResults` and
`LoadflowResultsPolars`, all following `LimitViolationSchema` (indexed by `timestep`, `contingency`, `element`,
`side`, `limit_type` and `limit_name` in pandas):

- `branch_violations`: the `CURRENT`, `ACTIVE_POWER` and `APPARENT_POWER` violations of branch sides,
- `va_diff_violations`: the `LOW_VOLTAGE_ANGLE` and `HIGH_VOLTAGE_ANGLE` violations, keyed by the voltage angle
  limits defined on the network (`Network.create_voltage_angle_limits`, this library does not create them),
- `node_violations`: every other violation, e.g. `LOW_VOLTAGE` and `HIGH_VOLTAGE` keyed by voltage level.

The base case violations are in these tables as well, next to the monitored base case results.

## Monitored elements

The monitored elements of the N-1 definition are translated exactly like for `run_contingency_analysis_powsybl`.
They are monitored for the base case only, so the base case branch, node and regulating element results and the
branch limits of both runners match.

Powsybl reports the violations of every element with limits. They are reduced to the monitored elements:

| Table | Kept |
| ----- | ---- |
| `branch_violations` | the monitored lines, tie lines, two and three winding transformers |
| `node_violations` | the voltage levels holding at least one monitored bus |
| `va_diff_violations` | all, they come from angle limits defined on the network on purpose |

Apart from that, no row is converted or dropped. A violation of a temporary limit carries that limit. A `loading`
column is added, with a magnitude of 1 at the violated limit:

| Limit type                               | `loading`                                                 |
| ---------------------------------------- | --------------------------------------------------------- |
| `CURRENT`, `ACTIVE_POWER`, `APPARENT_POWER` | `abs(value) / limit`, against the reported limit, which may be temporary |
| `HIGH_VOLTAGE`                           | `(value - nominal_v) / (limit - nominal_v)`, like `vm_loading` |
| `LOW_VOLTAGE`                            | `(value - nominal_v) / (nominal_v - limit)`, negative      |
| `LOW_VOLTAGE_ANGLE`, `HIGH_VOLTAGE_ANGLE` | `value / limit`                                          |

Other limit types have no loading.
`side` keeps the powsybl naming: `ONE`, `TWO` or `THREE` for branch sides, empty for voltage and angle
violations.
Pre-contingency violations carry the base case id. If the N-1 definition has no base case, they are dropped
like the base case rows of the other tables.

## Metrics

`compute_metrics` takes the N-0 metrics from the monitored base case, like for `run_contingency_analysis_powsybl`.
The N-1 branch metrics come from the `CURRENT` violations of `branch_violations` if the results have a violation
table, see `uses_limit_violations` and `get_branch_results_for_metrics`. The monitored runner leaves the table `None`,
the violation runner always sets it, so an empty table means no N-1 overload.

A branch side is reported once its current exceeds its lowest limit times the limit reduction. The artificial limits
are temporary current limits below the permanent limit, so every overload against the permanent or an artificial limit
is in the table. The loading is computed from the reported current against each limit of `branch_limits`.

| N-1 metric | From the violation table |
| ---------- | ------------------------ |
| `overload_current_*`, `critical_branch_count_*` | same as the monitored run |
| `max_flow_*` | same as the monitored run if a branch is reported, 0.0 otherwise, like for a result filter |
| `overload_energy_*` (in P) | `None`, the table has no active power |
| `max_va_diff_n_1`, `critical_va_diff_count_n_1`, `voltage_jump_count_n_1` | `None`, the table has no angle differences of the monitored switches and no voltages within the limits |

Powsybl reports an N-1 violation that already exists in N-0 only if it grew by more than the flow threshold of the
increased violation parameters, i.e. if `post > pre * (1 + threshold)`. `get_default_security_analysis_parameters`
sets the flow threshold to -1, so every N-1 flow violation is reported, also one that is equal to or lower than in N-0.
A negative threshold is outside of the range powsybl documents (positive), but it is not validated. With a threshold of
0 or more, an element overloaded in N-0 that no contingency increases has no N-1 violation, so its N-1 overload is
missing from the metrics. Powsybl's own default is 0.1.

A contingency that changes nothing, e.g. the outage of a busbar section nothing is connected to, gets no result
from powsybl once the contingency is propagated. It is `NO_CALCULATION` in the convergence table and has no
violations, where the monitored element runner reports it as `CONVERGED` with the base case flows.

## Parameters

The function sets no defaults and does not modify the network or the parameters it is given. The caller owns the
`pypowsybl.security.Parameters`, `get_default_security_analysis_parameters` builds a set that suits both runners.
Two things are decided by the runner, on a copy of the parameters:

| | `run_contingency_analysis_powsybl_violation_table` | `run_contingency_analysis_powsybl` |
| - | - | - |
| provider parameter `contingencyPropagation` | always `true`, powsybl propagates a busbar outage to the breakers | always `false`, busbar outages are expanded before the run, see [propagation](propagation.md) |
| distributed slack | used as given | baked into the network before the run; switched off for DC |

The slack matters for DC: with a distributed slack, powsybl redistributes the injection a contingency loses over
the remaining generators. The monitored element runner and the DC solver distribute the slack in N-0 only and leave
the lost injection to the slack bus. A caller that wants the same for the violation table has to bake the N-0 slack
into the network and pass parameters without a distributed slack, as `PowsyblRunner` does for its DC loadflows.

`current_limit_reduction` scales the permanent and temporary current limits for the detection, so 0.8 reports every
branch side loaded above 80 % of one of its limits. `branch_limit_cache` takes the prepared branch limits of
`build_branch_limit_cache`, which saves preparing them again for every run on the same network and monitored
branches. A stale cache is ignored.

## In the optimizer

`ACGAParameters.use_limit_violations` (default `False`) switches the loadflows of the AC optimizer to the violation
table. It is passed to `PowsyblRunner(use_limit_violations=True)`, whose N-1 loadflows, AC and DC, then come from
`run_contingency_analysis_powsybl_violation_table`. It is not available for pandapower grids. A caller of the runner
that needs the full N-1 flows, like `validate_loadflow_results`, uses a runner without it.

With it, the AC optimizer measures the fitness and the overload acceptance with the N-1 overload current instead of
the overload energy, and ranks the worst contingencies by current. The initial loadflow of the importer still comes
from the monitored element runner; a comparison of both then uses the overload current, which both have. The voltage
jump and angle difference rejections cannot be checked: `enable_critical_voltage_rejection` logs a warning and is
skipped. Of `ACGAParameters.result_filter`, only `branch_filters.loading_above` is used, as the current limit
reduction.

## Using it

```python
import pypowsybl
from toop_engine_contingency_analysis.pypowsybl_limit_table import run_contingency_analysis_powsybl_violation_table
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK, VIOLATION_THRESHOLD

results = run_contingency_analysis_powsybl_violation_table(
    net,
    n_minus_1_definition,
    job_id="job",
    timestep=0,
    security_analysis_params=pypowsybl.security.Parameters(
        load_flow_parameters=CGMES_DISTRIBUTED_SLACK,
        increased_violations_parameters=VIOLATION_THRESHOLD,
    ),
    current_limit_reduction=0.8,
    method="ac",
    polars=True,
    branch_limit_cache=None,  # optional, see build_branch_limit_cache
)
results.branch_violations.collect()
```

For the lower level calls, `create_powsybl_analysis_limit_table` builds the security analysis from a
`PowsyblLimitTableDefinition` (see `translate_nminus1_for_limit_table`), and `run_powsybl_analysis_limit_table`
runs it. The run returns the native result handle wrapped in `SecurityAnalysisResultPolars`, whose accessors
read each table as a polars lazy frame. The pypowsybl result
wrapper is bypassed on purpose: it builds a pandas multi-index of the violation table in its constructor. This uses
the private `_pypowsybl` bindings, so it is the place to look first after a pypowsybl upgrade.
