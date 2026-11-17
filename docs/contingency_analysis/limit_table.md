# Contingency analysis from the powsybl limit violation table

`run_contingency_analysis_powsybl_violation_table` in `toop_engine_contingency_analysis.pypowsybl_limit_table`
is the successor of `run_contingency_analysis_powsybl`. It runs the same powsybl security analysis, but it
reads the limit violation table instead of monitoring elements.

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

The AC optimizer measures the fitness and the overload acceptance with the N-1 overload current instead of the
overload energy if a loadflow comes from the violation table, and ranks the worst contingencies by current. The
voltage jump and angle difference rejections are then not checked.

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
)
results.branch_violations.collect()
```

For the lower level calls, `create_powsybl_analysis_limit_table` builds the security analysis from a
`PowsyblLimitTableDefinition` (see `translate_nminus1_for_limit_table`), and `run_powsybl_analysis_limit_table`
runs it. The run returns the native result handle wrapped in `SecurityAnalysisResultPolars`, whose accessors
read each table as a polars lazy frame. The pypowsybl result
wrapper is bypassed on purpose: it builds a pandas multi-index of the violation table in its constructor.
