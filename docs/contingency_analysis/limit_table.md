# Contingency analysis from the powsybl limit violation table

`run_contingency_analysis_powsybl_limit_table` in `toop_engine_contingency_analysis.pypowsybl_limit_table`
is the successor of `run_contingency_analysis_powsybl`. It runs the same powsybl security analysis, but it
reads the limit violation table instead of monitoring elements. The old runner is deprecated.

## What changes

- **The network is never modified.** The old runner wrote the distributed slack into the generator
  targets before the analysis. The new one hands the network to powsybl as it is. Lost injections of a
  contingency are redistributed by OpenLoadFlow according to the balance type of the loadflow parameters.
- **No monitored elements for contingencies.** Contingencies are only reported through the powsybl limit
  violation table, see [Limit violations](#limit-violations). Powsybl reports a current violation when the
  current exceeds the permanent or temporary limit times the *limit reduction*. The reduction is the
  `loading_above` threshold of the branch filter of a `LoadflowResultFilter`, capped at 1, so the branch
  violations are filtered inside powsybl. Without a threshold it is 1. Node filters cannot be pushed into powsybl, voltage
  violations are checked against the voltage limits of the network. They only apply to the base case node
  results, and the run adds a warning.
- **The base case is fully monitored** if the N-1 definition has one. Every branch, three winding
  transformer and voltage level of the network is registered for the pre-contingency state only, so the
  branch and node results hold the complete N-0 whatever the reduction. Without a base case they are empty.
  **Voltage angle difference results** are always empty.
- **Connectivity results** list the elements powsybl disconnected for each contingency, with the contingency
  id as outage group id.
- **Regulating element results** are empty. A security analysis does not report them.
- **Failed contingencies** only appear in the `converged` table. There are no NaN rows per element.
- **Contingency propagation** is the native powsybl one, switched on with `contingency_propagation=True`.
  Bus contingencies are not expanded over busbar sections beforehand.

## Limit violations

The violation table of powsybl is returned as it is, split in three fields of `LoadflowResultsPolars`, all
with the `LimitViolationSchemaPolars` columns:

- `branch_violations`: the `CURRENT`, `ACTIVE_POWER` and `APPARENT_POWER` violations of branch sides,
- `va_diff_violations`: the `LOW_VOLTAGE_ANGLE` and `HIGH_VOLTAGE_ANGLE` violations, keyed by the voltage angle
  limits defined on the network (`Network.create_voltage_angle_limits`, this library does not create them),
- `node_violations`: every other violation, e.g. `LOW_VOLTAGE` and `HIGH_VOLTAGE` keyed by voltage level.

The base case violations are in these tables as well, next to the monitored base case results.

No row is converted or dropped. A violation of a temporary limit carries that limit. A `loading` column is
added, with a magnitude of 1 at the violated limit:

| Limit type                               | `loading`                                                 |
| ---------------------------------------- | --------------------------------------------------------- |
| `CURRENT`, `ACTIVE_POWER`, `APPARENT_POWER` | `abs(value) / limit`, against the reported limit, which may be temporary |
| `HIGH_VOLTAGE`                           | `(value - nominal_v) / (limit - nominal_v)`, like `vm_loading` |
| `LOW_VOLTAGE`                            | `(value - nominal_v) / (nominal_v - limit)`, negative      |
| `LOW_VOLTAGE_ANGLE`, `HIGH_VOLTAGE_ANGLE` | `value / limit`                                          |

Other limit types have no loading.
Pre-contingency violations carry the base case id. If the N-1 definition has no base case, they are dropped
like the base case rows of the other tables. The pandas `LoadflowResults` have no violation tables, use
`polars=True` to get them.

## Using it

```python
import pypowsybl
from toop_engine_contingency_analysis.pypowsybl_limit_table import run_contingency_analysis_powsybl_limit_table
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK, VIOLATION_THRESHOLD

results = run_contingency_analysis_powsybl_limit_table(
    net,
    n_minus_1_definition,
    job_id="job",
    timestep=0,
    loadflow_params=CGMES_DISTRIBUTED_SLACK,
    security_params=pypowsybl.security.Parameters(increased_violations_parameters=VIOLATION_THRESHOLD),
    current_limit_reduction=0.8,
    method="ac",
    polars=True,
)
results.branch_violations.collect()
```

`run_powsybl_analysis_limit_table` is the lower level call. It returns the native result handle wrapped in
`PowsyblLimitTableRun`, whose accessors read each table as a polars lazy frame. The pypowsybl result
wrapper is bypassed on purpose: it builds a pandas multi-index of the violation table in its constructor.
