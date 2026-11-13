# Contingency analysis from the powsybl limit violation table

`run_contingency_analysis_powsybl_limit_table` in `toop_engine_contingency_analysis.pypowsybl_limit_table`
is the successor of `run_contingency_analysis_powsybl`. It runs the same powsybl security analysis, but it
reads the limit violation table instead of monitoring elements.

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
