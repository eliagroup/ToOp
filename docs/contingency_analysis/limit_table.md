# Contingency analysis from the powsybl limit violation table

`run_contingency_analysis_powsybl_limit_table` in `toop_engine_contingency_analysis.pypowsybl_limit_table`
is the successor of `run_contingency_analysis_powsybl`. It runs the same powsybl security analysis, but it
reads the limit violation table instead of monitoring elements. The old runner is deprecated.

## What changes

- **The network is never modified.** The old runner wrote the distributed slack into the generator
  targets before the analysis. The new one hands the network to powsybl as it is. Lost injections of a
  contingency are redistributed by OpenLoadFlow according to the balance type of the loadflow parameters.
- **No monitored elements for contingencies.** Every branch row of a contingency is a current violation.
  Powsybl reports a violation when the current exceeds the permanent limit times a *limit reduction*. The
  reduction defaults to a small epsilon, so every loaded branch is reported. If a `LoadflowResultFilter`
  with a branch loading threshold is given, the threshold is the reduction, so the filter runs inside
  powsybl. Temporary limits are not reduced, and the loading is always relative to the permanent limit.
- **The base case is fully monitored** (`basecase_full_monitoring=True`). Every branch, three winding
  transformer and voltage level of the network is registered for the pre-contingency state only, so N-0
  stays complete whatever the reduction. Switch it off and the base case is read from the violation table
  like any contingency.
- **Node results** of contingencies are the voltage violations. Powsybl keys them by voltage level, so the
  element of such a row is the voltage level id, `vm` is the violating voltage and `vm_loading` the distance
  from nominal relative to the violated bound. Angle, p and q are unknown. The base case deviation is
  measured against the base case bus of the same voltage level closest to the violated bound.
- **Voltage angle differences** are the violations of the voltage angle limits defined on the network
  (`Network.create_voltage_angle_limits`). This library does not create them.
- **Connectivity results** list the elements powsybl disconnected for each contingency, with the contingency
  id as outage group id.
- **Regulating element results** are empty. A security analysis does not report them.
- **Failed contingencies** only appear in the `converged` table. There are no NaN rows per element.
- **Contingency propagation** is the native powsybl one, switched on with `contingency_propagation=True`.
  Bus contingencies are not expanded over busbar sections beforehand.

## Keeping only the worst rows

`violation_filter` reduces the contingency rows of the branch and node results:

| `LimitViolationFilter`          | keeps                                                            |
| ------------------------------- | ---------------------------------------------------------------- |
| `ALL`                           | every row                                                        |
| `WORST_OVERLOAD_PER_ELEMENT`    | per element, the row with the highest loading over all contingencies and sides |
| `WORST_ELEMENT_PER_CONTINGENCY` | per contingency, the row of the most loaded element              |

Base case rows are never filtered.

## Using it

```python
from toop_engine_contingency_analysis.pypowsybl_limit_table import (
    LimitViolationFilter,
    run_contingency_analysis_powsybl_limit_table,
)
from toop_engine_interfaces.loadflow_result_filter import BranchLoadflowResultFilter, LoadflowResultFilter

results = run_contingency_analysis_powsybl_limit_table(
    net,
    n_minus_1_definition,
    job_id="job",
    timestep=0,
    method="ac",
    polars=True,
    result_filter=LoadflowResultFilter(branch_filters=BranchLoadflowResultFilter(loading_above=0.8)),
    violation_filter=LimitViolationFilter.WORST_OVERLOAD_PER_ELEMENT,
)
```

`run_powsybl_analysis_limit_table` is the lower level call. It returns the native result handle wrapped in
`PowsyblLimitTableRun`, whose accessors read each table as a polars lazy frame. The pypowsybl result
wrapper is bypassed on purpose: it builds a pandas multi-index of the violation table in its constructor.
