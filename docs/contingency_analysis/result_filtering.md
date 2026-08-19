# Filtering loadflow results

A contingency analysis produces one result row per timestep, contingency, element and side. Most of
those rows say nothing: a branch loaded to 12% of its rating, a busbar sitting at nominal voltage.
They are computed, carried between workers, and written to disk exactly like a violation is — and in
an optimization run, a full result set is stored for every candidate topology that gets evaluated.

A `LoadflowResultFilter` drops the rows that carry no decision value, as they are produced. It is
**off by default**: without one, results are exactly what they always were.

Furthrmore, the used filters are stored along the results.

## What it filters

Branch results and node results. Every other table — convergence, voltage-angle differences,
regulating elements, SpPS, cascade, connectivity — is small and passes through untouched.

## How a row is judged

- **Any active threshold can keep a row.** They are alternatives, not conditions to satisfy together.
- **Unknown is never dropped.** A branch with no rated current, a bus with no voltage limits, or any
  row from a contingency that did not converge has no value to compare, so it is kept.
- **The base case is exempt**, unless you say otherwise in `retain_basecase`.


## Using it

```python
from toop_engine_contingency_analysis.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_interfaces.loadflow_result_filter import (
    BranchLoadflowResultFilter,
    LoadflowResultFilter,
    NodeLoadflowResultFilter,
)

result_filter = LoadflowResultFilter(
    # Keep branches at or above 70% of their rated current.
    branch_filters=BranchLoadflowResultFilter(loading_above=0.7),
    # Keep buses within the top 30% of their voltage band, plus any bus that moves
    # more than 5% between N-0 and N-1 even while it stays inside the band.
    node_filters=NodeLoadflowResultFilter(vm_loading_above=0.7, vm_basecase_deviation_above=5.0),
)

results = get_ac_loadflow_results(net, nminus1_definition, result_filter=result_filter)
```

The two backends take the same policy directly:

```python
# pandapower
run_contingency_analysis_pandapower(
    net, nminus1_definition, job_id, timestep,
    cfg=ContingencyAnalysisConfig(method="ac", polars=True, result_filter=result_filter),
)

# powsybl
run_contingency_analysis_powsybl(
    net, nminus1_definition, job_id, timestep, method="ac", polars=True, result_filter=result_filter,
)
```

To filter results you already have, rather than as they are computed:

```python
from toop_engine_contingency_analysis.result_filter import apply_result_filter

filtered = apply_result_filter(results, result_filter, nminus1_definition.base_case.id)
```
