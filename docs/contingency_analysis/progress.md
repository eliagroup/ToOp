# Progress reporting

Pass `on_progress` to [`run_contingency_analysis_pandapower`][toop_engine_contingency_analysis.pandapower.contingency_analysis_pandapower.run_contingency_analysis_pandapower]
and the engine will call it as outage groups finish. It is **off by default**.

This hook is pandapower only. [`get_ac_loadflow_results`][toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service.get_ac_loadflow_results]
and the powsybl backend does not accept it.

## What it reports

`on_progress(done, total)`:

- **`total` is the outage-group count**, `len(grouped_contingencies)`. With
  grouping off that is the contingency list. With `cfg.apply_outage_grouping`
  several contingencies can share one group, so `total` can be smaller.
- **First call is `(0, total)`**, before the first load flow.
- **Later calls are about once a second**, then `(total, total)` if the run
  succeeds.
- **A failing callback does not fail the analysis.** The exception is logged
  and ignored.

## Sequential and parallel

The same callback works on both paths. Sequential calls it between load flows.
Parallel calls it on the driver process; a Python callable cannot be sent into
a Ray worker.

## Using it

```python
from toop_engine_contingency_analysis.pandapower.contingency_analysis_pandapower import (
    run_contingency_analysis_pandapower,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    ContingencyAnalysisConfig,
)

def report(done: int, total: int) -> None:
    print(f"{done}/{total} outage groups")

run_contingency_analysis_pandapower(
    net,
    nminus1_definition,
    job_id,
    timestep,
    cfg=ContingencyAnalysisConfig(method="ac", polars=True),
    on_progress=report,
)
```
