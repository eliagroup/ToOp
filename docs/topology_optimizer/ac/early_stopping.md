# Early Stopping in the AC Validation Loop

The AC optimizer now validates candidate topologies in two stages:

1. a fast-failing stage on a reduced contingency subset,
2. a remaining-contingency stage for survivors.

This page describes how the early-stopping subset is built and how it is used during AC validation. For the full worker-side control flow, see [AC Validation Loop](ac_loop.md).

## Overview

The goal of early stopping is not to prove optimality. It is a cheap rejection filter that prevents expensive full AC N-1 evaluation for clearly poor candidates.

The current implementation uses three sources of information:

- the worst `k` contingencies of the unsplit AC baseline,
- the worst `k` contingencies that came from the selected DC topology,
- the configured AC rejection thresholds for convergence, overload energy, and critical branch count.

The resulting subset is evaluated first. Only topologies that pass that stage continue to the remaining contingencies. The reason for this split is that full AC validation is the expensive part of the pipeline, so the optimizer benefits from spending a small amount of work on many candidates in order to reserve the large amount of work for the few candidates that still look promising.

## Build the AC Baseline

During AC initialization, `initialize_optimization` constructs the optimizer state and evaluates the unsplit topology once on AC before any candidate screening begins. Inside that initialization path, `update_initial_metrics_with_worst_k_contingencies` extracts the critical baseline contingencies and stores them as part of the unsplit reference topology. The reason for starting from the unsplit AC case is that early stopping needs a physically meaningful AC reference point: the optimizer is not asking whether a candidate is good in isolation, but whether it is already clearly worse than the trusted baseline from which the search began. The API surface for both functions is documented in the [Topology Optimizer Code Reference](../../references/reference_topology_optimizer.md).

That initialization computes and stores:

- `worst_k_contingency_cases`: the contingency IDs of the worst AC baseline cases,
- `top_k_overloads_n_1`: the aggregate overload across those cases,
- the full unsplit AC metrics used later as the acceptance baseline.

Conceptually, this gives the AC worker a baseline such as:

```json
{
    "worst_k_contingency_cases": ["a", "b", "c"],
    "top_k_overloads_n_1": 120.5
}
```

## Merge DC and AC Critical Cases When Pulling

When a DC topology is pulled into the AC repertoire, `pull` turns a DC candidate into an AC candidate and merges its `worst_k_contingency_cases` with the critical cases of the unsplit AC baseline. Conceptually, this is the point where the optimizer stops treating DC and AC evidence as separate hints and builds a single critical contingency set for the first AC screening pass. The reason for merging both sources is that neither one is sufficient alone: the unsplit AC baseline captures contingencies known to be critical in the high-fidelity model, while the selected DC topology can expose new critical cases introduced by the structural changes of that candidate. The implementation details of `pull` are also available through the [Topology Optimizer Code Reference](../../references/reference_topology_optimizer.md).

If the unsplit AC topology contains `[a, b, c]` and the pulled DC topology contains `[a, d, e]`, the AC topology receives the union:

```json
{
    "worst_k_contingency_cases": ["a", "b", "c", "d", "e"]
}
```

This is the set used for the first AC validation stage.

## Fast-Failing Stage

The fast-failing stage is executed by the worker loop before any remaining contingencies are considered. At the orchestration level, `run_optimization_epochs` decides when this stage is entered for a newly pulled batch. The actual batch execution is delegated to `run_fast_failing_epoch`, which in turn calls `score_strategy_worst_k_batch`. Inside that batched scorer, `score_strategy_worst_k` evaluates one topology at a time, and `get_early_stopping_contingency_ids` extracts the reduced subset of contingencies that should be solved for that topology. Together these functions define the first-stage rejection mechanism described on this page and in [AC Validation Loop](ac_loop.md); their signatures and surrounding abstractions are documented in the [Topology Optimizer Code Reference](../../references/reference_topology_optimizer.md). The key design idea is to make rejection decisions on the smallest subset that is still informative enough to rule out clearly inferior candidates.

For each candidate topology:

1. `get_early_stopping_contingency_ids` takes `topology.worst_k_contingency_cases`.
2. If a base case ID exists, it is appended so the subset also contains the N-0 reference.
3. The loadflow runner is temporarily restricted to only that subset.
4. The candidate metrics are compared against the unsplit AC baseline metrics for the same subset.

The comparison is done by `evaluate_acceptance` using:

- `reject_convergence_threshold`
- `reject_overload_threshold`
- `reject_critical_branch_threshold`

If one of these checks fails, the topology is rejected immediately with `early_stopping=True` in the rejection reason. This is intentionally conservative in one direction only: the stage is designed to reject candidates that already look bad, not to certify that surviving candidates are globally good.

## What Happens to Survivors

Passing the early-stopping stage does not mean the topology is accepted. It only means the topology is not obviously worse on the critical subset.

Survivors keep:

- their early-stage loadflow results,
- their early-stage metrics,
- the exact contingency subset that was already computed.

These are then forwarded to the remaining-contingency stage so the worker does not recompute the subset that was already solved.

In implementation terms, `process_fast_failing_results` separates survivors from immediate rejections, while `evaluate_remaining_contingencies` is the worker-side handoff into the full validation path. From there, `score_topology_remaining` reconstructs the full AC judgment for one survivor by reusing the already computed early-stage results, and `compute_remaining_loadflows` computes only the contingencies that were deliberately deferred. This separation is important conceptually: the worker manages pacing and batching, while the scoring layer manages reuse and metric consistency. The reason to preserve the early-stage results is straightforward: once the optimizer has already paid to solve the critical subset, recomputing it during the full stage would destroy much of the computational benefit that early stopping is meant to create. The interfaces of these functions are described in the [Topology Optimizer Code Reference](../../references/reference_topology_optimizer.md).

## Remaining-Contingency Stage

For surviving topologies, the AC optimizer evaluates all contingencies that were not part of the early-stopping subset.

The implementation:

1. computes only the missing contingencies,
2. concatenates them with the early-stage results,
3. recomputes the final metrics on the full AC result set,
4. applies the same acceptance logic again with `early_stopping=False`.

This is the stage that produces the final AC result persisted and emitted by the worker. The reason this second stage still recomputes the final metrics from the full result set is that acceptance must ultimately be based on the complete AC picture, not on the reduced subset that was only meant for screening.

## Example

Assume:

- unsplit AC baseline worst cases: `[a, b, c]`
- pulled DC worst cases: `[a, d, e]`
- merged early-stopping subset: `[a, b, c, d, e]`

If the candidate topology performs poorly enough on this subset to violate the configured thresholds against the unsplit AC baseline, it is rejected during the fast-failing stage and never enters the remaining-contingency stage.

If it stays within the thresholds, the optimizer evaluates only the contingencies outside that subset and then performs the final full acceptance check.
