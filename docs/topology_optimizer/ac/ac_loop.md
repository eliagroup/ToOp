# AC Validation Loop

This page describes the current AC worker loop implemented around `run_optimization_epochs` and the helper functions it calls.

The AC stage is no longer a monolithic "evaluate everything immediately" loop. It is a batched, two-stage validation loop that:

1. pulls candidate topologies from the DC repertoire,
2. runs a fast-failing AC subset evaluation,
3. accumulates survivors,
4. flushes survivors into full remaining-contingency evaluation,
5. persists and emits final results.

For the subset logic itself, see [Early Stopping in the AC Validation Loop](early_stopping.md).

At the top level, `optimization_loop` is the lifecycle boundary for one AC optimization run: it binds the optimization context, delegates initialization, executes the iterative epoch loop, and finally writes the summary. The transition from bootstrap to active optimization happens in `initialize_optimization_run`, which prepares the optimizer state and blocks until the first DC results are available. The central control law then lives in `run_optimization_epochs`, while the concrete batch operations are delegated to `run_fast_failing_epoch`, `process_fast_failing_results`, `evaluate_remaining_contingencies`, `run_remaining_epoch`, and `process_remaining_results`. This page focuses on the interaction between those abstractions; their full APIs are documented in the [Topology Optimizer Code Reference](../../references/reference_topology_optimizer.md). The selection side of the pipeline is described separately in [Selection Strategy](select_strategy.md), and the reduced-subset screening logic is described in [Early Stopping in the AC Validation Loop](early_stopping.md).

## Entry Point

`optimization_loop` creates the AC run context for a single optimization ID.

In conceptual terms, it performs three phases: initialization, iterative validation, and summarization. Initialization is delegated to `initialize_optimization_run`, which prepares the optimizer data and waits for the first DC repertoire entries; iterative validation is delegated to `run_optimization_epochs`; and the final phase writes the post-run summary. The consequence is a clean separation between lifecycle management and per-epoch decision making.

## Step-by-Step Flow of `run_optimization_epochs`

### 1. Initialize Loop State

At the start of the loop, the worker creates local state for:

- `start_time`: wall-clock start of the AC run,
- `last_full_run`: timestamp of the last survivor flush into full evaluation,
- `survivor_batch_size`: equal to `ga_config.runner_processes`,
- `epoch`: current AC loop iteration,
- `evaluated_topologies`: number of candidates seen by the fast-failing stage,
- `survivor_topologies`: buffered candidates that passed the fast-failing stage,
- `survivor_early_results`: matching early-stage results for those survivors.

This state is intentionally maintained in the worker, not in the scoring functions. The reason is architectural: these variables describe scheduling, batching, and lifecycle progress rather than electrical behavior. Keeping them in the worker lets the scoring layer stay focused on AC evaluation semantics while the worker remains responsible for throughput control and stop conditions.

### 2. Bind Epoch Context

Each iteration runs inside:

```python
with structlog.contextvars.bound_contextvars(epoch=epoch):
```

That keeps the epoch attached to all module-level logger calls inside the worker and optimizer helpers without passing bound logger objects around.

### 3. Poll the Results Topic

The loop begins by calling `poll_results_topic`.

This import step has three purposes: it moves newly produced DC topologies into the AC worker's local database, keeps the worker's repertoire view synchronized with the upstream DC process, and ensures that the next selection step can act on fresh candidates rather than stale local state. It happens once per epoch before any new AC topology is pulled. The reason for doing this eagerly is that the AC loop is not an isolated optimizer with a fixed search space; it is a downstream consumer of a continuously evolving DC repertoire, so delaying the import would directly reduce the usefulness of the AC selection stage.

### 4. Run the Fast-Failing Stage

The worker then calls `run_fast_failing_epoch`.

This helper first invokes `optimizer_data.evolution_fn()` to select and pull a new candidate batch from the DC repertoire into the AC domain. It then calls `optimizer_data.worst_k_scoring_fn(topologies)` to evaluate that batch on the reduced early-stopping subset and returns both the candidate topologies and their early-stage results. Even when `enable_ac_rejection=False`, this stage still runs because the early-stage results are needed later for reuse; what changes is not the computation itself but how the worker interprets and emits the intermediate outcomes. The deeper reason for keeping this stage separate is cost asymmetry: a reduced subset can cheaply reject obviously poor candidates, while a full remaining-contingency pass is expensive enough that it should be reserved for topologies that have already cleared a first barrier.

### 5. Process Fast-Failing Results

The loop branches depending on `ga_config.enable_ac_rejection`.

When `enable_ac_rejection=True`, `process_fast_failing_results` acts as an immediate classification boundary: rejected topologies are persisted and emitted at once, while only survivors are kept in the worker buffers for later full evaluation. When `enable_ac_rejection=False`, the same early-stage computation is retained, but the worker no longer treats that boundary as externally visible rejection; instead, all candidates are buffered and the early-stage outputs are used only as reusable intermediate state for the later full check. The structure of the loop therefore stays constant while the semantics of rejection change. This uniformity is deliberate: a single control flow is easier to reason about, easier to test, and avoids splitting the implementation into two partially duplicated pipelines that differ only in how rejections are surfaced.

### 6. Decide When to Flush Survivors

The worker does not immediately run the remaining-contingency stage for every survivor.

Instead, it flushes survivors when one of two conditions becomes true:

1. `len(survivor_topologies) >= survivor_batch_size`
2. `time.time() - last_full_run > ga_config.remaining_loadflow_wait_seconds` and at least one survivor is buffered

This gives the AC worker two controls:

batch for throughput when enough survivors are available, and a timeout-based flush to avoid unbounded waiting when survivor production is slow.

The reason for this deferred flush is that the remaining-contingency stage benefits much more from topology-level parallelization than the fast-failing stage. If survivors were pushed through one by one as soon as they passed early stopping, the worker would repeatedly launch small full-evaluation batches and underutilize the runner pool. By waiting until enough survivors accumulate, the worker can keep more runner processes busy with independent topologies and obtain better throughput from the expensive part of the AC pipeline.

When a flush is triggered, the worker calls:

```python
evaluate_remaining_contingencies(
    send_result_fn,
    optimizer_data,
    epoch,
    survivor_topologies[:survivor_batch_size],
    survivor_early_results[:survivor_batch_size],
)
```

After that call, the flushed prefix is removed from both survivor buffers.

The timeout condition exists for the opposite reason: maximizing batch size is useful, but waiting indefinitely for a perfectly full batch would hurt latency and can strand valid survivors when the incoming stream is sparse. The loop therefore balances throughput against forward progress.

### 7. Send Progress Heartbeats

Every epoch sends an `OptimizationStatsHeartbeat` containing:

- `optimization_id`
- current wall time
- current iteration
- number of branch topologies tried
- number of injection topologies tried

The branch topology counter is computed as:

```python
evaluated_topologies - len(survivor_topologies)
```

So buffered survivors are not counted as fully processed until they have been flushed into remaining-contingency evaluation. This accounting choice matches the actual cost structure of the loop: a topology that only passed the cheap screening stage has not yet consumed the full AC validation budget and should not be reported as fully processed.

### 8. Stop on Runtime Limit

If the configured `runtime_seconds` is exceeded, the worker performs a final flush before stopping.

If survivors are still buffered at that point, the worker pushes all of them through the remaining-contingency stage before it emits `OptimizationStoppedResult` with `reason="converged"` and `message="runtime limit"`. This last flush is conceptually important because the runtime boundary is treated as a stop condition for the search, not as permission to discard already screened candidates. In other words, once the system has already spent work to produce plausible survivors, the implementation prefers to finish their expensive validation rather than lose information at shutdown.

## What `evaluate_remaining_contingencies` Does

This helper is the handoff from the worker queue to the full AC evaluation path.

Internally, it first delegates to `run_remaining_epoch`, which computes the final AC results for the buffered survivors, and then to `process_remaining_results`, which persists the topologies and emits the corresponding push or rejection results. The worker never manipulates full-result persistence directly; its role is to decide when a survivor batch is mature enough to be handed off to the full evaluation path. This separation matters because persistence and acceptance semantics belong to the optimization domain layer, whereas the worker should stay focused on orchestration and scheduling.

## Why the Survivor Buffers Exist

The pair `survivor_topologies` and `survivor_early_results` is central to the current AC design. Together they let the worker decouple fast-failing screening from full evaluation, batch survivors according to available runner capacity, reuse the early-stage loadflows that have already been computed, and still flush partial batches after a timeout or at shutdown. Without this paired state, the design would collapse back to one of two less desirable alternatives: either every candidate would have to be fully evaluated immediately, which would waste topology-level parallelism in the expensive stage, or the early-stopping subset would have to be recomputed later, which would waste work that the first stage had already paid for.

## Related Parameters

The most important loop parameters are `runner_processes`, which determines how many topologies the remaining-contingency stage should process in parallel and therefore sets the default survivor flush batch size; `remaining_loadflow_wait_seconds`, which limits how long survivors may wait before being flushed even when the batch is not full; `runtime_seconds`, which acts as the hard runtime limit for the AC run; and `enable_ac_rejection`, which changes whether early-stage and final non-error rejections are emitted explicitly as rejections or folded into accepted push results. These parameters are not merely operational knobs; together they express the core tradeoff of the loop between search breadth, validation throughput, latency, and rejection strictness.

## Summary

`run_optimization_epochs` is the orchestration layer of AC validation.

It does not compute loadflows itself. Instead, it keeps the local repertoire in sync with DC results, pulls and screens new AC candidates, buffers survivors, flushes survivor batches into full evaluation, and sends heartbeats and stop messages. The actual scoring logic lives in the optimizer and scoring modules, while the worker remains responsible for pacing, batching, and lifecycle control.
