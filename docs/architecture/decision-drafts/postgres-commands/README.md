# Postgres instead of Kafka for optimizer coordination

State: draft
Author: Nico Westerbeck
Opened: 03.06.2026

This document describes how the kafka topics between the api service and the optimizer workers (commands, results,
heartbeats) could be replaced by a postgres database. The example code lives in
[PR #516](https://github.com/eliagroup/ToOp/pull/516) (`refactor/postgres-commands`), the diagrams are in
`postgres-commands.c4` next to this file and show up in the architecture app as `postgresCommands` and
`postgresCommandsLifecycle`. The as-is counterpart is the `optimizerCoordination` view.

## Motivation

Currently, the api service sends a `StartOptimizationCommand` to the commands topic, both stages consume it, push
topologies to the shared results topic and report liveliness through heartbeats. The state of an optimization does not
live anywhere, it has to be reconstructed from the heartbeat and result streams. That makes retries, cancellation and
"what is this job doing right now" questions harder than they should be.

Plainly put, kafka is not a priority work queue manager.

With a database, the request payload, the current per-stage work item and the historical execution record are kept
separate, so each concern can evolve without overloading a single table with conflicting responsibilities. The database
then also replaces the heartbeat topic (workers update their own row) and the results topic (topologies and their
per-stage evaluations become rows). Note that this proposal only covers the optimizer side, the importer topics are not
touched yet.

## Command side

Three tables plus one for liveliness:

- **OptimizationJob**: the immutable optimization request, i.e. the durable replacement for the old start command
  payload. Holds grid files, dc and ac parameters (as typed json columns), priority and `created_at`. Write-once by the
  api service, anchor id for everything else. There is deliberately no aggregate status field on this table, see below.
- **StageWorkItem**: one row per optimizer stage and job. It persists across pending, running, retry and terminal states
  and is the queue surface that workers poll with `SKIP LOCKED`. Only the hot-path state lives here: status, the current
  attempt counter and the lease expiry.
- **StageExecutionHistory**: one append-only row per attempt. Audit trail, not part of scheduling. This is roughly the
  equivalent of the optimization stats heartbeat in the kafka architecture, so it also carries the basic KPIs
  (loadflows computed, topologies checked, epoch, iteration) and the terminal error message.
- **ActiveWorker**: every worker periodically updates its own row with a heartbeat timestamp and the number of visible
  GPUs/CPUs. The worker id is the same as in the execution history, so a join reveals all executions a worker performed.

The sequence of actions is designed as follows:

1. A new optimization job is scheduled by the user, so a row in the OptimizationJob table is created. Also, a row for
   every stage in the StageWorkItem table is created in TRIGGERED.
2. A worker from each stage finds the StageWorkItem and locks it to itself by setting the status to RUNNING and the
   lease to a date in the future. The claim is a `SELECT ... FOR UPDATE SKIP LOCKED`, ordered by job priority descending
   and `created_at` ascending so the most urgent job goes first and equal priorities stay FIFO.
3. During their work, the workers repeatedly update the lease so no other worker picks up the same job. They also check
   for cancellations during this update.
4. Workers finish their work and set their rows to COMPLETED.

## Failures, leases and retries

We distinguish two types of failure: transient and deterministic failures. FAILED means we do not know exactly what
went wrong and would like to attempt a restart. BLOCKED means we know that a retry will not solve the problem (e.g.
non-converging initial loadflows) and the stage shall never be picked up again.

If a worker fails gracefully (can still write the db), it deletes the lease. If it can not write anymore e.g. due to a
segfault, the lease will just expire. Future workers will see the row, which is in RUNNING state without a valid lease
(SKIP LOCKED will only show it once the database session was killed). They will increase the attempt counter and

- if the attempt counter is less than a configured threshold, reacquire the lease on themselves and keep working.
- if the attempt threshold is exceeded, set the stage to FAILED.

If a worker took very long during an epoch and exceeded the lease, two things might happen: either the job has not yet
been picked up - in this case it can keep working, or another worker took over the stage and it has to drop all work
and assume the job is no longer allocated to itself. The ownership check happens on every update: the history row of
the current attempt has to belong to the updating worker, otherwise the update is refused.

## Cancellation

To cancel a run, the user sets every stage work item to CANCELLED upon which the workers will exit on their next
update. Note that setting this from None to CANCELLED is the desired flow, but once it was set and has propagated, this
should never be taken back - otherwise the job could end up in a state where some stages saw the cancellation and
dropped work while others still continue working. An update arriving after the cancellation does not revive the stage,
it only finalizes the current history row as CANCELLED.

## Job status is inferred

The optimization job purposely does not hold a mutable aggregate status field. Doing so would introduce a second
writable source of truth next to the stage-level tables and would make it too easy to create inconsistencies during
retries, cancellation or worker crashes. Instead, job status is inferred from StageWorkItem and StageExecutionHistory:

- an optimization is RUNNING if there is at least one RUNNING StageWorkItem with a valid lease,
- an optimization is COMPLETED when all required work items reached COMPLETED,
- an optimization is FAILED if at least one required work item is FAILED and no further retry is scheduled.

## Result side

This is the equivalent of the results topic. The lifecycle of a topology begins in a discovery stage. This is usually
the DC stage, but other stages could also discover new topologies. When that happens, a **Topology** is inserted into
the database and never modified again. It holds everything that is required to define the topology itself (actions,
disconnections, pst setpoints, unsplit flag, a hash) but no metrics. Topology ids are UUIDs so they survive a wipe of
the optimizer database and can still be disambiguated in the api-service database. For future multi-timestep support,
topologies are grouped into a **Strategy**, but this is not fully supported at the moment.

Stage-local persistence is then split into two concerns:

- **StageTopologyEvaluation** is the mutable coordination row for one stage and one topology. It is the scheduling,
  locking and lifecycle surface and moves TRIGGERED → RUNNING → ACCEPTED / WARN / REJECTED.
- **StageTopologyResult** is the payload row (fitness, metrics, worst-k contingency cases, rejection reason, a
  loadflow reference) that is only inserted once actual results are present. This keeps the lifecycle row small and
  allows non-null constraints such as a mandatory fitness.

The most naive approach would be that a stage picks up a topology, evaluates it and then inserts a new lifecycle row
and a result row. Here we choose a slightly different semantic: upon topology discovery or a previous stage evaluation,
a StageTopologyEvaluation for the next stage is already inserted with a TRIGGERED flag. Except for the terminal stage
(AC), all stages have the responsibility to create the evaluation rows for the next stage if they deem a topology
feasible for that stage (i.e. it ended in ACCEPTED or WARN). Stage workers are free to use the table as an additional
synchronization mechanism for in-worker parallelism.

Worst-k evaluation requires information from the previous stage - which N-1 cases were the most severe ones.
Concretely, the AC fast-failing stage requires information from the DC stage. As the worst-k cases are a denser form of
metrics, it semantically seems preferable to write them in the result entry of the stage that computed them.

In contrast to StageWorkItems, StageTopologyEvaluations do not carry a lease. This makes a recovery of individual
failed worker threads impossible. Instead, we assume that if a stage worker fails then all worker threads from the job
fail. A cleanup routine resets everything that is in RUNNING back to TRIGGERED. This must happen in a transaction
together with the reset of the StageWorkItem, otherwise there would be a race condition with a too short lease time
where a worker would still be working and trying to write results. As the workers check the StageWorkItem for a
possible termination (cancellation or lease expiry) __before__ writing results, this is remedied.

## Stages

The basic setup of the optimization becomes a staged run where first dc, then maybe dc+, then fast-failing ac and
finally ac review the same topology. `OptimizerType` therefore gains an `AC_FAST_FAILING` member (an ac based stage
which only computes a subset of N-1 cases to quickly assess topologies) next to `DC` and `AC`, and becomes a str enum
so it can be stored as a column.

## The example code

Everything lives in `toop_engine_topology_optimizer.database` in the PR:

- `command_models.py` and `result_models.py` are the SQLModel tables described above.
- `json_adapter.py` provides `TypedJson`, a sqlalchemy type decorator that dumps a pydantic type into a json column and
  validates it back on load. This is what lets the parameter objects and `StoredLoadflowReference` be stored as-is.
- `user_utils.py` is the api-service side: `start_optimization` inserts the job plus one TRIGGERED work item per
  stage, `cancel_optimization` locks and cancels all work items of a job.
- `utils.py` is the worker side: `poll_stage_work_item` claims the next eligible row (fresh, or running with an expired
  lease, retiring it as FAILED if `max_retries` is reached) and writes the history row, `update_stage_work_item`
  refreshes the lease, stores KPIs and moves to terminal states, honoring cancellation.

All utilities expect a session with `autobegin=False` and own their transaction boundaries, so locking happens inside
predictable transactions. The tests spin up a `postgres:15` container through the docker sdk, so they need docker on
the test machine.

## Open points

- The `infer_status` helper for the job status is referenced but not written yet, same for the cleanup routine of
  stale evaluations and the worker-side utilities of the result tables.
- Nothing is wired into the workers yet, the PR only contains the schema and the queue utilities with tests.
- The heartbeat replacement (ActiveWorker) has no utilities and no consumer yet.
- Multi-timestep is modelled (Strategy) but not supported, we currently only run single-timestep.
- Where the database lives in the deployment and who runs migrations is not decided.
