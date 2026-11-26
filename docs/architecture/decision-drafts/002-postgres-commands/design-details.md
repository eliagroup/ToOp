# Postgres instead of Kafka for optimizer coordination - design details

This document belongs to ADR-002 in [README.md](README.md). The ADR takes the decision and sets the high level picture.
This document describes how the approach is to be taken: the tables, the protocol of the workers, the requirements that
follow from the approach, the failure modes and the points that are still open. Nothing in here needs a decision of its
own, it is documented so that an implementation does not have to find it out again.

## Tables

The diagram shows the tables of the design and how they reference each other. The command tables replace the commands
and the heartbeat topic, the result tables replace the results topic.

![The tables of the optimizer database. Command tables: optimization_job (the immutable request), stage_work_item (one row per job and stage, polled by the workers), stage_execution_history (one row per attempt) and active_worker (one row per worker). Result tables: strategy, topology (the immutable definition of a topology), topology_work_item (the open work, one row per topology and stage) and stage_topology_result (the archive, one row per evaluated topology and stage). Arrows show the references between the tables.](images/postgres_commands_tables.png)

The command tables are three tables plus one for liveliness:

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

The result tables hold the topologies, their open work and their results:

- **Topology**: inserted when a stage discovers a topology and never modified again. It holds everything that is
  required to define the topology itself (actions, disconnections, pst setpoints, unsplit flag, a hash) but no metrics.
  Topology ids are UUIDs so they survive a wipe of the optimizer database and can still be disambiguated in the
  api-service database.
- **Strategy**: groups topologies for future multi-timestep support, but this is not fully supported at the moment.
- **TopologyWorkItem**: the queue of open work, with one row per topology and stage that still has to be evaluated.
  It holds the stage as a column, the id of the worker that claimed it (empty as long as it is unclaimed) and an
  attempt counter. A row stays in this table while it is being worked on and is deleted once the evaluation finished.
- **StageTopologyResult**: the archive, with one row per topology and stage that has been evaluated. It holds the
  outcome (ACCEPTED, WARN, REJECTED or an error) and the payload (fitness, metrics, worst-k contingency cases,
  rejection reason, a loadflow reference). It is only inserted once actual results are present and never updated
  afterwards, which allows non-null constraints such as a mandatory fitness.

There are two kinds of work items, on two levels. A StageWorkItem exists once per job and stage. For the DC stage it
is what a worker claims to run the stage, for the AC stages it is state that the workers of the job share. A
TopologyWorkItem exists once per topology and stage and is what a worker claims to evaluate that topology. The first
is created with the job and stays, the second is created when a topology is discovered or passes the previous stage
and is deleted once its result is written.

## Command side

The sequence of actions is designed as follows:

1. A new optimization job is scheduled by the user, so a row in the OptimizationJob table is created. Also, a row for
   every stage in the StageWorkItem table is created in TRIGGERED.
2. A worker from each stage finds the StageWorkItem and locks it to itself by setting the status to RUNNING and the
   lease to a date in the future. The claim is a `SELECT ... FOR UPDATE SKIP LOCKED`, ordered by job priority descending
   and `created_at` ascending so the most urgent job goes first and equal priorities stay FIFO.
3. During their work, the workers repeatedly update the lease so no other worker picks up the same job. They also check
   for cancellations during this update.
4. Workers finish their work and set their rows to COMPLETED.

For the AC stages, steps 2 to 4 differ, see sharing a stage between workers.

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

## Sharing a stage between workers

The sequence above lets one worker claim the StageWorkItem of a job. This fits the DC stage, where one worker runs the
optimization, but not the AC stages, on which several workers shall work for the same job. The AC stages therefore
work as follows.

**The StageWorkItem of an AC stage is not claimed.** It is a shared state row with the status and the start time for
the runtime limit, without a lease, an attempt counter or an owner. What an AC worker claims are TopologyWorkItems, and
it records the job it is on in its ActiveWorker row, so that others can see whether a job has a worker. As nobody owns
the stage, it has no FAILED status. If the cause of a failure lies in the job (e.g. the unsplit grid does not
converge), a worker sets the stage to BLOCKED for everyone. If the cause lies in the worker (a missing mount, no
memory), the worker ends itself and leaves the stage alone, as otherwise one broken worker could fail a job that
healthy workers are working on. The stage is COMPLETED when the DC stage has ended and no open work is left, or when
the runtime limit is reached. This is a rule that everyone who polls can apply and not the duty of one worker.

**The heartbeat of a worker is the lease for all of its topology work items.** A work item is open if it is unclaimed
or if the last heartbeat of the worker that claimed it is older than a timeout, which is the same for all workers. The
claim statement applies this rule itself, so there is no lease per work item and no process that cleans up. A lease
per work item would have to be refreshed for every held item while an evaluation runs, whereas one heartbeat per
worker covers all of them. A takeover increases the attempt counter, while a worker that stops on purpose clears its
claims without a counted attempt. The id of a worker identifies one start of a process, so a restarted worker does not
inherit claims that it does not know about. The heartbeat is sent by a background thread and keeps running when the
worker hangs, therefore a claim also becomes open once it is older than a generous maximum age. The timeout shall be
longer than the failover of the database that we plan for. After a longer outage all workers look dead and claims can
change hands, which for a topology work item costs one batch of evaluations, and we accept this.

**The ownership check is part of the write.** A worker writes a result by deleting its work item under the condition
that the item is still claimed by itself and that the stage is still RUNNING, and the result is only inserted for the
work items that were deleted. This is a single statement, so a worker whose claim was taken over or whose job was
cancelled meanwhile writes nothing and drops its result. A unique key over topology and stage on the
StageTopologyResult is the backstop. The loadflow results of a worker are stored under a path that contains its id, so
that a worker which lost its claim does not overwrite the files of the new owner.

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

- an optimization is RUNNING if there is at least one RUNNING StageWorkItem, with a valid lease where it is claimed,
- an optimization is COMPLETED when all required work items reached COMPLETED,
- an optimization is FAILED if at least one required work item is FAILED and no further retry is scheduled.

## Result side

This is the equivalent of the results topic. The lifecycle of a topology begins in a discovery stage. This is usually
the DC stage, but other stages could also discover new topologies. When that happens, a Topology is inserted into the
database and never modified again.

Stage-local persistence is then split into two concerns, the TopologyWorkItem as the queue of open work and the
StageTopologyResult as the archive of what has been evaluated.

The alternative would be to keep open and finished evaluations in one lifecycle table whose rows move from TRIGGERED
to a terminal state. We decided against this because the workers poll the open work all the time, and a table that
also keeps every finished evaluation of every past job grows without bound and makes exactly that poll slower over
time. With the split, the polled table only ever holds open work, the remaining open work of a finished or cancelled
job can be removed with a single delete and the archive is append-only.

The most naive approach would be that a stage picks up a topology, evaluates it and then inserts a result row. Here
we choose a slightly different semantic: upon topology discovery or a previous stage evaluation, a TopologyWorkItem
for the next stage is already inserted. Except for the terminal stage (AC), all stages have the responsibility to
create the work items for the next stage if they deem a topology feasible for that stage (i.e. it ended in ACCEPTED or
WARN). When a stage finishes a topology, it deletes its work item, inserts the result and inserts the work item for
the next stage in one transaction. For any topology and stage there is therefore either a work item, or a result, or
neither of them, but never both.

The following decisions were taken on the shape of the work item table:

- There is one table with a stage column and not one table per stage. The sequence of stages is not final (see the
  stages section, a dc+ stage is already being considered) and a table per stage would need a schema change for every
  new stage while behaving the same otherwise.
- Claimed work items stay in the table and are marked with the id of the worker. The alternative would be to hold the
  row lock for as long as the evaluation runs, but an AC evaluation takes long and this would keep a transaction and a
  database connection open for the entire time. No transaction shall stay open while a worker computes.
- The attempt counter is increased with every claim. Without it, a topology that reliably crashes its worker would be
  handed out again and again. Once a configured number of attempts is exceeded, the work item is not handed out again
  but moved to the results as an error.

A work item whose worker died has to be released again, which means the claim is cleared and it becomes available to
other workers. This can not be done by resetting all running evaluations of a stage together with the StageWorkItem
lease, as this would assume that there is only one worker per stage and job. It is done through the heartbeat of the
worker instead, see sharing a stage between workers.

Worst-k evaluation requires information from the previous stage - which N-1 cases were the most severe ones.
Concretely, the AC fast-failing stage requires information from the DC stage. As the worst-k cases are a denser form of
metrics, it semantically seems preferable to write them in the result entry of the stage that computed them.

### Duplicate topologies

The same topology can be found more than once, either by the same DC worker in a later epoch or by a second DC worker
on the same job, and it shall be evaluated only once per stage. The obvious solution is to check whether a result
already exists before a work item is inserted, but this is not safe. A work item that another transaction is just
moving to the results can be missed by the check, and a topology that is queued but not yet evaluated has no result
that could be found.

Instead, the Topology table is the single place that decides whether a topology is new. It has a unique key over the
job and the topology hash, rows are never deleted from it and a discovering stage inserts into it with conflicts being
ignored. Only for the rows that were actually inserted, the discovering stage writes its own result and the work item
for the next stage. This replaces the in-memory set of already sent topologies that every DC worker keeps today, which
can not see what other workers have sent.

As a consequence, if two DC workers find the same topology, the result of the second worker is dropped. We accept
this, as the result is for a topology that is already known and is not needed a second time.

## Selecting topologies for the AC stages

The AC side does not take open work in the order in which it arrived. When topologies are pulled from the DC results,
a filter strategy is applied (dominator, median and discriminator filter) and the final pick is a random sample that
is weighted by an interest score. These rules can not be stored as a state on the work item, because none of them is a
property of a single topology:

- The dominator and the median filter compare the candidates among each other. They are computed over the topologies
  that have not been pulled yet, so a dominated topology becomes eligible again once the topology dominating it has
  been pulled.
- The discriminator filter excludes candidates which are close to a topology that was already pulled, no matter
  whether that one is still being evaluated or already finished.
- If the filters remove all candidates, the unfiltered candidates are used, so a filtered topology can still be
  evaluated later on.

Therefore there is no DOMINATED state and dominated work items are not removed, they have to stay available. The
selection stays in the worker as it is today: the worker reads the candidates of its job, applies the filters and the
sampling locally and then claims exactly the selected work items with an update that only takes rows which are still
unclaimed. If another worker was faster on some of them, it works with the ones it got, and if it got none it selects
again right away. We decided against expressing the filters in SQL, as the metrics they work on are configurable and
the dominator filter depends on the order in which it iterates over the metric values.

Two requirements follow from this for the tables:

- The read for the selection has to be narrow. For the topologies of its job, the worker needs the id, whether the
  topology is open, claimed or finished, the DC fitness and the metrics that the filters work on. This read happens on
  every pull and must not load the result payloads.
- Claimed work items count as pulled for the discriminator. This is a second reason why they stay in the table.

### Several workers on the same job

As stated in the context of the ADR, several AC workers can pull for the same job. They shall not wait for each other,
so there is no lock around the selection and the only point where two workers meet is the claim, which is a single short
statement.

The price is that two workers which select at the same moment do not see each other's picks and can therefore pull
topologies which the discriminator would have kept apart. We accept that this breaks the domination and discrimination
logic a little, because a waiting AC worker is worse than a few AC evaluations that are too similar to each other. The
effect is also limited to one round, as the next selection sees the claims of the other workers and excludes their
neighbourhood again. For this to hold, the set of claimed and finished topologies has to be read fresh before every
selection, also if a worker caches the DC results locally.

There is one thing that has to change for this to work. Today all AC workers of a job seed their random generator
from the same seed in the AC parameters. With a shared set of candidates, two workers that read the same state would
draw the identical sample and all but one of them would lose the entire claim. Every worker therefore needs its own
random stream, derived from the seed of the job and the id of the worker.

## Database load

The setup we plan for is around 16 workers on 8 machines against a remote postgres. The DC stage computes a very high
number of loadflows, so the concern is that the workers overwhelm the database. The design answers this by making the
database traffic independent of the loadflow throughput:

- Loadflows never reach the database. A DC worker only writes the topologies that are new in its repertoire, once per
  epoch. This is bounded by the size of the repertoire, which with the default descriptors (5 x 45 cells, cell depth
  1) is at most 225 topologies per epoch and worker.
- The number of round trips per epoch is fixed and does not grow with the number of topologies. A DC worker writes an
  epoch in one transaction with three statements that each cover all new topologies: the insert into Topology, the DC
  results for the rows that were actually inserted and the work items for the next stage. The round trip to a remote
  database is the expensive part and the GPU idles while the worker waits for it, so a statement per topology is not
  acceptable.
- Loadflow results stay in the file storage and the database only holds a reference to them, so the rows stay small.
- No transaction stays open while a worker computes and the polled table only holds open work, for the reasons given
  in the result side section.

## Performance

To check the database load section, the design was implemented once in an experiment and measured. The load figures
come from simulated workers that issue the statements of the design at the rate of the planned setup: 8 DC and 8 AC
workers, 24 unfinished jobs, every DC worker writing 100 new topologies per epoch of one second, against a postgres 17
with 4 cores and 8 GB behind an injected network delay of 0.5 ms, for 10 minutes. All numbers are single runs. The
database was local with an artificial delay and not a remote one, no GPU was involved and no run was longer than 10
minutes, so these are first indications and not a confirmation.

| figure | value |
|---|---|
| transactions per second | 43 |
| database CPU | 0.08 of 4 cores |
| connections | 34 |
| WAL written | 1.4 MB/s |
| write of a DC epoch with 100 topologies | 13.4 ms at the median, 26 ms at p95, 189 ms at p99 |
| poll of an idle DC worker | 0.7 ms |
| job choice of an AC worker | 3.9 ms |
| read of the selection candidates (16 800 rows) | 79 ms |
| claim of 4 topologies | 10 ms |
| write of an AC result | 6.5 ms |

What we take from the experiments:

- The planned setup does not overwhelm the database. The write of an epoch takes 1.3 % of the epoch second, and 64
  workers at the same rate per worker used 0.3 cores.
- The number of topologies per epoch is what matters, not the number of workers. With 1 000 topologies per epoch on
  every DC worker the write of an epoch took 186 ms. With 5 000 it took 1.3 s, which is longer than the epoch, while the
  database wrote 15 000 topologies per second. A single write therefore needs an upper bound on its topologies.
- A write waits for the commit to be flushed to disk, which was 4 ms here and is the floor of every writing statement.
  The slow writes at p99 were all writers waiting for the same flush. Without the wait for the flush
  (`synchronous_commit = off`) the write of an epoch took 13 ms at p99.
- The network delay matters little up to 2 ms. The write of an epoch went from 12.8 ms at 0.2 ms of delay to 16.8 ms at
  2 ms and 22.9 ms at 5 ms when it was sent as a single statement, and to 45.2 ms at 5 ms when it was sent as a
  transaction with three statements.
- Storage is the larger cost. A DC topology with its result and its work item took about 1.5 kB of tables and indexes
  and 2 kB of WAL, which at the planned rate is 1 MB/s and in the order of 55 to 85 GB per day (extrapolated from 10
  minutes), while the AC stages evaluated about 0.1 % of these topologies. Removing a job of 18 000 topologies took
  0.4 s.
- The AC workers are bound by their computation and not by the database. On a grid with 1354 buses, on which one N-1
  analysis takes 12.6 s, a fast-failing evaluation took 3.5 s per topology and a full AC evaluation 25 to 45 s. In one
  job of 5 minutes with one DC and two AC workers, all statements of the workers together took 0.14 s in the database.
- Switching to a new job took 29 s on that grid and 51 s when two workers loaded it at the same time, not the 10 s
  assumed in the context of the ADR, and a worker held 10 GiB after loading and up to 20 GiB at the peak.

## Stages

The basic setup of the optimization becomes a staged run where first dc, then maybe dc+, then fast-failing ac and
finally ac review the same topology. `OptimizerType` therefore gains an `AC_FAST_FAILING` member (an ac based stage
which only computes a subset of N-1 cases to quickly assess topologies) next to `DC` and `AC`, and becomes a str enum
so it can be stored as a column.

## Failure modes

This section lists what can go wrong and what the design does about it. "Handled" means that the sections above define
the behaviour, "accepted" means that we know the cost and take it, and "open" means that the behaviour is not defined
yet. The open ones are part of the open points.

### Worker failures

1. **A worker fails gracefully.** It can still write to the database, deletes its lease and sets the stage to FAILED or
   BLOCKED. *Handled*, see failures, leases and retries.
2. **A worker dies while it holds a stage** (a segfault, a kill, the loss of the machine). The lease expires, and the
   next worker that polls increases the attempt counter and takes the stage over, or sets it to FAILED once the
   threshold is exceeded. *Handled.* The cost is that the state of a DC optimization lives in the memory of the worker,
   so the new attempt starts from scratch. *Open* is whether it gets the full runtime again or what is left of it.
3. **A worker dies while it holds topology work items.** Its heartbeat stops, and after the timeout its work items are
   open for other workers, who increase the attempt counter when they take them. *Handled*, see sharing a stage between
   workers.
4. **A worker hangs**, so the process is alive but makes no progress. *Handled.* The lease of a stage is refreshed by
   the loop that does the work and expires. The heartbeat comes from a background thread and keeps running, so for
   topology work items the maximum age of a claim opens them.
5. **A healthy worker exceeds its lease**, because an epoch or an evaluation took too long. Either nobody took the work
   and it keeps working, or another worker took over and it has to drop the work. The ownership check on every update
   decides which of the two it is. *Handled.*
6. **A worker is stopped on purpose** (a deployment, a rolling restart). For topology work items this is *handled*, the
   worker clears its claims without a counted attempt. For the DC stage it can delete its lease as in 1, but it is
   *open* whether this counts as an attempt. If it does, a few restarts in a row fail a healthy job with the attempt
   limit. A restart of all DC workers also costs every running DC stage what it has computed so far.
7. **A topology reliably crashes its worker.** The attempt counter of the work item limits how often it is handed out,
   afterwards it is moved to the results as an error. *Handled.*
8. **An evaluation raises an error without killing the worker.** *Open*: whether this counts as an attempt of the
   topology, and what happens to the other topologies that were claimed together with it. If the whole batch is charged,
   one bad topology turns the others into errors as well.
9. **A stage fails deterministically** (e.g. non-converging initial loadflows). The stage is set to BLOCKED and never
   picked up again. *Handled.* The worker has to tell a deterministic from a transient failure, and from an exception it
   often can not. A wrong guess either retries something that can not succeed or blocks a job that could run. *Open*:
   which failures a worker reports as BLOCKED.
10. **The cause of a failure is the worker and not the job** (a missing grid mount, no memory left, a full disk for the
    loadflow results). On a stage that several workers share, a worker that reports such a failure for the job takes the
    job away from the healthy workers, and one worker with an unreadable mount can fail every job of the queue.
    *Handled*: such a failure ends the worker and leaves the stage alone. *Open* is how a worker that is restarted
    declines a job it can not load, as it is offered the same job again.
11. **No worker picks up a stage**, because no worker of that kind is running or all of them are busy for a long time.
    The work item stays in TRIGGERED. *Open*: the kafka workers dropped commands that were too old, this has no
    counterpart yet.

### Database connection failures

12. **The database is unreachable for a short time.** The kafka producer buffers locally, postgres does not. *Open* (see
    the open points): a worker has to repeat its calls and keep its results in memory meanwhile, and for how long it
    does so is not defined.
13. **The database is unreachable for longer than the lease** (e.g. a failover). All leases expire, also those of
    healthy workers, and the first worker that reaches the database again takes over stages whose owners are still
    working on them. For topology work items this is *accepted*, it costs one batch of evaluations. For the DC stage it
    is *open* (see the open points): either the lease is longer than the failover we plan for, or a takeover needs more
    evidence than an expired lease.
14. **A response is lost.** The statement was committed, but the connection broke before the worker got the answer. The
    worker then does not know whether its claim or its write happened. *Open*: every call has to be safe to repeat, and
    a claim has to return what the caller holds already. Otherwise a row stays claimed by a worker that does not know
    about it.
15. **A call never returns**, because the connection is silent. The postgres client has no timeout for a response by
    default. *Open*: every call needs a deadline. Without one the worker hangs in the call, which then is failure mode
    4.
16. **The database answers with a lasting error** (a full disk, a statement that is too large or runs into a timeout).
    For the worker this looks like failure mode 12 on every repetition. *Open.* The size of a statement has to be
    bounded, as the 225 topologies per epoch are the default and not a limit.

### Concurrency

17. **Two workers poll the same stage work item.** `SKIP LOCKED` hands it to one of them. *Handled.*
18. **Two AC workers select the same topologies.** The claim only takes rows which are still unclaimed, and the slower
    worker works with what it got or selects again. *Handled.* That the two workers can pull topologies which the
    discriminator would have kept apart is *accepted*.
19. **The same topology is discovered twice.** The unique key of the Topology table decides, and the result of the
    second worker is dropped. *Handled* and *accepted*.
20. **An update arrives after the cancellation.** It does not revive the stage and only finalizes the history row. A
    result that arrives after the cancellation is not written, as the write checks that the stage is still RUNNING.
    *Handled.*
21. **A worker that lost its claim on a topology writes a result.** The write only takes work items that are still
    claimed by the worker, so it writes nothing. *Handled*, see sharing a stage between workers.
22. **The api service fails while it starts a job.** The job and its stage work items have to be inserted in one
    transaction, otherwise a job without work items exists that is never picked up. This is a requirement on the api
    service.
23. **A reader of the results misses rows.** With several writers, rows become visible in the order of their commits and
    not of their ids, so a reader that asks for everything after the last id it has seen can skip a row. *Open*: how the
    api service reads the results incrementally is not described yet.

### Jobs and stages

24. **All AC workers leave a job and none comes back**, or the DC stage ends without a single topology. The completion
    of a shared stage is a rule that everyone who polls can apply, so another AC worker completes it. *Open*: who
    applies the rule when no AC worker is running.
25. **A higher priority job preempts the workers of a job.** The context of the ADR says when a worker leaves its job,
    but not where it goes. If all workers move to the most urgent job, the job that triggered the preemption can stay
    without a worker. *Open.*
26. **The runtime limit is reached while an evaluation runs.** *Open*: whether a result that arrives after the limit is
    still accepted. A full AC evaluation takes tens of seconds on many cores, so throwing it away is a real cost.
27. **A stage is set to FAILED.** FAILED means that we would like to attempt a restart, but a stage is only set to
    FAILED once its attempts are used up. *Open*: whether anything restarts a FAILED stage. If not, FAILED and BLOCKED
    are the same.

### Storage and operations

28. **The database fills up.** Every topology and every result stays in the tables. *Open* (see the open points on
    retention). The performance section gives a first estimate of the growth.
29. **A polled statement gets slow.** The polls are cheap as long as postgres picks the plan we had in mind. A plan
    that is right for the jobs it knows can be wrong for a job that the planner statistics do not know yet, which is
    every new job. *Open*: checks of the plans on tables without statistics belong to the confirmation.
30. **A loadflow result and its row get out of step.** The files are written to the file storage and the row holds a
    reference to them. If the worker dies or loses its claim in between, the files stay without a row. *Open*: who
    removes them.
31. **The clocks of the workers differ.** Leases and runtime limits have to be compared on the clock of the database.
    Otherwise a worker with a wrong clock takes over healthy work or keeps its own forever. This is a requirement on the
    statements.

## Open points

- Multi-timestep is modelled (Strategy) but not supported, we currently only run single-timestep.
- Where the database lives in the deployment and who runs migrations is not decided.
- Who completes a shared stage when no AC worker is running, and how a worker declines a job that it can not load, is
  not decided.
- User-triggered AC validation jobs bring topologies that have no DC result. How they enter the work item table and
  how they pass the selection, which works on DC metrics, is not decided.
- Whether the full AC stage takes the accepted work items of the fast-failing stage in order of priority or also runs
  a selection on them is not decided.
- For a stage that selects from candidates, the open work could be derived (every DC result without a claim or result
  for that stage) and would not have to be inserted by the DC worker for every topology. This would remove one of the
  three statements per epoch and most rows of the work item table, but is not decided.
- Where the metrics for the selection live is not decided. As the metric names are configurable, this is probably a
  small json column that holds only these values.
- The number of topologies a DC worker actually pushes per epoch and the duration of an epoch have not been measured,
  the 225 above is only an upper bound. The DC worker logs the number of sent strategies for every epoch.
- How many database connections a worker process may hold (including the processes an AC worker starts) and whether
  there is a connection pooler in front of the database is not decided.
- How long the Topology and StageTopologyResult rows of finished jobs are kept is not decided.
- The kafka producer buffers locally if the broker is not reachable, postgres does not. What a worker does if the
  database is unreachable for some time, in which its lease can expire although it is healthy, is not defined.
- Topologies are deduplicated per job. Whether a re-run on the same grid and action set should reuse earlier AC results
  is not decided.
- The failure modes that are marked as open in the failure modes section.
