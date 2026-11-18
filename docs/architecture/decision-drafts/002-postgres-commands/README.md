---
title: "ADR-002: Postgres instead of Kafka for optimizer coordination"
status: Proposed
date: 2026-06-03
decision-makers: []
consulted: []
informed: []
---

# ADR-002: Postgres instead of Kafka for optimizer coordination

## Context and Problem Statement

This document describes how the kafka topics between the api service and the optimizer workers (commands, results,
heartbeats) could be replaced by a postgres database. The diagrams are in `postgres-commands.c4` next to this file and
show up in the architecture app as `postgresCommands`, `postgresCommandsTables` and `postgresCommandsLifecycle`. The
as-is counterpart is the `optimizerCoordination` view. The proposal was opened on 03.06.2026 by Nico Westerbeck.

Currently, the api service sends a `StartOptimizationCommand` to the commands topic, both stages consume it, push
topologies to the shared results topic and report liveliness through heartbeats. The state of an optimization does not
live anywhere, it has to be reconstructed from the heartbeat and result streams. That makes retries, cancellation,
high-priority jobs and "what is this job doing right now" questions harder than they should be.

Plainly put, kafka is not a priority work queue manager.

A further point of motivation is that currently, to-be-evaluated topologies only come from the DC solver. However, we
want to be able to accommodate AC validation jobs triggered by the user which should run with highest priority but use
the same worker set. This can be either custom topologies that the user designed by hand or a topology that is to be
evaluated in more timesteps.

In all of these, the work queue shall serve as an orchestrator for process parallelism. It shall be possible to have

- Single processes executing multiple work items at once (e.g. worst-k analysis where all k evaluations happen in
  parallel)
- Multiple workers evaluating on different optimization jobs (DC)
- Multiple workers evaluating on the same or different optimization job (AC)

The default setting for AC workers should be that they minimize job context switches. Switching to a new job means
reloading a grid file (ca 10s) and holding a lot of gridfiles in memory should be avoided due to the size of these
grids. So once they committed to a job, there should be an incentive to stay on them. Only reason to change would be if
a higher priority job came in that does not have a worker yet or if the current job does not have any topologies left at
the moment. If a job is abandoned due to these reasons, the time counter on it shall freeze and a worker shall pick it
back up at a later point and unfreeze the counter.

The question is therefore what shall coordinate the optimizer workers in place of the kafka topics.

## Decision Drivers

* The state of an optimization shall live in one place where it can be queried, so that retries, cancellation and "what
  is this job doing right now" do not need a reconstruction from streams.
* Jobs have priorities. AC validation jobs that the user triggers shall run with highest priority on the same worker set
  as the optimizations.
* The work queue shall orchestrate process parallelism: a single process executing multiple work items at once, multiple
  AC workers on different jobs and multiple AC workers on the same job.
* AC workers shall minimize job context switches, because a switch means reloading a grid file and the grids are large
  in memory.
* AC workers of the same job shall not wait for each other.
* Our deployment has a postgres database, but no rabbitmq or other message queue.
* Around 16 workers on 8 machines against a remote postgres shall not overwhelm the database, although the DC stage
  computes a very high number of loadflows.

## Considered Options

* Postgres tables with row locks as work queue, state and result store
* Keep the kafka topics
* Kafka share groups (Queues for Kafka)
* A dedicated queueing system

## Decision Outcome

Chosen option: "Postgres tables with row locks as work queue, state and result store", because it is the only option
that gives the state of an optimization a place to live, serves priorities and works with what our deployment has.

We stay on postgres for this and do not introduce a dedicated queueing system. Our deployment does not have rabbitmq or
another message queue available, while a postgres database is already there. 

Kafka share groups would take over the claiming and the redelivery of work, but not the selection among the open
topologies, the priorities or the state of an optimization, so a database would still be needed next to them.

With a database, the request payload, the current per-stage work item and the historical execution record are kept
separate, so each concern can evolve without overloading a single table with conflicting responsibilities. The database
then also replaces the heartbeat topic (workers update their own row) and the results topic (topologies and their
per-stage evaluations become rows). Note that this proposal only covers the optimizer side, the importer topics are not
touched yet but shall be reworked accordingly.

The picture below shows the tables of the design and how they reference each other.

![The tables of the optimizer database. Command tables: optimization_job (the immutable request), stage_work_item (one row per job and stage, polled by the workers), stage_execution_history (one row per attempt) and active_worker (one row per worker). Result tables: strategy, topology (the immutable definition of a topology), topology_work_item (the open work, one row per topology and stage) and stage_topology_result (the archive, one row per evaluated topology and stage). Arrows show the references between the tables.](images/postgres_commands_tables.png)

- The command tables replace the commands and the heartbeat topic: OptimizationJob is the immutable request,
  StageWorkItem holds the state of one stage of a job, StageExecutionHistory is the audit trail with one row per attempt
  and ActiveWorker holds the heartbeat of every worker.
- The result tables replace the results topic: Topology is the immutable definition of a topology, TopologyWorkItem is
  the queue of open work with one row per topology and stage, and StageTopologyResult is the archive with one row per
  topology and stage that has been evaluated.

The design in short:

- Workers poll the database and claim work with `SELECT ... FOR UPDATE SKIP LOCKED`, ordered by the priority of the job
  and then by its age.
- There are work items on two levels. The DC stage of a job is claimed by one worker through its StageWorkItem and a
  lease. The AC stages of a job may be shared by several workers, which claim TopologyWorkItems.
- A worker that dies loses its claims, either because the lease of its stage expires or because its heartbeat stops.
  Every write checks that the work still belongs to the worker that writes.
- Open work and finished results are separate tables, so the table that is polled only holds open work and the archive
  is append-only. A stage that finishes a topology inserts the work item for the next stage.
- The status of a job is inferred from its stages, there is no aggregate status that could get out of step. A
  cancellation is set on the stages and never taken back.
- Failures are either transient (FAILED, retried up to an attempt limit) or deterministic (BLOCKED, never picked up
  again).
- Whether a topology is new is decided by a unique key in the database, across all workers of a job.
- The selection of topologies for the AC stages (dominator, median and discriminator filter, weighted sampling) happens in a transaction.
- An optimization is a staged run of dc, maybe dc+, fast-failing ac and ac on the same topology.
- Optimization is the first refactor target, import to follow later.

How this works in detail, which requirements follow from it, which failure modes we see and what is still open is
documented in the [design details](design-details.md).

### Performance

The main concern against this option is the load on the database. In initial experiments, simulated workers issued the
statements of the design at the rate of the planned setup (8 DC and 8 AC workers, 24 unfinished jobs, 100 new topologies
per DC worker and second) against a postgres 17 with 4 cores behind a network delay of 0.5 ms. The database handled this
with 43 transactions per second on 0.08 of its 4 cores, and the write of a DC epoch took 13 ms at the median. The larger
cost is storage, which was about 1.5 kB of tables and 2 kB of WAL per DC topology, meaning cleanup/housekeeping might need to be relatively aggressive. The setup and all figures are in the [design details](design-details.md#performance).

## Consequences

* Good, because the state of an optimization lives in the database and does not have to be reconstructed from the
  heartbeat and result streams. Retries, cancellation, high-priority jobs and "what is this job doing right now" become
  queries.
* Good, because AC validation jobs that the user triggers can run with highest priority on the same worker set.
* Good, because duplicate topologies are detected across workers by a unique key, which is helpful for orchestrating multiple workers on the same job.
* Bad, because the failure handling is ours to define. Kafka brought an answer for an unreachable broker, a dead
  consumer and a message that is delivered twice. With plain tables and row locks every one of these cases has to be
  designed, see the [failure modes](design-details.md#failure-modes).
* Bad, because postgres does not buffer when it is not reachable, so a worker has to cope with an outage itself.
* Bad, because two AC workers that select at the same moment break the domination and discrimination logic a little. We
  accept this.
* Bad, because every topology and every result stays in the database, so retention and cleanup have to be decided and
  operated.

## Confirmation

The queue depends on the locking behaviour of postgres, which a mock does not show. The implementation is therefore
confirmed by tests that run against a real postgres.

What confirms the decision as a whole is not defined yet. We see two candidates: an end-to-end test in which the DC and
the AC worker run a job through the database, including a killed worker, a cancellation and an outage of the database,
and a load measurement that checks the database load against the planned setup, as outlined in the performance
section.

## Benefits and Drawbacks of each option

### Postgres tables with row locks as work queue, state and result store

* Good, because the state of an optimization, the open work and the results live in one place that can be queried.
* Good, because the claim orders by job priority and `created_at`, which is what a priority work queue needs.
* Good, because a postgres database is already in our deployment.
* Neutral, because everything has to work with plain tables and row locks, there are no queue features to rely on.
* Neutral, because the database load is the main concern against this option, see the performance section.
* Bad, because postgres does not buffer when it is not reachable, while the kafka producer does.
* Bad, because the failure handling has to be designed by us, see the [failure modes](design-details.md#failure-modes).

### Keep the kafka topics

* Good, because it exists and nothing has to be reworked.
* Good, because the kafka producer buffers locally if the broker is not reachable.
* Bad, because the state of an optimization does not live anywhere and has to be reconstructed from the heartbeat and
  result streams.
* Bad, because kafka is not a priority work queue manager, so a validation job of the user can not overtake the running
  optimizations.
* Bad, because every DC worker keeps its own set of sent topologies in memory and can not see what other workers have
  sent.

### Kafka share groups (Queues for Kafka)

Share groups (KIP-932, production-ready since Apache Kafka 4.2) let several consumers take records from the same topic.
Every record is acknowledged on its own, is delivered again if the consumer does not acknowledge it within a lock
duration, and is given up after a number of delivery attempts.

* Good, because claiming, redelivery after a dead consumer and an attempt limit come with the broker (a lock of 30 s by
  default, 5 delivery attempts by default). These are parts of the failure handling that we have to design ourselves on
  postgres.
* Good, because kafka is already in our deployment and the number of consumers is not limited by the number of
  partitions.
* Neutral, because delivery is at-least-once without an ordering guarantee and the delivery count is approximate, so a
  worker has to cope with work that is handed out twice, as with the other options.
* Bad, because a consumer gets the next available records and can not choose among them, while the AC stages select from
  all open topologies of a job with filters that compare the candidates with each other.
* Bad, because the state of an optimization and the results would still not live anywhere that can be queried. A
  database is needed next to it, also for the stickiness of AC workers to a job, the detection of duplicates and the
  cancellation, so we would operate two systems for the coordination.
* Bad, because we found no priorities for records in share groups, so a validation job of the user could only overtake
  through a topic of its own.
* Bad, because the lock on a record is limited to 60 s by default and to one hour at most, while a DC stage runs for
  minutes. The broker can renew a lock since 4.2, but the share consumer of the python client that we use
  (confluent-kafka) does not list this, is a preview since version 2.15.0 and is not recommended for production.

### A dedicated queueing system

* Good, because a queueing system is built for priority work queues.
* Neutral, because it would replace the commands topic, while the results and the state of a job would still need a
  store next to it.
* Bad, because our deployment does not have rabbitmq or another message queue available, while a postgres database is
  already there.

## Review Date

Not set yet. The ADR is still proposed, the review date defaults to one year after it has been accepted.

## References

* [Design details](design-details.md) of this ADR: tables, protocol, requirements, failure modes and open points.
* [KIP-932: Queues for Kafka](https://cwiki.apache.org/confluence/display/KAFKA/KIP-932%3A+Queues+for+Kafka), the
  design of share groups.
* [Broker configuration of Apache Kafka 4.2](https://kafka.apache.org/42/configuration/broker-configs/), for the limits
  of share groups (`group.share.*`).
* [Release notes of confluent-kafka-python 2.15.0](https://github.com/confluentinc/confluent-kafka-python/releases/tag/v2.15.0),
  for the state of the share consumer in the python client.
* `postgres-commands.c4` next to this file, with the views `postgresCommands`, `postgresCommandsTables` and
  `postgresCommandsLifecycle` in the architecture app. The as-is counterpart is the `optimizerCoordination` view.
