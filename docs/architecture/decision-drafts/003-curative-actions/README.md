---
title: "ADR-003: Implementation vision of curative actions"
status: Proposed
date: 2026-06-15
decision-makers: ["Nico Westerbeck"]
consulted: []
informed: []
---

# ADR-003: Implementation vision of curative actions

## Context and Problem Statement

This document describes how curative actions are to be implemented in ToOp.

The first implementation phase of curative actions would be an AC bruteforce stage, which requires a very short list of
potential curative action candidates. If we assume in the order of 10 actions per CO/CB, it would be feasible to just
brute-force all curative actions in AC after the preventative topology application.

There are two difficult decisions currently foreseen in the implementation

- How to store the action sets?
- How to integrate this into the contingency lib?

The options below are for the storage of the action set. The integration into the contingency lib is described in the
decision outcome and is not decided yet.

## Decision Drivers

* We want a very flexible action, as curative actions can consist of reassignments + coupler openings, disconnections,
  pst tap changes, active power changes and every combination of them.
* Combinations have to be part of the action set. ToOp currently does not have a concept of global actions in DC or any
  of its data structures, and for the AC side we can not afford to let the bruteforce loop find a combination, e.g.
  splitting the higher and lower voltage in parallel.
* We'd like to be compatible to DC, as the DC side might have to subselect/assemble the action set for AC.
* For the AC bruteforce to work in time, a relatively small action set is required.
* TSOs differ. Some have a notion of primary and secondary curative actions, others purely need only primary actions. (A secondary curative action is a different action from the primary that also manages to secure the grid, in case the primary action can not be implemented for some reason.)

## Considered Options

* Node/breaker actions
* ToOp actions

## Decision Outcome

Chosen option: "Node/breaker actions", because given the current vision it seems the more reasonable path forward. It is
more flexible, cleaner and has only one level of list.

The implementation is planned in three phases.

### Phase 1 - AC bruteforce

The curative actions of the action set are brute-forced in AC after the preventative topology application.

#### How to integrate into the contingency lib

The current contingency lib is mainly evaluation focussed. However, with the curative action feature, an optimization task
would join the list. At the end, we want every entry in the curative action set to be evaluated and ranked for the CO/CBs
that are remaining after the first preventative evaluation. The information we would like to obtain is, for each curative
action, how much overloads can I save.

Powsybl already contains an operator strategy logic which could be re-used for this purpose, but in theory we would like a
logic that

- performs the preventative Static Security Analysis (SSA)
- For remaining violations, performs the curative bruteforce.

This could be obtained through the operator actions interface, encoding violating N-1 cases and curative actions as operator
actions. Alternatively, it might be a thought whether we would like to integrate some logic into the java side of powsybl.

Here shall be noted the difference between different TSOs. Some have a notion of primary and secondary curative actions,
where a secondary action needs to be always available in case implementing the primary action fails (e.g. the switch is
defect). Other TSOs purely need only primary actions.

### Phase 2 - DC search

For the AC bruteforce to work in time, a relatively small action set is required. No combinatorics can happen on AC side
nor can the action set exceed more than maybe 10 actions for 10 CO/CBs. Hence, we might end up in a situation where the DC
stage should find combinations or preselect the curative action set for the AC validation

### Phase 3 - Checkpoint logic

Not in the first version, but in a later implementation of the AC stage we would like to include a checkpoint logic for
assessing intermediate states of curative actions. We separate the action into *checkpoints* which have an electric effect
on the grid and would like to know if after every intermediate checkpoint, there is still a secondary action that is
effective.

## Consequences

* Good, because an action can freely combine switch states, PST states and generator deltas, so combinations are part
  of the action set and do not have to be found by the bruteforce loop.
* Bad, because it works only on node/breaker grids.
* Bad, because the action set is not expressed in the data structures of the DC side, while something like that might
  be required in DC anyway if we ever want to get curative actions into DC.
* Bad, because the contingency lib, which is mainly evaluation focussed, gets an optimization task.
* Neutral, because the action set has to stay small until the DC stage can find combinations or preselect the action
  set.

## Confirmation

Not defined yet.

## Benefits and Drawbacks of each option

### Node/breaker actions

- Define an abstract action concept where each action can be
  - switch-id/target state
  - PST-id/target state
  - generator-id/delta
- Then, create an action set that has a combination of these actions for every CO/CB

* Good, because it is more flexible.
* Good, because it is cleaner and has only one level of list.
* Bad, because it works only on node/breaker grids.

### ToOp actions

- Use the existing concept of
  - Substation splits as Asset Topologies
  - Disconnectable branches as GridElement/index
  - PST setpoints as in the results (a setpoint for every PST)
  - Newly defined generator delta
- Then, define an abstract action that is a list of combinations of the above
- Then, create an action set that has a combination of these actions for every CO/CB

* Good, because it is easier to implement using pre-existing data structures.
* Good, because something like this might be required in DC anyway if we ever want to get curative actions into DC.
* Bad, because it is less flexible and needs a second level of list for the combinations.

## Review Date

Not set yet. The ADR is still proposed, the review date defaults to one year after it has been accepted.

## References

None yet.
