# Implementation vision of curative actions

This document describes how curative actions are to be implemented in ToOp.

# Phase 1 - AC bruteforce

The first implementation phase of curative actions would be an AC bruteforce stage, which requires a very short list of
potential curative action candidates. If we assume in the order of 10 actions per CO/CB, it would be feasible to just 
brute-force all curative actions in AC after the prevenative topology application. 

There are two difficult decisions currently foreseen in the implementation
- How to store the action sets?
- How to integrate this into the contingency lib?

## How to store the action set

The storage format for the action set needs to balance between two considerations. We want a very flexible action but at the
same time we'd like to be compatible to DC as the DC side might have to subselect/assemble the action set for AC.

Flexibility is required as curative actions can consist of reassignments + coupler openings, disconnections, pst tap changes, 
active power changes and every combination of them. The combinations feature is especially problematic as ToOp currently does 
not have a concept of global actions in DC or any of its data structures. For example, it might be required to split the
higher and lower voltage in parallel, and for the AC side we can not affort to let the bruteforce loop find this combination.

We currently see two possible ways how to implement this storage

**Node/breaker actions**

- Define an abstract action concept where each action can be 
    - switch-id/target state
    - PST-id/target state
    - generator-id/delta
- Then, create an action set that has a combination of these actions for every CO/CB

Discussion: More flexible, cleaner, only one level of list, works only on node/breaker grids

**ToOp Actions**

- Use the existing concept of 
    - Substation splits as Asset Topologies
    - Disconnectable branches as GridElement/index
    - PST setpoints as in the results (a setpoint for every PST)
    - Newly defined generator delta
- Then, define an abstract action that is a list of combinations of the above
- Then, create an action set that has a combination of these actions for every CO/CB

Discussion: Easier to implement using pre-existing data structures, something like this might be required in DC anyway if we
ever want to get curative actions into DC.

Given the current vision, the first approach seems the more reasonable path forward.


## How to integrate into the contingency lib

The current contingency lib is mainly evaluation focussed. However, with the curative action feature, an optimization task
would join the list. At the end, we want every entry in the curative action set to be evaluated and ranked for the CO/CBs
that are remaining after the first preventative evaluation. The information we would like to obtain is, for each curative
action, how much overloads can I safe.

Powsybl already contains an operator strategy logic which could be re-used for this purpose, but in theory we would like a
logic that
- performs the preventative SSA
- For remaining violations, performs the curative bruteforce.
This could be obtained through the operator actions interface, encoding violating N-1 cases and curative actions as operator
actions. Alternatively, it might be a thought whether we would like to integrate some logic into the java side of powsybl.

Here shall be noted the difference between different TSOs. Some have a notion of primary and secondary curative actions,
where a secondary action needs to be always available in case implementing the primary action fails (e.g. the switch is
defect). Other TSOs purely need only primary actions.

# Phase 2 - DC search

For the AC bruteforce to work in time, a relatively small action set is required. No combinatorics can happen on AC side
nor can the action set exceed more than maybe 10 actions for 10 CO/CBs. Hence, we might end up in a situation where the DC
stage should find combinations or preselect the curative action set for the AC validation 

# Phase 3 - Checkpoint logic

Not in the first version, but in a later implementation of the AC stage we would like to include a checkpoint logic for
assessing intermediate states of curative actions. We separate the action into *checkpoints* which have an electric effect
on the grid and would like to know if after every intermediate checkpoint, there is still a secondary action that is
effective. 
