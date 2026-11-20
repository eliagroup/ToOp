# Contingency Propagation

This page describes how contingency propagation currently works in the contingency-analysis package and what is delegated to the backend.

## Scope

In this repository, contingency propagation matters differently on the powsybl and pandapower paths.

On the powsybl side, there are two propagations to keep apart:
1. The native contingency propagation of powsybl inside the security analysis, used by every production path.
2. The propagation of the DC solver, applied to the N-1 definition before the security analysis, only to compare
   powsybl with the DC solver.

The security analysis runners do neither themselves: they run the contingencies and the parameters as given.

On the pandapower side, the comparable mechanism is not a backend-native fault-isolation feature. Instead, ToOp can expand contingencies into topology-aware outage groups based on connected components.

## Powsybl contingency propagation

Both powsybl runners, `run_contingency_analysis_powsybl` (monitored elements) and
`run_contingency_analysis_powsybl_violation_table` ([limit violation table](limit_table.md)), hand the contingencies
to powsybl exactly as they are defined and run the security analysis parameters as they are given. Neither of them
propagates a contingency itself.

`get_default_security_analysis_parameters` switches on the OpenLoadFlow provider parameter `contingencyPropagation`
(`CONTINGENCY_PROPAGATION`). Powsybl then isolates every contingency like a protection system would: starting from the
outaged equipment, it searches the station topology for the circuit breakers that have to open, and simulates
everything between them as tripped. Disconnectors and load-break switches do not interrupt a fault, so equipment
connected through them only is tripped as well. Every production path runs like this: the importer (the initial AC
loadflow and the artificial limits), the AC optimizer and the AC loadflow service.

A contingency whose isolation changes nothing, e.g. the outage of a busbar section nothing is connected to, gets no
result from powsybl. It is `NO_CALCULATION` in the convergence table, in both runners.

The elements powsybl tripped are listed per contingency in the connectivity result, see
[connectivity](limit_table.md#connectivity).

With the propagation, powsybl reports the voltages of node-breaker voltage levels at their busbar sections only. The
voltage angle differences of the outaged branches and the monitored switches are therefore read at the busbar section
each of their buses is connected to through closed switches, see `get_busbar_bus_breaker_buses`. For an outaged branch
that is the angle it would be reclosed onto, across the bay breakers powsybl opened.

## Propagation of the DC solver

The DC solver does not isolate a fault with breakers. It models a busbar outage on its simplified station topology:
the outaged busbar takes along the busbars it reaches through closed disconnectors, and the branches and injections
connected to them are outaged (see `preprocess_bb_outage` of the DC solver). Busbar outages it cannot model, e.g. one
that would split its station, are dropped from its N-1 definition.

`propagate_busbar_contingencies(nminus1_definition, net)` reproduces this outage for powsybl. It expands every busbar
section of a contingency to the busbar sections it reaches through closed disconnectors, stopping at breakers (see
`get_busbar_propagation_map`), and returns a new N-1 definition with the same contingency ids. The propagation depends
on the switch states, so it has to be repeated after every topology change.

It is only used to compare powsybl with the DC solver, e.g. in `validate_loadflow_results` and the DC parity tests.
`PowsyblRunner(dc_solver_propagation=True)` propagates the N-1 definition after the topology changes of every loadflow
and switches `contingencyPropagation` off, so a contingency is never propagated twice:

| Caller | Propagation | `contingencyPropagation` |
| ------ | ----------- | ------------------------ |
| production paths (importer, AC optimizer, loadflow service) | powsybl | `true` |
| `PowsyblRunner(dc_solver_propagation=True)`, DC parity tests | `propagate_busbar_contingencies` | `false` |

Whoever passes `contingencyPropagation` as `false` is responsible for an N-1 definition that is already propagated.

## Difference between the DC artificial limits and the DC solver

The importer derives the artificial DC limits from a DC security analysis with the propagation of powsybl, like
every other production path. The limits describe the N-1 flows of the grid as the contingencies really isolate,
not as the DC solver approximates them.

For branch and injection outages both are the same. For a busbar outage they can differ: powsybl opens the breakers
around the busbar, which can trip more than the DC solver outages, e.g. a feeder that is connected to the busbar
without its own breaker, or a busbar coupled to it by a breaker. The DC solver then computes a busbar contingency
against a limit that was derived from a slightly different outage. This is accepted on purpose: the limits follow the
physically correct isolation, and the remaining difference is a limitation of the DC solver's busbar outage model.
The DC parity tests show where it differs, as they fail without `dc_solver_propagation`.

## Pandapower side

Pandapower currently does not use a powsybl-style native contingencyPropagation parameter. Instead, the topology-aware propagation logic is implemented on the ToOp side through outage grouping.

When apply_outage_grouping is enabled in the pandapower contingency-analysis configuration, contingencies are mapped to the electrically connected component or components containing their directly outaged elements. Contingencies that touch the same set of connected components are grouped together into one outage group.

The resulting outage group then contains the union of all grid elements in those connected components. In practice, this means the modeled outage scope is expanded from the directly listed contingency elements to the full topology-aware disconnected area represented by the group.

Conceptually, the pandapower flow is:
1. Build connected components for the contingency-analysis graph.
2. Map each outaged element to its component.
3. Canonicalize the touched component set as the contingency signature.
4. Group contingencies with the same signature.
5. Expand each group to all elements contained in the affected connected components.

This is different from the powsybl path in two important ways:
1. The logic is implemented in ToOp, not delegated to a backend security-analysis propagation feature.
2. The expansion target is a connected-component-based outage group, not a breaker-opening sequence computed by the backend.

As a result, pandapower-side propagation is best understood as contingency expansion by outage scope rather than native fault isolation.

## Pandapower result semantics

If outage grouping is enabled, the pandapower result object also includes a connectivity table that maps each contingency to all affected elements through a shared outage_group_id.

That table expresses the propagated outage scope explicitly:
- the contingency identifies the initiating outage definition
- the affected elements are all elements in the same outage group
- the outage_group_id identifies the shared topology-aware disconnected scope

So on the pandapower path, propagation is visible as grouped connectivity metadata and grouped outage execution, rather than as a backend-side breaker isolation procedure.
