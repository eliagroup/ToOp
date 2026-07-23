# Contingency Propagation

This page describes how contingency propagation currently works in the contingency-analysis package and what is delegated to the backend.

## Scope

In this repository, contingency propagation matters differently on the powsybl and pandapower paths.

On the powsybl side, there are two distinct propagation layers to keep apart:
1. ToOp-side contingency expansion before the contingency is handed to powsybl.
2. Native powsybl contingency propagation inside the security analysis engine.

These two layers are related, but they are not the same mechanism.

On the pandapower side, the comparable mechanism is not a backend-native fault-isolation feature. Instead, ToOp can expand contingencies into topology-aware outage groups based on connected components.

## ToOp-side bus contingency expansion

Before contingencies are translated to powsybl, ToOp expands bus contingencies over all busbar sections that share the same bus-breaker bus identifier within the same voltage level.

Conceptually, the preprocessing builds a mapping like this:

- busbar_a -> [busbar_a, busbar_b, busbar_c]
- busbar_b -> [busbar_a, busbar_b, busbar_c]
- busbar_c -> [busbar_a, busbar_b, busbar_c]

If a contingency outaged one of these busbar sections, the translated powsybl contingency contains the full expanded set instead of just the originally named busbar section.

This expansion is derived from the bus map and groups busbar sections by:
- voltage_level_id
- bus_breaker_bus_id

The goal is to keep bus contingencies aligned with the bus-breaker representation even when the original contingency input names only one busbar section.

## Native powsybl contingency propagation

Powsybl itself also offers a security-analysis parameter called contingencyPropagation.

Its behavior is different from ToOp's busbar-section expansion. According to the powsybl model, the engine can perform a topological search around the fault and determine which circuit breakers must open to isolate it. Depending on the station topology, this can cause additional equipment to be simulated as tripped, while disconnectors and load-break switches are not treated as fault-interrupting devices.

In short:
- ToOp-side expansion broadens the list of outaged busbar-section ids before the security analysis starts.
- powsybl-side propagation changes how the security analysis isolates the contingency inside the network model.

## Current behavior in ToOp

On the current implementation path, ToOp already performs the bus contingency expansion described above when translating contingencies to powsybl.

At the same time, the provider parameter passed to powsybl security analysis currently sets contingencyPropagation to false. That means the backend-side breaker-based isolation propagation is not enabled in the actual run, even though the schema already contains a contingency_propagation flag.

So the current effective behavior is:
1. Bus contingencies are expanded by ToOp across busbar sections sharing the same bus-breaker bus.
2. powsybl native contingency propagation is currently not enabled during execution.

## Why this distinction matters

These two mechanisms answer different questions.

ToOp-side expansion answers:
- Which explicit busbar-section ids should belong to the contingency definition before it is submitted to the backend?

powsybl-side propagation answers:
- Once the contingency is applied inside the substation topology, which breakers must be opened to isolate the faulted area?

Because of that, enabling native powsybl contingency propagation later would not replace ToOp's current busbar expansion automatically. It would add a second layer of topology-aware isolation behavior inside the backend run.

## Practical implication for results

If you see a bus contingency represented by multiple busbar-section ids in the translated powsybl contingency, that comes from ToOp's preprocessing.

If in the future you enable powsybl native contingency propagation as well, then additional tripped equipment could appear because the backend would open the required circuit breakers to isolate the contingency according to the station topology.

## Current limitation

The powsybl-side contingency_propagation option is present in the powsybl helper schema, but the current execution path does not yet forward it into the provider parameters used for the security-analysis run. Documentation and behavior therefore intentionally distinguish between:
- implemented preprocessing-side bus contingency expansion
- not-yet-enabled native powsybl contingency propagation

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
