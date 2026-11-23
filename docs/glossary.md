# Glossary

The shared vocabulary of ToOp, for humans and coding agents alike.
Each entry gives a short definition, the identifiers used in code, variants to avoid and,
where one exists, the page with the full explanation. The explanation itself lives on that page only.

If you introduce, rename or sharpen a term, update this page in the same pull request.

## Grid model

### Branch

A two-terminal element with a `from_node` and a `to_node`: line, two-winding transformer, tie line, boundary line.
Three-winding transformers are split into branches during preprocessing according to the star-equivalent.

- **Code:** `NetworkData.branch_ids`, `from_node`, `to_node`; dimension `n_branches`; `GridElement.kind == "branch"`; `BranchAsset`.
- **Avoid:** "line" when transformers are included too.
- **See:** [DC solver preprocessing](dc_solver/preprocessing.md#backend-interface).

### Injection

A one-terminal element that injects or consumes active power at a node: generator, load, static generator, shunt, external grid, HVDC end.
Note that in DC, no PV/PQ differentiation happens, so all injecting elements are treated the same.


- **Code:** `NetworkData.injection_ids`, `injection_nodes`; `DynamicInformation.nodal_injections`; `GridElement.kind == "injection"`; `InjectionAsset`.
- **Abbreviation:** `inj`.

### Node / bus

An electrical node of the bus-branch model. Preprocessing (`NetworkData`) says *node* (`node_ids`, `relevant_node_mask`);
the JAX solver says *bus* (`n_bus` in the PTDF). Use whichever the surrounding module uses, consistently.

- **Avoid:** "bus" when you mean a physical [busbar](#busbar-physical-vs-electrical).
  Note that in the N-1 definition `GridElement.kind == "bus"` refers to a busbar section or bus-breaker bus.
- Note the different naming conventions in pandapower and powsybl. Pandapower has a unified "bus" concept that applies to across bus-branch buses and node-breaker nodes.
### Station / bus group

A switchable group of busbars together with the assets connected to them. The asset-topology model calls it a *bus group*
(`bus_group_id`); one physical substation can contain several structural bus groups (suffixes `_a`, `_b`, …).
In the solver, a *sub* is the node of a relevant station with its local branches and injections.

- **Code:** `MasterBusGroup`, `RuntimeBusGroup`, `SimplifiedBusGroup`; solver: `n_sub_relevant`, `branches_per_sub`, `substation_correspondence`.
- **Variants in use:** station, substation, sub. Prefer *bus group* in asset-topology code and *sub* only in solver dimension names.
- **See:** [Asset topology](interfaces/asset_topology.md#bus-group-identity-and-asset-scope), [bus-group identity](dc_solver/preprocessing.md#bus-group-identity).

### Busbar (physical vs electrical)

A *physical busbar* is a `Busbar` inside a bus group. An *electrical busbar* is what remains after closed couplers are merged.
The optimizer splits a station into at most two electrical busbars, called the A and B side.

- **Code:** `Busbar`, `RuntimeBusbar`.
- **See:** [Switching distance](dc_solver/switching_distance.md).

### Busbar coupler

A switch between two busbars of the same bus group. Opening it [splits](#split-and-unsplit) the station; the flow it carried while
closed is the *cross-coupler flow*.

- **Code:** `BusbarCoupler`, `RuntimeBusbarCoupler`, `CouplerBay`; mask `cross_coupler_limits`; metric `cross_coupler_flow`.

### Asset topology

The asset-level (node-breaker) description of stations which consists of busbars, couplers, bays and switching tables. It translates electrical
actions into real switch states.
- *Master* is the structural model, independent of switch states.
- *Runtime* is the live switch state.
- *Simplified* is the runtime state projected onto what the DC solver models.

- **Code:** `MasterAssetTopology`, `RuntimeAssetTopology`, `SimplifiedAssetTopology` and their `*BusGroup` classes.
  `AppliedStation`, `RealizedStation` and `RealizedTopology` are deprecated.
- **Avoid:** "AssetTopology" or "Station" as class names; they no longer exist.
- **See:** [Asset topology](interfaces/asset_topology.md).

### PST / parallel PST group

A phase-shifting transformer whose tap position is an optimization variable. A *parallel PST group* is a set of PSTs between the
same buses with identical parameters that share one tap change (powsybl grids with `enable_parallel_pst_group_optim`).

- **Code:** `PSTRange.pst_group`, `controllable_pst_indices`, `Topology.pst_setpoints`; metric `pst_activated`.
- **See:** [Parallel PST grouping](dc_solver/preprocessing.md#parallel-pst-grouping).

### Timestep

One snapshot of a time series. Arrays carry an `n_timesteps` dimension; a strategy holds one topology per timestep. Note that currently, timestep support is not fully implemented in ToOp.


## Contingency analysis

### N-0 / base case

The grid without any outage. In an N-1 definition it is the contingency with no elements.

- **Code:** `Nminus1Definition.base_case`; suffix `_n_0` on metrics; `n_0_matrix`.
- **See:** [Metrics](topology_optimizer/metrics.md).

### N-1 definition

The list of contingencies to simulate and the elements to monitor.
Contingencies consists of single and multi-outages as well as busbar outages if configured, i.e. literature usually refers to this as N-k.
It is stored separately from the grid model.
Three forms exist:
- The **input N-1 definition**, which is an authoritative and extensive list of monitored elements and contingencies to consider.
- The **grid-validated N-1 definition**, whose elements have been checked against the grid model (existence in the grid, type of element).
- The **DC abstraction** derived from it for the DC solver, limited to what the DC model can represent.

- **Code:** `Nminus1Definition` (`monitored_elements`, `contingencies`), `GridElement`, `MonitoredElement`; files `input_nminus1_definition.json`, `nminus1_definition.json`, `dc_nminus1_definition.json`
- **Avoid:** "business N-1 definition".
- **See:** [Data artifacts](dc_solver/preprocessing.md#data-artifacts).


### Contingency / outage / failure / out of service

A *contingency* is one N-1 case: a set of elements taken out of service together. An *outage* is the act of taking an element
out of service.

- **Code:** `Contingency`, `contingency_ids`; solver: `branches_to_fail`, dimension `n_failures`, `multi_outage_branches`.
- **Prefer:** *contingency* for the business concept, *outage* for the element being removed, *failure* only in existing solver dimension names.
- **See:** [Contingency propagation](contingency_analysis/propagation.md).

### Outage group

The set of elements that are de-energised together because breakers isolate them with a contingency. Several contingencies can map
to the same outage group.

- **Code:** `outage_group_id`, `element_outage_group_id` (loadflow results), `apply_outage_grouping`.
- **See:** [Contingency propagation](contingency_analysis/propagation.md).

### Busbar outage

The outage of a physical busbar. It propagates over closed non-breaker couplers and is then expressed as branch outages plus
injection changes, without islanding the grid. It can count as an N-1 case or as a separate penalty.

- **Code:** `SolverConfig.enable_bb_outages`, `bb_outage_as_nminus1`, `clip_bb_outage_penalty`; `RelBBOutageData`, `NonRelBBOutageData`;
  mask `busbar_for_nminus1`; metrics `bb_outage_*`.
- **Abbreviation:** `bb_outage`.
- **See:** [Busbar outage](dc_solver/busbar_outage.md).

### Monitored element / masks

A *monitored element* is an element whose loadflow results are computed and checked in every contingency.
For branches it is the union of [optimized and non-worsening](#optimized--non-worsening-branch) branches;
only monitored branches appear in loadflow results, limits and metrics. Each monitored element carries a *role*
(optimized, non-worsening), a [displayed](#displayed-element) flag and an overload weight.

The importer writes the same information as boolean and float masks.
Naming pattern: `*_for_nminus1` (outaged as a contingency), `*_for_optimized` and `*_for_non_worsening` (monitored, with that role),
`*_overload_weight`, `*_disconnectable`, `*_blacklisted` (excluded from monitoring).

- **Code:** `MonitoredElement` (`optimized`, `non_worsening`, `displayed`, `weight`, `monitoring_scope`);
  `NETWORK_MASK_NAMES` (`folder_structure.py`), `NetworkMasks`; `NetworkData.monitored_branch_mask`; `DynamicInformation.branches_monitored`;
  dimension `n_branches_monitored`.
- **Avoid:** `*_for_reward`; the masks were renamed to `*_for_optimized`.
- **See:** [Monitored branches](dc_solver/preprocessing.md#monitored-branches), [Importer masks](importer/pypowsybl/index.md).

### Optimized / non-worsening branch

The role of a monitored branch in the objective. An *optimized* branch is healed: the optimizer reduces its flow if it exceeds its limit.
A *non-worsening* branch is not healed, but its loading must not get worse than before the optimization.
Typical non-worsening branches are border lines, tie lines and DSO transformers. A branch can hold both roles.

The solver does not read the role directly; the role decides the branch's [artificial limit](#artificial-limit--permanent-limit).

- **Code:** `MonitoredElement.optimized`, `MonitoredElement.non_worsening`; masks `*_for_optimized`, `*_for_non_worsening`;
  `NetworkData.optimized_branch_mask`, `non_worsening_branch_mask`; `BranchLimits.optimized_mask`, `non_worsening_mask`;
  statistics `n_branches_optimized`, `n_branches_non_worsening`.
- **Avoid:** "observed" for optimized
- **See:** [Monitored branches](dc_solver/preprocessing.md#monitored-branches), [Artificial limit by branch role](importer/pypowsybl/artificial_limits.md#artificial-limit-by-branch-role).

### Artificial limit / permanent limit

The *permanent limit* is the physical limit of a branch. The *artificial limit* is the limit the optimizer measures overloads against.
The importer derives it per monitored branch from the physical limit, the worst current and the [branch role](#optimized--non-worsening-branch).

- **Code:** operational limits `permanent_limit`, `artificial_limit_n_0`, `artificial_limit_n_1`; `DoubleLimits` (`double_limits`);
  solver `BranchLimits.max_mw_flow`, `max_mw_flow_n_1`; metric suffixes `_permanent_n_*`, `_artificial_limit_n_*`; AC `limit_scope`.
- **See:** [Artificial limits](importer/pypowsybl/artificial_limits.md), [Metrics](topology_optimizer/metrics.md).

### Displayed element

A monitored element that is shown to the user. An element can be monitored only because the optimization needs it as a boundary
condition (for example a non-worsening border line) without being of interest to the user. Displaying has no effect on the
optimization; it only filters what is reported.

- **Code:** `MonitoredElement.displayed` (default `True`); `displayed_branch_ids`; metrics `max_flow_displayed_permanent_n_0`, `max_flow_displayed_permanent_n_1`.
- **See:** [Metrics](topology_optimizer/metrics.md#other-metrics).

### Overload

Flow above a branch limit. *Overload energy* is the MW above the limit summed over branches and timesteps (for N-1, over the worst
contingency).

- **Code:** `overload_energy_n_0`, `overload_energy_n_1`, `critical_branch_count`, `MetricType`.
- **See:** [Metrics](topology_optimizer/metrics.md).

## Actions and topologies

### Relevant substation

A station at which the optimizer may apply split actions.

- **Code:** mask `relevant_subs`; `NetworkData.relevant_node_mask`; pruned by `remove_relevant_subs` during DC preprocessing.
- **Caution:** the set after DC preprocessing can be smaller than the importer mask (see [open questions](#open-questions)).

### Split and unsplit

A *split* assigns the branches and injections of a relevant station to two electrical busbars. *Unsplit* is the original configuration.

- **Code:** dimension `n_splits`; metric `split_subs`; `unsplit_action_mask`, `unsplit_flow`, `get_unsplit_ac_topology`.

### Branch action / injection action / action set

A *branch action* assigns each branch of one station to busbar A or B; an *injection action* does the same for injections.
The *action set* holds all allowed actions of all relevant stations. Topologies refer to actions by index into it.

- **Code:** two classes share the name `ActionSet`: the persisted one in `stored_action_set.py` (file `action_set.json`) and
  the JAX one in `dc_solver/jax/types.py`. Always say which one you mean.
- **See:** [DC solver preprocessing](dc_solver/preprocessing.md).

### Disconnection

A remedial action that opens a branch. Only branches flagged as disconnectable may be disconnected.

- **Code:** masks `*_disconnectable`; `disconnectable_branches`; `Topology.disconnections`; metric `disconnected_branches`.

### Bridging branch

A *bridging branch* is a branch whose outage islands the grid.

- **Code:** `bridging_branch_mask`, `find_bridges`, `find_n_minus_2_safe_branches`, `filter_disconnectable_branches_nminus2`.
- **See:** [`preprocess()` routine](dc_solver/preprocessing.md#preprocess-routine).

### Switching distance

The number of changes needed to reach a topology from the starting one. *Physical* distance counts switch operations;
*electrical* distance counts assets assigned to an electrical busbar (electrical node). The optimizer uses the electrical switching distance.

- **Code:** metric `switching_distance`; `reassignment_distance`; `preprocess_switching.py`.
- **See:** [Switching distance](dc_solver/switching_distance.md).

### Topology / strategy

A *topology* is the set of actions for one timestep: action indices, disconnections and PST setpoints.
A *strategy* is one topology per timestep. The *topo-vect* format is an alternative flat boolean encoding of branch assignments.

- **Code:** `Topology`, `Strategy` (optimizer messages), `ACStrategy`; `ActionIndexComputations`, `TopoVectBranchComputations`, `convert_branch_topo_vect`.

## Solver

### PTDF / LODF / MODF / BSDF / PSDF

Sensitivity matrices of the DC loadflow:

- **PTDF**: change of branch flow per nodal injection (`n_branches × n_bus`).
- **LODF**: change of branch flows when a single branch is outaged.
- **MODF**: the same for multi-element outages.
- **BSDF**: bus split distribution factors, which update the PTDF when a station is split.
- **PSDF**: change of branch flow per phase-shifter angle (preprocessing only).

### NetworkData / StaticInformation / DynamicInformation / SolverConfig

- `NetworkData`: NumPy preprocessing state built from a backend.
- `StaticInformation`: the solver input, stored as `static_information.hdf5`. Contains `DynamicInformation` and `SolverConfig`.
- `DynamicInformation`: traced JAX arrays. Changing values (not shapes) does not trigger recompilation.
- `SolverConfig`: static configuration. Changing it triggers recompilation.

Note that `StaticInformation` contains the *dynamic* information; the names describe persistence, not traceability.

## Optimizer

### DC stage / AC stage

The *DC stage* is a GPU genetic algorithm that searches topologies with the DC loadflow. The *AC stage* re-validates promising
DC topologies with a full AC loadflow.

- **Code:** `OptimizerType.DC`, `OptimizerType.AC`.
- **See:** [Deep dive](topology_optimizer/deepdive.md).

### Repertoire and descriptor

The DC stage keeps a MAP-Elites *repertoire*: a grid of cells, each holding the best topology found for one combination of
*descriptor* values. Default descriptors are `split_subs` and `switching_distance`.

- **Code:** `DescriptorDef`, `me_descriptors`; `topology_optimizer/dc/repertoire/`; file `repertoire.json`.

### Pull

The AC operator that takes a topology from the DC repertoire into the AC stage. It is currently the only AC evolution operator.

- **Code:** `pull`, `select_strategy`, `FilterStrategy` (`ac/evolution_functions.py`).
- **See:** [Select strategy](topology_optimizer/ac/select_strategy.md), [AC loop](topology_optimizer/ac/ac_loop.md).

### Worst-k contingencies / fast-failing

The *k* contingencies with the highest overload. The AC stage evaluates them first and rejects candidates that already fail
there (*fast-failing*), before running the remaining contingencies.

- **Code:** `n_worst_contingencies`, `worst_k_contingency_cases`, `WorstKContingencyResults`, `process_fast_failing_results`.
- **See:** [Early stopping](topology_optimizer/ac/early_stopping.md).

## Naming conventions in code

- Counts and dimensions start with `n_` (`n_branches`, `n_timesteps`), never `num_` or `_count`.
- Boolean arrays end in `_mask`, string identifier lists in `_ids`.
- Branch ends are `from_node` / `to_node`, not source / target.
- JAX shape strings reuse existing dimension names; see `packages/dc_solver_pkg/AGENTS.md`.

## Open questions

Terminology the team has not settled yet. Do not pick a side in code; follow the decision once it is recorded here.

- **Element type strings.** `GridElement.type` differs by producer (`busbar` vs `BUSBAR_SECTION`, switch kinds vs `SWITCH`).
- **Relevant / monitored station scope.** The importer monitors all stations in `relevant_subs`, while the DC stage monitors only
  those still relevant after preprocessing.
- **`n_failures`.** Counts only single-branch outages in some solver arrays and all contingencies in others.
