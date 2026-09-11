# Preprocessing

The preprocessing flow is split into three parts:
- An importing procedure that prepares a processed grid folder from the raw source data. In this repository this is handled by the Importer package through [`convert_file`][toop_engine_importer.pypowsybl_import.preprocessing.convert_file]. It writes the backend-readable grid snapshot together with masks, loadflow parameters, topology metadata, and an initial contingency definition.
- A [`preprocess`][toop_engine_dc_solver.preprocess.preprocess] routine which extracts DC-loadflow relevant information from a backend and performs various data transformations.
- A [`convert_to_jax`][toop_engine_dc_solver.preprocess.convert_to_jax.convert_to_jax] routine which reformats the data from the Python format used during preprocessing to the format required by the solver. All processing happens in `preprocess`; this function purely reformats.

The [`load_grid`][toop_engine_dc_solver.preprocess.convert_to_jax.load_grid] routine combines the latter two steps, runs an initial loadflow, and persists the standard solver artifacts back into the same processed grid folder.

## Data artifacts

The processed grid folder layout is defined in the [`folder_structure`][toop_engine_interfaces.folder_structure.PREPROCESSING_PATHS]. The most important artifacts are split across the importer step and the DC solver step:

| Stage | Artifact | Purpose |
| --- | --- | --- |
| Importer | `grid.xiidm` or `grid.json` | Backend-readable grid snapshot used by the powsybl or pandapower backend. |
| Importer | `masks/` | Branch, node, and injection masks that define relevance and controllability. Contingencies are carried by the N-1 definition instead; the PandaPower backend still reads `*_for_nminus1` masks, the Powsybl backend no longer does. |
| Importer | `loadflow_parameters.json` | Loadflow parameters selected during import. |
| Importer | `importer_auxiliary_data.json` | Import statistics and auxiliary metadata produced during normalization. |
| Importer | `initial_topology/asset_topology_master_data.json` | Master asset-topology data keyed by `bus_group_id`. |
| Importer | `initial_topology/asset_topology_runtime.json` | Runtime bus-group snapshots aligned with the master asset topology. |
| Importer | `initial_topology/asset_topology.json` | Legacy combined asset-topology wrapper kept for compatibility where still needed. |
| Importer | `nminus1_definition.json` | Initial contingency definition derived from the imported grid and masks. |
| DC solver | `static_information.hdf5` | JAX-native solver input used by the DC solver and optimizer. |
| DC solver | `static_information_stats.json` | Summary statistics extracted from the preprocessed solver input. |
| DC solver | `action_set.json` | Persisted switching actions and controllable asset ranges used by postprocessing and optimization. |
| DC solver | `action_set_diffs.hdf5` | Companion diff representation for the persisted action set. |
| DC solver | `dc_nminus1_definition.json` | The DC projection of the canonical definition: exactly the contingencies DC computes, after preprocessing filters, with their source ids. Written by the DC solver, which never modifies `nminus1_definition.json`. |

The same processed grid folder is therefore both an input and an output of [`load_grid`][toop_engine_dc_solver.preprocess.convert_to_jax.load_grid].

## Parallel PST grouping

Parallel PST group optimization is currently supported only for Powsybl grids. The groups are identified during DC solver preprocessing from the imported Powsybl backend grid data. PSTs are considered part of the same supported group only when they connect the same voltage magnitude, share the same bus pair regardless of orientation, and have matching tap and phase-shifter parameters.

Controllable PSTs are serialized in `ActionSet.pst_ranges`. Each PST range carries a `pst_group` field that persists the preprocessing-derived group used by downstream tooling.

- PSTs with the same `pst_group` are treated as one optimization group.
- Group metadata is carried in static information when available, but grouped behavior is applied only when `enable_parallel_pst_group_optim=True`.
- Group members share the same tap delta during solver execution and optimization, then each member is clipped to its own tap domain.
- Different initial taps inside one group trigger a warning.
- Mixed linear and non-linear PSTs are not supported in one group when grouped optimization is enabled. We currently do not support optimization of non-linear/asymmetric PSTs.
- Parallel PST group optimization is not supported for the PandaPower backend.

The persisted `action_set.json` writes the group explicitly, so downstream tools can inspect the grouping selected during import.

## Backend interface

The [`backend`][toop_engine_interfaces.backend.BackendInterface] interface exposes a common format for both pandapower and powsybl-based grids. The main task of the backend is loading the processed grid folder and exposing the information in the required format. Instead of modelling lines, trafos, etc., the backend exposes branches, nodes, and injections.

For asset topology, the backend now exposes two distinct views:

- Master data via `get_master_data_asset_topology(...)`. This is the structural station description keyed by `bus_group_id`.
- Runtime station snapshots via `get_runtime_asset_topology(...)`. These snapshots contain the current busbar, coupler, switching-table, and bus-id state for the canonical stations.

This split is important during preprocessing because structural station grouping must not depend on the current open or closed state of busbar couplers, while runtime action generation still needs the current electrical station view.

## Bus-group identity

`bus_group_id` is the stable identifier of one structural bus-group view inside the master asset topology.

- It is the join key between master data, runtime station snapshots, simplified runtime projections, and stored actions.
- One physical substation can contribute multiple bus groups. Importers then assign deterministic suffixes such as `_a`, `_b`, and `_c`.
- Runtime bus ids can change with switching, but `bus_group_id` must stay stable.

## `preprocess()` routine

The [`preprocess`][toop_engine_dc_solver.preprocess.preprocess] function performs multiple steps to convert the backend information. The network data dataclass gets consecutively filled during these preprocessing steps:

- `extract_network_data_from_interface` pulls all backend information and stores it in the network data.
- `filter_relevant_nodes` checks if some relevant nodes have fewer than 4 branches connected. These nodes cannot be relevant as they can never be split N-1 safely and are thus de-designated as relevant nodes.
- `compute_ptdf_if_not_given` computes the PTDF matrix if the backend didn't provide one. Currently, no backend provides a pre-computed PTDF.
- `add_nodal_injections_to_network_data` sums the injections on a per-node basis.
- `compute_psdf_if_not_given` computes the PSDF matrix if the backend didn't provide one. Currently, no backend provides a pre-computed PSDF.
- `reduce_node_dimension` merges all nodes that are more than one hop away from a relevant sub or N-1 relevant branch. These nodes will never be used and the corresponding PTDF rows can be merged into one.
- `combine_phaseshift_and_injection` stacks the PSDF to the PTDF, and logically also stacks shift angles and nodal injections as the PSDF rows are then technically nodes in the resulting PTDF.
- `compute_bridging_branches` checks which branch will split the grid upon removal. These branches cannot be part of the N-1 definition and will later result in filterings.
- `exclude_bridges_from_outage_masks` removes the previously computed bridging branches from the N-1 masks, and drops any multi-outage that loses every element in the process.
- `reduce_branch_dimension` drops unnecessary branch columns from the PTDF matrix. A branch is unnecessary if it is neither monitored, outaged under N-1, nor at a relevant sub.
- `filter_disconnectable_branches_nminus2` reduces the disconnectable branches mask to only branches that are N-2 safe, i.e., that don't create additional bridges in the grid upon disconnection. These branches can never be disconnected as part of a remedial action.
- `compute_branch_topology_info` gathers information about branches at relevant nodes.
- `filter_inactive_injections` removes injections which have zero MW power in all timesteps.
- `compute_injection_topology_info` gathers information about injections at relevant nodes.
- `convert_multi_outages` removes one branch from trafo3w multi-outages and sorts the multi-outages by number of branches disconnected.
- `add_missing_asset_topo_info` ensures that all branches and injections from the network data are present in the asset topology.
- `simplify_asset_topology` creates a separate simplified asset topology that drops all assets not in the network data. The simplified asset topology exactly matches the network data view and projects runtime stations to the locally relevant assets of each node.
- `compute_electrical_actions` enumerates electrical (bus/branch) station reconfigurations for all relevant subs. The actions are also pre-filtered for suitability based on the bus/branch information. Currently, injection actions are not enumerated.
- `enumerate_station_realizations` finds a physical (node/breaker) representation for each electrical configuration.
- `remove_relevant_subs_without_actions` removes all relevant subs that have an empty action set and turns them into non-relevant subs.
- `enumerate_injection_actions` does not technically enumerate injection actions yet but just copies the assignment from the asset topology into the action set for each branch action.
- `process_injection_outages` finds the delta p and PTDF node for every injection outage. Injection outages at relevant subs are stored separately.
- `add_bus_b_columns_to_ptdf` adds a column for every relevant sub at the end of the PTDF.

## Master vs runtime semantics

Recent preprocessing logic relies on a strict distinction between master and runtime station information:

- `bus_group_id` is the master station identity used to align master data, runtime stations, station limits, and stored actions.
- Structural split groups are determined independently of open switches. Deterministic suffix ids such as `_a`, `_b`, and `_c` describe canonical groups inside one physical substation.
- Runtime lookup uses active `bus_branch_bus_ids` to map relevant electrical nodes back to master stations.
- Split filtering no longer relies on a raw "station has multiple busbars" check alone. It distinguishes materially split stations from technical multi-bus layouts and treats PST-linked internal bus components specially.
- Simplification and action generation operate on node-local projected runtime stations so the station-local asset view stays aligned with `branches_at_nodes` and `injections_at_nodes`.

## `convert_to_jax()` routine

The [`convert_to_jax`][toop_engine_dc_solver.preprocess.convert_to_jax.convert_to_jax] routine performs the following steps:

- `convert_tot_stat` pads out the branch topology info at the relevant subs.
- `convert_relevant_inj` pads out the injections at relevant subs to the upper bound length.
- `convert_masks` makes JAX arrays out of the branch masks and limits.
- `pad_out_branch_actions` pads the action set to upper bound length.
- `convert_rel_bb_outage_data` pads and transforms the busbar outage information.
- `create_static_information` copies all data into the static information dataclass.

## `load_grid()` routine

The [`load_grid`][toop_engine_dc_solver.preprocess.convert_to_jax.load_grid] routine performs the following tasks:

- Instantiate the backend, depending on whether it is a [`PandaPowerBackend`][toop_engine_dc_solver.preprocess.pandapower.pandapower_backend.PandaPowerBackend] or [`PowsyblBackend`][toop_engine_dc_solver.preprocess.powsybl.powsybl_backend.PowsyblBackend] grid. The backend reads the normalized grid files, masks, and loadflow parameters from the processed grid folder. For Powsybl grids, contingencies come from `nminus1_definition.json` rather than the `*_for_nminus1` masks, and PST grouping metadata is derived and exposed during preprocessing. (`load_grid_into_loadflow_solver_backend`)
- Call the [`preprocess`][toop_engine_dc_solver.preprocess.preprocess] routine.
- Call the [`convert_to_jax`][toop_engine_dc_solver.preprocess.convert_to_jax.convert_to_jax] routine.
- [`Validate`][toop_engine_dc_solver.jax.inputs.validate_static_information] the resulting static information.
- Run an [`initial loadflow`][toop_engine_dc_solver.preprocess.convert_to_jax.run_initial_loadflow] and update the double limits accordingly (`compute_base_loadflows`).
- Extract some [`DynamicInformationStats`][toop_engine_interfaces.messages.preprocess.preprocess_results.DynamicInformationStats].
- Save the [data artifacts](#data-artifacts), including `static_information.hdf5`, `action_set.json`, `action_set_diffs.hdf5`, `static_information_stats.json`, and the DC projection `dc_nminus1_definition.json` (`save_artifacts`).
