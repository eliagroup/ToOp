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
| Importer | `masks/` | Branch, node, and injection masks that define relevance, controllability, and contingency handling. |
| Importer | `loadflow_parameters.json` | Loadflow parameters selected during import. |
| Importer | `importer_auxiliary_data.json` | Import statistics and auxiliary metadata produced during normalization. |
| Importer | `initial_topology/asset_topology.json` | Asset-topology view of the imported grid. |
| Importer | `nminus1_definition.json` | Initial contingency definition derived from the imported grid and masks. |
| DC solver | `static_information.hdf5` | JAX-native solver input used by the DC solver and optimizer. |
| DC solver | `static_information_stats.json` | Summary statistics extracted from the preprocessed solver input. |
| DC solver | `action_set.json` | Persisted switching actions and controllable asset ranges used by postprocessing and optimization. |
| DC solver | `action_set_diffs.hdf5` | Companion diff representation for the persisted action set. |
| DC solver | `nminus1_definition.json` | Refreshed contingency definition after preprocessing filters have been applied. |

The same processed grid folder is therefore both an input and an output of [`load_grid`][toop_engine_dc_solver.preprocess.convert_to_jax.load_grid]. In particular, `action_set.json` is no longer only a postprocessing artifact: if it already exists when preprocessing starts, the backend reads its PST grouping metadata and preserves it through the preprocessing pipeline.

## Parallel PST grouping in `action_set.json`

Controllable PSTs are serialized in `ActionSet.pst_ranges`. Each PST range carries a `pst_group` field that defines which PSTs must move together during optimization.

- PSTs with the same `pst_group` are treated as one optimization group.
- If `action_set.json` is missing, or a controllable PST is absent from it, preprocessing falls back to one group per PST.
- During preprocessing, grouped PSTs are clipped to their common tap domain before the action set is written back to disk.
- Mixed linear and non-linear PSTs are rejected and cannot share the same group (We currently do not support optimization of non-linear/asymmetric PSTs).

The persisted `action_set.json` always writes the group explicitly, so downstream tools and subsequent preprocessing runs see the same grouping.

## Backend interface

The [`backend`][toop_engine_interfaces.backend.BackendInterface] interface exposes a common format for both pandapower and powsybl-based grids. The main task of the backend is loading the processed grid folder and exposing the information in the required format. Instead of modelling lines, trafos, etc., the backend exposes branches, nodes, and injections.

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
- `exclude_bridges_from_outage_masks` removes the previously computed bridging branches from the N-1 definition.
- `reduce_branch_dimension` drops unnecessary branch columns from the PTDF matrix. A branch is unnecessary if it is neither monitored, outaged under N-1, nor at a relevant sub.
- `filter_disconnectable_branches_nminus2` reduces the disconnectable branches mask to only branches that are N-2 safe, i.e., that don't create additional bridges in the grid upon disconnection. These branches can never be disconnected as part of a remedial action.
- `compute_branch_topology_info` gathers information about branches at relevant nodes.
- `filter_inactive_injections` removes injections which have zero MW power in all timesteps.
- `compute_injection_topology_info` gathers information about injections at relevant nodes.
- `convert_multi_outages` removes one branch from trafo3w multi-outages and sorts the multi-outages by number of branches disconnected.
- `add_missing_asset_topo_info` ensures that all branches and injections from the network data are present in the asset topology.
- `simplify_asset_topology` creates a separate simplified asset topology that drops all assets not in the network data. The simplified asset topology exactly matches the network data view.
- `compute_electrical_actions` enumerates electrical (bus/branch) station reconfigurations for all relevant subs. The actions are also pre-filtered for suitability based on the bus/branch information. Currently, injection actions are not enumerated.
- `enumerate_station_realizations` finds a physical (node/breaker) representation for each electrical configuration.
- `remove_relevant_subs_without_actions` removes all relevant subs that have an empty action set and turns them into non-relevant subs.
- `enumerate_injection_actions` does not technically enumerate injection actions yet but just copies the assignment from the asset topology into the action set for each branch action.
- `process_injection_outages` finds the delta p and PTDF node for every injection outage. Injection outages at relevant subs are stored separately.
- `add_bus_b_columns_to_ptdf` adds a column for every relevant sub at the end of the PTDF.

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

- Instantiate the backend, depending on whether it is a [`PandaPowerBackend`][toop_engine_dc_solver.preprocess.pandapower.pandapower_backend.PandaPowerBackend] or [`PowsyblBackend`][toop_engine_dc_solver.preprocess.powsybl.powsybl_backend.PowsyblBackend] grid. The backend reads the normalized grid files, masks, loadflow parameters, and any existing PST grouping metadata from the processed grid folder. (`load_grid_into_loadflow_solver_backend`)
- Call the [`preprocess`][toop_engine_dc_solver.preprocess.preprocess] routine.
- Call the [`convert_to_jax`][toop_engine_dc_solver.preprocess.convert_to_jax.convert_to_jax] routine.
- [`Validate`][toop_engine_dc_solver.jax.inputs.validate_static_information] the resulting static information.
- Run an [`initial loadflow`][toop_engine_dc_solver.preprocess.convert_to_jax.run_initial_loadflow] and update the double limits accordingly (`compute_base_loadflows`).
- Extract some [`StaticInformationStats`][toop_engine_interfaces.messages.preprocess.preprocess_results.StaticInformationStats].
- Save the [data artifacts](#data-artifacts), including `static_information.hdf5`, `action_set.json`, `action_set_diffs.hdf5`, `static_information_stats.json`, and the refreshed `nminus1_definition.json` (`save_artifacts`).
