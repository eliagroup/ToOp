# Preprocessing

The preprocessing process is split into three parts:
- An importing procedure that prepares the grid data—this is specific to the data source and power system modelling framework. The code for this is not hosted in this repository, but in the importer repo.

- A [`preprocess`][packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.preprocess] routine which extracts DC-loadflow relevant information from a backend and performs various data transformations.
- A [`convert_to_jax`][packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.convert_to_jax.convert_to_jax] routine which reformats the data from the Python format used during preprocessing to the format required by the solver. All processing happens in `preprocess`; this function purely reformats. The only exception is currently the N-2 unsplit analysis.

The [`load_grid`][packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.convert_to_jax.load_grid] routine combines the two and runs an initial loadflow. This routine serves as a top-level entrypoint for preprocessing.

## Data artifacts

Output data pieces are defined in the [`folder_structure`][packages.interfaces_pkg.src.toop_engine_interfaces.folder_structure.PREPROCESSING_PATHS]. Most notably, the following data objects are written out:

- A`static_information.hdf5` file containing all JAX data—this is all the DC solver and optimizer need to run an optimization.
- An `action_set.json` containing an asset-topology-based representation of the action set, used for postprocessing the raw bus/branch DC topologies to node/breaker topologies.
- A `nminus1_definition.json` containing the N-1 cases in the grid. AC validation can happen with the grid, action set, and N-1 definition.

## Backend interface

The [`backend`][packages.interfaces_pkg.src.toop_engine_interfaces.backend.BackendInterface] interface exposes a common format for both pandapower and powsybl-based grids. The main task of the backend is loading the masks and exposing the information in the required format. Instead of modelling lines, trafos, etc., the backend exposes branches, nodes, and injections.

## `preprocess()` routine

The [`preprocess`][packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.preprocess] function performs multiple steps to convert the backend information. The network data dataclass gets consecutively filled during these preprocessing steps:

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

The [`convert_to_jax`][packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.convert_to_jax.convert_to_jax] routine performs the following steps:

- `convert_tot_stat` pads out the branch topology info at the relevant subs.
- `convert_relevant_inj` pads out the injections at relevant subs to the upper bound length.
- `convert_masks` makes JAX arrays out of the branch masks and limits.
- `pad_out_branch_actions` pads the action set to upper bound length.
- `convert_rel_bb_outage_data` pads and transforms the busbar outage information.
- `create_static_information` copies all data into the static information dataclass.
- `unsplit_n2_analysis`, if N-2 is enabled, runs an unsplit N-2 analysis to determine branch limits for N-2.

## `load_grid()` routine

The [`load_grid`][packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.convert_to_jax.load_grid] routine performs the following tasks:

- Instantiate the backend, depending on whether it is a [`PandaPowerBackend`][packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.pandapower.pandapower_backend.PandaPowerBackend] or [`PowsyblBackend`][packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.powsybl.powsybl_backend.PowsyblBackend] grid. (`load_grid_into_loadflow_solver_backend`)
- Call the [`preprocess`][packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.preprocess] routine.
- Call the [`convert_to_jax`][packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.convert_to_jax.convert_to_jax] routine.
- [`Validate`][packages.dc_solver_pkg.src.toop_engine_dc_solver.jax.inputs.validate_static_information] the resulting static information.
- Run an [`initial loadflow`][packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.convert_to_jax.run_initial_loadflow] and update the double limits accordingly (`compute_base_loadflows`).
- Extract some [`StaticInformationStats`][packages.interfaces_pkg.src.toop_engine_interfaces.messages.preprocess.preprocess_results.StaticInformationStats].
- Save the [data artifacts](#data-artifacts) (`save_artifacts`).
