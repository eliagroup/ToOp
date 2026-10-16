# Importer worker

The importer worker is designed to run as one of the components in the ToOp architecture. The role is to preprocess grid files from the raw format (CGMES, UCTE) into the processed grid folder used by the optimizers. The worker takes preprocessing [`Command`][toop_engine_interfaces.messages.preprocess.preprocess_commands.Command] objects and, upon reception, starts the importing process. This entails:

- Creating the backend grid snapshot (`grid.xiidm` for powsybl or `grid.json` for pandapower).
- Writing masks (e.g. disconnectable branches, transformers, and PSTs in parallel), loadflow parameters, importer auxiliary data, asset topology metadata, and `nminus1_definition.json`.
- Running the AC N-1 security analysis of the import once, with the saved loadflow parameters, `n_contingency_worker` threads and `contingency_batch_size`. Its worst currents set the [artificial limits](../pypowsybl/artificial_limits.md) in the grid snapshot.
- Storing that analysis, with the permanent and artificial branch limits, as the initial loadflow in the loadflow result store. No second contingency analysis is run.
- Preprocessing through the [`load_grid`][toop_engine_dc_solver.preprocess.convert_to_jax.load_grid] function to create `static_information.hdf5`, `action_set.json`, `action_set_diffs.hdf5`, `static_information_stats.json`, and `dc_nminus1_definition.json`. `nminus1_definition.json` is left unchanged.

[`run_preprocessing`][toop_engine_importer.worker.preprocessor.run_preprocessing] runs these steps for each start command.

For Powsybl-imported grids, the importer worker writes the processed grid folder and then [`load_grid`][toop_engine_dc_solver.preprocess.convert_to_jax.load_grid] derives supported parallel PST groups from the Powsybl backend data. The resulting groups are persisted in `action_set.json` through `pst_ranges[*].pst_group`. The group metadata is carried into static information, while runtime grouped behavior is enabled only when `enable_parallel_pst_group_optim=True`. Parallel PST group optimization is not supported for the PandaPower backend.
