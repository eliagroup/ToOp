# Importer worker

The importer worker is designed to run as one of the components in the ToOp architecture. The role is to preprocess grid files from the raw format (CGMES, UCTE) into the processed grid folder used by the optimizers. The worker takes preprocessing [`Command`][toop_engine_interfaces.messages.preprocess.preprocess_commands.Command] objects and, upon reception, starts the importing process. This entails:

- Creating the backend grid snapshot (`grid.xiidm` for powsybl or `grid.json` for pandapower).
- Writing masks, loadflow parameters, importer auxiliary data, asset topology metadata, and an initial `nminus1_definition.json`.
- Preprocessing through the [`load_grid`][toop_engine_dc_solver.preprocess.convert_to_jax.load_grid] function to create `static_information.hdf5`, `action_set.json`, `action_set_diffs.hdf5`, `static_information_stats.json`, and the final filtered `nminus1_definition.json`.
- Running an initial loadflow using the contingency_analysis module.

If an `action_set.json` is already present when [`load_grid`][toop_engine_dc_solver.preprocess.convert_to_jax.load_grid] starts, its `pst_ranges[*].pst_group` values are reused so grouped PSTs stay synchronized through preprocessing and optimization.
