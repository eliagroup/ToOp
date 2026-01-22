# Importer worker

The importer worker is designed to run as one of the components in the ToOp architecture. The role is to preprocess gridfiles from the raw format (CGMES, UCTE) to an internal representation that can be used in the optimizers. The worker takes preprocessing [`Command`][toop_engine_interfaces.messages.preprocess.preprocess_commands.Command] objects and, upon reception starts the importing process. This entails
- Grid file creation
- Preprocessing through the [`load_grid`][toop_engine_dc_solver.preprocess.convert_to_jax.load_grid] function
- Running an initial loadflow using the contingency_analysis module.
