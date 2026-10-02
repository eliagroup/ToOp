# PyPowSyBl import

The PyPowSyBl import path is the supported path for CGMES import and for parallel PST group identification.

When grouped PST optimization is enabled downstream, parallel PST groups are derived during DC solver preprocessing from the Powsybl grid data. PSTs are considered part of the same supported group only when they connect the same voltage magnitude, share the same bus pair regardless of orientation, and have matching tap and phase-shifter parameters. The derived group metadata is persisted into the processed grid artifacts and later written to `action_set.json` as `pst_ranges[*].pst_group`.

[`pypowsybl_import`][toop_engine_importer.pypowsybl_import]

## Input N-1 definition

An input (or business) N-1 definition is a Pydantic JSON dump of [`Nminus1Definition`][toop_engine_interfaces.nminus1_definition.Nminus1Definition] whose element ids were already resolved by its producer. The importer turns it into a grid-validated N-1 definition in three stages: check it against the grid file, apply the area settings (the input N-1 definition takes precedence over the view area), and convert three-winding transformers into three two-winding transformers.

Only the grid-file stage exists so far, and the preprocessing pipeline does not call it yet. [`load_nminus1_definition_for_network`][toop_engine_importer.pypowsybl_import.contingency_from_file.nminus1_definition_input.load_nminus1_definition_for_network] loads a dump and passes it to [`filter_nminus1_definition_to_network`][toop_engine_importer.pypowsybl_import.contingency_from_file.nminus1_definition_input.filter_nminus1_definition_to_network]. It drops the elements, contingencies and SPPS rules the grid cannot resolve and corrects element types to the grid, logging a warning for every change. Pass the grid **before** the three-winding transformer conversion, because three-winding transformers are referenced by their original id.

An example dump for the complex test grid is `data/complex_grid/nminus1_definition_complex.json`.
