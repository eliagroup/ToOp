# PyPowSyBl import

The PyPowSyBl import path is the supported path for CGMES import and for parallel PST group identification.

When grouped PST optimization is enabled downstream, parallel PST groups are derived during DC solver preprocessing from the Powsybl grid data. PSTs are considered part of the same supported group only when they connect the same voltage magnitude, share the same bus pair regardless of orientation, and have matching tap and phase-shifter parameters. The derived group metadata is persisted into the processed grid artifacts and later written to `action_set.json` as `pst_ranges[*].pst_group`.

[`pypowsybl_import`][toop_engine_importer.pypowsybl_import]

## Input N-1 definition

An input (or business) N-1 definition is a Pydantic JSON dump of [`Nminus1Definition`][toop_engine_interfaces.nminus1_definition.Nminus1Definition] whose element ids were already resolved by its producer. The importer turns it into a grid-validated N-1 definition in stages:

1. **Grid file**: check the definition against the elements that exist in the grid.
2. **Area settings**: voltage level, control area and view area. The input N-1 definition takes precedence over the view area.
3. **Transformer conversion**: three-winding transformers are converted into three two-winding transformers.

Only the grid-file stage exists so far, and it is not yet called by the preprocessing pipeline. [`load_nminus1_definition_for_network`][toop_engine_importer.pypowsybl_import.contingency_from_file.nminus1_definition_input.load_nminus1_definition_for_network] loads a dump and passes it to [`filter_nminus1_definition_to_network`][toop_engine_importer.pypowsybl_import.contingency_from_file.nminus1_definition_input.filter_nminus1_definition_to_network]. The validation is lenient and logs a warning for every change:

- Elements that are not in the grid are dropped from contingencies and monitored elements.
- Elements whose type or kind differs from the grid are corrected to the grid's values.
- Contingencies that lose all their elements are dropped. The empty base case is kept.
- SPPS rules are dropped as a whole if their contingency was dropped or if any condition or action element is not in the grid.

Three-winding transformers are referenced by their original id, so the network passed in must be the grid **before** the three-winding transformer conversion.

An example dump for the complex test grid is `data/complex_grid/nminus1_definition_complex.json`.
