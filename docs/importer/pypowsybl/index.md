# PyPowSyBl import

The PyPowSyBl import path is the supported path for CGMES import and for parallel PST group identification.

When grouped PST optimization is enabled downstream, parallel PST groups are derived during DC solver preprocessing from the Powsybl grid data. PSTs are considered part of the same supported group only when they connect the same voltage magnitude, share the same bus pair regardless of orientation, and have matching tap and phase-shifter parameters. The derived group metadata is persisted into the processed grid artifacts and later written to `action_set.json` as `pst_ranges[*].pst_group`.

[`pypowsybl_import`][toop_engine_importer.pypowsybl_import]

## N-1 definition

The importer saves an N-1 definition next to the processed grid (`nminus1_definition.json`). It is built in the same order in two journeys, depending on whether `nminus1_definition_file` is set in the importer parameters:

| Stage | Input N-1 definition given | No input N-1 definition |
|---|---|---|
| 1. Source | The input definition, validated against the grid **before** the three-winding transformer conversion | Derived from the grid file through the network masks |
| 2. Area settings | The input definition is authoritative and not filtered. Network reduction keeps the voltage levels of all its elements. | View area, N-1 area and cutoff voltage, applied through the masks |
| 3. Transformer conversion | Three-winding transformers are replaced by their three two-winding legs | Same |

Without an input definition, every element selected by a `*_for_nminus1` mask becomes a single-element contingency. The `switch_for_nminus1` mask is always empty: opening a single switch usually only de-energizes the equipment behind it, which an AC loadflow cannot solve. Switches selected by `switch_for_reward` are still monitored. The busbars and couplers of the relevant stations (`relevant_subs`) are monitored as well, taken from the asset-topology master data.

With an input definition, the converted definition is finally checked against the processed grid again, dropping elements that later preprocessing steps removed. Either way, every element id in the saved definition exists in the saved grid. The network masks are computed in both journeys, so control area settings such as switchable stations still apply.

### Input N-1 definition

An input (or business) N-1 definition is a Pydantic JSON dump of [`Nminus1Definition`][toop_engine_interfaces.nminus1_definition.Nminus1Definition] whose element ids were already resolved by its producer. [`load_nminus1_definition_for_network`][toop_engine_importer.pypowsybl_import.contingency_from_file.nminus1_definition_input.load_nminus1_definition_for_network] loads it and passes it to [`filter_nminus1_definition_to_network`][toop_engine_importer.pypowsybl_import.contingency_from_file.nminus1_definition_input.filter_nminus1_definition_to_network]. The validation is soft and logs a warning for every change:

- Elements that are not in the grid are dropped from contingencies and monitored elements.
- Elements whose type or kind differs from the grid are corrected to the grid's values.
- Contingencies that lose all their elements are dropped. The empty base case is kept.
- SPPS rules are dropped as a whole if their contingency was dropped or if any condition or action element is not in the grid.

Three-winding transformers are referenced by their original id, which is why this validation runs on the grid before the conversion.

An example input definition for the complex test grid is [`create_complex_grid_nminus1_definition`][toop_engine_grid_helpers.powsybl.example_grids.create_complex_grid_nminus1_definition].

### Three-winding transformer conversion

The importer replaces every three-winding transformer `<id>` by the two-winding transformers `<id>-Leg1`, `<id>-Leg2` and `<id>-Leg3`. [`convert_three_winding_transformers_in_nminus1_definition`][toop_engine_importer.pypowsybl_import.contingency_from_file.nminus1_definition_conversion.convert_three_winding_transformers_in_nminus1_definition] rewrites the N-1 definition to match:

- A contingency or monitored element on `<id>` becomes the three legs, in place.
- An SPPS condition or action on `<id>` is copied once per leg, regardless of the rule's condition logic. Under `ANY` logic the rule therefore triggers as soon as one leg meets the condition.
- Legs that are already listed are not duplicated, so the conversion can safely be applied twice.

If network reduction cuts through a transformer, only the legs left in the reduced grid are used.
