# PyPowSyBl import

The PyPowSyBl import path is the supported path for CGMES import and for parallel PST group identification.

When grouped PST optimization is enabled downstream, parallel PST groups are derived during DC solver preprocessing from the Powsybl grid data. PSTs are considered part of the same supported group only when they connect the same voltage magnitude, share the same bus pair regardless of orientation, and have matching tap and phase-shifter parameters. The derived group metadata is persisted into the processed grid artifacts and later written to `action_set.json` as `pst_ranges[*].pst_group`.

[`pypowsybl_import`][toop_engine_importer.pypowsybl_import]

## Branch masks

The importer writes these branch masks to `masks/` for lines, transformers (`trafo`, `trafo3w`) and tie lines. Switches have the same two masks for monitoring.

| File | Meaning |
|---|---|
| `*_for_nminus1.npy` | The element is outaged as a single-element contingency. |
| `*_for_optimized.npy` | The element is monitored and healed if it exceeds its limit (optimized). |
| `*_for_non_worsening.npy` | The element is monitored, is not healed and must not get worse (non-worsening). Set for border lines and tie lines, and for DSO transformers. |

The monitored elements are the union of the optimization and non-worsening masks, see [Monitored branches](../../dc_solver/preprocessing.md#monitored-branches). The optimization limit of each monitored branch depends on its mask, see [Artificial limits](artificial_limits.md).

## N-1 definition

The importer saves an N-1 definition next to the processed grid (`nminus1_definition.json`). It is built in the same order in two journeys, depending on whether `input_nminus1_definition_file` is set in the importer parameters:

| Stage | Input N-1 definition given | No input N-1 definition |
|---|---|---|
| 1. Source | The input definition, validated against the grid **before** the three-winding transformer conversion | Derived from the grid file through the network masks |
| 2. Area settings | The input definition is authoritative and not filtered. Network reduction keeps the voltage levels of all its elements. | View area, N-1 area and cutoff voltage, applied through the masks |
| 3. Transformer conversion | Three-winding transformers are replaced by their three two-winding legs | Same |

Without an input definition, every element selected by a `*_for_nminus1` mask becomes a single-element contingency. The `switch_for_nminus1` mask is always empty: opening a single switch usually only de-energizes the equipment behind it, which an AC loadflow cannot solve. Switches selected by `switch_for_optimized` or `switch_for_non_worsening` are still monitored. The busbars and couplers of the relevant stations (`relevant_subs`) are monitored as well, taken from the asset-topology master data.

The importer splits the monitored branches into two masks that it never sets for the same element (the DC solver itself accepts an element in both, which then counts as optimized). Branches in the view area with operational limits are optimized (`*_for_optimized`): they are healed if they exceed their limit. Lines and tie lines leaving the N-1 area, and every DSO transformer (only one side at or above the cutoff voltage), are non-worsening (`*_for_non_worsening`): they are not healed but must not get worse. Border branches without limits are only non-worsening if `border_line_factors` or `dso_trafo_factors` is set, because the limits are then derived from the loadflow. Blacklisted lines are not monitored, DSO transformers are exempt from the blacklist.

With an input definition, the converted definition is finally checked against the processed grid again, dropping elements that later preprocessing steps removed. Either way, every element id in the saved definition exists in the saved grid. The network masks are computed in both journeys, so control area settings such as switchable stations still apply.

### Input N-1 definition

An input (or business) N-1 definition is a Pydantic JSON dump of [`Nminus1Definition`][toop_engine_interfaces.nminus1_definition.Nminus1Definition] whose element ids were already resolved by its producer. [`load_nminus1_definition_for_network`][toop_engine_importer.pypowsybl_import.contingency_from_file.nminus1_definition_input.load_nminus1_definition_for_network] loads it and passes it to [`filter_nminus1_definition_to_network`][toop_engine_importer.pypowsybl_import.contingency_from_file.nminus1_definition_input.filter_nminus1_definition_to_network]. The validation is soft and logs a warning for every change:

- Elements that are not in the grid are dropped from contingencies and monitored elements.
- Elements whose type or kind differs from the grid are corrected to the grid's values.
- Element names are overwritten by the grid's name, as the DC solver preprocessing names elements: UCTE lines and transformers are named `<id>: <elementName>`, UCTE tie lines by their id, other UCTE elements by their `elementName`, and all other elements by their Powsybl `name`. Elements without a name in the grid keep their input name.
- Contingencies that the DC solver outages as a single element (exactly one line, transformer or tie line, or else exactly one injection; isolating switches do not count) take that element's grid name. Multi-outages keep their input name.
- Contingencies that lose all their elements are dropped. The empty base case is kept.
- SPPS rules are dropped as a whole if their contingency was dropped or if any condition or action element is not in the grid.

Three-winding transformers are referenced by their original id, which is why this validation runs on the grid before the conversion.

The input and the saved definition are two separate files. `input_nminus1_definition_file` is read from the same filesystem as the grid model file. The grid-validated result is always written to `nminus1_definition.json` in the data folder, which is the file AC contingency analysis and the optimizer read. Do not store the input file at that path: if both filesystems point to the same location, the import overwrites it.

An example input definition for the complex test grid is [`create_complex_grid_nminus1_definition`][toop_engine_grid_helpers.powsybl.example_grids.create_complex_grid_nminus1_definition].

### Three-winding transformer conversion

The importer replaces every three-winding transformer `<id>` by the two-winding transformers `<id>-Leg1`, `<id>-Leg2` and `<id>-Leg3`. [`convert_three_winding_transformers_in_nminus1_definition`][toop_engine_importer.pypowsybl_import.contingency_from_file.nminus1_definition_conversion.convert_three_winding_transformers_in_nminus1_definition] rewrites the N-1 definition to match:

- A contingency or monitored element on `<id>` becomes the three legs, in place.
- An SPPS condition or action on `<id>` is copied once per leg, regardless of the rule's condition logic. Under `ANY` logic the rule therefore triggers as soon as one leg meets the condition.
- Legs that are already listed are not duplicated, so the conversion can safely be applied twice.

pypowsybl moves each operational limit of a three-winding transformer to the leg of its side, rescaled to the star voltage. Legs whose side had no limit get none.

`trafo3w_leg_monitoring` in the importer parameters sets how the legs of a monitored three-winding transformer inherit its `optimized`, `non_worsening` and `displayed` flags, in both journeys. The lead leg is the first leg with an operational limit, in the order Leg1, Leg2, Leg3, or Leg1 if no leg has a limit (logged as a warning).

| Mode | Lead leg | Other legs |
| --- | --- | --- |
| `trafo3w_all_legs_monitoring` (default) | Inherits all flags. | Inherit all flags. |
| `trafo3w_single_leg_monitoring` | Inherits all flags. | `optimized=False`, `displayed=False`. `non_worsening=True` only if the three-winding transformer is optimized and not non-worsening, else `False`. |
| `trafo3w_remove_non_lead_legs` | Inherits all flags. | Removed from the monitored elements. |

With `trafo3w_single_leg_monitoring`, an optimized transformer is optimized on one leg and keeps the others from getting worse. A non-worsening transformer, optimized or not, is represented by the lead leg only; the other legs stay in the monitored elements without any flag. `trafo3w_remove_non_lead_legs` reduces the number of monitored elements, and with it the runtime of the security analysis (N-1 analysis).

Contingencies and SPPS rules are not affected by the mode; outaging a three-winding transformer still outages all three legs. A leg that the definition already lists itself keeps its own flags. The limits are those of the active limit sets of the processed grid at the time the N-1 definition is created, before the default permanent limits of `create_artificial_limit_when_missing_permanent` are added.

If network reduction cuts through a transformer, only the legs left in the reduced grid are used.
