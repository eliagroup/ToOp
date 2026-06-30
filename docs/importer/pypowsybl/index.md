# PyPowSyBl import

The PyPowSyBl import path is the supported path for CGMES import and for parallel PST group identification.

When grouped PST optimization is enabled downstream, parallel PST groups are derived during import from the Powsybl grid data. PSTs are considered part of the same supported group only when they connect the same voltage magnitude, share the same bus pair regardless of orientation, and have matching tap and phase-shifter parameters. The derived group metadata is persisted into the processed grid artifacts and later written to `action_set.json` as `pst_ranges[*].pst_group`.

## CGMES Ignore List and Contingency List Behavior

For CGMES imports, `ignore_list_file` is applied during the default mask creation stage.

When a contingency list is also provided, elements listed as monitored in the contingency list remain monitored even if they are present in the ignore list.


[`pypowsybl_import`][toop_engine_importer.pypowsybl_import]
