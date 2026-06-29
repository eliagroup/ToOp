# Pandapower import

The PandaPower import path supports the backend workflows documented for PandaPower grids, but it is not a supported path for parallel PST group optimization.

Parallel PST group optimization is currently supported only for Powsybl-imported grids, where groups are identified from the imported grid data and persisted into the processed grid artifacts. A PandaPower grid may still contain ordinary PST information used by other backend functionality, but grouped PST optimization should not be enabled for PandaPower-based preprocessing artifacts.

[`pandapower`][toop_engine_importer.pandapower_import]
