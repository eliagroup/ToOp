# Grid Helpers

The Grid Helpers package provides essential utility functions and abstractions for working with electrical grid models in both [`pandapower`](https://www.pandapower.org/) and [`pypowsybl`](https://pypowsybl.readthedocs.io/) frameworks. It serves as a crucial bridge layer that enables seamless interoperability between different power system modeling tools within the ToOp ecosystem.

## Usage
This package is not intended to be used directly, but is rather a collection of functions used by other ToOp packages

## Overview

Grid Helpers acts as the foundational abstraction layer that standardizes operations across different power system modeling backends. It provides a unified interface for data extraction, manipulation, and conversion while preserving the specific characteristics and capabilities of each underlying framework.

### Key Features

- **Dual Backend Support**: Utilities for both pandapower and pypowsybl grid models
- **Data Standardization**: Consistent interfaces for extracting loadflow results, branch parameters, and injection data
- **ID Management**: Robust global identification system for grid elements across different backends
- **Example Networks**: Curated test grids including IEEE cases and synthetic networks
- **Loadflow Integration**: Parameter configuration and result extraction for power flow studies

## Pandapower Key Data Structures and Concepts

- **Loadflow parameter**: A collection of loadflow parameter used by the ToOp project.
- **Helpers**: A standardized function set to get grid information. Heavy used by the [`PandaPowerBackend`][packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.pandapower.pandapower_backend.PandaPowerBackend]. 
- **Id helpers**: A fix due to Pandapower not having global ids. It combines the row id of the dataframe with the table name.
- **Import helpers**: Some Functions to fix data quality issues
- **Asset Topology helpers**: TODO: currently in Importer package


## PyPowSyBl Key Data Structures and Concepts
- **Loadflow parameter**:A collection of loadflow parameter used by the ToOp project.
- **Helpers**: A standardized function set to get grid information. Heavy used by the [`PowsyblBackend`][packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.powsybl.powsybl_backend.PowsyblBackend]. 
- **Asset Topology helpers**: An implementation of the [AssetTopology][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology], main entry: [`get_list_of_stations`][packages.grid_helpers_pkg.src.toop_engine_grid_helpers.powsybl.powsybl_asset_topo.get_list_of_stations]
- **Single line diagram (SLD)**: A modified version of the powsybl SLD, with a bright and dark mode. This could be integrated into powsybl itself as it's own package (java).
- **Polars DataFrame**: Once you have millions of rows in you dataframe, the PowSyBl (Java) to PyPowSybl converter (Java->C++->Python->Pandas) becomes slow. You can eliminate one large bottleneck by removing pandas from this data extraction path. Ideally the transfer from Java to PyPowSyBl would deliver in this case parquet as a format. The speed boost only starts with large amount of N-1 cases: 3000 Bus network with more than 500 N-1 cases. Below that the speed of collection is roughly equal.

### Integration Examples

For complete integration examples with other ToOp packages, see:

- **[DC Solver Examples](../quickstart.md)**: Grid preprocessing and optimization setup
- **[Contingency Analysis](../contingency_analysis/intro.md)**: N-1 analysis configuration
- **[Topology Optimizer](../topology_optimizer/intro.md)**: Multi-objective switching optimization

## Reference Documentation

For detailed API documentation, see:

- **[Pandapower Helpers][packages.grid_helpers_pkg.src.toop_engine_grid_helpers.pandapower]**: Complete pandapower utility reference
- **[PyPowSyBL Helpers][packages.grid_helpers_pkg.src.toop_engine_grid_helpers.powsybl]**: Full pypowsybl functionality guide
- **Example Networks**: Comprehensive test network catalog
    - [Pandapower grids][packages.grid_helpers_pkg.src.toop_engine_grid_helpers.pandapower.example_grids]
    - [Powsybl grids][packages.grid_helpers_pkg.src.toop_engine_grid_helpers.powsybl.example_grids]
