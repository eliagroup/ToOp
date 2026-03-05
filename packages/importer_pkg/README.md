# Importer

The Importer package serves as the gateway for loading power system grid models and associated data into the ToOp ecosystem. This package handles the complex task of importing electrical transmission grid models from industry-standard formats and tools, enabling integration with topology optimization.

## Overview

The Importer package provides capabilities for importing grid models from multiple sources and formats. 
It supports grid models from UCTE files, and CGMES standards (currently only PyPowSyBl), converting them into standardized formats suitable for power system analysis and optimization.

At its core, the package leverages two Python libraries as backends: 

1. **PandaPower**
2. **PyPowSyBl**

## Package Structure

The Importer package is organized into several focused modules, each addressing specific aspects of the grid import process:

- **[Pandapower Import](https://eliagroup.github.io/ToOp/importer/pandapower/)**: Utilities for importing grid models into the PandaPower format
- **[PyPowSyBl Import](https://eliagroup.github.io/ToOp/importer/pypowsybl/)**: Utilities for importing grid models into the PowSyBl format
- **[Network Graph Processing](https://eliagroup.github.io/ToOp/importer/network_graph/)**: Tool to create an [Asset Topology](https://eliagroup.github.io/ToOp/interfaces/asset_topology/)
- **[PowerFactory Contingency Import](https://eliagroup.github.io/ToOp/importer/contingency_from_power_factory/)**: Specialized functionality for importing contingency definitions from PowerFactory projects
- **[Worker Processes](https://eliagroup.github.io/ToOp/importer/worker/worker/)**: A Kafka worker for deploying the importer as a service
- **[Data Export](https://eliagroup.github.io/ToOp/importer/exporter/)**: UCTE only: Tool to export the topological changes back as a UCTE format

Main entry point: [`convert_file`][toop_engine_importer.pypowsybl_import.preprocessing.convert_file]

TODO: add Importer example
