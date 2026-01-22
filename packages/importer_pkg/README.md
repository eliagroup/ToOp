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

- **[Pandapower Import](../../docs/importer/pandapower/index.md)**: Utilities for importing grid models into the PandaPower format, providing traditional power system analysis capabilities
- **[PyPowSyBl Import](../../docs/importer/pypowsybl/index.md)**: Advanced grid import functionality using the PyPowSyBl library for modern grid modeling standards
- **[Network Graph Processing](../../docs/importer/network_graph/index.md)**: Tool to create an [Asset Topology](../../docs/interfaces/asset_topology.md)
- **[PowerFactory Contingency Import](../../docs/importer/contingency_from_power_factory/index.md)**: Specialized functionality for importing contingency definitions from PowerFactory projects
- **[Worker Processes](../../docs/importer/worker/worker.md)**: A Kafka worker for deploying the importer as a service
- **[Data Export](../../docs/importer/exporter/index.md)**: UCTE only: Tool to export the topological changes back as a UCTE format

Main entry point: [`convert_file`][packages.importer_pkg.src.toop_engine_importer.pypowsybl_import.preprocessing.convert_file]

TODO: add Importer example
