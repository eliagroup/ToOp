# ToOp-Interfaces

The `interfaces` package provides a set of abstractions and adapters to enable interoperability between different grid modeling tools and data formats. It is designed to facilitate the integration of various power system analysis libraries, such as pandapower and pypowsybl, within a unified workflow.

## Interfaces

[Asset Topology][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology] - Defines the physical structure and configuration of electrical assets in the grid system.
A full description is found here: [Asset Topology](asset_topology.md)

The [`BackendInterface`][packages.interfaces_pkg.src.toop_engine_interfaces.backend.BackendInterface] is an abstract interface to read grid data for a bus-branch model. It provides raw numeric inputs for the solver — no validation or processing. Specifically not task of this interface is to perform any validations or processing of the data.
Implementations can be found in the repository: [`powsybl_backend.py`][packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.powsybl.powsybl_backend.PowsyblBackend] and [`pandapower_backend.py`][packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.pandapower.pandapower_backend.PandaPowerBackend]

[Folder Structure][packages.interfaces_pkg.src.toop_engine_interfaces.folder_structure] - Manages the file system organization for preprocessing and post-processing optimization data.

[Loadflow Results][packages.interfaces_pkg.src.toop_engine_interfaces.loadflow_results] - Contains power flow calculation results and electrical network analysis data.

[Loadflow Results Polars][packages.interfaces_pkg.src.toop_engine_interfaces.loadflow_results_polars] - Polars-based implementation for efficient handling of large-scale loadflow result datasets.

[N-minus-1 Definition][packages.interfaces_pkg.src.toop_engine_interfaces.nminus1_definition] - Defines contingency scenarios for reliability analysis where one component is out of service.

[Stored Action Set][packages.interfaces_pkg.src.toop_engine_interfaces.stored_action_set] - Contains pre-computed optimization actions and topology configurations for the optimizer.

[Types][packages.interfaces_pkg.src.toop_engine_interfaces.types] - Provides type definitions and metric types used throughout the optimization engine.

## Message Interfaces
[Loadflow Commands][packages.interfaces_pkg.src.toop_engine_interfaces.messages.lf_service.loadflow_commands] - Message interface for requesting and controlling loadflow calculations.

[Loadflow Heartbeat][packages.interfaces_pkg.src.toop_engine_interfaces.messages.lf_service.loadflow_heartbeat] - Heartbeat messages for monitoring loadflow service health and status.

[Preprocess Commands][packages.interfaces_pkg.src.toop_engine_interfaces.messages.preprocess.preprocess_commands] - Message interface for preprocessing operations and commands.

[Preprocess Heartbeat][packages.interfaces_pkg.src.toop_engine_interfaces.messages.preprocess.preprocess_heartbeat] - Heartbeat messages for monitoring preprocessing service health and status.
