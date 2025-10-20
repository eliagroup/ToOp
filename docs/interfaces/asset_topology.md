# Asset Topology

The Asset Topology is a central object for the Topology optimizer. It holds the topological information mapped from the node-breaker to the bus-branch model and back, enabling translation of topological actions between models using real switches.

Asset Topology is essential when stations do not allow free assignment of lines to busbars. It helps track valid assignments and ensures correct topology application.

## Class Structure

- [`Strategy`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.Strategy]  
  Collection of time steps, each represented by a [`Topology`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.Topology].

- [`Topology`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.Topology]  
  Contains multiple [`Station`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.Station] objects and optional [`AssetSetpoint`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.AssetSetpoint] objects.

- [`Station`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.Station]  
  Contains lists of [`Busbar`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.Busbar], [`BusbarCoupler`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.BusbarCoupler], and [`SwitchableAsset`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.SwitchableAsset].  
  Includes asset_switching_table, the current switch connection layout and asset_connectivity, all possible selections.

- [`Busbar`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.Busbar]  
  Represents a single busbar in a station.

- [`BusbarCoupler`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.BusbarCoupler]  
  Represents a coupler connecting two [`Busbar`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.Busbar].
  Note: the current implementation only supports a busbar connection between two busbars. It is planned to add an asset bay for both connection sides.

- [`SwitchableAsset`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.SwitchableAsset]  
  Represents an asset (line, transformer, generator, etc.) that can be switched. You may leave out non-switchable assets or assign them to a single busbar to have a complete representation of the physical [`Station`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.Station].

- [`AssetBay`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.AssetBay]  
  Describes the physical connection (switches) between an asset and busbars. It may contain breaker and disconnector switches.

  The AssetBay class currently supports a bay setup with a Disconnector on [`SwitchableAsset`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.SwitchableAsset] - Breaker - multiple Disconnectors on [`Busbar`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.Busbar].

  - The Disconnector on the line side (example Line1) is supported, as this is commonly found in CGMES data, but is not further used in the current implementation.

  - The Breaker of a branch is expected to be the one that connects and disconnects a line. Setups like T1 should have a selection process to decide which breaker will be written into the AssetBay class.

  - The Disconnectors on [`Busbar`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.Busbar] are selector switches, where only one of them should be closed at any time. Any preprocessing should find double connections, as this 
  will break the later assumption that a [`Busbar`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.Busbar] split can be performed by opening the [`BusbarCoupler`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.BusbarCoupler].
  
  ![Example of AssetBay configurations and data issues](src/asset_bay_example.png){width=50%}

  Note: An AssetBay expects that one Asset has its own bay. Combinations where two assets use shared switches are not supported.
    
    ![Example of unsupported AssetBay configuration](src/asset_bay_example_not_supported.png){width=50%}
    
    *Example: AssetBay configuration with two assets sharing switches (not supported).*

- [`AssetSetpoint`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.AssetSetpoint]  
  Represents an asset with a setpoint (e.g., PST or HVDC).

- [`RealizedStation`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.RealizedStation]  
  Contains a station and the changes made to it.

- [`RealizedTopology`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology.RealizedTopology]  
  Contains a topology and the changes made to it.

---

## API Reference

See the [`Asset Topology Reference`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology] for full class and method documentation.

## How to use / Implementation

To populate Asset Topology data from grid models, use the [`Network Graph module`][packages.importer_pkg.src.toop_engine_importer.network_graph].  

A Pandapower to Asset Topology implementation is found in the importer: [`get_list_of_stations_ids`][packages.importer_pkg.src.toop_engine_importer.pandapower_import.asset_topology.get_list_of_stations_ids]

Note: the Pandapower version does not currently use the Network Graph module.

A PyPowSyBl to Asset Topology implementation is found in the importer Network Graph module: [`get_topology`][packages.importer_pkg.src.toop_engine_importer.network_graph.powsybl_station_to_graph]
