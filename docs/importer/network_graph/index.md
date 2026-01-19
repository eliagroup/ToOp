# Network Graph for Topologies
## Motivation
To use the optimizer, an [AssetTopology][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology] is needed. The [AssetTopology][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology] stores information about a Topology, incl. Station layouts, assets and their topological state. The creation of an [AssetTopology][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology] is easy if an asset is directly connected to a busbar with no additional nodes and switches in between (e.g., the UCTE Format).
In a closer-to-reality Node-Breaker Model like CGMES, assets can be connected to multiple busbars via switches and additional non-busbar-nodes. With this info we can create possible splits for the optimizer and directly translate the optimization results into switching actions on the actual grid.
The network graph model maps all important elements to the [AssetTopology][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology] logic.

## Core Concepts

The data classes for the Network Graph:

1. **[`NetworkGraphData`][packages.importer_pkg.src.toop_engine_importer.network_graph.NetworkGraphData]**
   This class is used as a starting point. All information needed to map the elements is included in these dataframes. The graph's edges are divided into switches and branches, as the "open" state of the switch is central information to get the zero impedance connection (zero impedance connection = currently connected via closed switches).
   - **[`NodeSchema`][packages.importer_pkg.src.toop_engine_importer.network_graph.NodeSchema]**: Defines all the nodes.
        [`NODE_TYPES`][packages.importer_pkg.src.toop_engine_importer.network_graph.NODE_TYPES]: "busbar" a true physical busbar you can touch (but probably shouldn't)
                   "node" is a connection point between elements or fictive nodes where two TSO meet (border nodes)
   - **[`SwitchSchema`][packages.importer_pkg.src.toop_engine_importer.network_graph.SwitchSchema]**: This is core information. If no switches are present, the mapping will do nothing.
        [`SWITCH_TYPES`][packages.importer_pkg.src.toop_engine_importer.network_graph.SWITCH_TYPES]: "DISCONNECTOR" (Power Switch), "BREAKER" (Non Power Switch)
   - **[`BranchSchema`][packages.importer_pkg.src.toop_engine_importer.network_graph.BranchSchema]**: Optional. If only a substation is represented, a branch would be represented in the [`NodeAssetSchema`][packages.importer_pkg.src.toop_engine_importer.network_graph.NodeAssetSchema] at the outer edge of the network.
        [`BRANCH_TYPES`][packages.importer_pkg.src.toop_engine_importer.network_graph.BRANCH_TYPES] depend on the framework you use.
        [`BRANCH_TYPES_POWSYBL`][packages.importer_pkg.src.toop_engine_importer.network_graph.BRANCH_TYPES_POWSYBL]: "LINE", "TWO_WINDING_TRANSFORMER", "PHASE_SHIFTER", "TWO_WINDING_TRANSFORMER_WITH_TAP_CHANGER", "THREE_WINDINGS_TRANSFORMER",
        [`BRANCH_TYPES_PANDAPOWER`][packages.importer_pkg.src.toop_engine_importer.network_graph.BRANCH_TYPES_PANDAPOWER]: "line", "trafo", "trafo3w", "dcline", "tcsc", "impedance"
   - **[`NodeAssetSchema`][packages.importer_pkg.src.toop_engine_importer.network_graph.NodeAssetSchema]**: Optional. Mainly used to represent node assets like generators, loads, etc. It can also be used to represent border lines of the network (either of a substation or a line leaving the network).
   - **[`HelperBranchSchema`][packages.importer_pkg.src.toop_engine_importer.network_graph.HelperBranchSchema]**: A concept mainly introduced to support the exact layout of `powsybl.net.get_node_breaker_topology(substation_info.voltage_level_id)`. For example, a `BusbarSection` in the powsybl NodeBreakerTopology is a dead-end node. The true connection hub is a helper node, where all switches are connected (again with a helper node in between).

2. **[`BusbarConnectionInfo`][packages.importer_pkg.src.toop_engine_importer.network_graph.BusbarConnectionInfo] and **[`EdgeConnectionInfo`][packages.importer_pkg.src.toop_engine_importer.network_graph.EdgeConnectionInfo]**
   These data classes are initialized blank. The data gets filled step by step using the [`run_default_filter_strategy`][packages.importer_pkg.src.toop_engine_importer.network_graph.run_default_filter_strategy]. Most of the data is used for the [AssetTopology][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology], and some for tracking dependencies between the filter process.

## Workflow: From a Grid Model to the Asset Topology

1. **Extract the Grid or a Station into the NetworkGraphData Format:**
   - [`pandapower_network_to_graph`][packages.importer_pkg.src.toop_engine_importer.network_graph.get_network_graph]
   - [`powsybl_station_to_graph`][packages.importer_pkg.src.toop_engine_importer.network_graph.get_node_breaker_topology_graph]

2. **Enrich NetworkGraphData with Generic Data:**
   - Add [`filter_weights`][packages.importer_pkg.src.toop_engine_importer.network_graph.set_all_weights] to NetworkGraphData – preparation for the filter process.
     Sets weights based on the three DataFrames:
     - [`NetworkGraphData.branches`][packages.importer_pkg.src.toop_engine_importer.network_graph.NetworkGraphData]
     - [`NetworkGraphData.switches`][packages.importer_pkg.src.toop_engine_importer.network_graph.NetworkGraphData]
     - [`NetworkGraphData.helper_branches`][packages.importer_pkg.src.toop_engine_importer.network_graph.NetworkGraphData]

3. **Generate the Graph:**
   A `networkx.Graph` is generated based on the [`NetworkGraphData`][packages.importer_pkg.src.toop_engine_importer.network_graph.NetworkGraphData] class. [`BusbarConnectionInfo`][packages.importer_pkg.src.toop_engine_importer.network_graph.BusbarConnectionInfo] (node only) and [`EdgeConnectionInfo`][packages.importer_pkg.src.toop_engine_importer.network_graph.EdgeConnectionInfo] (edge only) are initialized for all elements.
   - Add a Substation ID if missing (pandapower has no substation logic, but CGMES import does).

4. **Run [`run_default_filter_strategy`][packages.importer_pkg.src.toop_engine_importer.network_graph.run_default_filter_strategy]:**
   Here the mapping logic is implemented. The order of these steps is important, as there are dependencies between the steps.
   - **[`set_switch_busbar_connection_info`][packages.importer_pkg.src.toop_engine_importer.network_graph.set_switch_busbar_connection_info]**
     Sets a filter value `busbar_weight`. It is applied to all switches directly connected to a busbar and is used as an indicator during the shortest path calculation to indicate that a busbar has been found.
   - **[`set_bay_weights`][packages.importer_pkg.src.toop_engine_importer.network_graph.set_bay_weights]**
     The mapping of bays is a core functionality of the network graph module. We want to know which switches belong to a bay of an Branch/node-asset (e.g. Generator or in a reduced grid could also be a branch). After the algorithm is done, the `bay_id` is used to identify bays, where the `bay_id` is set to the Branch/node-asset id that it belongs to.

     The search starts at an node-asset or branch. From there, the shortest path is calculated to all busbars with a cutoff limit. The weights used for the shortest path are `busbar_weight` and `bay_weight`. The cutoff limit is chosen so that `busbar_weight` is encountered only once.
     The `bay_weight` (default value set by `filter_weights`) is high for branches that leave a station; hence the cutoff is always within a station. To find the branches, the "from" and "to" nodes are used separately. Finally, it sets the `bay_id` (needed to map the switches to an asset) and the `bay_weight`.
   - **[`set_empty_bay_weights`][packages.importer_pkg.src.toop_engine_importer.network_graph.filter_strategy.empty_bay.set_empty_bay_weights]**
     An empty bay is a collection of switches that have no asset connected at the end. This can happen e.g. if an out of service asset is not being exported.
     Finding and categorizing couplers depends on all bays weights being set.
   - **[`set_connectable_busbars`][packages.importer_pkg.src.toop_engine_importer.network_graph.set_connectable_busbars]**
     Uses `coupler_weight` (default value set by `filter_weights`) and the updated `bay_weight` to mark couplers between busbars. The `bay_weight` needs to be set first; otherwise, the bay connection is always an option to "couple" busbars. The `bay_weight` now blocks these paths.
     Sets the connectable busbars result from `calculate_connectable_busbars` to the [`BusbarConnectionInfo`][packages.importer_pkg.src.toop_engine_importer.network_graph.BusbarConnectionInfo]. It sets `connectable_busbars` and `connectable_busbars_node_ids` in the same order.
   - **[`set_all_busbar_coupling_switches`][packages.importer_pkg.src.toop_engine_importer.network_graph.filter_strategy.switches.set_all_busbar_coupling_switches]**
     Set all connection paths between busbars, be it "BREAKER" or "DISCONNECTOR".
     These paths will be busbar coupler, cross coupler or busbar disconnector.
   - **[`set_zero_impedance_connected`][packages.importer_pkg.src.toop_engine_importer.network_graph.set_zero_impedance_connected]**
     Sets the zero impedance connection for assets and busbars.

5. **Extract the Filled [`BusbarConnectionInfo`][packages.importer_pkg.src.toop_engine_importer.network_graph.BusbarConnectionInfo] and [`EdgeConnectionInfo`][packages.importer_pkg.src.toop_engine_importer.network_graph.EdgeConnectionInfo] from the Network.**

6. **Create the AssetTopology** using the [`NetworkGraphData`][packages.importer_pkg.src.toop_engine_importer.network_graph.NetworkGraphData], [`BusbarConnectionInfo`][packages.importer_pkg.src.toop_engine_importer.network_graph.BusbarConnectionInfo], and [`EdgeConnectionInfo`][packages.importer_pkg.src.toop_engine_importer.network_graph.EdgeConnectionInfo].

**Note:** A star equivalent transformation for three-winding transformers is not needed before using this module. The graph module only needs the connection and does no calculation.

## Missing Features / Limitations

1. PSTs are not implemented, currently outside of the station due to the branch [`bay_weight`][packages.importer_pkg.src.toop_engine_importer.network_graph.set_all_weights].
