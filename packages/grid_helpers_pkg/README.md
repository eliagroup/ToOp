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
- **Helpers**: A standardized function set to get grid information. Heavy used by the [`PandaPowerBackend`][toop_engine_dc_solver.preprocess.pandapower.pandapower_backend.PandaPowerBackend]. 
- **Id helpers**: A fix due to Pandapower not having global ids. It combines the row id of the dataframe with the table name.
- **Import helpers**: Some Functions to fix data quality issues
- **Asset Topology helpers**: TODO: currently in Importer package


## PyPowSyBl Key Data Structures and Concepts
- **Loadflow parameter**:A collection of loadflow parameter used by the ToOp project.
- **Helpers**: A standardized function set to get grid information. Heavy used by the [`PowsyblBackend`][toop_engine_dc_solver.preprocess.powsybl.powsybl_backend.PowsyblBackend]. 
- **Asset Topology helpers**: An implementation of the [AssetTopology][toop_engine_interfaces.asset_topology], main entry: [`get_list_of_stations`][toop_engine_grid_helpers.powsybl.powsybl_asset_topo.get_list_of_stations]
- **Single line diagram (SLD)**: A modified version of the powsybl SLD, with a bright and dark mode. This could be integrated into powsybl itself as it's own package (java).
- **Polars DataFrame**: Once you have millions of rows in you dataframe, the PowSyBl (Java) to PyPowSybl converter (Java->C++->Python->Pandas) becomes slow. You can eliminate one large bottleneck by removing pandas from this data extraction path. Ideally the transfer from Java to PyPowSyBl would deliver in this case parquet as a format. The speed boost only starts with large amount of N-1 cases: 3000 Bus network with more than 500 N-1 cases. Below that the speed of collection is roughly equal.

## Electrical Circuit Groups for PyPowSyBl

The electrical circuit group feature identifies breaker-bounded connectivity in a PyPowSyBl bus-breaker topology and turns it into a fast lookup index for failure propagation queries.

At a high level, it can:

- Group branches, breaker switches, injections, and busbar sections into electrical circuit groups.
- Answer branch-driven queries such as "which elements fail if this branch is outaged?".
- Answer busbar-driven queries such as "which elements and switches are reachable behind this busbar section?".
- It returns in the case of the busbar the primary switches, the secondary elements and the secondary switches
- Reuse one cached identification result for many downstream queries instead of recomputing graph connectivity.
- Build a human-readable group map when you want to inspect the grouped assets directly.

The feature is designed for the PyPowSyBl bus-breaker model and uses breaker boundaries as the stopping condition. Disconnectors are not treated as retained connections during group detection, and fictitious switches can be kept or filtered depending on the use case.

This differs from the Powsybl porpagation logic, as this opens switches. With this logic
we can also get the affected elements e.g. for a bus-branch solver or the corresponding 
swithes for an Open-Rao exchange.

### What Gets Computed

`identify_circuit_groups` prepares a temporary working variant, updates switch retention so the relevant breaker topology is exposed, computes connected components on the bus-breaker graph, and returns:

- a lookup index for fast branch and busbar queries,
- a branch table annotated with `electrical_circuit_group`,
- a switch table annotated with `electrical_circuit_group_bus1` and `electrical_circuit_group_bus2`,
- an injection table annotated with `electrical_circuit_group`.

Once that result exists, the helper functions can cheaply answer repeated lookup questions without rebuilding the graph.

### Topology Intuition

The grouping stops at breakers. Assets connected through retained non-breaker topology belong to the same electrical circuit group, while breaker switches sit on the boundary between groups.

```text
     Branch A                    Branch B
        |                           |
        |                           |
   bus_a1 ---- disconnector ---- bus_mid ---- disconnector ---- bus_b1
                                         \
                                          \
                                           Branch C
                                              |
                                              |
                                           bus_c1

   breaker_a between external grid and bus_a1
   breaker_b between external grid and bus_b1
   breaker_c between external grid and bus_c1
```

In that simplified picture, `Branch A`, `Branch B`, and `Branch C` sit inside the same electrical circuit group because they remain connected through the retained bus-breaker topology. The breakers mark the boundaries where propagation stops.

Busbar lookups use the breaker boundary to expand from a busbar section into the asset groups behind the connected breakers:

```text
busbar section
     |
     | primary circuit group
     v
[ Group 12 ] --breaker--> [ Group 27 ]
                       \
                        \--breaker--> [ Group 31 ]

Query result for the busbar section:
- failing elements from Group 27 and Group 31
- failing switches on the breaker boundaries of Group 27 and Group 31
```

### Typical Workflow

1. Load a PyPowSyBl network in bus-breaker topology.
2. Replace three-winding transformers before running the feature.
3. Run `identify_circuit_groups(...)` once.
4. Reuse `lookup_index` for repeated branch or busbar queries.
5. Optionally build a circuit-group map for inspection or debugging.

### Example

```python
import pypowsybl

from toop_engine_grid_helpers.powsybl.network_graph.electrical_circuit_groups import (
    build_circuit_group_map,
    get_failing_elements_by_branch_ids,
    get_failing_switches_by_busbar_ids,
    identify_circuit_groups,
)

net = pypowsybl.network.load("grid.xiidm")
pypowsybl.network.replace_3_windings_transformers_with_3_2_windings_transformers(net)

identified = identify_circuit_groups(net=net, keep_fictitious=False)

# Reuse one cached lookup index for many queries.
lookup_index = identified.lookup_index

branch_failures = get_failing_elements_by_branch_ids(
    branch_ids=["LINE_12_34"],
    lookup_index=lookup_index,
    include_busbar_id=True,
)
busbar_switches = get_failing_switches_by_busbar_ids(
    busbar_section_ids=["BUSBAR_SECTION_1"],
    lookup_index=lookup_index,
)

# Optional inspection view keyed by electrical circuit group id.
circuit_group_map = build_circuit_group_map(identified)

print(branch_failures["LINE_12_34"])
print(busbar_switches["BUSBAR_SECTION_1"])
print(circuit_group_map[0])
```

### Important Constraints

- The current implementation requires three-winding transformers to be replaced before calling `identify_circuit_groups`.
- The feature is based on PyPowSyBl bus-breaker topology, not on a node-breaker abstraction from another backend.
- Query helpers expect ids that are present in the precomputed lookup index and raise a `ValueError` for unknown branch or busbar-section ids.
- The identification step computes the graph once; performance-sensitive workflows should reuse the returned lookup index instead of recomputing it per query.

### Integration Examples

For complete integration examples with other ToOp packages, see:

- **[DC Solver Examples](https://eliagroup.github.io/ToOp/quickstart/)**: Grid preprocessing and optimization setup
- **[Contingency Analysis](https://eliagroup.github.io/ToOp/contingency_analysis/intro/)**: N-1 analysis configuration
- **[Topology Optimizer](https://eliagroup.github.io/ToOp/topology_optimizer/intro/)**: Multi-objective switching optimization

## Reference Documentation

For detailed API documentation, see:

- **[Pandapower Helpers][toop_engine_grid_helpers.pandapower]**: Complete pandapower utility reference
- **[PyPowSyBL Helpers][toop_engine_grid_helpers.powsybl]**: Full pypowsybl functionality guide
- **Example Networks**: Comprehensive test network catalog
    - [Pandapower grids][toop_engine_grid_helpers.pandapower.example_grids]
    - [Powsybl grids][toop_engine_grid_helpers.powsybl.example_grids]
