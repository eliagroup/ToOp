# Package Structure and Interactions

## Core Components

### DC Optimizer (`dc/`)
- **`worker/`**: Kafka-based worker infrastructure for distributed optimization
  - [`optimizer.py`][packages.topology_optimizer_pkg.src.toop_engine_topology_optimizer.dc.worker.optimizer]: Core optimization logic and genetic algorithm execution
  - [`worker.py`][packages.topology_optimizer_pkg.src.toop_engine_topology_optimizer.dc.worker.worker]: Kafka consumer/producer for distributed coordination
- **`genetic_functions/`**: Evolutionary operators and fitness evaluation
- **`repertoire/`**: Map-Elites repertoire management and visualization

### AC Optimizer (`ac/`)
- **[`worker.py`][packages.topology_optimizer_pkg.src.toop_engine_topology_optimizer.ac.worker]**: AC validation worker with database storage
- **[`optimizer.py`][packages.topology_optimizer_pkg.src.toop_engine_topology_optimizer.ac.optimizer]**: AC loadflow execution and strategy management
- **[`evolution_functions.py`][packages.topology_optimizer_pkg.src.toop_engine_topology_optimizer.ac.evolution_functions]**: Evolution operators (pull, reconnect, close_coupler)
- **[`scoring_functions.py`][packages.topology_optimizer_pkg.src.toop_engine_topology_optimizer.ac.scoring_functions]**: AC power flow computation and metrics

### Interfaces (`interfaces/`)
- **Message protocols**: Kafka message definitions for inter-optimizer communication
- **Parameters**: Configuration classes for [`DC`][packages.topology_optimizer_pkg.src.toop_engine_topology_optimizer.interfaces.messages.dc_params.DCOptimizerParameters] and [`AC`][packages.topology_optimizer_pkg.src.toop_engine_topology_optimizer.interfaces.messages.ac_params.ACOptimizerParameters] optimizer parameters
- **Results**: Standardized topology and metrics data structures

## Integration with Other Packages

### DC Solver Integration
The topology optimizer leverages the [`DC Solver`](../dc_solver/intro.md) package for high-performance loadflow computation:

- **PTDF/BSDF matrices**: Pre-computed sensitivity matrices for rapid topology evaluation
- **Batch processing**: Simultaneous evaluation of multiple topologies and contingencies
- **Injection bruteforcing**: Efficient exploration of injection patterns
- **Cross-coupler flows**: Advanced switching modeling

### Interfaces Package
Uses the [`Interfaces`](../interfaces/intro.md) package for standardized data structures:

- **[`Asset Topology`][packages.interfaces_pkg.src.toop_engine_interfaces.asset_topology]**: Physical substation and switching device models
- **[`Stored Action Set`][packages.interfaces_pkg.src.toop_engine_interfaces.stored_action_set]**: Pre-computed topology actions and switching sequences
- **[`N-1 Definition`][packages.interfaces_pkg.src.toop_engine_interfaces.nminus1_definition]**: Contingency analysis specifications
- **[`Loadflow Results`][packages.interfaces_pkg.src.toop_engine_interfaces.loadflow_results]**: Standardized power flow output formats

### Grid Import Integration
Works with the [`Importer`](../importer/intro.md) package for data preprocessing:

- **Static Information**: Grid topology and electrical parameters
- **Network Data**: Processed grid models compatible with optimization algorithms
- **Action enumeration**: Systematic generation of feasible switching actions
