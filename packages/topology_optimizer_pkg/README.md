# Topology Optimizer

This package implements a topology optimizer for electrical transmission grids using a two-stage approach. It employs multi-objective optimization to determine substation reconfigurations that reduce line overloads and improve grid stability through reconfiguration of branches in a substation + opening a busbar coupler and branch disconnections.

## Overview

The topology optimizer consists of two main optimization stages working in a coordinated pipeline:

### 1. DC Stage (High-Performance Screening)
The DC optimizer performs rapid exploration of the topology space using an accelerated DC loadflow solver with GPU support. It uses a **discrete Map-Elites genetic algorithm** to maintain a diverse repertoire of solutions across multiple objectives:

- **Fitness metrics**: Line overload energy, switching distance, number of split substations
- **Search algorithm**: Quality-diversity optimization maintaining solutions across descriptor dimensions
- **Performance**: GPU-accelerated computation using JAX for massive parallel evaluation
- **Output**: Pareto-optimal topologies sent to the AC stage for validation

Entry point: [`initialize_optimization`][packages.topology_optimizer_pkg.src.toop_engine_topology_optimizer.dc.worker.optimizer.initialize_optimization]

### 2. AC Stage (Validation and Refinement)
The AC optimizer validates promising DC solutions using full AC loadflow calculations and applies further evolutionary refinements:

- **Validation**: Full AC power flow analysis with N-1 contingency checking. Use [`optimization_loop`][packages.topology_optimizer_pkg.src.toop_engine_topology_optimizer.ac.worker.optimization_loop]  
- **Evolution operators**: Pull (from DC), reconnection, coupler closing: Use [`evolution_try`][packages.topology_optimizer_pkg.src.toop_engine_topology_optimizer.ac.evolution_functions.evolution_try]  
- **[Early stopping](../../docs/topology_optimizer/ac/early_stopping.md)**: Rejection based on worst-case overload comparison
- **[Selection strategy](../../docs/topology_optimizer/ac/select_strategy.md)**: Filtering method using median, dominator, and discriminator filters

## Key Data Structures

### Topology Representation
- **Actions**: List of substation switching indices from the [`ActionSet`][packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.action_set]
- **Disconnections**: Branch outage specifications for N-1 analysis
- **PST Setpoints**: Phase-shifting transformer positions
- **Metrics**: Multi-objective fitness values and constraint violations

### Strategy Collections  
- **DC Repertoire**: Map-Elites grid maintaining diversity across descriptor dimensions
- **AC Database**: SQLite storage for validated topologies with loadflow references
- **Message Protocols**: Standardized formats for inter-optimizer communication

## Prerequisites
The optimization process requires preprocessed grid data from the **[`Importer`](../../docs/importer/intro.md)** package:

1. **Static Information**: Grid electrical parameters and topology
2. **Action Set**: Enumerated switching possibilities  
3. **N-1 Definition**: Contingency analysis specifications

## Running an Optimization

### Method 1: Kafka-based Distributed Execution
```bash
# Start Kafka infrastructure
cd dev-deployment
docker-compose up

# Launch DC optimizer worker
cd packages/topology_optimizer_pkg
python -m topology_optimizer.dc.worker.worker \
    --processed_gridfile_folder=/path/to/preprocessed/data

# Launch AC optimizer worker  
python -m topology_optimizer.ac.worker \
    --processed_gridfile_folder=/path/to/preprocessed/data \
    --loadflow_result_folder=/path/to/results
```

### Method 2: Direct Python Integration
```python
from topology_optimizer.dc.worker.optimizer import initialize_optimization, run_epoch
from topology_optimizer.interfaces.messages.dc_params import DCOptimizerParameters

# Configure optimization parameters
params = DCOptimizerParameters(
    ga_config=BatchedMEParameters(runtime_seconds=300),
    loadflow_solver_config=LoadflowSolverParameters(max_num_splits=4)
)

# Initialize and run optimization
optimizer_data, stats, initial_strategy = initialize_optimization(
    params=params,
    optimization_id="my_optimization", 
    static_information_files=["grid_data.pkl"]
)
```

## Advanced Topics

- **[AC Selection Strategy](../../docs/topology_optimizer/ac/select_strategy.md)**: Sophisticated filtering for AC candidate selection
- **[Early Stopping](../../docs/topology_optimizer/ac/early_stopping.md)**: Efficient topology rejection in N-1 analysis  
- **[DC Solver Configuration](../../docs/dc_solver/intro.md)**: GPU optimization and batch processing parameters
- **[Interface Definitions](../../docs/interfaces/intro.md)**: Data structure specifications and message protocols
