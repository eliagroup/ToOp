# How to use this package

## Overview 
There are 6 packages:

1. [Importer](./importer/intro.md): This packages handles grid import and additional data files necessary to perform topology optimization. It relies on the packages `PandaPower` and `PyPowsybl` for grid import, which we refer to as *backends*.
2. [DC Solver](./dc_solver/intro.md): This packages implements an accelerated DC loadflow solver with GPU support.
3. [Topology Optimizer](./topology_optimizer/intro.md): This package implements a topology optimizer for electrical transmission grids. 
It uses multi-objective optimization to determine reconfigurations of substations that reduce for example line overloads.
4. [Interfaces](./interfaces/intro.md): This package provides a set of abstractions and adapters to enable interoperability between different grid modeling tools and data formats.
5. [Grid Helpers](./grid_helpers/intro.md): Contains several helping functions to streamline the use of both backends for grid importing.
6. [Contingency  Analysis](./contingency_analysis/intro.md): Allows to run post-optimization evaluation of topologies by passing them back to the backends (`PandaPower` and `PyPowsybl`)

ToOp delivers as an output AC validated Optimization results in the form of:

- The changes needed to get from the base grid to the optimized topology. As a Diff (e.g. dgs format) or a full Grid (e.g. Pandapower or PyPowSyBl)
- PyPowSyBl only: Two sets of Single Line Diagrams (SLD). The Layout of the Substation before and after the Split.
- AC and DC load-flow tables before and after the split.
- A map elites repertoire for AC and DC, with the best topologies picked by the optimizer.

If you want to use all capabilities of this repository, you need to understand how the packages interact and integrate with each other.
To see this in action, you can either use the direct python interface or run the whole repo as a composition of Docker containers.

To see the whole repo in action, check out the [`end-to-end pipeline`](https://github.com/eliagroup/ToOp/blob/develop/notebooks/e2e_pipeline.ipynb).
If you want to use Kafka workers instead, read on.

## Kafka messaging

!!! Work-in-progress

    We will extend the usage guide of this package incrementally. For now, please refer to the example notebooks.
    If you are interested in creating Kafka workers, inspect the interfaces for the Kafka topics and trace their usage.

### Step 1: Import a grid file

To use the tool, you need to import the grid into your file. This entails two fundamental steps:

- The [convert_file][packages.importer_pkg.src.toop_engine_importer.pypowsybl_import.preprocessing.convert_file] function, taking an import command. This will prepare masks and perform initial preprocessing tasks in the grid.
- The [load_grid][packages.dc_solver_pkg.src.toop_engine_dc_solver.preprocess.load_grid] function writes data into the data folder, creating a folder with several artifacts. The most relevant one being the `static_information.hdf5` which holds the data relevant for the DC GPU optimizer.

### Step 2: Perform an optimization

For this, both the DC optimizer and AC validator should be running at the same time to keep runtime constraints. In principle they could be run one after the other, which would greatly simplify deployment, but as the whole point of this project is to have a fast response, we run them at the same time.
Note: The AC optimizer consumes the results of the DC optimizer. Therefore the AC Optimizer should run a bit longer/after to be able to consume the latest DC tppologies.

This is set up through kafka messaging. For the beginning we will run everything on the same machine, but in principle you can deploy this on a cluster.

First, set up kafka with 
```
cd dev-deployment
docker-compose up
```

This will spin up a kafka server with 6 topics, 3 importer topics and 3 optimizer topics. The importer topics are only required in case you want to run the importer as a kafka component, but they do no harm if you just want to run an optimization. For the optimizer, we have [`commands`][packages.topology_optimizer_pkg.src.toop_engine_topology_optimizer.interfaces.messages.commands] which is a topic where usually only an [`StartOptimizationCommand`][packages.topology_optimizer_pkg.src.toop_engine_topology_optimizer.interfaces.messages.commands.StartOptimizationCommand] will be sent to. Also there is the `results` topic, which contains topologies with their metrics. Both the DC and AC stage push their topologies there, and the AC stage also pulls DC topologies for validation.

Now, we need to spin up the DC and AC optimizer:

```
cd packages/topology_optimizer_pkg/dc/worker
python3 worker.py --processed_gridfile_folder=/path/to/your/import/
```

```
cd packages/topology_optimizer_pkg/ac/
python3 worker.py --processed_gridfile_folder=/path/to/your/import/
```

They should connect to the kafka that was spun up earlier on their own, you should see new consumers in the kafka logs.

Now, we need to send an optimization command and listen for results.
