# Quickstart

The ToOp repository contains a lot of functionality, which can be overwhelming if you look at it for the first time.
But you do not need to understand all of it to use the parts that matter to you.
If you would like to grasp the key concepts and data structures first, have a look at [Usage](./usage.md).
If ToOp is not installed yet, start with the [installation guide](./installation.md).

## A batched DC load flow in a few lines

Write the IEEE 57-bus test grid to a folder, preprocess it for the GPU solver and compute the flow on every branch:

```python
from pathlib import Path

import jax
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.example_grids import case57_data_powsybl
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid, run_initial_loadflow

jax.config.update("jax_enable_x64", True)

folder = Path("data/case57")
case57_data_powsybl(folder)  # write the example grid
stats, static_information, network_data = load_grid(DirFileSystem(str(folder)))

static_information, metrics = run_initial_loadflow(static_information)
print(static_information.dynamic_information.unsplit_flow.shape)  # (timesteps, branches) -> (1, 82)
```

## The whole pipeline: import, optimize, validate

The pipeline helpers take a grid file all the way to AC-validated topologies:

```python
from toop_engine_topology_optimizer.benchmark.benchmark_utils import (
    perform_ac_analysis,
    run_dc_optimization_stage,
    run_preprocessing,
)

run_preprocessing(importer_parameters, data_folder, preprocessing_parameters)  # import + DC preprocessing
run_dc_optimization_stage(dc_config, run_dir)  # genetic search on the GPU
perform_ac_analysis(data_folder, run_dir, ac_validation_cfg=ac_validation_cfg)  # AC validation of the best
```

Most workflows start like this, by turning a raw grid file into a processed grid folder. The importer creates the backend grid snapshot, masks, loadflow parameters and topology metadata; [`load_grid`][toop_engine_dc_solver.preprocess.load_grid] then adds `static_information.hdf5`, `action_set.json`, `action_set_diffs.hdf5` and the final `nminus1_definition.json` used by the solver and optimizer.

## Learn by example

The configuration objects above are set up step by step in the example notebooks, which are the best place to continue:

<!-- markdown-link-check-disable -->
| Notebook | What it shows |
| --- | --- |
| [`example1_dc_loadflow_example`](https://github.com/eliagroup/ToOp/blob/main/notebooks/example1_dc_loadflow_example.ipynb) | the DC load flow above, with more detail – a good first step |
| [`example2_small_grid_toop`](https://github.com/eliagroup/ToOp/blob/main/notebooks/example2_small_grid_toop.ipynb) | optimizing a small node-breaker grid and looking at the changed substation |
| [`example3_e2e_pipeline`](https://github.com/eliagroup/ToOp/blob/main/notebooks/example3_e2e_pipeline.ipynb) | the whole pipeline on any PyPowSyBl-compatible grid file of your choice |
| [`example4_data_contracts`](https://github.com/eliagroup/ToOp/blob/main/notebooks/example4_data_contracts.ipynb) | the same journey with the packages called directly, so every hand-over is visible |
| [`example5_e2e_ac_validation`](https://github.com/eliagroup/ToOp/blob/main/notebooks/example5_e2e_ac_validation.ipynb) | the whole pipeline called by hand, AC-validated end to end |
<!-- markdown-link-check-enable -->

To see how the stages fit together, take the tour in the [interactive pipeline explorer](architecture/pipeline_explorer.html).
