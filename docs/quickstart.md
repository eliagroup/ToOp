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

The pipeline helpers take a grid file all the way to AC-validated topologies. On a laptop CPU this takes about a minute:

```python
from pathlib import Path

from omegaconf import DictConfig
from toop_engine_grid_helpers.powsybl.example_grids import create_complex_grid_battery_hvdc_svc_3w_trafo
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    AreaSettings,
    CgmesImporterParameters,
    PreprocessParameters,
)
from toop_engine_topology_optimizer.benchmark.benchmark_utils import (
    get_run_dir,
    perform_ac_analysis,
    run_dc_optimization_stage,
    run_preprocessing,
)
from toop_engine_topology_optimizer.dc.main import CLIArgs
from toop_engine_topology_optimizer.interfaces.messages.commons import DescriptorDef
from toop_engine_topology_optimizer.interfaces.messages.dc_params import BatchedMEParameters, LoadflowSolverParameters

# 1. Write a small demo grid
grid_file = Path("data/demo/grid.xiidm")
data_folder = grid_file.with_suffix("")
data_folder.mkdir(parents=True, exist_ok=True)
create_complex_grid_battery_hvdc_svc_3w_trafo().save(grid_file)

# 2. Import it and prepare everything the GPU needs
importer_parameters = CgmesImporterParameters(
    grid_model_file=grid_file,
    data_folder=data_folder,
    area_settings=AreaSettings(cutoff_voltage=10, control_area=[""], view_area=[""], nminus1_area=[""]),
)
run_preprocessing(importer_parameters, data_folder, PreprocessParameters())

# 3. Search for good topologies for ten seconds
run_dir = get_run_dir(data_folder / "optimizer_snapshot")
dc_config = DictConfig(
    CLIArgs(
        ga_config=BatchedMEParameters(
            runtime_seconds=10,
            me_descriptors=(
                DescriptorDef(metric="split_subs", num_cells=3),
                DescriptorDef(metric="disconnected_branches", num_cells=3),
            ),
        ),
        lf_config=LoadflowSolverParameters(max_num_splits=2, max_num_disconnections=2),
        fixed_files=(str(data_folder / "static_information.hdf5"),),
        tensorboard_dir=str(run_dir / "tensorboard"),
        stats_dir=str(run_dir / "stats"),
    ).model_dump()
)
dc_config.update(
    task_name="demo",
    num_cuda_devices=0,  # set to 1 to run on a GPU
    omp_num_threads=4,
    xla_force_host_platform_device_count=None,
    double_precision=True,
)
run_dc_optimization_stage(dc_config, run_dir)

# 4. Check the three best topologies with a full AC contingency analysis
ac_config = DictConfig(
    {"n_processes": 1, "k_best_topos": 3, "critical_voltage_jump_percent": 5.0, "critical_va_diff_degree": 20.0}
)
print(perform_ac_analysis(data_folder, run_dir, ac_validation_cfg=ac_config))
```

Most workflows start like this, by turning a raw grid file into a processed grid folder. The importer creates the backend grid snapshot, masks, loadflow parameters and topology metadata; [`load_grid`][toop_engine_dc_solver.preprocess.load_grid] then adds `static_information.hdf5`, `action_set.json`, `action_set_diffs.hdf5` and the final `nminus1_definition.json` used by the solver and optimizer.

## Learn by example

The example notebooks explain every setting step by step and are the best place to continue:

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
