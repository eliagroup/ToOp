# Benchmark

ToOp comes with a Python file `toop-engine-benchmark/benchmark_toop.py` to benchmark and profile it.
The script supports:

1. Multirun to sweep across different parameter configurations
2. Profiling of all stages (Preprocessing, Optimization, Validation)
3. Insights into the performance of the GPU-based topology optimizer stage:
    - Number of computed injection combinations
    - Number of computed topologies (per epoch)
    - Tensorboard traces to analyze the optimization in-depth
    - Plotting cell distribution of genetic algorithm
4. Assessment to propose the best parameter configuration using `toop-engine-benchmark/assess_benchmark.py`, including:
    - Timing metrics across runs for each stage
    - Best run based on metric
    - Best parameter combination

We use the common [hydra](https://hydra.cc/docs/intro/) library for multiruns and configuration parameter optimization.
It uses YAML files for its configuration which can be found in `toop-engine-benchmark/config/`.
These files pose the *default* config with which the benchmark runs.

You can run the a single benchmark with
```bash
uv run python -m toop-engine-benchmark.benchmark_toop
```
and read the results from the `benchmark_summary.json`.

In most cases, you probably want to run several benchmarks with different parameters.

## Multirun and parameter sweep

If you aim to find the best parameter configuration for ToOp and your grid file(s), you want to run the algorithm with several combinations of parameters.
You start a multirun by passing the multirun flag `--multirun` to hydra. At the project's root directory, we run:
```bash
uv run python -m toop-engine-benchmark.benchmark_toop --multirun grid=config_grid_node_breaker ga_config.runtime_seconds=10,20 ga_config.split_subs=2,5
```

This will run the benchmark on the grid node breaker example using two different runtimes for the genetic algorithm and two different maximum number of split substations, in total four different combinations.
The output directory of the benchmark is configured in `toop-engine-benchmark/configs/grid/config_grid_node_breaker.yaml` via the `output_dir` field.

## Extensibility

Since ToOp requires a large number of configuration parameters that rarely require tuning but may still investigated, we decided to prioritize usability over exhaustive parameter selection.
We restricted the configurable parameters to the subset we deem most relevant to investigate the overall performance.

If you wish to investigate other parameters, you can extend the main loop of `toop-engine-benchmark/benchmark_toop.py` to support this.
You need to perform three steps to extend the set of parameters that can be considered for hyperparameter search:
1. Create (or extend an existing) hydra YAML file with the parameter you would like to include.
2. Extend the main loop of the benchmark to pass the config's new parameter to its corresponding internal config (e.g. `ga_config`, `lf_config` etc.).
3. Vary the parameter via CLI multirun (or using hydra experiment configs).

For example, `ga_config` contains `mutation_repetition` which determines the number of candidates to mutate during an iteration.
One may want to adjust this to steer the algorithm to explore more randomly.

## Assessment

We include an assessment script that computes a number of interesting metrics to observe the performance over time.
These metrics could be displayed in a health dashboard of the software.

When prompted with a directory of results of a multirun, the assessment script assesses the hyperparameter sweep, providing the best combination found.
It also computes timing metrics of each stage of the pipeline.

Let's imagine you have previously run a multirun.
You can then assess it via:
```bash
uv run python -m toop-engine-benchmark.assess_benchmarks root=/workspaces/ToOp/data/grid_node_breaker/results/multirun save=/workspaces/ToOp/data/grid_node_breaker/results/aggregate_report.json print=true
```
