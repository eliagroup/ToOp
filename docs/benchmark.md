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

## Parent selection study

The parent-selection study compares MAP-Elites policies under identical DC
optimization settings and shared random seeds. The default configuration uses
`data/complex_grid/grid.xiidm` with the Powsybl backend, preprocesses the grid
once, and then runs every enabled mode-seed combination sequentially.

The checked-in smoke preset evaluates three policies across two shared seeds:

- `uniform` (`UNIi`): samples occupied repertoire candidates uniformly.
- `ucb_snapshot` (`UCBs`): samples from one GPU-friendly UCB score snapshot per emission.
- `greedy` (`G`): samples uniformly among occupied candidates with the highest fitness.

Other available modes can be enabled in
`toop-engine-benchmark/configs/parent_selection.yaml`:

- `uniform_cell` (`UNIc`): samples an occupied cell uniformly, then an occupied candidate within that cell.
- `ucb` (`UCBc`): selects cells by empirical survival rate plus an exact UCB exploration term.
- `ucb_batched` (`UCBb`): approximates UCB with vectorized virtual-pull blocks.
- `exploitation` (`Ec`): favors cells with the highest empirical survival rate.
- `exploration` (`Xc`): prioritizes unvisited and then least-selected cells.

The outer `tqdm` bar reports study progress. The nested bar reports epoch-level
fitness, QD score, archive coverage, and evaluation throughput.

Run the study from the repository root:

```bash
uv run python toop-engine-benchmark/benchmark_parent_selection.py
```

For a quick DC-only smoke run, override the runtime and seeds:

```bash
uv run python toop-engine-benchmark/benchmark_parent_selection.py ga.runtime_seconds=10 seeds=[103]
```

### Repertoire descriptor resolution

The default study enables nodal-injection optimization and PST mutation. Its
`descriptor_resolution` configuration can automatically append
`pst_switching_distance` or `pst_activated` to the explicitly configured
MAP-Elites descriptors after each grid has been preprocessed. The default
benchmark uses `pst_activated`. This lets every grid use a resolution that
matches its controllable PSTs instead of sharing a guessed fixed value.

For each controllable PST, the resolver determines the farthest tap position
from its starting tap. With $T$ timesteps, it creates cells for every integer
distance from zero through

$$
D_{max} = T \sum_i \max(s_i, n_i - 1 - s_i),
$$

where $s_i$ is the starting tap index and $n_i$ is the number of available tap
positions. Thus, the resolved PST descriptor has $D_{max} + 1$ cells. The
upper bound can leave unreachable cells for coupled PSTs, which is intentional:
the descriptor never silently clips a physically possible distance. A grid
without a controllable, movable PST is rejected when this automatic descriptor
is enabled.

`pst_activated` counts every movable PST whose tap differs from its initial
position at each timestep. With $N_{movable}$ movable PSTs, it therefore uses
$T \cdot N_{movable} + 1$ cells for all integer counts from zero through
$T \cdot N_{movable}$; it is not a binary descriptor.

`descriptor_resolution.max_logical_cells` defaults to `100000` and is checked
before JAX allocates the repertoire. It limits the product of all descriptor
dimensions; the resulting flat repertoire has that product multiplied by
`cell_depth` slots. Increase the limit deliberately if a grid needs a larger
repertoire.

### Repertoire trajectory artifacts

Every completed run writes `repertoire_trajectory.jsonl` alongside
`trajectory.jsonl`, `res.json`, and `archive_cells.json`. It has one sparse
snapshot for epoch `0` and one after every completed optimizer epoch. Each
record contains the epoch, JAX iteration, and aligned `cell_indices`,
`elite_fitnesses`, and `selection_counts` arrays. Cell indices refer to the
logical repertoire shape stored in `run_manifest.json`.

Snapshots retain cells with either a finite elite or a positive count. A
missing cell means an empty fitness and selection count zero; a `null` fitness
with a positive count means a previously selected cell has no current elite.
Selection counts are cumulative parent-selection counts, not per-epoch deltas.
Run manifests use schema version `2` and persist the descriptor names, cell
dimensions, depth, logical-cell count, and the configured automatic-resolution
policy under `repertoire_layout`.

`notebooks/parent_selection_results.ipynb` uses these snapshots for its
interactive repertoire analyzer. It can project any two to four configured
descriptors into all descriptor-pair heatmaps, switch between fitness and
cumulative selection counts, and aggregate available seeds at a selected
epoch. Older studies have no historical snapshot data: the notebook can show a
final fitness fallback from `archive_cells.json`, but count history and earlier
epochs require a newly run study.

Each run writes a `trajectory.jsonl` record at initialization and after every
optimizer epoch. Records include best fitness, observed metrics, cell and
candidate coverage, raw and baseline-improvement QD scores, evaluation
throughput, and parent-selection feedback statistics. The final optimizer result
is written as `res.json`.

AC validation is disabled by default. Enable it to validate the final top-k DC
candidates for every completed run:

```bash
uv run python toop-engine-benchmark/benchmark_parent_selection.py ac_validation.enabled=true ac_validation.k_best_topos=3
```

Add future real or larger grids as entries in
`toop-engine-benchmark/configs/parent_selection.yaml`.
