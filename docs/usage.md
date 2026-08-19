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
<!-- markdown-link-check-disable -->
To see the whole repo in action, check out the **[notebooks/example3_e2e_pipeline.ipynb](https://github.com/EliaGroup/ToOp/blob/main/notebooks/example3_e2e_pipeline.ipynb)**.
If you want to use Kafka workers instead, read on.
<!-- markdown-link-check-enable -->

## Running in a container

A container is the supported route on Windows, and a convenient one anywhere else, since it needs no
local Python. The image is built from the `Dockerfile` at the repository root — the same one the
VS Code dev container uses.

Build it once, and create a volume to hold the Python environment:

```bash
docker build -t toop:dev .
docker volume create toop-venv
```

Then start Jupyter Lab, with the repository mounted at `/app`:

```bash
docker run -d --name toop-jl -v "${PWD}:/app" -v toop-venv:/opt/venv -w /app --shm-size=2gb -p 127.0.0.1:8888:8888 -e JAX_PLATFORMS=cpu toop:dev uv run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token= --ServerApp.password= --ServerApp.root_dir=/app
```

Open `http://127.0.0.1:8888/lab` and run the notebooks in `notebooks/`. Stop it with
`docker rm -f toop-jl`.

For a shell, or to run the tests:

```bash
docker run --rm -it -v "${PWD}:/app" -v toop-venv:/opt/venv -w /app --shm-size=2gb toop:dev bash
# then, inside the container:
git config --global --add safe.directory /app
uv sync --all-groups --frozen
uv run pytest packages/interfaces_pkg/tests -q
```

The first `uv sync` takes a few minutes; afterwards it is a sub-second no-op, because the environment
persists in the `toop-venv` volume.

A few details that are easy to get wrong:

- **`--shm-size=2gb` is not optional.** Ray and JAX allocate through `/dev/shm`, and Docker's 64 MB
  default surfaces as opaque crashes rather than a clear out-of-space error.
- **The Jupyter server runs without a token**, so keep the port published on `127.0.0.1` as shown
  above and it stays unreachable from the local network.
- **Prefer an explicit `-n <N>` over `-n auto`** when running the larger test suites. One JAX-loading
  worker per core can outgrow the memory a laptop VM allows. `--dist loadgroup` remains mandatory
  whenever `-n` is used.
- **The environment is at `/opt/venv`, not `/app/.venv`.** `uv sync` and `uv run` are unaffected;
  only the location moves. See `.devcontainer/README.md` for why.
- **From Git Bash on Windows**, prefix the command with `MSYS_NO_PATHCONV=1` and use
  `"$(pwd -W):/app"`. MSYS rewrites container-absolute paths into Windows paths before `docker.exe`
  sees them, and the daemon rejects the result.

Editing a `.py` file on the host takes effect inside the container on the next import: the six
packages are installed editable and the repository is bind-mounted, not copied.

## Kafka messaging

!!! Work-in-progress

    We will extend the usage guide of this package incrementally. For now, please refer to the example notebooks.
    If you are interested in creating Kafka workers, inspect the interfaces for the Kafka topics and trace their usage.

### Step 1: Prepare a processed grid folder

To use the tool, you first create a processed grid folder and then derive the solver artifacts from it. This entails two fundamental steps:

- The [convert_file][toop_engine_importer.pypowsybl_import.preprocessing.convert_file] function takes an import command and writes the normalized backend grid file, masks, loadflow parameters, asset topology metadata, importer auxiliary data, and an initial `nminus1_definition.json`.
- The [load_grid][toop_engine_dc_solver.preprocess.load_grid] function consumes that processed grid folder and writes the solver-facing artifacts, most notably `static_information.hdf5`, `action_set.json`, `action_set_diffs.hdf5`, `static_information_stats.json`, and a refreshed `nminus1_definition.json`.

### Step 2: Perform an optimization

For this, both the DC optimizer and AC validator should be running at the same time to keep runtime constraints. In principle they could be run one after the other, which would greatly simplify deployment, but as the whole point of this project is to have a fast response, we run them at the same time.
Note: The AC optimizer consumes the results of the DC optimizer. Therefore the AC Optimizer should run a bit longer/after to be able to consume the latest DC tppologies.

This is set up through kafka messaging. For the beginning we will run everything on the same machine, but in principle you can deploy this on a cluster.

First, set up kafka with
```
cd dev-deployment
docker-compose up
```

This will spin up a kafka server with 6 topics, 3 importer topics and 3 optimizer topics. The importer topics are only required in case you want to run the importer as a kafka component, but they do no harm if you just want to run an optimization. For the optimizer, we have [`commands`][toop_engine_topology_optimizer.interfaces.messages.commands] which is a topic where usually only an [`StartOptimizationCommand`][toop_engine_topology_optimizer.interfaces.messages.commands.StartOptimizationCommand] will be sent to. Also there is the `results` topic, which contains topologies with their metrics. Both the DC and AC stage push their topologies there, and the AC stage also pulls DC topologies for validation.

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
