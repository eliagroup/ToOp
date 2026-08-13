# Architecture diagrams

Architecture-as-code for the ToOp engine, written in [LikeC4](https://likec4.dev).
The `.c4` files in this folder are the source of truth. The PNGs under
`generated/` are rendered from them by
`scripts/render_architecture_diagrams.sh` and are **not committed** — CI renders
them before every docs build, so edit the model, never the images.

```
specification.c4        element kinds, tags, colours
model/
  01-externals.c4       who drives the engine, what goes in and out
  02-messaging.c4       Kafka and its topics
  03-storage.c4         the two shared filesystems and their contents
  04-engine.c4          the ToOp system itself; every file below extends it
  05-importer.c4        \
  06-dc-optimizer.c4     |  one per stage
  07-ac-validator.c4    /
  08-contingency.c4     \
  09-interfaces.c4       |  shared packages
  10-postprocess.c4     /
  11-relations.c4       every relation, grouped by the stage that drives it
  12-parameters.c4      the parameter object each stage takes
views/
  02-data-flow.c4 … 10-parameters.c4   one file per audience
  09-landscape.c4       holds `index` and the `overview` sequence
```

Elements that map onto a specific module or function carry a `link` to the
published API reference, so you can go from a box in the diagram to its
docstring in one click.

Those links resolve to mkdocstrings anchors, which are derived from python
dotted paths — so renaming or moving a function breaks them silently. CI catches
that: `test-docs-build` builds the site and then resolves every link against it.
To run the same check locally:

```bash
uv run mkdocs build -d ./site
./scripts/check_architecture_doc_links.sh ./site
```

It reports the offending `.c4` file and line, and suggests the closest surviving
anchor when a symbol was renamed rather than removed. If a module has no anchor
at all, it is missing from `docs/references/` and needs a `:::` directive there.

## Views

Each view is written for one audience. Pick yours.

| View | Written for | What it shows |
| --- | --- | --- |
| `overview` | Anyone new to ToOp | **Start here.** One end-to-end run as a sequence: grid file in, import, DC search and AC validation running concurrently, topologies out. |
| `parameters` | Anyone tuning a run | The three parameter objects, and which stage owns which constraint. |
| `dataFlow` | Architects | Technologies, the two storage layers and every data flow labelled with its payload. |
| `importerInternals` | Importer engineers | `convert_file`, then the DC preprocessing chain, then the reference loadflow. |
| `dcWorkerInternals` | Optimizer engineers | Repertoire, mutation and crossover, scoring, and the stages inside the GPU solver. |
| `acValidatorInternals` | AC engineers | The selection funnel: three filters, worst-k rejection, then full N-1. |
| `contingencyAnalysis` | Anyone picking a backend | What pandapower and PyPowSyBl can each do, and which `LoadflowResults` tables each one fills. |
| `assetTopology` | Anyone touching topologies | The backend-neutral node-breaker model: who builds it, who reads it, what gets exported from it. |
| `loadflowFormat` | Anyone reading stored results | The on-disk Parquet layout behind a `StoredLoadflowReference`, with the index and columns of every file. |
| `index` | — | Everything at one level. |

In `contingencyAnalysis`, colour carries the meaning: **green** works on both
backends, **amber** is pandapower only, **violet** is PyPowSyBl only.

![One end-to-end ToOp run as a sequence diagram](./generated/overview-sequence.png)

Note the two `par` blocks: the DC-Optimizer and the AC-Validator start from the
*same* `StartOptimizationCommand` and run concurrently, with AC consuming DC
topologies off the shared `results` topic as they appear. They are not
sequential stages, and the AC stage is expected to outlive the DC stage so the
last candidates still get validated.

## Where the model comes from

Naming follows the two papers, so the diagrams and the literature agree:

- [Transmission Topology Optimization using accelerated MapElites](https://arxiv.org/abs/2605.10128) —
  names the three components the **Importer**, **DC-Optimizer** and
  **AC-Validator**, and describes the repertoire descriptors, the mutation and
  crossover operators, the fitness heuristics and the AC elimination strategy.
- [Accelerated DC Loadflow Solver for Topology Optimization](https://arxiv.org/abs/2501.17529) —
  describes the solver: PTDF low-rank updates via BSDF/LODF/MODF, and the split
  between work that changes the PTDF and work that does not.

Everything else was read off the code. Where the two disagree, the code wins and
the model follows the code — the solver paper still describes a "branch module"
and an "injection module", but the code now front-loads the PTDF-changing work
into `compute_bsdf_lodf_static_flows` with the N-0 and N-1 passes after it.

## Viewing and editing

No Node.js toolchain is needed — everything runs through the published Docker
image, and the devcontainer already has the docker-in-docker feature.

### Run the diagram server

This is how to work on the model. It serves every view as a browsable app with
search and click-through navigation, and the sources are mounted live, so
editing a `.c4` file reloads the browser.

```bash
./scripts/serve_architecture_diagrams.sh
```

Then open `http://localhost:5173`. Ctrl-C stops it. The devcontainer forwards
5173 and the live-reload port 24678, so it works the same inside the container
as on the host.

If port 5173 is taken, pass another one — the browser URL changes to match:

```bash
./scripts/serve_architecture_diagrams.sh docs/architecture 5199
```

### Render the PNGs

Only needed to preview what the docs site will embed; CI renders them itself
before every docs build.

```bash
./scripts/render_architecture_diagrams.sh
```

One gotcha this handles for you: the image runs as root so it can reach its
bundled Chromium, which lives under `/root/.cache` with mode 0700. Passing
`--user` makes the export fail to find the browser, so the script chowns the
output back to you afterwards instead.

### Check the model parses

No browser needed, so this one is quick:

```bash
docker run --rm -v "$PWD:/app" -w /app likec4/likec4:1.59.2 validate docs/architecture
```

Both scripts pin the LikeC4 image by digest. Bump the version in both when
upgrading.

## Keeping it honest

The model describes real code paths. When changing any of these, check whether the
model still holds:

- the Kafka topics in `dev-deployment/docker-compose.yaml`
- the artifact names in `packages/interfaces_pkg/src/toop_engine_interfaces/folder_structure.py`
- the three worker entry points: `importer_pkg/.../worker/worker.py`,
  `topology_optimizer_pkg/.../dc/worker/worker.py`,
  `topology_optimizer_pkg/.../ac/worker.py`
