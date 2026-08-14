# Architecture diagrams

Architecture-as-code for the ToOp engine, written in [LikeC4](https://likec4.dev).
The `.c4` files here are the source of truth. The PNGs under `generated/` are
rendered from them by CI and are not committed — edit the model, never the images.

## Start the viewer

```bash
./scripts/serve_architecture_diagrams.sh
```

Then open `http://localhost:5173`. Every view is browsable with search and
click-through navigation, and editing a `.c4` file reloads the browser. Ctrl-C
stops it.

Docker is the only requirement — no Node toolchain. The devcontainer forwards
5173 and the live-reload port 24678. If 5173 is taken, pass another one and the
URL follows: `./scripts/serve_architecture_diagrams.sh docs/architecture 5199`.

## What is where

Each view answers one question. Start with `overview`.

| View | Answers |
| --- | --- |
| `overview` | What does a ToOp run actually do, start to finish? |
| `parameters` | Where is this limit / weight / threshold configured? |
| `dataFlow` | What talks to what, over which technology, carrying what? |
| `importerInternals` | What happens to a grid file, in what order? |
| `dcWorkerInternals` | How do repertoire, mutation, scoring and the GPU solver fit together? |
| `acValidatorInternals` | How does a DC candidate get selected, checked and accepted? |
| `contingencyAnalysis` | What can pandapower and PyPowSyBl each do, and what does each fill in? |
| `assetTopology` | How does a topology travel from the grid file to an export? |
| `loadflowFormat` | What is inside a stored loadflow folder? |
| `index` | Everything at one level. |

In `contingencyAnalysis`, colour carries meaning: **green** works on both
backends, **amber** is pandapower only, **violet** is PyPowSyBl only.

Boxes that map onto a module or function link to the published API reference, so
you can click from a diagram straight to the docstring.

The model itself is split by concern:

```
specification.c4        element kinds, tags, colours
model/
  01-externals.c4       who drives the engine, what goes in and out
  02-messaging.c4       Kafka and its topics
  03-storage.c4         the three filesystems and their contents
  04-engine.c4          the ToOp system; every file below extends it
  05..07                one per stage: importer, DC optimizer, AC validator
  08..10                shared packages: contingency, interfaces, postprocess
  11-relations.c4       every relation, grouped by the stage that drives it
  12-parameters.c4      the parameter object each stage takes
views/                  one file per view, numbered as listed above
```

## Reading more about C4

- [c4model.com](https://c4model.com) — the C4 model: context, container,
  component, code. Worth ten minutes before editing the model.
- [likec4.dev](https://likec4.dev) — LikeC4, the DSL and viewer used here.
- [likec4.dev/docs](https://likec4.dev/docs/) — DSL reference, for when you need
  the exact syntax of a view predicate or a style rule.
