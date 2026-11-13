# Topology optimisation for power grids, on the GPU

[**What is ToOp?**](#what-is-toop)
| [**Installation**](installation.md)
| [**Quickstart**](quickstart.md)
| [**Usage**](usage.md)
| [**How it works**](how_it_works.md)
| [**Pipeline explorer**](architecture/pipeline_explorer.html)

## What is ToOp?

Electricity does not take the route we pick for it. It spreads over every line it can reach, simply following the laws of physics. On busy days some of these lines end up carrying more power than they safely can – grid operators call this *congestion*. The usual fix is *redispatch*: asking one power plant to produce less and another, somewhere else, to produce more. It works, but it is expensive, and it often means starting up plants that emit more CO₂.

ToOp, short for *topology optimisation*, looks for a cheaper and cleaner way out. Rather than changing who produces the electricity, it changes the layout of the grid itself – for example by switching off a line, or by splitting a substation so that it behaves like two separate junctions. Much like closing a street or opening a new junction in a city, these *topological actions* gently steer the power away from the lines that are full, and they cost next to nothing to carry out. ToOp searches through the many possible combinations and suggests the most promising ones to the operators in the control room, who always have the final say.

> Our goal: help grid operators relieve congestion with clever switching instead of costly redispatch – saving money and CO₂ along the way.

This repository holds the engine behind ToOp at Elia Group. It reads operational grid data through an importer, searches for good topologies in a fast DC optimisation stage, and double-checks the best candidates with a full AC validation. It also contains the GPU-based DC load flow solver that makes the search fast. Today ToOp considers transmission line switching, busbar splitting, busbar reassignment and linear tap optimisation of phase-shifting transformers (PSTs); for grids imported with PowSyBl, it can also optimise groups of PSTs together.

<img src="./illustrations/TopoActions.jpg" alt="Topological actions: line switching, busbar splitting and busbar reassignment" width="70%">

Curious how it all fits together? The **[interactive pipeline explorer](architecture/pipeline_explorer.html)** walks you through a whole run – from the grid file to the proposals shown to the operator – with a small animated sketch for every component.

## Where to go next

* [Installation](installation.md) – get ToOp running on your machine, with or without a GPU
* [Quickstart](quickstart.md) – a first DC load flow in a few lines, and the example notebooks
* [Usage](usage.md) – the packages, and the two ways to drive ToOp: from Python or through Kafka
* [How it works](how_it_works.md) – the three stages of a run, the architecture and the GPU solver
* Code reference – the API of every package, in the navigation on the left
* Talks at LF Energy: <!-- markdown-link-check-disable -->[2025](https://www.youtube.com/watch?v=XteDpNsX75A) and [2024](https://lfenergy.org/lf-energy-summit-recap-and-video-a-gpu-native-approach-on-tackling-grid-topology-optimization/)<!-- markdown-link-check-enable -->

ToOp ships no GUI or system integration: you drive it from Python or through Kafka commands.

## Working together

We believe topology optimisation works best when it is developed jointly and applied to the grid as a whole, taking into account the operational constraints of different TSOs. So please try ToOp, ask questions, share your feedback and your constraints – and if you like it, contribute. We would also love to hear your ideas for combining ToOp with other open-source tools.

* **Contributing:** have a look at our [contribution guide](contribution_guide.md).
* **Roadmap:** current work broadens the supported controllable assets – a wider range of asset topologies, and non-linear or asymmetric phase-shifting transformers – improves preprocessing fidelity and hardens the end-to-end workflow. A more high-level roadmap will follow.
* **Contact:** do not hesitate to reach out if you are interested in the project or need help with the setup. Your first point of contact is <a href="mailto:ToOp@eliagroup.eu">ToOp@eliagroup.eu</a>.

## Citing ToOp

If you use our work in scientific research, please cite [our paper on load flow solving](https://arxiv.org/abs/2501.17529). The paper on the optimiser architecture will follow soon.

ToOp builds on JAX, and we gladly credit its authors:

```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}
```

## License

Distributed under MPL 2.0. <!-- markdown-link-check-disable -->See [LICENSE](https://github.com/eliagroup/ToOp/blob/main/LICENSE).<!-- markdown-link-check-enable -->
