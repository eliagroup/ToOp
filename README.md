
# Topology Optimization Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
-------
## About The Project

This repo builds the engine behind the topology optimization project ToOp at EliaGroup. This provides a tool to perform topology optimization on a grid file including import, DC optimization and AC validation. Note that this does NOT provide a GUI or system integration code, you are expected to interact with the module through either python or kafka commands. You can check the [paper](https://arxiv.org/abs/2501.17529) for a high level academic introduction.
Please check out our [full documentation](https://eliagroup.github.io/ToOp.pages.github.io).

## Getting Started

If you want to get started with the engine, we highly recommend

### Prerequisites

If you want to contribute to this repository, we recommend using VS Code's Devcontainer Environment. This allows the developers to use the same environment to develop in.

For this setup, you need to install:
1. `uv`
2. `Microsoft VS Code`
3. `Docker`

### Installation

You can follow our installation guide on our [Contributing page](./CONTRIBUTING.md#local-development-setup).

# Usage

In order to understand the functionalities of this repo, please have a look at our examples in `notebooks/`.
There you can find several Jupyter notebooks that explain how to use the engine.
For example, you can load a grid file and compute the DC loadflow using our GPU-based loadflow solver.
Or you can load an example grid and minimise the branch overload by running the topology optimizer.

You can also build the documentation and open it on your web browser by running
```bash
uv sync --all-groups
uv run mkdocs serve
```


## Roadmap

Outline planned features, improvements, or milestones.
Use checkboxes for clarity:

- [x] MVP
- [ ] Add authentication
- [ ] Multi-language support

---

## Contributing

Please have a look at our [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

Specify the license under which the project is distributed.
Example: Distributed under the MIT License. See [LICENSE](./LICENSE) for more information.

## Citation

If you use our work in scientific research, please cite [our paper on loadflowsolving](https://arxiv.org/abs/2501.17529) and soon also the work on the optimizer architecture, which is to be released soon.

---

## Contact

Project Lead – [Christian Merz](mailto:Christian.Merz@50hertz.com)

Team – [ToOp](mailto:ToOp@eliagroup.eu)

---

## Acknowledgments (if applicable)

Credit any tools, libraries, contributors, or inspirations used in the project.
