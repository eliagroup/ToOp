# Installation

ToOp is not published on PyPI yet. Clone the repository and install it with [`uv`](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/eliagroup/ToOp.git
cd ToOp
uv sync --all-groups
```

This installs all six packages of the monorepo together with their dependencies.

## Running on a GPU

ToOp runs on a CPU out of the box. To use an NVIDIA GPU, add JAX with CUDA support:

```bash
uv pip install "jax[cuda12]"
```

The DC solver and the genetic search then run on the GPU automatically.

## Using ToOp in your own project

If your own project uses `uv`, you can add ToOp as a local dependency in your `pyproject.toml`, pointing to the cloned repository.

## Setting up for development

If you would like to contribute, we recommend the VS Code devcontainer, so that everyone develops in the same environment. You need:

1. `uv`
2. Python 3.14
3. Visual Studio Code
4. Docker

The [contribution guide](contribution_guide.md#local-development-setup) walks you through the setup.

## Reading the documentation locally

```bash
uv run mkdocs serve
```

Then open the address `mkdocs` prints, usually `http://127.0.0.1:8000`.
