# tests/test_notebooks.py
from pathlib import Path

import pytest
from nbclient import NotebookClient
from nbformat import read

NOTEBOOK_DIR = Path("notebooks")
TIMEOUT = 600

notebook_paths = sorted(NOTEBOOK_DIR.rglob("example*.ipynb"))


@pytest.mark.parametrize("nb_path", notebook_paths, ids=[p.name for p in notebook_paths])
def test_notebook_executes(nb_path: Path):
    with nb_path.open("r", encoding="utf-8") as f:
        nb = read(f, as_version=4)
    client = NotebookClient(
        nb,
        timeout=TIMEOUT,
        kernel_name="python3",
        resources={"metadata": {"path": nb_path.parent}},
    )
    client.execute()  # will raise on failure
