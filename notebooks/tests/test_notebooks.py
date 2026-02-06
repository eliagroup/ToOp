# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

# tests/test_notebooks.py
from pathlib import Path

import pytest
from nbclient import NotebookClient
from nbformat import read

NOTEBOOK_DIR = Path("notebooks")
TIMEOUT = 600

example_notebook_paths = sorted(NOTEBOOK_DIR.rglob("example*.ipynb"))


# Executes each example notebook and ensure it executes without errors
@pytest.mark.parametrize("nb_path", example_notebook_paths, ids=[p.name for p in example_notebook_paths])
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
