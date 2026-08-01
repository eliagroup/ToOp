# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from uuid import uuid4

from sqlmodel import Session, SQLModel, create_engine, select
from toop_engine_topology_optimizer.database.command_models import GRID_FILES_JSON, OptimizationJob, StageWorkItem, StageWorkItemStatus
from toop_engine_topology_optimizer.interfaces.messages.ac_params import ACOptimizerParameters
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.messages.dc_params import DCOptimizerParameters


def test_json_adapter() -> None:
    grid_files = [GridFile(framework=Framework.PANDAPOWER, grid_folder="dummy-grid")]

    encoded_grid_files = GRID_FILES_JSON.encode(grid_files)
    decoded_grid_files = GRID_FILES_JSON.decode(encoded_grid_files)

    assert encoded_grid_files == [
        {"framework": "pandapower", "grid_folder": "dummy-grid", "timestep_correspondence": None, "coupling": "none"}
    ]
    assert decoded_grid_files == grid_files