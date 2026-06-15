# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Helper functions to write a summary after an optimization."""

import base64
import json
from pathlib import Path

import structlog
from fsspec import AbstractFileSystem
from sqlmodel import Session, select
from toop_engine_dc_solver.export.export import get_node_breaker_updates_from_action_set
from toop_engine_interfaces.folder_structure import POSTPROCESSING_PATHS
from toop_engine_interfaces.node_breaker_update import NodeBreakerUpdate
from toop_engine_interfaces.stored_action_set import ActionSet
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile, OptimizerType

logger = structlog.get_logger(__name__)


def node_breaker_updates_to_orao_dict(
    node_breaker_updates: NodeBreakerUpdate,
) -> dict[str, dict[str, dict[str, str | list[dict[str, str | bool]]]]]:
    """Write node-breaker updates into an ORAO-compatible dictionary format.

    The format looks like this:
    ```
    "forced-actions": {
        "preventive-actions-list": {
            "version": "1.2",
            "actions": [
            {
                "type": "PHASE_TAP_CHANGER_TAP_POSITION",
                "id": "PRA_PST_BE",
                "transformerId": "BBE2AA1  BBE3AA1  1",
                "tapPosition": -16,
                "relativeValue": false,
                "side": "TWO"
            },
            {
                "type": "TERMINALS_CONNECTION",
                "id": "Open FR1 FR2",
                "elementId": "FFR1AA1  FFR2AA1  1",
                "open": true
            }
            ]
        }
    }
    ```

    Parameters
    ----------
    node_breaker_updates : NodeBreakerUpdate
        The raw switch and PST updates to export.
    """
    switch_actions = [
        {
            "type": "SWITCH",
            "id": f"{'Open' if switch_update['open'] else 'Close'} {switch_update['grid_model_id']}",
            "switchId": switch_update["grid_model_id"],
            "open": bool(switch_update["open"]),
        }
        for switch_update in node_breaker_updates.switch_updates.to_dict(orient="records")
    ]
    pst_actions = [
        {
            "type": "PHASE_TAP_CHANGER_TAP_POSITION",
            "id": f"Set {pst_update['grid_model_id']} to {pst_update['tap']}",
            "transformerId": pst_update["grid_model_id"],
            "tapPosition": int(pst_update["tap"]),
            "relativeValue": False,
            "side": "TWO",
        }
        for pst_update in node_breaker_updates.pst_updates.to_dict(orient="records")
    ]
    actions = [*switch_actions, *pst_actions]

    return {
        "forced-actions": {
            "preventive-actions-list": {
                "version": "1.2",
                "actions": actions,
            }
        }
    }


def db_topology_to_node_breaker_updates(
    db_topology: ACOptimTopology,
    action_set: ActionSet,
) -> NodeBreakerUpdate:
    """Get raw node-breaker updates from a topology in the database.

    Parameters
    ----------
    db_topology : ACOptimTopology
        The topology to be converted to low-level node-breaker updates.
    action_set : ActionSet
        The action set that was applied to the topology, to read up the stored action definitions.
    """
    return get_node_breaker_updates_from_action_set(
        action_set=action_set,
        actions=db_topology.actions,
        disconnections=db_topology.disconnections,
        pst_setpoints=db_topology.pst_setpoints,
    )


def export_topology(
    db_topology: ACOptimTopology,
    action_set: ActionSet,
    processed_gridfile_fs: AbstractFileSystem,
    root_folder: str,
) -> None:
    """Write a summary of an optimization run for a single timestep

    This will export to
    - Openrao compatible json format
    - .dgs format for visualization in PowerFactory (TODO, not implemented yet)

    Parameters
    ----------
    db_topology : ACOptimTopology
        The topology that was explored during the optimization run for the given grid file.
    action_set : ActionSet
        The action set that was applied during the optimization run for the given grid file, to be
        included in the summary.
    processed_gridfile_fs : AbstractFileSystem
        The file system where the summary should be written to. This is typically the same file system
        where the processed gridfiles are stored.
    root_folder : str
        The root folder where the summary should be written to. This is typically the root folder of the processed grid file
    """
    node_breaker_updates = db_topology_to_node_breaker_updates(
        db_topology=db_topology,
        action_set=action_set,
    )

    # Write orao summary
    dict_repr = node_breaker_updates_to_orao_dict(node_breaker_updates=node_breaker_updates)
    hash_b64 = base64.urlsafe_b64encode(db_topology.strategy_hash).decode("utf-8").rstrip("=")
    summary_path = (
        Path(root_folder) / POSTPROCESSING_PATHS["orao_summary"] / f"{hash_b64}_timestep_{db_topology.timestep}.json"
    )
    processed_gridfile_fs.makedirs(summary_path.parent.as_posix(), exist_ok=True)
    with processed_gridfile_fs.open(summary_path.as_posix(), "w") as f:
        json.dump(dict_repr, f)


def write_summary(
    grid_file: GridFile,
    db: Session,
    optimization_id: str,
    processed_gridfile_fs: AbstractFileSystem,
    action_set: ActionSet,
) -> None:
    """Write a summary of the optimization run to the file system.

    Parameters
    ----------
    grid_file : GridFile
        The grid file that was optimized during the optimization run.
    db : Session
        The in-memory database that holds topologies including the ones from the current optimization. It will be queried
        to perform a topology summary
    optimization_id : str
        The optimization id of the optimization run for which the summary should be written. Topologies will be filtered for
        this optimization id
    processed_gridfile_fs : AbstractFileSystem
        The file system where the summary should be written to. This is typically the same file system where the processed
        gridfiles are stored.
    action_set : ActionSet
        The action set that was applied during the optimization run, to be included in the summary.
    """
    topologies = db.exec(
        select(ACOptimTopology).where(
            ACOptimTopology.optimization_id == optimization_id,
            ACOptimTopology.optimizer_type == OptimizerType.AC,
            ACOptimTopology.acceptance == True,  # noqa: E712
        )
    ).all()

    if grid_file.framework == Framework.PYPOWSYBL:
        for topology in topologies:
            export_topology(
                db_topology=topology,
                action_set=action_set,
                processed_gridfile_fs=processed_gridfile_fs,
                root_folder=grid_file.grid_folder,
            )
    else:
        logger.warning(
            f"Framework {grid_file.framework} is currently not supported for summary export, "
            f"skipping summary export for grid file {grid_file.grid_file}"
        )
