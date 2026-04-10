# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from __future__ import annotations

import base64
import json
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from pypowsybl.network import Network
from sqlmodel import Session
from toop_engine_dc_solver.export.asset_topology_to_dgs import SwitchUpdateSchema, get_changing_switches_from_topology
from toop_engine_grid_helpers.powsybl.powsybl_helpers import load_powsybl_from_fs
from toop_engine_interfaces.folder_structure import POSTPROCESSING_PATHS
from toop_engine_interfaces.stored_action_set import ActionSet, load_action_set_fs, random_actions
from toop_engine_topology_optimizer.ac.storage import ACOptimTopology, create_session
from toop_engine_topology_optimizer.ac.summary import (
    changing_switches_to_orao_dict,
    db_topology_to_changing_switches,
    export_topology,
    write_summary,
)
from toop_engine_topology_optimizer.interfaces.messages.commons import Framework, GridFile, OptimizerType
from toop_engine_topology_optimizer.interfaces.models.base_storage import hash_topo_data


@dataclass
class SummaryGridContext:
    """Shared handles for running the summary export against a processed grid."""

    filesystem: DirFileSystem
    grid_file: GridFile
    action_set: ActionSet
    network: Network
    grid_root: Path


@dataclass
class StoredTopologyContext:
    """Database state used to verify accepted vs filtered-out summary exports."""

    session: Session
    accepted_topologies: list[ACOptimTopology]
    rejected_topology: ACOptimTopology
    other_optimization_topology: ACOptimTopology


@pytest.fixture
def complex_grid_summary_context(grid_folder: Path) -> SummaryGridContext:
    """Load the complex-grid snapshot once per test from the copied test data folder."""

    filesystem = DirFileSystem(str(grid_folder))
    grid_file = GridFile(framework=Framework.PYPOWSYBL, grid_folder="complex_grid")
    # The action set and grid file both live under the same processed-grid snapshot.
    action_set = load_action_set_fs(filesystem, grid_file.action_set_file, grid_file.action_set_diff_file)
    network = load_powsybl_from_fs(filesystem, grid_file.grid_file)
    return SummaryGridContext(
        filesystem=filesystem,
        grid_file=grid_file,
        action_set=action_set,
        network=network,
        grid_root=grid_folder,
    )


def _sample_topologies(
    action_set: ActionSet,
    network: Network,
    optimization_id: str,
    rng_seed: int,
    n_topologies: int,
) -> list[ACOptimTopology]:
    """Sample DB topologies from the stored action set until they trigger actual switch changes."""

    rng = np.random.default_rng(rng_seed)
    n_substations = len({station.grid_model_id for station in action_set.local_actions})
    sampled_topologies: list[ACOptimTopology] = []
    seen_actions: set[tuple[int, ...]] = set()

    for _ in range(500):
        if len(sampled_topologies) == n_topologies:
            break

        # Sample a small but non-empty set of station actions from the preprocessed action set.
        actions = sorted(
            random_actions(
                action_set=action_set,
                rng=rng,
                n_split_subs=3,
            )
        )
        action_key = tuple(actions)
        if not action_key or action_key in seen_actions:
            continue

        # The DB topology stores only action indices; summary reconstruction expands them again later.
        topology = ACOptimTopology(
            actions=actions,
            disconnections=[],
            pst_setpoints=[],
            unsplit=False,
            timestep=0,
            strategy_hash=hash_topo_data([(actions, [], [])]),
            optimization_id=optimization_id,
            optimizer_type=OptimizerType.AC,
            fitness=float(len(actions)),
            metrics={},
            worst_k_contingency_cases=[],
            acceptance=True,
        )
        # Keep only topologies that produce a non-empty switch diff on the grid.
        switch_updates = db_topology_to_changing_switches(network=network, db_topology=topology, action_set=action_set)
        if switch_updates.empty:
            continue

        sampled_topologies.append(topology)
        seen_actions.add(action_key)

    if len(sampled_topologies) != n_topologies:
        raise RuntimeError("Failed to sample enough topologies with changing switches from the action set.")

    return sampled_topologies


@pytest.fixture
def stored_topologies(complex_grid_summary_context: SummaryGridContext) -> Generator[StoredTopologyContext, None, None]:
    """Populate an in-memory DB with accepted, rejected, and foreign-optimization topologies."""

    session = create_session()

    # Two accepted topologies should end up on disk for the target optimization.
    accepted_topologies = _sample_topologies(
        action_set=complex_grid_summary_context.action_set,
        network=complex_grid_summary_context.network,
        optimization_id="opt-1",
        rng_seed=42,
        n_topologies=2,
    )
    for topology in accepted_topologies:
        topology.acceptance = True

    # This one belongs to the same optimization but should be filtered out by acceptance.
    rejected_topology = _sample_topologies(
        action_set=complex_grid_summary_context.action_set,
        network=complex_grid_summary_context.network,
        optimization_id="opt-1",
        rng_seed=84,
        n_topologies=1,
    )[0]
    rejected_topology.acceptance = False

    # This one is accepted, but belongs to a different optimization run.
    other_optimization_topology = _sample_topologies(
        action_set=complex_grid_summary_context.action_set,
        network=complex_grid_summary_context.network,
        optimization_id="opt-2",
        rng_seed=126,
        n_topologies=1,
    )[0]
    other_optimization_topology.acceptance = True

    for topology in [*accepted_topologies, rejected_topology, other_optimization_topology]:
        session.add(topology)
    session.commit()

    try:
        yield StoredTopologyContext(
            session=session,
            accepted_topologies=accepted_topologies,
            rejected_topology=rejected_topology,
            other_optimization_topology=other_optimization_topology,
        )
    finally:
        session.close()


def _output_path(grid_root: Path, grid_file: GridFile, topology: ACOptimTopology) -> Path:
    """Return the expected JSON export path for a topology hash under the processed grid folder."""

    hash_b64 = base64.b64encode(topology.strategy_hash).decode("utf-8")
    return grid_root / grid_file.grid_folder / POSTPROCESSING_PATHS["orao_summary"] / f"{hash_b64}.json"


def test_changing_switches_to_orao_dict_formats_switch_updates(
    complex_grid_summary_context: SummaryGridContext,
    stored_topologies: StoredTopologyContext,
) -> None:
    switch_updates = db_topology_to_changing_switches(
        network=complex_grid_summary_context.network,
        db_topology=stored_topologies.accepted_topologies[0],
        action_set=complex_grid_summary_context.action_set,
    )

    result = changing_switches_to_orao_dict(switch_updates=switch_updates)

    expected_actions = [
        {
            "type": "TERMINALS_CONNECTION",
            "id": f"{'Open' if switch_update['open'] else 'Close'} {switch_update['grid_model_id']}",
            "elementId": switch_update["grid_model_id"],
            "open": bool(switch_update["open"]),
        }
        for switch_update in switch_updates.to_dict(orient="records")
    ]
    assert result == {
        "forced-actions": {
            "preventive-actions-list": {
                "version": "1.2",
                "actions": expected_actions,
            }
        }
    }


def test_db_topology_to_changing_switches_matches_direct_grid_computation(
    complex_grid_summary_context: SummaryGridContext,
    stored_topologies: StoredTopologyContext,
) -> None:
    db_topology = stored_topologies.accepted_topologies[0]

    result = db_topology_to_changing_switches(
        network=complex_grid_summary_context.network,
        db_topology=db_topology,
        action_set=complex_grid_summary_context.action_set,
    )
    # Rebuild the target topology the same way as the summary code, then compare against the low-level helper directly.
    direct_target_topology = complex_grid_summary_context.action_set.starting_topology.model_copy(
        update={
            "stations": [
                complex_grid_summary_context.action_set.local_actions[action_id] for action_id in db_topology.actions
            ]
        }
    )
    expected = get_changing_switches_from_topology(
        network=complex_grid_summary_context.network,
        target_topology=direct_target_topology,
    )

    SwitchUpdateSchema.validate(result)
    assert not result.empty
    assert result.reset_index(drop=True).equals(expected.reset_index(drop=True))


def test_export_topology_writes_expected_json_for_grid(
    complex_grid_summary_context: SummaryGridContext,
    stored_topologies: StoredTopologyContext,
) -> None:
    db_topology = stored_topologies.accepted_topologies[0]

    # This is the single-topology export path used internally by write_summary.
    export_topology(
        network=complex_grid_summary_context.network,
        db_topology=db_topology,
        action_set=complex_grid_summary_context.action_set,
        processed_gridfile_fs=complex_grid_summary_context.filesystem,
        root_folder=complex_grid_summary_context.grid_file.grid_folder,
    )

    expected_switch_updates = db_topology_to_changing_switches(
        network=complex_grid_summary_context.network,
        db_topology=db_topology,
        action_set=complex_grid_summary_context.action_set,
    )
    output_path = _output_path(
        complex_grid_summary_context.grid_root,
        complex_grid_summary_context.grid_file,
        db_topology,
    )

    assert output_path.exists()
    assert json.loads(output_path.read_text()) == changing_switches_to_orao_dict(expected_switch_updates)


def test_write_summary_exports_only_accepted_topologies(
    complex_grid_summary_context: SummaryGridContext,
    stored_topologies: StoredTopologyContext,
) -> None:
    # This runs the full query + export loop against the grid snapshot and action set.
    write_summary(
        grid_files=[complex_grid_summary_context.grid_file],
        db=stored_topologies.session,
        optimization_id="opt-1",
        processed_gridfile_fs=complex_grid_summary_context.filesystem,
        action_sets=[complex_grid_summary_context.action_set],
    )

    output_dir = (
        complex_grid_summary_context.grid_root
        / complex_grid_summary_context.grid_file.grid_folder
        / POSTPROCESSING_PATHS["orao_summary"]
    )
    written_files = sorted(output_dir.rglob("*.json")) if output_dir.exists() else []

    expected_paths = [
        _output_path(complex_grid_summary_context.grid_root, complex_grid_summary_context.grid_file, topology)
        for topology in stored_topologies.accepted_topologies
    ]
    expected_payloads = {
        path: changing_switches_to_orao_dict(
            db_topology_to_changing_switches(
                network=complex_grid_summary_context.network,
                db_topology=topology,
                action_set=complex_grid_summary_context.action_set,
            )
        )
        for path, topology in zip(expected_paths, stored_topologies.accepted_topologies, strict=True)
    }

    # Only accepted topologies from the requested optimization should have been exported.
    assert sorted(written_files) == sorted(expected_paths)
    for path in expected_paths:
        assert json.loads(path.read_text()) == expected_payloads[path]

    rejected_path = _output_path(
        complex_grid_summary_context.grid_root,
        complex_grid_summary_context.grid_file,
        stored_topologies.rejected_topology,
    )
    other_optimization_path = _output_path(
        complex_grid_summary_context.grid_root,
        complex_grid_summary_context.grid_file,
        stored_topologies.other_optimization_topology,
    )

    assert not rejected_path.exists()
    assert not other_optimization_path.exists()
