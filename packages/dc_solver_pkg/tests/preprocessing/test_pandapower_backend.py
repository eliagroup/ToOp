# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from collections import Counter
from pathlib import Path

import numpy as np
import pandapower as pp
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.preprocess.pandapower.pandapower_backend import PandaPowerBackend
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import (
    table_id,
    table_ids,
)
from toop_engine_interfaces.folder_structure import (
    CHRONICS_FILE_NAMES,
    PREPROCESSING_PATHS,
)


def test_pandapower_backend(data_folder: Path) -> None:
    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir)

    assert backend.net is not None

    for i, bus_id in enumerate(backend.net.bus.index):
        assert table_id(backend.get_node_ids()[i]) == bus_id
        assert backend.get_node_names()[i] == backend.net.bus.loc[bus_id, "name"]
        assert backend.get_node_types()[i] == "bus"

    from_lookup = {
        "line": "from_bus",
        "trafo": "hv_bus",
    }

    for i, node in enumerate(backend.get_from_nodes()):
        pp_type = backend.get_branch_types()[i]
        pp_id = table_id(backend.get_branch_ids()[i])
        if pp_type not in from_lookup:
            continue
        assert backend.net[pp_type].loc[pp_id, from_lookup[pp_type]] == table_id(backend.get_node_ids()[node])

    to_lookup = {
        "line": "to_bus",
        "trafo": "lv_bus",
    }

    for i, node in enumerate(backend.get_to_nodes()):
        pp_type = backend.get_branch_types()[i]
        pp_id = table_id(backend.get_branch_ids()[i])
        if pp_type not in to_lookup:
            continue
        assert backend.net[pp_type].loc[pp_id, to_lookup[pp_type]] == table_id(backend.get_node_ids()[node])

    assert len(backend.get_multi_outage_names()) == len(backend.get_multi_outage_branches())
    assert len(backend.get_multi_outage_nodes()) == len(backend.get_multi_outage_branches())
    assert len(backend.get_multi_outage_names()) >= backend.net.trafo3w.shape[0]
    trafo3w_multi_outages = backend.get_multi_outage_branches()[: len(backend.net.trafo3w)]
    assert np.all(np.sum(trafo3w_multi_outages, axis=1) == 3)
    trafo3w_multi_outages = backend.get_multi_outage_nodes()[: len(backend.net.trafo3w)]
    assert np.all(np.sum(trafo3w_multi_outages, axis=1) == 1)

    assert len(backend.get_disconnectable_branch_mask()) == len(backend.get_branch_types())
    assert backend.get_branches_in_maintenance().shape == (
        len(backend.get_max_mw_flows()),
        len(backend.get_branch_types()),
    )
    assert len(backend.get_monitored_branch_mask()) == len(backend.get_branch_types())

    assert backend.get_asset_topology() is not None


def test_mw_injections(data_folder: Path) -> None:
    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir)
    net = backend.net
    pp.rundcpp(net)

    n_injections = len(backend.get_injection_ids())
    assert len(backend.get_injection_types()) == n_injections
    assert backend.get_mw_injections().shape[1] == n_injections
    assert len(backend.get_outaged_injection_mask()) == n_injections

    for inj_type, inj_id, inj_mw, inj_out in zip(
        backend.get_injection_types(),
        table_ids(backend.get_injection_ids()),
        backend.get_mw_injections()[0],
        backend.get_outaged_injection_mask(),
    ):
        if inj_type not in ["gen", "sgen"]:
            assert not inj_out

        table_mapper = {f"{key}{val}": val for key, val in backend.INJECTION_TYPE_MAPPING}
        if len(table_mapper[inj_type]):
            continue  # TODO implement

        ref_p = net[f"res_{inj_type}"].p_mw.values[inj_id]
        ref_p *= pp.signing_system_value(inj_type)

        assert np.isclose(inj_mw, ref_p)


def test_multi_timestep(data_folder: Path) -> None:
    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir)
    mw_injections_net = backend.get_mw_injections()
    injection_types = np.array(backend.get_injection_types())
    assert np.array_equal(
        mw_injections_net[0, injection_types == "gen"],
        -backend.net.gen.p_mw.values[backend.net.gen.in_service.values],
    )
    assert np.array_equal(
        mw_injections_net[0, injection_types == "load"],
        backend.net.load.p_mw.values[backend.net.load.in_service.values],
    )
    assert np.array_equal(
        mw_injections_net[0, injection_types == "sgen"],
        -backend.net.sgen.p_mw.values[backend.net.sgen.in_service.values],
    )
    assert np.array_equal(
        mw_injections_net[0, injection_types == "dcline_from"],
        backend.net.dcline.p_mw.values[backend.net.dcline.in_service.values],
    )

    chronics_path = Path(data_folder) / PREPROCESSING_PATHS["chronics_path"]
    load_p = np.load(chronics_path / "0000" / CHRONICS_FILE_NAMES["load_p"])
    gen_p = np.load(chronics_path / "0000" / CHRONICS_FILE_NAMES["gen_p"])
    sgen_p = np.load(chronics_path / "0000" / CHRONICS_FILE_NAMES["sgen_p"])
    dcline_p = np.load(chronics_path / "0000" / CHRONICS_FILE_NAMES["dcline_p"])
    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir, chronics_id=0, chronics_slice=None)
    mw_injections = backend.get_mw_injections()
    assert mw_injections.shape == (load_p.shape[0], len(injection_types))
    assert np.array_equal(
        mw_injections[:, injection_types == "gen"],
        -gen_p[:, backend.net.gen.in_service.values],
    )
    assert np.array_equal(
        mw_injections[:, injection_types == "load"],
        load_p[:, backend.net.load.in_service.values],
    )
    assert np.array_equal(
        mw_injections[:, injection_types == "sgen"],
        -sgen_p[:, backend.net.sgen.in_service.values],
    )
    assert np.array_equal(
        mw_injections[:, injection_types == "dcline_from"],
        dcline_p[:, backend.net.dcline.in_service.values],
    )
    loss_percent = backend.net.dcline.loss_percent.values[backend.net.dcline.in_service.values]
    loss_mw = backend.net.dcline.loss_mw.values[backend.net.dcline.in_service.values]
    expected_dc_line_to = -1 * (
        backend.dcline_p[:, backend.net.dcline.in_service] * (1 - loss_percent[:, None] / 100) - loss_mw[:, None]
    )
    assert np.array_equal(mw_injections[:, injection_types == "dcline_to"], expected_dc_line_to)

    # First timestep should match the values in the net
    assert np.allclose(mw_injections[0], mw_injections_net[0])

    # Test with a slice
    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir, chronics_id=0, chronics_slice=slice(0, 5))
    mw_injections_slice = backend.get_mw_injections()

    assert mw_injections_slice.shape == (5, len(injection_types))
    assert np.array_equal(mw_injections_slice, mw_injections[:5])


def test_globally_unique_ids(data_folder: Path) -> None:
    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir)

    assert len(backend.get_branch_ids()) == len(set(backend.get_branch_ids())), (
        f"Duplicates: {[item for item, count in Counter(backend.get_branch_ids()).items() if count > 1]}"
    )
    assert len(backend.get_node_ids()) == len(set(backend.get_node_ids())), (
        f"Duplicates: {[item for item, count in Counter(backend.get_node_ids()).items() if count > 1]}"
    )
    assert len(backend.get_injection_ids()) == len(set(backend.get_injection_ids())), (
        f"Duplicates: {[item for item, count in Counter(backend.get_injection_ids()).items() if count > 1]}"
    )
    assert len(backend.get_multi_outage_ids()) == len(set(backend.get_multi_outage_ids()))


def test_psts(case30_data_folder: Path) -> None:
    filesystem_dir = DirFileSystem(str(case30_data_folder))
    backend = PandaPowerBackend(filesystem_dir)

    assert backend.get_phase_shift_mask().sum() == 4
    assert backend.get_controllable_phase_shift_mask().sum() == 3
    assert not np.any(backend.get_controllable_phase_shift_mask() & ~backend.get_phase_shift_mask())
    assert len(backend.get_phase_shift_taps_and_angles()[0]) == backend.get_controllable_phase_shift_mask().sum()
