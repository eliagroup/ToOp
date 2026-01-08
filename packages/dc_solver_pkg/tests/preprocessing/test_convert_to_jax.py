# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import os
import shutil
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import numpy as np
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.example_grids import case30_with_psts, case30_with_psts_powsybl
from toop_engine_dc_solver.jax.injections import default_injection
from toop_engine_dc_solver.jax.inputs import (
    load_static_information,
    save_static_information,
    validate_static_information,
)
from toop_engine_dc_solver.jax.topology_computations import default_topology
from toop_engine_dc_solver.jax.topology_looper import run_solver_symmetric
from toop_engine_dc_solver.jax.types import BBOutageBaselineAnalysis
from toop_engine_dc_solver.preprocess.convert_to_jax import (
    convert_rel_bb_outage_data,
    convert_relevant_injections,
    convert_to_jax,
    get_bb_outage_baseline_analysis,
    load_grid,
    run_initial_loadflow,
)
from toop_engine_dc_solver.preprocess.preprocess import NetworkData
from toop_engine_dc_solver.preprocess.preprocess_bb_outage import (
    preprocess_bb_outages,
)
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.stored_action_set import load_action_set


def test_convert_relevant_injections(network_data_preprocessed: NetworkData) -> None:
    relevant_injections = convert_relevant_injections(
        injection_idx_at_nodes=network_data_preprocessed.injection_idx_at_nodes,
        mw_injections=network_data_preprocessed.mw_injections,
    )
    assert relevant_injections.ndim == 3
    assert relevant_injections.shape[0] == network_data_preprocessed.mw_injections.shape[0]
    assert relevant_injections.shape[1] == len(network_data_preprocessed.injection_idx_at_nodes)
    assert relevant_injections.shape[2] == max(len(x) for x in network_data_preprocessed.injection_idx_at_nodes)

    for sub, inj_idx_at_sub in enumerate(network_data_preprocessed.injection_idx_at_nodes):
        relevant_injections_sub = relevant_injections[:, sub, :]
        power_ref = network_data_preprocessed.mw_injections[:, inj_idx_at_sub].sum()
        assert np.allclose(relevant_injections_sub.sum(), power_ref)


def test_run_initial_loadflow(network_data_preprocessed: NetworkData) -> None:
    static_information = convert_to_jax(network_data_preprocessed)
    validate_static_information(static_information)
    static_info, (n_0_overload, n_1_overload) = run_initial_loadflow(static_information)
    topo = default_topology(static_info.solver_config)
    inj = default_injection(
        n_splits=topo.action.shape[1],
        max_inj_per_sub=static_info.dynamic_information.max_inj_per_sub,
        batch_size=topo.action.shape[0],
    )
    (n_0_ref, n_1_ref), success = run_solver_symmetric(
        topo,
        None,
        inj.injection_topology,
        static_info.dynamic_information,
        static_info.solver_config,
        lambda lf_res: (lf_res.n_0_matrix, lf_res.n_1_matrix),
    )

    n_0_overload_ref = jnp.sum(
        jnp.clip(np.abs(n_0_ref[0, 0]) - static_info.dynamic_information.branch_limits.max_mw_flow, min=0, max=None)
    )
    n_1_overload_ref = jnp.sum(
        jnp.clip(
            jnp.max(np.abs(n_1_ref[0, 0]), axis=0) - static_info.dynamic_information.branch_limits.max_mw_flow,
            min=0,
            max=None,
        )
    )
    assert abs(n_0_overload - n_0_overload_ref) <= 1e-5
    assert abs(n_1_overload - n_1_overload_ref) <= 1e-5


def test_convert_to_jax(network_data_preprocessed: NetworkData) -> None:
    static_information = convert_to_jax(network_data_preprocessed)
    validate_static_information(static_information)
    assert static_information.dynamic_information.n_nminus1_cases == len(static_information.solver_config.contingency_ids)


def test_convert_to_jax_n_2(network_data_preprocessed: NetworkData) -> None:
    static_information = convert_to_jax(
        network_data_preprocessed,
        enable_n_2=True,
    )
    assert static_information.dynamic_information.n2_baseline_analysis is not None
    validate_static_information(static_information)
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        save_static_information(temp_dir / "static_information.hdf5", static_information)
        static_information2 = load_static_information(temp_dir / "static_information.hdf5")
        assert np.array_equal(
            static_information.dynamic_information.n2_baseline_analysis.n_2_overloads,
            static_information2.dynamic_information.n2_baseline_analysis.n_2_overloads,
        )


def test_convert_to_jax_bb_outage(network_data_preprocessed: NetworkData) -> None:
    # 71%%bus is a relevant node while 67%%bus is non-relevant node
    static_information = convert_to_jax(network_data_preprocessed, enable_bb_outage=True)
    validate_static_information(static_information)

    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        save_static_information(temp_dir / "static_information.hdf5", static_information)
        static_information2 = load_static_information(temp_dir / "static_information.hdf5")
        assert np.array_equal(
            static_information.dynamic_information.action_set.rel_bb_outage_data.deltap_set,
            static_information2.dynamic_information.action_set.rel_bb_outage_data.deltap_set,
        )
        assert np.array_equal(
            static_information.dynamic_information.action_set.rel_bb_outage_data.nodal_indices,
            static_information2.dynamic_information.action_set.rel_bb_outage_data.nodal_indices,
        )
        assert np.array_equal(
            static_information.dynamic_information.action_set.rel_bb_outage_data.articulation_node_mask,
            static_information2.dynamic_information.action_set.rel_bb_outage_data.articulation_node_mask,
        )
        assert np.array_equal(
            static_information.dynamic_information.action_set.rel_bb_outage_data.branch_outage_set,
            static_information2.dynamic_information.action_set.rel_bb_outage_data.branch_outage_set,
        )
        assert np.array_equal(
            static_information.dynamic_information.non_rel_bb_outage_data.branch_outages,
            static_information2.dynamic_information.non_rel_bb_outage_data.branch_outages,
        )
        assert np.array_equal(
            static_information.dynamic_information.non_rel_bb_outage_data.nodal_indices,
            static_information2.dynamic_information.non_rel_bb_outage_data.nodal_indices,
        )
        assert np.array_equal(
            static_information.dynamic_information.non_rel_bb_outage_data.deltap,
            static_information2.dynamic_information.non_rel_bb_outage_data.deltap,
        )


def test_load_grid(data_folder: Path) -> None:
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        shutil.copytree(data_folder, temp_dir, dirs_exist_ok=True)
        filesystem_dir = DirFileSystem(str(temp_dir))
        stats, static_information, network_data = load_grid(
            data_folder_dirfs=filesystem_dir,
            chronics_id=0,
            timesteps=slice(0, 1),
            pandapower=True,
        )
        static_information_path = temp_dir / PREPROCESSING_PATHS["static_information_file_path"]
        network_data_path = temp_dir / PREPROCESSING_PATHS["network_data_file_path"]
        assert os.path.exists(static_information_path)
        assert os.path.exists(network_data_path)
        validate_static_information(static_information)
        assert network_data is not None
        assert stats is not None
        assert static_information.dynamic_information.branch_limits.max_mw_flow_limited is not None


def test_load_grid_case30(tmp_path_factory: pytest.TempPathFactory) -> None:
    folder = tmp_path_factory.mktemp("case30")
    case30_with_psts(folder)
    filesystem_dir = DirFileSystem(str(folder))
    _, static_information, _ = load_grid(data_folder_dirfs=filesystem_dir, pandapower=True)
    validate_static_information(static_information)
    assert static_information.dynamic_information.nodal_injection_information.shift_degree_max.shape == (3,)

    action_set = load_action_set(folder / PREPROCESSING_PATHS["action_set_file_path"])
    assert len(action_set.pst_ranges) == 3
    assert np.array_equal(
        static_information.dynamic_information.nodal_injection_information.shift_degree_max,
        np.array([max(taps.shift_steps) for taps in action_set.pst_ranges]),
    )
    assert np.array_equal(
        static_information.dynamic_information.nodal_injection_information.shift_degree_min,
        np.array([min(taps.shift_steps) for taps in action_set.pst_ranges]),
    )


def test_load_grid_case30_powsybl(tmp_path_factory: pytest.TempPathFactory) -> None:
    folder = tmp_path_factory.mktemp("case30")
    case30_with_psts_powsybl(folder)
    filesystem_dir = DirFileSystem(str(folder))
    _, static_information, _ = load_grid(data_folder_dirfs=filesystem_dir, pandapower=False)
    validate_static_information(static_information)
    assert static_information.dynamic_information.nodal_injection_information.shift_degree_max.shape == (2,)

    assert static_information.dynamic_information.nodal_injection_information.pst_n_taps.shape == (2,)
    pst_n_taps = static_information.dynamic_information.nodal_injection_information.pst_n_taps
    assert pst_n_taps[0] != pst_n_taps[1], "Case30 should have different number of taps for the two PSTs."
    assert static_information.dynamic_information.nodal_injection_information.pst_tapped_angle_values.shape == (
        2,
        jnp.max(pst_n_taps),
    )

    action_set = load_action_set(folder / PREPROCESSING_PATHS["action_set_file_path"])
    assert len(action_set.pst_ranges) == 2
    assert np.array_equal(
        static_information.dynamic_information.nodal_injection_information.shift_degree_max,
        np.array([max(taps.shift_steps) for taps in action_set.pst_ranges]),
    )
    assert np.array_equal(
        static_information.dynamic_information.nodal_injection_information.shift_degree_min,
        np.array([min(taps.shift_steps) for taps in action_set.pst_ranges]),
    )


# TODO: Complete this test method
def test_convert_rel_bb_outage_data(network_data_preprocessed: NetworkData, oberrhein_outage_station_busbars_map) -> None:
    outage_station_busbars_map = oberrhein_outage_station_busbars_map
    network_data_preprocessed = replace(
        network_data_preprocessed,
        busbar_outage_map=outage_station_busbars_map,
    )
    network_data_preprocessed = preprocess_bb_outages(network_data_preprocessed)
    rel_bb_outage_data = convert_rel_bb_outage_data(network_data_preprocessed)
    assert rel_bb_outage_data.deltap_set.shape[:2] == rel_bb_outage_data.nodal_indices.shape
    # There are no critical busbars in the Oberrhein network
    assert jnp.all(rel_bb_outage_data.articulation_node_mask == False)

    # Case 2: When busbar_outage_map is empty.
    # In this case, busbar_outage_map is comprises of all relevant busbars in the
    # default setting.
    network_data_preprocessed = replace(
        network_data_preprocessed,
        busbar_outage_map=None,
    )
    network_data_preprocessed = preprocess_bb_outages(network_data_preprocessed)
    rel_bb_outage_data = convert_rel_bb_outage_data(network_data_preprocessed)
    assert rel_bb_outage_data.deltap_set.shape[:2] == rel_bb_outage_data.nodal_indices.shape

    assert len(network_data_preprocessed.non_rel_bb_outage_br_indices) == 0
    assert len(network_data_preprocessed.non_rel_bb_outage_deltap) == 0
    assert len(network_data_preprocessed.non_rel_bb_outage_nodal_indices) == 0


def test_get_bb_outage_baseline_analysis(jax_inputs_oberrhein):
    static_information = jax_inputs_oberrhein[1]
    result = get_bb_outage_baseline_analysis(
        static_information.dynamic_information,
        1000.0,
    )
    assert isinstance(result, BBOutageBaselineAnalysis)
