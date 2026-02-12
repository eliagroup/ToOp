# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import gc
import json
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl
import pypowsybl
import pypowsybl.loadflow.impl
import pypowsybl.loadflow.impl.loadflow
import pytest
import ray
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from jax_dataclasses import replace
from toop_engine_contingency_analysis.pypowsybl.powsybl_helpers import set_target_values_to_lf_values_incl_distributed_slack
from toop_engine_dc_solver.jax.compute_batch import compute_symmetric_batch
from toop_engine_dc_solver.jax.injections import default_injection
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_dc_solver.jax.topology_computations import (
    default_topology,
)
from toop_engine_dc_solver.jax.topology_looper import run_solver_symmetric
from toop_engine_dc_solver.jax.types import ActionIndexComputations, NodalInjOptimResults, NodalInjStartOptions
from toop_engine_dc_solver.postprocess.postprocess_powsybl import (
    PowsyblRunner,
    apply_disconnections,
    apply_topology,
    compute_cross_coupler_flows,
)
from toop_engine_dc_solver.preprocess.convert_to_jax import convert_to_jax
from toop_engine_dc_solver.preprocess.network_data import (
    extract_action_set,
    extract_nminus1_definition,
    load_lf_params,
    load_network_data,
)
from toop_engine_dc_solver.preprocess.powsybl.powsybl_backend import PowsyblBackend
from toop_engine_grid_helpers.powsybl.loadflow_parameters import DISTRIBUTED_SLACK
from toop_engine_interfaces.folder_structure import (
    OUTPUT_FILE_NAMES,
    POSTPROCESSING_PATHS,
    PREPROCESSING_PATHS,
)
from toop_engine_interfaces.loadflow_result_helpers_polars import (
    extract_node_matrices_polars,
    extract_solver_matrices_polars,
)
from toop_engine_interfaces.nminus1_definition import GridElement, load_nminus1_definition
from toop_engine_interfaces.stored_action_set import ActionSet, load_action_set


def test_apply_topology(preprocessed_powsybl_data_folder: Path) -> None:
    # Load grid, network data and topology
    network_data = load_network_data(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["network_data_file_path"])
    action_set = extract_action_set(network_data)
    post_process_file_path = (
        preprocessed_powsybl_data_folder
        / POSTPROCESSING_PATHS["dc_optimizer_snapshots_path"]
        / OUTPUT_FILE_NAMES["multiple_topologies"]
    )
    with open(post_process_file_path, "r", encoding="utf-8") as f:
        res_json = json.load(f)

    for i in range(10):
        action = res_json["best_topos"][i]["actions"]

        # Apply the topology
        net = pypowsybl.network.load(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
        n_buses_before = len(net.get_buses())
        _ = apply_topology(net, action, action_set)
        assert len(net.get_buses()) == n_buses_before + len(action)

        assert sum(net.get_switches()["open"]) == len(action)

        # Check that the loadflow still converges
        dc_res = pypowsybl.loadflow.run_dc(net)
        assert dc_res[0].status == pypowsybl.loadflow.ComponentStatus.CONVERGED


def test_apply_disconnections(preprocessed_powsybl_data_folder: Path) -> None:
    net = pypowsybl.network.load(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    assert net.get_branches().connected1.iloc[0]
    assert net.get_branches().connected2.iloc[0]

    action_set = ActionSet.model_construct(
        disconnectable_branches=[GridElement(id=net.get_branches().index[0], type="LINE", name="test", kind="branch")]
    )

    disconnections = [0]

    apply_disconnections(net, disconnections, action_set)

    assert not net.get_branches().connected1.iloc[0]
    assert not net.get_branches().connected2.iloc[0]


@pytest.mark.parametrize("fixture_name", ["preprocessed_powsybl_data_folder", "node_breaker_grid_preprocessed_data_folder"])
def test_apply_topology_matches_loadflows(
    request,
    fixture_name: str,
) -> None:
    # Sometimes ray crashes during this test, make sure it is shut down.
    ray.shutdown()
    data_folder = request.getfixturevalue(fixture_name)
    assert isinstance(data_folder, Path)

    net = pypowsybl.network.load(data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    network_data = load_network_data(data_folder / PREPROCESSING_PATHS["network_data_file_path"])
    lf_params = load_lf_params(data_folder / PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    action_set = extract_action_set(network_data)
    runner = PowsyblRunner(lf_params=lf_params)
    runner.replace_grid(net)
    runner.store_action_set(action_set)
    nminus1_def = extract_nminus1_definition(network_data)
    runner.store_nminus1_definition(nminus1_def)
    static_information = load_static_information(data_folder / PREPROCESSING_PATHS["static_information_file_path"])
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=1),
    )
    post_process_file_path = (
        data_folder / POSTPROCESSING_PATHS["dc_optimizer_snapshots_path"] / OUTPUT_FILE_NAMES["multiple_topologies"]
    )
    with open(post_process_file_path, "r", encoding="utf-8") as f:
        optim_res = json.load(f)
    for i in range(len(optim_res["best_topos"])):
        actions = optim_res["best_topos"][i]["actions"]

        (n_0, n_1), success = run_solver_symmetric(
            ActionIndexComputations(action=jnp.array([actions], dtype=int), pad_mask=jnp.array([True])),
            None,
            None,
            static_information.dynamic_information,
            static_information.solver_config,
            lambda lf_res: (lf_res.n_0_matrix, lf_res.n_1_matrix),
        )
        n_0 = np.abs(n_0[0, 0])
        n_1 = np.abs(n_1[0, 0])
        assert np.all(success)

        res = runner.run_dc_loadflow(actions, [])
        assert runner.get_last_action_info() is not None
        n_0_runner, n_1_runner, success = extract_solver_matrices_polars(res, nminus1_def, 0)
        assert np.all(success)
        n_0_runner = np.abs(n_0_runner)
        n_1_runner = np.abs(n_1_runner)
        assert n_0.shape == n_0_runner.shape
        assert np.allclose(n_0, n_0_runner)
        assert n_1.shape == n_1_runner.shape
        assert np.allclose(n_1, n_1_runner)


def test_apply_disconnections_matches_loadflows(
    preprocessed_powsybl_data_folder: Path,
) -> None:
    net = pypowsybl.network.load(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    network_data = load_network_data(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["network_data_file_path"])
    lf_params = load_lf_params(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["loadflow_parameters_file_path"])
    runner = PowsyblRunner(lf_params=lf_params)
    runner.replace_grid(net)
    runner.store_action_set(extract_action_set(network_data))
    nminus1_definition = extract_nminus1_definition(network_data)
    runner.store_nminus1_definition(nminus1_definition)
    static_information = load_static_information(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=1),
    )
    post_process_file_path = (
        preprocessed_powsybl_data_folder
        / POSTPROCESSING_PATHS["dc_optimizer_snapshots_path"]
        / OUTPUT_FILE_NAMES["multiple_topologies"]
    )
    with open(post_process_file_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    for i in range(len(res["best_topos"])):
        disconnections = np.array(res["best_topos"][i]["disconnection"])

        disconnections_repeated = jnp.repeat(
            disconnections[None, :],
            static_information.solver_config.batch_size_bsdf,
            axis=0,
        )
        topo = default_topology(static_information.solver_config)
        inj = default_injection(
            n_splits=topo.action.shape[1],
            max_inj_per_sub=static_information.dynamic_information.max_inj_per_sub,
            batch_size=topo.action.shape[0],
        )

        (n_0, n_1), success = run_solver_symmetric(
            topo,
            disconnections_repeated,
            inj.injection_topology,
            static_information.dynamic_information,
            static_information.solver_config,
            lambda lf_res: (lf_res.n_0_matrix, lf_res.n_1_matrix),
        )
        n_0 = np.abs(n_0[0, 0])
        n_1 = np.abs(n_1[0, 0])
        assert np.all(success)

        ref_result = runner.run_dc_loadflow([], disconnections.tolist())
        n_0_ref, n_1_ref, success = extract_solver_matrices_polars(ref_result, nminus1_definition, 0)
        # assert np.all(success)
        n_0_ref = np.abs(n_0_ref)
        n_1_ref = np.abs(n_1_ref)

        assert n_0.shape == n_0_ref.shape
        assert jnp.allclose(n_0, n_0_ref)
        assert n_1.shape == n_1_ref.shape
        assert jnp.allclose(n_1, n_1_ref)

        single_result = runner.run_dc_n_0([], disconnections.tolist())
        n_0_single, _, _ = extract_solver_matrices_polars(single_result, nminus1_definition, 0)
        n_0_single = np.abs(n_0_single)

        assert n_0_single.shape == n_0_ref.shape
        assert np.allclose(n_0_single, n_0_ref)


@pytest.mark.parametrize(
    "fixture_name", ["three_node_pst_example_data_folder", "complex_grid_battery_hvdc_svc_3w_trafo_data_folder"]
)
def test_change_pst_matches_loadflows(
    request,
    fixture_name: str,
) -> None:
    if fixture_name == "complex_grid_battery_hvdc_svc_3w_trafo_data_folder":
        pytest.xfail("PSDF implementation has a bug on complex grids")

    preprocessed_powsybl_data_folder = request.getfixturevalue(fixture_name)
    net = pypowsybl.network.load(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    network_data = load_network_data(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["network_data_file_path"])
    runner = PowsyblRunner()
    runner.replace_grid(net)
    runner.store_action_set(extract_action_set(network_data))
    nminus1_definition = extract_nminus1_definition(network_data)

    runner.store_nminus1_definition(nminus1_definition)
    static_information = load_static_information(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )
    di = static_information.dynamic_information
    solver_config = replace(
        static_information.solver_config,
        batch_size_bsdf=1,
    )
    assert di.nodal_injection_information is not None, "Grid should have nodal injection information for this test"

    # With random PST tap changes, DC solver and runner should match
    n_pst = len(di.nodal_injection_information.pst_n_taps)
    rel_taps = jax.random.randint(
        jax.random.PRNGKey(42),
        shape=(n_pst,),
        minval=0,
        maxval=di.nodal_injection_information.pst_n_taps,
    )
    abs_taps = rel_taps + di.nodal_injection_information.grid_model_low_tap

    solver_res, success_dc = compute_symmetric_batch(
        topology_batch=default_topology(solver_config),
        disconnection_batch=None,
        injections=None,
        nodal_inj_start_options=NodalInjStartOptions(
            previous_results=NodalInjOptimResults(pst_tap_idx=rel_taps[None, None, :]),
            precision_percent=jnp.array(0.0),
        ),
        dynamic_information=di,
        solver_config=solver_config,
    )
    assert np.all(success_dc), "DC solver with PST changes should succeed"

    # Get PST branch IDs from the action set (which knows about controllable PSTs)
    action_set = extract_action_set(network_data)
    pst_indices = [pst.id for pst in action_set.pst_ranges]

    net = pypowsybl.network.load(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    net.update_phase_tap_changers(id=pst_indices, tap=(abs_taps).tolist())
    net = set_target_values_to_lf_values_incl_distributed_slack(net, "dc")
    pypowsybl.loadflow.run_dc(net)
    n_0_direct = net.get_branches().loc[network_data.branch_ids][network_data.monitored_branch_mask].p1.values

    runner_res = runner.run_dc_loadflow([], [], np.array(abs_taps).tolist())
    n_0_runner_pst, n_1_runner_pst, success_ref = extract_solver_matrices_polars(runner_res, nminus1_definition, 0)
    assert np.all(success_ref), "Pypowsybl runner with PST changes should succeed"

    n_0_with_pst = -solver_res.n_0_matrix[0, 0]
    n_1_with_pst = -solver_res.n_1_matrix[0, 0]

    # First verify the two powsybl native computations
    assert np.allclose(n_0_direct, n_0_runner_pst, atol=1e-2)

    # Then verify runner also matches direct computation
    assert np.allclose(n_0_runner_pst, n_0_with_pst, atol=1e-2), "Runner should match direct pypowsybl computation"

    # Finally verify runner matches DC solver
    assert np.allclose(np.abs(n_1_runner_pst), np.abs(n_1_with_pst), atol=1e-2), "N-1 with PST changes should match"


def test_runner_load_from_fs(preprocessed_powsybl_data_folder: Path) -> None:
    runner = PowsyblRunner()
    runner.load_base_grid(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])

    runner2 = PowsyblRunner()
    runner2.load_base_grid_fs(
        LocalFileSystem(), preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    )
    assert runner.net.get_buses().equals(runner2.net.get_buses())


def test_powsybl_runner(preprocessed_powsybl_data_folder: Path) -> None:
    runner = PowsyblRunner()
    runner.load_base_grid(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    nminus1_def = load_nminus1_definition(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["nminus1_definition_file_path"]
    )
    action_set = load_action_set(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["action_set_file_path"])
    runner.store_nminus1_definition(nminus1_def)
    runner.store_action_set(action_set)

    res = runner.run_dc_loadflow([], [])
    n_0, n_1, success = extract_solver_matrices_polars(res, nminus1_def, 0)
    assert np.all(success)

    res_single = runner.run_dc_n_0([], [])
    n_0_single, _, _ = extract_solver_matrices_polars(res_single, nminus1_def, 0)
    assert np.allclose(n_0, n_0_single)
    assert n_0.shape == n_0_single.shape

    res_ac = runner.run_ac_loadflow([], [])
    n_0_ac, n_1_ac, success_ac = extract_solver_matrices_polars(res_ac, nminus1_def, 0)
    # TODO find out why last generator outage does not converge on AC
    # assert np.all(success_ac)
    assert np.sum(success_ac) > 0, "At least one AC loadflow should have converged"
    assert success_ac.shape == success.shape
    assert n_0.shape == n_0_ac.shape
    assert n_1.shape == n_1_ac.shape

    # Try the distributed version
    runner = PowsyblRunner(n_processes=2)
    runner.load_base_grid(preprocessed_powsybl_data_folder / "grid.xiidm")
    runner.store_nminus1_definition(nminus1_def)
    runner.store_action_set(action_set)

    # Work against out-of-memory errors on CI
    gc.collect()
    res_mp = runner.run_dc_loadflow([], [])
    n_0_mp, n_1_mp, success_mp = extract_solver_matrices_polars(res_mp, nminus1_def, 0)
    assert np.all(success_mp)

    assert n_0.shape == n_0_mp.shape
    assert np.allclose(n_0, n_0_mp)
    assert n_1.shape == n_1_mp.shape
    assert np.allclose(n_1, n_1_mp)
    ray.shutdown()


def test_compute_cross_coupler_flows(preprocessed_powsybl_data_folder: Path) -> None:
    fs_dir = DirFileSystem(str(preprocessed_powsybl_data_folder))
    backend = PowsyblBackend(fs_dir)
    net = backend.net
    network_data = load_network_data(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["network_data_file_path"])
    action_set = extract_action_set(network_data)
    post_process_file_path = (
        preprocessed_powsybl_data_folder
        / POSTPROCESSING_PATHS["dc_optimizer_snapshots_path"]
        / OUTPUT_FILE_NAMES["multiple_topologies"]
    )
    with open(post_process_file_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    actions = res["best_topos"][0]["actions"]

    cross_coupler_p_ref, _, success = compute_cross_coupler_flows(net, actions, action_set, DISTRIBUTED_SLACK, "dc")
    assert np.all(success)
    assert cross_coupler_p_ref.shape == (len(actions),)

    static_information = load_static_information(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )

    static_information = replace(
        static_information,
        solver_config=replace(
            static_information.solver_config,
            batch_size_bsdf=1,
        ),
    )

    cc_flows, success = run_solver_symmetric(
        topologies=ActionIndexComputations(
            action=jnp.array([actions], dtype=int),
            pad_mask=jnp.array([True]),
        ),
        disconnections=None,
        injections=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
        aggregate_output_fn=lambda lf_res: lf_res.cross_coupler_flows,
    )
    cc_flows = cc_flows[0, :, 0]

    assert cc_flows.shape == cross_coupler_p_ref.shape
    assert np.allclose(np.abs(cc_flows), np.abs(cross_coupler_p_ref))


@pytest.mark.parametrize(
    "data_folder_fixture", ["preprocessed_powsybl_data_folder", "node_breaker_grid_preprocessed_data_folder"]
)
def test_compute_n_1_ac(data_folder_fixture: str, request) -> None:
    data_folder = request.getfixturevalue(data_folder_fixture)

    network_data = load_network_data(data_folder / PREPROCESSING_PATHS["network_data_file_path"])
    grid_file_path = Path(data_folder) / PREPROCESSING_PATHS["grid_file_path_powsybl"]
    runner = PowsyblRunner()
    runner.load_base_grid(grid_file_path)
    runner.store_action_set(extract_action_set(network_data))
    nminus1_definition = extract_nminus1_definition(network_data)
    runner.store_nminus1_definition(nminus1_definition)
    switches = [elem for elem in nminus1_definition.monitored_elements if elem.kind == "switch"]
    # Open a switch for va_diff results
    runner.net.open_switch(switches[0].id)
    net = runner.net
    res = runner.run_ac_loadflow([], [])
    n_0, n_1, success = extract_solver_matrices_polars(
        loadflow_results=res,
        nminus1_definition=nminus1_definition,
        timestep=0,
    )
    vm_n0, va_n0, vm_n1, va_n1 = extract_node_matrices_polars(
        node_results=res.node_results,
        timestep=0,
        basecase="BASECASE",
        contingencies=[cont.id for cont in nminus1_definition.contingencies if cont.is_basecase() is False],
        monitored_nodes=[element for element in nminus1_definition.monitored_elements if element.kind == "bus"],
    )
    assert n_1.shape[0] == sum(network_data.outaged_branch_mask) + len(network_data.multi_outage_types) + sum(
        network_data.outaged_injection_mask
    )

    assert n_1.shape[1] == sum(network_data.monitored_branch_mask)

    outaged_branch_types = np.array(network_data.branch_types)[network_data.outaged_branch_mask]
    outaged_branch_ids = [
        elem.id for contingency in nminus1_definition.contingencies for elem in contingency.elements if elem.kind == "branch"
    ]
    monitored_node_ids = [elem.id for elem in nminus1_definition.monitored_elements if elem.kind == "bus"]
    assert vm_n1.shape == (n_1.shape[0], len(monitored_node_ids))
    assert va_n1.shape == (n_1.shape[0], len(monitored_node_ids))

    va_diff_res = res.va_diff_results

    open_switches = net.get_switches(attributes=["open", "bus_breaker_bus1_id", "bus_breaker_bus2_id"]).query("open")
    monitored_switches = [
        elem.id for elem in nminus1_definition.monitored_elements if elem.kind == "switch" and elem.id in open_switches.index
    ]

    cases = va_diff_res.select("contingency").unique().collect()["contingency"].to_numpy()

    if len(monitored_switches) > 0:
        assert "BASECASE" in cases, "Basecase should be present in the va_diff_results if there are monitored switches"

    elements = va_diff_res.select("element").unique().collect()["element"].to_numpy()
    assert all([switch_id in elements for switch_id in monitored_switches]), (
        "All existing monitored switches should show up in the va_diff_results"
    )
    if monitored_switches:
        assert all([cont.id in cases for cont in nminus1_definition.contingencies if not cont.is_basecase()]), (
            "All contingencies should show up in the va_diff_results"
        )

    assert all([outage_id in elements for outage_id in outaged_branch_ids]), (
        "All outaged branches should show up in the va_diff_results"
    )

    monitored_branches = [elem.id for elem in nminus1_definition.monitored_elements if elem.kind == "branch"]
    original_branches = net.get_branches(attributes=["bus_breaker_bus1_id", "bus_breaker_bus2_id", "bus1_id", "bus2_id"])

    n0_res, *_ = pypowsybl.loadflow.run_ac(net, DISTRIBUTED_SLACK)
    lf_params = deepcopy(DISTRIBUTED_SLACK)
    # Fix the slack to always use the same bus as in the base case
    lf_params.read_slack_bus = False
    lf_params.provider_parameters["slackBusSelectionMode"] = "NAME"
    lf_params.provider_parameters["slackBusesIds"] = n0_res.reference_bus_id
    lf_params.provider_parameters["alwaysUpdateNetwork"] = "true"
    for i, unique_id in enumerate(outaged_branch_ids):
        outage_net = deepcopy(net)
        was_connected = outage_net.disconnect(unique_id)
        result, *_ = pypowsybl.loadflow.run_ac(outage_net, lf_params)
        if result.status != pypowsybl.loadflow.ComponentStatus.CONVERGED:
            assert success[i] == False, f"Loadflow for outage {unique_id} should not have converged"
            continue
        branches = outage_net.get_branches(attributes=["p1", "bus_breaker_bus1_id", "bus_breaker_bus2_id"])
        monitored_loadflows = branches.loc[monitored_branches, "p1"].fillna(0.0).values
        # Compare the first time step only
        assert n_1[i].shape == monitored_loadflows.shape
        if was_connected:
            assert np.allclose(
                np.abs(n_1[i]),
                np.abs(monitored_loadflows),
                atol=1,  # Allow some tolerance for numerical differences # TODO find out why this is needed
            )
        else:
            np.allclose(
                np.abs(n_0),
                np.abs(monitored_loadflows),
            )
        buses = outage_net.get_bus_breaker_view_buses(attributes=["v_mag", "v_angle", "bus_id"])
        buses.loc[buses.bus_id == "", ["v_mag", "v_angle"]] = np.nan, np.nan
        busbars = outage_net.get_busbar_sections(attributes=["v", "angle", "bus_id"]).rename(
            columns={"v": "v_mag", "angle": "v_angle"}
        )
        all_buses = pd.concat([buses, busbars], axis=0)
        monitored_buses = all_buses.reindex(monitored_node_ids, fill_value=np.nan)
        # Compare the voltage results
        monitored_buses.loc[monitored_buses.bus_id == "", ["v_mag", "v_angle"]] = np.nan, np.nan
        assert np.allclose(vm_n1[i], monitored_buses.v_mag.values, equal_nan=True, atol=1e-3)
        assert np.allclose(va_n1[i], monitored_buses.v_angle.values, equal_nan=True, atol=1e-1)

        # Compare the va_diff_results
        for switch_id, row in open_switches.loc[monitored_switches].iterrows():
            va_diff = (
                va_diff_res.filter(
                    (pl.col("timestep") == 0) & (pl.col("contingency") == unique_id) & (pl.col("element") == switch_id)
                )
                .collect()["va_diff"]
                .item()
                or np.nan
            )
            va_diff_expected = buses.loc[row["bus_breaker_bus1_id"]].v_angle - buses.loc[row["bus_breaker_bus2_id"]].v_angle

            assert np.isclose(va_diff, va_diff_expected, atol=1e-3, equal_nan=True), (
                f"Va Diff does not match for switch {switch_id} in outage {unique_id}"
            )
        bus_1 = original_branches.loc[unique_id, "bus1_id"]
        bus_2 = original_branches.loc[unique_id, "bus2_id"]
        electrical_buses = outage_net.get_buses(attributes=["v_mag", "v_angle"])
        va_diff = electrical_buses.loc[bus_1].v_angle - electrical_buses.loc[bus_2].v_angle

        lf_res_va_diff = (
            va_diff_res.filter(
                (pl.col("timestep") == 0) & (pl.col("contingency") == unique_id) & (pl.col("element") == unique_id)
            )
            .collect()["va_diff"]
            .item()
        )
        assert np.isclose(va_diff, lf_res_va_diff, atol=1e-1), "Va Diff does not match for branch outage"

    offset = len(outaged_branch_types)

    # Test multi outages
    multi_outages = [elem for elem in nminus1_definition.contingencies if len(elem.elements) > 1]
    for i, contingency in enumerate(multi_outages):
        unique_id = contingency.id
        outage_net = deepcopy(net)
        for outage in contingency.elements:
            outage_net.disconnect(outage.id)
        result, *_ = pypowsybl.loadflow.run_ac(outage_net, lf_params)
        if result.status != pypowsybl.loadflow.ComponentStatus.CONVERGED.value:
            assert success[i] == False, f"Loadflow for outage {unique_id} should not have converged"
        branches = outage_net.get_branches(attributes=["p1", "bus_breaker_bus1_id", "bus_breaker_bus2_id"])
        monitored_loadflows = branches.loc[monitored_branches, "p1"].fillna(0.0)
        # Compare the first time step only
        assert n_1[i + offset].shape == monitored_loadflows.shape
        assert np.allclose(
            np.abs(n_1[i + offset]),
            np.abs(monitored_loadflows),
            atol=1e-2,  # Allow some tolerance for numerical differences # TODO find out why this is needed
        )
        buses = outage_net.get_bus_breaker_view_buses(attributes=["v_mag", "v_angle"])
        busbars = outage_net.get_busbar_sections(attributes=["v", "angle"]).rename(
            columns={"v": "v_mag", "angle": "v_angle"}
        )
        all_buses = pd.concat([buses, busbars], axis=0)
        monitored_buses = all_buses.reindex(monitored_node_ids, fill_value=np.nan)
        # Compare the voltage results
        assert np.allclose(vm_n1[i + offset], monitored_buses.v_mag.values, equal_nan=True)
        assert np.allclose(va_n1[i + offset], monitored_buses.v_angle.values, equal_nan=True, atol=1e-4)

        # Compare the va_diff_results
        for switch_id, row in open_switches.loc[monitored_switches].iterrows():
            va_diff = (
                va_diff_res.filter(
                    (pl.col("timestep") == 0) & (pl.col("contingency") == unique_id) & (pl.col("element") == unique_id)
                )
                .collect()["va_diff"]
                .item()
            )
            va_diff_expected = buses.loc[row["bus_breaker_bus1_id"]].v_angle - buses.loc[row["bus_breaker_bus2_id"]].v_angle
            assert np.isclose(va_diff, va_diff_expected, atol=1e-4), (
                f"Va Diff does not match for switch {switch_id} in outage {unique_id}"
            )


def test_n0_in_ac_unsplit(preprocessed_powsybl_data_folder: Path) -> None:
    network_data = load_network_data(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["network_data_file_path"])

    static_information_dc = load_static_information(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )
    (n_0_dc, n_1_dc), success_dc = run_solver_symmetric(
        topologies=default_topology(static_information_dc.solver_config),
        disconnections=None,
        injections=None,
        dynamic_information=static_information_dc.dynamic_information,
        solver_config=static_information_dc.solver_config,
        aggregate_output_fn=lambda lf_res: (lf_res.n_0_matrix, lf_res.n_1_matrix),
    )
    assert np.all(success_dc)

    # Run the unsplit topology with N-0 in AC
    static_information = convert_to_jax(
        network_data=network_data,
        ac_dc_interpolation=1.0,
    )

    (n_0, n_1), success = run_solver_symmetric(
        topologies=default_topology(static_information.solver_config),
        disconnections=None,
        injections=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
        aggregate_output_fn=lambda lf_res: (lf_res.n_0_matrix, lf_res.n_1_matrix),
    )
    assert np.all(success)

    assert n_0.shape == n_0_dc.shape
    assert n_1.shape == n_1_dc.shape
    assert not np.allclose(n_0, n_0_dc)
    assert not np.allclose(n_1, n_1_dc)

    # First batch, first timestep
    n_0 = n_0[0, 0]
    n_1 = n_1[0, 0]
    n_0_dc = n_0_dc[0, 0]
    n_1_dc = n_1_dc[0, 0]

    runner = PowsyblRunner()
    runner.load_base_grid(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    nminus1_def = extract_nminus1_definition(network_data)
    runner.store_nminus1_definition(nminus1_def)
    runner.store_action_set(extract_action_set(network_data))
    ref = runner.run_ac_loadflow([], [])
    n_0_ref, n_1_ref, success_ref = extract_solver_matrices_polars(ref, nminus1_def, 0)
    assert np.sum(success_ref) > len(success_ref) / 2, "At least half of the AC loadflows should have converged"

    assert len(success_ref) == len(n_1_ref)
    assert n_0.shape == n_0_ref.shape
    assert np.allclose(np.abs(n_0), np.abs(n_0_ref))
    assert n_1.shape == n_1_ref.shape

    # Look only at successful loadflows
    n_1_ref = n_1_ref[success_ref]
    n_1 = n_1[success_ref]
    n_1_dc = n_1_dc[success_ref]

    # We assume that the N-1 loadflows on AC are closer
    ac_dist = np.abs(n_1) - np.abs(n_1_ref)
    dc_dist = np.abs(n_1_dc) - np.abs(n_1_ref)

    ac_better = np.abs(ac_dist) < np.abs(dc_dist)
    dc_better = np.abs(dc_dist) < np.abs(ac_dist)

    assert np.sum(ac_better) > np.sum(dc_better)


# @pytest.mark.skip(reason="It's not guaranteed that the N-0 in AC must be better after splits than the pure DC approach")
@pytest.mark.parametrize("topo_idx", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_n0_in_ac_split_ignoring_disconnections(preprocessed_powsybl_data_folder: Path, topo_idx: int) -> None:
    # TODO Validate if this is a meaningful way to test this
    # Due to the random nature of the topologies, this may or may not fail. We need a better way to test this.
    # For most random seeds, this should have at least more than 50% success rate. If this suddenly drops to 0% you
    # probably broke something
    network_data = load_network_data(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["network_data_file_path"])

    post_process_file_path = (
        preprocessed_powsybl_data_folder
        / POSTPROCESSING_PATHS["dc_optimizer_snapshots_path"]
        / OUTPUT_FILE_NAMES["multiple_topologies"]
    )
    with open(post_process_file_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    topo_data = res["best_topos"][topo_idx]
    actions = topo_data["actions"]

    static_information_dc = load_static_information(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )
    (n_0_dc, n_1_dc), success_dc = run_solver_symmetric(
        topologies=ActionIndexComputations(action=jnp.array([actions], dtype=int), pad_mask=jnp.array([True])),
        disconnections=None,
        injections=None,
        dynamic_information=static_information_dc.dynamic_information,
        solver_config=static_information_dc.solver_config,
        aggregate_output_fn=lambda lf_res: (lf_res.n_0_matrix, lf_res.n_1_matrix),
    )
    assert np.all(success_dc)

    # Run the unsplit topology with N-0 in AC
    static_information = convert_to_jax(
        network_data=network_data,
        ac_dc_interpolation=1.0,
    )

    (n_0, n_1), success = run_solver_symmetric(
        topologies=ActionIndexComputations(action=jnp.array([actions], dtype=int), pad_mask=jnp.array([True])),
        disconnections=None,
        injections=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
        aggregate_output_fn=lambda lf_res: (lf_res.n_0_matrix, lf_res.n_1_matrix),
    )
    assert np.all(success)

    assert n_0.shape == n_0_dc.shape
    assert n_1.shape == n_1_dc.shape
    assert not np.allclose(n_0, n_0_dc)
    assert not np.allclose(n_1, n_1_dc)

    # First batch, first timestep
    n_0 = n_0[0, 0]
    n_1 = n_1[0, 0]
    n_0_dc = n_0_dc[0, 0]
    n_1_dc = n_1_dc[0, 0]

    runner = PowsyblRunner()
    runner.load_base_grid(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    nminus1_def = extract_nminus1_definition(network_data)
    runner.store_nminus1_definition(nminus1_def)
    runner.store_action_set(extract_action_set(network_data))
    ref = runner.run_ac_loadflow(actions, [])
    n_0_ref, n_1_ref, success_ref = extract_solver_matrices_polars(ref, nminus1_def, 0)
    assert np.sum(success_ref) > len(success_ref) / 2, "At least half of the AC loadflows should have converged"

    assert len(success_ref) == len(n_1_ref)
    assert n_0.shape == n_0_ref.shape
    assert n_1.shape == n_1_ref.shape

    # Look only at successful loadflows
    n_1_ref = n_1_ref[success_ref]
    n_1 = n_1[success_ref]
    n_1_dc = n_1_dc[success_ref]

    # We assume that the N-1 loadflows on AC are closer
    ac_dist = np.abs(n_1) - np.abs(n_1_ref)
    dc_dist = np.abs(n_1_dc) - np.abs(n_1_ref)

    ac_better = np.abs(ac_dist) < np.abs(dc_dist)
    dc_better = np.abs(dc_dist) < np.abs(ac_dist)

    assert np.sum(ac_better) > np.sum(dc_better), "DC Better"
    # Comment this in if you want the test to always fail
    # assert np.sum(dc_better) > np.sum(ac_better), "AC Better"


@pytest.mark.parametrize("topo_idx", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_n0_in_ac_split_with_disconnections(preprocessed_powsybl_data_folder: Path, topo_idx: int) -> None:
    network_data = load_network_data(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["network_data_file_path"])

    post_process_file_path = (
        preprocessed_powsybl_data_folder
        / POSTPROCESSING_PATHS["dc_optimizer_snapshots_path"]
        / OUTPUT_FILE_NAMES["multiple_topologies"]
    )
    with open(post_process_file_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    topo_data = res["best_topos"][topo_idx]
    actions = topo_data["actions"]
    disconnections = topo_data["disconnection"]

    static_information_dc = load_static_information(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )
    (n_0_dc, n_1_dc), success_dc = run_solver_symmetric(
        topologies=ActionIndexComputations(action=jnp.array([actions], dtype=int), pad_mask=jnp.array([True])),
        disconnections=jnp.array(disconnections)[None],
        injections=None,
        dynamic_information=static_information_dc.dynamic_information,
        solver_config=static_information_dc.solver_config,
        aggregate_output_fn=lambda lf_res: (lf_res.n_0_matrix, lf_res.n_1_matrix),
    )
    if not all(success_dc):
        pytest.skip("Solver did not converge for all of the disconnections")

    # Run the unsplit topology with N-0 in AC
    static_information = convert_to_jax(
        network_data=network_data,
        ac_dc_interpolation=1.0,
    )

    (n_0, n_1), success = run_solver_symmetric(
        topologies=ActionIndexComputations(action=jnp.array([actions], dtype=int), pad_mask=jnp.array([True])),
        disconnections=jnp.array(disconnections)[None],
        injections=None,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
        aggregate_output_fn=lambda lf_res: (lf_res.n_0_matrix, lf_res.n_1_matrix),
    )
    assert np.all(success)

    assert n_0.shape == n_0_dc.shape
    assert n_1.shape == n_1_dc.shape
    assert not np.allclose(n_0, n_0_dc)
    assert not np.allclose(n_1, n_1_dc)

    # First batch, first timestep
    n_0 = n_0[0, 0]
    n_1 = n_1[0, 0]
    n_0_dc = n_0_dc[0, 0]
    n_1_dc = n_1_dc[0, 0]

    runner = PowsyblRunner()
    runner.load_base_grid(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    nminus1_def = extract_nminus1_definition(network_data)
    runner.store_nminus1_definition(nminus1_def)
    runner.store_action_set(extract_action_set(network_data))
    ref = runner.run_ac_loadflow(actions, disconnections)
    n_0_ref, n_1_ref, success_ref = extract_solver_matrices_polars(ref, nminus1_def, 0)
    assert np.sum(success_ref) > len(success_ref) / 2, "At least half of the AC loadflows should have converged"

    assert len(success_ref) == len(n_1_ref)
    assert n_0.shape == n_0_ref.shape
    assert n_1.shape == n_1_ref.shape

    # Look only at successful loadflows
    n_1_ref = n_1_ref[success_ref]
    n_1 = n_1[success_ref]
    n_1_dc = n_1_dc[success_ref]

    # We assume that the N-1 loadflows on AC are closer
    ac_dist = np.abs(n_1) - np.abs(n_1_ref)
    dc_dist = np.abs(n_1_dc) - np.abs(n_1_ref)

    ac_better = np.abs(ac_dist) < np.abs(dc_dist)
    dc_better = np.abs(dc_dist) < np.abs(ac_dist)

    assert np.sum(ac_better) > np.sum(dc_better), "DC Better"
    # Comment this in if you want the test to always fail
    # assert np.sum(dc_better) > np.sum(ac_better), "AC Better"
