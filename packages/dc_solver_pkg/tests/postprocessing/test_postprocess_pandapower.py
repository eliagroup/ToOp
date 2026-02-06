# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import json
from copy import deepcopy
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandapower as pp
import pytest
import ray
from fsspec.implementations.dirfs import DirFileSystem
from jax_dataclasses import replace
from toop_engine_dc_solver.jax.injections import default_injection
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_dc_solver.jax.topology_computations import (
    default_topology,
)
from toop_engine_dc_solver.jax.topology_looper import run_solver_symmetric
from toop_engine_dc_solver.jax.types import ActionIndexComputations
from toop_engine_dc_solver.postprocess.postprocess_pandapower import (
    PandapowerRunner,
    apply_disconnections,
    apply_topology,
    compute_cross_coupler_flows,
)
from toop_engine_dc_solver.preprocess.convert_to_jax import convert_to_jax
from toop_engine_dc_solver.preprocess.network_data import (
    extract_action_set,
    extract_nminus1_definition,
    load_network_data,
)
from toop_engine_dc_solver.preprocess.pandapower.pandapower_backend import PandaPowerBackend
from toop_engine_dc_solver.preprocess.preprocess import preprocess
from toop_engine_grid_helpers.pandapower.pandapower_helpers import (
    get_pandapower_loadflow_results_in_ppc,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import (
    get_globally_unique_id,
    table_ids,
)
from toop_engine_interfaces.folder_structure import (
    OUTPUT_FILE_NAMES,
    POSTPROCESSING_PATHS,
    PREPROCESSING_PATHS,
)
from toop_engine_interfaces.loadflow_result_helpers_polars import (
    extract_node_matrices_polars,
    extract_solver_matrices_polars,
)
from toop_engine_interfaces.nminus1_definition import GridElement
from toop_engine_interfaces.stored_action_set import ActionSet


def test_apply_topology_unsplit(data_folder: str) -> None:
    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir)
    net = backend.net
    pp.rundcpp(net)
    network_data = preprocess(backend)

    action_set = extract_action_set(network_data)

    new_net, _ = apply_topology(net, [], action_set)

    pp.rundcpp(new_net)
    assert np.allclose(net.res_line.loading_percent, new_net.res_line.loading_percent, equal_nan=True)

    # Pick an action from the action set
    action = [2]

    new_net, _ = apply_topology(net, action, action_set)

    pp.rundcpp(new_net)

    assert not np.allclose(net.res_line.loading_percent, new_net.res_line.loading_percent, equal_nan=True)

    with pytest.raises(ValueError):
        apply_topology(net, [99999999], action_set)


def test_apply_disconnections(data_folder: str) -> None:
    grid_file_path = Path(data_folder) / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    net = pp.from_json(grid_file_path)

    action_set = ActionSet.model_construct(
        disconnectable_branches=[GridElement(id=get_globally_unique_id(0, "line"), type="line", name="test", kind="branch")]
    )

    assert net.line.loc[0, "in_service"]
    net = apply_disconnections(
        net,
        [0],
        action_set,
    )

    assert not net.line.loc[0, "in_service"]
    pp.rundcpp(net)

    with pytest.raises(ValueError):
        apply_disconnections(
            net,
            [99999999],
            action_set,
        )


def test_compute_n_1_dc(data_folder: str) -> None:
    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir)
    network_data = preprocess(backend)
    static_information = convert_to_jax(network_data)

    topo = default_topology(static_information.solver_config)
    inj = default_injection(
        n_splits=topo.action.shape[1],
        max_inj_per_sub=static_information.dynamic_information.max_inj_per_sub,
        batch_size=topo.action.shape[0],
    )

    (n_0_solver, n_1_solver), success = run_solver_symmetric(
        topologies=topo,
        disconnections=None,
        injections=inj.injection_topology,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
        aggregate_output_fn=lambda lf_res: (lf_res.n_0_matrix, lf_res.n_1_matrix),
    )
    n_0_solver = np.abs(n_0_solver)[0, 0]
    n_1_solver = np.abs(n_1_solver)[0, 0]
    assert np.all(success)

    grid_file_path = Path(data_folder) / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    runner = PandapowerRunner()
    runner.load_base_grid(grid_file_path)
    action_set = extract_action_set(network_data)
    runner.store_action_set(action_set)
    nminus1_def = extract_nminus1_definition(network_data)
    runner.store_nminus1_definition(nminus1_def)

    res = runner.run_dc_loadflow([], [])
    n_0, n_1, success = extract_solver_matrices_polars(
        loadflow_results=res,
        nminus1_definition=nminus1_def,
        timestep=0,
    )
    assert np.all(success)

    assert n_0.shape == n_0_solver.shape
    assert np.allclose(np.abs(n_0), n_0_solver)
    assert n_1.shape == n_1_solver.shape
    assert np.allclose(np.abs(n_1), n_1_solver)

    # Test N-0 alone
    res_n0 = runner.run_dc_n_0([], [])
    n_0_single, _, _ = extract_solver_matrices_polars(
        loadflow_results=res_n0,
        nminus1_definition=nminus1_def,
        timestep=0,
    )
    assert np.allclose(n_0_single, n_0)

    grid_file_path = Path(data_folder) / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    # Test parallelization
    runner = PandapowerRunner(n_processes=2)
    runner.load_base_grid(grid_file_path)
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(nminus1_def)

    res_mt = runner.run_dc_loadflow([], [])
    n_0_mt, n_1_mt, success_mt = extract_solver_matrices_polars(
        loadflow_results=res_mt,
        nminus1_definition=nminus1_def,
        timestep=0,
    )
    assert np.all(success_mt)

    assert n_0_mt.shape == n_0.shape
    assert np.allclose(n_0_mt, n_0)
    assert n_1_mt.shape == n_1.shape
    assert np.allclose(n_1_mt, n_1)

    # Test parallelization again with batch size set
    runner = PandapowerRunner(n_processes=2, batch_size=4)
    runner.load_base_grid(grid_file_path)
    runner.store_action_set(extract_action_set(network_data))
    runner.store_nminus1_definition(extract_nminus1_definition(network_data))

    res_mt = runner.run_dc_loadflow([], [])
    n_0_mt, n_1_mt, success_mt = extract_solver_matrices_polars(
        loadflow_results=res_mt,
        nminus1_definition=nminus1_def,
        timestep=0,
    )
    assert np.all(success_mt)
    assert n_0_mt.shape == n_0.shape
    assert np.allclose(n_0_mt, n_0)
    assert n_1_mt.shape == n_1.shape
    assert np.allclose(n_1_mt, n_1)

    outaged_branch_types = np.array(network_data.branch_types)[network_data.outaged_branch_mask]
    outaged_branch_ids = np.array(table_ids(network_data.branch_ids))[network_data.outaged_branch_mask]

    assert n_1.shape[0] == len(outaged_branch_types) + len(network_data.multi_outage_types) + sum(
        network_data.outaged_injection_mask
    )
    assert n_1.shape[1] == sum(network_data.monitored_branch_mask)

    original_in_service = backend.net._ppc["internal"]["branch_is"]
    for i, (pp_type, pp_id) in enumerate(zip(outaged_branch_types, outaged_branch_ids, strict=True)):
        outage_net = deepcopy(backend.net)
        outage_net[pp_type].loc[int(pp_id), "in_service"] = False
        pp.rundcpp(outage_net)

        ppc_loadflows = get_pandapower_loadflow_results_in_ppc(outage_net)
        # We use original in service, since the ppci from the outaged net would exclude the outaged branch and our masks would not fit
        ppci_loadflows = ppc_loadflows[original_in_service]
        monitored_loadflows = ppci_loadflows[backend.get_monitored_branch_mask()]
        # Compare the first time step only
        assert n_1[i].shape == monitored_loadflows.shape
        assert np.allclose(
            np.abs(n_1[i]),
            np.abs(monitored_loadflows),
        )

    offset = len(outaged_branch_types)

    # Test multi outages
    for i, (pp_type, pp_id) in enumerate(
        zip(network_data.multi_outage_types, table_ids(network_data.multi_outage_ids), strict=True)
    ):
        outage_net = deepcopy(backend.net)
        outage_net[pp_type].loc[int(pp_id), "in_service"] = False
        pp.rundcpp(outage_net)

        ppc_loadflows = get_pandapower_loadflow_results_in_ppc(outage_net)
        # We use original in service, since the ppci from the outaged net would exclude the outaged branch and our masks would not fit
        ppci_loadflows = ppc_loadflows[original_in_service]
        monitored_loadflows = ppci_loadflows[backend.get_monitored_branch_mask()]
        # Compare the first time step only
        assert n_1[i + offset].shape == monitored_loadflows.shape
        assert np.allclose(
            np.abs(n_1[i + offset]),
            np.abs(monitored_loadflows),
        )

    offset += len(network_data.multi_outage_types)

    # Test nonrel injections
    for i, (ref_id) in enumerate(network_data.nonrel_io_global_inj_index):
        pp_id = table_ids(network_data.injection_ids)[ref_id]
        pp_type = network_data.injection_types[ref_id]
        outage_net = deepcopy(backend.net)
        outage_net[pp_type].loc[int(pp_id), "in_service"] = False
        pp.rundcpp(outage_net)

        ppc_loadflows = get_pandapower_loadflow_results_in_ppc(outage_net)
        # We use original in service, since the ppci from the outaged net would exclude the outaged branch and our masks would not fit
        ppci_loadflows = ppc_loadflows[original_in_service]
        monitored_loadflows = ppci_loadflows[backend.get_monitored_branch_mask()]
        # Compare the first time step only
        assert n_1[i + offset].shape == monitored_loadflows.shape
        assert np.allclose(
            np.abs(n_1[i + offset]),
            np.abs(monitored_loadflows),
        )

    offset += len(network_data.nonrel_io_global_inj_index)

    # Test rel injections
    for i, (ref_id) in enumerate(network_data.rel_io_global_inj_index):
        pp_id = table_ids(network_data.injection_ids)[ref_id]
        pp_type = network_data.injection_types[ref_id]
        outage_net = deepcopy(backend.net)
        outage_net[pp_type].loc[int(pp_id), "in_service"] = False
        pp.rundcpp(outage_net)

        ppc_loadflows = get_pandapower_loadflow_results_in_ppc(outage_net)
        # We use original in service, since the ppci from the outaged net would exclude the outaged branch and our masks would not fit
        ppci_loadflows = ppc_loadflows[original_in_service]
        monitored_loadflows = ppci_loadflows[backend.get_monitored_branch_mask()]
        # Compare the first time step only
        assert n_1[i + offset].shape == monitored_loadflows.shape
        assert np.allclose(
            np.abs(n_1[i + offset]),
            np.abs(monitored_loadflows),
        )

    offset += len(network_data.rel_io_global_inj_index)
    assert offset == n_1.shape[0]
    ray.shutdown()


def test_compute_n_1_ac(data_folder: str) -> None:
    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir)
    backend.net
    network_data = preprocess(backend)

    grid_file_path = Path(data_folder) / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    runner = PandapowerRunner()
    runner.load_base_grid(grid_file_path)
    runner.store_action_set(extract_action_set(network_data))
    nminus1_definition = extract_nminus1_definition(network_data)
    runner.store_nminus1_definition(nminus1_definition)

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

    assert np.all(success)
    assert n_1.shape[0] == sum(network_data.outaged_branch_mask) + len(network_data.multi_outage_types) + sum(
        network_data.outaged_injection_mask
    )

    assert n_1.shape[1] == sum(network_data.monitored_branch_mask)

    outaged_branch_types = np.array(network_data.branch_types)[network_data.outaged_branch_mask]
    outaged_branch_ids = np.array(table_ids(network_data.branch_ids))[network_data.outaged_branch_mask]
    monitored_node_ids = table_ids([elem.id for elem in nminus1_definition.monitored_elements if elem.kind == "bus"])
    assert vm_n1.shape == (n_1.shape[0], len(monitored_node_ids))
    assert va_n1.shape == (n_1.shape[0], len(monitored_node_ids))

    original_in_service = backend.net._ppc["internal"]["branch_is"]
    bus_voltages = backend.net.bus.reindex(monitored_node_ids).vn_kv.values
    for i, (pp_type, pp_id) in enumerate(zip(outaged_branch_types, outaged_branch_ids)):
        outage_net = deepcopy(backend.net)
        outage_net[pp_type].loc[int(pp_id), "in_service"] = False
        pp.runpp(outage_net)

        ppc_loadflows = get_pandapower_loadflow_results_in_ppc(outage_net)
        # We use original in service, since the ppci from the outaged net would exclude the outaged branch and our masks would not fit
        ppci_loadflows = ppc_loadflows[original_in_service]
        monitored_loadflows = ppci_loadflows[backend.get_monitored_branch_mask()]
        # Compare the first time step only
        assert n_1[i].shape == monitored_loadflows.shape
        assert np.allclose(
            np.abs(n_1[i]),
            np.abs(monitored_loadflows),
        )
        monitored_buses_reindexed = outage_net.res_bus.reindex(monitored_node_ids)
        assert np.allclose(vm_n1[i], monitored_buses_reindexed.vm_pu.values * bus_voltages, equal_nan=True)
        assert np.allclose(va_n1[i], monitored_buses_reindexed.va_degree.values, equal_nan=True)
    offset = len(outaged_branch_types)

    # Test multi outages
    for i, (pp_type, pp_id) in enumerate(zip(network_data.multi_outage_types, table_ids(network_data.multi_outage_ids))):
        outage_net = deepcopy(backend.net)
        outage_net[pp_type].loc[int(pp_id), "in_service"] = False
        pp.runpp(outage_net)

        ppc_loadflows = get_pandapower_loadflow_results_in_ppc(outage_net)
        # We use original in service, since the ppci from the outaged net would exclude the outaged branch and our masks would not fit
        ppci_loadflows = ppc_loadflows[original_in_service]
        monitored_loadflows = ppci_loadflows[backend.get_monitored_branch_mask()]
        # Compare the first time step only
        assert n_1[i + offset].shape == monitored_loadflows.shape
        assert np.allclose(
            np.abs(n_1[i + offset]),
            np.abs(monitored_loadflows),
        )
        monitored_buses_reindexed = outage_net.res_bus.reindex(monitored_node_ids)

        assert np.allclose(vm_n1[i + offset], monitored_buses_reindexed.vm_pu.values * bus_voltages, equal_nan=True)
        assert np.allclose(va_n1[i + offset], monitored_buses_reindexed.va_degree.values, equal_nan=True)


@pytest.mark.xdist_group("performance")
@pytest.mark.timeout(600)
def test_runner_matches_split_loadflows(preprocessed_data_folder: str) -> None:
    data_path = Path(preprocessed_data_folder)
    network_data = load_network_data(data_path / PREPROCESSING_PATHS["network_data_file_path"])
    action_set = extract_action_set(network_data)
    grid_file_path = Path(preprocessed_data_folder) / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    runner = PandapowerRunner(n_processes=8)
    runner.load_base_grid(grid_file_path)
    runner.store_action_set(action_set)
    runner.store_nminus1_definition(extract_nminus1_definition(network_data))
    static_information = load_static_information(
        preprocessed_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )
    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=1),
    )
    post_process_file_path = (
        preprocessed_data_folder
        / POSTPROCESSING_PATHS["dc_optimizer_snapshots_path"]
        / OUTPUT_FILE_NAMES["multiple_topologies"]
    )
    with open(post_process_file_path, "r", encoding="utf-8") as f:
        optim_res = json.load(f)
    for i in range(len(optim_res["best_topos"])):
        actions = optim_res["best_topos"][i]["actions"]
        disconnections = optim_res["best_topos"][i]["disconnection"]

        disconnections_jax = jnp.array(disconnections)[None]

        (n_0, n_1), success = run_solver_symmetric(
            ActionIndexComputations(action=jnp.array([actions], dtype=int), pad_mask=jnp.array([True])),
            disconnections_jax,
            None,
            static_information.dynamic_information,
            static_information.solver_config,
            lambda lf_res: (lf_res.n_0_matrix, lf_res.n_1_matrix),
        )
        n_0 = np.abs(n_0[0, 0])
        n_1 = np.abs(n_1[0, 0])
        assert np.all(success)

        res = runner.run_dc_loadflow(actions, disconnections)
        assert runner.get_last_action_info() is not None
        n_0_ref, n_1_ref, success = extract_solver_matrices_polars(
            loadflow_results=res,
            nminus1_definition=extract_nminus1_definition(network_data),
            timestep=0,
        )
        assert np.all(success)
        n_0_ref = np.abs(n_0_ref)
        n_1_ref = np.abs(n_1_ref)
        assert n_0.shape == n_0_ref.shape
        assert np.allclose(n_0, n_0_ref)
        assert n_1.shape == n_1_ref.shape
        assert np.allclose(n_1, n_1_ref)
    ray.shutdown()


def test_compute_cross_coupler_flows(preprocessed_data_folder: str) -> None:
    filesystem_dir = DirFileSystem(str(preprocessed_data_folder))
    backend = PandaPowerBackend(filesystem_dir)
    net = backend.net
    data_path = Path(preprocessed_data_folder)
    network_data = load_network_data(data_path / PREPROCESSING_PATHS["network_data_file_path"])
    action_set = extract_action_set(network_data)
    post_process_file_path = (
        data_path / POSTPROCESSING_PATHS["dc_optimizer_snapshots_path"] / OUTPUT_FILE_NAMES["multiple_topologies"]
    )
    grid_file_path = Path(data_path) / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    with open(post_process_file_path, "r", encoding="utf-8") as f:
        res = json.load(f)
    actions = res["best_topos"][0]["actions"]
    assert len(actions)

    cross_coupler_p_ref, _, success = compute_cross_coupler_flows(net, actions, action_set, "dc")
    assert np.all(success)
    assert cross_coupler_p_ref.shape == (len(actions),)

    static_information = load_static_information(data_path / PREPROCESSING_PATHS["static_information_file_path"])

    static_information = replace(
        static_information,
        solver_config=replace(
            static_information.solver_config,
            batch_size_bsdf=1,
        ),
    )

    cc_flows, success = run_solver_symmetric(
        topologies=ActionIndexComputations(
            action=jnp.array([actions]),
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
