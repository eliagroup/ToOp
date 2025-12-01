from copy import deepcopy
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandapower as pp
from fsspec.implementations.dirfs import DirFileSystem
from jax_dataclasses import replace
from pandapower.pypower.idx_brch import PF
from toop_engine_dc_solver.jax.compute_batch import compute_symmetric_batch
from toop_engine_dc_solver.jax.injections import default_injection
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_dc_solver.jax.topology_computations import (
    convert_topo_sel_sorted,
    convert_topo_to_action_set_index,
    default_topology,
)
from toop_engine_dc_solver.postprocess.postprocess_pandapower import (
    compute_n_1_dc,
)
from toop_engine_dc_solver.preprocess.convert_to_jax import convert_to_jax
from toop_engine_dc_solver.preprocess.helpers.find_bridges import (
    find_n_minus_2_safe_branches,
)
from toop_engine_dc_solver.preprocess.network_data import load_network_data
from toop_engine_dc_solver.preprocess.pandapower.pandapower_backend import PandaPowerBackend
from toop_engine_dc_solver.preprocess.preprocess import preprocess
from toop_engine_grid_helpers.pandapower.pandapower_helpers import (
    check_for_splits,
    get_pandapower_branch_loadflow_results_sequence,
    get_pandapower_loadflow_results_in_ppc,
    get_pandapower_loadflow_results_injection,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import table_id
from toop_engine_interfaces.folder_structure import (
    CHRONICS_FILE_NAMES,
    PREPROCESSING_PATHS,
)


def test_grid_loadflow_uptodate(data_folder: Path) -> None:
    grid_file_path = data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"]
    net = pp.from_json(grid_file_path)
    net_copy = deepcopy(net)
    pp.rundcpp(net_copy)
    assert np.allclose(net.res_bus.p_mw.values, net_copy.res_bus.p_mw.values)
    assert np.allclose(net.res_line.p_from_mw, net_copy.res_line.p_from_mw)


def test_n_0_results(data_folder: Path, preprocessed_data_folder: Path) -> None:
    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir)
    pp.rundcpp(backend.net)

    abs_backend_loadflow = np.abs(backend.net._ppc["internal"]["branch"][backend.get_monitored_branch_mask(), PF].real)

    static_information = load_static_information(
        preprocessed_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )
    lf_res, success = compute_symmetric_batch(
        default_topology(static_information.solver_config),
        None,
        None,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
    )
    assert np.all(success)
    n_0 = lf_res.n_0_matrix

    assert np.allclose(abs_backend_loadflow, np.abs(n_0), rtol=1e-3, atol=1e-3)
    # Two batch dimensions, time and batch
    assert abs_backend_loadflow.shape == n_0[0, 0, :].shape


def test_nminus1_results_one_timestep(data_folder: Path, preprocessed_data_folder: Path) -> None:
    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir)
    pp.rundcpp(backend.net)
    network_data = load_network_data(preprocessed_data_folder / PREPROCESSING_PATHS["network_data_file_path"])
    static_information = load_static_information(
        preprocessed_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )

    lf_res, success = compute_symmetric_batch(
        default_topology(static_information.solver_config),
        None,
        None,
        dynamic_information=static_information.dynamic_information,
        solver_config=static_information.solver_config,
    )
    assert np.all(success)
    n_1 = lf_res.n_1_matrix

    n_1_loadflows = jnp.abs(n_1)

    n_1_ref, _ = compute_n_1_dc(backend.net, network_data)

    n_1_ref = np.abs(n_1_ref)
    assert n_1_loadflows[0, 0].shape == n_1_ref.shape
    assert np.allclose(n_1_loadflows[0, 0], n_1_ref, rtol=1e-3, atol=1e-3)


def test_n_0_results_with_disconnection(data_folder: Path) -> None:
    # Fake backend that always returns all branches as monitored
    # This is needed to ensure we're not accidentally making N-2 safe branches only N-1 safe
    # by removing the fallback-branches from the monitored branches
    class FakeBackend(PandaPowerBackend):
        def get_monitored_branch_mask(self):
            branch_mask = np.ones_like(super().get_monitored_branch_mask())
            # Exclude xwards aux-branches since they would always lead to islanding
            from_branch, to_branch = self.net._pd2ppc_lookups["branch"]["xward"]
            branch_mask[from_branch:to_branch] = False
            return branch_mask

    filesystem_dir = DirFileSystem(str(data_folder))
    backend = FakeBackend(filesystem_dir)
    network_data = preprocess(backend)
    static_information = convert_to_jax(network_data)

    static_information = replace(
        static_information,
        solver_config=replace(static_information.solver_config, batch_size_bsdf=1, batch_size_injection=1),
    )

    n_branch = np.sum(static_information.solver_config.branches_per_sub.val).item()
    branch_topologies = jnp.zeros((static_information.solver_config.batch_size_bsdf, n_branch), dtype=bool)
    branch_topology_batch = convert_topo_sel_sorted(branch_topologies, static_information.solver_config.branches_per_sub)
    action_index_topo, _ = convert_topo_to_action_set_index(
        topologies=branch_topology_batch,
        branch_actions=static_information.dynamic_information.action_set,
        extend_action_set=False,
    )

    n_2_safe_mask = find_n_minus_2_safe_branches(
        from_node=np.array(static_information.dynamic_information.from_node),
        to_node=np.array(static_information.dynamic_information.to_node),
        number_of_branches=static_information.dynamic_information.ptdf.shape[0],
        number_of_nodes=static_information.dynamic_information.ptdf.shape[1],
    )

    assert np.any(n_2_safe_mask)

    disconnected_branch = 0

    disconnection_batch = jnp.full(
        (static_information.solver_config.batch_size_bsdf, 1),
        disconnected_branch,
        dtype=int,
    )
    inj_topology_batch = jnp.zeros(
        (
            static_information.solver_config.batch_size_bsdf,
            action_index_topo.action.shape[1],
            static_information.dynamic_information.generators_per_sub.max().item(),
        ),
        dtype=bool,
    )

    lf_res, success = compute_symmetric_batch(
        action_index_topo,
        disconnection_batch,
        inj_topology_batch,
        static_information.dynamic_information,
        static_information.solver_config,
    )
    assert jnp.all(success)
    abs_solver_loadflow = np.abs(lf_res.n_0_matrix)
    actual_disconnected_branch = static_information.dynamic_information.disconnectable_branches[disconnected_branch]
    if actual_disconnected_branch in static_information.dynamic_information.branches_monitored:
        del_idx = np.argwhere(static_information.dynamic_information.branches_monitored == actual_disconnected_branch).item()
        abs_solver_loadflow = np.delete(abs_solver_loadflow.flatten(), del_idx)

    pp_type = network_data.branch_types[actual_disconnected_branch]
    pp_id = table_id(network_data.branch_ids[actual_disconnected_branch])
    full_id = network_data.branch_ids[actual_disconnected_branch]
    net_copy = deepcopy(backend.net)
    net_copy[pp_type].loc[pp_id, "in_service"] = False
    orig_idx = backend.get_branch_ids().index(full_id)

    pp.rundcpp(net_copy)

    abs_backend_loadflow = np.abs(
        net_copy._ppc["internal"]["branch"][np.delete(backend.get_monitored_branch_mask(), orig_idx), PF].real
    )

    assert np.allclose(abs_backend_loadflow, abs_solver_loadflow)
    assert abs_backend_loadflow.shape == abs_solver_loadflow.shape


def test_multi_timestep(data_folder: Path) -> None:
    chronics_path = Path(data_folder) / PREPROCESSING_PATHS["chronics_path"]
    load_p = np.load(chronics_path / "0000" / CHRONICS_FILE_NAMES["load_p"])
    gen_p = np.load(chronics_path / "0000" / CHRONICS_FILE_NAMES["gen_p"])
    sgen_p = np.load(chronics_path / "0000" / CHRONICS_FILE_NAMES["sgen_p"])
    dcline_p = np.load(chronics_path / "0000" / CHRONICS_FILE_NAMES["dcline_p"])

    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir, chronics_id=0, chronics_slice=slice(0, 3))
    network_data = preprocess(backend)
    static_information = convert_to_jax(network_data)

    loadflows = []
    net = deepcopy(backend.net)
    for timestep in range(3):
        net.load.loc[:, "p_mw"] = load_p[timestep]
        net.gen.loc[:, "p_mw"] = gen_p[timestep]
        net.sgen.loc[:, "p_mw"] = sgen_p[timestep]
        net.dcline.loc[:, "p_mw"] = dcline_p[timestep]
        pp.rundcpp(net)

        abs_backend_loadflow = np.abs(net._ppc["internal"]["branch"][backend.get_monitored_branch_mask(), PF].real)

        loadflows.append(abs_backend_loadflow)

    abs_backend_loadflow = np.stack(loadflows, axis=0)

    unsplit_topo = default_topology(static_information.solver_config)
    unsplit_inj = default_injection(
        n_splits=unsplit_topo.action.shape[1],
        max_inj_per_sub=static_information.dynamic_information.max_inj_per_sub,
        batch_size=unsplit_topo.action.shape[0],
    )

    lf_res, success = compute_symmetric_batch(
        unsplit_topo,
        None,
        unsplit_inj.injection_topology,
        static_information.dynamic_information,
        static_information.solver_config,
    )
    assert np.all(success)
    abs_solver_loadflow = np.abs(lf_res.n_0_matrix)
    assert abs_backend_loadflow.shape == abs_solver_loadflow[0, :, :].shape
    assert np.allclose(abs_backend_loadflow, abs_solver_loadflow, rtol=1e-3, atol=1e-3)


def test_extract_loadflow_results(data_folder: Path) -> None:
    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir)
    network_data = preprocess(backend)
    net = backend.net
    pp.rundcpp(net)

    direct_loadflows = net._ppc["internal"]["branch"][backend.get_monitored_branch_mask(), PF].real
    direct_loadflows = np.abs(direct_loadflows)

    ppc_loadflows = get_pandapower_loadflow_results_in_ppc(net)
    ppc_loadflows = ppc_loadflows[net._ppc["internal"]["branch_is"]]
    ppc_loadflows = ppc_loadflows[backend.get_monitored_branch_mask()]
    ppc_loadflows = np.abs(ppc_loadflows)

    monitored_branch_types = [network_data.branch_types[i] for i in np.flatnonzero(network_data.monitored_branch_mask)]
    monitored_branch_ids = [table_id(network_data.branch_ids[i]) for i in np.flatnonzero(network_data.monitored_branch_mask)]
    sequence_loadflows = get_pandapower_branch_loadflow_results_sequence(
        net, monitored_branch_types, monitored_branch_ids, "active"
    )
    sequence_loadflows = np.abs(sequence_loadflows)

    assert direct_loadflows.shape == ppc_loadflows.shape
    assert direct_loadflows.shape == sequence_loadflows.shape
    assert np.allclose(direct_loadflows, ppc_loadflows)
    assert np.allclose(direct_loadflows, sequence_loadflows)

    isnan = check_for_splits(net, monitored_branch_types, monitored_branch_ids)
    assert isnan is False

    # Check injections
    inj_types = [t for t in network_data.injection_types if t != "PST"]
    inj_ids = [table_id(i) for t, i in zip(network_data.injection_types, network_data.injection_ids) if t != "PST"]

    inj_p = get_pandapower_loadflow_results_injection(net, inj_types, inj_ids)

    assert inj_p.shape == (len(inj_ids),)
    assert np.isclose(np.sum(inj_p), net.res_ext_grid.p_mw.sum())
