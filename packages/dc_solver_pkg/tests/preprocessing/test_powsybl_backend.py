from pathlib import Path

import numpy as np
import pypowsybl
import pypowsybl.loadflow.impl
import pypowsybl.loadflow.impl.loadflow
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.example_grids import case30_with_psts_powsybl
from toop_engine_dc_solver.jax.injections import get_all_injection_outage_deltap
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_dc_solver.jax.lodf import calc_lodf
from toop_engine_dc_solver.jax.topology_computations import (
    default_topology,
)
from toop_engine_dc_solver.jax.topology_looper import run_solver_symmetric
from toop_engine_dc_solver.postprocess.postprocess_powsybl import (
    PowsyblRunner,
)
from toop_engine_dc_solver.preprocess.helpers.psdf import compute_psdf
from toop_engine_dc_solver.preprocess.helpers.ptdf import compute_ptdf
from toop_engine_dc_solver.preprocess.network_data import (
    extract_action_set,
    extract_nminus1_definition,
    load_network_data,
    validate_network_data,
)
from toop_engine_dc_solver.preprocess.powsybl.powsybl_backend import PowsyblBackend
from toop_engine_dc_solver.preprocess.preprocess import preprocess
from toop_engine_grid_helpers.powsybl.loadflow_parameters import DISTRIBUTED_SLACK, SINGLE_SLACK
from toop_engine_interfaces.folder_structure import (
    PREPROCESSING_PATHS,
)
from toop_engine_interfaces.loadflow_result_helpers_polars import extract_solver_matrices_polars
from toop_engine_interfaces.nminus1_definition import load_nminus1_definition


def test_get_branches(powsybl_data_folder: Path) -> None:
    filesystem_dir_powsybl = DirFileSystem(str(powsybl_data_folder))
    backend = PowsyblBackend(filesystem_dir_powsybl)
    n_branches = len(backend.get_branch_ids())
    n_timesteps = 1

    assert backend.get_from_nodes().shape == (n_branches,)
    assert backend.get_to_nodes().shape == (n_branches,)
    assert backend.get_susceptances().shape == (n_branches,)
    assert np.all(np.abs(backend.get_susceptances()) > 1e-10)
    assert backend.get_max_mw_flows().shape == (n_timesteps, n_branches)
    assert backend.get_max_mw_flows_n_1().shape == (n_timesteps, n_branches)
    assert backend.get_overload_weights().shape == (n_branches,)
    assert len(backend.get_branch_ids()) == n_branches
    assert len(backend.get_branch_names()) == n_branches
    assert len(backend.get_branch_types()) == n_branches
    assert backend.get_branches_in_maintenance().shape == (n_timesteps, n_branches)
    assert backend.get_disconnectable_branch_mask().shape == (n_branches,)
    assert backend.get_outaged_branch_mask().shape == (n_branches,)
    assert backend.get_monitored_branch_mask().shape == (n_branches,)
    psts = backend.net.get_phase_tap_changers()
    pst_trafos = backend.net.get_2_windings_transformers().loc[psts.index]
    out_of_services_psts = ~pst_trafos.connected1 | ~pst_trafos.connected2
    node_ids = backend.get_node_ids()
    masked_nodes = ~(pst_trafos.bus1_id.isin(node_ids) & pst_trafos.bus2_id.isin(node_ids))
    assert backend.get_phase_shift_mask().shape == (n_branches,)
    assert backend.get_phase_shift_mask().sum() == backend.net.get_phase_tap_changers().shape[0] - sum(
        masked_nodes | out_of_services_psts
    )
    assert backend.get_shift_angles().shape == (n_timesteps, n_branches)
    assert np.all(backend.get_shift_angles()[0, ~backend.get_phase_shift_mask()] == 0.0)
    ac_dc_diff = backend.get_ac_dc_mismatch()
    assert ac_dc_diff.shape == (n_timesteps, n_branches)
    assert np.sum(ac_dc_diff == 0) < 10
    assert np.all(np.isfinite(ac_dc_diff))


def test_get_nodes(powsybl_data_folder: Path) -> None:
    filesystem_dir_powsybl = DirFileSystem(str(powsybl_data_folder))
    backend = PowsyblBackend(filesystem_dir_powsybl)
    busses = backend.net.get_buses()
    n_connected_nodes = sum(busses.connected_component == 0)

    assert set(backend.get_node_ids()) == set(busses[busses.connected_component == 0].index.values.tolist())
    assert len(backend.get_node_ids()) == n_connected_nodes
    assert len(backend.get_node_names()) == n_connected_nodes
    assert len(backend.get_node_types()) == n_connected_nodes
    slack_id = backend.get_node_ids()[backend.get_slack()]
    assert busses.loc[slack_id]["v_angle"] == 0
    assert backend.get_relevant_node_mask().shape == (n_connected_nodes,)
    assert backend.get_cross_coupler_limits().shape == (n_connected_nodes,)


def test_get_injections(powsybl_data_folder: Path) -> None:
    filesystem_dir_powsybl = DirFileSystem(str(powsybl_data_folder))
    backend = PowsyblBackend(filesystem_dir_powsybl)
    n_injections = len(backend.get_injection_ids())
    n_timesteps = 1

    assert backend.get_injection_nodes().shape == (n_injections,)
    assert backend.get_mw_injections().shape == (n_timesteps, n_injections)
    assert len(backend.get_injection_ids()) == n_injections
    assert len(backend.get_injection_names()) == n_injections
    assert len(backend.get_injection_types()) == n_injections
    assert backend.get_outaged_injection_mask().shape == (n_injections,)


def test_ptdf_matrix(powsybl_data_folder: Path) -> None:
    filesystem_dir_powsybl = DirFileSystem(str(powsybl_data_folder))
    backend = PowsyblBackend(filesystem_dir_powsybl)
    net = backend.net
    loadflow_ref = -net.get_branches()["p1"].loc[backend.get_branch_ids()].values

    ptdf = compute_ptdf(
        backend.get_from_nodes(),
        backend.get_to_nodes(),
        backend.get_susceptances(),
        backend.get_slack(),
    )
    psdf = compute_psdf(
        ptdf,
        backend.get_from_nodes(),
        backend.get_to_nodes(),
        backend.get_susceptances(),
        backend.get_phase_shift_mask(),
        backend.get_base_mva(),
    )

    busses = backend.net.get_buses()
    injections = backend.net.get_injections()
    injections["tie_line_id"] = backend.net.get_dangling_lines()["tie_line_id"]
    injections = injections[(injections.tie_line_id == "") | (injections.tie_line_id.isna())]
    busses["p"] = injections.groupby("bus_id")["p"].sum()
    busses["p"] = busses["p"].fillna(0)
    nodal_injections = busses["p"].loc[backend.get_node_ids()].values
    shift_angles = backend.get_shift_angles()[0, backend.get_phase_shift_mask()]
    loadflow = ptdf @ nodal_injections
    loadflow += psdf @ shift_angles

    assert np.allclose(loadflow, loadflow_ref)

    pypowsybl.loadflow.run_ac(net, DISTRIBUTED_SLACK)
    ac_loadflow_ref = net.get_branches()["p1"].loc[backend.get_branch_ids()].values

    ac_loadflow = loadflow + backend.get_ac_dc_mismatch()[0, :]
    assert np.allclose(np.abs(ac_loadflow), np.abs(ac_loadflow_ref))


def test_extract_network_data(powsybl_data_folder: Path) -> None:
    filesystem_dir_powsybl = DirFileSystem(str(powsybl_data_folder))
    backend = PowsyblBackend(filesystem_dir_powsybl)
    network_data = preprocess(backend)
    assert network_data.ptdf.size > 0
    assert network_data.nodal_injection.size > 0

    lf_results = np.abs(network_data.ptdf[network_data.monitored_branch_mask, :] @ network_data.nodal_injection[0])

    backend_branches = backend._get_branches()
    lf_reference = np.abs(backend_branches[backend_branches["for_reward"]]["p1"].values)

    assert lf_reference.shape == lf_results.shape
    assert np.allclose(lf_results, lf_reference)
    validate_network_data(network_data)


def test_lodf(preprocessed_powsybl_data_folder: Path) -> None:
    net = pypowsybl.network.load(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    nminus1_definition = load_nminus1_definition(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["nminus1_definition_file_path"]
    )
    runner = PowsyblRunner()
    runner.replace_grid(net)
    runner.store_nminus1_definition(nminus1_definition)
    lf_res = runner.run_dc_loadflow([], [])
    n_0, n_1, success = extract_solver_matrices_polars(
        loadflow_results=lf_res,
        nminus1_definition=nminus1_definition,
        timestep=0,
    )
    assert np.all(success)
    n_0 = np.abs(n_0)
    n_1 = np.abs(n_1)

    static_information = load_static_information(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )

    outage_idx = 63
    branch_to_outage = static_information.dynamic_information.branches_to_fail[outage_idx]

    lodf, success = calc_lodf(
        branch_to_outage=branch_to_outage,
        ptdf=static_information.dynamic_information.ptdf,
        from_node=static_information.dynamic_information.from_node,
        to_node=static_information.dynamic_information.to_node,
        branches_monitored=static_information.dynamic_information.branches_monitored,
    )
    assert np.all(success)

    base_loadflow = static_information.dynamic_information.ptdf @ static_information.dynamic_information.nodal_injections[0]
    assert np.allclose(np.abs(base_loadflow), n_0)

    diff_flow = lodf * base_loadflow[branch_to_outage]
    n_1_loadflow = np.abs(base_loadflow + diff_flow)

    assert np.allclose(n_1_loadflow, n_1[outage_idx])


def test_injection_outages_match(preprocessed_powsybl_data_folder: Path) -> None:
    net = pypowsybl.network.load(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    network_data = load_network_data(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["network_data_file_path"])
    static_information = load_static_information(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )
    dynamic_information = static_information.dynamic_information

    nminus1_def = extract_nminus1_definition(network_data)
    outaged_injections = [
        cont for cont in nminus1_def.contingencies if not cont.is_basecase() and cont.elements[0].kind == "injection"
    ]
    pypowsybl.loadflow.run_dc(net, DISTRIBUTED_SLACK if network_data.metadata["distributed_slack"] else SINGLE_SLACK)

    assert len(outaged_injections) == dynamic_information.n_inj_failures
    assert (
        len(outaged_injections)
        == dynamic_information.nonrel_injection_outage_deltap.shape[1]
        + dynamic_information.relevant_injection_outage_idx.shape[0]
    )
    assert (
        len(outaged_injections)
        == dynamic_information.nonrel_injection_outage_node.shape[0]
        + dynamic_information.relevant_injection_outage_idx.shape[0]
    )

    injections = net.get_injections()
    all_delta_p = get_all_injection_outage_deltap(
        injection_outage_deltap=dynamic_information.nonrel_injection_outage_deltap,
        relevant_injections=dynamic_information.relevant_injections,
        relevant_injection_outage_idx=dynamic_information.relevant_injection_outage_idx,
        relevant_injection_outage_sub=dynamic_information.relevant_injection_outage_sub,
    )
    for i, inj in enumerate(outaged_injections):
        assert inj.elements[0].id in injections.index
        assert np.isclose(
            injections.loc[inj.elements[0].id, "p"],
            -all_delta_p[0, i],
        )


def test_loadflows_match(preprocessed_powsybl_data_folder: Path) -> None:
    net = pypowsybl.network.load(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    network_data = load_network_data(preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["network_data_file_path"])
    static_information = load_static_information(
        preprocessed_powsybl_data_folder / PREPROCESSING_PATHS["static_information_file_path"]
    )

    (n_0, n_1), success = run_solver_symmetric(
        default_topology(static_information.solver_config),
        None,
        None,
        static_information.dynamic_information,
        static_information.solver_config,
        lambda lf_res: (lf_res.n_0_matrix, lf_res.n_1_matrix),
    )
    n_0 = np.abs(n_0[0, 0])
    n_1 = np.abs(n_1[0, 0])
    assert np.all(success)

    runner = PowsyblRunner()
    runner.replace_grid(net)
    runner.store_action_set(extract_action_set(network_data))
    nminus1_def = extract_nminus1_definition(network_data)
    runner.store_nminus1_definition(nminus1_def)

    res_ref = runner.run_dc_loadflow([], [])
    n_0_ref, n_1_ref, success_ref = extract_solver_matrices_polars(
        loadflow_results=res_ref,
        nminus1_definition=runner.nminus1_definition,
        timestep=0,
    )

    assert np.all(success)
    n_0_ref = np.abs(n_0_ref)
    n_1_ref = np.abs(n_1_ref)

    assert n_0.shape == n_0_ref.shape
    assert np.allclose(n_0, n_0_ref)
    assert n_1.shape == n_1_ref.shape
    assert np.allclose(n_1, n_1_ref)


def test_globally_unique_ids(powsybl_data_folder: Path) -> None:
    filesystem_dir_powsybl = DirFileSystem(str(powsybl_data_folder))
    backend = PowsyblBackend(filesystem_dir_powsybl)

    assert len(backend.get_branch_ids()) == len(set(backend.get_branch_ids()))
    assert len(backend.get_node_ids()) == len(set(backend.get_node_ids()))
    assert len(backend.get_injection_ids()) == len(set(backend.get_injection_ids()))
    assert len(backend.get_multi_outage_ids()) == len(set(backend.get_multi_outage_ids()))


def test_psts(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_dir = tmp_path_factory.mktemp("psts")
    case30_with_psts_powsybl(tmp_dir)
    filesystem_dir_powsybl = DirFileSystem(str(tmp_dir))
    backend = PowsyblBackend(filesystem_dir_powsybl)

    assert backend.get_controllable_phase_shift_mask().sum() == 2
    assert len(backend.get_phase_shift_taps()) == 2
    for taps in backend.get_phase_shift_taps():
        assert len(taps)
        assert taps[0] == taps.min()
        assert taps[-1] == taps.max()
