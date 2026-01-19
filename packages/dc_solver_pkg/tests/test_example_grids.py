# Copyright 2025 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandapower as pp
import pypowsybl
import pytest
from fsspec.implementations.dirfs import DirFileSystem
from tests.numpy_reference import calc_bsdf as calc_bsdf_numpy
from toop_engine_dc_solver.example_grids import (
    PandapowerCounters,
    case14_pandapower,
    case30_with_psts,
    case30_with_psts_powsybl,
    case57_data_pandapower,
    case57_data_powsybl,
    case57_non_converging,
    case300_pandapower,
    case300_powsybl,
    case9241_pandapower,
    case9241_powsybl,
    create_complex_grid_battery_hvdc_svc_3w_trafo_data_folder,
    create_ucte_data_folder,
    node_breaker_folder_powsybl,
    oberrhein_data,
    random_topology_info_backend,
)
from toop_engine_dc_solver.jax.bsdf import _apply_bus_split, calc_bsdf, init_bsdf_results
from toop_engine_dc_solver.jax.inputs import validate_static_information
from toop_engine_dc_solver.jax.lodf import calc_lodf
from toop_engine_dc_solver.preprocess.convert_to_jax import convert_to_jax, load_grid
from toop_engine_dc_solver.preprocess.pandapower.pandapower_backend import PandaPowerBackend
from toop_engine_dc_solver.preprocess.powsybl.powsybl_backend import PowsyblBackend
from toop_engine_dc_solver.preprocess.preprocess import preprocess
from toop_engine_grid_helpers.pandapower.example_grids import example_multivoltage_cross_coupler
from toop_engine_grid_helpers.powsybl.example_grids import (
    basic_node_breaker_network_powsybl,
    case14_matching_asset_topo_powsybl,
)
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import assert_station_in_network
from toop_engine_interfaces.asset_topology import Topology
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters


def test_random_topology_info(data_folder: Path) -> None:
    filesystem_dir = DirFileSystem(str(data_folder))
    backend = PandaPowerBackend(filesystem_dir)
    pp_counters = PandapowerCounters(
        highest_bus_id=int(backend.net.bus.index.max()),
        highest_switch_id=int(backend.net.switch.index.max()),
    )
    topology = random_topology_info_backend(backend, pp_counters)

    assert len(topology.stations) == sum(backend.get_relevant_node_mask())


def test_oberrhein_data() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        oberrhein_data(tmp)
        filesystem_dir = DirFileSystem(str(tmp))
        pp_backend = PandaPowerBackend(filesystem_dir)
        network_data = preprocess(pp_backend)
        assert len(network_data.branches_at_nodes) > 0

        assert (tmp / "initial_topology" / "asset_topology.json").exists()


def test_case57_match():
    with tempfile.TemporaryDirectory() as pp_folder:
        with tempfile.TemporaryDirectory() as powsybl_folder:
            case57_data_powsybl(Path(powsybl_folder))
            case57_data_pandapower(Path(pp_folder))

            filesystem_dir_pp = DirFileSystem(str(pp_folder))
            filesystem_dir_powsybl = DirFileSystem(str(powsybl_folder))
            pp_backend = PandaPowerBackend(filesystem_dir_pp)
            powsybl_backend = PowsyblBackend(filesystem_dir_powsybl, distributed_slack=False)

            assert np.allclose(
                pp_backend.net.res_bus["va_degree"].values,
                powsybl_backend.net.get_buses()["v_angle"].values,
            )


def test_case57_backends_match():
    with tempfile.TemporaryDirectory() as pp_folder:
        with tempfile.TemporaryDirectory() as powsybl_folder:
            case57_data_powsybl(Path(powsybl_folder))
            case57_data_pandapower(Path(pp_folder))

            filesystem_dir_pp = DirFileSystem(str(pp_folder))
            filesystem_dir_powsybl = DirFileSystem(str(powsybl_folder))
            pp_backend = PandaPowerBackend(filesystem_dir_pp)
            powsybl_backend = PowsyblBackend(filesystem_dir_powsybl, distributed_slack=False)

            assert len(pp_backend.get_susceptances()) == len(powsybl_backend.get_susceptances())
            assert np.allclose(pp_backend.get_susceptances(), powsybl_backend.get_susceptances())
            assert np.array_equal(pp_backend.get_from_nodes(), powsybl_backend.get_from_nodes())
            assert np.array_equal(pp_backend.get_to_nodes(), powsybl_backend.get_to_nodes())

            assert len(pp_backend.get_mw_injections()) == len(powsybl_backend.get_mw_injections())
            # The slack injection can be different, filter that out
            rel_inj_pp = pp_backend.get_mw_injections()[0, pp_backend.get_injection_nodes() != pp_backend.get_slack()]
            rel_inj_powsybl = powsybl_backend.get_mw_injections()[
                0, powsybl_backend.get_injection_nodes() != powsybl_backend.get_slack()
            ]
            assert np.isclose(
                np.sum(rel_inj_pp),
                np.sum(rel_inj_powsybl),
            )
            assert len(pp_backend.get_shift_angles()) == len(powsybl_backend.get_shift_angles())
            assert np.isclose(
                np.sum(pp_backend.get_shift_angles()),
                np.sum(powsybl_backend.get_shift_angles()),
            )

            pp_network_data = preprocess(pp_backend)
            pp_static_information = convert_to_jax(pp_network_data)
            powsybl_network_data = preprocess(powsybl_backend)
            powsybl_static_information = convert_to_jax(powsybl_network_data)

            ###
            # Check a base loadflow
            lf_pp = pp_network_data.ptdf @ pp_network_data.nodal_injection[0]
            lf_powsybl = powsybl_network_data.ptdf @ powsybl_network_data.nodal_injection[0]
            assert np.isclose(np.sum(np.abs(lf_pp)), np.sum(np.abs(lf_powsybl)))

            ###
            # Check LODFs
            # Outage the PST in pandapower
            pst_idx = pp_backend.get_branch_names().index("PST")

            lodf, success = calc_lodf(
                branch_to_outage=jnp.array(pst_idx),
                ptdf=pp_static_information.dynamic_information.ptdf,
                from_node=pp_static_information.dynamic_information.from_node,
                to_node=pp_static_information.dynamic_information.to_node,
                branches_monitored=pp_static_information.dynamic_information.branches_monitored,
            )
            assert success.item()

            diff_flow = lodf * lf_pp[pst_idx]
            n_1_pp = np.abs(lf_pp + diff_flow)

            # Outage the PST in powsybl
            pst_idx = powsybl_backend.get_branch_ids().index("PST")

            lodf, success = calc_lodf(
                branch_to_outage=jnp.array(pst_idx),
                ptdf=powsybl_static_information.dynamic_information.ptdf,
                from_node=powsybl_static_information.dynamic_information.from_node,
                to_node=powsybl_static_information.dynamic_information.to_node,
                branches_monitored=powsybl_static_information.dynamic_information.branches_monitored,
            )
            assert success.item()

            diff_flow = lodf * lf_powsybl[pst_idx]
            n_1_powsybl = np.abs(lf_powsybl + diff_flow)

            assert np.isclose(np.sum(np.abs(n_1_pp)), np.sum(np.abs(n_1_powsybl)))

            ###
            # Check BSDFs
            substation_topology = jnp.array([True, True, False, False, False, False], dtype=bool)
            sub_id = jnp.array(0, dtype=int)

            pp_bsdf_res = _apply_bus_split(
                current_results=init_bsdf_results(
                    ptdf=pp_static_information.dynamic_information.ptdf,
                    from_node=pp_static_information.dynamic_information.from_node,
                    to_node=pp_static_information.dynamic_information.to_node,
                    n_splits=1,
                ),
                substation_configuration=substation_topology,
                substation_id=sub_id,
                split_idx=0,
                tot_stat=pp_static_information.dynamic_information.tot_stat,
                from_stat_bool=pp_static_information.dynamic_information.from_stat_bool,
                susceptance=pp_static_information.dynamic_information.susceptance,
                rel_stat_map=pp_static_information.solver_config.rel_stat_map,
                slack=pp_static_information.solver_config.slack,
                n_stat=pp_static_information.solver_config.n_stat,
            )

            assert jnp.all(pp_bsdf_res.success)
            n_0_split_pp = pp_bsdf_res.ptdf @ pp_static_information.dynamic_information.nodal_injections[0]
            n_0_split_pp = np.abs(n_0_split_pp)[pp_static_information.dynamic_information.branches_monitored]

            powsybl_bsdf_res = _apply_bus_split(
                current_results=init_bsdf_results(
                    ptdf=powsybl_static_information.dynamic_information.ptdf,
                    from_node=powsybl_static_information.dynamic_information.from_node,
                    to_node=powsybl_static_information.dynamic_information.to_node,
                    n_splits=1,
                ),
                substation_configuration=substation_topology,
                substation_id=sub_id,
                split_idx=0,
                tot_stat=powsybl_static_information.dynamic_information.tot_stat,
                from_stat_bool=powsybl_static_information.dynamic_information.from_stat_bool,
                susceptance=powsybl_static_information.dynamic_information.susceptance,
                rel_stat_map=powsybl_static_information.solver_config.rel_stat_map,
                slack=powsybl_static_information.solver_config.slack,
                n_stat=powsybl_static_information.solver_config.n_stat,
            )

            assert jnp.all(powsybl_bsdf_res.success)
            n_0_split_powsybl = powsybl_bsdf_res.ptdf @ powsybl_static_information.dynamic_information.nodal_injections[0]
            n_0_split_powsybl = np.abs(n_0_split_powsybl)[powsybl_static_information.dynamic_information.branches_monitored]

            assert np.isclose(np.sum(n_0_split_pp), np.sum(n_0_split_powsybl))

            ###
            # Compare the inner BSDF with the numpy reference
            bsdf, ptdf_th_sw, succ = calc_bsdf(
                substation_topology=substation_topology,
                ptdf=pp_static_information.dynamic_information.ptdf,
                i_stat=pp_static_information.solver_config.rel_stat_map.val[sub_id],
                i_stat_rel=sub_id,
                tot_stat=pp_static_information.dynamic_information.tot_stat[sub_id],
                from_stat_bool=pp_static_information.dynamic_information.from_stat_bool[sub_id],
                to_node=pp_static_information.dynamic_information.to_node,
                from_node=pp_static_information.dynamic_information.from_node,
                susceptance=pp_static_information.dynamic_information.susceptance,
                slack=pp_static_information.solver_config.slack,
                n_stat=pp_static_information.solver_config.n_stat,
            )
            assert np.all(succ)

            bsdf_ref, ptdf_ref, from_node_ref, to_node_ref = calc_bsdf_numpy(
                switched_node=int(sub_id),
                assignment=np.array(substation_topology[: len(pp_network_data.branches_at_nodes[sub_id])]),
                is_slack=bool(pp_network_data.slack == pp_network_data.relevant_nodes[sub_id]),
                bus_a_ptdf=int(pp_network_data.relevant_nodes[sub_id]),
                bus_b_ptdf=int(pp_network_data.n_original_nodes + sub_id),
                ptdf=pp_network_data.ptdf,
                susceptance=pp_network_data.susceptances,
                from_node=pp_network_data.from_nodes,
                to_node=pp_network_data.to_nodes,
                branches_at_nodes=pp_network_data.branches_at_nodes,
                branch_direction=pp_network_data.branch_direction,
            )
            assert np.allclose(bsdf, bsdf_ref)


def test_case57_non_converging():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        case57_non_converging(tmp)
        filesystem_dir_pp = DirFileSystem(str(tmp))
        pp_backend = PandaPowerBackend(filesystem_dir_pp)

        assert np.all(pp_backend.get_ac_dc_mismatch() == 0)


def test_case300() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        case300_pandapower(tmp)

        filesystem_dir_powsybl = DirFileSystem(str(tmp))
        pp_backend = PandaPowerBackend(filesystem_dir_powsybl)

        assert len(pp_backend.net.bus) == 300

        network_data = preprocess(pp_backend)
        static_information = convert_to_jax(network_data)

        assert static_information.n_sub_relevant > 0


def test_case300_powsybl() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        case300_powsybl(tmp)

        filesystem_dir_powsybl = DirFileSystem(str(tmp))
        powsybl_backend = PowsyblBackend(filesystem_dir_powsybl)

        assert len(powsybl_backend.net.get_buses()) == 300

        network_data = preprocess(powsybl_backend)
        static_information = convert_to_jax(network_data)

        assert static_information.n_sub_relevant > 0


@pytest.mark.xdist_group("performance")
@pytest.mark.timeout(300)
def test_case9241_pp() -> None:
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        case9241_pandapower(folder)
        grid_file_path = folder / PREPROCESSING_PATHS["grid_file_path_pandapower"]
        net = pp.from_json(grid_file_path)

        masks_folder = folder / PREPROCESSING_PATHS["masks_path"]

        line_for_reward_0 = np.load(masks_folder / "line_for_reward_0.npy")
        line_for_reward_1 = np.load(masks_folder / "line_for_reward_1.npy")
        line_for_reward_2 = np.load(masks_folder / "line_for_reward_2.npy")
        line_for_reward_3 = np.load(masks_folder / "line_for_reward_3.npy")
        assert line_for_reward_0.sum() < len(net.line)
        assert line_for_reward_1.sum() < len(net.line)
        assert line_for_reward_2.sum() < len(net.line)
        assert line_for_reward_3.sum() < len(net.line)
        total_reward = line_for_reward_0.sum() + line_for_reward_1.sum() + line_for_reward_2.sum() + line_for_reward_3.sum()
        assert total_reward >= len(net.line)

        trafo_for_reward_0 = np.load(masks_folder / "trafo_for_reward_0.npy")
        trafo_for_reward_1 = np.load(masks_folder / "trafo_for_reward_1.npy")
        trafo_for_reward_2 = np.load(masks_folder / "trafo_for_reward_2.npy")
        trafo_for_reward_3 = np.load(masks_folder / "trafo_for_reward_3.npy")
        assert trafo_for_reward_0.sum() < len(net.trafo)
        assert trafo_for_reward_1.sum() < len(net.trafo)
        assert trafo_for_reward_2.sum() < len(net.trafo)
        assert trafo_for_reward_3.sum() < len(net.trafo)
        total_reward = (
            trafo_for_reward_0.sum() + trafo_for_reward_1.sum() + trafo_for_reward_2.sum() + trafo_for_reward_3.sum()
        )
        assert total_reward >= len(net.trafo)

        relevant_subs_0 = np.load(masks_folder / "relevant_subs_0.npy")
        relevant_subs_1 = np.load(masks_folder / "relevant_subs_1.npy")
        relevant_subs_2 = np.load(masks_folder / "relevant_subs_2.npy")
        relevant_subs_3 = np.load(masks_folder / "relevant_subs_3.npy")

        assert relevant_subs_0.sum() < len(net.bus)
        assert relevant_subs_1.sum() < len(net.bus)
        assert relevant_subs_2.sum() < len(net.bus)
        assert relevant_subs_3.sum() < len(net.bus)

        relevant_subs = np.load(masks_folder / "relevant_subs.npy")

        assert relevant_subs.sum() < len(net.bus)
        assert (
            relevant_subs.sum()
            == relevant_subs_0.sum() + relevant_subs_1.sum() + relevant_subs_2.sum() + relevant_subs_3.sum()
        )
        assert relevant_subs.sum() == 400

        filesystem_dir_pp = DirFileSystem(str(folder))
        backend = PandaPowerBackend(filesystem_dir_pp)
        assert sum(backend.get_controllable_phase_shift_mask())


@pytest.mark.skip(reason="This test takes too long to run")
def test_case9241_pp_load_grid() -> None:
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        case9241_pandapower(folder)
        filesystem_dir = DirFileSystem(str(folder))
        stats, static_information, network_data = load_grid(
            filesystem_dir,
            chronics_id=0,
            timesteps=slice(0, 1),
            pandapower=True,
            parameters=PreprocessParameters(
                action_set_clip=2**3,
                action_set_filter_bridge_lookup=False,
                action_set_filter_bsdf_lodf=False,
            ),
        )
        assert stats.overload_energy_n0 > 0
        assert stats.overload_energy_n1 > 0
        assert static_information.n_sub_relevant <= 400
        assert static_information.n_sub_relevant > 300
        assert static_information.dynamic_information.n_controllable_pst > 0
        validate_static_information(static_information)


@pytest.mark.xdist_group("performance")
@pytest.mark.timeout(300)
def test_case9241_powsybl() -> None:
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        case9241_powsybl(folder)
        filesystem_dir = DirFileSystem(str(folder))
        backend = PowsyblBackend(filesystem_dir)
        assert len(backend.net.get_buses()) == 9241
        assert sum(backend.get_relevant_node_mask()) == 400
        assert np.isfinite(backend.get_susceptances()).all()
        assert all(np.abs(backend.get_susceptances()) > 0.005)
        assert len(backend.net.get_buses()["synchronous_component"].value_counts()) == 1


@pytest.mark.skip(reason="This test takes too long to run")
def test_case9241_powsybl_load_grid() -> None:
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        case9241_powsybl(folder)
        filesystem_dir = DirFileSystem(str(folder))
        clip_to_n_actions = 2**3
        stats, static_information, network_data = load_grid(
            filesystem_dir,
            chronics_id=0,
            timesteps=slice(0, 1),
            pandapower=False,
            parameters=PreprocessParameters(
                action_set_clip=clip_to_n_actions,
                action_set_filter_bridge_lookup=False,
                action_set_filter_bsdf_lodf=False,
            ),
        )
        assert stats.overload_energy_n0 > 0
        assert stats.overload_energy_n1 > 0
        assert static_information.n_sub_relevant <= 400
        assert static_information.n_sub_relevant > 300
        validate_static_information(static_information)


def test_example_multivoltage_cross_coupler() -> None:
    net = example_multivoltage_cross_coupler()
    net_org = pp.networks.example_multivoltage()
    pp.runpp(net_org)
    pp.runpp(net)
    # Check if the results are the same
    # modifications should not change the results
    for id_org in net_org.res_bus.index:
        for col in net_org.res_bus:
            assert net.res_bus.loc[id_org, col] == net_org.res_bus.loc[id_org, col]
    for id_org in net_org.res_line.index:
        for col in net_org.res_line:
            assert net.res_line.loc[id_org, col] == net_org.res_line.loc[id_org, col]


def test_case14_pandapower() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        case14_pandapower(tmp_dir)

        filesystem_dir = DirFileSystem(str(tmp_dir))
        pp_backend = PandaPowerBackend(filesystem_dir)
        assert sum(pp_backend.get_relevant_node_mask()) == 5
        assert len(pp_backend.get_relevant_node_mask()) == 14
        assert sum(pp_backend.get_monitored_branch_mask()) == 20
        assert sum(pp_backend.get_outaged_branch_mask()) == 19


def test_case30_with_psts() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        case30_with_psts(tmp_dir)

        filesystem_dir = DirFileSystem(str(tmp_dir))
        pp_backend = PandaPowerBackend(filesystem_dir)
        assert pp_backend.get_phase_shift_mask().sum() == 4
        assert pp_backend.get_controllable_phase_shift_mask().sum() == 3


def test_case30_with_psts_powsybl() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        case30_with_psts_powsybl(tmp_dir)

        filesystem_dir = DirFileSystem(str(tmp_dir))
        powsybl_backend = PowsyblBackend(filesystem_dir)
        assert powsybl_backend.get_phase_shift_mask().sum() == 2
        assert powsybl_backend.get_controllable_phase_shift_mask().sum() == 2


def test_case14_with_matching_asset_topo() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        case14_matching_asset_topo_powsybl(tmp_dir)

        filesystem_dir = DirFileSystem(str(tmp_dir))
        backend = PowsyblBackend(filesystem_dir)
        preprocess(backend, parameters=PreprocessParameters())

        # Check the asset topology
        with open(tmp_dir / PREPROCESSING_PATHS["asset_topology_file_path"], "r") as f:
            asset_topo = Topology.model_validate_json(f.read())
        for station in asset_topo.stations:
            assert_station_in_network(backend.net, station)


def test_basic_node_breaker_network_powsybl() -> None:
    net = basic_node_breaker_network_powsybl()
    pypowsybl.loadflow.run_dc(net)


def test_node_breaker_folder_powsybl() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        node_breaker_folder_powsybl(tmp_dir)
        filesystem_dir = DirFileSystem(str(tmp_dir))
        backend = PowsyblBackend(filesystem_dir)
        assert sum(backend.get_relevant_node_mask())
        network_data = preprocess(backend)
        assert sum(network_data.relevant_node_mask) > 0
        assert len(network_data.branch_action_set)


def test_create_complex_grid_battery_hvdc_svc_3w_trafo_data_folder() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        create_complex_grid_battery_hvdc_svc_3w_trafo_data_folder(tmp_dir)
        filesystem_dir = DirFileSystem(str(tmp_dir))
        backend = PowsyblBackend(filesystem_dir)
        assert sum(backend.get_relevant_node_mask())
        network_data = preprocess(backend)
        assert sum(network_data.relevant_node_mask) > 0
        assert len(network_data.branch_action_set)


def test_create_ucte_data_folder(ucte_file) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        create_ucte_data_folder(tmp_dir, ucte_file=ucte_file)
        filesystem_dir = DirFileSystem(str(tmp_dir))
        backend = PowsyblBackend(filesystem_dir)
        assert sum(backend.get_relevant_node_mask())
        network_data = preprocess(backend)
        assert sum(network_data.relevant_node_mask) > 0
        assert len(network_data.branch_action_set)
