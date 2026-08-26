# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import time
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import pandapower as pp
import pandas as pd
import pypowsybl
import pytest
import structlog
from beartype.typing import Optional
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem
from pypowsybl.network import Network
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_grid_helpers.powsybl.loadflow_parameters import SINGLE_SLACK
from toop_engine_grid_helpers.powsybl.powsybl_helpers import load_lf_params_from_fs
from toop_engine_importer.pandapower_import.preprocessing import modify_constan_z_load
from toop_engine_importer.pypowsybl_import import powsybl_masks, preprocessing
from toop_engine_importer.pypowsybl_import.data_classes import PreProcessingStatistics
from toop_engine_importer.pypowsybl_import.network_analysis import set_tie_line_boundary_equivalents
from toop_engine_importer.pypowsybl_import.network_reduction import reduce_network_based_on_area_settings
from toop_engine_importer.pypowsybl_import.preprocessing import create_nminus1_definition_from_masks
from toop_engine_interfaces.folder_structure import (
    NETWORK_MASK_NAMES,
    PREPROCESSING_PATHS,
)
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    AreaSettings,
    CgmesImporterParameters,
    PreprocessParameters,
    UcteImporterParameters,
)
from toop_engine_interfaces.messages.preprocess.preprocess_heartbeat import (
    PreprocessStage,
)
from toop_engine_interfaces.messages.preprocess.preprocess_results import (
    DynamicInformationStats,
    ImportResult,
)
from toop_engine_interfaces.nminus1_definition import (
    load_nminus1_definition,
    validate_spps_rule_referential_integrity,
)
from toop_engine_interfaces.status_update import NetworkDataStats

logger = structlog.get_logger(__name__)

ASSET_TOPOLOGY_RUNTIME_STATE_PATH = Path("initial_topology/asset_topology_runtime_state.json")
ASSET_TOPOLOGY_COMPACT_RUNTIME_STATE_PATH = Path("initial_topology/asset_topology_compact_runtime_state.json")
ASSET_TOPOLOGY_RUNTIME_PATH = Path("initial_topology/asset_topology_runtime.json")


def test_save_load_preprocessing_statistics():
    importer_parameters = UcteImporterParameters(
        grid_model_file="uct_file",
        data_folder="files_path",
        white_list_file="files_path/CB_White-Liste.csv",
        black_list_file="files_path/CB_Black-Liste.csv",
        area_settings=AreaSettings(
            cutoff_voltage=220,
            control_area=["D8"],
            view_area=["D2", "D4", "D7", "D8"],
            nminus1_area=["D2", "D4", "D7", "D8"],
        ),
    )

    # test 1 - apply white and black list
    import_result = ImportResult(
        data_folder=Path(""),
    )
    statistics = PreProcessingStatistics(
        id_lists={"relevant_subs": 1},
        import_result=import_result,
        border_current={"3": 3},
        network_changes={"4": 4},
        import_parameter=importer_parameters,
    )
    with TemporaryDirectory() as temp_dir:
        json_test_file = Path(temp_dir) / "test_save_statistics.json"
        preprocessing.save_preprocessing_statistics_filesystem(
            statistics, file_path=json_test_file, filesystem=LocalFileSystem()
        )
        loaded_statistics = preprocessing.load_preprocessing_statistics_filesystem(
            json_test_file, filesystem=LocalFileSystem()
        )
        assert isinstance(loaded_statistics, PreProcessingStatistics)
        assert statistics == loaded_statistics
        assert loaded_statistics.id_lists == statistics.id_lists
        assert loaded_statistics.import_result == statistics.import_result
        assert loaded_statistics.border_current == statistics.border_current
        assert loaded_statistics.network_changes == statistics.network_changes
        assert loaded_statistics.import_parameter == statistics.import_parameter
        assert isinstance(loaded_statistics.import_result, ImportResult)
        assert isinstance(loaded_statistics.import_parameter, UcteImporterParameters)


def test_fill_statistics_for_network_masks(ucte_file, ucte_importer_parameters):
    network = pypowsybl.network.load(ucte_file)

    import_result = ImportResult(
        data_folder=Path(""),
    )
    statistics = PreProcessingStatistics(
        id_lists={},
        import_result=import_result,
        border_current={},
        network_changes={},
    )
    masks = powsybl_masks.create_default_network_masks(network=network)
    preprocessing.fill_statistics_for_network_masks(network=network, statistics=statistics, network_masks=masks)

    assert statistics.import_result == ImportResult(
        data_folder=Path(""),
    )
    assert statistics.border_current == {}
    assert statistics.network_changes == {}
    assert statistics.import_parameter is None
    for key, value in statistics.id_lists.items():
        assert value == []
    lf_result, *_ = pypowsybl.loadflow.run_dc(network)

    masks = powsybl_masks.make_masks(
        network=network, slack_id=lf_result.reference_bus_id, importer_parameters=ucte_importer_parameters
    )
    preprocessing.fill_statistics_for_network_masks(network=network, statistics=statistics, network_masks=masks)
    for key, value in statistics.id_lists.items():
        assert len(value) > 0
        assert isinstance(value, list)
        assert isinstance(value[0], str)
        assert statistics.import_result.model_dump()[f"n_{key}"] == len(value)


def test_convert_file(ucte_file):
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # def parameters for function

        def heartbeat_working(
            stage: PreprocessStage,
            message: Optional[str],
            preprocess_id: str,
            start_time: float,
            stats: Optional[NetworkDataStats] = None,
        ):
            logger.info(
                f"Preprocessing stage {stage} for job {preprocess_id} after {(time.time() - start_time):f}s: "
                f"{message}, {stats}"
            )

        start_time = time.time()
        heartbeat_fn = partial(
            heartbeat_working,
            preprocess_id="test_id",
            start_time=start_time,
        )
        importer_parameters = UcteImporterParameters(
            grid_model_file=ucte_file,
            fail_on_non_convergence=False,
            data_folder=temp_dir,
            white_list_file=None,
            black_list_file=None,
            area_settings=AreaSettings(
                cutoff_voltage=220,
                control_area=["D8"],
                view_area=["D2", "D4", "D7", "D8"],
                nminus1_area=["D2", "D4", "D7", "D8"],
            ),
        )

        import_result = preprocessing.convert_file(
            importer_parameters=importer_parameters,
            status_update_fn=heartbeat_fn,
        )
        importer_auxiliary_file = temp_dir / PREPROCESSING_PATHS["importer_auxiliary_file_path"]
        grid_file_path = temp_dir / PREPROCESSING_PATHS["grid_file_path_powsybl"]
        mask_dir = temp_dir / PREPROCESSING_PATHS["masks_path"]
        asset_topology_master_data_file = temp_dir / PREPROCESSING_PATHS["asset_topology_master_data_file_path"]
        asset_topology_runtime_state_file = temp_dir / ASSET_TOPOLOGY_RUNTIME_STATE_PATH
        asset_topology_compact_runtime_state_file = temp_dir / ASSET_TOPOLOGY_COMPACT_RUNTIME_STATE_PATH
        assert importer_auxiliary_file.exists()
        assert grid_file_path.exists()
        assert not (temp_dir / PREPROCESSING_PATHS["asset_topology_file_path"]).exists()
        assert not (temp_dir / ASSET_TOPOLOGY_RUNTIME_PATH).exists()
        assert asset_topology_master_data_file.exists()
        assert not asset_topology_runtime_state_file.exists()
        assert not asset_topology_compact_runtime_state_file.exists()
        for file_name in powsybl_masks.NetworkMasks.__annotations__.keys():
            assert (mask_dir / NETWORK_MASK_NAMES[file_name]).exists(), f"{NETWORK_MASK_NAMES[file_name]} does not exist"
        assert isinstance(import_result, ImportResult)

        # test without status_update_fn
        temp_dir_test2 = temp_dir / "test2"
        temp_dir_test2.mkdir(exist_ok=True)
        importer_parameters.data_folder = temp_dir_test2
        import_result = preprocessing.convert_file(
            importer_parameters=importer_parameters,
        )
        importer_auxiliary_file = temp_dir_test2 / PREPROCESSING_PATHS["importer_auxiliary_file_path"]
        grid_file_path = temp_dir_test2 / PREPROCESSING_PATHS["grid_file_path_powsybl"]
        mask_dir = temp_dir_test2 / PREPROCESSING_PATHS["masks_path"]
        asset_topology_master_data_file = temp_dir_test2 / PREPROCESSING_PATHS["asset_topology_master_data_file_path"]
        asset_topology_runtime_state_file = temp_dir_test2 / ASSET_TOPOLOGY_RUNTIME_STATE_PATH
        asset_topology_compact_runtime_state_file = temp_dir_test2 / ASSET_TOPOLOGY_COMPACT_RUNTIME_STATE_PATH
        assert importer_auxiliary_file.exists()
        assert grid_file_path.exists()
        assert not (temp_dir_test2 / PREPROCESSING_PATHS["asset_topology_file_path"]).exists()
        assert not (temp_dir_test2 / ASSET_TOPOLOGY_RUNTIME_PATH).exists()
        assert asset_topology_master_data_file.exists()
        assert not asset_topology_runtime_state_file.exists()
        assert not asset_topology_compact_runtime_state_file.exists()
        for file_name in powsybl_masks.NetworkMasks.__annotations__.keys():
            assert (mask_dir / NETWORK_MASK_NAMES[file_name]).exists(), f"{NETWORK_MASK_NAMES[file_name]} does not exist"
        assert isinstance(import_result, ImportResult)


@pytest.mark.parametrize("reduction_range", [1, 2, 5, 100])
def test_reduce_network_to_view_area_preserves_dc_branch_flows(
    reduction_range: int, complex_grid_network: Network, cgmes_importer_parameters: CgmesImporterParameters
) -> None:
    net = complex_grid_network
    importer_parameters = cgmes_importer_parameters.model_copy(
        update={"network_reduction_voltage_level_range": reduction_range}
    )
    pypowsybl.loadflow.run_dc(net)
    original_branches = net.get_lines()
    reduce_network_based_on_area_settings(net=net, importer_parameters=importer_parameters)

    pypowsybl.loadflow.run_dc(net)
    reduced_branches = net.get_lines()

    common_branch_ids = original_branches.index.intersection(reduced_branches.index)
    assert len(common_branch_ids) > 0
    for column in ["p1", "p2"]:
        pd.testing.assert_series_equal(
            original_branches.loc[common_branch_ids, column],
            reduced_branches.loc[common_branch_ids, column],
            check_names=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-9,
        )


@pytest.mark.parametrize("run_ac", [True, False])
def test_reduce_network_to_view_area_preserves_dc_branch_flows_tie_lines_edge_case(
    complex_grid_network: Network, cgmes_importer_parameters: CgmesImporterParameters, run_ac: bool
) -> None:
    """Tests Tieline reduction.

    It can happen, that a Tieline is exactly on the border of the view area.
    If the boundary node has a different setting for p0 and q0, the resulting load flow will be different after the reduction.
    This test checks that the reduction does not change the load flow of the remaining branches, even in this edge case.
    """
    loadflow_parameters = deepcopy(SINGLE_SLACK)
    loadflow_parameters.provider_parameters["newtonRaphsonConvEpsPerEq"] = "1e-9"
    loadflow_parameters.provider_parameters["maxNewtonRaphsonIterations"] = "20"
    if run_ac:
        columns = ["p1", "p2", "q1", "q2"]
        atol = 1e-7
    else:
        columns = ["p1", "p2"]
        atol = 1e-9

    reduction_range = 0
    net = complex_grid_network
    importer_parameters = cgmes_importer_parameters.model_copy(
        update={"network_reduction_voltage_level_range": reduction_range}
    )

    # VL_NL_TIE_REDUCTION_REMOTE_380 is a tie line that is exactly on the border of the view area.
    # it is set to p0=0.0 and q0=0.0, but the tie line has a flow due to load_NL_tie_reduction_remote
    importer_parameters.area_settings = AreaSettings(
        cutoff_voltage=220,
        control_area=["BE"],
        view_area=[
            "BE",
        ],
        nminus1_area=[
            "BE",
        ],
        dso_trafo_factors=None,
        border_line_factors=None,
    )
    if run_ac:
        pypowsybl.loadflow.run_ac(net, loadflow_parameters)
    else:
        pypowsybl.loadflow.run_dc(net, loadflow_parameters)
    net_success = deepcopy(net)
    original_branches = net.get_lines()
    reduce_network_based_on_area_settings(net=net, importer_parameters=importer_parameters)

    if run_ac:
        pypowsybl.loadflow.run_ac(net, loadflow_parameters)
    else:
        pypowsybl.loadflow.run_dc(net, loadflow_parameters)
    reduced_branches = net.get_lines()

    common_branch_ids = original_branches.index.intersection(reduced_branches.index)
    assert len(common_branch_ids) > 0
    for column in columns:
        with pytest.raises(AssertionError):
            pd.testing.assert_series_equal(
                original_branches.loc[common_branch_ids, column],
                reduced_branches.loc[common_branch_ids, column],
                check_names=False,
                check_exact=False,
                rtol=0.0,
                atol=atol,
            )

    # Set boundary lines to their solved tie-line flow before reducing the successful variant.
    set_tie_line_boundary_equivalents(net=net_success)
    reduce_network_based_on_area_settings(net=net_success, importer_parameters=importer_parameters)

    if run_ac:
        pypowsybl.loadflow.run_ac(net_success, loadflow_parameters)
    else:
        pypowsybl.loadflow.run_dc(net_success, loadflow_parameters)
    reduced_branches = net_success.get_lines()

    common_branch_ids = original_branches.index.intersection(reduced_branches.index)
    assert len(common_branch_ids) > 0
    for column in columns:
        pd.testing.assert_series_equal(
            original_branches.loc[common_branch_ids, column],
            reduced_branches.loc[common_branch_ids, column],
            check_names=False,
            check_exact=False,
            rtol=0.0,
            atol=atol,
        )


def test_convert_file_complex_grid_with_network_reduction(
    complex_grid_network: Network, cgmes_importer_parameters: CgmesImporterParameters, tmp_path: Path
) -> None:
    input_grid_path = tmp_path / "complex_grid.xiidm"
    complex_grid_network.save(input_grid_path)
    importer_parameters = cgmes_importer_parameters.model_copy(
        update={
            "grid_model_file": input_grid_path,
            "data_folder": tmp_path / "processed",
            "fail_on_non_convergence": False,
            "network_reduction_voltage_level_range": 1,
        }
    )

    import_result = preprocessing.convert_file(importer_parameters=importer_parameters)

    assert isinstance(import_result, ImportResult)
    assert (import_result.data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"]).exists()
    assert (import_result.data_folder / PREPROCESSING_PATHS["nminus1_definition_file_path"]).exists()


def test_convert_file_complex_contingencies_persists_grouped_definition(
    complex_grid_network: Network, cgmes_importer_parameters: CgmesImporterParameters, tmp_path: Path
) -> None:
    input_grid_path = tmp_path / "complex_grid.xiidm"
    complex_grid_network.save(input_grid_path)
    contingency_file = Path(__file__).parents[4] / "data/complex_grid/contingency_list_complex.json"
    importer_parameters = cgmes_importer_parameters.model_copy(
        update={
            "grid_model_file": input_grid_path,
            "data_folder": tmp_path / "processed",
            "contingency_list_file": contingency_file,
            "schema_format": "ContingencyImportSchemaComplex",
            "fail_on_non_convergence": False,
        }
    )

    import_result = preprocessing.convert_file(importer_parameters=importer_parameters)

    definition = load_nminus1_definition(import_result.data_folder / PREPROCESSING_PATHS["nminus1_definition_file_path"])
    assert [contingency.id for contingency in definition.contingencies] == [
        "BASECASE",
        "C_L_DE_BE_1",
        "C_L_NL_1_2",
        "C_L8_WITH_LINE_OUT_OF_SERVICE",
        "C_3W",
        "C_NL_3W_1",
        "C_HVDC_LCC",
        "C_MV_COUPLER",
    ]
    l8_contingency = next(
        contingency for contingency in definition.contingencies if contingency.id == "C_L8_WITH_LINE_OUT_OF_SERVICE"
    )
    assert [element.id for element in l8_contingency.elements] == ["L8", "L81_BREAKER", "L82_BREAKER"]
    three_winding_contingency = next(contingency for contingency in definition.contingencies if contingency.id == "C_3W")
    assert [element.id for element in three_winding_contingency.elements[:3]] == [
        "3W-Leg1",
        "3W-Leg2",
        "3W-Leg3",
    ]
    assert definition.spps_rules is not None
    assert [rule.scheme_name for rule in definition.spps_rules] == [
        "C_L_DE_BE_1",
        "C_L8_WITH_LINE_OUT_OF_SERVICE",
        "C_3W",
    ]
    validate_spps_rule_referential_integrity(definition)
    assert [condition.condition_element_unique_id for condition in definition.spps_rules[1].conditions] == [
        "L8",
        "L81_BREAKER",
        "L82_BREAKER",
    ]
    assert [action.measure_element_unique_id for action in definition.spps_rules[1].actions] == [
        "LINE_out_of_service_BREAKER1",
        "LINE_out_of_service_BREAKER2",
    ]
    assert definition.source_schema == "complex"


def test_convert_file_reduced_network_preserves_ac_branch_flows(
    complex_grid_network: Network, cgmes_importer_parameters: CgmesImporterParameters, tmp_path: Path
) -> None:
    """Tests that a reduced network is has the same loadflow with tie line reductions"""
    complex_grid_network_org = deepcopy(complex_grid_network)
    input_grid_path = tmp_path / "complex_grid.xiidm"
    complex_grid_network_org.save(input_grid_path)
    importer_parameters = cgmes_importer_parameters.model_copy(
        update={
            "grid_model_file": input_grid_path,
            "data_folder": tmp_path / "processed",
            "fail_on_non_convergence": False,
            "network_reduction_voltage_level_range": 0,
            "area_settings": AreaSettings(
                cutoff_voltage=220,
                control_area=["BE"],
                view_area=["BE"],
                nminus1_area=["BE"],
                dso_trafo_factors=None,
                border_line_factors=None,
            ),
        }
    )

    import_result = preprocessing.convert_file(importer_parameters=importer_parameters)
    loadflow_parameters = load_lf_params_from_fs(
        filesystem=LocalFileSystem(),
        file_path=import_result.data_folder / PREPROCESSING_PATHS["loadflow_parameters_file_path"],
    )
    pypowsybl.loadflow.run_ac(complex_grid_network_org, loadflow_parameters)
    original_branches = complex_grid_network_org.get_lines()

    converted_network = pypowsybl.network.load(import_result.data_folder / PREPROCESSING_PATHS["grid_file_path_powsybl"])
    pypowsybl.loadflow.run_ac(converted_network, loadflow_parameters)
    converted_branches = converted_network.get_lines()

    # check that the reduction happened
    voltage_levels_org = complex_grid_network_org.get_voltage_levels()
    voltage_levels_converted = converted_network.get_voltage_levels()
    assert len(voltage_levels_converted) < len(voltage_levels_org)

    common_branch_ids = original_branches.index.intersection(converted_branches.index)
    assert len(common_branch_ids) > 0
    for column in ["p1", "p2", "q1", "q2"]:
        # Note: the other tests use 1e-7
        # because of loadflow_parameters.provider_parameters["newtonRaphsonConvEpsPerEq"] = "1e-9"
        # this loadflow runs with the standard settings of newtonRaphsonConvEpsPerEq = "1e-6"
        # hence the reduced accuracy of the loadflow results here
        pd.testing.assert_series_equal(
            original_branches.loc[common_branch_ids, column],
            converted_branches.loc[common_branch_ids, column],
            check_names=False,
            check_exact=False,
            rtol=0.0,
            atol=1e-5,
        )


def test_convert_file_node_breaker_with_svc(basic_node_breaker_network_powsybl_grid: Network):
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        temp_grid_file = temp_dir / "node_breaker_network.xiidm"
        # add SVC to network
        svc = pd.DataFrame.from_records(
            data=[
                {
                    "id": "SVC",
                    "name": "SVC",
                    "b_max": 0.01,
                    "b_min": -0.01,
                    "regulation_mode": "VOLTAGE",
                    "regulating": True,
                    "target_v": 220.0,
                    "target_q": 0.0,
                    "bus_or_busbar_section_id": "BBS5_1",
                    "position_order": 10,
                }
            ]
        ).set_index("id")

        pypowsybl.network.create_static_var_compensator_bay(basic_node_breaker_network_powsybl_grid, df=svc)
        basic_node_breaker_network_powsybl_grid.save(temp_grid_file)
        # def parameters for function

        def heartbeat_working(
            stage: PreprocessStage,
            message: Optional[str],
            preprocess_id: str,
            start_time: float,
            stats: Optional[NetworkDataStats] = None,
        ):
            logger.info(
                f"Preprocessing stage {stage} for job {preprocess_id} after {(time.time() - start_time):f}s: "
                f"{message}, {stats}"
            )

        start_time = time.time()
        heartbeat_fn = partial(
            heartbeat_working,
            preprocess_id="test_id",
            start_time=start_time,
        )
        importer_parameters = CgmesImporterParameters(
            grid_model_file=temp_grid_file,
            data_folder=temp_dir,
            white_list_file=None,
            black_list_file=None,
            area_settings=AreaSettings(
                cutoff_voltage=110,
                control_area=[""],
                view_area=[""],
                nminus1_area=[""],
            ),
        )

        import_result = preprocessing.convert_file(
            importer_parameters=importer_parameters,
            status_update_fn=heartbeat_fn,
        )
        importer_auxiliary_file = temp_dir / PREPROCESSING_PATHS["importer_auxiliary_file_path"]
        grid_file_path = temp_dir / PREPROCESSING_PATHS["grid_file_path_powsybl"]
        mask_dir = temp_dir / PREPROCESSING_PATHS["masks_path"]
        asset_topology_master_data_file = temp_dir / PREPROCESSING_PATHS["asset_topology_master_data_file_path"]
        asset_topology_runtime_state_file = temp_dir / ASSET_TOPOLOGY_RUNTIME_STATE_PATH
        asset_topology_compact_runtime_state_file = temp_dir / ASSET_TOPOLOGY_COMPACT_RUNTIME_STATE_PATH
        assert importer_auxiliary_file.exists()
        assert grid_file_path.exists()
        assert not (temp_dir / PREPROCESSING_PATHS["asset_topology_file_path"]).exists()
        assert not (temp_dir / ASSET_TOPOLOGY_RUNTIME_PATH).exists()
        assert asset_topology_master_data_file.exists()
        assert not asset_topology_runtime_state_file.exists()
        assert not asset_topology_compact_runtime_state_file.exists()
        for file_name in powsybl_masks.NetworkMasks.__annotations__.keys():
            assert (mask_dir / NETWORK_MASK_NAMES[file_name]).exists(), f"{NETWORK_MASK_NAMES[file_name]} does not exist"
        assert isinstance(import_result, ImportResult)

        # test without status_update_fn
        temp_dir_test2 = temp_dir / "test2"
        temp_dir_test2.mkdir(exist_ok=True)
        importer_parameters.data_folder = temp_dir_test2
        import_result = preprocessing.convert_file(
            importer_parameters=importer_parameters,
        )
        importer_auxiliary_file = temp_dir_test2 / PREPROCESSING_PATHS["importer_auxiliary_file_path"]
        grid_file_path = temp_dir_test2 / PREPROCESSING_PATHS["grid_file_path_powsybl"]
        mask_dir = temp_dir_test2 / PREPROCESSING_PATHS["masks_path"]
        asset_topology_master_data_file = temp_dir_test2 / PREPROCESSING_PATHS["asset_topology_master_data_file_path"]
        asset_topology_runtime_state_file = temp_dir_test2 / ASSET_TOPOLOGY_RUNTIME_STATE_PATH
        asset_topology_compact_runtime_state_file = temp_dir_test2 / ASSET_TOPOLOGY_COMPACT_RUNTIME_STATE_PATH
        assert importer_auxiliary_file.exists()
        assert grid_file_path.exists()
        assert not (temp_dir_test2 / PREPROCESSING_PATHS["asset_topology_file_path"]).exists()
        assert not (temp_dir_test2 / ASSET_TOPOLOGY_RUNTIME_PATH).exists()
        assert asset_topology_master_data_file.exists()
        assert not asset_topology_runtime_state_file.exists()
        assert not asset_topology_compact_runtime_state_file.exists()
        for file_name in powsybl_masks.NetworkMasks.__annotations__.keys():
            assert (mask_dir / NETWORK_MASK_NAMES[file_name]).exists(), f"{NETWORK_MASK_NAMES[file_name]} does not exist"
        assert isinstance(import_result, ImportResult)

        net_loaded = pypowsybl.network.load(grid_file_path)
        assert len(net_loaded.get_static_var_compensators()) == 1

        # make sure the dc solver does not crash with the svc
        filesystem_dir = DirFileSystem(str(import_result.data_folder))
        info, _, _ = load_grid(
            data_folder_dirfs=filesystem_dir,
            pandapower=False,
            parameters=PreprocessParameters(),
            status_update_fn=heartbeat_fn,
        )
        assert isinstance(info, DynamicInformationStats)


def test_modify_constan_z_load():
    # Create a simple pandapower network
    net = pp.create_empty_network()

    # Add some buses
    b1 = pp.create_bus(net, vn_kv=20)
    b2 = pp.create_bus(net, vn_kv=0.4)

    # Add a load with const_z_percent = 100.0
    pp.create_load(net, bus=b1, p_mw=0.1, q_mvar=0.05, const_z_percent=100.0)
    pp.create_load(net, bus=b2, p_mw=0.2, q_mvar=0.1, const_z_percent=50.0)

    # Modify constant z load
    modify_constan_z_load(net, value=0.0)

    # Check if the load with const_z_percent = 100.0 is modified
    assert net.load.loc[net.load["const_z_percent"] == 0.0].shape[0] == 1
    assert net.load.loc[net.load["const_z_percent"] == 100.0].shape[0] == 0

    # Check if the load with const_z_percent != 100.0 is not modified
    assert net.load.loc[net.load["const_z_percent"] == 50.0].shape[0] == 1


def test_modify_constan_z_load_with_different_value():
    # Create a simple pandapower network
    net = pp.create_empty_network()

    # Add some buses
    b1 = pp.create_bus(net, vn_kv=20)
    b2 = pp.create_bus(net, vn_kv=0.4)

    # Add a load with const_z_percent = 100.0
    pp.create_load(net, bus=b1, p_mw=0.1, q_mvar=0.05, const_z_percent=100.0)
    pp.create_load(net, bus=b2, p_mw=0.2, q_mvar=0.1, const_z_percent=50.0)

    # Modify constant z load
    modify_constan_z_load(net, value=75.0)

    # Check if the load with const_z_percent = 100.0 is modified
    assert net.load.loc[net.load["const_z_percent"] == 75.0].shape[0] == 1
    assert net.load.loc[net.load["const_z_percent"] == 100.0].shape[0] == 0

    # Check if the load with const_z_percent != 100.0 is not modified
    assert net.load.loc[net.load["const_z_percent"] == 50.0].shape[0] == 1


def test_create_nminus1_definition_from_masks_basic(ucte_file):
    network = pypowsybl.network.load(ucte_file)
    masks = powsybl_masks.create_default_network_masks(network=network)
    # Set some masks to True to create monitored elements and contingencies
    masks.line_for_reward[0] = True
    masks.line_for_nminus1[1] = True
    masks.trafo_for_reward[2] = True
    masks.trafo_for_nminus1[3] = True
    masks.tie_line_for_reward[0] = True
    masks.tie_line_for_nminus1[0] = True
    masks.generator_for_nminus1[0] = True
    masks.load_for_nminus1[0] = True
    masks.switch_for_reward[0] = True
    masks.switch_for_nminus1[0] = True
    nminus1_def = create_nminus1_definition_from_masks(network, masks)
    monitored_ids = [e.id for e in nminus1_def.monitored_elements]
    contingency_ids = [c.id for c in nminus1_def.contingencies]
    lines = network.get_lines()
    assert lines.index[0] in monitored_ids  # line_for_reward
    assert lines.index[1] in contingency_ids  # line_for_nminus1
    trafos = network.get_2_windings_transformers()
    assert trafos.index[2] in monitored_ids  # trafo_for_reward
    assert trafos.index[3] in contingency_ids  # trafo_for_nminus1
    tie_lines = network.get_tie_lines()
    assert tie_lines.index[0] in monitored_ids  # tie_line_for_reward
    assert tie_lines.index[0] in contingency_ids  # tie_line_for_n

    generators = network.get_generators()
    assert generators.index[0] in contingency_ids  # generator_for_nminus1
    loads = network.get_loads()
    assert loads.index[0] in contingency_ids  # load_for_nminus1
    switches = network.get_switches()
    assert switches.index[0] in monitored_ids  # switch_for_reward
    assert switches.index[0] in contingency_ids  # switch_for_nminus1
    # BASECASE contingency should exist
    assert "BASECASE" in contingency_ids


def test_create_nminus1_definition_from_masks_busbars(basic_node_breaker_network_powsybl_grid: Network) -> None:
    network = basic_node_breaker_network_powsybl_grid
    masks = powsybl_masks.create_default_network_masks(network=network)
    masks.busbar_for_nminus1[0] = True

    nminus1_def = create_nminus1_definition_from_masks(network, masks)
    contingency_ids = [contingency.id for contingency in nminus1_def.contingencies]
    busbar_sections = network.get_busbar_sections(attributes=["name"])

    assert busbar_sections.index[0] in contingency_ids
