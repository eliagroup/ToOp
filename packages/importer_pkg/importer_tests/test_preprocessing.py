import time
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import logbook
import pandapower as pp
import pandas as pd
import pypowsybl
from fsspec import filesystem
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_importer.pandapower_import.preprocessing import modify_constan_z_load
from toop_engine_importer.pypowsybl_import import powsybl_masks, preprocessing
from toop_engine_importer.pypowsybl_import.data_classes import PreProcessingStatistics
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
    StaticInformationStats,
    UcteImportResult,
)


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
    import_result = UcteImportResult(
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
            statistics, file_path=json_test_file, filesystem=filesystem("file")
        )
        loaded_statistics = preprocessing.load_preprocessing_statistics_filesystem(
            json_test_file, filesystem=filesystem("file")
        )
        assert isinstance(loaded_statistics, PreProcessingStatistics)
        assert statistics == loaded_statistics
        assert loaded_statistics.id_lists == statistics.id_lists
        assert loaded_statistics.import_result == statistics.import_result
        assert loaded_statistics.border_current == statistics.border_current
        assert loaded_statistics.network_changes == statistics.network_changes
        assert loaded_statistics.import_parameter == statistics.import_parameter
        assert isinstance(loaded_statistics.import_result, UcteImportResult)
        assert isinstance(loaded_statistics.import_parameter, UcteImporterParameters)


def test_fill_statistics_for_network_masks(ucte_file, ucte_importer_parameters):
    network = pypowsybl.network.load(ucte_file)

    import_result = UcteImportResult(
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

    assert statistics.import_result == UcteImportResult(
        data_folder=Path(""),
    )
    assert statistics.border_current == {}
    assert statistics.network_changes == {}
    assert statistics.import_parameter is None
    for key, value in statistics.id_lists.items():
        assert value == []

    masks = powsybl_masks.make_masks(network=network, importer_parameters=ucte_importer_parameters)
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
        logger = logbook.Logger(__name__)

        def heartbeat_working(
            stage: PreprocessStage,
            message: Optional[str],
            preprocess_id: str,
            start_time: float,
        ):
            logger.info(
                f"Preprocessing stage {stage} for job {preprocess_id} after {(time.time() - start_time):f}s: {message}"
            )

        start_time = time.time()
        heartbeat_fn = partial(
            heartbeat_working,
            preprocess_id="test_id",
            start_time=start_time,
        )
        importer_parameters = UcteImporterParameters(
            grid_model_file=ucte_file,
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
        assert importer_auxiliary_file.exists()
        assert grid_file_path.exists()
        for file_name in powsybl_masks.NetworkMasks.__annotations__.keys():
            assert (mask_dir / NETWORK_MASK_NAMES[file_name]).exists(), f"{NETWORK_MASK_NAMES[file_name]} does not exist"
        assert isinstance(import_result, UcteImportResult)

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
        assert importer_auxiliary_file.exists()
        assert grid_file_path.exists()
        for file_name in powsybl_masks.NetworkMasks.__annotations__.keys():
            assert (mask_dir / NETWORK_MASK_NAMES[file_name]).exists(), f"{NETWORK_MASK_NAMES[file_name]} does not exist"
        assert isinstance(import_result, UcteImportResult)


def test_convert_file_node_breaker_with_svc(basic_node_breaker_network_powsybl: pypowsybl.network.Network):
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

        pypowsybl.network.create_static_var_compensator_bay(basic_node_breaker_network_powsybl, df=svc)
        basic_node_breaker_network_powsybl.save(temp_grid_file)
        # def parameters for function
        logger = logbook.Logger(__name__)

        def heartbeat_working(
            stage: PreprocessStage,
            message: Optional[str],
            preprocess_id: str,
            start_time: float,
        ):
            logger.info(
                f"Preprocessing stage {stage} for job {preprocess_id} after {(time.time() - start_time):f}s: {message}"
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
        assert importer_auxiliary_file.exists()
        assert grid_file_path.exists()
        for file_name in powsybl_masks.NetworkMasks.__annotations__.keys():
            assert (mask_dir / NETWORK_MASK_NAMES[file_name]).exists(), f"{NETWORK_MASK_NAMES[file_name]} does not exist"
        assert isinstance(import_result, UcteImportResult)

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
        assert importer_auxiliary_file.exists()
        assert grid_file_path.exists()
        for file_name in powsybl_masks.NetworkMasks.__annotations__.keys():
            assert (mask_dir / NETWORK_MASK_NAMES[file_name]).exists(), f"{NETWORK_MASK_NAMES[file_name]} does not exist"
        assert isinstance(import_result, UcteImportResult)

        net_loaded = pypowsybl.network.load(grid_file_path)
        assert len(net_loaded.get_static_var_compensators()) == 1

        # make sure the dc solver does not crash with the svc
        info, _, _ = load_grid(
            data_folder=import_result.data_folder,
            pandapower=False,
            parameters=PreprocessParameters(),
            status_update_fn=heartbeat_fn,
        )
        assert isinstance(info, StaticInformationStats)


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
