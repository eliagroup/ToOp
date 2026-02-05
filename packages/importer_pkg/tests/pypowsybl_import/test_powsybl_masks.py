# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import tempfile
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pypowsybl
import pytest
from fsspec.implementations.local import LocalFileSystem
from toop_engine_importer.pypowsybl_import import powsybl_masks
from toop_engine_importer.pypowsybl_import.ucte.powsybl_masks_ucte import get_switchable_buses_ucte
from toop_engine_interfaces.folder_structure import (
    NETWORK_MASK_NAMES,
    PREPROCESSING_PATHS,
)
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    AreaSettings,
    CgmesImporterParameters,
    LimitAdjustmentParameters,
    UcteImporterParameters,
)


def test_create_default_network_masks():
    network = pypowsybl.network.create_micro_grid_be_network()
    masks = powsybl_masks.create_default_network_masks(network)
    assert isinstance(masks, powsybl_masks.NetworkMasks)
    assert isinstance(masks.relevant_subs, np.ndarray)
    assert isinstance(masks.line_for_nminus1, np.ndarray)
    assert isinstance(masks.line_for_reward, np.ndarray)
    assert isinstance(masks.line_overload_weight, np.ndarray)
    assert isinstance(masks.line_disconnectable, np.ndarray)
    assert isinstance(masks.trafo_for_nminus1, np.ndarray)
    assert isinstance(masks.trafo_for_reward, np.ndarray)
    assert isinstance(masks.trafo_overload_weight, np.ndarray)
    assert isinstance(masks.trafo_disconnectable, np.ndarray)
    assert isinstance(masks.trafo_n0_n1_max_diff_factor, np.ndarray)
    assert isinstance(masks.tie_line_for_reward, np.ndarray)
    assert isinstance(masks.tie_line_for_nminus1, np.ndarray)
    assert isinstance(masks.tie_line_overload_weight, np.ndarray)
    assert isinstance(masks.tie_line_disconnectable, np.ndarray)
    assert isinstance(masks.dangling_line_for_nminus1, np.ndarray)
    assert isinstance(masks.generator_for_nminus1, np.ndarray)
    assert isinstance(masks.load_for_nminus1, np.ndarray)
    assert isinstance(masks.switch_for_nminus1, np.ndarray)
    assert isinstance(masks.switch_for_reward, np.ndarray)


def test_validate_network_masks(ucte_importer_parameters: UcteImporterParameters):
    network = pypowsybl.network.create_micro_grid_be_network()
    masks_default = powsybl_masks.create_default_network_masks(network)
    masks = powsybl_masks.make_masks(network=network, importer_parameters=ucte_importer_parameters)
    assert powsybl_masks.validate_network_masks(masks, masks_default)
    masks = replace(masks, line_disconnectable=[1])
    assert not powsybl_masks.validate_network_masks(masks, masks_default)


def test_get_mask_for_area_codes():
    line_df = pd.DataFrame(
        {
            "id": ["line1", "line2", "line3"],
            "bus1_id": ["A1", "A2", "A1"],
            "bus2_id": ["A1", "A2", "A1"],
        }
    )
    area_codes = ["A1", "A2"]
    area_mask = powsybl_masks.get_mask_for_area_codes(line_df, area_codes, "bus1_id", "bus2_id")
    assert all(area_mask == np.array([True, True, True]))

    area_codes = ["A3", "A2"]
    area_mask = powsybl_masks.get_mask_for_area_codes(line_df, area_codes, "bus1_id", "bus2_id")
    assert all(area_mask == np.array([False, True, False]))

    area_codes = ["A1"]
    area_mask = powsybl_masks.get_mask_for_area_codes(line_df, area_codes, "bus1_id")
    assert all(area_mask == np.array([True, False, True]))


def test_update_line_masks(ucte_file_with_border, ucte_importer_parameters: UcteImporterParameters):
    network = pypowsybl.network.load(ucte_file_with_border)
    lines = network.get_lines(attributes=["voltage_level1_id"])
    default_masks = powsybl_masks.create_default_network_masks(network)
    ucte_importer_parameters.area_settings.nminus1_area = ["D2", "D4", "D7", "D8"]
    ucte_importer_parameters.area_settings.view_area = ["D2", "D4", "D7", "D8"]
    ucte_importer_parameters.area_settings.control_area = ["D8"]
    network_masks = powsybl_masks.update_line_masks(
        default_masks,
        network,
        ucte_importer_parameters,
        blacklisted_ids=[],
    )

    assert np.array_equal(network_masks.line_for_nminus1, np.array([True, True, True, True, False, False]))
    assert np.array_equal(network_masks.line_for_reward, np.array([True, True, True, True, False, False]))
    assert np.array_equal(network_masks.line_overload_weight, np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    assert np.array_equal(
        network_masks.line_disconnectable,
        np.array([True, True, True, False, False, False]),
    )
    assert np.array_equal(
        network_masks.line_blacklisted,
        np.array([False, False, False, False, False, False]),
    )
    assert np.array_equal(
        network_masks.line_tso_border,
        np.array([False, False, False, False, False, False]),
    )

    ucte_importer_parameters.area_settings.nminus1_area = ["D8"]
    ucte_importer_parameters.area_settings.view_area = ["D8"]

    network_masks = powsybl_masks.update_line_masks(
        default_masks,
        network,
        ucte_importer_parameters,
        blacklisted_ids=[lines.index[0]],
    )
    assert np.array_equal(
        network_masks.line_for_nminus1,
        np.array([True, True, True, False, False, False]),
    )
    assert np.array_equal(network_masks.line_for_reward, np.array([True, True, True, False, False, False]))
    assert np.array_equal(network_masks.line_overload_weight, np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    assert np.array_equal(
        network_masks.line_disconnectable,
        np.array([True, True, True, False, False, False]),
    )
    assert np.array_equal(
        network_masks.line_blacklisted,
        np.array([True, False, False, False, False, False]),
    )
    assert np.array_equal(
        network_masks.line_tso_border,
        np.array([False, True, True, False, False, False]),
    )
    ucte_importer_parameters.area_settings.border_line_weight = 10.0
    network_masks = powsybl_masks.update_line_masks(
        default_masks,
        network,
        ucte_importer_parameters,
        blacklisted_ids=[lines.index[0]],
    )
    assert np.array_equal(network_masks.line_overload_weight, np.array([1.0, 10.0, 10.0, 1.0, 1.0, 1.0]))


def test_update_tie_and_dangling_lines(ucte_file_with_border, ucte_importer_parameters: UcteImporterParameters):
    network = pypowsybl.network.load(ucte_file_with_border)
    default_masks = powsybl_masks.create_default_network_masks(network)
    network_masks = powsybl_masks.update_tie_and_dangling_line_masks(default_masks, network, ucte_importer_parameters)
    assert np.array_equal(network_masks.tie_line_for_reward, np.array([True, True]))
    assert np.array_equal(network_masks.tie_line_for_nminus1, np.array([True, True]))
    assert np.array_equal(network_masks.tie_line_overload_weight, np.array([1.0, 1.0]))
    assert np.array_equal(network_masks.tie_line_disconnectable, np.array([False, False]))
    assert np.array_equal(network_masks.tie_line_tso_border, np.array([True, True]))
    assert np.array_equal(
        network_masks.dangling_line_for_nminus1,
        np.array([False, True, False, True, True]),
    )
    ucte_importer_parameters.area_settings.border_line_weight = 10.0
    network_masks = powsybl_masks.update_tie_and_dangling_line_masks(default_masks, network, ucte_importer_parameters)
    assert np.array_equal(
        network_masks.tie_line_overload_weight,
        np.array(
            [
                10.0,
                10.0,
            ]
        ),
    )


def test_update_switches_mask(ucte_file_with_border, ucte_importer_parameters: UcteImporterParameters):
    network = pypowsybl.network.load(ucte_file_with_border)
    default_masks = powsybl_masks.create_default_network_masks(network)
    network_masks = powsybl_masks.update_switch_masks(default_masks, network, ucte_importer_parameters)
    assert np.array_equal(network_masks.switch_for_nminus1, np.array([True]))
    assert np.array_equal(network_masks.switch_for_reward, np.array([False]))


def test_update_load_and_generation_masks(ucte_file_with_border, ucte_importer_parameters: UcteImporterParameters):
    network = pypowsybl.network.load(ucte_file_with_border)
    default_masks = powsybl_masks.create_default_network_masks(network)
    network_masks = powsybl_masks.update_load_and_generation_masks(default_masks, network, ucte_importer_parameters)
    assert np.array_equal(
        network_masks.generator_for_nminus1,
        np.array([False, False, False, False, True, False]),
    )
    assert np.array_equal(network_masks.load_for_nminus1, np.array([False, True, False, False, False]))


def test_update_bus_masks(ucte_file_with_border, ucte_importer_parameters: UcteImporterParameters):
    importer_parameters = deepcopy(ucte_importer_parameters)
    network = pypowsybl.network.load(ucte_file_with_border)
    default_masks = powsybl_masks.create_default_network_masks(network)

    network_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, filesystem=LocalFileSystem())

    expected_bus_mask = np.zeros(17, dtype=bool)
    expected_bus_mask[3] = True
    assert np.array_equal(network_masks.relevant_subs, expected_bus_mask)

    # test ignore file
    file_content = "grid_model_id;reason\nD8SU1_1;to complex"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = f"{temp_dir}/ignore_list.csv"
        with open(temp_file_path, "w") as temp_file:
            temp_file.write(file_content)

        importer_parameters.ignore_list_file = temp_file_path
        updated_masks = powsybl_masks.update_bus_masks(
            default_masks, network, importer_parameters, filesystem=LocalFileSystem()
        )
        expected_bus_mask[3] = False
        assert np.array_equal(updated_masks.relevant_subs, expected_bus_mask)

    # test ignore file with id not complete
    file_content = "grid_model_id;reason\nD8SU1;to complex"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = f"{temp_dir}/ignore_list.csv"
        with open(temp_file_path, "w") as temp_file:
            temp_file.write(file_content)

        importer_parameters.ignore_list_file = temp_file_path
        updated_masks = powsybl_masks.update_bus_masks(
            default_masks, network, importer_parameters, filesystem=LocalFileSystem()
        )
        expected_bus_mask[3] = True
        assert np.array_equal(updated_masks.relevant_subs, expected_bus_mask)

    # test select_station_grid_model_id_list
    importer_parameters.ignore_list_file = None
    importer_parameters.select_by_voltage_level_id_list = list(network.get_voltage_levels().index)
    updated_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, filesystem=LocalFileSystem())
    assert np.array_equal(updated_masks.relevant_subs, network_masks.relevant_subs)
    importer_parameters.select_by_voltage_level_id_list = ["D8SU1_1"]
    updated_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, filesystem=LocalFileSystem())
    assert np.array_equal(updated_masks.relevant_subs, network_masks.relevant_subs)
    importer_parameters.select_by_voltage_level_id_list = ["D8SU1_2"]
    updated_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, filesystem=LocalFileSystem())
    assert not (updated_masks.relevant_subs).any()

    # test independent of area codes
    importer_parameters.area_settings.control_area = ["D2"]
    importer_parameters.area_settings.cutoff_voltage = 1000
    importer_parameters.select_by_voltage_level_id_list = ["D8SU1_1"]
    updated_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, filesystem=LocalFileSystem())
    assert np.array_equal(updated_masks.relevant_subs, network_masks.relevant_subs)


def test_update_bus_masks_node_breaker_select_station(basic_node_breaker_network_powsybl_network_graph):
    network = basic_node_breaker_network_powsybl_network_graph
    importer_parameters = CgmesImporterParameters(
        grid_model_file=Path("cgmes_file.zip"),
        data_folder="data_folder",
        area_settings=AreaSettings(cutoff_voltage=220, control_area=["BE"], view_area=["BE"], nminus1_area=["BE"]),
    )
    default_masks = powsybl_masks.create_default_network_masks(network)
    updated_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, filesystem=LocalFileSystem())

    expected_bus_mask = np.array([True, True, True, False, False])
    assert np.array_equal(updated_masks.relevant_subs, expected_bus_mask)

    # make sure the slack is removed from relevant subs
    network_masks = powsybl_masks.make_masks(
        network=network,
        importer_parameters=importer_parameters,
        blacklisted_ids=[],
    )
    expected_bus_mask_no_slack = np.array([False, True, True, False, False])
    assert np.array_equal(network_masks.relevant_subs, expected_bus_mask_no_slack)

    importer_parameters.select_by_voltage_level_id_list = list(network.get_voltage_levels().index)
    updated_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, filesystem=LocalFileSystem())
    assert np.array_equal(updated_masks.relevant_subs, expected_bus_mask)

    importer_parameters.select_by_voltage_level_id_list = list(network.get_voltage_levels().index)[:2]
    updated_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, filesystem=LocalFileSystem())
    expected_bus_mask = np.array([True, True, False, False, False])
    assert np.array_equal(updated_masks.relevant_subs, expected_bus_mask)

    # make sure the slack is removed from relevant subs
    network_masks = powsybl_masks.make_masks(
        network=network,
        importer_parameters=importer_parameters,
        blacklisted_ids=[],
    )
    expected_bus_mask_no_slack = np.array([False, True, False, False, False])
    assert np.array_equal(network_masks.relevant_subs, expected_bus_mask_no_slack)

    # test independent of area codes
    importer_parameters.area_settings.control_area = ["FR"]
    importer_parameters.area_settings.cutoff_voltage = 1000
    importer_parameters.select_by_voltage_level_id_list = list(network.get_voltage_levels().index)[:2]
    updated_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, filesystem=LocalFileSystem())
    expected_bus_mask = np.array([True, True, False, False, False])
    assert np.array_equal(updated_masks.relevant_subs, expected_bus_mask)


def test_update_trafo_masks(ucte_file_with_border, ucte_importer_parameters: UcteImporterParameters):
    network = pypowsybl.network.load(ucte_file_with_border)
    trafos = network.get_2_windings_transformers(attributes=["voltage_level1_id", "voltage_level2_id"])
    default_masks = powsybl_masks.create_default_network_masks(network)

    ucte_importer_parameters.area_settings.nminus1_area = ["D2", "D4", "D7", "D8"]
    ucte_importer_parameters.area_settings.control_area = ["D2", "D4", "D7", "D8"]
    network_masks = powsybl_masks.update_trafo_masks(
        default_masks,
        network,
        ucte_importer_parameters,
        blacklisted_ids=[],
    )

    assert np.array_equal(
        network_masks.trafo_for_nminus1,
        np.array([False, False, True, False, False, False]),
    )
    assert np.array_equal(
        network_masks.trafo_for_reward,
        np.array([False, False, True, False, False, False]),
    )
    assert np.array_equal(network_masks.trafo_overload_weight, np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    assert np.array_equal(
        network_masks.trafo_disconnectable,
        np.array([False, False, True, False, False, False]),
    )
    assert np.array_equal(
        network_masks.trafo_n0_n1_max_diff_factor,
        np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
    )
    assert np.array_equal(
        network_masks.trafo_blacklisted,
        np.array([False, False, False, False, False, False]),
    )
    assert np.array_equal(
        network_masks.trafo_dso_border,
        np.array([False, False, False, False, False, False]),
    )
    assert np.array_equal(
        network_masks.trafo_pst_controllable,
        np.array([False, False, True, False, False, False]),
    )

    ucte_importer_parameters.area_settings.nminus1_area = ["D8"]

    network_masks = powsybl_masks.update_trafo_masks(
        default_masks,
        network,
        importer_parameters=ucte_importer_parameters,
        blacklisted_ids=[trafos.index[2]],
    )
    assert np.array_equal(
        network_masks.trafo_blacklisted,
        np.array([False, False, True, False, False, False]),
    )


def test_trafo_dso_border(ucte_file_with_border, ucte_importer_parameters: UcteImporterParameters):
    network = pypowsybl.network.load(ucte_file_with_border)
    trafos = network.get_2_windings_transformers(attributes=["voltage_level1_id", "voltage_level2_id"])
    # Make sure there is a dso trafo
    ucte_importer_parameters.area_settings.nminus1_area = [trafos.iloc[0].voltage_level1_id[:2]]
    default_masks = powsybl_masks.create_default_network_masks(network)

    network_masks = powsybl_masks.update_trafo_masks(
        default_masks,
        network,
        importer_parameters=ucte_importer_parameters,
        blacklisted_ids=[],
    )
    assert network_masks.trafo_for_nminus1[0]
    assert np.array_equal(
        network_masks.trafo_dso_border,
        np.array([True, False, False, False, False, False]),
    )
    ucte_importer_parameters.area_settings.dso_trafo_weight = 10.0

    network_masks = powsybl_masks.update_trafo_masks(
        default_masks,
        network,
        importer_parameters=ucte_importer_parameters,
        blacklisted_ids=[],
    )
    assert np.array_equal(network_masks.trafo_overload_weight, np.array([10.0, 1.0, 1.0, 1.0, 1.0, 1.0]))


def test_make_masks(ucte_file_with_border, ucte_importer_parameters: UcteImporterParameters):
    network = pypowsybl.network.load(ucte_file_with_border)
    default_masks = powsybl_masks.create_default_network_masks(network)
    masks = powsybl_masks.make_masks(network=network, importer_parameters=ucte_importer_parameters)
    assert powsybl_masks.validate_network_masks(masks, default_masks)


def test_make_masks_node_breaker(
    basic_node_breaker_network_powsybl_not_disconnectable, cgmes_importer_parameters: CgmesImporterParameters
):
    network = basic_node_breaker_network_powsybl_not_disconnectable
    default_masks = powsybl_masks.create_default_network_masks(network)
    masks = powsybl_masks.make_masks(network=network, importer_parameters=cgmes_importer_parameters)
    assert powsybl_masks.validate_network_masks(masks, default_masks)


def test_update_masks_from_contingency_list_file(
    ucte_file_with_border, ucte_importer_parameters: UcteImporterParameters, tmp_path_factory: pytest.TempPathFactory
):
    contingency_file = tmp_path_factory.mktemp("contingency_file") / "contingency.csv"
    contingency_file.write_text(
        """contingency_name;contingency_id;power_factory_grid_model_name;power_factory_grid_model_fid;power_factory_grid_model_rdf_id
        line1;1;D8SU1_12 D8SU1_11 2;D8SU1_12 D8SU1_11 2;D8SU1_12 D8SU1_11 2
        line2;2;B_SU2_11 B_SU1_11 1;B_SU2_11 B_SU1_11 1;B_SU2_11 B_SU1_11 1
        line3;3;line3;line3;line3
        trafo1;3;B_SU1_11 B_SU1_21 1;B_SU1_11 B_SU1_21 1;B_SU1_11 B_SU1_21 1
        gen1;4;D7SU2_11_generator;D7SU2_11_generator;D7SU2_11_generator
        load1;5;HCCCCC1 _load;HCCCCC1 _load;HCCCCC1 _load
        switch1;6;D8SU1_12 D8SU1_11 1;D8SU1_12 D8SU1_11 1;D8SU1_12 D8SU1_11 1
        dangling;7;XB__F_21 B_SU1_21 1;XB__F_21 B_SU1_21 1;XB__F_21 B_SU1_21 1
        """
    )
    ucte_importer_parameters.contingency_list_file = contingency_file
    network = pypowsybl.network.load(ucte_file_with_border)
    default_masks = powsybl_masks.create_default_network_masks(network)

    network_masks = powsybl_masks.create_default_network_masks(network)
    network_masks = powsybl_masks.update_masks_from_power_factory_contingency_list_file(
        network_masks=network_masks,
        network=network,
        importer_parameters=ucte_importer_parameters,
        filesystem=LocalFileSystem(),
    )
    assert powsybl_masks.validate_network_masks(network_masks, default_masks)
    assert np.array_equal(network_masks.line_for_nminus1, np.array([True, False, False, False, False, True]))
    assert np.array_equal(network_masks.trafo_for_nminus1, np.array([False, True, False, False, False, False]))
    assert np.array_equal(network_masks.generator_for_nminus1, np.array([False, False, False, False, True, False]))
    assert np.array_equal(network_masks.load_for_nminus1, np.array([False, False, False, False, True]))
    assert np.array_equal(network_masks.switch_for_nminus1, np.array([True]))
    assert np.array_equal(network_masks.dangling_line_for_nminus1, np.array([False, False, True, False, False]))


def test_make_masks_with_contingency_file(
    ucte_file_with_border, ucte_importer_parameters: UcteImporterParameters, tmp_path_factory: pytest.TempPathFactory
):
    contingency_file = tmp_path_factory.mktemp("contingency_file") / "contingency.csv"
    contingency_file.write_text(
        """contingency_name;contingency_id;power_factory_grid_model_name;power_factory_grid_model_fid;power_factory_grid_model_rdf_id
        line1;1;D8SU1_12 D8SU1_11 2;D8SU1_12 D8SU1_11 2;D8SU1_12 D8SU1_11 2
        """
    )
    ucte_importer_parameters.contingency_list_file = contingency_file
    network = pypowsybl.network.load(ucte_file_with_border)
    default_masks = powsybl_masks.create_default_network_masks(network)

    network_masks = powsybl_masks.create_default_network_masks(network)
    network_masks = powsybl_masks.update_masks_from_power_factory_contingency_list_file(
        network_masks=network_masks,
        network=network,
        importer_parameters=ucte_importer_parameters,
        filesystem=LocalFileSystem(),
    )
    assert powsybl_masks.validate_network_masks(network_masks, default_masks)
    assert np.array_equal(network_masks.line_for_nminus1, np.array([True, False, False, False, False, False]))


def test_validate_masks(ucte_file_with_border, ucte_importer_parameters: UcteImporterParameters):
    network = pypowsybl.network.load(ucte_file_with_border)
    default_masks = powsybl_masks.create_default_network_masks(network)
    assert powsybl_masks.validate_network_masks(ucte_importer_parameters, default_masks) is False

    wrong_shape_masks = replace(default_masks, line_for_reward=default_masks.line_for_reward[:2])
    assert powsybl_masks.validate_network_masks(wrong_shape_masks, default_masks) is False

    wrong_dtype_masks = replace(default_masks, line_for_reward=default_masks.line_for_reward.astype(float))

    assert powsybl_masks.validate_network_masks(wrong_dtype_masks, default_masks) is False


def test_save_masks_to_files(ucte_file_with_border, ucte_importer_parameters: UcteImporterParameters):
    network = pypowsybl.network.load(ucte_file_with_border)
    default_masks = powsybl_masks.create_default_network_masks(network)
    powsybl_masks.save_masks_to_files(default_masks, ucte_importer_parameters.data_folder)
    for file_name in powsybl_masks.NetworkMasks.__annotations__.keys():
        assert (
            ucte_importer_parameters.data_folder / PREPROCESSING_PATHS["masks_path"] / NETWORK_MASK_NAMES[file_name]
        ).exists(), f"{NETWORK_MASK_NAMES[file_name]} does not exist"


def test_update_reward_masks_to_include_border_branches(
    ucte_file_with_border, ucte_importer_parameters: UcteImporterParameters
):
    network = pypowsybl.network.load(ucte_file_with_border)
    default_masks = powsybl_masks.create_default_network_masks(network)

    # Fake masks where all lines/tie_lines/trafos, that are not rewarded already are set as border=True
    all_borders_masks = replace(
        default_masks,
        line_tso_border=~default_masks.line_tso_border,
        tie_line_tso_border=~default_masks.tie_line_tso_border,
        trafo_dso_border=~default_masks.trafo_dso_border,
    )
    assert not all(all_borders_masks.tie_line_for_reward)
    assert not all(all_borders_masks.line_for_reward)
    assert not all(all_borders_masks.trafo_for_reward)

    # Test no update if limit factors are not set
    assert ucte_importer_parameters.area_settings.dso_trafo_factors is None
    assert ucte_importer_parameters.area_settings.border_line_factors is None
    updated_masks = powsybl_masks.update_reward_masks_to_include_border_branches(
        network_masks=all_borders_masks, importer_parameters=ucte_importer_parameters
    )

    assert np.array_equal(default_masks.tie_line_for_reward, updated_masks.tie_line_for_reward)
    assert np.array_equal(default_masks.line_for_reward, updated_masks.line_for_reward)
    assert np.array_equal(default_masks.trafo_for_reward, updated_masks.trafo_for_reward)

    # Test update when border_line_factors is set
    ucte_importer_parameters.area_settings.dso_trafo_factors = None
    ucte_importer_parameters.area_settings.border_line_factors = LimitAdjustmentParameters()

    updated_masks = powsybl_masks.update_reward_masks_to_include_border_branches(
        network_masks=all_borders_masks, importer_parameters=ucte_importer_parameters
    )

    assert np.array_equal(
        updated_masks.tie_line_for_reward,
        all_borders_masks.tie_line_for_reward | all_borders_masks.tie_line_tso_border,
    )
    assert np.array_equal(
        updated_masks.line_for_reward,
        all_borders_masks.line_for_reward | all_borders_masks.line_tso_border,
    )
    assert np.array_equal(updated_masks.trafo_for_reward, default_masks.trafo_for_reward)

    # Test update when dso trafo factor is set
    ucte_importer_parameters.area_settings.dso_trafo_factors = LimitAdjustmentParameters()
    ucte_importer_parameters.area_settings.border_line_factors = None
    updated_masks = powsybl_masks.update_reward_masks_to_include_border_branches(
        network_masks=all_borders_masks, importer_parameters=ucte_importer_parameters
    )

    assert np.array_equal(default_masks.tie_line_for_reward, updated_masks.tie_line_for_reward)
    assert np.array_equal(default_masks.line_for_reward, updated_masks.line_for_reward)
    assert np.array_equal(
        updated_masks.trafo_for_reward,
        all_borders_masks.trafo_for_reward | all_borders_masks.trafo_dso_border,
    )

    # Test when both are set
    # Test update when border_line_factors is set
    ucte_importer_parameters.area_settings.dso_trafo_factors = LimitAdjustmentParameters()
    ucte_importer_parameters.area_settings.border_line_factors = LimitAdjustmentParameters()
    updated_masks = powsybl_masks.update_reward_masks_to_include_border_branches(
        network_masks=all_borders_masks, importer_parameters=ucte_importer_parameters
    )

    assert np.array_equal(
        updated_masks.tie_line_for_reward,
        all_borders_masks.tie_line_for_reward | all_borders_masks.tie_line_tso_border,
    )
    assert np.array_equal(
        updated_masks.line_for_reward,
        all_borders_masks.line_for_reward | all_borders_masks.line_tso_border,
    )
    assert np.array_equal(
        updated_masks.trafo_for_reward,
        all_borders_masks.trafo_for_reward | all_borders_masks.trafo_dso_border,
    )


def test_get_switchable_buses():
    # test 1
    expected = ["S1VL1_0", "S1VL2_0", "S2VL1_0", "S3VL1_0", "S4VL1_0"]
    network = pypowsybl.network.create_four_substations_node_breaker_network_with_extensions()
    voltage_level_prefix = ["S"]
    buses = get_switchable_buses_ucte(network, voltage_level_prefix)
    assert isinstance(buses, list)
    assert all(isinstance(bus, str) for bus in buses)
    assert all(bus.startswith(tuple(voltage_level_prefix)) for bus in buses)
    assert len(buses) > 0
    assert len(buses) <= len(network.get_buses())
    assert buses == expected

    # test 2
    voltage_level_prefix = ["S3"]
    expected = ["S3VL1_0"]
    buses = get_switchable_buses_ucte(network, voltage_level_prefix)
    assert isinstance(buses, list)
    assert all(isinstance(bus, str) for bus in buses)
    assert all(bus.startswith(tuple(voltage_level_prefix)) for bus in buses)
    assert len(buses) > 0
    assert len(buses) <= len(network.get_buses())
    assert buses == expected

    # test select by voltage level
    expected = ["S1VL1_0", "S1VL2_0", "S2VL1_0", "S3VL1_0", "S4VL1_0"]
    select_by_voltage_level_id_list = ["S1VL1", "S1VL2", "S2VL1", "S3VL1", "S4VL1"]
    network = pypowsybl.network.create_four_substations_node_breaker_network_with_extensions()
    voltage_level_prefix = ["OVERWRITE"]
    cutoff_voltage = 1000
    buses = get_switchable_buses_ucte(
        network,
        voltage_level_prefix,
        cutoff_voltage=cutoff_voltage,
        select_by_voltage_level_id_list=select_by_voltage_level_id_list,
    )
    assert buses == expected

    # test select by voltage level
    expected = ["S1VL1_0", "S1VL2_0", "S4VL1_0"]
    select_by_voltage_level_id_list = ["S1VL1", "S1VL2", "S4VL1"]
    network = pypowsybl.network.create_four_substations_node_breaker_network_with_extensions()
    voltage_level_prefix = ["OVERWRITE"]
    cutoff_voltage = 1000
    buses = get_switchable_buses_ucte(
        network,
        voltage_level_prefix,
        cutoff_voltage=cutoff_voltage,
        select_by_voltage_level_id_list=select_by_voltage_level_id_list,
    )
    assert buses == expected


def test_get_switchable_buses_no_switches():
    network = pypowsybl.network.create_four_substations_node_breaker_network_with_extensions()
    switches = network.get_switches()
    network.remove_elements(switches.index)
    voltage_level_prefix = ["S"]
    buses = get_switchable_buses_ucte(network, voltage_level_prefix)
    assert len(buses) == 0


def test_remove_slack_from_relevant_subs(ucte_file_with_border):
    network = pypowsybl.network.load(ucte_file_with_border)
    default_masks = powsybl_masks.create_default_network_masks(network)
    bus_df = network.get_buses(attributes=[])
    network_masks = replace(
        default_masks,
        relevant_subs=np.ones(len(bus_df), dtype=bool),
    )
    result = powsybl_masks.remove_slack_from_relevant_subs(network, network_masks)
    expected = np.array(
        [True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True]
    )
    assert np.array_equal(result.relevant_subs, expected)


def test_update_masks_contingency_list_file(tmp_path, ucte_file_with_border, ucte_importer_parameters):
    network = pypowsybl.network.load(ucte_file_with_border)
    lines = network.get_lines()
    trafos = network.get_2_windings_transformers()
    dangling_lines = network.get_dangling_lines()
    tie_lines = network.get_tie_lines()
    # create a random boolean array of length lines

    monitored_lines = np.random.choice([True, False], size=len(lines))
    monitored_trafos = np.random.choice([True, False], size=len(trafos))

    contingency_lines = np.random.choice([True, False], size=len(lines))
    contingency_trafos = np.random.choice([True, False], size=len(trafos))

    monitored_dangling = np.random.choice([True, False], size=len(dangling_lines))
    contingency_dangling = np.random.choice([True, False], size=len(dangling_lines))

    monitored_tie_lines = tie_lines.index.isin(dangling_lines[monitored_dangling].tie_line_id)
    contingency_tie_lines = tie_lines.index.isin(dangling_lines[contingency_dangling].tie_line_id)

    # Prepare a fake contingency list file
    contingency_data = pd.DataFrame(
        {
            "observe_std": list(monitored_lines) + list(monitored_trafos) + list(monitored_dangling),
            "contingency_case": list(contingency_lines) + list(contingency_trafos) + list(contingency_dangling),
        },
        index=list(lines.index) + list(trafos.index) + list(dangling_lines.index),
    )
    contingency_data.index.name = "mrid"
    contingency_file = tmp_path / "contingency.csv"
    contingency_data.to_csv(contingency_file)

    # Set the contingency file in the importer parameters
    ucte_importer_parameters.contingency_list_file = contingency_file
    ucte_importer_parameters.schema_format = "ContingencyImportSchema"

    # Call the function
    default_masks = powsybl_masks.create_default_network_masks(network)
    updated_masks = powsybl_masks.update_masks_from_contingency_list_file(
        network_masks=default_masks,
        network=network,
        importer_parameters=ucte_importer_parameters,
        filesystem=LocalFileSystem(),
        process_multi_outages=False,
    )

    # Check that the masks are set as expected
    assert np.array_equal(updated_masks.line_for_nminus1, contingency_lines), "Line for n-1 mask not updated correctly"
    assert np.array_equal(updated_masks.line_for_reward, monitored_lines), "Line for reward mask not updated correctly"
    assert np.array_equal(updated_masks.trafo_for_nminus1, contingency_trafos), "Trafo for n-1 mask not updated correctly"
    assert np.array_equal(updated_masks.trafo_for_reward, monitored_trafos), "Trafo for reward mask not updated correctly"
    assert np.array_equal(updated_masks.dangling_line_for_nminus1, contingency_dangling), (
        "Dangling line for n-1 mask not updated correctly"
    )
    assert np.array_equal(updated_masks.tie_line_for_nminus1, contingency_tie_lines), (
        "Tie line for n-1 mask not updated correctly"
    )
    assert np.array_equal(updated_masks.tie_line_for_reward, monitored_tie_lines), (
        "Tie line for reward mask not updated correctly"
    )

    # Test NotImplementedError for multi-outages
    with pytest.raises(NotImplementedError):
        powsybl_masks.update_masks_from_contingency_list_file(
            network_masks=default_masks,
            network=network,
            importer_parameters=ucte_importer_parameters,
            filesystem=LocalFileSystem(),
            process_multi_outages=True,
        )

    # Test AssertionError for 3wtrafos
    network.get_3_windings_transformers = network.get_2_windings_transformers
    with pytest.raises(AssertionError):
        powsybl_masks.update_masks_from_contingency_list_file(
            network_masks=default_masks,
            network=network,
            importer_parameters=ucte_importer_parameters,
            filesystem=LocalFileSystem(),
            process_multi_outages=False,
        )


def test_is_disconnectable(basic_node_breaker_network_powsybl_not_disconnectable):
    network = basic_node_breaker_network_powsybl_not_disconnectable
    res = powsybl_masks._is_disconnectable(network, ["not_disconnectable_line"])
    assert np.array_equal(res, np.array([False]))
    res = powsybl_masks._is_disconnectable(network, ["L1", "not_disconnectable_line"])
    assert np.array_equal(res, np.array([True, False]))
