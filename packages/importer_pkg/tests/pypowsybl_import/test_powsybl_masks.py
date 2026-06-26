# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
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
from pypowsybl.network import Network
from toop_engine_grid_helpers.powsybl.example_grids import grouped_pst_grid_example, parallel_pst_example
from toop_engine_importer.pypowsybl_import import network_analysis, powsybl_masks, preprocessing
from toop_engine_importer.pypowsybl_import.data_classes import PreProcessingStatistics
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
from toop_engine_interfaces.messages.preprocess.preprocess_results import ImportResult


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
    assert isinstance(masks.boundary_line_for_nminus1, np.ndarray)
    assert isinstance(masks.generator_for_nminus1, np.ndarray)
    assert isinstance(masks.load_for_nminus1, np.ndarray)
    assert isinstance(masks.switch_for_nminus1, np.ndarray)
    assert isinstance(masks.switch_for_reward, np.ndarray)
    assert isinstance(masks.pst_group_labels, np.ndarray)
    # Default group labels are -1 (no parallel-PST grouping identified yet).
    assert np.all(masks.pst_group_labels == -1)


def test_validate_network_masks(ucte_importer_parameters: UcteImporterParameters):
    network = pypowsybl.network.create_micro_grid_be_network()
    lf_result, *_ = pypowsybl.loadflow.run_dc(network)
    masks_default = powsybl_masks.create_default_network_masks(network)
    masks = powsybl_masks.make_masks(
        network=network, slack_id=lf_result.reference_bus_id, importer_parameters=ucte_importer_parameters
    )
    assert powsybl_masks.validate_network_masks(masks, masks_default)
    masks = replace(masks, line_disconnectable=np.array([1]))
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
        np.array([False, True, True, False, False, False]),
    )
    assert np.array_equal(network_masks.line_for_reward, np.array([False, True, True, False, False, False]))
    assert np.array_equal(network_masks.line_overload_weight, np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    assert np.array_equal(
        network_masks.line_disconnectable,
        np.array([False, True, True, False, False, False]),
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
    network_masks = powsybl_masks.update_tie_and_dangling_line_masks(
        default_masks, network, ucte_importer_parameters, blacklisted_ids=[]
    )
    assert np.array_equal(network_masks.tie_line_for_reward, np.array([True, True]))
    assert np.array_equal(network_masks.tie_line_for_nminus1, np.array([True, True]))
    assert np.array_equal(network_masks.tie_line_overload_weight, np.array([1.0, 1.0]))
    assert np.array_equal(network_masks.tie_line_disconnectable, np.array([False, False]))
    assert np.array_equal(network_masks.tie_line_tso_border, np.array([True, True]))
    assert np.array_equal(
        network_masks.boundary_line_for_nminus1,
        np.array([False, True, False, True, True]),
    )
    ucte_importer_parameters.area_settings.border_line_weight = 10.0
    network_masks = powsybl_masks.update_tie_and_dangling_line_masks(
        default_masks, network, ucte_importer_parameters, blacklisted_ids=[]
    )
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
    network_masks = powsybl_masks.update_switch_masks(default_masks, network, ucte_importer_parameters, blacklisted_ids=[])
    assert np.array_equal(network_masks.switch_for_nminus1, np.array([True]))
    assert np.array_equal(network_masks.switch_for_reward, np.array([False]))


def test_update_masks_apply_ignore_list(ucte_file_with_border, ucte_importer_parameters: UcteImporterParameters):
    importer_parameters = deepcopy(ucte_importer_parameters)
    network = pypowsybl.network.load(ucte_file_with_border)
    default_masks = powsybl_masks.create_default_network_masks(network)

    importer_parameters.area_settings.nminus1_area = ["D2", "D4", "D7", "D8"]
    importer_parameters.area_settings.view_area = ["D2", "D4", "D7", "D8"]
    importer_parameters.area_settings.control_area = ["D2", "D4", "D7", "D8"]

    line_masks = powsybl_masks.update_line_masks(default_masks, network, importer_parameters, blacklisted_ids=[])
    trafo_masks = powsybl_masks.update_trafo_masks(default_masks, network, importer_parameters, blacklisted_ids=[])
    tie_and_dangling_masks = powsybl_masks.update_tie_and_dangling_line_masks(
        default_masks, network, importer_parameters, blacklisted_ids=[]
    )
    generation_and_load_masks = powsybl_masks.update_load_and_generation_masks(
        default_masks, network, importer_parameters, blacklisted_ids=[]
    )
    switch_masks = powsybl_masks.update_switch_masks(default_masks, network, importer_parameters, blacklisted_ids=[])

    assert line_masks.line_for_nminus1.any()
    assert trafo_masks.trafo_for_nminus1.any()
    assert tie_and_dangling_masks.tie_line_for_nminus1.any()
    assert tie_and_dangling_masks.boundary_line_for_nminus1.any()
    assert generation_and_load_masks.generator_for_nminus1.any()
    assert generation_and_load_masks.load_for_nminus1.any()
    assert switch_masks.switch_for_nminus1.any()

    line_df = network.get_lines(attributes=[])
    trafo_df = network.get_2_windings_transformers(attributes=[])
    tie_df = network.get_tie_lines(attributes=[])
    dangling_df = network.get_boundary_lines(attributes=[])
    generator_df = network.get_generators(attributes=[])
    load_df = network.get_loads(attributes=[])
    switch_df = network.get_switches(attributes=[])

    ignored_line_id = line_df.index[np.flatnonzero(line_masks.line_for_nminus1)[0]]
    ignored_trafo_id = trafo_df.index[np.flatnonzero(trafo_masks.trafo_for_nminus1)[0]]
    ignored_tie_id = tie_df.index[np.flatnonzero(tie_and_dangling_masks.tie_line_for_nminus1)[0]]
    ignored_dangling_id = dangling_df.index[np.flatnonzero(tie_and_dangling_masks.boundary_line_for_nminus1)[0]]
    ignored_generator_id = generator_df.index[np.flatnonzero(generation_and_load_masks.generator_for_nminus1)[0]]
    ignored_load_id = load_df.index[np.flatnonzero(generation_and_load_masks.load_for_nminus1)[0]]
    ignored_switch_id = switch_df.index[np.flatnonzero(switch_masks.switch_for_nminus1)[0]]

    file_content = "grid_model_id;reason\n" + "\n".join(
        [
            f"{ignored_line_id};line",
            f"{ignored_trafo_id};trafo",
            f"{ignored_tie_id};tie",
            f"{ignored_dangling_id};dangling",
            f"{ignored_generator_id};generator",
            f"{ignored_load_id};load",
            f"{ignored_switch_id};switch",
        ]
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = f"{temp_dir}/ignore_list.csv"
        with open(temp_file_path, "w") as temp_file:
            temp_file.write(file_content)

        importer_parameters.ignore_list_file = temp_file_path

        line_masks_ignored = powsybl_masks.update_line_masks(
            default_masks,
            network,
            importer_parameters,
            blacklisted_ids=[ignored_line_id],
        )
        trafo_masks_ignored = powsybl_masks.update_trafo_masks(
            default_masks,
            network,
            importer_parameters,
            blacklisted_ids=[ignored_trafo_id],
        )
        tie_and_dangling_masks_ignored = powsybl_masks.update_tie_and_dangling_line_masks(
            default_masks,
            network,
            importer_parameters,
            blacklisted_ids=[ignored_tie_id, ignored_dangling_id],
        )
        generation_and_load_masks_ignored = powsybl_masks.update_load_and_generation_masks(
            default_masks,
            network,
            importer_parameters,
            blacklisted_ids=[ignored_generator_id, ignored_load_id],
        )
        switch_masks_ignored = powsybl_masks.update_switch_masks(
            default_masks,
            network,
            importer_parameters,
            blacklisted_ids=[ignored_switch_id],
        )

    ignored_line_idx = line_df.index.get_loc(ignored_line_id)
    ignored_trafo_idx = trafo_df.index.get_loc(ignored_trafo_id)
    ignored_tie_idx = tie_df.index.get_loc(ignored_tie_id)
    ignored_dangling_idx = dangling_df.index.get_loc(ignored_dangling_id)
    ignored_generator_idx = generator_df.index.get_loc(ignored_generator_id)
    ignored_load_idx = load_df.index.get_loc(ignored_load_id)
    ignored_switch_idx = switch_df.index.get_loc(ignored_switch_id)

    assert not line_masks_ignored.line_for_nminus1[ignored_line_idx]
    assert not line_masks_ignored.line_for_reward[ignored_line_idx]
    assert not line_masks_ignored.line_disconnectable[ignored_line_idx]

    assert not trafo_masks_ignored.trafo_for_nminus1[ignored_trafo_idx]
    assert not trafo_masks_ignored.trafo_for_reward[ignored_trafo_idx]
    assert not trafo_masks_ignored.trafo_disconnectable[ignored_trafo_idx]
    assert not trafo_masks_ignored.trafo_pst_linear[ignored_trafo_idx]

    assert not tie_and_dangling_masks_ignored.tie_line_for_nminus1[ignored_tie_idx]
    assert not tie_and_dangling_masks_ignored.tie_line_for_reward[ignored_tie_idx]
    assert not tie_and_dangling_masks_ignored.tie_line_tso_border[ignored_tie_idx]
    assert not tie_and_dangling_masks_ignored.boundary_line_for_nminus1[ignored_dangling_idx]

    assert not generation_and_load_masks_ignored.generator_for_nminus1[ignored_generator_idx]
    assert not generation_and_load_masks_ignored.load_for_nminus1[ignored_load_idx]

    assert not switch_masks_ignored.switch_for_nminus1[ignored_switch_idx]
    assert not switch_masks_ignored.switch_for_reward[ignored_switch_idx]


def test_update_masks_apply_ignore_list_cgmes(
    test_pypowsybl_cgmes_with_3w_trafo: Path,
    cgmes_importer_parameters: CgmesImporterParameters,
):
    importer_parameters = deepcopy(cgmes_importer_parameters)
    network = pypowsybl.network.load(test_pypowsybl_cgmes_with_3w_trafo)
    default_masks = powsybl_masks.create_default_network_masks(network)

    line_masks = powsybl_masks.update_line_masks(default_masks, network, importer_parameters, blacklisted_ids=[])
    trafo_masks = powsybl_masks.update_trafo_masks(default_masks, network, importer_parameters, blacklisted_ids=[])
    tie_and_dangling_masks = powsybl_masks.update_tie_and_dangling_line_masks(
        default_masks, network, importer_parameters, blacklisted_ids=[]
    )
    generation_and_load_masks = powsybl_masks.update_load_and_generation_masks(
        default_masks, network, importer_parameters, blacklisted_ids=[]
    )
    switch_masks = powsybl_masks.update_switch_masks(default_masks, network, importer_parameters, blacklisted_ids=[])

    assert any(
        [
            line_masks.line_for_nminus1.any(),
            trafo_masks.trafo_for_nminus1.any(),
            tie_and_dangling_masks.tie_line_for_nminus1.any(),
            tie_and_dangling_masks.boundary_line_for_nminus1.any(),
            generation_and_load_masks.generator_for_nminus1.any(),
            generation_and_load_masks.load_for_nminus1.any(),
            switch_masks.switch_for_nminus1.any(),
        ]
    )

    line_df = network.get_lines(attributes=[])
    trafo_df = network.get_2_windings_transformers(attributes=[])
    tie_df = network.get_tie_lines(attributes=[])
    dangling_df = network.get_boundary_lines(attributes=[])
    generator_df = network.get_generators(attributes=[])
    load_df = network.get_loads(attributes=[])
    switch_df = network.get_switches(attributes=[])

    ignored_line_id = (
        line_df.index[np.flatnonzero(line_masks.line_for_nminus1)[0]] if line_masks.line_for_nminus1.any() else None
    )
    ignored_trafo_id = (
        trafo_df.index[np.flatnonzero(trafo_masks.trafo_for_nminus1)[0]] if trafo_masks.trafo_for_nminus1.any() else None
    )
    ignored_tie_id = (
        tie_df.index[np.flatnonzero(tie_and_dangling_masks.tie_line_for_nminus1)[0]]
        if tie_and_dangling_masks.tie_line_for_nminus1.any()
        else None
    )
    ignored_dangling_id = (
        dangling_df.index[np.flatnonzero(tie_and_dangling_masks.boundary_line_for_nminus1)[0]]
        if tie_and_dangling_masks.boundary_line_for_nminus1.any()
        else None
    )
    ignored_generator_id = (
        generator_df.index[np.flatnonzero(generation_and_load_masks.generator_for_nminus1)[0]]
        if generation_and_load_masks.generator_for_nminus1.any()
        else None
    )
    ignored_load_id = (
        load_df.index[np.flatnonzero(generation_and_load_masks.load_for_nminus1)[0]]
        if generation_and_load_masks.load_for_nminus1.any()
        else None
    )
    ignored_switch_id = (
        switch_df.index[np.flatnonzero(switch_masks.switch_for_nminus1)[0]]
        if switch_masks.switch_for_nminus1.any()
        else None
    )

    ignore_entries = [
        (ignored_line_id, "line"),
        (ignored_trafo_id, "trafo"),
        (ignored_tie_id, "tie"),
        (ignored_dangling_id, "dangling"),
        (ignored_generator_id, "generator"),
        (ignored_load_id, "load"),
        (ignored_switch_id, "switch"),
    ]
    file_content = "grid_model_id;reason\n" + "\n".join(
        [f"{grid_model_id};{reason}" for grid_model_id, reason in ignore_entries if grid_model_id is not None]
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = f"{temp_dir}/ignore_list.csv"
        with open(temp_file_path, "w") as temp_file:
            temp_file.write(file_content)

        importer_parameters.ignore_list_file = temp_file_path

        line_blacklist = [ignored_line_id] if ignored_line_id is not None else []
        line_masks_ignored = powsybl_masks.update_line_masks(
            default_masks,
            network,
            importer_parameters,
            blacklisted_ids=line_blacklist,
        )
        trafo_blacklist = [ignored_trafo_id] if ignored_trafo_id is not None else []
        trafo_masks_ignored = powsybl_masks.update_trafo_masks(
            default_masks,
            network,
            importer_parameters,
            blacklisted_ids=trafo_blacklist,
        )
        tie_and_dangling_blacklist = [
            element_id for element_id in [ignored_tie_id, ignored_dangling_id] if element_id is not None
        ]
        tie_and_dangling_masks_ignored = powsybl_masks.update_tie_and_dangling_line_masks(
            default_masks,
            network,
            importer_parameters,
            blacklisted_ids=tie_and_dangling_blacklist,
        )
        generation_and_load_blacklist = [
            element_id for element_id in [ignored_generator_id, ignored_load_id] if element_id is not None
        ]
        generation_and_load_masks_ignored = powsybl_masks.update_load_and_generation_masks(
            default_masks,
            network,
            importer_parameters,
            blacklisted_ids=generation_and_load_blacklist,
        )
        switch_blacklist = [ignored_switch_id] if ignored_switch_id is not None else []
        switch_masks_ignored = powsybl_masks.update_switch_masks(
            default_masks,
            network,
            importer_parameters,
            blacklisted_ids=switch_blacklist,
        )

    if ignored_line_id is not None:
        ignored_line_idx = line_df.index.get_loc(ignored_line_id)
        assert not line_masks_ignored.line_for_nminus1[ignored_line_idx]
        assert not line_masks_ignored.line_for_reward[ignored_line_idx]
        assert not line_masks_ignored.line_disconnectable[ignored_line_idx]

    if ignored_trafo_id is not None:
        ignored_trafo_idx = trafo_df.index.get_loc(ignored_trafo_id)
        assert not trafo_masks_ignored.trafo_for_nminus1[ignored_trafo_idx]
        assert not trafo_masks_ignored.trafo_for_reward[ignored_trafo_idx]
        assert not trafo_masks_ignored.trafo_disconnectable[ignored_trafo_idx]
        assert not trafo_masks_ignored.trafo_pst_linear[ignored_trafo_idx]

    if ignored_tie_id is not None:
        ignored_tie_idx = tie_df.index.get_loc(ignored_tie_id)
        assert not tie_and_dangling_masks_ignored.tie_line_for_nminus1[ignored_tie_idx]
        assert not tie_and_dangling_masks_ignored.tie_line_for_reward[ignored_tie_idx]
        assert not tie_and_dangling_masks_ignored.tie_line_tso_border[ignored_tie_idx]

    if ignored_dangling_id is not None:
        ignored_dangling_idx = dangling_df.index.get_loc(ignored_dangling_id)
        assert not tie_and_dangling_masks_ignored.boundary_line_for_nminus1[ignored_dangling_idx]

    if ignored_generator_id is not None:
        ignored_generator_idx = generator_df.index.get_loc(ignored_generator_id)
        assert not generation_and_load_masks_ignored.generator_for_nminus1[ignored_generator_idx]

    if ignored_load_id is not None:
        ignored_load_idx = load_df.index.get_loc(ignored_load_id)
        assert not generation_and_load_masks_ignored.load_for_nminus1[ignored_load_idx]

    if ignored_switch_id is not None:
        ignored_switch_idx = switch_df.index.get_loc(ignored_switch_id)
        assert not switch_masks_ignored.switch_for_nminus1[ignored_switch_idx]
        assert not switch_masks_ignored.switch_for_reward[ignored_switch_idx]


def test_update_load_and_generation_masks(ucte_file_with_border, ucte_importer_parameters: UcteImporterParameters):
    network = pypowsybl.network.load(ucte_file_with_border)
    default_masks = powsybl_masks.create_default_network_masks(network)
    network_masks = powsybl_masks.update_load_and_generation_masks(
        default_masks, network, ucte_importer_parameters, blacklisted_ids=[]
    )
    assert np.array_equal(
        network_masks.generator_for_nminus1,
        np.array([False, False, False, False, True, False]),
    )
    assert np.array_equal(network_masks.load_for_nminus1, np.array([False, True, False, False, False]))


def test_update_bus_masks(ucte_file_with_border, ucte_importer_parameters: UcteImporterParameters):
    importer_parameters = deepcopy(ucte_importer_parameters)
    network = pypowsybl.network.load(ucte_file_with_border)
    default_masks = powsybl_masks.create_default_network_masks(network)

    network_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, blacklisted_ids=[])

    expected_bus_mask = np.zeros(17, dtype=bool)
    expected_bus_mask[3] = True
    assert np.array_equal(network_masks.relevant_subs, expected_bus_mask)

    # test select_station_grid_model_id_list
    importer_parameters.select_by_voltage_level_id_list = list(network.get_voltage_levels().index)
    updated_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, blacklisted_ids=[])
    assert np.array_equal(updated_masks.relevant_subs, network_masks.relevant_subs)
    importer_parameters.select_by_voltage_level_id_list = ["D8SU1_1"]
    updated_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, blacklisted_ids=[])
    assert np.array_equal(updated_masks.relevant_subs, network_masks.relevant_subs)
    importer_parameters.select_by_voltage_level_id_list = ["D8SU1_2"]
    updated_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, blacklisted_ids=[])
    assert not (updated_masks.relevant_subs).any()

    # test independent of area codes
    importer_parameters.area_settings.control_area = ["D2"]
    importer_parameters.area_settings.cutoff_voltage = 1000
    importer_parameters.select_by_voltage_level_id_list = ["D8SU1_1"]
    updated_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, blacklisted_ids=[])
    assert np.array_equal(updated_masks.relevant_subs, network_masks.relevant_subs)

    # test blacklisted ids
    importer_parameters.area_settings.control_area = ["D8"]
    importer_parameters.area_settings.cutoff_voltage = 0
    importer_parameters.select_by_voltage_level_id_list = []
    updated_masks = powsybl_masks.update_bus_masks(
        default_masks,
        network,
        importer_parameters,
        blacklisted_ids=["D8SU1_1"],
    )
    expected_bus_mask[3] = False
    assert np.array_equal(updated_masks.relevant_subs, expected_bus_mask)


def test_update_bus_masks_node_breaker_select_station(basic_node_breaker_network_powsybl_grid: Network):
    network = basic_node_breaker_network_powsybl_grid
    importer_parameters = CgmesImporterParameters(
        grid_model_file=Path("cgmes_file.zip"),
        data_folder="data_folder",
        area_settings=AreaSettings(cutoff_voltage=220, control_area=["BE"], view_area=["BE"], nminus1_area=["BE"]),
    )
    default_masks = powsybl_masks.create_default_network_masks(network)
    updated_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, blacklisted_ids=[])

    expected_bus_mask = np.array([True, True, True, False, False])
    assert np.array_equal(updated_masks.relevant_subs, expected_bus_mask)

    # make sure the slack is removed from relevant subs
    lf_result, *_ = pypowsybl.loadflow.run_dc(network)
    network_masks = powsybl_masks.make_masks(
        network=network,
        slack_id=lf_result.reference_bus_id,
        importer_parameters=importer_parameters,
        blacklisted_ids=[],
    )
    expected_bus_mask_no_slack = np.array([False, True, True, False, False])
    assert np.array_equal(network_masks.relevant_subs, expected_bus_mask_no_slack)

    importer_parameters.select_by_voltage_level_id_list = list(network.get_voltage_levels().index)
    updated_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, blacklisted_ids=[])
    assert np.array_equal(updated_masks.relevant_subs, expected_bus_mask)

    importer_parameters.select_by_voltage_level_id_list = list(network.get_voltage_levels().index)[:2]
    updated_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, blacklisted_ids=[])
    expected_bus_mask = np.array([True, True, False, False, False])
    assert np.array_equal(updated_masks.relevant_subs, expected_bus_mask)

    # make sure the slack is removed from relevant subs
    lf_result, *_ = pypowsybl.loadflow.run_dc(network)

    network_masks = powsybl_masks.make_masks(
        network=network,
        slack_id=lf_result.reference_bus_id,
        importer_parameters=importer_parameters,
        blacklisted_ids=[],
    )
    expected_bus_mask_no_slack = np.array([False, True, False, False, False])
    assert np.array_equal(network_masks.relevant_subs, expected_bus_mask_no_slack)

    # test independent of area codes
    importer_parameters.area_settings.control_area = ["FR"]
    importer_parameters.area_settings.cutoff_voltage = 1000
    importer_parameters.select_by_voltage_level_id_list = list(network.get_voltage_levels().index)[:2]
    updated_masks = powsybl_masks.update_bus_masks(default_masks, network, importer_parameters, blacklisted_ids=[])
    expected_bus_mask = np.array([True, True, False, False, False])
    assert np.array_equal(updated_masks.relevant_subs, expected_bus_mask)

    updated_masks = powsybl_masks.update_bus_masks(
        default_masks,
        network,
        importer_parameters,
        blacklisted_ids=[network.get_buses().iloc[0]["name"][:-2]],
    )
    expected_bus_mask_blacklisted = np.array([False, True, False, False, False])
    assert np.array_equal(updated_masks.relevant_subs, expected_bus_mask_blacklisted)


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
        network_masks.trafo_pst_linear,
        np.array([False, False, False, False, False, False]),
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
    lf_result, *_ = pypowsybl.loadflow.run_dc(network)
    masks = powsybl_masks.make_masks(
        network=network, slack_id=lf_result.reference_bus_id, importer_parameters=ucte_importer_parameters
    )
    assert powsybl_masks.validate_network_masks(masks, default_masks)


def test_make_masks_node_breaker(
    basic_node_breaker_network_powsybl_not_disconnectable, cgmes_importer_parameters: CgmesImporterParameters
):
    network = basic_node_breaker_network_powsybl_not_disconnectable
    default_masks = powsybl_masks.create_default_network_masks(network)
    lf_result, *_ = pypowsybl.loadflow.run_dc(network)
    masks = powsybl_masks.make_masks(
        network=network, slack_id=lf_result.reference_bus_id, importer_parameters=cgmes_importer_parameters
    )
    assert powsybl_masks.validate_network_masks(masks, default_masks)


def test_make_masks_node_breaker_with_ignore_file(
    basic_node_breaker_network_powsybl_not_disconnectable: Network,
    cgmes_importer_parameters: CgmesImporterParameters,
    tmp_path: Path,
):
    network = basic_node_breaker_network_powsybl_not_disconnectable
    importer_parameters = deepcopy(cgmes_importer_parameters)
    importer_parameters.data_folder = tmp_path
    default_masks = powsybl_masks.create_default_network_masks(network)

    lf_result, *_ = pypowsybl.loadflow.run_dc(network)
    baseline_masks = powsybl_masks.make_masks(
        network=network,
        slack_id=lf_result.reference_bus_id,
        importer_parameters=importer_parameters,
        blacklisted_ids=[],
    )
    relevant_sub_idx = np.flatnonzero(baseline_masks.relevant_subs)[0]
    ignored_station_id = network.get_buses().iloc[relevant_sub_idx]["name"][:-2]

    ignore_list_file = tmp_path / "ignore_list.csv"
    ignore_list_file.write_text(f"grid_model_id;reason\n{ignored_station_id};station\n")
    importer_parameters.ignore_list_file = ignore_list_file

    statistics = PreProcessingStatistics(
        id_lists={},
        import_result=ImportResult(data_folder=tmp_path),
        border_current={},
        network_changes={},
        import_parameter=importer_parameters,
    )
    statistics = network_analysis.apply_cb_lists_cgmes(
        statistics=statistics,
        white_list_file=None,
        ignore_list_file=ignore_list_file,
        filesystem=LocalFileSystem(),
    )

    ignored_masks = preprocessing.get_network_masks(
        network=network,
        slack_id=lf_result.reference_bus_id,
        importer_parameters=importer_parameters,
        statistics=statistics,
        filesystem=LocalFileSystem(),
    )

    assert statistics.id_lists["black_list"] == [ignored_station_id]
    assert baseline_masks.relevant_subs[relevant_sub_idx]
    assert not ignored_masks.relevant_subs[relevant_sub_idx]
    assert powsybl_masks.validate_network_masks(ignored_masks, default_masks)


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
    assert np.array_equal(network_masks.boundary_line_for_nminus1, np.array([False, False, True, False, False]))


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
    assert powsybl_masks.validate_network_masks.__wrapped__(ucte_importer_parameters, default_masks) is False

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


def test_build_pst_group_labels_groups_parallel_psts():
    """Parallel PSTs (same bus pair, voltage and tap-changer parameters) share a group label."""
    net = parallel_pst_example()
    trafos = net.get_2_windings_transformers(attributes=["bus1_id", "bus2_id", "voltage_level1_id", "voltage_level2_id"])
    control_area_hv_trafo_mask = trafos.index.isin(net.get_phase_tap_changers().index)

    trafo_has_pst_tap, trafo_pst_linear, pst_group_labels = powsybl_masks.filter_and_group_linear_psts(
        network=net, trafos_df=trafos, control_area_hv_trafo_mask=control_area_hv_trafo_mask
    )

    assert trafo_has_pst_tap.sum() == 3
    assert trafo_pst_linear.sum() == 3
    label_by_id = dict(zip(trafos.index, pst_group_labels, strict=True))
    # PST1 and PST2 connect the same bus pair with identical tap-changer parameters -> same group.
    assert label_by_id["PST1"] == label_by_id["PST2"]
    # PST3 connects a different bus pair with a different tap range -> its own group.
    assert label_by_id["PST3"] >= 0
    assert label_by_id["PST3"] != label_by_id["PST1"]


def test_build_pst_group_labels_marks_non_controllable_as_ungrouped():
    """Trafos that are not controllable PSTs keep the -1 sentinel and are never grouped."""
    net = parallel_pst_example()
    trafos = net.get_2_windings_transformers(attributes=["bus1_id", "bus2_id", "voltage_level1_id", "voltage_level2_id"])
    # Only PST1 is controllable; the parallel PST2 and the distinct PST3 are excluded.
    control_area_hv_trafo_mask = np.asarray(trafos.index == "PST1")

    trafo_has_pst_tap, trafo_pst_linear, pst_group_labels = powsybl_masks.filter_and_group_linear_psts(
        network=net, trafos_df=trafos, control_area_hv_trafo_mask=control_area_hv_trafo_mask
    )

    assert trafo_has_pst_tap.sum() == 1
    assert trafo_pst_linear.sum() == 1
    label_by_id = dict(zip(trafos.index, pst_group_labels, strict=True))
    assert label_by_id["PST1"] >= 0
    assert label_by_id["PST2"] == -1
    assert label_by_id["PST3"] == -1


@pytest.mark.parametrize(
    ("linear_pst", "split_pst_station", "expected_group_count", "expected_grouped_pairs"),
    [
        (
            [True, True, True, True],
            False,
            1,
            [("PST_1_group_1", "PST_2_group_1"), ("PST_3_group_2", "PST_4_group_2")],
        ),
        (
            [True, True, True, True],
            True,
            2,
            [("PST_1_group_1", "PST_3_group_2"), ("PST_2_group_1", "PST_4_group_2")],
        ),
        (
            [True, False, True, False],
            False,
            2,
            [("PST_1_group_1", "PST_3_group_2"), ("PST_2_group_1", "PST_4_group_2")],
        ),
    ],
)
def test_grouped_pst_grid_importer_masks_include_pst_groups(
    linear_pst: list[bool],
    split_pst_station: bool,
    expected_group_count: int,
    expected_grouped_pairs: list[tuple[str, str]],
    cgmes_importer_parameters: CgmesImporterParameters,
) -> None:
    net = grouped_pst_grid_example(linear_pst=linear_pst)
    if split_pst_station:
        net.open_switch("VL2_BREAKER#0")
    trafos = net.get_2_windings_transformers(attributes=[])
    importer_parameters = deepcopy(cgmes_importer_parameters)
    importer_parameters.area_settings.control_area = ["BE"]
    importer_parameters.area_settings.view_area = ["BE"]
    importer_parameters.area_settings.nminus1_area = ["BE"]
    importer_parameters.area_settings.cutoff_voltage = 220

    network_masks = powsybl_masks.make_masks(
        network=net,
        slack_id="VL1_0",
        importer_parameters=importer_parameters,
        blacklisted_ids=[],
    )
    label_by_id = dict(zip(trafos.index, network_masks.pst_group_masks, strict=True))

    assert powsybl_masks.validate_network_masks(network_masks, powsybl_masks.create_default_network_masks(net))
    assert np.array_equal(network_masks.trafo_pst_controllable, np.ones(len(trafos), dtype=bool))
    assert set(label_by_id) == {"PST_1_group_1", "PST_2_group_1", "PST_3_group_2", "PST_4_group_2"}
    assert len(set(network_masks.pst_group_masks)) == expected_group_count
    for first_pst_id, second_pst_id in expected_grouped_pairs:
        assert label_by_id[first_pst_id] == label_by_id[second_pst_id]


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
    voltage_level_prefix = ["S"]
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
    voltage_level_prefix = ["S"]
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
    lf_result, *_ = pypowsybl.loadflow.run_dc(network)
    result = powsybl_masks.remove_slack_from_relevant_subs(network_masks, network, slack_id=lf_result.reference_bus_id)
    expected = np.array(
        [True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True, True]
    )
    assert np.array_equal(result.relevant_subs, expected)


def test_update_masks_contingency_list_file(tmp_path, ucte_file_with_border, ucte_importer_parameters):
    network = pypowsybl.network.load(ucte_file_with_border)
    lines = network.get_lines()
    trafos = network.get_2_windings_transformers()
    dangling_lines = network.get_boundary_lines()
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
    assert np.array_equal(updated_masks.boundary_line_for_nminus1, contingency_dangling), (
        "Boundary line for n-1 mask not updated correctly"
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
