# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import tempfile
from dataclasses import asdict
from pathlib import Path

import logbook
import numpy as np
import pandapower as pp
import pytest
from pandas import Index
from toop_engine_grid_helpers.pandapower.pandapower_import_helpers import fuse_closed_switches_fast
from toop_engine_importer.pandapower_import.pp_masks import (
    NetworkMasks,
    count_assets,
    count_assets_in_substation,
    count_branches_at_buses,
    count_busbar_coupler_at_station,
    count_busbars_at_station,
    create_default_network_masks,
    get_relevant_subs,
    make_pp_masks,
    mask_min_branches_per_station,
    mask_min_busbar_coupler,
    mask_min_busbar_per_station,
    save_masks_to_files,
    save_preprocessing,
    validate_network_masks,
)
from toop_engine_interfaces.folder_structure import (
    NETWORK_MASK_NAMES,
    PREPROCESSING_PATHS,
)

logger = logbook.Logger(__name__)


def test_make_pp_masks(pp_network_w_switches):
    net = pp_network_w_switches
    masks = make_pp_masks(net)
    assert isinstance(masks, NetworkMasks)
    assert validate_network_masks(net, masks)


def test_make_pp_masks_foreign_id_column(pp_network_w_switches):
    net = pp_network_w_switches
    net.bus.loc[0:15, "zone"] = "region1"

    # test base function
    masks = make_pp_masks(
        net,
        substation_column="substat",
        region="region1",
        min_branches_per_station=0,
        min_busbars_per_substation=0,
        min_busbar_coupler_per_station=0,
        voltage_level=0,
    )
    assert isinstance(masks, NetworkMasks)
    assert validate_network_masks(net, masks)
    assert all(masks.relevant_subs[:16])
    assert all(~masks.relevant_subs[16:])

    # set region
    net.bus.loc[:, "zone"] = "region1"
    # set substation for all buses
    for i in range(33, net.bus.index.size):
        net.bus.loc[i, "substat"] = "substation" + str(i)
    masks = make_pp_masks(
        net,
        substation_column="substat",
        region="region1",
        min_branches_per_station=0,
        min_busbars_per_substation=0,
        min_busbar_coupler_per_station=0,
        voltage_level=0,
    )
    assert all(masks.relevant_subs)
    assert all(masks.line_for_reward)

    # set fid for all elements
    for element, column in pp.element_bus_tuples():
        net[element].loc[0:2, "fid"] = net[element].index[0:3]
    masks = make_pp_masks(
        net,
        substation_column="substat",
        region="region1",
        min_branches_per_station=0,
        min_busbars_per_substation=0,
        min_busbar_coupler_per_station=0,
        voltage_level=0,
        foreign_id_column="fid",
    )
    assert all(masks.line_for_reward[:3])
    assert all(~masks.line_for_reward[3:])


def test_get_relevant_subs(pp_network_w_switches):
    net = pp_network_w_switches

    net.bus["zone"] = "region1"
    exclude_stations = [""]
    # test base function
    res = get_relevant_subs(
        network=net,
        substation_column="substat",
        region="region1",
        min_branches_per_station=0,
        min_busbars_per_substation=1,
        min_busbar_coupler_per_station=0,
        voltage_level=380,
        exclude_stations=exclude_stations,
    )
    assert np.all(res[:16])
    assert np.all(~res[16:])
    assert isinstance(res, np.ndarray)
    # test voltage_level
    res = get_relevant_subs(
        network=net,
        substation_column="substat",
        region="region1",
        min_branches_per_station=0,
        min_busbars_per_substation=1,
        min_busbar_coupler_per_station=0,
        voltage_level=110,
        exclude_stations=exclude_stations,
    )
    assert all(res[:32])
    assert all(~res[32:])
    assert isinstance(res, np.ndarray)
    # test min_busbar_coupler_per_station
    res = get_relevant_subs(
        network=net,
        substation_column="substat",
        region="region1",
        min_branches_per_station=0,
        min_busbars_per_substation=1,
        min_busbar_coupler_per_station=1,
        voltage_level=110,
        exclude_stations=exclude_stations,
    )
    assert np.all(res[:16])
    assert np.all(~res[16:])
    assert isinstance(res, np.ndarray)
    # test min_busbars_per_substation
    res = get_relevant_subs(
        network=net,
        substation_column="substat",
        region="region1",
        min_branches_per_station=0,
        min_busbars_per_substation=1,
        min_busbar_coupler_per_station=0,
        voltage_level=110,
        exclude_stations=exclude_stations,
    )
    assert all(res[:32])
    assert all(~res[32:])
    assert isinstance(res, np.ndarray)
    res = get_relevant_subs(
        network=net,
        substation_column="substat",
        region="region1",
        min_branches_per_station=0,
        min_busbars_per_substation=2,
        min_busbar_coupler_per_station=0,
        voltage_level=110,
        exclude_stations=exclude_stations,
    )
    assert np.all(res[:16])
    assert np.all(~res[16:])
    assert isinstance(res, np.ndarray)
    # test min_branches_per_station
    res = get_relevant_subs(
        network=net,
        substation_column="substat",
        region="region1",
        min_branches_per_station=2,
        min_busbars_per_substation=1,
        min_busbar_coupler_per_station=0,
        voltage_level=110,
        exclude_stations=exclude_stations,
    )
    assert all(~res[:16])
    assert all(res[16:32])
    assert all(~res[32:])
    assert isinstance(res, np.ndarray)
    # test exclude_stations
    res = get_relevant_subs(
        network=net,
        substation_column="substat",
        region="region1",
        min_branches_per_station=1,
        min_busbars_per_substation=1,
        min_busbar_coupler_per_station=0,
        voltage_level=110,
        exclude_stations=["Double Busbar 1"],
    )
    assert all(~res[:16])
    assert all(res[16:32])
    assert all(~res[32:])
    assert isinstance(res, np.ndarray)
    res = get_relevant_subs(
        network=net,
        substation_column="substat",
        region="region1",
        min_branches_per_station=1,
        min_busbars_per_substation=1,
        min_busbar_coupler_per_station=0,
        voltage_level=0,
    )
    assert all(res[:32])
    assert all(~res[32:])
    assert isinstance(res, np.ndarray)
    # test min_busbars_per_substation and exclude_stations
    res = get_relevant_subs(
        network=net,
        substation_column="substat",
        region="region1",
        min_branches_per_station=0,
        min_busbars_per_substation=0,
        min_busbar_coupler_per_station=0,
        voltage_level=0,
    )
    assert all(res)
    assert isinstance(res, np.ndarray)
    # test region
    res = get_relevant_subs(
        network=net,
        substation_column="substat",
        region="region2",
        min_branches_per_station=1,
        min_busbars_per_substation=1,
        min_busbar_coupler_per_station=0,
        voltage_level=0,
    )
    assert all(~res)
    assert isinstance(res, np.ndarray)
    net.bus.loc[0:15, "zone"] = "region2"
    res = get_relevant_subs(
        network=net,
        substation_column="substat",
        region="region2",
        min_branches_per_station=1,
        min_busbars_per_substation=1,
        min_busbar_coupler_per_station=0,
        voltage_level=0,
    )
    assert np.all(res[:16])
    assert np.all(~res[16:])


def test_get_relevant_subs_fused_network(pp_network_w_switches):
    net = pp_network_w_switches
    net.bus["zone"] = "region1"
    exclude_stations = [""]
    # test base function
    fuse_closed_switches_fast(net)
    # test reduced network - min_busbar_coupler_per_station
    res = get_relevant_subs(
        network=net,
        substation_column="substat",
        region="region1",
        min_branches_per_station=0,
        min_busbars_per_substation=1,
        min_busbar_coupler_per_station=1,
        voltage_level=380,
        exclude_stations=exclude_stations,
    )
    assert isinstance(res, np.ndarray)
    assert all(~res)
    # test min_busbar_coupler_per_station
    res = get_relevant_subs(
        network=net,
        substation_column="substat",
        region="region1",
        min_branches_per_station=0,
        min_busbars_per_substation=1,
        min_busbar_coupler_per_station=0,
        voltage_level=380,
        exclude_stations=exclude_stations,
    )
    assert isinstance(res, np.ndarray)
    assert res[0]
    assert all(~res[1:])
    # test voltage_level
    res = get_relevant_subs(
        network=net,
        substation_column="substat",
        region="region1",
        min_branches_per_station=0,
        min_busbars_per_substation=1,
        min_busbar_coupler_per_station=0,
        voltage_level=110,
        exclude_stations=exclude_stations,
    )
    assert isinstance(res, np.ndarray)
    assert all(res[:2])
    assert all(~res[2:])
    # test min_branches_per_station
    res = get_relevant_subs(
        network=net,
        substation_column="substat",
        region="region1",
        min_branches_per_station=1,
        min_busbars_per_substation=1,
        min_busbar_coupler_per_station=0,
        voltage_level=110,
        exclude_stations=exclude_stations,
    )
    assert isinstance(res, np.ndarray)
    assert all(res[:2])
    assert all(~res[2:])
    res = get_relevant_subs(
        network=net,
        substation_column="substat",
        region="region1",
        min_branches_per_station=2,
        min_busbars_per_substation=1,
        min_busbar_coupler_per_station=0,
        voltage_level=110,
        exclude_stations=exclude_stations,
    )
    assert isinstance(res, np.ndarray)
    assert not res[0]
    assert res[1]
    assert all(~res[2:])
    # test substation_column not set
    res = get_relevant_subs(
        network=net,
        substation_column="N/A",
        region="region1",
        min_branches_per_station=0,
        min_busbars_per_substation=1,
        min_busbar_coupler_per_station=0,
        voltage_level=50,
        exclude_stations=exclude_stations,
    )
    assert all(res[:6])
    assert all(~res[6:])


def test_station_mask_min_branches(pp_network_w_switches):
    net = pp_network_w_switches
    res = mask_min_branches_per_station(net, substation_column="substat", min_branches_per_station=0)
    assert all(res[:32])
    assert all(~res[32:])
    res = mask_min_branches_per_station(net, substation_column="substat", min_branches_per_station=1)
    assert all(res[:32])
    assert all(~res[32:])
    res = mask_min_branches_per_station(net, substation_column="substat", min_branches_per_station=2)
    assert all(~res[:16])
    assert all(res[16:32])
    assert all(~res[32:])
    res2 = mask_min_branches_per_station(net, substation_column="substat", min_branches_per_station=2)
    assert all(res == res2)
    res2 = mask_min_branches_per_station(net, substation_column="substat", min_branches_per_station=3)
    assert all(res == res2)
    res = mask_min_branches_per_station(net, substation_column="substat", min_branches_per_station=4)
    assert all(~res)
    assert isinstance(res, np.ndarray)


def test_count_assets_in_substation(pp_network_w_switches):
    net = pp_network_w_switches
    res = count_assets_in_substation(network=net, substation_column="substat", station_name="Single Busbar")
    assert res == 3
    res = count_assets_in_substation(network=net, substation_column="substat", station_name="Double Busbar 1")
    assert res == 1
    # test all stations except Single Busbar and Double Busbar 1
    res = count_assets_in_substation(network=net, substation_column="substat", station_name="")
    assert res == 26  # onyl trafo 0 is missing -> 24 lines + 1 trafo + 1 trafo3w = 26
    assert isinstance(res, int)


def test_count_assets(pp_network_w_switches):
    net = pp_network_w_switches
    bus_index = [el for el in range(0, 16)]
    branches = pp.toolbox.get_connected_elements_dict(net, bus_index, include_empty_lists=True)

    res = count_assets(branches=branches)
    assert res == 1
    bus_index = [el for el in range(16, 32)]
    branches = pp.toolbox.get_connected_elements_dict(net, bus_index, include_empty_lists=True)

    res = count_assets(branches=branches)
    assert res == 3
    bus_index = net.bus.index
    branches = pp.toolbox.get_connected_elements_dict(net, bus_index, include_empty_lists=True)

    res = count_assets(branches=branches)
    # one line is disconnected
    assert res == len(net.line) - 1 + len(net.trafo) + len(net.trafo3w)

    res = count_assets(branches=branches, include_branches=False)
    assert res == 0

    res = count_assets(branches=branches, include_branches=False, include_gen=True)
    assert res == len(net.gen) + len(net.sgen)

    res = count_assets(branches=branches, include_branches=False, include_gen=False, include_load=True)
    assert res == len(net.load)

    res = count_assets(
        branches=branches,
        include_branches=False,
        include_gen=False,
        include_load=False,
        include_impedance=True,
    )
    assert res == len(net.impedance)
    assert isinstance(res, int)


def test_count_branches_at_buses():
    # Create a simple pandapower network
    net = pp.create_empty_network()

    # Create buses
    bus1 = pp.create_bus(net, vn_kv=110)
    bus2 = pp.create_bus(net, vn_kv=110)
    bus3 = pp.create_bus(net, vn_kv=110)

    # Create lines
    pp.create_line(net, bus1, bus2, length_km=1.0, std_type="NAYY 4x50 SE")
    pp.create_line(net, bus2, bus3, length_km=1.0, std_type="NAYY 4x50 SE")

    # Test with all buses
    buses = Index([bus1, bus2, bus3])
    expected_counts = np.array([1, 2, 1])
    counts = count_branches_at_buses(net, buses)
    assert np.array_equal(counts, expected_counts), f"Expected {expected_counts}, but got {counts}"

    # Create transformers
    pp.create_transformer(net, bus1, bus3, std_type="25 MVA 110/20 kV")
    expected_counts = np.array([2, 2, 2])
    counts = count_branches_at_buses(net, buses)
    assert np.array_equal(counts, expected_counts), f"Expected {expected_counts}, but got {counts}"

    # Test with a subset of buses
    buses = Index([bus1, bus2])
    expected_counts = np.array([2, 2])
    counts = count_branches_at_buses(net, buses)
    assert np.array_equal(counts, expected_counts), f"Expected {expected_counts}, but got {counts}"

    # Test with a single bus
    buses = Index([bus1])
    expected_counts = np.array([2])
    counts = count_branches_at_buses(net, buses)
    assert np.array_equal(counts, expected_counts), f"Expected {expected_counts}, but got {counts}"

    # Test with no buses
    buses = Index([])
    expected_counts = np.array([])
    counts = count_branches_at_buses(net, buses)
    assert np.array_equal(counts, expected_counts), f"Expected {expected_counts}, but got {counts}"
    assert isinstance(counts, np.ndarray)


def test_mask_min_busbar_per_station(pp_network_w_switches):
    net = pp_network_w_switches
    res = mask_min_busbar_per_station(net, substation_column="substat", min_branches_per_station=0)
    assert all(res[:32])
    assert all(~res[32:])
    res = mask_min_busbar_per_station(net, substation_column="substat", min_branches_per_station=1)
    assert all(res[:32])
    assert all(~res[32:])
    res = mask_min_busbar_per_station(net, substation_column="substat", min_branches_per_station=2)
    assert np.all(res[:16])
    assert np.all(~res[16:])
    res = mask_min_busbar_per_station(net, substation_column="substat", min_branches_per_station=3)
    assert all(~res)
    net.bus.loc[2, "type"] = "b"
    res = mask_min_busbar_per_station(net, substation_column="substat", min_branches_per_station=3)
    assert np.all(res[:16])
    assert np.all(~res[16:])
    assert isinstance(res, np.ndarray)


def test_count_busbars_at_station(pp_network_w_switches):
    net = pp_network_w_switches
    res = count_busbars_at_station(net, substation_column="substat", station_name="Single Busbar")
    assert res == 1
    res = count_busbars_at_station(net, substation_column="substat", station_name="Double Busbar 1")
    assert res == 2
    assert isinstance(res, int)


def test_mask_min_busbar_coupler(pp_network_w_switches, pp_network_w_switches_parallel_coupler):
    net = pp_network_w_switches
    net2 = pp_network_w_switches_parallel_coupler

    res = mask_min_busbar_coupler(net2, min_busbar_coupler_per_station=0, substation_column="substat")
    assert isinstance(res, np.ndarray)
    assert all(res[:32])
    assert all(~res[32:])
    res = mask_min_busbar_coupler(net2, min_busbar_coupler_per_station=1, substation_column="substat")
    assert np.all(res[:16])
    assert np.all(~res[16:])
    res = mask_min_busbar_coupler(net2, min_busbar_coupler_per_station=2, substation_column="substat")
    assert np.all(res[:16])
    assert np.all(~res[16:])
    res = mask_min_busbar_coupler(net2, min_busbar_coupler_per_station=3, substation_column="substat")
    assert isinstance(res, np.ndarray)
    assert all(~res)

    res = mask_min_busbar_coupler(net, min_busbar_coupler_per_station=1, substation_column="substat")
    assert np.all(res[:16])
    assert np.all(~res[16:])


def test_count_busbar_coupler_at_station(pp_network_w_switches, pp_network_w_switches_parallel_coupler):
    net = pp_network_w_switches
    net2 = pp_network_w_switches_parallel_coupler
    res = count_busbar_coupler_at_station(net, substation_column="substat", station_name="Double Busbar 1")
    assert res == 1
    res = count_busbar_coupler_at_station(net, substation_column="substat", station_name="Single Busbar")
    assert res == 0
    res = count_busbar_coupler_at_station(net2, substation_column="substat", station_name="Double Busbar 1")
    assert res == 2
    assert isinstance(res, int)


def test_create_default_network_masks(pp_network_w_switches):
    net = pp_network_w_switches
    masks = create_default_network_masks(net)
    assert len(masks.__annotations__) == 16, (
        "test has been created with 16 annotations -> if this changes, the test has to be adapted"
    )

    # Check that all masks are created correctly and are of the correct type and shape
    assert isinstance(masks, NetworkMasks)
    assert masks.relevant_subs.shape == (len(net.bus),)
    assert masks.line_for_nminus1.shape == (len(net.line),)
    assert masks.line_for_reward.shape == (len(net.line),)
    assert masks.line_overload_weight.shape == (len(net.line),)
    assert masks.line_disconnectable.shape == (len(net.line),)
    assert masks.trafo_for_nminus1.shape == (len(net.trafo),)
    assert masks.trafo_for_reward.shape == (len(net.trafo),)
    assert masks.trafo_overload_weight.shape == (len(net.trafo),)
    assert masks.trafo_disconnectable.shape == (len(net.trafo),)
    assert masks.trafo3w_for_nminus1.shape == (len(net.trafo3w),)
    assert masks.trafo3w_for_reward.shape == (len(net.trafo3w),)
    assert masks.trafo3w_overload_weight.shape == (len(net.trafo3w),)
    assert masks.trafo3w_disconnectable.shape == (len(net.trafo3w),)
    assert masks.generator_for_nminus1.shape == (len(net.gen),)
    assert masks.sgen_for_nminus1.shape == (len(net.sgen),)
    assert masks.load_for_nminus1.shape == (len(net.load),)

    # Check that all masks are initialized to the correct default values
    assert np.all(~masks.relevant_subs)
    assert np.all(~masks.line_for_nminus1)
    assert np.all(~masks.line_for_reward)
    assert np.all(masks.line_overload_weight == 0.0)
    assert np.all(~masks.line_disconnectable)
    assert np.all(~masks.trafo_for_nminus1)
    assert np.all(~masks.trafo_for_reward)
    assert np.all(masks.trafo_overload_weight == 0.0)
    assert np.all(~masks.trafo_disconnectable)
    assert np.all(~masks.trafo3w_for_nminus1)
    assert np.all(~masks.trafo3w_for_reward)
    assert np.all(masks.trafo3w_overload_weight == 0.0)
    assert np.all(~masks.trafo3w_disconnectable)
    assert np.all(~masks.generator_for_nminus1)
    assert np.all(~masks.sgen_for_nminus1)
    assert np.all(~masks.load_for_nminus1)


def test_validate_network_masks(pp_network_w_switches):
    net = pp_network_w_switches
    masks = create_default_network_masks(net)
    assert validate_network_masks(net, masks)

    with logbook.handlers.TestHandler() as caplog:
        assert not validate_network_masks(net, 1)
        assert "network_masks are not of type NetworkMasks" in "".join(caplog.formatted_records)

    with logbook.handlers.TestHandler() as caplog:
        masks.relevant_subs = 1
        assert not validate_network_masks(net, masks)
        assert "Mask relevant_subs is not a numpy array" in "".join(caplog.formatted_records)

    masks = create_default_network_masks(net)
    with logbook.handlers.TestHandler() as caplog:
        masks.relevant_subs = np.array([1])
        assert not validate_network_masks(net, masks)
        assert "Shape of mask relevant_subs is not correct" in "".join(caplog.formatted_records)

    masks = create_default_network_masks(net)
    with logbook.handlers.TestHandler() as caplog:
        masks.relevant_subs = np.zeros(len(net.bus), dtype=int)
        assert not validate_network_masks(net, masks)
        assert "Dtype of mask relevant_subs is not correct" in "".join(caplog.formatted_records)


def test_save_masks_to_files(pp_network_w_switches):
    net = pp_network_w_switches
    masks = create_default_network_masks(net)

    with tempfile.TemporaryDirectory() as tmpdirname:
        data_folder = Path(tmpdirname)
        save_masks_to_files(masks, data_folder)

        masks_folder = data_folder / PREPROCESSING_PATHS["masks_path"]
        assert masks_folder.exists(), "Masks folder was not created."

        for mask_key in masks.__annotations__.keys():
            mask_file = masks_folder / NETWORK_MASK_NAMES[mask_key]
            assert mask_file.exists(), f"Mask file {mask_file} was not created."
            loaded_mask = np.load(mask_file)
            assert np.array_equal(loaded_mask, asdict(masks)[mask_key]), f"Mask {mask_key} was not saved correctly."


def test_save_preprocessing(pp_network_w_switches):
    net = pp_network_w_switches
    masks = create_default_network_masks(net)

    with tempfile.TemporaryDirectory() as tmpdirname:
        data_folder = Path(tmpdirname)
        save_preprocessing(data_folder, net, masks)

        # Check if the grid file was created
        grid_folder = data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"]
        assert grid_folder.exists(), "Grid file was not created."

        # Check if the masks were saved correctly
        masks_folder = data_folder / PREPROCESSING_PATHS["masks_path"]
        assert masks_folder.exists(), "Masks folder was not created."

        for mask_key in masks.__annotations__.keys():
            mask_file = masks_folder / NETWORK_MASK_NAMES[mask_key]
            assert mask_file.exists(), f"Mask file {mask_file} was not created."
            loaded_mask = np.load(mask_file)
            assert np.array_equal(loaded_mask, asdict(masks)[mask_key]), f"Mask {mask_key} was not saved correctly."


def test_save_preprocessing_invalid_masks(pp_network_w_switches, caplog):
    net = pp_network_w_switches
    masks = create_default_network_masks(net)
    masks.relevant_subs = np.array([1])  # Invalid mask for testing

    with tempfile.TemporaryDirectory() as tmpdirname:
        data_folder = Path(tmpdirname)
        with pytest.raises(RuntimeError, match="Network masks are not created correctly"):
            save_preprocessing(data_folder, net, masks)

        # Ensure no files were created
        grid_folder = data_folder / PREPROCESSING_PATHS["grid_file_path_pandapower"]
        assert not grid_folder.exists(), "Grid file should not be created for invalid masks."

        masks_folder = data_folder / PREPROCESSING_PATHS["masks_path"]
        assert not masks_folder.exists(), "Masks folder should not be created for invalid masks."
