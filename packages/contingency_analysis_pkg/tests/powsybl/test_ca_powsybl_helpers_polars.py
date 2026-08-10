# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import polars as pl
from toop_engine_contingency_analysis.pypowsybl import (
    add_name_column_polars,
    get_branch_results_polars,
    get_node_results_polars,
    get_va_diff_results_polars,
    update_basename_polars,
)
from toop_engine_interfaces.interface_helpers import get_empty_dataframe_from_model
from toop_engine_interfaces.loadflow_results import (
    BranchResultSchema,
    NodeResultSchema,
    VADiffResultSchema,
)
from toop_engine_interfaces.loadflow_results_polars import (
    BranchResultSchemaPolars,
    NodeResultSchemaPolars,
    VADiffResultSchemaPolars,
)


def test_get_va_diff_results():
    blank_va_diff_with_buses = pl.LazyFrame(
        {
            "contingency": ["contingency_1", "contingency_1", "contingency_2", "contingency_2"],
            "element": ["element_1", "element_2", "element_1", "element_2"],
            "bus_breaker_bus1_id": ["bus_1", "bus_2", "bus_2", "bus_2"],
            "bus_breaker_bus2_id": ["bus_2", "bus_1", "bus_1", "bus_1"],
        }
    )

    bus_results = pl.LazyFrame(
        {
            "contingency_id": ["contingency_1", "contingency_1", "contingency_2", "contingency_2"],
            "operator_strategy_id": ["", "", "", ""],
            "voltage_level_id": ["placeholder"] * 4,
            "bus_id": ["bus_1", "bus_2", "bus_1", "bus_2"],
            "v_mag": [10.0] * 4,
            "v_angle": [180.0, 0, 10, np.nan],
        }
    )
    bus_map = pl.LazyFrame({"id": ["bus_1", "bus_2"], "bus_breaker_bus_id": ["bus_1", "bus_2"]})
    outages = ["contingency_1", "contingency_2"]
    timestep = 0
    va_results = get_va_diff_results_polars(
        bus_results,
        outages,
        va_diff_with_buses=blank_va_diff_with_buses,
        bus_map=bus_map,
        timestep=timestep,
    )

    # make sure it is a lazy frame and the validation works for both lazy and df
    assert isinstance(va_results, pl.LazyFrame)
    VADiffResultSchemaPolars.validate(va_results)
    va_results = va_results.collect()
    VADiffResultSchemaPolars.validate(va_results)

    assert va_results.height == 4, "As all contingencies are considered, there should be 4 rows"

    outages = ["contingency_1"]
    va_results = get_va_diff_results_polars(
        bus_results,
        outages,
        va_diff_with_buses=blank_va_diff_with_buses,
        bus_map=bus_map,
        timestep=timestep,
    )
    assert isinstance(va_results, pl.LazyFrame)
    VADiffResultSchemaPolars.validate(va_results)
    va_results = va_results.collect()
    VADiffResultSchemaPolars.validate(va_results)
    assert va_results.height == 2, "As only the first contingency is considered, there should be 2 rows"

    # test empty input
    va_results = get_va_diff_results_polars(
        bus_results.limit(0),
        outages,
        va_diff_with_buses=blank_va_diff_with_buses,
        bus_map=bus_map,
        timestep=timestep,
    )
    assert isinstance(va_results, pl.LazyFrame)
    VADiffResultSchemaPolars.validate(va_results)
    va_results = va_results.collect()
    VADiffResultSchemaPolars.validate(va_results)
    assert va_results.height == 2, "Rows should be preserved even when voltage angles are unavailable"
    assert va_results["va_diff"].is_null().sum() == 2, "Missing voltage angles should produce null va_diff values"

    blank_va_diff_basecase = pl.LazyFrame(
        {
            "contingency": ["", ""],
            "element": ["element_1", "element_2"],
            "bus_breaker_bus1_id": ["bus_1", "bus_2"],
            "bus_breaker_bus2_id": ["bus_2", "bus_1"],
        }
    )
    bus_results_basecase = pl.LazyFrame(
        {
            "contingency_id": ["", ""],
            "operator_strategy_id": ["", ""],
            "voltage_level_id": ["placeholder", "placeholder"],
            "bus_id": ["bus_1", "bus_2"],
            "v_mag": [10.0, 10.0],
            "v_angle": [180.0, 0.0],
        }
    )
    va_results = get_va_diff_results_polars(
        bus_results_basecase,
        [],
        va_diff_with_buses=blank_va_diff_basecase,
        bus_map=bus_map,
        timestep=timestep,
    ).collect()
    VADiffResultSchemaPolars.validate(va_results)
    assert va_results.height == 2, "Basecase switch rows should be preserved when there are no outages"
    assert va_results.filter(pl.col("element") == "element_1")["va_diff"][0] == 180.0
    assert va_results.filter(pl.col("element") == "element_2")["va_diff"][0] == -180.0


def test_get_va_diff_results_all_null_bus_and_contingency_ids() -> None:
    blank_va_diff_basecase = pl.LazyFrame(
        {
            "contingency": ["", ""],
            "element": ["element_1", "element_2"],
            "bus_breaker_bus1_id": ["bus_1", "bus_2"],
            "bus_breaker_bus2_id": ["bus_2", "bus_1"],
        }
    )
    bus_results_basecase = pl.LazyFrame(
        {
            "contingency_id": [None, None],
            "operator_strategy_id": ["", ""],
            "voltage_level_id": ["placeholder", "placeholder"],
            "bus_id": [None, None],
            "v_mag": [None, None],
            "v_angle": [None, None],
        }
    )
    bus_map = pl.LazyFrame({"id": ["bus_1", "bus_2"], "bus_breaker_bus_id": ["bus_1", "bus_2"]})

    va_results = get_va_diff_results_polars(
        bus_results_basecase,
        [],
        va_diff_with_buses=blank_va_diff_basecase,
        bus_map=bus_map,
        timestep=0,
    ).collect()

    VADiffResultSchemaPolars.validate(va_results)
    assert va_results.height == 2
    assert va_results["va_diff"].is_null().sum() == 2


def test_get_node_results_polars_keeps_absolute_ac_voltage() -> None:
    bus_results = pl.LazyFrame(
        {
            "contingency_id": ["contingency_1", "contingency_1"],
            "operator_strategy_id": ["", ""],
            "voltage_level_id": ["VL_1", "VL_1"],
            "bus_id": ["bus_1", "bus_2"],
            "v_mag": [224.4, 224.8],
            "v_angle": [-1.2, -0.4],
        }
    )
    voltage_levels = pl.LazyFrame(
        {
            "id": ["VL_1"],
            "nominal_v": [225.0],
            "high_voltage_limit": [245.0],
            "low_voltage_limit": [205.0],
        }
    )
    monitored_buses = ["bus_1", "bus_2"]
    busmap = pl.LazyFrame({"id": monitored_buses, "bus_breaker_bus_id": monitored_buses})

    node_results = get_node_results_polars(
        bus_results,
        monitored_buses,
        busmap,
        voltage_levels,
        failed_outages=[],
        timestep=0,
        method="ac",
    ).collect()

    assert node_results.filter(pl.col("element") == "bus_1").select("vm").item() == 224.4
    assert node_results.filter(pl.col("element") == "bus_2").select("vm").item() == 224.8


def test_get_node_results_polars_computes_vm_basecase_deviation() -> None:
    bus_results = pl.LazyFrame(
        {
            "contingency_id": ["", "", "contingency_1", "contingency_1"],
            "operator_strategy_id": ["", "", "", ""],
            "voltage_level_id": ["VL_1", "VL_1", "VL_1", "VL_1"],
            "bus_id": ["bus_1", "bus_2", "bus_1", "bus_2"],
            "v_mag": [220.0, 225.0, 231.0, 213.75],
            "v_angle": [-1.0, -0.5, -1.2, -0.3],
        }
    )
    voltage_levels = pl.LazyFrame(
        {
            "id": ["VL_1"],
            "nominal_v": [225.0],
            "high_voltage_limit": [245.0],
            "low_voltage_limit": [205.0],
        }
    )
    monitored_buses = ["bus_1", "bus_2"]
    busmap = pl.LazyFrame({"id": monitored_buses, "bus_breaker_bus_id": monitored_buses})

    node_results = get_node_results_polars(
        bus_results,
        monitored_buses,
        busmap,
        voltage_levels,
        failed_outages=[],
        timestep=0,
        method="ac",
    ).collect()

    assert (
        node_results.filter((pl.col("contingency") == "") & (pl.col("element") == "bus_1"))
        .select("vm_basecase_deviation")
        .item()
        == 0.0
    )
    assert (
        node_results.filter((pl.col("contingency") == "contingency_1") & (pl.col("element") == "bus_1"))
        .select("vm_basecase_deviation")
        .item()
        == 5.0
    )
    assert (
        node_results.filter((pl.col("contingency") == "contingency_1") & (pl.col("element") == "bus_2"))
        .select("vm_basecase_deviation")
        .item()
        == 5.0
    )


def test_get_branch_results():
    ca_branch_results = pl.LazyFrame(
        {
            "branch_id": ["branch_1", "branch_2", "branch_1", "branch_2"],
            "contingency_id": ["cont_1", "cont_1", "cont_2", "cont_2"],
            "operator_strategy_id": ["placeholder"] * 4,
            "p1": [100.0, 200.0, 0.0, np.nan],
            "q1": [50.0, 100.0, 0.0, np.nan],
            "i1": [10.0, 20.0, 0.0, np.nan],
            "p2": [90.0, 190.0, 0.0, np.nan],
            "q2": [40.0, 90.0, 0.0, np.nan],
            "i2": [5.0, 10.0, 0.0, np.nan],
            "flow_transfer": [np.nan] * 4,
        }
    )

    three_winding_results = pl.LazyFrame(
        {
            "transformer_id": ["trafo_1", "trafo_2", "trafo_1", "trafo_2"],
            "contingency_id": ["cont_1", "cont_1", "cont_2", "cont_2"],
            "p1": [100.0, 200.0, 0.0, np.nan],
            "q1": [50.0, 100.0, 0.0, np.nan],
            "i1": [10.0, 20.0, 0.0, np.nan],
            "p2": [90.0, 190.0, 0.0, np.nan],
            "q2": [40.0, 90.0, 0.0, np.nan],
            "i2": [5.0, 10.0, 0.0, np.nan],
            "p3": [90.0, 190.0, 0.0, np.nan],
            "q3": [40.0, 90.0, 0.0, np.nan],
            "i3": [5.0, 10.0, 0.0, np.nan],
            "flow_transfer": [np.nan] * 4,
        }
    )
    monitored_branches = ["branch_1", "branch_2"]
    monitored_trafo3w = ["trafo_1", "trafo_2"]
    failed_outages = ["cont_3"]  # This wont show up in the results
    timestep = 0
    branch_limits = pl.LazyFrame(
        {
            "element_id": ["branch_1", "branch_1", "trafo_1"],
            "side": [1, 2, 1],
            "value": [100, 100, 100],
            "acceptable_duration": [-1, -1, -1],
        }
    )
    branch_results = get_branch_results_polars(
        ca_branch_results,
        three_winding_results,
        monitored_branches,
        monitored_trafo3w,
        failed_outages,
        timestep,
        branch_limits,
    )

    n_contingencies = 3  # 1 failed + 2 successful contingencies
    n_branches = 2  # 2 branches
    n_trafo3w = 2  # 2 three winding transformers
    n_expected_rows = n_contingencies * (n_branches * 2 + n_trafo3w * 3)

    # make sure the schema is correct for the lazy frame
    branch_results = BranchResultSchemaPolars.validate(branch_results)
    assert isinstance(branch_results, pl.LazyFrame), "The result should be a Polars LazyFrame"
    # collect lazy frame for testing
    branch_results = branch_results.collect()
    # make sure the schema is correct for the collected frame
    branch_results = BranchResultSchemaPolars.validate(branch_results)

    assert branch_results.height == n_expected_rows, (
        "The number of rows in the branch results should match the expected number"
    )
    assert all(branch_results["timestep"] == timestep), "All rows should have the same timestep"
    assert branch_results.select(pl.col("contingency").is_in(["cont_1", "cont_2", "cont_3"])).to_numpy().all(), (
        "All contingencies should be present in the results"
    )
    assert branch_results.select(pl.col("element").is_in(monitored_branches + monitored_trafo3w)).to_numpy().all(), (
        "All elements should be present in the results"
    )

    # assert no null values in df, expected nans instead of nulls
    assert not branch_results.select(pl.all().is_null()).to_numpy().any(), "There should be no null values in the results"

    # check that the values for the branches are correctly translated
    ca_branch_results_df = ca_branch_results.collect()
    branch_limits_df = branch_limits.collect()
    for row in ca_branch_results_df.iter_rows(named=True):
        contingency = row["contingency_id"]
        element = row["branch_id"]
        for side in [1, 2]:
            for value in ["p", "q", "i"]:
                original_value = row[f"{value}{side}"]
                result_value = (
                    branch_results.filter(
                        (pl.col("timestep") == timestep)
                        & (pl.col("contingency") == contingency)
                        & (pl.col("element") == element)
                        & (pl.col("side") == side)
                    )
                    .select(value)
                    .item()
                )
                assert result_value == original_value or (np.isnan(result_value) and np.isnan(original_value)), (
                    f"Power flow {value}{side} for {element} should match"
                )
            limit_row = branch_limits_df.filter((pl.col("element_id") == element) & (pl.col("side") == side))
            expected_loading = row[f"i{side}"] / limit_row.select("value").item() if limit_row.height > 0 else np.nan
            result_loading = (
                branch_results.filter(
                    (pl.col("timestep") == timestep)
                    & (pl.col("contingency") == contingency)
                    & (pl.col("element") == element)
                    & (pl.col("side") == side)
                )
                .select("loading")
                .item()
            )
            assert result_loading == expected_loading or (np.isnan(result_loading) and np.isnan(expected_loading)), (
                f"Loading for {element} should match"
            )

    # Check that the values for the three-winding transformers are correctly translated
    three_winding_results_df = three_winding_results.collect()
    for row in three_winding_results_df.iter_rows(named=True):
        contingency = row["contingency_id"]
        element = row["transformer_id"]
        for side in [1, 2, 3]:
            for value in ["p", "q", "i"]:
                original_value = row[f"{value}{side}"]
                result_value = (
                    branch_results.filter(
                        (pl.col("timestep") == timestep)
                        & (pl.col("contingency") == contingency)
                        & (pl.col("element") == element)
                        & (pl.col("side") == side)
                    )
                    .select(value)
                    .item()
                )
                assert result_value == original_value or (np.isnan(result_value) and np.isnan(original_value)), (
                    f"Power flow {value}{side} for {element} should match"
                )
            limit_row = branch_limits_df.filter((pl.col("element_id") == element) & (pl.col("side") == side))
            expected_loading = row[f"i{side}"] / limit_row.select("value").item() if limit_row.height > 0 else np.nan
            result_loading = (
                branch_results.filter(
                    (pl.col("timestep") == timestep)
                    & (pl.col("contingency") == contingency)
                    & (pl.col("element") == element)
                    & (pl.col("side") == side)
                )
                .select("loading")
                .item()
            )
            assert result_loading == expected_loading or (np.isnan(result_loading) and np.isnan(expected_loading)), (
                f"Loading for {element} should match"
            )

    # Check that all rows for the failed outages are NaN
    failed_rows = branch_results.filter(pl.col("contingency") == failed_outages[0])
    assert np.isnan(failed_rows.select(["i", "p", "q", "loading"]).to_numpy()).all(), (
        "All rows for failed outages should be NaN"
    )


def test_get_node_results_dc():
    bus_results = pl.LazyFrame(
        {
            "contingency_id": ["contingency_1", "contingency_1", "contingency_2", "contingency_2"],
            "operator_strategy_id": ["", "", "", ""],
            "voltage_level_id": ["VL_1"] * 4,
            "bus_id": ["bus_1", "bus_2", "bus_1", "bus_2"],
            "v_mag": [10.0, 10.0, 10.0, np.nan],
            "v_angle": [180.0, 0, 10, np.nan],
        }
    )

    voltage_levels = pl.LazyFrame(
        {
            "id": ["VL_1", "VL_2"],
            "nominal_v": [10.0, 20.0],
            "high_voltage_limit": [11.0, 22.0],
            "low_voltage_limit": [9.0, 18.0],
        }
    )

    monitored_buses = ["bus_1", "bus_2"]
    busmap = pl.LazyFrame({"id": monitored_buses, "bus_breaker_bus_id": monitored_buses})
    failed_outages = ["contingency_3"]  # This wont show up in the results
    timestep = 0
    method = "dc"
    node_results = get_node_results_polars(
        bus_results, monitored_buses, busmap, voltage_levels, failed_outages, timestep, method
    )

    # make sure the schema is correct for the lazy frame
    node_results = NodeResultSchemaPolars.validate(node_results)
    assert isinstance(node_results, pl.LazyFrame), "The result should be a Polars LazyFrame"
    # collect lazy frame for testing
    node_results = node_results
    # make sure the schema is correct for the collected frame
    node_results = NodeResultSchemaPolars.validate(node_results)

    assert node_results.collect().height == 6, "There should be 6 rows in the node results (2 for each contingency)"
    assert node_results.filter(pl.col("timestep") != timestep).collect().is_empty(), "All rows should have the same timestep"
    assert np.isnan(
        node_results.filter(pl.col("element") == failed_outages[0])
        .select(["vm", "va", "p", "q", "vm_loading"])
        .collect()
        .to_numpy()
    ).all(), "All rows for failed outages should be NaN"
    bus_results_df = bus_results.collect()
    voltage_levels_df = voltage_levels.collect()
    for row in bus_results_df.iter_rows(named=True):
        contingency = row["contingency_id"]
        bus_id = row["bus_id"]
        if bus_id in monitored_buses:
            vm_result = (
                node_results.filter(
                    (pl.col("timestep") == timestep) & (pl.col("contingency") == contingency) & (pl.col("element") == bus_id)
                )
                .select("vm")
                .collect()
                .item()
            )
            orig_vm = row["v_mag"]
            voltage = voltage_levels_df.filter(pl.col("id") == "VL_1").select("nominal_v").item()
            assert vm_result == voltage or (np.isnan(vm_result) and np.isnan(orig_vm)), (
                f"Voltage magnitude for {bus_id} in {contingency} in {method} should match"
            )
            vm_loading = (
                node_results.filter(
                    (pl.col("timestep") == timestep) & (pl.col("contingency") == contingency) & (pl.col("element") == bus_id)
                )
                .select("vm_loading")
                .collect()
                .item()
            )
            assert vm_loading == 0.0 or (np.isnan(vm_loading) and np.isnan(orig_vm)), (
                f"Voltage magnitude loading for {bus_id} in {contingency} in {method} should be 0"
            )
            va_result = (
                node_results.filter(
                    (pl.col("timestep") == timestep) & (pl.col("contingency") == contingency) & (pl.col("element") == bus_id)
                )
                .select("va")
                .collect()
                .item()
            )
            orig_va = row["v_angle"]
            assert va_result == orig_va or (np.isnan(va_result) and np.isnan(orig_va)), (
                f"Voltage angle for {bus_id} in {contingency} in {method} should match"
            )
        else:
            assert bus_id not in node_results["element"].to_list(), (
                f"Bus {bus_id} should not be in the node results if it is not monitored"
            )
    monitored_buses = ["bus_1"]
    node_results = get_node_results_polars(
        bus_results, monitored_buses, busmap, voltage_levels, failed_outages, timestep, method
    )
    node_results = node_results.collect()
    assert node_results.height == 3, (
        "There should be 3 rows in the node results (1 for each contingency), since only one bus is monitored"
    )

    # test empty input
    node_results = get_node_results_polars(
        bus_results.limit(0), monitored_buses, busmap, voltage_levels, failed_outages, timestep, method
    )
    assert isinstance(node_results, pl.LazyFrame), "The result should be a Polars LazyFrame"
    node_results = node_results.collect()
    assert isinstance(node_results, pl.DataFrame), "The result should be a Polars DataFrame"
    node_results = NodeResultSchemaPolars.validate(node_results)
    assert node_results.height == len(monitored_buses), "There should be as many rows as monitored buses"


def test_get_node_results_ac():
    bus_results = pl.LazyFrame(
        {
            "contingency_id": ["contingency_1", "contingency_1", "contingency_2", "contingency_2"],
            "operator_strategy_id": ["", "", "", ""],
            "voltage_level_id": ["VL_1"] * 4,
            "bus_id": ["bus_1", "bus_2", "bus_1", "bus_2"],
            "v_mag": [10.0, 10.1, 9.9, np.nan],
            "v_angle": [180.0, 0, 10, np.nan],
        }
    )

    voltage_levels = pl.LazyFrame(
        {
            "id": ["VL_1", "VL_2"],
            "nominal_v": [10.0, 20.0],
            "high_voltage_limit": [11.0, 22.0],
            "low_voltage_limit": [9.0, 18.0],
        }
    )

    monitored_buses = ["bus_1", "bus_2"]
    failed_outages = ["contingency_3"]  # This wont show up in the results
    timestep = 0
    method = "ac"
    busmap = pl.LazyFrame({"id": monitored_buses, "bus_breaker_bus_id": monitored_buses})
    node_results = get_node_results_polars(
        bus_results, monitored_buses, busmap, voltage_levels, failed_outages, timestep, method
    )

    # make sure the schema is correct for the lazy frame
    node_results = NodeResultSchemaPolars.validate(node_results)
    assert isinstance(node_results, pl.LazyFrame), "The result should be a Polars LazyFrame"
    # collect lazy frame for testing
    node_results = node_results.collect()
    # make sure the schema is correct for the collected frame
    node_results = NodeResultSchemaPolars.validate(node_results)

    assert node_results.height == 6, "There should be 6 rows in the node results (2 for each contingency)"
    assert node_results.filter(pl.col("timestep") != timestep).is_empty(), "All rows should have the same timestep"
    assert np.isnan(
        node_results.filter(pl.col("element") == failed_outages[0]).select(["vm", "va", "p", "q", "vm_loading"]).to_numpy()
    ).all(), "All rows for failed outages should be NaN"
    bus_results_df = bus_results.collect()
    voltage_levels_df = voltage_levels.collect()
    for row in bus_results_df.iter_rows(named=True):
        contingency = row["contingency_id"]
        bus_id = row["bus_id"]
        if bus_id in monitored_buses:
            vm_result = (
                node_results.filter(
                    (pl.col("timestep") == timestep) & (pl.col("contingency") == contingency) & (pl.col("element") == bus_id)
                )
                .select("vm")
                .item()
            )
            orig_vm = row["v_mag"]
            nominal_v = voltage_levels_df.filter(pl.col("id") == "VL_1").select("nominal_v").item()
            expected_vm = np.nan
            if not np.isnan(orig_vm):
                expected_vm = orig_vm * nominal_v if abs(orig_vm) <= nominal_v * 0.1 else orig_vm
            assert vm_result == expected_vm or (np.isnan(vm_result) and np.isnan(expected_vm)), (
                f"Voltage magnitude for {bus_id} in {contingency} in {method} should match"
            )
            vm_loading = (
                node_results.filter(
                    (pl.col("timestep") == timestep) & (pl.col("contingency") == contingency) & (pl.col("element") == bus_id)
                )
                .select("vm_loading")
                .item()
            )
            if np.isnan(vm_loading):
                assert np.isnan(orig_vm), "Loading should only be NaN if the original voltage is NaN"
            elif vm_result >= nominal_v:
                voltage_max = voltage_levels_df.filter(pl.col("id") == "VL_1").select("high_voltage_limit").item()
                expected_loading = (vm_result - nominal_v) / (voltage_max - nominal_v)
                assert vm_loading == expected_loading, (
                    f"Voltage loading for {bus_id} in {contingency} in {method} should match"
                )
            elif vm_loading == 0.0:
                assert vm_loading == 0.0, "Loading should be 0 if the voltage is equal to the nominal voltage"
            else:
                voltage_min = voltage_levels_df.filter(pl.col("id") == "VL_1").select("low_voltage_limit").item()
                expected_loading = (vm_result - nominal_v) / (nominal_v - voltage_min)
                assert vm_loading == expected_loading, (
                    f"Voltage loading for {bus_id} in {contingency} in {method} should match"
                )
            va_result = (
                node_results.filter(
                    (pl.col("timestep") == timestep) & (pl.col("contingency") == contingency) & (pl.col("element") == bus_id)
                )
                .select("va")
                .item()
            )
            orig_va = row["v_angle"]
            assert va_result == orig_va or (np.isnan(va_result) and np.isnan(orig_va)), (
                f"Voltage angle for {bus_id} in {contingency} in {method} should match"
            )
        else:
            assert bus_id not in node_results["element"].to_list(), (
                f"Bus {bus_id} should not be in the node results if it is not monitored"
            )
    monitored_buses = ["bus_1"]
    node_results = get_node_results_polars(
        bus_results, monitored_buses, busmap, voltage_levels, failed_outages, timestep, method
    )
    node_results = node_results.collect()
    assert node_results.height == 3, (
        "There should be 3 rows in the node results (1 for each contingency), since only one bus is monitored"
    )


def test_update_basename_with_new_name():
    # Test with a valid base case name
    empty_base_case_name = ""
    base_case_name = "BASECASE"
    timestep = 0
    contingency = ""
    element = "test_element"

    node_df = get_empty_dataframe_from_model(NodeResultSchema)
    node_df = pl.from_pandas(node_df, include_index=True)
    # Fill the specified values and fill the rest of the columns with their default values
    node_df = pl.DataFrame(
        [
            {
                "timestep": timestep,
                "contingency": contingency,
                "element": element,
                "vm": 2.0,
                "vm_loading": np.nan,
                "va": np.nan,
                "p": np.nan,
                "q": np.nan,
                "vm_basecase_deviation": np.nan,
                "element_name": "",
                "contingency_name": "",
            }
        ]
    ).lazy()
    NodeResultSchemaPolars.validate(node_df)
    updated_df = update_basename_polars(node_df, base_case_name)
    assert node_df.filter(pl.col("contingency") == empty_base_case_name).collect().height == 1, (
        "The contingency should not be updated to BASECASE"
    )
    assert updated_df.filter(pl.col("contingency") == base_case_name).collect().height == 1, (
        "The contingency should be updated to BASECASE"
    )

    branch_df = get_empty_dataframe_from_model(BranchResultSchema)
    branch_df = pl.from_pandas(branch_df, include_index=True).lazy()
    # Fill the specified values and fill the rest of the columns with their default values
    branch_df = pl.DataFrame(
        [
            {
                "timestep": timestep,
                "contingency": contingency,
                "element": element,
                "side": 1,
                "p": np.nan,
                "q": np.nan,
                "i": 2.0,
                "loading": np.nan,
                "element_name": "",
                "contingency_name": "",
            }
        ]
    ).lazy()
    updated_df = update_basename_polars(branch_df, base_case_name)
    assert branch_df.filter(pl.col("contingency") == empty_base_case_name).collect().height == 1, (
        "The contingency should not be updated to BASECASE"
    )
    assert updated_df.filter(pl.col("contingency") == base_case_name).collect().height == 1, (
        "The contingency should be updated to BASECASE"
    )

    va_diff_df = get_empty_dataframe_from_model(VADiffResultSchema)
    va_diff_df = pl.from_pandas(va_diff_df, include_index=True).lazy()
    # Fill the specified values and fill the rest of the columns with their default values
    va_diff_df = pl.DataFrame(
        [
            {
                "timestep": timestep,
                "contingency": contingency,
                "element": element,
                "va_diff": 5.0,
                "element_name": "",
                "contingency_name": "",
            }
        ]
    ).lazy()
    updated_df = update_basename_polars(va_diff_df, base_case_name)
    assert va_diff_df.filter(pl.col("contingency") == empty_base_case_name).collect().height == 1, (
        "The contingency should not be updated to BASECASE"
    )
    assert updated_df.filter(pl.col("contingency") == base_case_name).collect().height == 1, (
        "The contingency should be updated to BASECASE"
    )

    node_df = get_empty_dataframe_from_model(NodeResultSchema)
    node_df = pl.from_pandas(node_df, include_index=True).lazy()
    # Fill the specified values and fill the rest of the columns with their default values
    node_df = pl.DataFrame(
        [
            {
                "timestep": timestep,
                "contingency": contingency,
                "element": element,
                "vm": 2.0,
                "vm_loading": np.nan,
                "va": np.nan,
                "p": np.nan,
                "q": np.nan,
                "vm_basecase_deviation": np.nan,
                "element_name": "",
                "contingency_name": "",
            },
            {
                "timestep": timestep,
                "contingency": contingency,
                "element": element + "_2",
                "vm": 2.0,
                "vm_loading": np.nan,
                "va": np.nan,
                "p": np.nan,
                "q": np.nan,
                "vm_basecase_deviation": np.nan,
                "element_name": "",
                "contingency_name": "",
            },
            {
                "timestep": timestep,
                "contingency": "OTHER_CONTINGENCY",
                "element": element,
                "vm": 2.0,
                "vm_loading": np.nan,
                "va": np.nan,
                "p": np.nan,
                "q": np.nan,
                "vm_basecase_deviation": np.nan,
                "element_name": "",
                "contingency_name": "",
            },
        ]
    ).lazy()
    NodeResultSchemaPolars.validate(node_df)
    updated_df = update_basename_polars(node_df, base_case_name)
    assert node_df.filter(pl.col("contingency") == empty_base_case_name).collect().height == 2, (
        "The contingency should be updated to BASECASE for two rows"
    )
    assert updated_df.filter(pl.col("contingency") == "OTHER_CONTINGENCY").collect().height == 1, (
        "The non-basecase contingency should not be updated to BASECASE"
    )

    empty_df = get_empty_dataframe_from_model(NodeResultSchema)
    empty_df = pl.from_pandas(empty_df, include_index=True).lazy()
    updated_empty_df = update_basename_polars(empty_df, base_case_name)
    assert empty_df.collect().is_empty(), "The empty dataframe should remain empty"
    assert updated_empty_df.collect().is_empty(), "The updated empty dataframe should remain empty"


def test_update_basename_drops():
    base_case_name = None
    timestep = 0
    contingency = ""
    element = "test_element"

    node_df = get_empty_dataframe_from_model(NodeResultSchema)
    node_df = pl.from_pandas(node_df, include_index=True)
    node_df = pl.DataFrame(
        [
            {
                "timestep": timestep,
                "contingency": contingency,
                "element": element,
                "vm": 2.0,
                "vm_loading": np.nan,
                "va": np.nan,
                "p": np.nan,
                "q": np.nan,
                "vm_basecase_deviation": np.nan,
                "element_name": "",
                "contingency_name": "",
            }
        ]
    ).lazy()
    NodeResultSchemaPolars.validate(node_df)
    updated_df = update_basename_polars(node_df, base_case_name)
    assert updated_df.collect().is_empty(), "The dataframe should be empty when base_case_name is None"
    assert not node_df.collect().is_empty(), "The original stay dataframe should also be empty when base_case_name is None"

    node_df = get_empty_dataframe_from_model(NodeResultSchema)
    node_df = pl.from_pandas(node_df, include_index=True)
    node_df = pl.DataFrame(
        [
            {
                "timestep": timestep,
                "contingency": contingency,
                "element": element,
                "vm": 2.0,
                "vm_loading": np.nan,
                "va": np.nan,
                "p": np.nan,
                "q": np.nan,
                "vm_basecase_deviation": np.nan,
                "element_name": "",
                "contingency_name": "",
            },
            {
                "timestep": timestep,
                "contingency": contingency,
                "element": element + "_2",
                "vm": 2.0,
                "vm_loading": np.nan,
                "va": np.nan,
                "p": np.nan,
                "q": np.nan,
                "vm_basecase_deviation": np.nan,
                "element_name": "",
                "contingency_name": "",
            },
            {
                "timestep": timestep,
                "contingency": "OTHER_CONTINGENCY",
                "element": element,
                "vm": 2.0,
                "vm_loading": np.nan,
                "va": np.nan,
                "p": np.nan,
                "q": np.nan,
                "vm_basecase_deviation": np.nan,
                "element_name": "",
                "contingency_name": "",
            },
        ]
    ).lazy()
    NodeResultSchemaPolars.validate(node_df)
    updated_df = update_basename_polars(node_df, base_case_name)
    assert updated_df.collect().height == 1, "Only the non-basecase contingency should remain"
    assert node_df.collect().height == 3, "should not change"

    empty_df = get_empty_dataframe_from_model(NodeResultSchema)
    empty_df = pl.from_pandas(empty_df, include_index=True).lazy()
    updated_empty_df = update_basename_polars(empty_df, base_case_name)
    assert empty_df.collect().is_empty(), "The empty dataframe should remain empty"
    assert updated_empty_df.collect().is_empty(), "The updated empty dataframe should remain empty"


def test_translate_element_names():
    timestep = 0
    contingency = "test_contingency"
    element_id = "to_be_translated"
    element_name = "translated_element"
    element_mapping = {element_id: element_name, "another_element": "another_translated_element"}

    node_df = get_empty_dataframe_from_model(NodeResultSchema)
    node_df = pl.from_pandas(node_df, include_index=True).lazy()

    # Test with empty
    updated_df = add_name_column_polars(node_df, element_mapping, index_level="element")
    assert updated_df.collect().is_empty(), "The dataframe should remain empty when no elements are present"

    node_df = pl.DataFrame(
        [
            {
                "timestep": timestep,
                "contingency": contingency,
                "element": element_id,
                "vm": 2.0,
                "vm_loading": np.nan,
                "va": np.nan,
                "p": np.nan,
                "q": np.nan,
                "vm_basecase_deviation": np.nan,
                "element_name": "",
                "contingency_name": "",
            }
        ]
    ).lazy()
    NodeResultSchemaPolars.validate(node_df)
    updated_df = add_name_column_polars(node_df, element_mapping, index_level="element")
    assert updated_df.collect()["element"][0] == element_id, "The element should still be as before"
    assert node_df.collect()["element"][0] == element_id, "The original element should still be as before"
    assert updated_df.collect()["element_name"][0] == element_name, (
        "The element_name column should contain the translated name"
    )
    assert node_df.collect()["element_name"][0] != element_name, (
        "The original element_name column not should contain the translated name"
    )

    # Adding another row without entry
    missing_element_id = "missing"
    node_df = pl.DataFrame(
        [
            {
                "timestep": timestep,
                "contingency": contingency,
                "element": element_id,
                "vm": 2.0,
                "vm_loading": np.nan,
                "va": np.nan,
                "p": np.nan,
                "q": np.nan,
                "vm_basecase_deviation": np.nan,
                "element_name": "",
                "contingency_name": "",
            },
            {
                "timestep": timestep,
                "contingency": contingency,
                "element": missing_element_id,
                "vm": 2.0,
                "vm_loading": np.nan,
                "va": np.nan,
                "p": np.nan,
                "q": np.nan,
                "vm_basecase_deviation": np.nan,
                "element_name": "",
                "contingency_name": "",
            },
        ]
    ).lazy()
    NodeResultSchemaPolars.validate(node_df)
    updated_df = add_name_column_polars(node_df, element_mapping, index_level="element")
    assert updated_df.collect()["element"][0] == element_id, "The element should still be as before"
    assert node_df.collect()["element"][0] == element_id, "The original element should still be as before"
    assert updated_df.collect()["element_name"][0] == element_name, (
        "The element_name column should contain the translated name"
    )

    assert updated_df.filter(pl.col("element") == missing_element_id).collect()["element_name"][0] == "", (
        "The map does not contain the key, so the name should be empty ('')"
    )
    assert (node_df.collect()["element_name"] == "").all(), "Nothing should change in the original dataframe"

    branch_df = get_empty_dataframe_from_model(BranchResultSchema)
    branch_df = pl.from_pandas(branch_df, include_index=True).lazy()
    branch_df = pl.DataFrame(
        [
            {
                "timestep": timestep,
                "contingency": contingency,
                "element": element_id,
                "side": 1,
                "p": 2.0,
                "q": np.nan,
                "i": np.nan,
                "loading": np.nan,
                "element_name": "",
                "contingency_name": "",
            }
        ]
    ).lazy()
    BranchResultSchemaPolars.validate(branch_df)
    updated_branch_df = add_name_column_polars(branch_df, element_mapping, index_level="element")
    assert updated_branch_df.collect()["element"][0] == element_id, "The element should still be as before"
    assert branch_df.collect()["element"][0] == element_id, "The original element should still be as before"
    assert updated_branch_df.collect()["element_name"][0] == element_name, (
        "The element_name column should contain the translated name"
    )
    assert (branch_df.collect()["element_name"] == "").all(), "Nothing should change in the original dataframe"

    va_diff_df = get_empty_dataframe_from_model(VADiffResultSchema)
    va_diff_df = pl.from_pandas(va_diff_df, include_index=True).lazy()
    va_diff_df = pl.DataFrame(
        [
            {
                "timestep": timestep,
                "contingency": contingency,
                "element": element_id,
                "va_diff": 5.0,
                "element_name": "",
                "contingency_name": "",
            }
        ]
    ).lazy()
    VADiffResultSchemaPolars.validate(va_diff_df)
    updated_va_diff_df = add_name_column_polars(va_diff_df, element_mapping, index_level="element")
    assert updated_va_diff_df.collect()["element"][0] == element_id, "The element should still be as before"
    assert va_diff_df.collect()["element"][0] == element_id, "The original element should still be as before"
    assert updated_va_diff_df.collect()["element_name"][0] == element_name, (
        "The element_name column should contain the translated name"
    )
    assert (va_diff_df.collect()["element_name"] == "").all(), "Nothing should change in the original dataframe"
