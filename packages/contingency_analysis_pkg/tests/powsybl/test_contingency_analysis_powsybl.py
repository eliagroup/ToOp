# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from dataclasses import replace

import numpy as np
import pandas as pd
import pandera as pa
import polars as pl
import pypowsybl
import pytest
from polars.testing import assert_frame_equal
from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_contingency_analysis.pypowsybl import (
    get_full_nminus1_definition_powsybl,
    run_powsybl_analysis,
    translate_nminus1_for_powsybl,
)
from toop_engine_contingency_analysis.pypowsybl.contingency_analysis_powsybl import (
    PowsyblBranchLimitCache,
    build_branch_limit_cache,
    run_contingency_analysis_powsybl,
)
from toop_engine_grid_helpers.powsybl.loadflow_parameters import CGMES_DISTRIBUTED_SLACK, SINGLE_SLACK
from toop_engine_importer.network_graph.powsybl_station_to_graph import (
    get_relevant_voltage_levels,
    get_station_list,
)
from toop_engine_importer.pypowsybl_import import powsybl_masks
from toop_engine_interfaces.loadflow_result_helpers import (
    convert_pandas_loadflow_results_to_polars,
    convert_polars_loadflow_results_to_pandas,
    extract_branch_results,
    extract_solver_matrices,
)
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, MonitoredElement, Nminus1Definition


def _normalize_contingency_frame(
    frame: pl.DataFrame,
    contingency_id: str,
    excluded_element_prefixes: set[str] | None = None,
    excluded_elements: set[str] | None = None,
    excluded_branches: set[str] | None = None,
) -> pl.DataFrame:
    """Normalize contingency-specific result frames for direct equality checks."""
    normalized = frame.filter(pl.col("contingency") == contingency_id)

    if excluded_element_prefixes and "element" in normalized.columns:
        excluded_prefix_expr = pl.lit(False)
        for prefix in sorted(excluded_element_prefixes):
            excluded_prefix_expr = excluded_prefix_expr | pl.col("element").str.starts_with(prefix)
        normalized = normalized.filter(~excluded_prefix_expr)

    if excluded_elements and "element" in normalized.columns:
        normalized = normalized.filter(~pl.col("element").is_in(sorted(excluded_elements)))

    if excluded_branches and {"element", "side"}.issubset(set(normalized.columns)):
        normalized = normalized.filter(~pl.col("element").is_in(sorted(excluded_branches)))

    if "contingency_name" in normalized.columns:
        normalized = normalized.with_columns(pl.lit("").alias("contingency_name"))
    normalized = normalized.with_columns(pl.lit("reference").alias("contingency"))
    sort_columns = [column for column in normalized.columns if column not in {"contingency_name"}]
    return normalized.sort(sort_columns)


def test_run_powsybl_analysis(powsybl_bus_breaker_net: pypowsybl.network.Network) -> None:
    nminus1_definition = get_full_nminus1_definition_powsybl(powsybl_bus_breaker_net)

    pow_n1_def = translate_nminus1_for_powsybl(nminus1_definition, powsybl_bus_breaker_net)
    result, basecase_name = run_powsybl_analysis(powsybl_bus_breaker_net, pow_n1_def, CGMES_DISTRIBUTED_SLACK, "dc")
    assert all(result.branch_results["q1"].isna())
    assert all(result.bus_results["v_mag"].isna())
    assert result is not None
    assert basecase_name == "BASECASE"

    result, basecase_name = run_powsybl_analysis(powsybl_bus_breaker_net, pow_n1_def, CGMES_DISTRIBUTED_SLACK, "ac")
    assert not all(result.branch_results["q1"].isna())
    assert not all(result.bus_results["v_mag"].isna())
    assert result is not None
    assert basecase_name == "BASECASE"


def test_run_contingency_analysis_powsybl_with_branch_limit_cache(
    powsybl_bus_breaker_net: pypowsybl.network.Network,
) -> None:
    nminus1_definition = get_full_nminus1_definition_powsybl(powsybl_bus_breaker_net)
    translated_nminus1 = translate_nminus1_for_powsybl(nminus1_definition, powsybl_bus_breaker_net)
    branch_limit_cache = build_branch_limit_cache(
        powsybl_bus_breaker_net,
        monitored_branches=translated_nminus1.monitored_elements["branches"],
    )

    uncached_result = run_contingency_analysis_powsybl(
        net=powsybl_bus_breaker_net,
        n_minus_1_definition=nminus1_definition,
        job_id="test_job",
        timestep=0,
        method="dc",
        polars=True,
    )
    cached_result = run_contingency_analysis_powsybl(
        net=powsybl_bus_breaker_net,
        n_minus_1_definition=nminus1_definition,
        job_id="test_job",
        timestep=0,
        method="dc",
        polars=True,
        branch_limit_cache=branch_limit_cache,
    )

    assert uncached_result == cached_result


def test_run_contingency_analysis_powsybl_ignores_stale_branch_limit_cache(
    powsybl_bus_breaker_net: pypowsybl.network.Network,
) -> None:
    nminus1_definition = get_full_nminus1_definition_powsybl(powsybl_bus_breaker_net)
    stale_cache = PowsyblBranchLimitCache(
        chosen_limit="permanent_limit",
        monitored_branches=("definitely-stale",),
        current_limit_fingerprint="stale-fingerprint",
        branch_limits=pd.DataFrame(
            {"value": [999.0]},
            index=pd.MultiIndex.from_tuples([("stale", 1)], names=["element_id", "side"]),
        ),
    )

    uncached_result = run_contingency_analysis_powsybl(
        net=powsybl_bus_breaker_net,
        n_minus_1_definition=nminus1_definition,
        job_id="test_job",
        timestep=0,
        method="dc",
        polars=True,
    )
    cached_result = run_contingency_analysis_powsybl(
        net=powsybl_bus_breaker_net,
        n_minus_1_definition=nminus1_definition,
        job_id="test_job",
        timestep=0,
        method="dc",
        polars=True,
        branch_limit_cache=stale_cache,
    )

    assert uncached_result == cached_result


@pytest.mark.parametrize("powsybl_net", ["powsybl_bus_breaker_net", "powsybl_node_breaker_net"])
def test_run_ac_contingency_analysis_powsybl(powsybl_net: str, request) -> None:
    net = request.getfixturevalue(powsybl_net)
    nminus1_definition = get_full_nminus1_definition_powsybl(net)

    lf_result_sequential_polars = get_ac_loadflow_results(net, nminus1_definition, job_id="test_job", n_processes=1)
    assert len(lf_result_sequential_polars.branch_results.collect()) > 0
    assert len(lf_result_sequential_polars.node_results.collect()) > 0
    assert len(lf_result_sequential_polars.va_diff_results.collect()) > 0
    assert len(lf_result_sequential_polars.regulating_element_results.collect()) > 0
    assert len(lf_result_sequential_polars.converged.collect()) > 0
    # Run the loadflow in parallel
    lf_result_parallel_polars = get_ac_loadflow_results(net, nminus1_definition, job_id="test_job", n_processes=2)

    assert lf_result_sequential_polars is not None
    assert lf_result_parallel_polars is not None

    assert len(lf_result_sequential_polars.branch_results.collect()) == len(
        lf_result_parallel_polars.branch_results.collect()
    )
    assert len(lf_result_sequential_polars.node_results.collect()) == len(lf_result_parallel_polars.node_results.collect())
    assert len(lf_result_sequential_polars.va_diff_results.collect()) == len(
        lf_result_parallel_polars.va_diff_results.collect()
    )
    assert len(lf_result_sequential_polars.regulating_element_results.collect()) == len(
        lf_result_parallel_polars.regulating_element_results.collect()
    )
    assert len(lf_result_sequential_polars.converged.collect()) == len(lf_result_parallel_polars.converged.collect())


def test_contingency_analysis_validated_or_not(powsybl_node_breaker_net: pypowsybl.network.Network) -> None:
    nminus1_definition = get_full_nminus1_definition_powsybl(powsybl_node_breaker_net)

    with pa.config.config_context(validation_enabled=True, validation_depth=pa.config.ValidationDepth.SCHEMA_AND_DATA):
        lf_result_with_val_polars = get_ac_loadflow_results(
            powsybl_node_breaker_net, nminus1_definition, job_id="test_job", n_processes=2
        )
    with pa.config.config_context(validation_enabled=False):
        lf_result_no_val_polars = get_ac_loadflow_results(
            powsybl_node_breaker_net, nminus1_definition, job_id="test_job", n_processes=2
        )

    assert lf_result_with_val_polars == lf_result_no_val_polars


def test_run_ac_contingency_analysis_powsybl_with_busbar_outage(
    powsybl_node_breaker_net: pypowsybl.network.Network,
) -> None:
    nminus1_definition = get_full_nminus1_definition_powsybl(powsybl_node_breaker_net)
    busbar_sections = powsybl_node_breaker_net.get_busbar_sections()
    # select a busbar section that is not the slack
    lf_res = pypowsybl.loadflow.run_ac(powsybl_node_breaker_net)[0]
    busbar_sections = busbar_sections[busbar_sections.bus_id != lf_res.reference_bus_id]
    selected_busbar_id = selected_busbar_name = busbar_sections.index[0]
    nminus1_definition.contingencies.append(
        Contingency(
            id=selected_busbar_id,
            name=selected_busbar_name or "",
            elements=[
                GridElement(
                    id=selected_busbar_id,
                    name=selected_busbar_name or "",
                    type="BUSBAR_SECTION",
                    kind="bus",
                )
            ],
        )
    )

    lf_result = get_ac_loadflow_results(powsybl_node_breaker_net, nminus1_definition, job_id="test_job", n_processes=1)

    converged = lf_result.converged.collect()
    assert selected_busbar_id in converged["contingency"].to_list()
    assert not any(selected_busbar_id in warning for warning in lf_result.warnings)


def test_busbar_outage_equals_connected_element_outage(
    powsybl_node_breaker_net: pypowsybl.network.Network,
) -> None:
    busbar_sections = powsybl_node_breaker_net.get_busbar_sections()
    lf_res = pypowsybl.loadflow.run_ac(powsybl_node_breaker_net)[0]
    non_slack_busbar_sections = busbar_sections[busbar_sections.bus_id != lf_res.reference_bus_id]
    assert len(non_slack_busbar_sections) > 0

    selected_voltage_levels = {"VL2", "VL3"}
    selected_busbar_sections = non_slack_busbar_sections[
        non_slack_busbar_sections["voltage_level_id"].astype(str).isin(selected_voltage_levels)
    ]
    assert len(selected_busbar_sections) > 0

    network_masks = powsybl_masks.create_default_network_masks(powsybl_node_breaker_net)
    network_masks = replace(
        network_masks,
        relevant_subs=np.ones_like(network_masks.relevant_subs, dtype=bool),
        busbar_for_nminus1=np.ones_like(network_masks.busbar_for_nminus1, dtype=bool),
    )
    relevant_voltage_level_with_region = get_relevant_voltage_levels(
        network=powsybl_node_breaker_net,
        network_masks=network_masks,
    )
    station_list = get_station_list(
        network=powsybl_node_breaker_net,
        relevant_voltage_level_with_region=relevant_voltage_level_with_region,
    )
    assert len(station_list) == len(powsybl_node_breaker_net.get_buses())

    monitored_votlages = []
    for vl in relevant_voltage_level_with_region["voltage_level_id"].values:
        bus_breaker_topology = powsybl_node_breaker_net.get_bus_breaker_topology(vl)
        elements = bus_breaker_topology.elements
        elements.rename(columns={"bus_id": "bus_breaker_id"}, inplace=True)
        elements = elements.merge(
            powsybl_node_breaker_net.get_bus_breaker_view_buses()[["bus_id"]],
            left_on="bus_breaker_id",
            right_on="id",
            how="left",
        )
        elements = elements[elements["type"] == "BUSBAR_SECTION"]
        monitored_votlages += elements["bus_breaker_id"].to_list()

    branches = powsybl_node_breaker_net.get_branches(attributes=["type"])
    injections = powsybl_node_breaker_net.get_injections(attributes=["type"])

    monitored_elements = get_full_nminus1_definition_powsybl(powsybl_node_breaker_net).monitored_elements
    # filter monitored bus elements to only monitor buses from powsybl_node_breaker_net.get_buses()
    monitored_elements = [
        element
        for element in monitored_elements
        if (element.type == "BUS" and element.id in monitored_votlages) or (element.type != "BUS")
    ]

    for selected_busbar_id in selected_busbar_sections.index:
        selected_busbar_name = selected_busbar_id
        selected_voltage_level_id = str(selected_busbar_sections.loc[selected_busbar_id, "voltage_level_id"])

        selected_station = next(
            station
            for station in station_list
            if any(busbar.grid_model_id == selected_busbar_id for busbar in station.busbars)
        )
        selected_busbar_index = next(
            i for i, busbar in enumerate(selected_station.busbars) if busbar.grid_model_id == selected_busbar_id
        )
        connected_asset_ids = {
            asset.grid_model_id
            for i, asset in enumerate(selected_station.assets)
            if selected_station.asset_switching_table[selected_busbar_index, i]
        }

        explicit_elements = [
            GridElement(id=asset_id, name=asset_id, type=branches.loc[asset_id, "type"], kind="branch")
            for asset_id in connected_asset_ids
            if asset_id in branches.index
        ] + [
            GridElement(
                id=asset_id,
                name=asset_id,
                type=injections.loc[asset_id, "type"],
                kind="injection",
            )
            for asset_id in connected_asset_ids
            if asset_id in injections.index and injections.loc[asset_id, "type"] != "BUSBAR_SECTION"
        ]

        nminus1_definition = Nminus1Definition(
            monitored_elements=monitored_elements,
            contingencies=[
                Contingency(id="BASECASE", elements=[]),
                Contingency(
                    id=selected_busbar_id,
                    name=selected_busbar_name or "",
                    elements=[
                        GridElement(
                            id=selected_busbar_id,
                            name=selected_busbar_name or "",
                            type="BUSBAR_SECTION",
                            kind="bus",
                        )
                    ],
                ),
                Contingency(id="explicit_busbar_outage", elements=explicit_elements),
            ],
            id_type="powsybl",
        )

        lf_result = get_ac_loadflow_results(
            powsybl_node_breaker_net, nminus1_definition, job_id="test_job", n_processes=1, lf_params=SINGLE_SLACK
        )

        selected_node_elements = set(
            lf_result.node_results.collect().filter(pl.col("contingency") == selected_busbar_id)["element"].to_list()
        )
        explicit_node_elements = set(
            lf_result.node_results.collect().filter(pl.col("contingency") == "explicit_busbar_outage")["element"].to_list()
        )
        excluded_selected_voltage_level_elements = {
            element_id
            for element_id in (explicit_node_elements - selected_node_elements)
            if element_id.startswith(f"{selected_voltage_level_id}_")
        }

        selected_regulating_elements = set(
            lf_result.regulating_element_results.collect()
            .filter(pl.col("contingency") == selected_busbar_id)["element"]
            .to_list()
        )
        explicit_regulating_elements = set(
            lf_result.regulating_element_results.collect()
            .filter(pl.col("contingency") == "explicit_busbar_outage")["element"]
            .to_list()
        )
        excluded_explicit_regulating_elements = explicit_regulating_elements - selected_regulating_elements

        selected_va_diff_elements = set(
            lf_result.va_diff_results.collect().filter(pl.col("contingency") == selected_busbar_id)["element"].to_list()
        )
        explicit_va_diff_elements = set(
            lf_result.va_diff_results.collect()
            .filter(pl.col("contingency") == "explicit_busbar_outage")["element"]
            .to_list()
        )
        excluded_explicit_va_diff_elements = explicit_va_diff_elements - selected_va_diff_elements

        excluded_branch_ids = {asset_id for asset_id in connected_asset_ids if asset_id in branches.index}

        assert_frame_equal(
            _normalize_contingency_frame(
                lf_result.branch_results.collect(),
                selected_busbar_id,
                excluded_branches=excluded_branch_ids,
            ),
            _normalize_contingency_frame(
                lf_result.branch_results.collect(),
                "explicit_busbar_outage",
                excluded_branches=excluded_branch_ids,
            ),
            check_row_order=False,
            check_exact=False,
            abs_tol=1e-4,
        )
        assert_frame_equal(
            _normalize_contingency_frame(lf_result.node_results.collect(), selected_busbar_id),
            _normalize_contingency_frame(
                lf_result.node_results.collect(),
                "explicit_busbar_outage",
                excluded_elements=excluded_selected_voltage_level_elements,
            ),
            check_row_order=False,
        )
        assert_frame_equal(
            _normalize_contingency_frame(lf_result.regulating_element_results.collect(), selected_busbar_id),
            _normalize_contingency_frame(
                lf_result.regulating_element_results.collect(),
                "explicit_busbar_outage",
                excluded_elements=excluded_explicit_regulating_elements,
            ),
            check_row_order=False,
        )
        assert_frame_equal(
            _normalize_contingency_frame(lf_result.va_diff_results.collect(), selected_busbar_id),
            _normalize_contingency_frame(
                lf_result.va_diff_results.collect(),
                "explicit_busbar_outage",
                excluded_elements=excluded_explicit_va_diff_elements,
            ),
            check_row_order=False,
        )
        assert_frame_equal(
            _normalize_contingency_frame(lf_result.converged.collect(), selected_busbar_id),
            _normalize_contingency_frame(lf_result.converged.collect(), "explicit_busbar_outage"),
            check_row_order=False,
        )


@pytest.mark.parametrize("powsybl_net", ["powsybl_bus_breaker_net", "powsybl_node_breaker_net"])
def test_contingency_analysis_ray_vs_powsybl(powsybl_net: str, request, init_ray) -> None:
    net = request.getfixturevalue(powsybl_net)

    nminus1_definition = get_full_nminus1_definition_powsybl(net)

    lf_parallel_ray_polars = get_ac_loadflow_results(
        net, nminus1_definition, job_id="test_job", n_processes=2, batch_size=10
    )

    lf_parallel_native_polars = get_ac_loadflow_results(
        net,
        nminus1_definition,
        job_id="test_job",
        timestep=0,
        n_processes=2,
    )
    assert lf_parallel_ray_polars == lf_parallel_native_polars


def test_run_contingency_analysis_powsybl_not_converging_basecase() -> None:
    net = pypowsybl.network.create_ieee14()
    loads = net.get_loads(attributes=["q0"])
    # make the ac loadflow fail
    loads["q0"] = loads["q0"] * 10
    net.update_loads(loads)

    nminus1_def_1 = Nminus1Definition(
        monitored_elements=[
            MonitoredElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[Contingency(id="BASECASE", elements=[])],
    )

    result = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def_1, job_id="same_test_job", timestep=0, method="ac", polars=True
    )
    assert len(result.converged.collect()) == 1
    assert result.converged.collect()["contingency"][0] == "BASECASE"
    assert result.converged.collect()["status"][0] == "MAX_ITERATION_REACHED"


def test_extract_branch_results_disconnected():
    net = pypowsybl.network.create_ieee14()
    net.disconnect(net.get_branches().index[0])
    nminus1_def = Nminus1Definition(
        monitored_elements=[
            MonitoredElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[
            Contingency(id=index, elements=[GridElement(id=index, name=row.name, kind="branch", type=row.type)])
            for index, row in net.get_branches().iterrows()
        ],
    )

    res = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def, job_id="test_job", timestep=0, method="dc"
    )
    _, matrix = extract_branch_results(
        branch_results=res.branch_results,
        basecase="BASECASE",
        contingencies=[contingency.id for contingency in nminus1_def.contingencies if not contingency.is_basecase()],
        monitored_branches=[element for element in nminus1_def.monitored_elements if element.kind == "branch"],
        timestep=0,
    )
    assert matrix.shape == (len(nminus1_def.contingencies), len(nminus1_def.monitored_elements))
    assert matrix.dtype == float
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix[0, :] == 0.0)  # Check that the disconnected branch has a loading of 0.0 in all contingencies
    assert np.all(matrix[:, 0] == 0.0)  # Check that the first monitored branch has a loading of 0.0 in all contingencies


def test_extract_solver_matrices():
    net = pypowsybl.network.create_ieee14()
    nminus1_def = Nminus1Definition(
        monitored_elements=[
            MonitoredElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[Contingency(id="BASECASE", elements=[])]
        + [
            Contingency(id=index, elements=[GridElement(id=index, name=row.name, kind="branch", type=row.type)])
            for index, row in net.get_branches().iterrows()
        ],
    )

    res = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def, job_id="test_job", timestep=0, method="dc"
    )
    n_0, n_1, success = extract_solver_matrices(
        loadflow_results=res,
        nminus1_definition=nminus1_def,
        timestep=0,
    )
    assert n_0.shape == (len(nminus1_def.monitored_elements),)
    assert n_1.shape == (len(nminus1_def.contingencies) - 1, len(nminus1_def.monitored_elements))
    assert success.shape == (len(nminus1_def.contingencies) - 1,)
    assert n_0.dtype == float
    assert n_1.dtype == float
    assert success.dtype == bool
    assert np.all(np.isfinite(n_0))
    assert np.all(np.isfinite(n_1))
    assert np.all(success)
    contingencies = [contingency for contingency in nminus1_def.contingencies if not contingency.is_basecase()]
    nminus1_def.contingencies = contingencies
    with pytest.raises(AssertionError):
        extract_solver_matrices(
            loadflow_results=res,
            nminus1_definition=nminus1_def,
            timestep=0,
        )


def test_extract_solver_matrices_disconnected():
    net = pypowsybl.network.create_ieee14()
    net.disconnect(net.get_branches().index[0])
    nminus1_def = Nminus1Definition(
        monitored_elements=[
            MonitoredElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[Contingency(id="BASECASE", elements=[])]
        + [
            Contingency(id=index, elements=[GridElement(id=index, name=row.name, kind="branch", type=row.type)])
            for index, row in net.get_branches().iterrows()
        ],
    )

    res = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def, job_id="test_job", timestep=0, method="dc"
    )
    n_0, n_1, success = extract_solver_matrices(
        loadflow_results=res,
        nminus1_definition=nminus1_def,
        timestep=0,
    )
    assert n_0.shape == (len(nminus1_def.monitored_elements),)
    assert n_1.shape == (len(nminus1_def.contingencies) - 1, len(nminus1_def.monitored_elements))
    assert success.shape == (len(nminus1_def.contingencies) - 1,)
    assert n_0.dtype == float
    assert n_1.dtype == float
    assert success.dtype == bool
    assert np.all(np.isfinite(n_0))
    assert np.all(np.isfinite(n_1))
    assert success[0].item() is True, "Outages that are disconnected should be considered successful"
    assert np.all(success[1:])
    assert n_0[0] == 0.0  # Check that the disconnected branch has a loading of 0.0 in the base case
    assert np.all(n_1[:, 0] == 0.0)  # Check that the first monitored branch has a loading of 0.0 in all contingencies
    assert np.all(n_1[0, :] == 0.0)  # Check that the first contingency has a loading of 0.0 in all monitored branches


def test_extract_branch_results_disconnected():
    net = pypowsybl.network.create_ieee14()
    net.disconnect(net.get_branches().index[0])
    nminus1_def = Nminus1Definition(
        monitored_elements=[
            MonitoredElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[
            Contingency(id=index, elements=[GridElement(id=index, name=row.name, kind="branch", type=row.type)])
            for index, row in net.get_branches().iterrows()
        ],
    )

    res = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def, job_id="test_job", timestep=0, method="dc"
    )
    _, matrix = extract_branch_results(
        branch_results=res.branch_results,
        basecase="BASECASE",
        contingencies=[contingency.id for contingency in nminus1_def.contingencies if not contingency.is_basecase()],
        monitored_branches=[element for element in nminus1_def.monitored_elements if element.kind == "branch"],
        timestep=0,
    )
    assert matrix.shape == (len(nminus1_def.contingencies), len(nminus1_def.monitored_elements))
    assert matrix.dtype == float
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix[0, :] == 0.0)  # Check that the disconnected branch has a loading of 0.0 in all contingencies
    assert np.all(matrix[:, 0] == 0.0)  # Check that the first monitored branch has a loading of 0.0 in all contingencies


def test_extract_solver_matrices():
    net = pypowsybl.network.create_ieee14()
    nminus1_def = Nminus1Definition(
        monitored_elements=[
            MonitoredElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[Contingency(id="BASECASE", elements=[])]
        + [
            Contingency(id=index, elements=[GridElement(id=index, name=row.name, kind="branch", type=row.type)])
            for index, row in net.get_branches().iterrows()
        ],
    )

    res = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def, job_id="test_job", timestep=0, method="dc"
    )
    n_0, n_1, success = extract_solver_matrices(
        loadflow_results=res,
        nminus1_definition=nminus1_def,
        timestep=0,
    )
    assert n_0.shape == (len(nminus1_def.monitored_elements),)
    assert n_1.shape == (len(nminus1_def.contingencies) - 1, len(nminus1_def.monitored_elements))
    assert success.shape == (len(nminus1_def.contingencies) - 1,)
    assert n_0.dtype == float
    assert n_1.dtype == float
    assert success.dtype == bool
    assert np.all(np.isfinite(n_0))
    assert np.all(np.isfinite(n_1))
    assert np.all(success)
    contingencies = [contingency for contingency in nminus1_def.contingencies if not contingency.is_basecase()]
    nminus1_def.contingencies = contingencies
    with pytest.raises(AssertionError):
        extract_solver_matrices(
            loadflow_results=res,
            nminus1_definition=nminus1_def,
            timestep=0,
        )


def test_extract_solver_matrices_disconnected():
    net = pypowsybl.network.create_ieee14()
    net.disconnect(net.get_branches().index[0])
    nminus1_def = Nminus1Definition(
        monitored_elements=[
            MonitoredElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[Contingency(id="BASECASE", elements=[])]
        + [
            Contingency(id=index, elements=[GridElement(id=index, name=row.name, kind="branch", type=row.type)])
            for index, row in net.get_branches().iterrows()
        ],
    )

    res = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def, job_id="test_job", timestep=0, method="dc"
    )
    n_0, n_1, success = extract_solver_matrices(
        loadflow_results=res,
        nminus1_definition=nminus1_def,
        timestep=0,
    )
    assert n_0.shape == (len(nminus1_def.monitored_elements),)
    assert n_1.shape == (len(nminus1_def.contingencies) - 1, len(nminus1_def.monitored_elements))
    assert success.shape == (len(nminus1_def.contingencies) - 1,)
    assert n_0.dtype == float
    assert n_1.dtype == float
    assert success.dtype == bool
    assert np.all(np.isfinite(n_0))
    assert np.all(np.isfinite(n_1))
    assert success[0].item() is True, "Outages that are disconnected should be considered successful"
    assert np.all(success[1:])
    assert n_0[0] == 0.0  # Check that the disconnected branch has a loading of 0.0 in the base case
    assert np.all(n_1[:, 0] == 0.0)  # Check that the first monitored branch has a loading of 0.0 in all contingencies
    assert np.all(n_1[0, :] == 0.0)  # Check that the first contingency has a loading of 0.0 in all monitored branches


def test_convert_polars_loadflow_results_to_pandas():
    net = pypowsybl.network.create_ieee14()
    nminus1_def_1 = Nminus1Definition(
        monitored_elements=[
            MonitoredElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[Contingency(id="BASECASE", elements=[])],
    )

    nminus1_def_2 = Nminus1Definition(
        monitored_elements=[
            MonitoredElement(id=index, name=row.name, kind="branch", type=row.type)
            for index, row in net.get_branches().iterrows()
        ],
        contingencies=[
            Contingency(id=index, elements=[GridElement(id=index, name=row.name, kind="branch", type=row.type)])
            for index, row in net.get_branches().iterrows()
        ],
    )

    loadflow_data_polars = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def_1, job_id="same_test_job", timestep=0, method="dc", polars=True
    )
    loadflow_data_pandas = run_contingency_analysis_powsybl(
        net=net, n_minus_1_definition=nminus1_def_1, job_id="same_test_job", timestep=0, method="dc", polars=False
    )

    loadflow_data_pandas_2 = convert_polars_loadflow_results_to_pandas(loadflow_data_polars)

    loadflow_data_polars_2 = convert_pandas_loadflow_results_to_polars(loadflow_data_pandas_2)

    kw_args_testing = {
        "check_row_order": False,
        "check_column_order": False,
        "check_dtypes": True,
        "check_exact": False,
        "abs_tol": 1e-9,
    }

    # this is for debugging purposes
    assert loadflow_data_polars.job_id == loadflow_data_polars_2.job_id
    assert_frame_equal(loadflow_data_polars.branch_results, loadflow_data_polars_2.branch_results, **kw_args_testing)
    assert_frame_equal(loadflow_data_polars.node_results, loadflow_data_polars_2.node_results, **kw_args_testing)
    assert_frame_equal(
        loadflow_data_polars.regulating_element_results,
        loadflow_data_polars_2.regulating_element_results,
        **kw_args_testing,
    )
    assert_frame_equal(loadflow_data_polars.va_diff_results, loadflow_data_polars_2.va_diff_results, **kw_args_testing)
    assert_frame_equal(loadflow_data_polars.converged, loadflow_data_polars_2.converged, **kw_args_testing)
    assert loadflow_data_polars.warnings == loadflow_data_polars_2.warnings

    # this is the actual test
    assert loadflow_data_polars.__eq__(loadflow_data_polars_2)

    assert loadflow_data_pandas.__eq__(loadflow_data_pandas_2)
