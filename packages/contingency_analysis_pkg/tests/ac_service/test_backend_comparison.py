# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pandapower
import pandas as pd
import pypowsybl
import pytest
from fsspec.implementations.local import LocalFileSystem
from pandapower import networks
from pandapower import pandapowerNet as PandapowerNetwork
from pandapower.converter import from_cim, to_mpc
from pypowsybl.loadflow import Parameters
from toop_engine_contingency_analysis.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_contingency_analysis.pandapower import get_full_nminus1_definition_pandapower
from toop_engine_contingency_analysis.pypowsybl import get_full_nminus1_definition_powsybl
from toop_engine_grid_helpers.pandapower.pandapower_helpers import load_pandapower_from_fs
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_grid_helpers.powsybl.loadflow_parameters import OPENLOADFLOW_PARAM_PF, UNIFORM_SINGLE_SLACK
from toop_engine_grid_helpers.powsybl.powsybl_helpers import load_powsybl_from_fs
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Nminus1Definition

BranchKey = tuple[str, int, int]
NodeKey = tuple[str, int]
ContingencyKey = str | BranchKey

LARGE_GRID_EXPECTATIONS = {
    "case57": {
        "branch_p_max": 5.0,
        "branch_q_max": 3.0,
        "node_vm_pu_max": 1e-6,
        "node_va_max": 1e-5,
    },
    "case300": {
        "branch_p_max": 1.1,
        "branch_q_max": 1.1,
        "node_vm_pu_max": 0.004,
        "node_va_max": 0.7,
    },
}

EXTERNAL_CGMES_BASECASE_EXPECTATIONS = {
    "external_cgmes_basecase": {
        "path": "",
        "forced_slack_generator_id": "",
        "branch_p_max": 28.0,
        "branch_q_max": 3.5,
        "switch_va_diff_abs_max": 3e-2,
    },
}

EXTERNAL_CGMES_PROVIDER_OVERRIDES = {
    "voltageInitModeOverride": "NONE",
    "newtonRaphsonStoppingCriteriaType": "UNIFORM_CRITERIA",
    "stateVectorScalingMode": "MAX_VOLTAGE_CHANGE",
    "linePerUnitMode": "RATIO",
}


PANDAPOWER_LOADFLOW_PARAM_PPCI = {
    "calculate_voltage_angles": True,
    "trafo_model": "t",
    "check_connectivity": True,
    "mode": "pf",
    "switch_rx_ratio": 2,
    "enforce_q_lims": False,
    "voltage_depend_loads": False,
    "consider_line_temperature": False,
    "tdpf": False,
    "tdpf_update_r_theta": True,
    "tdpf_delay_s": None,
    "distributed_slack": False,
    "delta": 0,
    "trafo3w_losses": "hv",
    "init_results": True,
    "p_lim_default": 1000000000.0,
    "q_lim_default": 1000000000.0,
    "neglect_open_switch_branches": False,
    "tolerance_mva": 1e-08,
    "trafo_loading": "current",
    "numba": True,
    "algorithm": "nr",  # Normal Newton-Rhapson
    "max_iteration": 25,
    "v_debug": False,
    "only_v_results": False,
    "use_umfpack": True,
    "permc_spec": None,
    "lightsim2grid": False,
    "recycle": False,
}


def _pp_branch_key_maps(net: PandapowerNetwork) -> dict[str, BranchKey]:
    element_key: dict[str, BranchKey] = {}
    for idx, row in net.line.iterrows():
        element_key[get_globally_unique_id(idx, "line")] = (
            "line",
            min(int(row.from_bus) + 1, int(row.to_bus) + 1),
            max(int(row.from_bus) + 1, int(row.to_bus) + 1),
        )
    for idx, row in net.trafo.iterrows():
        element_key[get_globally_unique_id(idx, "trafo")] = (
            "trafo",
            min(int(row.hv_bus) + 1, int(row.lv_bus) + 1),
            max(int(row.hv_bus) + 1, int(row.lv_bus) + 1),
        )
    return element_key


def _py_branch_key_maps(net: pypowsybl.network.Network) -> dict[str, BranchKey]:
    element_key: dict[str, BranchKey] = {}
    for idx in net.get_lines().index:
        match = re.fullmatch(r"LINE-(\d+)-(\d+)", str(idx))
        if match:
            bus_1, bus_2 = int(match.group(1)), int(match.group(2))
            element_key[str(idx)] = ("line", min(bus_1, bus_2), max(bus_1, bus_2))
    for idx in net.get_2_windings_transformers().index:
        match = re.fullmatch(r"TWT-(\d+)-(\d+)", str(idx))
        if match:
            bus_1, bus_2 = int(match.group(1)), int(match.group(2))
            element_key[str(idx)] = ("trafo", min(bus_1, bus_2), max(bus_1, bus_2))
    return element_key


def _pp_node_key_maps(net: PandapowerNetwork) -> dict[str, NodeKey]:
    return {get_globally_unique_id(idx, "bus"): ("bus", int(idx) + 1) for idx in net.bus.index}


def _pp_node_nominal_kv_map(net: PandapowerNetwork) -> dict[NodeKey, float]:
    return {("bus", int(idx) + 1): float(row.vn_kv) for idx, row in net.bus.iterrows()}


def _py_node_key_maps(net: pypowsybl.network.Network) -> dict[str, NodeKey]:
    element_key: dict[str, NodeKey] = {}
    buses = net.get_busbar_sections(attributes=[])
    if buses.empty:
        buses = net.get_bus_breaker_view_buses(attributes=[])
    for idx in buses.index:
        match = re.fullmatch(r"BUS-(\d+)", str(idx))
        if match:
            element_key[str(idx)] = ("bus", int(match.group(1)))
    return element_key


def _py_node_nominal_kv_map(net: pypowsybl.network.Network) -> dict[NodeKey, float]:
    buses = net.get_busbar_sections(attributes=["voltage_level_id"])
    if buses.empty:
        buses = net.get_bus_breaker_view_buses(attributes=["voltage_level_id"])
    voltage_levels = net.get_voltage_levels(attributes=["nominal_v"])

    nominal_map: dict[NodeKey, float] = {}
    for idx, row in buses.iterrows():
        match = re.fullmatch(r"BUS-(\d+)", str(idx))
        if match:
            nominal_map[("bus", int(match.group(1)))] = float(voltage_levels.loc[row["voltage_level_id"], "nominal_v"])
    return nominal_map


def _normalize_cgmes_origin_id(value: str) -> str:
    return value[1:] if value.startswith("_") else value


def _pp_branch_origin_id_key_map(net: PandapowerNetwork) -> dict[str, str]:
    element_key: dict[str, str] = {}
    for idx, row in net.line.dropna(subset=["origin_id"]).iterrows():
        element_key[get_globally_unique_id(idx, "line")] = _normalize_cgmes_origin_id(str(row.origin_id))
    for idx, row in net.trafo.dropna(subset=["origin_id"]).iterrows():
        element_key[get_globally_unique_id(idx, "trafo")] = _normalize_cgmes_origin_id(str(row.origin_id))
    for idx, row in net.trafo3w.dropna(subset=["origin_id"]).iterrows():
        element_key[get_globally_unique_id(idx, "trafo3w")] = _normalize_cgmes_origin_id(str(row.origin_id))
    return element_key


def _py_branch_origin_id_key_map(net: pypowsybl.network.Network) -> dict[str, str]:
    branch_ids = list(net.get_lines(attributes=[]).index)
    branch_ids += list(net.get_2_windings_transformers(attributes=[]).index)
    branch_ids += list(net.get_3_windings_transformers(attributes=[]).index)
    return {str(idx): _normalize_cgmes_origin_id(str(idx)) for idx in branch_ids}


def _pp_switch_origin_id_key_map(net: PandapowerNetwork) -> dict[str, str]:
    element_key: dict[str, str] = {}
    for idx, row in net.switch.dropna(subset=["origin_id"]).iterrows():
        element_key[get_globally_unique_id(idx, "switch")] = _normalize_cgmes_origin_id(str(row.origin_id))
    return element_key


def _external_cgmes_powsybl_switch_va_diff_frame(net: pypowsybl.network.Network) -> pd.DataFrame:
    switches = net.get_switches(attributes=["kind", "open", "retained", "bus_breaker_bus1_id", "bus_breaker_bus2_id"]).copy()
    if switches.empty:
        return pd.DataFrame(columns=["contingency_key", "element_key", "va_diff"])

    switches = switches[(switches["kind"] == "BREAKER") & switches["open"] & switches["retained"]].copy()
    if switches.empty:
        return pd.DataFrame(columns=["contingency_key", "element_key", "va_diff"])

    bus_breaker_buses = net.get_bus_breaker_view_buses(attributes=["v_angle"]).copy()
    switches = switches.join(bus_breaker_buses[["v_angle"]], on="bus_breaker_bus1_id").rename(
        columns={"v_angle": "v_angle_1"}
    )
    switches = switches.join(bus_breaker_buses[["v_angle"]], on="bus_breaker_bus2_id").rename(
        columns={"v_angle": "v_angle_2"}
    )
    switches = switches.dropna(subset=["v_angle_1", "v_angle_2"]).copy()
    if switches.empty:
        return pd.DataFrame(columns=["contingency_key", "element_key", "va_diff"])

    return pd.DataFrame(
        {
            "contingency_key": "BASECASE",
            "element_key": switches.index.map(str).map(_normalize_cgmes_origin_id),
            "va_diff": switches["v_angle_1"] - switches["v_angle_2"],
        }
    )


def _pp_external_cgmes_slack_candidates(net: PandapowerNetwork) -> list[dict[str, str | int | None]]:
    slack_candidates: list[dict[str, str | int | None]] = []
    if "slack" not in net.gen.columns or "origin_id" not in net.gen.columns or "origin_id" not in net.bus.columns:
        return slack_candidates

    for _, row in net.gen.loc[net.gen["slack"] == True].iterrows():
        bus_idx = int(row.bus)
        slack_candidates.append(
            {
                "type": "gen",
                "name": None if pd.isna(row.get("name")) else str(row.get("name")),
                "bus_idx": bus_idx,
                "bus_origin_id": _normalize_cgmes_origin_id(str(net.bus.at[bus_idx, "origin_id"])),
                "element_origin_id": _normalize_cgmes_origin_id(str(row.origin_id)),
            }
        )
    return slack_candidates


def _external_cgmes_slack_diagnostics(
    pandapower_net: PandapowerNetwork,
    powsybl_net: pypowsybl.network.Network,
) -> dict[str, object]:
    pandapower_slack_candidates = _pp_external_cgmes_slack_candidates(pandapower_net)
    powsybl_loadflow_result = pypowsybl.loadflow.run_ac(powsybl_net, parameters=_external_cgmes_powsybl_params())[0]
    slack_bus_results = getattr(powsybl_loadflow_result, "slack_bus_results", [])
    powsybl_slack_bus_ids = [str(item.id) for item in slack_bus_results]
    powsybl_reference_bus_id = (
        None if powsybl_loadflow_result.reference_bus_id is None else str(powsybl_loadflow_result.reference_bus_id)
    )

    buses = powsybl_net.get_bus_breaker_view_buses(attributes=["voltage_level_id"])
    if buses.empty:
        buses = powsybl_net.get_busbar_sections(attributes=["voltage_level_id"])
    slack_voltage_level_ids = [
        str(buses.loc[bus_id, "voltage_level_id"]) for bus_id in powsybl_slack_bus_ids if bus_id in buses.index
    ]

    generators = powsybl_net.get_generators(attributes=["bus_id", "voltage_level_id"])
    powsybl_generators_at_slack_bus = sorted(
        str(generator_id) for generator_id, row in generators.iterrows() if str(row["bus_id"]) in powsybl_slack_bus_ids
    )
    powsybl_generators_at_slack_voltage_level = sorted(
        str(generator_id)
        for generator_id, row in generators.iterrows()
        if str(row["voltage_level_id"]) in slack_voltage_level_ids
    )
    pandapower_slack_generator_ids = sorted(
        candidate["element_origin_id"]
        for candidate in pandapower_slack_candidates
        if candidate["element_origin_id"] is not None
    )
    slack_overlap_with_pandapower = sorted(
        set(pandapower_slack_generator_ids).intersection(
            powsybl_generators_at_slack_bus + powsybl_generators_at_slack_voltage_level
        )
    )

    powsybl_slack_identity = sorted(set(powsybl_slack_bus_ids + slack_voltage_level_ids + powsybl_generators_at_slack_bus))
    pandapower_slack_identity = sorted(
        {
            str(candidate["bus_origin_id"])
            for candidate in pandapower_slack_candidates
            if candidate["bus_origin_id"] is not None
        }.union(
            {
                str(candidate["element_origin_id"])
                for candidate in pandapower_slack_candidates
                if candidate["element_origin_id"] is not None
            }
        )
    )

    return {
        "pandapower_slack_candidates": pandapower_slack_candidates,
        "pandapower_slack_generator_ids": pandapower_slack_generator_ids,
        "powsybl_reference_bus_id": powsybl_reference_bus_id,
        "powsybl_slack_bus_ids": powsybl_slack_bus_ids,
        "powsybl_reference_equals_slack": powsybl_reference_bus_id in powsybl_slack_bus_ids,
        "powsybl_slack_voltage_level_ids": slack_voltage_level_ids,
        "powsybl_generators_at_slack_bus": powsybl_generators_at_slack_bus,
        "powsybl_generators_at_slack_voltage_level": powsybl_generators_at_slack_voltage_level,
        "slack_overlap_with_pandapower": slack_overlap_with_pandapower,
        "matches_pandapower_slack_family": bool(slack_overlap_with_pandapower),
        "is_identical": bool(set(pandapower_slack_identity).intersection(powsybl_slack_identity)),
    }


def _apply_external_cgmes_slack_terminal(net: pypowsybl.network.Network, generator_id: str) -> None:
    generators = net.get_generators(attributes=["bus_id", "voltage_level_id"])
    row = generators.loc[generator_id]
    pypowsybl.network.Network.create_extensions(
        net,
        extension_name="slackTerminal",
        voltage_level_id=str(row["voltage_level_id"]),
        bus_id=str(row["bus_id"]),
    )


def _external_cgmes_powsybl_params() -> Parameters:
    provider_parameters = dict(OPENLOADFLOW_PARAM_PF)
    provider_parameters.update(EXTERNAL_CGMES_PROVIDER_OVERRIDES)
    return Parameters(
        balance_type=UNIFORM_SINGLE_SLACK.balance_type,
        component_mode=UNIFORM_SINGLE_SLACK.component_mode,
        countries_to_balance=None,
        dc_power_factor=UNIFORM_SINGLE_SLACK.dc_power_factor,
        dc_use_transformer_ratio=UNIFORM_SINGLE_SLACK.dc_use_transformer_ratio,
        distributed_slack=UNIFORM_SINGLE_SLACK.distributed_slack,
        phase_shifter_regulation_on=UNIFORM_SINGLE_SLACK.phase_shifter_regulation_on,
        provider_parameters=provider_parameters,
        read_slack_bus=UNIFORM_SINGLE_SLACK.read_slack_bus,
        shunt_compensator_voltage_control_on=UNIFORM_SINGLE_SLACK.shunt_compensator_voltage_control_on,
        transformer_voltage_control_on=UNIFORM_SINGLE_SLACK.transformer_voltage_control_on,
        use_reactive_limits=UNIFORM_SINGLE_SLACK.use_reactive_limits,
        twt_split_shunt_admittance=UNIFORM_SINGLE_SLACK.twt_split_shunt_admittance,
        voltage_init_mode=UNIFORM_SINGLE_SLACK.voltage_init_mode,
        write_slack_bus=UNIFORM_SINGLE_SLACK.write_slack_bus,
    )


def _branch_contingency_with_all_monitored(full_definition: Nminus1Definition) -> Nminus1Definition:
    monitored_elements = [
        element for element in full_definition.monitored_elements if element.kind in {"branch", "bus", "switch"}
    ]
    contingencies = [
        contingency
        for contingency in full_definition.contingencies
        if contingency.is_basecase() or (len(contingency.elements) == 1 and contingency.elements[0].kind == "branch")
    ]
    return Nminus1Definition(
        monitored_elements=monitored_elements,
        contingencies=contingencies,
        id_type=full_definition.id_type,
    )


def _basecase_with_all_monitored(full_definition: Nminus1Definition) -> Nminus1Definition:
    monitored_elements = [
        element for element in full_definition.monitored_elements if element.kind in {"branch", "bus", "switch"}
    ]
    contingencies = [contingency for contingency in full_definition.contingencies if contingency.is_basecase()]
    return Nminus1Definition(
        monitored_elements=monitored_elements,
        contingencies=contingencies,
        id_type=full_definition.id_type,
    )


def _aggregate_branch_frame(
    loadflow_results: LoadflowResultsPolars,
    element_key_map: dict[str, BranchKey],
    contingency_key_map: dict[str, ContingencyKey],
) -> pd.DataFrame:
    branch_results = loadflow_results.branch_results.collect().to_pandas()
    branch_results["element_key"] = branch_results["element"].map(element_key_map)
    branch_results["contingency_key"] = branch_results["contingency"].map(contingency_key_map)
    branch_results = branch_results[branch_results["element_key"].notna() & branch_results["contingency_key"].notna()].copy()
    branch_results["abs_p"] = branch_results["p"].abs()
    branch_results["abs_q"] = branch_results["q"].abs()
    return (
        branch_results.groupby(["contingency_key", "element_key"], dropna=False)
        .agg(max_abs_p=("abs_p", "max"), max_abs_q=("abs_q", "max"))
        .reset_index()
    )


def _aggregate_node_frame(
    loadflow_results: LoadflowResultsPolars,
    element_key_map: dict[str, NodeKey],
    contingency_key_map: dict[str, ContingencyKey],
) -> pd.DataFrame:
    node_results = loadflow_results.node_results.collect().to_pandas()
    node_results["element_key"] = node_results["element"].map(element_key_map)
    node_results["contingency_key"] = node_results["contingency"].map(contingency_key_map)
    node_results = node_results[node_results["element_key"].notna() & node_results["contingency_key"].notna()].copy()
    return node_results[["contingency_key", "element_key", "vm", "va"]]


def _aggregate_va_diff_frame(
    loadflow_results: LoadflowResultsPolars,
    element_key_map: dict[str, BranchKey],
    contingency_key_map: dict[str, ContingencyKey],
) -> pd.DataFrame:
    va_diff_results = loadflow_results.va_diff_results.collect().to_pandas()
    va_diff_results["element_key"] = va_diff_results["element"].map(element_key_map)
    va_diff_results["contingency_key"] = va_diff_results["contingency"].map(contingency_key_map)
    return va_diff_results[va_diff_results["element_key"].notna() & va_diff_results["contingency_key"].notna()].copy()


def _build_shared_backend_comparison(
    source_net: PandapowerNetwork,
    *,
    basecase_only: bool,
) -> dict[str, pd.DataFrame]:
    with tempfile.TemporaryDirectory() as temp_dir:
        matpower_path = Path(temp_dir) / "shared_backend_case.mat"
        pandapower.rundcpp(source_net)
        to_mpc(source_net, matpower_path)

        filesystem = LocalFileSystem()
        pandapower_net = load_pandapower_from_fs(filesystem, matpower_path)
        powsybl_net = load_powsybl_from_fs(filesystem, matpower_path)

        pandapower_definition_full = get_full_nminus1_definition_pandapower(pandapower_net)
        powsybl_definition_full = get_full_nminus1_definition_powsybl(powsybl_net)
        selector = _basecase_with_all_monitored if basecase_only else _branch_contingency_with_all_monitored
        pandapower_definition = selector(pandapower_definition_full)
        powsybl_definition = selector(powsybl_definition_full)

        pandapower_branch_keys = _pp_branch_key_maps(pandapower_net)
        powsybl_branch_keys = _py_branch_key_maps(powsybl_net)
        pandapower_node_keys = _pp_node_key_maps(pandapower_net)
        powsybl_node_keys = _py_node_key_maps(powsybl_net)
        pandapower_node_nominal = _pp_node_nominal_kv_map(pandapower_net)
        powsybl_node_nominal = _py_node_nominal_kv_map(powsybl_net)

        pandapower_contingency_keys: dict[str, ContingencyKey] = {"BASECASE": "BASECASE"}
        powsybl_contingency_keys: dict[str, ContingencyKey] = {"BASECASE": "BASECASE"}
        for contingency in pandapower_definition.contingencies:
            if not contingency.is_basecase():
                pandapower_contingency_keys[contingency.id] = pandapower_branch_keys[contingency.elements[0].id]
        for contingency in powsybl_definition.contingencies:
            if not contingency.is_basecase():
                powsybl_contingency_keys[contingency.id] = powsybl_branch_keys[contingency.elements[0].id]

        pandapower_results = get_ac_loadflow_results(
            pandapower_net,
            pandapower_definition,
            job_id="pp_compare",
            n_processes=1,
            lf_params=PANDAPOWER_LOADFLOW_PARAM_PPCI,
        )
        powsybl_results = get_ac_loadflow_results(
            powsybl_net,
            powsybl_definition,
            job_id="py_compare",
            n_processes=1,
            lf_params=UNIFORM_SINGLE_SLACK,
        )

    merged_branch_results = _aggregate_branch_frame(
        pandapower_results,
        pandapower_branch_keys,
        pandapower_contingency_keys,
    ).merge(
        _aggregate_branch_frame(
            powsybl_results,
            powsybl_branch_keys,
            powsybl_contingency_keys,
        ),
        on=["contingency_key", "element_key"],
        suffixes=("_pp", "_py"),
    )
    merged_node_results = _aggregate_node_frame(
        pandapower_results,
        pandapower_node_keys,
        pandapower_contingency_keys,
    ).merge(
        _aggregate_node_frame(
            powsybl_results,
            powsybl_node_keys,
            powsybl_contingency_keys,
        ),
        on=["contingency_key", "element_key"],
        suffixes=("_pp", "_py"),
    )
    merged_node_results["nominal_vm_pp"] = merged_node_results["element_key"].map(pandapower_node_nominal)
    merged_node_results["nominal_vm_py"] = merged_node_results["element_key"].map(powsybl_node_nominal)
    merged_node_results["vm_pu_pp"] = merged_node_results["vm_pp"] / merged_node_results["nominal_vm_pp"]
    merged_node_results["vm_pu_py"] = merged_node_results["vm_py"] / merged_node_results["nominal_vm_py"]
    pandapower_va_diff_results = _aggregate_va_diff_frame(
        pandapower_results,
        pandapower_branch_keys,
        pandapower_contingency_keys,
    )
    powsybl_va_diff_results = _aggregate_va_diff_frame(
        powsybl_results,
        powsybl_branch_keys,
        powsybl_contingency_keys,
    )

    pandapower_converged = pandapower_results.converged.collect().to_pandas()
    pandapower_converged["contingency_key"] = pandapower_converged["contingency"].map(pandapower_contingency_keys)
    powsybl_converged = powsybl_results.converged.collect().to_pandas()
    powsybl_converged["contingency_key"] = powsybl_converged["contingency"].map(powsybl_contingency_keys)
    merged_convergence = pandapower_converged[["contingency_key", "status"]].merge(
        powsybl_converged[["contingency_key", "status"]],
        on="contingency_key",
        suffixes=("_pp", "_py"),
    )

    return {
        "branch": merged_branch_results,
        "node": merged_node_results,
        "pandapower_va_diff": pandapower_va_diff_results,
        "powsybl_va_diff": powsybl_va_diff_results,
        "convergence": merged_convergence,
    }


def _build_external_cgmes_backend_basecase_comparison(grid_name: str, grid_path: Path) -> dict[str, object]:
    pandapower_net = from_cim.from_cim(str(grid_path))
    powsybl_net = pypowsybl.network.load(str(grid_path))
    forced_slack_generator_id = EXTERNAL_CGMES_BASECASE_EXPECTATIONS[grid_name]["forced_slack_generator_id"]
    _apply_external_cgmes_slack_terminal(powsybl_net, forced_slack_generator_id)

    pandapower_definition = _basecase_with_all_monitored(get_full_nminus1_definition_pandapower(pandapower_net))
    powsybl_definition = _basecase_with_all_monitored(get_full_nminus1_definition_powsybl(powsybl_net))

    pandapower_results = get_ac_loadflow_results(
        pandapower_net,
        pandapower_definition,
        job_id="pp_compare_external_cgmes",
        n_processes=1,
        lf_params=PANDAPOWER_LOADFLOW_PARAM_PPCI,
    )
    powsybl_results = get_ac_loadflow_results(
        powsybl_net,
        powsybl_definition,
        job_id="py_compare_external_cgmes",
        n_processes=1,
        lf_params=_external_cgmes_powsybl_params(),
    )

    contingency_keys = {"BASECASE": "BASECASE"}
    merged_branch_results = _aggregate_branch_frame(
        pandapower_results,
        _pp_branch_origin_id_key_map(pandapower_net),
        contingency_keys,
    ).merge(
        _aggregate_branch_frame(
            powsybl_results,
            _py_branch_origin_id_key_map(powsybl_net),
            contingency_keys,
        ),
        on=["contingency_key", "element_key"],
        suffixes=("_pp", "_py"),
    )
    pandapower_switch_va_diff_results = _aggregate_va_diff_frame(
        pandapower_results,
        _pp_switch_origin_id_key_map(pandapower_net),
        contingency_keys,
    )
    powsybl_switch_va_diff_results = _external_cgmes_powsybl_switch_va_diff_frame(powsybl_net)
    merged_switch_va_diff_results = pandapower_switch_va_diff_results[["contingency_key", "element_key", "va_diff"]].merge(
        powsybl_switch_va_diff_results,
        on=["contingency_key", "element_key"],
        suffixes=("_pp", "_py"),
    )

    pandapower_converged = pandapower_results.converged.collect().to_pandas()
    pandapower_converged["contingency_key"] = pandapower_converged["contingency"].map(contingency_keys)
    powsybl_converged = powsybl_results.converged.collect().to_pandas()
    powsybl_converged["contingency_key"] = powsybl_converged["contingency"].map(contingency_keys)
    merged_convergence = pandapower_converged[["contingency_key", "status"]].merge(
        powsybl_converged[["contingency_key", "status"]],
        on="contingency_key",
        suffixes=("_pp", "_py"),
    )

    slack = _external_cgmes_slack_diagnostics(pandapower_net, powsybl_net)

    return {
        "branch": merged_branch_results,
        "convergence": merged_convergence,
        "pandapower_switch_va_diff": pandapower_switch_va_diff_results,
        "powsybl_switch_va_diff": powsybl_switch_va_diff_results,
        "switch_va_diff": merged_switch_va_diff_results,
        "slack": slack,
    }


@pytest.fixture(scope="module")
def shared_ieee14_backend_comparison() -> dict[str, pd.DataFrame]:
    return _build_shared_backend_comparison(networks.case14(), basecase_only=False)


@pytest.fixture(scope="module", params=["case57", "case300"])
def shared_large_grid_backend_basecase_comparison(request: pytest.FixtureRequest) -> tuple[str, dict[str, pd.DataFrame]]:
    network_factory = {
        "case57": networks.case57,
        "case300": networks.case300,
    }[request.param]
    return request.param, _build_shared_backend_comparison(network_factory(), basecase_only=True)


@pytest.fixture(scope="module", params=["external_cgmes_basecase"])
def shared_external_cgmes_backend_basecase_comparison(request: pytest.FixtureRequest) -> tuple[str, dict[str, object]]:
    grid_name = request.param
    grid_path_str = EXTERNAL_CGMES_BASECASE_EXPECTATIONS[grid_name]["path"]
    if not grid_path_str:
        pytest.skip("External CGMES grid path not configured.")
    grid_path = Path(grid_path_str)
    if not grid_path.exists():
        pytest.skip(f"External CGMES grid not available: {grid_path}")
    return grid_name, _build_external_cgmes_backend_basecase_comparison(grid_name, grid_path)


def test_shared_ieee14_backend_comparison_has_matching_convergence_status(
    shared_ieee14_backend_comparison: dict[str, pd.DataFrame],
) -> None:
    merged_convergence = shared_ieee14_backend_comparison["convergence"]

    assert not merged_convergence.empty

    mismatches = merged_convergence[merged_convergence["status_pp"] != merged_convergence["status_py"]]
    assert mismatches.empty


def test_shared_ieee14_backend_comparison_has_small_branch_deviations(
    shared_ieee14_backend_comparison: dict[str, pd.DataFrame],
) -> None:
    merged_branch_results = shared_ieee14_backend_comparison["branch"].copy()

    assert not merged_branch_results.empty

    merged_branch_results["max_abs_p_diff"] = (
        merged_branch_results["max_abs_p_pp"] - merged_branch_results["max_abs_p_py"]
    ).abs()
    merged_branch_results["max_abs_q_diff"] = (
        merged_branch_results["max_abs_q_pp"] - merged_branch_results["max_abs_q_py"]
    ).abs()

    basecase_rows = merged_branch_results[merged_branch_results["contingency_key"] == "BASECASE"]
    assert not basecase_rows.empty
    assert basecase_rows["max_abs_p_diff"].max() < 1e-3
    assert basecase_rows["max_abs_q_diff"].max() < 1e-3
    assert merged_branch_results["max_abs_p_diff"].max() < 1e-3
    assert merged_branch_results["max_abs_q_diff"].max() < 1e-3


def test_shared_ieee14_backend_comparison_has_small_node_deviations(
    shared_ieee14_backend_comparison: dict[str, pd.DataFrame],
) -> None:
    merged_node_results = shared_ieee14_backend_comparison["node"].copy()

    assert not merged_node_results.empty

    merged_node_results["vm_diff"] = (merged_node_results["vm_pu_pp"] - merged_node_results["vm_pu_py"]).abs()
    merged_node_results["va_diff"] = (merged_node_results["va_pp"] - merged_node_results["va_py"]).abs()

    basecase_rows = merged_node_results[merged_node_results["contingency_key"] == "BASECASE"]
    assert not basecase_rows.empty
    assert basecase_rows["vm_diff"].max() < 1e-3
    assert basecase_rows["va_diff"].max() < 1e-3
    assert merged_node_results["vm_diff"].max() < 1e-3
    assert merged_node_results["va_diff"].max() < 1e-3


def test_shared_ieee14_backend_comparison_has_asymmetric_va_diff_results(
    shared_ieee14_backend_comparison: dict[str, pd.DataFrame],
) -> None:
    pandapower_va_diff_results = shared_ieee14_backend_comparison["pandapower_va_diff"]
    powsybl_va_diff_results = shared_ieee14_backend_comparison["powsybl_va_diff"]

    assert pandapower_va_diff_results.empty
    assert not powsybl_va_diff_results.empty


def test_shared_large_grid_backend_basecase_has_matching_convergence_status(
    shared_large_grid_backend_basecase_comparison: tuple[str, dict[str, pd.DataFrame]],
) -> None:
    _, comparison = shared_large_grid_backend_basecase_comparison
    merged_convergence = comparison["convergence"]

    assert not merged_convergence.empty

    mismatches = merged_convergence[merged_convergence["status_pp"] != merged_convergence["status_py"]]
    assert mismatches.empty


def test_shared_large_grid_backend_basecase_deviations_are_bounded(
    shared_large_grid_backend_basecase_comparison: tuple[str, dict[str, pd.DataFrame]],
) -> None:
    grid_name, comparison = shared_large_grid_backend_basecase_comparison
    expectations = LARGE_GRID_EXPECTATIONS[grid_name]

    merged_branch_results = comparison["branch"].copy()
    merged_node_results = comparison["node"].copy()

    assert not merged_branch_results.empty
    assert not merged_node_results.empty

    merged_branch_results["max_abs_p_diff"] = (
        merged_branch_results["max_abs_p_pp"] - merged_branch_results["max_abs_p_py"]
    ).abs()
    merged_branch_results["max_abs_q_diff"] = (
        merged_branch_results["max_abs_q_pp"] - merged_branch_results["max_abs_q_py"]
    ).abs()
    merged_node_results["vm_diff"] = (merged_node_results["vm_pu_pp"] - merged_node_results["vm_pu_py"]).abs()
    merged_node_results["va_diff"] = (merged_node_results["va_pp"] - merged_node_results["va_py"]).abs()

    assert merged_branch_results["max_abs_p_diff"].max() < expectations["branch_p_max"]
    assert merged_branch_results["max_abs_q_diff"].max() < expectations["branch_q_max"]
    assert merged_node_results["vm_diff"].max() < expectations["node_vm_pu_max"]
    assert merged_node_results["va_diff"].max() < expectations["node_va_max"]


def test_shared_external_cgmes_backend_basecase_has_matching_convergence_status(
    shared_external_cgmes_backend_basecase_comparison: tuple[str, dict[str, object]],
) -> None:
    _, comparison = shared_external_cgmes_backend_basecase_comparison
    merged_convergence = comparison["convergence"]

    assert not merged_convergence.empty

    mismatches = merged_convergence[merged_convergence["status_pp"] != merged_convergence["status_py"]]
    assert mismatches.empty


def test_shared_external_cgmes_backend_basecase_branch_deviations_are_bounded(
    shared_external_cgmes_backend_basecase_comparison: tuple[str, dict[str, object]],
) -> None:
    grid_name, comparison = shared_external_cgmes_backend_basecase_comparison
    expectations = EXTERNAL_CGMES_BASECASE_EXPECTATIONS[grid_name]

    merged_branch_results = comparison["branch"].copy()

    assert not merged_branch_results.empty

    merged_branch_results["max_abs_p_diff"] = (
        merged_branch_results["max_abs_p_pp"] - merged_branch_results["max_abs_p_py"]
    ).abs()
    merged_branch_results["max_abs_q_diff"] = (
        merged_branch_results["max_abs_q_pp"] - merged_branch_results["max_abs_q_py"]
    ).abs()

    assert merged_branch_results["max_abs_p_diff"].max() < expectations["branch_p_max"]
    assert merged_branch_results["max_abs_q_diff"].max() < expectations["branch_q_max"]


def test_shared_external_cgmes_backend_basecase_reports_slack_identity(
    shared_external_cgmes_backend_basecase_comparison: tuple[str, dict[str, object]],
) -> None:
    _, comparison = shared_external_cgmes_backend_basecase_comparison
    slack = comparison["slack"]

    assert slack["pandapower_slack_candidates"]
    assert slack["powsybl_reference_bus_id"] is not None
    assert slack["powsybl_slack_bus_ids"]
    assert slack["powsybl_reference_equals_slack"] is True
    assert slack["matches_pandapower_slack_family"] is True
    assert slack["slack_overlap_with_pandapower"]


def test_shared_external_cgmes_backend_basecase_switch_va_diff_deviations_are_bounded(
    shared_external_cgmes_backend_basecase_comparison: tuple[str, dict[str, object]],
) -> None:
    grid_name, comparison = shared_external_cgmes_backend_basecase_comparison
    expectations = EXTERNAL_CGMES_BASECASE_EXPECTATIONS[grid_name]

    pandapower_switch_va_diff = comparison["pandapower_switch_va_diff"].copy()
    powsybl_switch_va_diff = comparison["powsybl_switch_va_diff"].copy()
    merged_switch_va_diff = comparison["switch_va_diff"].copy()

    assert not pandapower_switch_va_diff.empty
    assert not powsybl_switch_va_diff.empty
    assert not merged_switch_va_diff.empty

    merged_switch_va_diff["abs_va_diff_pp"] = merged_switch_va_diff["va_diff_pp"].abs()
    merged_switch_va_diff["abs_va_diff_py"] = merged_switch_va_diff["va_diff_py"].abs()
    merged_switch_va_diff["abs_va_diff_diff"] = (
        merged_switch_va_diff["abs_va_diff_pp"] - merged_switch_va_diff["abs_va_diff_py"]
    ).abs()

    assert merged_switch_va_diff["abs_va_diff_diff"].max() < expectations["switch_va_diff_abs_max"]
