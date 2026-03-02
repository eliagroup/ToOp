# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import numpy as np
import pandapower as pp
import pandas as pd
import pandera as pa
import pytest
from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_contingency_analysis.pandapower import get_full_nminus1_definition_pandapower
from toop_engine_contingency_analysis.pandapower.contingency_analysis_pandapower import (
    run_contingency_analysis_pandapower,
    update_results_with_names,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    ContingencyAnalysisConfig,
    PandapowerContingency,
)
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_interfaces.loadflow_result_helpers import extract_branch_results
from toop_engine_interfaces.loadflow_results import RegulatingElementType
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, Nminus1Definition


def test_run_ac_contingency_analysis_pandapower(pandapower_net: pp.pandapowerNet, init_ray) -> None:
    nminus1_definition = get_full_nminus1_definition_pandapower(pandapower_net)
    with pa.config.config_context(validation_enabled=True, validation_depth=pa.config.ValidationDepth.SCHEMA_AND_DATA):
        lf_result_sequential_polars = get_ac_loadflow_results(
            pandapower_net, nminus1_definition, job_id="test_job", n_processes=1
        )
    with pa.config.config_context(validation_enabled=False):
        lf_result_sequential_polars_no_val = get_ac_loadflow_results(
            pandapower_net, nminus1_definition, job_id="test_job", n_processes=1
        )
    assert lf_result_sequential_polars is not None
    assert lf_result_sequential_polars_no_val == lf_result_sequential_polars


def test_update_results_with_names_sets_missing_values_and_contingency_name() -> None:
    contingency = PandapowerContingency(unique_id="cont1", name="Contingency A", elements=[])

    branch_index = pd.MultiIndex.from_tuples(
        [(0, "cont1", "branch_1", 1), (0, "cont1", "branch_2", 1)],
        names=["timestep", "contingency", "element", "side"],
    )
    branch_results_df = pd.DataFrame(
        {
            "i": [1.0, 2.0],
            "p": [10.0, 20.0],
            "q": [0.1, 0.2],
            "loading": [50.0, 60.0],
            "element_name": ["", "Existing Branch"],
            "contingency_name": ["", ""],
        },
        index=branch_index,
    )

    node_index = pd.MultiIndex.from_tuples(
        [(0, "cont1", "node_1")],
        names=["timestep", "contingency", "element"],
    )
    node_results_df = pd.DataFrame(
        {
            "vm": [110.0],
            "vm_loading": [0.0],
            "va": [0.0],
            "p": [5.0],
            "q": [1.0],
            "vm_basecase_deviation": [0.0],
            "element_name": [np.nan],
            "contingency_name": [""],
        },
        index=node_index,
    )

    va_diff_index = pd.MultiIndex.from_tuples(
        [(0, "cont1", "va_1")],
        names=["timestep", "contingency", "element"],
    )
    va_diff_results = pd.DataFrame(
        {
            "va_diff": [1.5],
            "element_name": [""],
            "contingency_name": [""],
        },
        index=va_diff_index,
    )

    regulating_index = pd.MultiIndex.from_tuples(
        [(0, "cont1", "reg_1")],
        names=["timestep", "contingency", "element"],
    )
    regulating_elements_df = pd.DataFrame(
        {
            "value": [0.5],
            "regulating_element_type": [RegulatingElementType.OTHER.value],
            "element_name": [""],
            "contingency_name": [""],
        },
        index=regulating_index,
    )

    element_name_map = {
        "branch_1": "Branch 1",
        "branch_2": "Branch 2",
        "node_1": "Node 1",
        "va_1": "VA 1",
        "reg_1": "Reg 1",
    }

    regulating_elements_df, branch_results_df, node_results_df, va_diff_results = update_results_with_names(
        contingency,
        regulating_elements_df,
        branch_results_df,
        node_results_df,
        va_diff_results,
        element_name_map,
    )

    assert branch_results_df.loc[(0, "cont1", "branch_1", 1), "element_name"] == "Branch 1"
    assert branch_results_df.loc[(0, "cont1", "branch_2", 1), "element_name"] == "Existing Branch"
    assert node_results_df.loc[(0, "cont1", "node_1"), "element_name"] == "Node 1"
    assert va_diff_results.loc[(0, "cont1", "va_1"), "element_name"] == "VA 1"
    assert regulating_elements_df.loc[(0, "cont1", "reg_1"), "element_name"] == "Reg 1"

    assert branch_results_df["contingency_name"].unique().tolist() == ["Contingency A"]
    assert node_results_df["contingency_name"].unique().tolist() == ["Contingency A"]
    assert va_diff_results["contingency_name"].unique().tolist() == ["Contingency A"]
    assert regulating_elements_df["contingency_name"].unique().tolist() == ["Contingency A"]


@pytest.mark.xdist_group("performance")
@pytest.mark.timeout(300)
@pytest.mark.skip(reason="Does not work on CI")
def test_run_ac_contingency_analysis_pandapower_mt(pandapower_net: pp.pandapowerNet, init_ray) -> None:
    nminus1_definition = get_full_nminus1_definition_pandapower(pandapower_net)

    lf_result_parallel_polars = get_ac_loadflow_results(pandapower_net, nminus1_definition, job_id="test_job", n_processes=2)
    assert lf_result_parallel_polars is not None


def test_extract_branch_results_pandapower_disconnected():
    net = pp.networks.case14()
    net.line.loc[0, "in_service"] = False  # Disconnect the first line
    nminus1_def = Nminus1Definition(
        monitored_elements=[
            GridElement(id=get_globally_unique_id(index, "line"), name=str(row.name), kind="branch", type="line")
            for index, row in net.line.iterrows()
        ],
        contingencies=[
            Contingency(
                id=str(index),
                elements=[
                    GridElement(id=get_globally_unique_id(index, "line"), name=str(row.name), kind="branch", type="line")
                ],
            )
            for index, row in net.line.iterrows()
        ],
    )
    cfg = ContingencyAnalysisConfig(method="dc")
    res = run_contingency_analysis_pandapower(
        net=net,
        n_minus_1_definition=nminus1_def,
        job_id="test_job",
        timestep=0,
        cfg=cfg,
    )
    contingencies = [contingency.id for contingency in nminus1_def.contingencies if not contingency.is_basecase()]
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


def test_extract_branch_results_pandapower_disconnected():
    net = pp.networks.case14()

    net.line.loc[0, "in_service"] = False  # Disconnect the first line
    nminus1_def = Nminus1Definition(
        monitored_elements=[
            GridElement(id=get_globally_unique_id(index, "line"), name=str(row.name), kind="branch", type="line")
            for index, row in net.line.iterrows()
        ],
        contingencies=[
            Contingency(
                id=str(index),
                elements=[
                    GridElement(id=get_globally_unique_id(index, "line"), name=str(row.name), kind="branch", type="line")
                ],
            )
            for index, row in net.line.iterrows()
        ],
    )

    cfg = ContingencyAnalysisConfig(method="dc")
    res = run_contingency_analysis_pandapower(
        net=net,
        n_minus_1_definition=nminus1_def,
        job_id="test_job",
        timestep=0,
        cfg=cfg,
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


def test_outage_grouping_combines_connected_elements_into_single_contingency():
    # ------------------------------------------------------------------
    # Build test network
    # ------------------------------------------------------------------
    net = pp.create_empty_network()

    # Slack side
    b_slack = pp.create_bus(net, vn_kv=110, name="bus_slack")
    b0 = pp.create_bus(net, vn_kv=110, name="bus0")
    pp.create_switch(net, b_slack, b0, et="b", closed=True, type="CB", name="switch1")

    # Remaining buses
    b1 = pp.create_bus(net, vn_kv=110, name="bus1")
    b2 = pp.create_bus(net, vn_kv=110, name="bus2")
    b3 = pp.create_bus(net, vn_kv=110, name="bus3")

    # Slack generator
    pp.create_gen(
        net,
        bus=b_slack,
        p_mw=0.0,
        vm_pu=1.02,
        slack=True,
        name="SlackGen@bus0",
    )

    # Loads
    pp.create_load(net, bus=b0, p_mw=10.0, q_mvar=10.0, name="Load@bus0")
    pp.create_load(net, bus=b1, p_mw=10.0, q_mvar=10.0, name="Load@bus1")

    # Lines
    l1 = pp.create_line_from_parameters(
        net,
        b0,
        b1,
        length_km=1,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.1,
        c_nf_per_km=0,
        max_i_ka=1,
        name="line0",
    )

    # Bus-bus switch between line sections
    pp.create_switch(net, b1, b2, et="b", closed=True, type="DS", name="switch0")

    l2 = pp.create_line_from_parameters(
        net,
        b2,
        b3,
        length_km=1,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.1,
        c_nf_per_km=0,
        max_i_ka=1,
        name="line1",
    )

    # ------------------------------------------------------------------
    # Define N-1: outage only for line1
    # ------------------------------------------------------------------
    nminus1_def = Nminus1Definition(
        monitored_elements=[
            GridElement(
                id=get_globally_unique_id(int(index), "line"),
                name=str(row.name),
                kind="branch",
                type="line",
            )
            for index, row in net.line.iterrows()
        ],
        contingencies=[
            Contingency(
                id=str(l2),
                elements=[
                    GridElement(
                        id=get_globally_unique_id(int(l2), "line"),
                        name=str(l2),
                        kind="branch",
                        type="line",
                    )
                ],
            )
        ],
    )

    # ------------------------------------------------------------------
    # Run WITHOUT outage grouping
    # ------------------------------------------------------------------
    cfg = ContingencyAnalysisConfig(method="ac", apply_outage_grouping=False)

    res = run_contingency_analysis_pandapower(
        net=net,
        n_minus_1_definition=nminus1_def,
        job_id="test_job",
        timestep=0,
        cfg=cfg,
    )

    branch_results = res.branch_results.reset_index()

    # line0 should still be energized
    l1_loading = branch_results.loc[branch_results.element == f"{l1}%%line", "loading"]
    assert not l1_loading.empty
    assert l1_loading.notna().all()

    # line1 is the outage -> loading must be NaN
    l2_loading = branch_results.loc[branch_results.element == f"{l2}%%line", "loading"]
    assert not l2_loading.empty
    assert l2_loading.isna().all()

    # ------------------------------------------------------------------
    # Run WITH outage grouping
    # ------------------------------------------------------------------
    cfg = ContingencyAnalysisConfig(method="ac", apply_outage_grouping=True)

    res = run_contingency_analysis_pandapower(
        net=net,
        n_minus_1_definition=nminus1_def,
        job_id="test_job",
        timestep=0,
        cfg=cfg,
    )

    branch_results = res.branch_results.reset_index()

    # both lines belong to the same connected outage group
    l1_loading = branch_results.loc[branch_results.element == f"{l1}%%line", "loading"]
    l2_loading = branch_results.loc[branch_results.element == f"{l2}%%line", "loading"]

    assert l1_loading.isna().all()
    assert l2_loading.isna().all()


def test_basecase_deviation_is_nan_when_basecase_fails_and_defined_when_basecase_converges():
    # ------------------------------------------------------------------
    # Build test network
    # ------------------------------------------------------------------
    net = pp.create_empty_network()

    b0 = pp.create_bus(net, vn_kv=110, name="bus0")
    b1 = pp.create_bus(net, vn_kv=110, name="bus1")
    b2 = pp.create_bus(net, vn_kv=110, name="bus2")

    # Two slack generators connected through a near-zero impedance branch
    # → intended to make BASECASE numerically unstable / fail
    pp.create_gen(net, bus=b0, p_mw=0.0, vm_pu=1.02, slack=True, name="SlackGen0")
    pp.create_gen(net, bus=b2, p_mw=0.0, vm_pu=1.01, slack=True, name="SlackGen1")

    pp.create_load(net, bus=b1, p_mw=50.0, q_mvar=20.0, name="Load")

    # Normal line
    pp.create_line_from_parameters(
        net,
        b0,
        b1,
        length_km=1,
        r_ohm_per_km=0.05,
        x_ohm_per_km=0.20,
        c_nf_per_km=0,
        max_i_ka=1,
        name="line01",
    )

    # Problematic near-zero impedance tie
    l2 = pp.create_line_from_parameters(
        net,
        b1,
        b2,
        length_km=1,
        r_ohm_per_km=1e-9,
        x_ohm_per_km=1e-9,
        c_nf_per_km=0,
        max_i_ka=1,
        name="line02",
    )

    # ------------------------------------------------------------------
    # Define N-1 case (outage of line02)
    # ------------------------------------------------------------------
    monitored_elements = [
        GridElement(
            id=get_globally_unique_id(int(index), "line"),
            name=str(row.name),
            kind="branch",
            type="line",
        )
        for index, row in net.line.iterrows()
    ]

    monitored_elements += [
        GridElement(
            id=get_globally_unique_id(int(index), "line"),
            name=str(row.name),
            kind="bus",
            type="bus",
        )
        for index, row in net.bus.iterrows()
    ]

    nminus1_def = Nminus1Definition(
        monitored_elements=monitored_elements,
        contingencies=[
            Contingency(
                id=str(l2),
                elements=[
                    GridElement(
                        id=get_globally_unique_id(int(l2), "line"),
                        name=str(l2),
                        kind="branch",
                        type="line",
                    )
                ],
            )
        ],
    )

    cfg = ContingencyAnalysisConfig(method="ac", apply_outage_grouping=False)

    # ------------------------------------------------------------------
    # Case 1: basecase fails -> deviation must be NaN
    # ------------------------------------------------------------------
    res = run_contingency_analysis_pandapower(
        net=net,
        n_minus_1_definition=nminus1_def,
        job_id="test_job",
        timestep=0,
        cfg=cfg,
    )

    node_results = res.node_results.reset_index()

    assert not node_results.empty
    assert node_results["vm_basecase_deviation"].isna().all()

    # ------------------------------------------------------------------
    # Case 2: make network numerically stable
    # (increase line impedances so basecase converges)
    # ------------------------------------------------------------------
    net.line["r_ohm_per_km"] = 0.02
    net.line["x_ohm_per_km"] = 0.02

    res = run_contingency_analysis_pandapower(
        net=net,
        n_minus_1_definition=nminus1_def,
        job_id="test_job",
        timestep=0,
        cfg=cfg,
    )

    node_results = res.node_results.reset_index()

    assert not node_results.empty
    assert node_results["vm_basecase_deviation"].notna().all()
