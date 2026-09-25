# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Tests that rows controllers record in ``net.controller_data`` come back as ``controller_results``.

The MCCS pandapower fork's DiscreteTapControl always writes those rows. Stock
pandapower never does, so these tests use a small controller that writes the same keys. What is under test
is the engine's side of the contract: rows are picked up per outage, rows inherited from the parent net
are dropped, and every row is stamped with the contingency and timestep it belongs to - and nothing else.
"""

import pandapower as pp
import pandas as pd
import pytest
from pandapower.control.basic_controller import Controller
from toop_engine_contingency_analysis.pandapower.contingency_analysis_pandapower import (
    run_contingency_analysis_pandapower,
)
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import ContingencyAnalysisConfig
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_interfaces.loadflow_results import LoadflowResults
from toop_engine_interfaces.loadflow_results_polars import ControllerResultSchemaPolars, LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import (
    Contingency,
    GridElement,
    MonitoredElement,
    Nminus1Definition,
)

BASECASE = "BASECASE"


class _RecordingController(Controller):
    """Writes one row per run_control call, under the keys the pandapower fork uses."""

    def __init__(self, net: pp.pandapowerNet) -> None:
        super().__init__(net, in_service=True)
        self.recorded = False

    def initialize_control(self, _net: pp.pandapowerNet) -> None:
        self.recorded = False

    def is_converged(self, _net: pp.pandapowerNet) -> bool:
        return self.recorded

    def control_step(self, net: pp.pandapowerNet) -> None:
        rows = net.get("_controller_data_rows")
        if not isinstance(rows, list):
            rows = []
            net["_controller_data_rows"] = rows
        rows.append(_controller_row(float(net.res_bus.at[0, "vm_pu"])))
        net["controller_data"] = pd.DataFrame(rows)
        self.recorded = True


def _controller_row(vm_pu: float) -> dict:
    """A final row with every ControllerResultSchema column, typed as the pandapower fork writes it.

    Returns
    -------
    dict
        One row of ``net.controller_data``.
    """
    return {
        "row_type": "final",
        "element": "trafo",
        "tap_control_method": "direct",
        "tap_step_method": "tap_step_percent",
        "tap_direction_method": "pandapower",
        "controlled_bus": 0,
        "vn_kv": 135.0,
        "vm_set_pu": 1.06,
        "vm_lower_pu": 1.05,
        "vm_upper_pu": 1.07,
        "tap_min": -9.0,
        "tap_max": 9.0,
        "control_step": 1,
        "vm_start_pu": vm_pu,
        "vm_pu": vm_pu,
        "in_band": 1.05 <= vm_pu <= 1.07,
        "voltage_violation": 0.0,
        "voltage_region": 0.0,
        "tap_start": 0.0,
        "tap_pos_before": 0.0,
        "tap_pos_after": 0.0,
        "proposed_increment": float("nan"),
        "actual_increment": float("nan"),
        "controller_output": float("nan"),
        "tap_limit": None,
        "tap_limit_reached": False,
        "total_tap_changes": 0.0,
        "hunting_transition_detected": False,
        "hunting_counter": 0,
        "hunting_limit_reached": False,
        "hunting_detected": False,
    }


@pytest.fixture
def nminus1_definition() -> Nminus1Definition:
    """case14 with the basecase and one contingency per line.

    Returns
    -------
    Nminus1Definition
        Monitored buses, the basecase and one contingency per line.
    """
    net = pp.networks.case14()
    return Nminus1Definition(
        monitored_elements=[
            MonitoredElement(id=get_globally_unique_id(index, "bus"), name=str(index), kind="bus", type="bus")
            for index in net.bus.index
        ],
        contingencies=[Contingency(id=BASECASE, elements=[])]
        + [
            Contingency(
                id=str(index),
                elements=[
                    GridElement(id=get_globally_unique_id(index, "line"), name=str(index), kind="branch", type="line")
                ],
            )
            for index in net.line.index
        ],
    )


def _run(
    nminus1_definition: Nminus1Definition, with_controller: bool, polars: bool = True
) -> LoadflowResultsPolars | LoadflowResults:
    net = pp.networks.case14()
    if with_controller:
        _RecordingController(net)
    return run_contingency_analysis_pandapower(
        net=net,
        n_minus_1_definition=nminus1_definition,
        job_id="test_job",
        timestep=3,
        cfg=ContingencyAnalysisConfig(method="ac", polars=polars, runpp_kwargs={"run_control": True}),
    )


def test_controller_rows_come_back_stamped_per_contingency(nminus1_definition: Nminus1Definition) -> None:
    """Every outage returns its own rows, stamped with its contingency and the run's timestep."""
    results = _run(nminus1_definition, with_controller=True)

    assert results.controller_results is not None
    rows = results.controller_results.collect()

    expected = {c.id for c in nminus1_definition.contingencies}
    assert set(rows["contingency"].to_list()) == expected
    assert set(rows["timestep"].to_list()) == {3}
    assert "contingency_name" not in rows.columns, "ControllerResultSchema does not carry contingency names"


def test_rows_inherited_from_the_parent_net_are_dropped(nminus1_definition: Nminus1Definition) -> None:
    """The base-case load flow also records a row on the parent net; outages must not carry it along."""
    rows = _run(nminus1_definition, with_controller=True).controller_results.collect()

    per_contingency = rows.group_by("contingency").len()
    assert per_contingency["len"].to_list() == [1] * per_contingency.height, (
        "each outage recorded exactly one row; more means the parent's rows leaked into the copy"
    )


def test_no_controller_data_means_no_controller_results(nminus1_definition: Nminus1Definition) -> None:
    """Without anything writing controller data, the field stays None."""
    assert _run(nminus1_definition, with_controller=False).controller_results is None


def test_controller_results_match_the_polars_schema(nminus1_definition: Nminus1Definition) -> None:
    """The collected rows validate against ControllerResultSchemaPolars, timestep included as int64."""
    results = _run(nminus1_definition, with_controller=True)

    ControllerResultSchemaPolars.validate(results.controller_results).collect()


def test_controller_results_survive_the_pandas_conversion(nminus1_definition: Nminus1Definition) -> None:
    """polars=False returns the rows indexed by timestep and contingency, validated by ControllerResultSchema."""
    results = _run(nminus1_definition, with_controller=True, polars=False)

    assert isinstance(results.controller_results, pd.DataFrame)
    assert results.controller_results.index.names == ["timestep", "contingency"]
    contingencies = set(results.controller_results.index.get_level_values("contingency"))
    assert contingencies == {c.id for c in nminus1_definition.contingencies}
