# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the MPL was not distributed with this file,
# You can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import pandapower
import pytest
from pypowsybl.network.impl.network import Network
from toop_engine_contingency_analysis.ac_loadflow_service import ac_loadflow_service
from toop_engine_interfaces.loadflow_results_polars import LoadflowResultsPolars
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, Nminus1Definition, SppsRule


def test_get_ac_loadflow_results_sanitizes_complex_definition_for_pandapower(
    pandapower_net: pandapower.pandapowerNet,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = Nminus1Definition(
        monitored_elements=[],
        contingencies=[
            Contingency(
                id="grouped_case",
                name="Grouped case",
                elements=[
                    GridElement(id="line_a", type="line", kind="branch"),
                    GridElement(id="line_b", type="line", kind="branch"),
                ],
            )
        ],
        spps_rules=[SppsRule(scheme_name="grouped_case", conditions=[], actions=[])],
        id_type="unique_pandapower",
    )
    captured: dict[str, Nminus1Definition] = {}
    result = LoadflowResultsPolars(job_id="test")

    def fake_run_contingency_analysis_pandapower(
        _net: pandapower.pandapowerNet,
        n_minus_1_definition: Nminus1Definition,
        _job_id: str,
        _timestep: int,
        cfg: object,
    ) -> LoadflowResultsPolars:
        del _net, _job_id, _timestep, cfg
        captured["definition"] = n_minus_1_definition
        return result

    monkeypatch.setattr(ac_loadflow_service, "run_contingency_analysis_pandapower", fake_run_contingency_analysis_pandapower)

    actual = ac_loadflow_service.get_ac_loadflow_results(
        net=pandapower_net,
        n_minus_1_definition=definition,
        sanitize_spps_rules=True,
    )

    assert actual is result
    sanitized_definition = captured["definition"]
    assert sanitized_definition.spps_rules is None
    assert sanitized_definition.id_type == definition.id_type
    assert sanitized_definition.contingencies == definition.contingencies
    assert sanitized_definition.contingencies is not definition.contingencies
    assert definition.spps_rules is not None


def test_get_ac_loadflow_results_keeps_spps_rules_by_default(
    pandapower_net: pandapower.pandapowerNet,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rule = SppsRule(scheme_name="case", conditions=[], actions=[])
    definition = Nminus1Definition(
        monitored_elements=[],
        contingencies=[Contingency(id="case", elements=[])],
        spps_rules=[rule],
        id_type="unique_pandapower",
    )
    captured: dict[str, Nminus1Definition] = {}

    def fake_run_contingency_analysis_pandapower(
        _net: pandapower.pandapowerNet,
        n_minus_1_definition: Nminus1Definition,
        _job_id: str,
        _timestep: int,
        cfg: object,
    ) -> LoadflowResultsPolars:
        del _net, _job_id, _timestep, cfg
        captured["definition"] = n_minus_1_definition
        return LoadflowResultsPolars(job_id="test")

    monkeypatch.setattr(ac_loadflow_service, "run_contingency_analysis_pandapower", fake_run_contingency_analysis_pandapower)

    ac_loadflow_service.get_ac_loadflow_results(net=pandapower_net, n_minus_1_definition=definition)

    assert captured["definition"] is definition
    assert captured["definition"].spps_rules == [rule]


def test_get_ac_loadflow_results_sanitizes_complex_definition_for_powsybl(
    powsybl_bus_breaker_net: Network,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = Nminus1Definition(
        monitored_elements=[],
        contingencies=[
            Contingency(
                id="grouped_case",
                name="Grouped case",
                elements=[
                    GridElement(id="line_a", type="LINE", kind="branch"),
                    GridElement(id="line_b", type="LINE", kind="branch"),
                ],
            )
        ],
        spps_rules=[SppsRule(scheme_name="grouped_case", conditions=[], actions=[])],
        id_type="powsybl",
    )
    captured: dict[str, Nminus1Definition] = {}

    def fake_run_contingency_analysis_powsybl(
        _net: Network,
        n_minus_1_definition: Nminus1Definition,
        _job_id: str,
        _timestep: int,
        n_processes: int,
        method: str,
        polars: bool,
        lf_params: object,
    ) -> LoadflowResultsPolars:
        del _net, _job_id, _timestep, n_processes, method, polars, lf_params
        captured["definition"] = n_minus_1_definition
        return LoadflowResultsPolars(job_id="test")

    monkeypatch.setattr(ac_loadflow_service, "run_contingency_analysis_powsybl", fake_run_contingency_analysis_powsybl)

    ac_loadflow_service.get_ac_loadflow_results(
        net=powsybl_bus_breaker_net,
        n_minus_1_definition=definition,
        sanitize_spps_rules=True,
    )

    sanitized_definition = captured["definition"]
    assert sanitized_definition.spps_rules is None
    assert sanitized_definition.id_type == definition.id_type
    assert sanitized_definition.contingencies == definition.contingencies
    assert sanitized_definition.contingencies is not definition.contingencies
    assert definition.spps_rules is not None
