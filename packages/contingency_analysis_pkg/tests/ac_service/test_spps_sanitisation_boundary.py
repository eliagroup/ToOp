# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""The SPPS sanitisation boundary in front of both CA backends.

Imported complex contingency lists carry SPPS rules that neither CA backend may execute yet.
``get_ac_loadflow_results`` is the single point where both backends are dispatched, so the rules are
stripped there rather than at each caller. These tests pin that boundary: an earlier attempt passed a
``sanitize_spps_rules`` flag to a function that never accepted one, so the boundary silently did not
exist.
"""

import pandapower as pp
import pypowsybl
import pytest
from toop_engine_contingency_analysis.ac_loadflow_service import ac_loadflow_service
from toop_engine_contingency_analysis.ac_loadflow_service.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_interfaces.nminus1_definition import (
    Action,
    Condition,
    Contingency,
    GridElement,
    Nminus1Definition,
    SppsRule,
)
from toop_engine_interfaces.spps_parameters import (
    SppsConditionCheckType,
    SppsConditionLogic,
    SppsConditionType,
    SppsMeasureType,
)

BACKEND_RUNNERS = {
    "powsybl": "run_contingency_analysis_powsybl",
    "pandapower": "run_contingency_analysis_pandapower",
}


class _DispatchedError(Exception):
    """Raised by the stubbed backend runner once it has captured its input."""


def _definition_with_spps_rules(source_schema: str | None) -> Nminus1Definition:
    """Build a definition whose single contingency carries one SPPS rule."""
    return Nminus1Definition(
        monitored_elements=[],
        contingencies=[
            Contingency(id="BASECASE", name="base_case", elements=[]),
            Contingency(id="c1", elements=[GridElement(id="branch1", type="LINE", kind="branch")]),
        ],
        spps_rules=[
            SppsRule(
                scheme_name="c1",
                condition_logic=SppsConditionLogic.ALL,
                conditions=[
                    Condition(
                        condition_type=SppsConditionType.STATE,
                        condition_check_type=SppsConditionCheckType.DE_ENERGIZED,
                        condition_element_unique_id="branch1",
                    )
                ],
                actions=[
                    Action(
                        measure_element_unique_id="switch1",
                        measure_type=SppsMeasureType.SWITCHING_STATE,
                        measure_value="closed",
                    )
                ],
            )
        ],
        id_type="powsybl",
        source_schema=source_schema,
    )


def _net_for(backend: str) -> pp.pandapowerNet | pypowsybl.network.Network:
    """Return the smallest network that routes dispatch to the requested backend."""
    return pp.create_empty_network() if backend == "pandapower" else pypowsybl.network.create_empty()


def _capture_definition_passed_to_backend(
    monkeypatch: pytest.MonkeyPatch, backend: str, definition: Nminus1Definition
) -> Nminus1Definition:
    """Dispatch to one backend and return the definition that backend actually received."""
    received: list[Nminus1Definition] = []

    def _stub(_net: object, n_minus_1_definition: Nminus1Definition, *_args: object, **_kwargs: object) -> None:
        received.append(n_minus_1_definition)
        raise _DispatchedError

    monkeypatch.setattr(ac_loadflow_service, BACKEND_RUNNERS[backend], _stub)
    with pytest.raises(_DispatchedError):
        get_ac_loadflow_results(net=_net_for(backend), n_minus_1_definition=definition)

    assert len(received) == 1
    return received[0]


@pytest.mark.parametrize("backend", sorted(BACKEND_RUNNERS))
def test_complex_definition_reaches_backend_without_spps_rules(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    """A complex-schema definition must lose its SPPS rules before either backend sees them."""
    definition = _definition_with_spps_rules(source_schema="complex")

    received = _capture_definition_passed_to_backend(monkeypatch, backend, definition)

    assert received.spps_rules is None
    # Everything else must survive the sanitisation.
    assert received.contingencies == definition.contingencies
    assert received.monitored_elements == definition.monitored_elements
    assert received.id_type == definition.id_type
    assert received.source_schema == definition.source_schema


@pytest.mark.parametrize("backend", sorted(BACKEND_RUNNERS))
def test_caller_definition_keeps_its_spps_rules(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    """Sanitisation is an in-memory copy, so the caller's definition is left untouched."""
    definition = _definition_with_spps_rules(source_schema="complex")

    _capture_definition_passed_to_backend(monkeypatch, backend, definition)

    assert definition.spps_rules is not None
    assert [rule.scheme_name for rule in definition.spps_rules] == ["c1"]


@pytest.mark.parametrize("backend", sorted(BACKEND_RUNNERS))
def test_non_complex_definition_keeps_its_spps_rules(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    """Sanitisation is scoped to complex input, so other import paths keep their SPPS behaviour."""
    definition = _definition_with_spps_rules(source_schema=None)

    received = _capture_definition_passed_to_backend(monkeypatch, backend, definition)

    assert received.spps_rules == definition.spps_rules
