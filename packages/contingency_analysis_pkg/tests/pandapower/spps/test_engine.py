# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

from copy import deepcopy
from unittest import mock

import pandapower as pp
import pandas as pd
import pytest
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.schemas import (
    SppsActionsPandapowerSchema,
    SppsConditionsPandapowerSchema,
)
from toop_engine_contingency_analysis.pandapower.spps.engine import (
    _apply_actions,
    _apply_switch_actions,
    _evaluate_conditions,
    _extract_condition_values,
    _populate_energized,
    _populate_failed,
    _restore_res_tables,
    _run_power_flow,
    _snapshot_res_tables,
    run_spps,
)
from toop_engine_contingency_analysis.pandapower.spps.errors import SppsPowerFlowError
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import SEPARATOR
from toop_engine_interfaces.spps_parameters import (
    SppsConditionCheckType,
    SppsConditionLogic,
    SppsConditionSide,
    SppsConditionType,
    SppsMeasureType,
    SppsPowerFlowFailurePolicy,
    SppsSwitchActionTarget,
)


def _cond_row(
    *,
    scheme: str = "s1",
    logic: str = SppsConditionLogic.ALL.value,
    ctype: str = SppsConditionType.CURRENT.value,
    check: str = SppsConditionCheckType.GT.value,
    side: str = SppsConditionSide.PRIMARY.value,
    limit: float | None = 0.0,
    table: str = "line",
    table_id: int = 0,
) -> dict:
    return {
        "scheme_name": scheme,
        "condition_logic": logic,
        "condition_type": ctype,
        "condition_check_type": check,
        "condition_side": side,
        "condition_limit_value": limit,
        "condition_element_table": table,
        "condition_element_table_id": table_id,
    }


def _act_row(
    *,
    scheme: str = "s1",
    mtype: str = SppsMeasureType.SWITCHING_STATE.value,
    mvalue: str | float = SppsSwitchActionTarget.OPEN.value,
    table: str = "switch",
    table_id: int = 0,
) -> dict:
    return {
        "scheme_name": scheme,
        "measure_type": mtype,
        "measure_value": mvalue,
        "measure_element_table": table,
        "measure_element_table_id": table_id,
    }


def _validate_conditions(df: pd.DataFrame) -> pd.DataFrame:
    return SppsConditionsPandapowerSchema.validate(df)


def _validate_actions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "measure_value" in df.columns:
        # Schema expects object dtype; literals otherwise become float64.
        df["measure_value"] = df["measure_value"].astype(object)
    return SppsActionsPandapowerSchema.validate(df)


def test_populate_energized_line_reads_in_service(pandapower_net: pp.pandapowerNet) -> None:
    net = deepcopy(pandapower_net)
    lid = int(net.line.index[0])
    cond = _validate_conditions(pd.DataFrame([_cond_row(table="line", table_id=lid)]))
    _populate_energized(cond, net)
    assert cond.loc[0, "energized"] == bool(net.line.loc[lid, "in_service"])


def test_populate_energized_switch_reads_closed(pandapower_net: pp.pandapowerNet) -> None:
    net = deepcopy(pandapower_net)
    sid = int(net.switch.index[0])
    cond = _validate_conditions(
        pd.DataFrame([_cond_row(table="switch", table_id=sid, check=SppsConditionCheckType.DE_ENERGIZED.value, limit=None)])
    )
    _populate_energized(cond, net)
    assert cond.loc[0, "energized"] == bool(net.switch.loc[sid, "closed"])


def test_evaluate_de_energized_passes_when_switch_open(pandapower_net: pp.pandapowerNet) -> None:
    net = deepcopy(pandapower_net)
    sid = int(net.switch.index[0])
    net.switch.loc[sid, "closed"] = False
    cond = _validate_conditions(
        pd.DataFrame(
            [
                _cond_row(
                    table="switch",
                    table_id=sid,
                    ctype=SppsConditionType.CURRENT.value,
                    check=SppsConditionCheckType.DE_ENERGIZED.value,
                    limit=None,
                )
            ]
        )
    )
    _populate_energized(cond, net)
    cond["failed"] = False
    cond["condition_element_value"] = 0.0
    _evaluate_conditions(cond)
    assert cond.loc[0, "is_condition"]


def test_evaluate_gt_lt_eq(pandapower_net: pp.pandapowerNet) -> None:
    lid = int(pandapower_net.line.index[0])
    cond = _validate_conditions(
        pd.DataFrame(
            [
                _cond_row(scheme="a", check=SppsConditionCheckType.GT.value, limit=1.0, table_id=lid),
                _cond_row(scheme="b", check=SppsConditionCheckType.LT.value, limit=10.0, table_id=lid),
                _cond_row(scheme="c", check=SppsConditionCheckType.EQ.value, limit=3.0, table_id=lid),
            ]
        )
    )
    cond["failed"] = False
    cond["energized"] = True
    cond.loc[0, "condition_element_value"] = 2.0
    cond.loc[1, "condition_element_value"] = 2.0
    cond.loc[2, "condition_element_value"] = 3.0
    _evaluate_conditions(cond)
    assert cond.loc[0, "is_condition"]
    assert cond.loc[1, "is_condition"]
    assert cond.loc[2, "is_condition"]


def test_evaluate_failed_uses_failed_column(pandapower_net: pp.pandapowerNet) -> None:
    lid = int(pandapower_net.line.index[0])
    cond = _validate_conditions(
        pd.DataFrame(
            [
                _cond_row(
                    check=SppsConditionCheckType.FAILED.value,
                    limit=None,
                    table_id=lid,
                )
            ]
        )
    )
    cond["failed"] = True
    cond["energized"] = True
    cond["condition_element_value"] = 0.0
    _evaluate_conditions(cond)
    assert cond.loc[0, "is_condition"]


def test_populate_failed_explicit_uid(pandapower_net: pp.pandapowerNet) -> None:
    net = deepcopy(pandapower_net)
    lid = int(net.line.index[0])
    uid = f"{lid}{SEPARATOR}line"
    cond = _validate_conditions(
        pd.DataFrame(
            [
                _cond_row(
                    check=SppsConditionCheckType.FAILED.value,
                    limit=None,
                    table_id=lid,
                )
            ]
        )
    )
    _populate_energized(cond, net)
    _populate_failed(cond, {uid}, net)
    assert cond.loc[0, "failed"]


def test_extract_condition_values_line_current(pandapower_net: pp.pandapowerNet) -> None:
    net = deepcopy(pandapower_net)
    pp.runpp(net, lightsim2grid=False)
    lid = int(net.line.index[0])
    cond = _validate_conditions(
        pd.DataFrame(
            [
                _cond_row(
                    ctype=SppsConditionType.CURRENT.value,
                    check=SppsConditionCheckType.GT.value,
                    side=SppsConditionSide.PRIMARY.value,
                    limit=0.0,
                    table_id=lid,
                )
            ]
        )
    )
    _extract_condition_values(cond, net)
    assert pd.notna(cond.loc[0, "condition_element_value"])
    assert cond.loc[0, "condition_element_value"] >= 0.0


def test_apply_switch_actions_open_and_closed(pandapower_net: pp.pandapowerNet) -> None:
    net = deepcopy(pandapower_net)
    s0 = int(net.switch.index[0])
    s1 = int(net.switch.index[1]) if len(net.switch.index) > 1 else None
    net.switch.loc[s0, "closed"] = True
    if s1 is not None:
        net.switch.loc[s1, "closed"] = True
    rows = [
        {
            "measure_element_table": "switch",
            "measure_element_table_id": s0,
            "measure_value": SppsSwitchActionTarget.OPEN.value,
        },
    ]
    if s1 is not None:
        rows.append(
            {
                "measure_element_table": "switch",
                "measure_element_table_id": s1,
                "measure_value": SppsSwitchActionTarget.CLOSED.value,
            }
        )
    _apply_switch_actions(pd.DataFrame(rows), net)
    assert not net.switch.loc[s0, "closed"]
    if s1 is not None:
        assert net.switch.loc[s1, "closed"]


def test_apply_actions_gen_active_power(pandapower_net: pp.pandapowerNet) -> None:
    net = deepcopy(pandapower_net)
    if net.gen.empty:
        pytest.skip("network has no generators")
    gid = int(net.gen.index[0])
    old_p = float(net.gen.loc[gid, "p_mw"])
    actions = _validate_actions(
        pd.DataFrame(
            [
                {
                    "scheme_name": "g",
                    "measure_type": SppsMeasureType.ACTIVE_POWER.value,
                    "measure_value": old_p + 1.0,
                    "measure_element_table": "gen",
                    "measure_element_table_id": gid,
                }
            ]
        )
    )
    _apply_actions(actions, net)
    assert net.gen.loc[gid, "p_mw"] == pytest.approx(old_p + 1.0)


def test_snapshot_and_restore_res_tables(pandapower_net: pp.pandapowerNet) -> None:
    net = deepcopy(pandapower_net)
    pp.runpp(net, lightsim2grid=False)
    snap = _snapshot_res_tables(net)
    assert "res_bus" in snap
    orig_vm = net.res_bus["vm_pu"].copy()
    net.res_bus["vm_pu"] = -999.0
    _restore_res_tables(net, snap)
    pd.testing.assert_series_equal(net.res_bus["vm_pu"], orig_vm, check_names=True)


def test_run_power_flow_dc(pandapower_net: pp.pandapowerNet) -> None:
    net = deepcopy(pandapower_net)
    _run_power_flow(net, "dc", {})
    assert net.converged


def test_run_spps_activates_scheme_applies_switch_dc(pandapower_net: pp.pandapowerNet) -> None:
    """Use max_iterations>1 so the engine does not treat hitting max_iters as an error after one activation."""
    net = deepcopy(pandapower_net)
    lid = int(net.line.index[0])
    sid = int(net.switch.index[0])
    was_closed = bool(net.switch.loc[sid, "closed"])

    conditions = _validate_conditions(
        pd.DataFrame(
            [
                _cond_row(
                    scheme="trip",
                    table="line",
                    table_id=lid,
                    ctype=SppsConditionType.CURRENT.value,
                    check=SppsConditionCheckType.GT.value,
                    side=SppsConditionSide.PRIMARY.value,
                    limit=-1.0,
                )
            ]
        )
    )
    actions = _validate_actions(
        pd.DataFrame(
            [
                _act_row(
                    scheme="trip",
                    table_id=sid,
                    mvalue=SppsSwitchActionTarget.OPEN.value,
                )
            ]
        )
    )
    failed: set[str] = set()
    result = run_spps(net, conditions, actions, failed, method="dc", max_iterations=5, runpp_kwargs={})
    assert result.iterations >= 1
    assert result.activated_schemes_per_iter
    assert any("trip" in batch for batch in result.activated_schemes_per_iter)
    assert not result.power_flow_failed
    assert not result.max_iterations_reached
    if was_closed:
        assert not net.switch.loc[sid, "closed"]
        assert f"{sid}{SEPARATOR}switch" in failed


def test_run_spps_no_activation_when_condition_false(pandapower_net: pp.pandapowerNet) -> None:
    net = deepcopy(pandapower_net)
    lid = int(net.line.index[0])
    conditions = _validate_conditions(
        pd.DataFrame(
            [
                _cond_row(
                    scheme="never",
                    table="line",
                    table_id=lid,
                    ctype=SppsConditionType.CURRENT.value,
                    check=SppsConditionCheckType.GT.value,
                    side=SppsConditionSide.PRIMARY.value,
                    limit=1.0e30,
                )
            ]
        )
    )
    actions = _validate_actions(pd.DataFrame([_act_row(scheme="never", table_id=int(net.switch.index[0]))]))
    result = run_spps(net, conditions, actions, set(), method="dc", max_iterations=3, runpp_kwargs={})
    assert result.iterations == 0
    assert result.activated_schemes_per_iter == []


def test_run_spps_raises_on_initial_pf_failure() -> None:
    net = pp.create_empty_network()
    conditions = _validate_conditions(pd.DataFrame([_cond_row(table_id=0)]))
    actions = _validate_actions(pd.DataFrame([_act_row()]))
    with pytest.raises(Exception):
        run_spps(net, conditions, actions, set(), method="dc", max_iterations=2, runpp_kwargs={})


def test_run_spps_max_iterations_exhausted_raises(pandapower_net: pp.pandapowerNet) -> None:
    """Two schemes always eligible; max_iterations=1 completes one iter then raises (engine contract)."""
    net = deepcopy(pandapower_net)
    lid = int(net.line.index[0])
    s0, s1 = int(net.switch.index[0]), int(net.switch.index[1]) if len(net.switch.index) > 1 else None
    if s1 is None:
        pytest.skip("need two switches")
    cond_rows = [
        _cond_row(scheme="A", table_id=lid, limit=-1.0),
        _cond_row(scheme="B", table_id=lid, limit=-1.0),
    ]
    conditions = _validate_conditions(pd.DataFrame(cond_rows))
    actions = _validate_actions(
        pd.DataFrame(
            [
                _act_row(scheme="A", table_id=s0, mvalue=SppsSwitchActionTarget.OPEN.value),
                _act_row(scheme="B", table_id=s1, mvalue=SppsSwitchActionTarget.OPEN.value),
            ]
        )
    )
    with pytest.raises(SppsPowerFlowError, match="max_iterations"):
        run_spps(
            net,
            conditions,
            actions,
            set(),
            method="dc",
            max_iterations=1,
            runpp_kwargs={},
            on_power_flow_error=SppsPowerFlowFailurePolicy.RAISE,
        )


def test_run_spps_keep_previous_restores_res_after_pf_failure(pandapower_net: pp.pandapowerNet) -> None:
    """Second in-loop PF fails; ``res_*`` are restored from the snapshot taken before that call."""
    net = deepcopy(pandapower_net)
    pp.runpp(net, lightsim2grid=False)
    vm_after_initial = net.res_bus["vm_pu"].copy()

    lid = int(net.line.index[0])
    sid = int(net.switch.index[0])
    conditions = _validate_conditions(pd.DataFrame([_cond_row(scheme="x", table_id=lid, limit=-1.0)]))
    actions = _validate_actions(pd.DataFrame([_act_row(scheme="x", table_id=sid)]))

    calls = {"n": 0}

    def _run_pw(n: pp.pandapowerNet, method: str, runpp_kwargs: dict) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            pp.runpp(n, **runpp_kwargs)
        elif calls["n"] == 2:
            raise RuntimeError("simulated pf failure")
        else:
            pp.runpp(n, **runpp_kwargs)

    with mock.patch("toop_engine_contingency_analysis.pandapower.spps.engine._run_power_flow", side_effect=_run_pw):
        result = run_spps(
            net,
            conditions,
            actions,
            set(),
            method="dc",
            max_iterations=2,
            runpp_kwargs={"lightsim2grid": False},
            on_power_flow_error=SppsPowerFlowFailurePolicy.KEEP_PREVIOUS,
        )

    assert result.power_flow_failed is True
    pd.testing.assert_series_equal(net.res_bus["vm_pu"], vm_after_initial, check_names=True, rtol=0, atol=0)
