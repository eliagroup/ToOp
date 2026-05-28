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
    SppsConditionMode,
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
    condition_mode: str = SppsConditionMode.CON.value,
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
        "condition_mode": condition_mode,
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
    basecase_net = deepcopy(pandapower_net)
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
    result = run_spps(net, conditions, actions, failed, basecase_net, method="dc", max_iterations=5, runpp_kwargs={})
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
    basecase_net = deepcopy(pandapower_net)
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
    result = run_spps(net, conditions, actions, set(), basecase_net, method="dc", max_iterations=3, runpp_kwargs={})
    assert result.iterations == 0
    assert result.activated_schemes_per_iter == []


def test_run_spps_raises_on_initial_pf_failure() -> None:
    net = pp.create_empty_network()
    basecase_net = pp.create_empty_network()
    conditions = _validate_conditions(pd.DataFrame([_cond_row(table_id=0)]))
    actions = _validate_actions(pd.DataFrame([_act_row()]))
    with pytest.raises(Exception):
        run_spps(net, conditions, actions, set(), basecase_net, method="dc", max_iterations=2, runpp_kwargs={})


def test_run_spps_max_iterations_exhausted_raises(pandapower_net: pp.pandapowerNet) -> None:
    """Two schemes always eligible; max_iterations=1 completes one iter then raises (engine contract)."""
    net = deepcopy(pandapower_net)
    basecase_net = deepcopy(pandapower_net)
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
            basecase_net,
            method="dc",
            max_iterations=1,
            runpp_kwargs={},
            on_power_flow_error=SppsPowerFlowFailurePolicy.RAISE,
        )


def test_run_spps_keep_previous_restores_res_after_pf_failure(pandapower_net: pp.pandapowerNet) -> None:
    """Second in-loop PF fails; ``res_*`` are restored from the snapshot taken before that call."""
    net = deepcopy(pandapower_net)
    pp.runpp(net, lightsim2grid=False)
    basecase_net = deepcopy(net)
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
            basecase_net,
            method="dc",
            max_iterations=2,
            runpp_kwargs={"lightsim2grid": False},
            on_power_flow_error=SppsPowerFlowFailurePolicy.KEEP_PREVIOUS,
        )

    assert result.power_flow_failed is True
    pd.testing.assert_series_equal(net.res_bus["vm_pu"], vm_after_initial, check_names=True, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# BC-mode condition tests
# ---------------------------------------------------------------------------


def _make_simple_loaded_net() -> pp.pandapowerNet:
    """3-bus series network used by BC-mode tests.

    Topology:  ext_grid (b0) --line0-- (b1) --line1-- (b2) load
    Switches:  sw0 on line0 at b0 side, sw1 on line1 at b1 side.

    With 50 MW / 10 Mvar load and AC power flow the lines carry significant
    current.  Reducing the load to near-zero makes the contingency-state
    current nearly zero, giving a clean BC vs CON contrast.
    """
    net = pp.create_empty_network(sn_mva=100)
    b0 = pp.create_bus(net, vn_kv=110)
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0)
    pp.create_load(net, bus=b2, p_mw=50, q_mvar=10)
    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=1,
        r_ohm_per_km=0.05,
        x_ohm_per_km=0.1,
        c_nf_per_km=0,
        max_i_ka=1,
    )
    pp.create_line_from_parameters(
        net,
        from_bus=b1,
        to_bus=b2,
        length_km=1,
        r_ohm_per_km=0.05,
        x_ohm_per_km=0.1,
        c_nf_per_km=0,
        max_i_ka=1,
    )
    pp.create_switch(net, bus=b0, element=0, et="l", closed=True)
    pp.create_switch(net, bus=b1, element=1, et="l", closed=True)
    return net


def test_bc_mode_activates_scheme_using_basecase_current() -> None:
    """BC condition reads base-case current; scheme fires even when post-contingency current is below limit.

    Setup
    -----
    * ``basecase_net``: AC PF with 50 MW load → significant current on line 0.
    * Contingency ``net``: load reduced to 0.1 MW → near-zero current on line 0.
    * Condition threshold: halfway between 0 and the base-case current.

    Expected
    --------
    BC condition evaluates to True (base-case current > threshold) → scheme activates.
    """
    net = _make_simple_loaded_net()
    pp.runpp(net, lightsim2grid=False)
    basecase_net = deepcopy(net)

    # Drastically reduce load so the contingency PF gives near-zero current.
    net.load.at[net.load.index[0], "p_mw"] = 0.1
    net.load.at[net.load.index[0], "q_mvar"] = 0.0

    lid = int(net.line.index[0])
    bc_current = float(basecase_net.res_line.at[lid, "i_from_ka"])
    # Threshold sits between near-zero (CON) and bc_current → True for BC, False for CON.
    threshold = bc_current * 0.5

    conditions = _validate_conditions(
        pd.DataFrame(
            [
                _cond_row(
                    scheme="bc_trip",
                    table="line",
                    table_id=lid,
                    ctype=SppsConditionType.CURRENT.value,
                    check=SppsConditionCheckType.GT.value,
                    side=SppsConditionSide.PRIMARY.value,
                    limit=threshold,
                    condition_mode=SppsConditionMode.BC.value,
                )
            ]
        )
    )
    sid = int(net.switch.index[0])
    actions = _validate_actions(
        pd.DataFrame([_act_row(scheme="bc_trip", table_id=sid, mvalue=SppsSwitchActionTarget.OPEN.value)])
    )

    result = run_spps(
        net,
        conditions,
        actions,
        set(),
        basecase_net,
        method="ac",
        max_iterations=3,
        runpp_kwargs={"lightsim2grid": False},
    )

    assert result.iterations >= 1
    activated = [s for batch in result.activated_schemes_per_iter for s in batch]
    assert "bc_trip" in activated


def test_con_mode_does_not_activate_when_only_basecase_exceeds_limit() -> None:
    """Mirror of the BC test: CON condition on the same setup must NOT activate.

    The contingency current (near-zero load) is far below the threshold that
    the base-case current exceeds.  With ``condition_mode="CON"`` the engine
    must evaluate against the post-contingency result → condition is False →
    scheme is never triggered.
    """
    net = _make_simple_loaded_net()
    pp.runpp(net, lightsim2grid=False)
    basecase_net = deepcopy(net)

    net.load.at[net.load.index[0], "p_mw"] = 0.1
    net.load.at[net.load.index[0], "q_mvar"] = 0.0

    lid = int(net.line.index[0])
    bc_current = float(basecase_net.res_line.at[lid, "i_from_ka"])
    threshold = bc_current * 0.5  # same threshold as the BC test above

    conditions = _validate_conditions(
        pd.DataFrame(
            [
                _cond_row(
                    scheme="con_trip",
                    table="line",
                    table_id=lid,
                    ctype=SppsConditionType.CURRENT.value,
                    check=SppsConditionCheckType.GT.value,
                    side=SppsConditionSide.PRIMARY.value,
                    limit=threshold,
                    condition_mode=SppsConditionMode.CON.value,  # CON → reads post-contingency PF
                )
            ]
        )
    )
    sid = int(net.switch.index[0])
    actions = _validate_actions(
        pd.DataFrame([_act_row(scheme="con_trip", table_id=sid, mvalue=SppsSwitchActionTarget.OPEN.value)])
    )

    result = run_spps(
        net,
        conditions,
        actions,
        set(),
        basecase_net,
        method="ac",
        max_iterations=3,
        runpp_kwargs={"lightsim2grid": False},
    )

    assert result.iterations == 0
    assert result.activated_schemes_per_iter == []


def test_bc_values_frozen_across_iterations() -> None:
    """BC condition values must not be refreshed after in-loop power flows.

    Network
    -------
    Three buses, three lines forming two paths from the slack (b0) to the load (b2):

    * path A: b0 --line0-- b1 --line1-- b2  (series, 2×Z)
    * path B: b0 --line2-- b2              (direct,  Z)

    With identical line impedances the parallel split gives roughly 1/3 of total
    current on line 0.  After scheme A opens the line2 switch, line 0 carries
    the full load current (≈ 3× its base-case value).

    Test logic
    ----------
    * Scheme A: CON condition on line 2 current > −1 (always True) → opens
      line2's switch → in-loop PF shifts all current to line 0.
    * Scheme B: BC condition on line 0 current > ``threshold``, where
      ``threshold`` lies between the base-case value (1/3 × total) and the
      post-scheme-A value (≈ total).

    If BC values are properly frozen, scheme B's condition stays False (BC
    current < threshold).  If they were incorrectly refreshed from the new
    PF, scheme B would wrongly activate.
    """
    net = pp.create_empty_network(sn_mva=100)
    b0 = pp.create_bus(net, vn_kv=110)
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0)
    pp.create_load(net, bus=b2, p_mw=60, q_mvar=0)

    lid0 = pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=1,
        r_ohm_per_km=0.05,
        x_ohm_per_km=0.1,
        c_nf_per_km=0,
        max_i_ka=1,
    )
    pp.create_line_from_parameters(
        net,
        from_bus=b1,
        to_bus=b2,
        length_km=1,
        r_ohm_per_km=0.05,
        x_ohm_per_km=0.1,
        c_nf_per_km=0,
        max_i_ka=1,
    )
    lid2 = pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b2,
        length_km=1,
        r_ohm_per_km=0.05,
        x_ohm_per_km=0.1,
        c_nf_per_km=0,
        max_i_ka=1,
    )
    # sw_a: switch for scheme A (opens line 2's direct path)
    sw_a = pp.create_switch(net, bus=b0, element=lid2, et="l", closed=True)
    # sw_b: switch for scheme B action (if B wrongly fires)
    sw_b = pp.create_switch(net, bus=b0, element=lid0, et="l", closed=True)

    pp.runpp(net, lightsim2grid=False)
    basecase_net = deepcopy(net)

    bc_current_line0 = float(basecase_net.res_line.at[lid0, "i_from_ka"])
    # With identical parallel paths, line0 carries ~1/3 of total current in BC.
    # After scheme A disconnects the direct path, line0 carries ~all current (≥ 2× BC).
    # Place threshold between BC and the expected post-A value.
    threshold = bc_current_line0 * 1.6

    # Scheme A: CON condition always True → fires in iteration 1, opens direct path.
    cond_a = _cond_row(
        scheme="A",
        table="line",
        table_id=lid2,
        ctype=SppsConditionType.CURRENT.value,
        check=SppsConditionCheckType.GT.value,
        side=SppsConditionSide.PRIMARY.value,
        limit=-1.0,
        condition_mode=SppsConditionMode.CON.value,
    )
    act_a = _act_row(scheme="A", table_id=sw_a, mvalue=SppsSwitchActionTarget.OPEN.value)

    # Scheme B: BC condition on line 0 current > threshold.
    # BC value (1/3 × total) < threshold → must NOT fire.
    # If BC values were refreshed from the post-A PF (all current on line 0), B would wrongly fire.
    cond_b = _cond_row(
        scheme="B",
        table="line",
        table_id=lid0,
        ctype=SppsConditionType.CURRENT.value,
        check=SppsConditionCheckType.GT.value,
        side=SppsConditionSide.PRIMARY.value,
        limit=threshold,
        condition_mode=SppsConditionMode.BC.value,
    )
    act_b = _act_row(scheme="B", table_id=sw_b, mvalue=SppsSwitchActionTarget.OPEN.value)

    conditions = _validate_conditions(pd.DataFrame([cond_a, cond_b]))
    actions = _validate_actions(pd.DataFrame([act_a, act_b]))

    result = run_spps(
        net,
        conditions,
        actions,
        set(),
        basecase_net,
        method="ac",
        max_iterations=3,
        runpp_kwargs={"lightsim2grid": False},
    )

    activated = [s for batch in result.activated_schemes_per_iter for s in batch]
    assert "A" in activated, "Scheme A (always-true CON) should have fired"
    assert "B" not in activated, "Scheme B (BC frozen below threshold) must NOT fire"


def test_mixed_bc_con_conditions_all_logic_scheme_activates() -> None:
    """A scheme with ALL logic activates only when every condition row is True.

    Two condition rows in the same scheme:

    * Row 1 (BC):  line 0 current > low_threshold  → True  (base-case has meaningful current)
    * Row 2 (CON): line 1 current > −1             → True  (always True post-contingency)

    Both rows must be True for the scheme to activate; this verifies that BC
    and CON rows interact correctly within one scheme.

    A second call with Row 1 having a limit above the base-case current (BC
    condition False) confirms that the scheme stays inactive when any row fails.
    """
    net = _make_simple_loaded_net()
    pp.runpp(net, lightsim2grid=False)
    basecase_net = deepcopy(net)

    lid0 = int(net.line.index[0])
    lid1 = int(net.line.index[1])
    sid = int(net.switch.index[0])

    bc_current = float(basecase_net.res_line.at[lid0, "i_from_ka"])

    # --- sub-case 1: both conditions True → scheme activates ---
    cond1 = _cond_row(
        scheme="S",
        table="line",
        table_id=lid0,
        ctype=SppsConditionType.CURRENT.value,
        check=SppsConditionCheckType.GT.value,
        side=SppsConditionSide.PRIMARY.value,
        limit=bc_current * 0.5,  # BC current (large) > limit → True
        condition_mode=SppsConditionMode.BC.value,
    )
    cond2 = _cond_row(
        scheme="S",
        table="line",
        table_id=lid1,
        ctype=SppsConditionType.CURRENT.value,
        check=SppsConditionCheckType.GT.value,
        side=SppsConditionSide.PRIMARY.value,
        limit=-1.0,  # always True
        condition_mode=SppsConditionMode.CON.value,
    )
    conditions_ok = _validate_conditions(pd.DataFrame([cond1, cond2]))
    actions_ok = _validate_actions(
        pd.DataFrame([_act_row(scheme="S", table_id=sid, mvalue=SppsSwitchActionTarget.OPEN.value)])
    )

    result_ok = run_spps(
        net,
        conditions_ok,
        actions_ok,
        set(),
        basecase_net,
        method="ac",
        max_iterations=2,
        runpp_kwargs={"lightsim2grid": False},
    )
    assert result_ok.iterations >= 1, "Scheme S should activate when both BC and CON conditions are True"
    activated_ok = [s for batch in result_ok.activated_schemes_per_iter for s in batch]
    assert "S" in activated_ok

    # --- sub-case 2: BC condition False (limit above BC current) → scheme stays inactive ---
    net2 = _make_simple_loaded_net()
    pp.runpp(net2, lightsim2grid=False)
    basecase_net2 = deepcopy(net2)

    cond1_false = _cond_row(
        scheme="S2",
        table="line",
        table_id=lid0,
        ctype=SppsConditionType.CURRENT.value,
        check=SppsConditionCheckType.GT.value,
        side=SppsConditionSide.PRIMARY.value,
        limit=bc_current * 2.0,  # BC current < limit → False
        condition_mode=SppsConditionMode.BC.value,
    )
    cond2_true = _cond_row(
        scheme="S2",
        table="line",
        table_id=lid1,
        ctype=SppsConditionType.CURRENT.value,
        check=SppsConditionCheckType.GT.value,
        side=SppsConditionSide.PRIMARY.value,
        limit=-1.0,  # still always True
        condition_mode=SppsConditionMode.CON.value,
    )
    conditions_fail = _validate_conditions(pd.DataFrame([cond1_false, cond2_true]))
    actions_fail = _validate_actions(
        pd.DataFrame([_act_row(scheme="S2", table_id=int(net2.switch.index[0]), mvalue=SppsSwitchActionTarget.OPEN.value)])
    )

    result_fail = run_spps(
        net2,
        conditions_fail,
        actions_fail,
        set(),
        basecase_net2,
        method="ac",
        max_iterations=2,
        runpp_kwargs={"lightsim2grid": False},
    )
    assert result_fail.iterations == 0, "Scheme S2 must not activate when the BC condition row is False"
    assert result_fail.activated_schemes_per_iter == []


def test_bc_mode_bus_voltage_uses_basecase_voltage() -> None:
    """BC condition on bus voltage reads base-case vm_pu, not post-contingency vm_pu.

    Setup
    -----
    * ``basecase_net``: load bus voltage is slightly below 1.0 p.u. (50 MW load).
    * Contingency ``net``: load reduced to 0.1 MW → voltage rises close to 1.0 p.u.
    * Threshold: midpoint between base-case voltage and 1.0.

    Scheme ``V_bc`` (BC mode, ``vm < threshold``):
        Base-case voltage is below the midpoint → condition True → scheme fires.

    Scheme ``V_con`` (CON mode, ``vm < threshold``):
        Post-contingency voltage is above the midpoint → condition False → no fire.
    """
    net = _make_simple_loaded_net()
    pp.runpp(net, lightsim2grid=False)
    basecase_net = deepcopy(net)

    # Load bus is the last bus (index 2 in the simple network).
    b2 = int(net.bus.index[2])
    bc_vm = float(basecase_net.res_bus.at[b2, "vm_pu"])

    # Reduce load so the contingency voltage rises above the threshold.
    net.load.at[net.load.index[0], "p_mw"] = 0.1
    net.load.at[net.load.index[0], "q_mvar"] = 0.0

    # Threshold sits between bc_vm (low, with load) and ~1.0 (no load).
    threshold = (bc_vm + 1.0) / 2.0

    sid0 = int(net.switch.index[0])
    sid1 = int(net.switch.index[1])

    cond_bc = _cond_row(
        scheme="V_bc",
        table="bus",
        table_id=b2,
        ctype=SppsConditionType.VOLTAGE.value,
        check=SppsConditionCheckType.LT.value,
        side=SppsConditionSide.PRIMARY.value,
        limit=threshold,
        condition_mode=SppsConditionMode.BC.value,  # evaluates bc_vm < threshold → True
    )
    cond_con = _cond_row(
        scheme="V_con",
        table="bus",
        table_id=b2,
        ctype=SppsConditionType.VOLTAGE.value,
        check=SppsConditionCheckType.LT.value,
        side=SppsConditionSide.PRIMARY.value,
        limit=threshold,
        condition_mode=SppsConditionMode.CON.value,  # evaluates con_vm < threshold → False
    )

    conditions = _validate_conditions(pd.DataFrame([cond_bc, cond_con]))
    actions = _validate_actions(
        pd.DataFrame(
            [
                _act_row(scheme="V_bc", table_id=sid0, mvalue=SppsSwitchActionTarget.OPEN.value),
                _act_row(scheme="V_con", table_id=sid1, mvalue=SppsSwitchActionTarget.OPEN.value),
            ]
        )
    )

    result = run_spps(
        net,
        conditions,
        actions,
        set(),
        basecase_net,
        method="ac",
        max_iterations=2,
        runpp_kwargs={"lightsim2grid": False},
    )

    activated = [s for batch in result.activated_schemes_per_iter for s in batch]
    assert "V_bc" in activated, "BC voltage condition (base-case vm below threshold) should activate scheme"
    assert "V_con" not in activated, "CON voltage condition (post-contingency vm above threshold) must not activate"
