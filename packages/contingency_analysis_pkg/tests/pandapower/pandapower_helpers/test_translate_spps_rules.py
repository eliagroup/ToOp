# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

import uuid

import pandapower as pp
import pytest
from toop_engine_contingency_analysis.pandapower.pandapower_helpers.translators import translate_spps_rules
from toop_engine_grid_helpers.pandapower.pandapower_id_helpers import get_globally_unique_id
from toop_engine_interfaces.nminus1_definition import Action, Condition, SppsRule
from toop_engine_interfaces.spps_parameters import (
    SppsConditionCheckType,
    SppsConditionLogic,
    SppsConditionSide,
    SppsConditionType,
    SppsMeasureType,
    SppsSwitchActionTarget,
)


def _sample_rule(line_idx_cond: int, switch_idx_meas: int, scheme: str = "scheme_a") -> SppsRule:
    cond_uid = get_globally_unique_id(line_idx_cond, "line")
    meas_uid = get_globally_unique_id(switch_idx_meas, "switch")
    return SppsRule(
        scheme_name=scheme,
        conditions=[
            Condition(
                condition_type=SppsConditionType.CURRENT,
                condition_check_type=SppsConditionCheckType.GT,
                condition_side=SppsConditionSide.PRIMARY,
                condition_limit_value=100.0,
                condition_element_unique_id=cond_uid,
            )
        ],
        actions=[
            Action(
                measure_element_unique_id=meas_uid,
                measure_type=SppsMeasureType.SWITCHING_STATE,
                measure_value=SppsSwitchActionTarget.CLOSED,
            )
        ],
    )


def test_translate_spps_rules_none_returns_empty_tables(pandapower_net: pp.pandapowerNet) -> None:
    cond, act, missing, dupes = translate_spps_rules(pandapower_net, None)
    assert cond.empty and act.empty
    assert missing == [] and dupes == []


def test_translate_spps_rules_empty_list_returns_empty_tables(pandapower_net: pp.pandapowerNet) -> None:
    cond, act, missing, dupes = translate_spps_rules(pandapower_net, [], id_type="unique_pandapower")
    assert cond.empty and act.empty
    assert missing == [] and dupes == []


def test_translate_spps_rules_unique_pandapower_default_id_type(pandapower_net: pp.pandapowerNet) -> None:
    first_line = int(pandapower_net.line.index[0])
    first_switch = int(pandapower_net.switch.index[0])
    rule = _sample_rule(first_line, first_switch)
    cond, act, missing, dupes = translate_spps_rules(pandapower_net, [rule])
    assert missing == [] and dupes == []
    assert len(cond) == 1 and len(act) == 1
    assert cond.iloc[0]["scheme_name"] == rule.scheme_name
    assert cond.iloc[0]["condition_element_table"] == "line"
    assert int(cond.iloc[0]["condition_element_table_id"]) == first_line
    assert act.iloc[0]["measure_element_table"] == "switch"
    assert int(act.iloc[0]["measure_element_table_id"]) == first_switch
    assert act.iloc[0]["measure_type"] == SppsMeasureType.SWITCHING_STATE
    assert act.iloc[0]["measure_value"] == SppsSwitchActionTarget.CLOSED
    assert (cond["condition_logic"] == SppsConditionLogic.ALL.value).all()


def test_translate_spps_rules_unique_pandapower_unknown_element_in_missing(
    pandapower_net: pp.pandapowerNet,
) -> None:
    bad_id = "999999%%line"
    rule = SppsRule(
        scheme_name="bad",
        conditions=[
            Condition(
                condition_type=SppsConditionType.VOLTAGE,
                condition_check_type=SppsConditionCheckType.LT,
                condition_side=SppsConditionSide.PRIMARY,
                condition_limit_value=0.9,
                condition_element_unique_id=bad_id,
            )
        ],
        actions=[
            Action(
                measure_element_unique_id=get_globally_unique_id(int(pandapower_net.switch.index[0]), "switch"),
                measure_type=SppsMeasureType.SWITCHING_STATE,
                measure_value=SppsSwitchActionTarget.OPEN,
            )
        ],
    )
    cond, act, missing, dupes = translate_spps_rules(pandapower_net, [rule], id_type="unique_pandapower")
    assert len(missing) == 1 and missing[0] is rule
    assert dupes == []
    assert cond.empty and act.empty


def test_translate_spps_rules_condition_logic_any_on_condition_rows(
    pandapower_net: pp.pandapowerNet,
) -> None:
    net = pandapower_net
    line_a = int(net.line.index[0])
    line_b = int(net.line.index[1]) if len(net.line.index) > 1 else line_a
    sw = int(net.switch.index[0])
    rule = SppsRule(
        scheme_name="any_scheme",
        condition_logic=SppsConditionLogic.ANY,
        conditions=[
            Condition(
                condition_type=SppsConditionType.CURRENT,
                condition_check_type=SppsConditionCheckType.GT,
                condition_side=SppsConditionSide.PRIMARY,
                condition_limit_value=1.0,
                condition_element_unique_id=get_globally_unique_id(line_a, "line"),
            ),
            Condition(
                condition_type=SppsConditionType.VOLTAGE,
                condition_check_type=SppsConditionCheckType.LT,
                condition_side=SppsConditionSide.PRIMARY,
                condition_limit_value=2.0,
                condition_element_unique_id=get_globally_unique_id(line_b, "line"),
            ),
        ],
        actions=[
            Action(
                measure_element_unique_id=get_globally_unique_id(sw, "switch"),
                measure_type=SppsMeasureType.SWITCHING_STATE,
                measure_value=SppsSwitchActionTarget.CLOSED,
            )
        ],
    )
    cond, act, missing, dupes = translate_spps_rules(net, [rule], id_type="unique_pandapower")
    assert missing == [] and dupes == []
    assert len(cond) == 2 and len(act) == 1
    assert (cond["condition_logic"] == SppsConditionLogic.ANY.value).all()

    rule = SppsRule(
        scheme_name="no_cond",
        conditions=[],
        actions=[
            Action(
                measure_element_unique_id=get_globally_unique_id(int(pandapower_net.switch.index[0]), "switch"),
                measure_type=SppsMeasureType.SWITCHING_STATE,
                measure_value=SppsSwitchActionTarget.CLOSED,
            )
        ],
    )
    cond, act, missing, _ = translate_spps_rules(pandapower_net, [rule], id_type="unique_pandapower")
    assert missing == [rule] and cond.empty and act.empty


def test_translate_spps_rules_unique_pandapower_no_actions_in_missing(
    pandapower_net: pp.pandapowerNet,
) -> None:
    rule = SppsRule(
        scheme_name="no_act",
        conditions=[
            Condition(
                condition_type=SppsConditionType.CURRENT,
                condition_check_type=SppsConditionCheckType.GT,
                condition_side=SppsConditionSide.PRIMARY,
                condition_limit_value=1.0,
                condition_element_unique_id=get_globally_unique_id(int(pandapower_net.line.index[0]), "line"),
            )
        ],
        actions=[],
    )
    cond, act, missing, _ = translate_spps_rules(pandapower_net, [rule], id_type="unique_pandapower")
    assert missing == [rule] and cond.empty and act.empty


def test_translate_spps_rules_unsupported_id_type_raises(pandapower_net: pp.pandapowerNet) -> None:
    rule = _sample_rule(int(pandapower_net.line.index[0]), int(pandapower_net.switch.index[0]))
    with pytest.raises(ValueError, match="Unsupported id_type"):
        translate_spps_rules(pandapower_net, [rule], id_type="ucte")  # type: ignore[arg-type]


def test_translate_spps_rules_cgmes_resolves_origin_id(pandapower_net: pp.pandapowerNet) -> None:
    net = pandapower_net
    line_idx = int(net.line.index[0])
    sw_idx = int(net.switch.index[0])
    line_guid = str(uuid.uuid4())
    switch_guid = str(uuid.uuid4())
    net.line.loc[line_idx, "origin_id"] = line_guid
    net.switch.loc[sw_idx, "origin_id"] = switch_guid

    rule = SppsRule(
        scheme_name="cgmes_scheme",
        conditions=[
            Condition(
                condition_type=SppsConditionType.ACTIVE_POWER,
                condition_check_type=SppsConditionCheckType.GT,
                condition_side=SppsConditionSide.PRIMARY,
                condition_limit_value=50.0,
                condition_element_unique_id=line_guid,
            )
        ],
        actions=[
            Action(
                measure_element_unique_id=switch_guid,
                measure_type=SppsMeasureType.SWITCHING_STATE,
                measure_value=SppsSwitchActionTarget.OPEN,
            )
        ],
    )
    cond, act, missing, dupes = translate_spps_rules(net, [rule], id_type="cgmes")
    assert missing == [] and dupes == []
    assert len(cond) == 1 and len(act) == 1
    assert cond.iloc[0]["condition_element_table"] == "line" and int(cond.iloc[0]["condition_element_table_id"]) == line_idx
    assert act.iloc[0]["measure_element_table"] == "switch" and int(act.iloc[0]["measure_element_table_id"]) == sw_idx


def test_translate_spps_rules_cgmes_unknown_guid_missing(pandapower_net: pp.pandapowerNet) -> None:
    net = pandapower_net
    sw_idx = int(net.switch.index[0])
    switch_guid = str(uuid.uuid4())
    net.switch.loc[sw_idx, "origin_id"] = switch_guid
    rule = SppsRule(
        scheme_name="missing_guid",
        conditions=[
            Condition(
                condition_type=SppsConditionType.VOLTAGE,
                condition_check_type=SppsConditionCheckType.LT,
                condition_side=SppsConditionSide.PRIMARY,
                condition_limit_value=1.0,
                condition_element_unique_id=str(uuid.uuid4()),
            )
        ],
        actions=[
            Action(
                measure_element_unique_id=switch_guid,
                measure_type=SppsMeasureType.SWITCHING_STATE,
                measure_value=SppsSwitchActionTarget.CLOSED,
            )
        ],
    )
    cond, act, missing, dupes = translate_spps_rules(net, [rule], id_type="cgmes")
    assert len(missing) == 1 and missing[0] is rule
    assert cond.empty and act.empty
    assert dupes == []


def test_translate_spps_rules_cgmes_duplicate_origin_id_reported(pandapower_net: pp.pandapowerNet) -> None:
    net = pandapower_net
    if len(net.switch.index) < 2:
        pytest.skip("need at least two switches for duplicate origin_id test")
    sw0 = int(net.switch.index[0])
    sw1 = int(net.switch.index[1])
    shared = str(uuid.uuid4())
    net.switch.loc[sw0, "origin_id"] = shared
    net.switch.loc[sw1, "origin_id"] = shared

    rule = SppsRule(
        scheme_name="dup",
        conditions=[
            Condition(
                condition_type=SppsConditionType.CURRENT,
                condition_check_type=SppsConditionCheckType.GT,
                condition_side=SppsConditionSide.PRIMARY,
                condition_limit_value=0.0,
                condition_element_unique_id=shared,
            )
        ],
        actions=[
            Action(
                measure_element_unique_id=shared,
                measure_type=SppsMeasureType.SWITCHING_STATE,
                measure_value=SppsSwitchActionTarget.OPEN,
            )
        ],
    )
    _cond, _act, missing, dupes = translate_spps_rules(net, [rule], id_type="cgmes")
    assert missing == []
    assert dupes == [shared]
