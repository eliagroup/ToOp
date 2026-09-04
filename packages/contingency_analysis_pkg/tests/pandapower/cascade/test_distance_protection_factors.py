# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Distance-protection zones and the factor selection behind them.

The three zones nest danger inside alarm inside warning. Danger is the relay polygon as the
relay defines it and takes no factors; alarm and warning are each configurable on two axes,
case and protected element type.
"""

import math

import numpy as np
import pandas as pd
import pytest
import shapely
from toop_engine_contingency_analysis.pandapower.cascade.configuration import (
    CascadeConfig,
    DistanceProtectionConfig,
    DistanceProtectionFactors,
    DistanceProtectionSeverity,
    OverloadConfig,
)
from toop_engine_contingency_analysis.pandapower.cascade.detection.distance_protection import (
    _build_poly,
    _resolve_one_case,
    describe_relay_zones,
    evaluate_distance_protection_triggers,
    get_alarm_area,
    get_danger_area,
    get_warning_area,
    resolve_effective_factors,
)

#: ``_build_poly`` with ``r_i == r_v == x_v`` degenerates to the square [0, 10] x [0, 10],
#: so a measurement is inside exactly while both scaled coordinates are <= 10.
REACH = 10.0

#: What describe_relay_zones adds on top of the relay table it is given.
DERIVED_COLUMNS = {
    "poly",
    "effective_alarm_basecase",
    "effective_alarm_contingency",
    "effective_warning_basecase",
    "effective_warning_contingency",
}

SLOTS = (
    "basecase_line",
    "basecase_transformer",
    "basecase_bus_coupler",
    "contingency_line",
    "contingency_transformer",
    "contingency_bus_coupler",
)

OVERRIDE_COLUMNS = (
    "custom_base_alarm",
    "custom_base_warning",
    "custom_contingency_alarm",
    "custom_contingency_warning",
)


def _square_poly() -> shapely.Polygon:
    row = pd.Series({"r_i": REACH, "r_v": REACH, "x_v": REACH, "angle": math.radians(30.0)})
    return _build_poly(row)


def _relays(
    protection_element: list,
    r_ohm: list[float],
    contingency: list[str] | None = None,
    **overrides: list,
) -> pd.DataFrame:
    """Build a switch-result frame of relays that all share one square protection zone."""
    count = len(protection_element)
    data = {
        "contingency": contingency if contingency is not None else ["BASECASE"] * count,
        "protection_element": protection_element,
        "r_ohm": r_ohm,
        "x_ohm": [0.0] * count,
        "poly": [_square_poly()] * count,
    }
    for column in OVERRIDE_COLUMNS:
        data[column] = overrides.get(column, [np.nan] * count)
    return pd.DataFrame(data)


def _prepared(df: pd.DataFrame, cfg: CascadeConfig) -> pd.DataFrame:
    """Add the factor columns a job precomputes, so the area helpers can read them."""
    prepared = df.copy()
    for column, values in resolve_effective_factors(df, cfg).items():
        prepared[column] = values
    return prepared


def _factors(**overrides: float) -> DistanceProtectionFactors:
    """Neutral on every axis except the ones a test names.

    1.0 leaves the relay polygon exactly as the relay defines it, so a test that cares about
    one axis stays readable while still supplying all four required values.
    """
    values = dict.fromkeys(SLOTS, 1.0)
    values.update(overrides)
    return DistanceProtectionFactors(**values)


def _config(alarm: dict | None = None, warning: dict | None = None) -> CascadeConfig:
    return CascadeConfig(
        depth_limit=1,
        min_island_size=1,
        cascade_log_elements=[],
        overload=OverloadConfig(current_loading_threshold=1.0),
        distance_protection=DistanceProtectionConfig(
            alarm=_factors(**(alarm or {})),
            warning=_factors(**(warning or {})),
        ),
    )


def _severities(df: pd.DataFrame, cfg: CascadeConfig) -> list[str]:
    """Label each row the way the cascade does: by the innermost zone it reached."""
    danger = get_danger_area(df)
    alarm = get_alarm_area(_prepared(df, cfg))
    warning = get_warning_area(_prepared(df, cfg))
    return [
        DistanceProtectionSeverity.innermost(
            danger_inside=bool(is_danger), alarm_inside=bool(is_alarm), warning_inside=bool(is_warning)
        )
        for is_danger, is_alarm, is_warning in zip(danger, alarm, warning, strict=True)
    ]


class TestDangerArea:
    """The innermost zone: the relay polygon, with nothing applied to it."""

    def test_is_the_raw_relay_polygon(self):
        """No factor touches the danger area, so it matches a bare polygon test value for value."""
        r_ohm = [0.0, 5.0, REACH, REACH + 1e-9, 20.0, -5.0]
        df = _relays(protection_element=["line"] * len(r_ohm), r_ohm=r_ohm)

        expected = shapely.covers(
            df["poly"].to_numpy(),
            shapely.points(np.abs(df["r_ohm"].to_numpy()), np.abs(df["x_ohm"].to_numpy())),
        )

        assert get_danger_area(df).tolist() == expected.tolist()

    def test_ignores_the_element_type(self):
        """With no factors there is no element-type axis to select on."""
        df = _relays(protection_element=["trafo", "line", None], r_ohm=[5.0, 5.0, 5.0])

        assert get_danger_area(df).tolist() == [True, True, True]


class TestAlarmArea:
    """The middle zone, configurable on case and element type."""

    def test_a_wider_factor_widens_only_the_matching_relays(self):
        """A transformer alarm factor must not move line relays."""
        cfg = _config(alarm={"basecase_transformer": 2.0}, warning={"basecase_transformer": 2.0})
        # 20 / 2.0 = 10 -> on the boundary, inside. At 1.0 it would be 20, outside.
        df = _relays(protection_element=["trafo", "line"], r_ohm=[20.0, 20.0])

        assert get_alarm_area(_prepared(df, cfg)).tolist() == [True, False]

    def test_reads_the_alarm_override_not_the_warning_one(self):
        """The alarm area reads custom_*_alarm; the warning override must not reach it."""
        cfg = _config()
        df = _relays(
            protection_element=["line", "line"],
            r_ohm=[20.0, 20.0],
            custom_base_alarm=[2.0, np.nan],
            custom_base_warning=[np.nan, 2.0],
        )

        assert get_alarm_area(_prepared(df, cfg)).tolist() == [True, False]


class TestWarningFactorSelection:
    """The warning factor is chosen by case and protected element type."""

    def test_transformer_relay_uses_the_transformer_factor(self):
        """A trafo relay with no override takes the transformer factor, not the line one."""
        cfg = _config(warning={"basecase_transformer": 2.0})
        df = _relays(protection_element=["trafo", "line"], r_ohm=[20.0, 20.0])

        assert get_warning_area(_prepared(df, cfg)).tolist() == [True, False]

    def test_an_override_wins_over_the_transformer_factor(self):
        """The per-relay override replaces the global factor for its severity and case."""
        cfg = _config(warning={"basecase_transformer": 2.0})
        df = _relays(protection_element=["trafo"], r_ohm=[20.0], custom_base_warning=[1.0])

        assert get_warning_area(_prepared(df, cfg)).tolist() == [False]

    def test_the_override_is_picked_by_case(self):
        """A base-case row reads custom_base_warning, a contingency row the contingency one."""
        cfg = _config()
        df = _relays(
            protection_element=["line", "line"],
            r_ohm=[20.0, 20.0],
            contingency=["BASECASE", "outage-1"],
            custom_base_warning=[2.0, 1.0],
            custom_contingency_warning=[1.0, 2.0],
        )

        assert get_warning_area(_prepared(df, cfg)).tolist() == [True, True]

    def test_the_global_factor_is_picked_by_case(self):
        """Without overrides, contingency rows use the contingency factor."""
        cfg = _config(warning={"contingency_line": 2.0})
        df = _relays(
            protection_element=["line", "line"],
            r_ohm=[20.0, 20.0],
            contingency=["BASECASE", "outage-1"],
        )

        assert get_warning_area(_prepared(df, cfg)).tolist() == [False, True]


class TestBusCouplerFactor:
    """bus_coupler is a third protection_element value with a factor of its own."""

    def test_a_bus_coupler_relay_uses_the_bus_coupler_factor(self):
        """Only the coupler moves; the line and transformer relays keep the neutral factor."""
        cfg = _config(warning={"basecase_bus_coupler": 2.0})
        df = _relays(protection_element=["bus_coupler", "line", "trafo"], r_ohm=[20.0, 20.0, 20.0])

        assert get_warning_area(_prepared(df, cfg)).tolist() == [True, False, False]

    def test_a_line_factor_does_not_move_bus_couplers(self):
        cfg = _config(warning={"basecase_line": 2.0})
        df = _relays(protection_element=["bus_coupler"], r_ohm=[20.0])

        assert get_warning_area(_prepared(df, cfg)).tolist() == [False]

    def test_it_is_picked_by_case_like_the_others(self):
        cfg = _config(warning={"contingency_bus_coupler": 2.0})
        df = _relays(
            protection_element=["bus_coupler", "bus_coupler"],
            r_ohm=[20.0, 20.0],
            contingency=["BASECASE", "outage-1"],
        )

        assert get_warning_area(_prepared(df, cfg)).tolist() == [False, True]

    def test_a_per_relay_override_still_wins(self):
        """The custom_* columns are per severity and case, so they cover every element type."""
        cfg = _config(warning={"basecase_bus_coupler": 2.0})
        df = _relays(protection_element=["bus_coupler"], r_ohm=[20.0], custom_base_warning=[1.0])

        assert get_warning_area(_prepared(df, cfg)).tolist() == [False]


class TestUnknownProtectionElement:
    """protection_element is None when the side carries both elements, or neither."""

    def test_takes_the_wider_of_the_two_factors(self):
        """The larger factor widens the area, so the ambiguous relay is screened as widely."""
        cfg = _config(warning={"basecase_transformer": 2.0})
        df = _relays(protection_element=[None], r_ohm=[20.0])

        assert get_warning_area(_prepared(df, cfg)).tolist() == [True]

    def test_the_wider_factor_wins_whichever_type_it_belongs_to(self):
        """The rule is the maximum over all three, not "always the transformer"."""
        cfg = _config(warning={"basecase_line": 2.0})
        df = _relays(protection_element=[None], r_ohm=[20.0])

        assert get_warning_area(_prepared(df, cfg)).tolist() == [True]

    def test_the_bus_coupler_factor_takes_part_in_the_maximum(self):
        """A wide bus-coupler factor reaches unclassified relays like the other two do."""
        cfg = _config(warning={"basecase_bus_coupler": 2.0})
        df = _relays(protection_element=[None], r_ohm=[20.0])

        assert get_warning_area(_prepared(df, cfg)).tolist() == [True]


class TestInnermostSeverity:
    """A trip is labelled by the innermost zone it reached."""

    def test_danger_wins_over_everything(self):
        severity = DistanceProtectionSeverity.innermost(danger_inside=True, alarm_inside=True, warning_inside=True)

        assert severity == DistanceProtectionSeverity.DANGER.value

    def test_alarm_when_the_danger_polygon_was_not_reached(self):
        severity = DistanceProtectionSeverity.innermost(danger_inside=False, alarm_inside=True, warning_inside=True)

        assert severity == DistanceProtectionSeverity.ALARM.value

    def test_warning_when_only_the_outer_zone_was_reached(self):
        severity = DistanceProtectionSeverity.innermost(danger_inside=False, alarm_inside=False, warning_inside=True)

        assert severity == DistanceProtectionSeverity.WARNING.value

    def test_a_narrow_outer_zone_still_reports_the_inner_one(self):
        """Factors below 1.0 can leave a danger hit outside both configured zones."""
        severity = DistanceProtectionSeverity.innermost(danger_inside=True, alarm_inside=False, warning_inside=False)

        assert severity == DistanceProtectionSeverity.DANGER.value

    def test_no_zone_at_all_is_rejected(self):
        """Such a row was never a trip, so asking for its severity is a caller bug."""
        with pytest.raises(ValueError, match="no distance-protection zone"):
            DistanceProtectionSeverity.innermost(danger_inside=False, alarm_inside=False, warning_inside=False)

    def test_the_result_is_a_plain_string(self):
        """It lands in a pandera Series[str]; an Enum member would not hash like its value."""
        severity = DistanceProtectionSeverity.innermost(danger_inside=True, alarm_inside=False, warning_inside=False)

        assert severity in {"DANGER", "ALARM", "WARNING"}


class TestFactorGrouping:
    """Each of the eight configurable values must reach its own slot."""

    def test_each_slot_is_read_from_its_own_field(self):
        """A distinct value per slot catches any two of them being crossed."""
        cfg = _config(
            warning={
                "basecase_line": 1.0,
                "contingency_line": 2.0,
                "basecase_transformer": 3.0,
                "contingency_transformer": 4.0,
                "basecase_bus_coupler": 5.0,
                "contingency_bus_coupler": 6.0,
            }
        )

        assert cfg.distance_protection.warning == DistanceProtectionFactors(
            basecase_line=1.0,
            basecase_transformer=3.0,
            basecase_bus_coupler=5.0,
            contingency_line=2.0,
            contingency_transformer=4.0,
            contingency_bus_coupler=6.0,
        )

    def test_the_severities_are_independent(self):
        """Setting the warning factors must not move the alarm ones."""
        cfg = _config(warning={"basecase_line": 1.41, "contingency_transformer": 1.6})

        assert cfg.distance_protection.alarm == _factors()

    @pytest.mark.parametrize("missing", SLOTS)
    def test_every_factor_is_required(self, missing: str):
        """No factor has a default: an un-migrated caller fails loudly instead of silently."""
        values = dict.fromkeys(SLOTS, 1.0)
        del values[missing]

        with pytest.raises(ValueError):
            DistanceProtectionFactors(**values)

    @pytest.mark.parametrize("missing", ["alarm", "warning"])
    def test_both_configurable_zones_are_required(self, missing: str):
        groups = {"alarm": _factors(), "warning": _factors()}
        del groups[missing]

        with pytest.raises(ValueError):
            DistanceProtectionConfig(**groups)


class TestCoincidingZones:
    """Two zones with the same factor are the same area, so the innermost name always wins.

    This is not special-cased anywhere: if the alarm and warning factors are equal, every row
    inside the warning area is inside the identical alarm area, so the alarm flag is already
    set and ``innermost`` reports the inner name. The tests pin that down because the labels
    are what operators read.
    """

    def test_alarm_equal_to_warning_never_reports_warning(self):
        """The two areas coincide, so a trip in them is an ALARM, not a WARNING."""
        cfg = _config(alarm={"basecase_line": 2.0}, warning={"basecase_line": 2.0})
        # 15 / 2.0 = 7.5, inside both; outside the danger polygon, which reaches only to 10.
        df = _relays(protection_element=["line"], r_ohm=[15.0])

        assert get_warning_area(_prepared(df, cfg)).tolist() == [True]
        assert _severities(df, cfg) == [DistanceProtectionSeverity.ALARM.value]

    def test_alarm_equal_to_danger_never_reports_alarm(self):
        """An alarm factor of 1.0 makes the alarm area the raw polygon, so trips are DANGER."""
        cfg = _config(alarm={"basecase_line": 1.0}, warning={"basecase_line": 2.0})
        df = _relays(protection_element=["line"], r_ohm=[5.0])

        assert _severities(df, cfg) == [DistanceProtectionSeverity.DANGER.value]

    def test_all_three_equal_reports_danger(self):
        """With every factor at 1.0 the three zones are one area, reported as the innermost."""
        cfg = _config(alarm={"basecase_line": 1.0}, warning={"basecase_line": 1.0})
        df = _relays(protection_element=["line"], r_ohm=[5.0])

        assert _severities(df, cfg) == [DistanceProtectionSeverity.DANGER.value]

    def test_distinct_zones_still_reach_every_label(self):
        """The collapse above is the zones coinciding, not the labels being unreachable."""
        cfg = _config(alarm={"basecase_line": 1.5}, warning={"basecase_line": 2.0})
        # 5 is inside the polygon; 15 / 1.5 = 10 is on the alarm edge; 18 / 1.5 = 12 is outside
        # the alarm area but 18 / 2.0 = 9 is inside the warning one.
        df = _relays(protection_element=["line"] * 3, r_ohm=[5.0, 15.0, 18.0])

        assert _severities(df, cfg) == [
            DistanceProtectionSeverity.DANGER.value,
            DistanceProtectionSeverity.ALARM.value,
            DistanceProtectionSeverity.WARNING.value,
        ]


def _measurements(r_ohm: list[float]) -> pd.DataFrame:
    """Build the raw switch columns that produce the given resistances.

    ``get_complex_impedance`` derives r and x from vm, i, p and q. With ``q = 0`` the angle is
    zero, so all of the impedance lands on the resistance axis; ``i = 1000`` then makes
    ``vm = r * sqrt(3)`` give exactly ``r`` ohms.
    """
    count = len(r_ohm)
    return pd.DataFrame(
        {
            "vm": [value * math.sqrt(3) for value in r_ohm],
            "i": [1000.0] * count,
            "p": [1.0] * count,
            "q": [0.0] * count,
            "contingency": ["BASECASE"] * count,
            "protection_element": ["line"] * count,
            "poly": [_square_poly()] * count,
            **{column: [np.nan] * count for column in OVERRIDE_COLUMNS},
        }
    )


class TestEvaluateDistanceProtectionTriggers:
    """The entry point: derive the impedance, flag all three zones, keep what tripped."""

    def test_it_derives_r_and_x_from_the_measurements(self):
        cfg = _config()
        # Bind the prepared frame: the function writes r_ohm and x_ohm onto what it is given.
        df = _prepared(_measurements([5.0]), cfg)

        evaluate_distance_protection_triggers(df)

        assert df["r_ohm"].tolist() == pytest.approx([5.0])
        assert df["x_ohm"].tolist() == pytest.approx([0.0])

    def test_it_flags_all_three_zones(self):
        """A row deep inside the polygon is inside every zone."""
        cfg = _config(warning={"basecase_line": 2.0})
        df = _measurements([5.0])

        tripped = evaluate_distance_protection_triggers(_prepared(df, cfg))

        assert tripped["danger_inside"].tolist() == [True]
        assert tripped["alarm_inside"].tolist() == [True]
        assert tripped["warning_inside"].tolist() == [True]

    def test_it_keeps_only_rows_inside_a_zone(self):
        """5 is in the polygon, 15 only in the widened warning area, 30 in nothing."""
        cfg = _config(warning={"basecase_line": 2.0})
        df = _measurements([5.0, 15.0, 30.0])

        tripped = evaluate_distance_protection_triggers(_prepared(df, cfg))

        assert tripped.index.tolist() == [0, 1]
        assert tripped["danger_inside"].tolist() == [True, False]
        assert tripped["warning_inside"].tolist() == [True, True]

    def test_a_danger_hit_survives_a_narrower_warning_area(self):
        """Selection is the union, so factors below 1.0 cannot hide a relay in its polygon."""
        cfg = _config(alarm={"basecase_line": 0.5}, warning={"basecase_line": 0.5})
        # 8 is inside the polygon, but 8 / 0.5 = 16 is outside both configured zones.
        df = _measurements([8.0])

        tripped = evaluate_distance_protection_triggers(_prepared(df, cfg))

        assert tripped["danger_inside"].tolist() == [True]
        assert tripped["alarm_inside"].tolist() == [False]
        assert tripped["warning_inside"].tolist() == [False]
        assert len(tripped) == 1

    def test_nothing_inside_gives_an_empty_result(self):
        cfg = _config()
        df = _measurements([30.0])

        assert evaluate_distance_protection_triggers(_prepared(df, cfg)).empty


def test_the_danger_zone_has_no_factors():
    """The resolver covers the configurable zones only; danger is the raw polygon."""
    cfg = _config()
    df = _relays(protection_element=["line"], r_ohm=[5.0])

    with pytest.raises(ValueError, match="raw relay polygon"):
        _resolve_one_case(df, cfg, DistanceProtectionSeverity.DANGER, basecase=True)


def _sw_characteristics(protection_element: list, **overrides: list) -> pd.DataFrame:
    """A relay table shaped like net.sw_characteristics, with angle still in degrees."""
    count = len(protection_element)
    data = {
        "breaker_uuid": [f"relay-{index}" for index in range(count)],
        "relay_side": ["bus"] * count,
        "protection_side": ["element"] * count,
        "protection_element": protection_element,
        "angle": [30.0] * count,
        "r_i": [REACH] * count,
        "r_v": [REACH] * count,
        "x_v": [REACH] * count,
    }
    for column in OVERRIDE_COLUMNS:
        data[column] = overrides.get(column, [np.nan] * count)
    return pd.DataFrame(data)


class TestDescribeRelayZones:
    """Resolve polygons and effective factors straight from a relay table."""

    def test_it_reports_one_row_per_relay_with_its_polygon(self):
        cfg = _config()
        relays = _sw_characteristics(protection_element=["line", "trafo"])

        described = describe_relay_zones(relays, cfg)

        assert described["breaker_uuid"].tolist() == ["relay-0", "relay-1"]
        assert described["poly"].map(lambda poly: poly.equals(_square_poly())).all()

    def test_it_resolves_both_cases_per_zone(self):
        """Four factor columns: two zones times two cases."""
        cfg = _config(
            alarm={"basecase_line": 1.1, "contingency_line": 1.2},
            warning={"basecase_line": 1.3, "contingency_line": 1.4},
        )
        described = describe_relay_zones(_sw_characteristics(protection_element=["line"]), cfg)

        assert described["effective_alarm_basecase"].tolist() == [1.1]
        assert described["effective_alarm_contingency"].tolist() == [1.2]
        assert described["effective_warning_basecase"].tolist() == [1.3]
        assert described["effective_warning_contingency"].tolist() == [1.4]

    def test_it_follows_the_element_type(self):
        cfg = _config(warning={"basecase_transformer": 2.0, "basecase_bus_coupler": 3.0})
        relays = _sw_characteristics(protection_element=["line", "trafo", "bus_coupler"])

        described = describe_relay_zones(relays, cfg)

        assert described["effective_warning_basecase"].tolist() == [1.0, 2.0, 3.0]

    def test_a_per_relay_override_shows_up_in_the_result(self):
        """This is the point of the helper: see the value a relay really gets."""
        cfg = _config(warning={"basecase_line": 2.0})
        relays = _sw_characteristics(protection_element=["line", "line"], custom_base_warning=[5.0, np.nan])

        described = describe_relay_zones(relays, cfg)

        assert described["effective_warning_basecase"].tolist() == [5.0, 2.0]

    def test_it_does_not_modify_the_input(self):
        """angle stays in degrees on the caller's frame, and no poly column is added to it."""
        cfg = _config()
        relays = _sw_characteristics(protection_element=["line"])

        describe_relay_zones(relays, cfg)

        assert relays["angle"].tolist() == [30.0]
        assert "poly" not in relays.columns

    def test_an_already_prepared_table_is_not_converted_twice(self):
        """A poly column marks a table prepare_cascade_run_constants already handled."""
        cfg = _config()
        prepared = _sw_characteristics(protection_element=["line"])
        prepared["angle"] = np.radians(prepared["angle"])
        prepared["poly"] = [_square_poly()]

        described = describe_relay_zones(prepared, cfg)

        assert described["poly"].iloc[0].equals(_square_poly())

    def test_it_keeps_every_original_column(self):
        """prepare_cascade_run_constants puts this frame on the net in place of the original.

        The per-outage merge then selects relay_side and protection_side from it, so dropping
        a column here would only surface as a KeyError once an outage runs.
        """
        relays = _sw_characteristics(protection_element=["line"])

        described = describe_relay_zones(relays, _config())

        assert set(relays.columns) <= set(described.columns)
        assert DERIVED_COLUMNS <= set(described.columns)

    def test_an_empty_relay_table_is_handled(self):
        relays = _sw_characteristics(protection_element=[])

        described = describe_relay_zones(relays, _config())

        assert described.empty
        assert set(relays.columns) <= set(described.columns)
        assert DERIVED_COLUMNS <= set(described.columns)


def test_an_empty_relay_frame_is_handled():
    """The base-case screen calls these with whatever the outage produced, empty included."""
    cfg = _config()
    df = _relays(protection_element=[], r_ohm=[])

    assert get_danger_area(df).tolist() == []
    assert get_alarm_area(_prepared(df, cfg)).tolist() == []
    assert get_warning_area(_prepared(df, cfg)).tolist() == []
