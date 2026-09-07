# Copyright 2026 50Hertz Transmission GmbH and Elia Transmission Belgium SA/NV
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file,
# you can obtain one at https://mozilla.org/MPL/2.0/.
# Mozilla Public License, version 2.0

"""Detect cascade triggers caused by distance-protection relays."""

import math

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon
from toop_engine_contingency_analysis.pandapower.cascade.configuration import (
    CascadeConfig,
    DistanceProtectionSeverity,
)
from toop_engine_contingency_analysis.pandapower.cascade.detection.switch_preparation import (
    get_complex_impedance,
)


def _build_poly(row: pd.Series) -> Polygon:
    """Build the protection zone shape for one relay.

    Parameters
    ----------
    row : pd.Series
        Row with relay zone dimensions and angle.

    Returns
    -------
    Polygon
        Polygon that represents the relay protection area.
    """
    third_point_y = row.r_i * math.tan(row.angle)
    fourth_point_y = row.r_v * math.tan(row.angle)

    return Polygon(
        [
            (0.0, 0.0),
            (row["r_i"], 0.0),
            (row["r_i"], third_point_y),
            (row["r_v"], fourth_point_y),
            (row["r_v"], row["x_v"]),
            (0.0, row["x_v"]),
        ]
    )


def _resolve_one_case(
    relays: pd.DataFrame,
    cascade_configuration: CascadeConfig,
    severity: DistanceProtectionSeverity,
    *,
    basecase: bool,
) -> np.ndarray:
    """Resolve the factor each relay gets for one zone and one case.

    A measurement is divided by its factor before being tested against the relay polygon
    (``x = |r_ohm| / factor``), so the factor is what widens or narrows the effective zone.
    Two sources can supply it, the first that has one winning:

    1. the per-relay override column for this zone and case, where it is not NaN;
    2. otherwise the global factor configured for this zone, case and element type.

    Zone and case are both fixed by the arguments, so only the element type varies per row and
    the global factors are plain scalars here.

    Parameters
    ----------
    relays : pd.DataFrame
        Relay table with ``protection_element`` and the four ``custom_*`` override columns.
    cascade_configuration : CascadeConfig
        Cascade settings holding the alarm and warning factors.
    severity : DistanceProtectionSeverity
        Which zone to resolve. Only ``ALARM`` and ``WARNING`` are configurable.
    basecase : bool
        Resolve the base-case factors rather than the contingency ones.

    Returns
    -------
    np.ndarray
        One factor per relay, as a per-unit ratio.
    """
    # Step 1: zone. Fixes the factor group and which override column applies.
    distance_protection = cascade_configuration.distance_protection
    if severity is DistanceProtectionSeverity.ALARM:
        factors = distance_protection.alarm
        override_column = "custom_base_alarm" if basecase else "custom_contingency_alarm"
    elif severity is DistanceProtectionSeverity.WARNING:
        factors = distance_protection.warning
        override_column = "custom_base_warning" if basecase else "custom_contingency_warning"
    else:
        # DANGER has no factors: it is the relay polygon itself.
        raise ValueError(f"{severity} has no factors; the danger area is the raw relay polygon")

    # Step 2: case. Fixed for the whole call, so one scalar per element type.
    if basecase:
        line, transformer, bus_coupler = (
            factors.basecase_line,
            factors.basecase_transformer,
            factors.basecase_bus_coupler,
        )
    else:
        line, transformer, bus_coupler = (
            factors.contingency_line,
            factors.contingency_transformer,
            factors.contingency_bus_coupler,
        )

    # Step 3: element type. Which of those scalars each row is entitled to.
    #
    # Use .eq, not ==: on an empty frame the column can arrive as float64, and numpy would
    # then return a scalar False instead of an empty mask.
    is_line = relays["protection_element"].eq("line").to_numpy()
    is_transformer = relays["protection_element"].eq("trafo").to_numpy()
    is_bus_coupler = relays["protection_element"].eq("bus_coupler").to_numpy()

    # Step 4: start every row on the unknown-type factor.
    #
    # protection_element is None when the protected side carries no single element type. The
    # largest factor gives the widest area, so a relay of unknown type is never screened too
    # narrowly.
    global_factors = np.full(len(relays.index), max(line, transformer, bus_coupler), dtype=float)

    # Step 5: rows whose type is known take that type's factor instead.
    global_factors = np.where(is_line, line, global_factors)
    global_factors = np.where(is_transformer, transformer, global_factors)
    global_factors = np.where(is_bus_coupler, bus_coupler, global_factors)

    # Step 6: the relay's own override wins where it set one.
    overrides = relays[override_column].to_numpy()
    return np.where(pd.isna(overrides), global_factors, overrides).astype(float)


def resolve_effective_factors(
    relays: pd.DataFrame,
    cascade_configuration: CascadeConfig,
) -> dict[str, np.ndarray]:
    """Resolve every relay's factor for both configurable zones and both cases.

    The result depends only on the relay and the configuration, never on which contingency is
    being computed, so this runs once per job rather than once per outage.

    Parameters
    ----------
    relays : pd.DataFrame
        Relay table with ``protection_element`` and the four ``custom_*`` override columns.
    cascade_configuration : CascadeConfig
        Cascade settings holding the alarm and warning factors.

    Returns
    -------
    dict[str, np.ndarray]
        One entry per zone and case, keyed by the column name it is written to.
    """
    alarm = DistanceProtectionSeverity.ALARM
    warning = DistanceProtectionSeverity.WARNING
    return {
        "effective_alarm_basecase": _resolve_one_case(relays, cascade_configuration, alarm, basecase=True),
        "effective_alarm_contingency": _resolve_one_case(relays, cascade_configuration, alarm, basecase=False),
        "effective_warning_basecase": _resolve_one_case(relays, cascade_configuration, warning, basecase=True),
        "effective_warning_contingency": _resolve_one_case(relays, cascade_configuration, warning, basecase=False),
    }


def _inside_area(df: pd.DataFrame, effective_factors: np.ndarray) -> pd.Series:
    """Test each relay measurement against its polygon, scaled by ``effective_factors``.

    Parameters
    ----------
    df : pd.DataFrame
        Switch result table with ``r_ohm``, ``x_ohm`` and ``poly``.
    effective_factors : np.ndarray
        One factor per row.

    Returns
    -------
    pd.Series
        Boolean series where True means the row is inside the scaled area.
    """
    x = (np.abs(df["r_ohm"].to_numpy()) / effective_factors).astype(float)
    y = (np.abs(df["x_ohm"].to_numpy()) / effective_factors).astype(float)
    flags = shapely.covers(df["poly"].to_numpy(), shapely.points(x, y))
    return pd.Series(flags, index=df.index)


def get_warning_area(df: pd.DataFrame) -> pd.Series:
    """Check whether each relay measurement is inside the warning area.

    The warning area is the outermost of the three zones, so this is what decides whether a
    relay trips at all; the other two only say how deep the trip went.

    Parameters
    ----------
    df : pd.DataFrame
        Switch result table with impedance values, protection polygons, ``contingency`` and
        the warning factor columns written by :func:`resolve_effective_factors`.

    Returns
    -------
    pd.Series
        Boolean series where True means the row is inside the warning area.
    """
    is_basecase = (df["contingency"] == "BASECASE").to_numpy()
    effective_factors = np.where(
        is_basecase,
        df["effective_warning_basecase"].to_numpy(),
        df["effective_warning_contingency"].to_numpy(),
    ).astype(float)
    return _inside_area(df, effective_factors)


def get_alarm_area(df: pd.DataFrame) -> pd.Series:
    """Check whether each relay measurement is inside the alarm area.

    The alarm area sits between the danger polygon and the warning area. It does not add trips
    of its own - every row inside it is already inside the wider warning area - it only
    separates an ``ALARM``-labelled trip from a ``WARNING``-labelled one.

    Parameters
    ----------
    df : pd.DataFrame
        Switch result table with impedance values, protection polygons, ``contingency`` and
        the alarm factor columns written by :func:`resolve_effective_factors`.

    Returns
    -------
    pd.Series
        Boolean series where True means the row is inside the alarm area.
    """
    is_basecase = (df["contingency"] == "BASECASE").to_numpy()
    effective_factors = np.where(
        is_basecase,
        df["effective_alarm_basecase"].to_numpy(),
        df["effective_alarm_contingency"].to_numpy(),
    ).astype(float)
    return _inside_area(df, effective_factors)


def get_danger_area(df: pd.DataFrame) -> pd.Series:
    """Check whether each relay measurement is inside the danger area.

    The danger area is the relay polygon exactly as the relay defines it - the real trip
    boundary. It takes no configuration on purpose: a factor here would make the simulation
    report trips the physical relay would never perform.

    Parameters
    ----------
    df : pd.DataFrame
        Switch result table with impedance values and protection polygons.

    Returns
    -------
    pd.Series
        Boolean series where True means the row is inside the danger area.
    """
    # A factor of 1.0 on every row leaves the measurement untouched.
    return _inside_area(df, np.ones(len(df.index)))


def describe_relay_zones(
    sw_characteristics: pd.DataFrame,
    cascade_configuration: CascadeConfig,
) -> pd.DataFrame:
    """Prepare a relay table: derive its polygons and resolve its effective factors.

    Everything a relay needs before any load flow runs. Also answers "what will this
    configuration actually apply to each relay", which otherwise only becomes visible once a
    cascade has run.

    The danger zone gets no column: it is the polygon itself, so its factor is always ``1.0``.

    Parameters
    ----------
    sw_characteristics : pd.DataFrame
        Relay table as attached to ``net.sw_characteristics``: ``breaker_uuid``,
        ``protection_element``, the polygon dimensions (``angle``, ``r_i``, ``r_v``, ``x_v``)
        and the four ``custom_*`` override columns. Not modified. ``angle`` is read in degrees
        unless a ``poly`` column is already present, which marks a table already prepared.
    cascade_configuration : CascadeConfig
        Cascade settings holding the alarm and warning factors.

    Returns
    -------
    pd.DataFrame
        A copy of the input carrying every original column, with ``angle`` converted to
        radians, the derived ``poly`` polygon, and one factor column per zone and case as
        written by :func:`resolve_effective_factors`.
    """
    relays = sw_characteristics.copy()
    if "poly" not in relays.columns:
        relays["angle"] = np.radians(relays["angle"])
        # apply over an empty frame returns a DataFrame, which cannot be assigned as a column.
        relays["poly"] = (
            relays.apply(_build_poly, axis=1) if not relays.empty else pd.Series(dtype=object, index=relays.index)
        )

    for column, values in resolve_effective_factors(relays, cascade_configuration).items():
        relays[column] = values
    return relays


def evaluate_distance_protection_triggers(switch_results: pd.DataFrame) -> pd.DataFrame:
    """Find switches that should trip because of distance protection.

    A row trips when it is inside any of the three zones. With the zones nested as expected
    that is just the warning area, but the union is what is actually taken, so a configuration
    with a warning area narrower than the relay polygon still trips on the danger zone. All
    three flags travel with the rows so the caller can label each trip with
    :meth:`DistanceProtectionSeverity.innermost`.

    Parameters
    ----------
    switch_results : pd.DataFrame
        Switch result table already joined with relay characteristics, so it carries the
        polygons and the precomputed factor columns.

    Returns
    -------
    pd.DataFrame
        Subset of switch_results inside at least one zone, carrying the ``danger_inside``,
        ``alarm_inside`` and ``warning_inside`` flags.
    """
    switch_results["r_ohm"], switch_results["x_ohm"] = get_complex_impedance(switch_results)
    switch_results["danger_inside"] = get_danger_area(switch_results)
    switch_results["alarm_inside"] = get_alarm_area(switch_results)
    switch_results["warning_inside"] = get_warning_area(switch_results)
    return switch_results[
        switch_results["warning_inside"] | switch_results["alarm_inside"] | switch_results["danger_inside"]
    ]
