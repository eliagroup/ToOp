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


def _effective_factors(
    df: pd.DataFrame,
    cascade_configuration: CascadeConfig,
    severity: DistanceProtectionSeverity,
) -> np.ndarray:
    """Resolve the impedance factor that applies to each relay measurement.

    A measurement is divided by its factor before being tested against the relay polygon
    (``x = |r_ohm| / factor``), so the factor is what widens or narrows the effective zone.
    Which factor a given row gets is decided on three independent axes:

    - **severity** - from the ``severity`` argument. Fixed for the whole call rather than
      per row, because one call tests one zone.
    - **case** - whether the row is the base case or a contingency, from the ``contingency``
      column.
    - **protected element type** - line or transformer, from the ``protection_element``
      column.

    Two sources can supply the value for those three axes, the first that has one winning:

    1. the per-relay override column for this row's severity and case, where it is not NaN;
    2. otherwise the global factor configured for that severity, case and element type.

    The steps below take one axis at a time. Each is a whole-column numpy operation: this
    runs twice per outage over every monitored relay, so nothing walks the frame row by row.

    Parameters
    ----------
    df : pd.DataFrame
        Switch result table, carrying ``contingency``, ``protection_element`` and the four
        ``custom_*`` override columns.
    cascade_configuration : CascadeConfig
        Cascade settings holding the eight global factors.
    severity : DistanceProtectionSeverity
        Which zone is being tested, and so which factor group and override columns apply.
        Only ``ALARM`` and ``WARNING`` are configurable; ``DANGER`` raises.

    Returns
    -------
    np.ndarray
        One factor per row, as a per-unit ratio.
    """
    # Step 1: severity. One call tests one zone, so this is picked once, not per row. It
    # decides both the factor group and which two override columns the steps below read.
    distance_protection = cascade_configuration.distance_protection
    if severity is DistanceProtectionSeverity.ALARM:
        factors = distance_protection.alarm
        basecase_column, contingency_column = "custom_base_alarm", "custom_contingency_alarm"
    elif severity is DistanceProtectionSeverity.WARNING:
        factors = distance_protection.warning
        basecase_column, contingency_column = "custom_base_warning", "custom_contingency_warning"
    else:
        # DANGER has no factors: it is the relay polygon itself.
        raise ValueError(f"{severity} has no factors; the danger area is the raw relay polygon")

    # Step 2: case. Leaves one line factor and one transformer factor per row.
    is_basecase = (df["contingency"] == "BASECASE").to_numpy()
    line = np.where(is_basecase, factors.basecase_line, factors.contingency_line)
    transformer = np.where(is_basecase, factors.basecase_transformer, factors.contingency_transformer)

    # Step 3: element type. Which of the two factors each row is entitled to.
    #
    # Use .eq, not ==: on an empty frame the column can arrive as float64, and numpy would
    # then return a scalar False instead of an empty mask.
    is_line = df["protection_element"].eq("line").to_numpy()
    is_transformer = df["protection_element"].eq("trafo").to_numpy()

    # Step 4: start every row on the unknown-type factor.
    #
    # protection_element is None when the protected side has both a line and a transformer,
    # or neither. The larger factor gives the wider area, so a relay of unknown type is
    # never screened too narrowly.
    global_factors = np.maximum(line, transformer)

    # Step 5: rows whose type is known take that type's factor instead.
    global_factors = np.where(is_line, line, global_factors)
    global_factors = np.where(is_transformer, transformer, global_factors)

    # Step 6: the relay's own override wins where it set one. Only the column for the row's
    # case is read, so a relay can override the base case and not the contingency, or vice versa.
    overrides = np.where(is_basecase, df[basecase_column].to_numpy(), df[contingency_column].to_numpy())
    return np.where(pd.isna(overrides), global_factors, overrides).astype(float)


def _inside_area(df: pd.DataFrame, effective_factors: np.ndarray) -> pd.Series:
    """Test each relay measurement against its polygon, scaled by ``effective_factors``.

    Parameters
    ----------
    df : pd.DataFrame
        Switch result table with ``r_ohm``, ``x_ohm`` and ``poly``.
    effective_factors : np.ndarray
        One factor per row, from :func:`_effective_factors`.

    Returns
    -------
    pd.Series
        Boolean series where True means the row is inside the scaled area.
    """
    x = (np.abs(df["r_ohm"].to_numpy()) / effective_factors).astype(float)
    y = (np.abs(df["x_ohm"].to_numpy()) / effective_factors).astype(float)
    flags = shapely.covers(df["poly"].to_numpy(), shapely.points(x, y))
    return pd.Series(flags, index=df.index)


def get_warning_area(df: pd.DataFrame, cascade_configuration: CascadeConfig) -> pd.Series:
    """Check whether each relay measurement is inside the warning area.

    The warning area is the outermost of the three zones, so this is what decides whether a
    relay trips at all; the other two only say how deep the trip went.

    Parameters
    ----------
    df : pd.DataFrame
        Switch result table with impedance values and protection polygons.
    cascade_configuration : CascadeConfig
        Cascade settings holding the warning factors.

    Returns
    -------
    pd.Series
        Boolean series where True means the row is inside the warning area.
    """
    return _inside_area(df, _effective_factors(df, cascade_configuration, DistanceProtectionSeverity.WARNING))


def get_alarm_area(df: pd.DataFrame, cascade_configuration: CascadeConfig) -> pd.Series:
    """Check whether each relay measurement is inside the alarm area.

    The alarm area sits between the danger polygon and the warning area. It does not add
    trips of its own - every row inside it is already inside the wider warning area - it only
    separates an ``ALARM``-labelled trip from a ``WARNING``-labelled one.

    Parameters
    ----------
    df : pd.DataFrame
        Switch result table with impedance values and protection polygons.
    cascade_configuration : CascadeConfig
        Cascade settings holding the alarm factors.

    Returns
    -------
    pd.Series
        Boolean series where True means the row is inside the alarm area.
    """
    return _inside_area(df, _effective_factors(df, cascade_configuration, DistanceProtectionSeverity.ALARM))


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


def evaluate_distance_protection_triggers(
    switch_results: pd.DataFrame,
    cascade_configuration: CascadeConfig,
) -> pd.DataFrame:
    """Find switches that should trip because of distance protection.

    A row trips when it is inside any of the three zones. With the zones nested as expected
    that is just the warning area, but the union is what is actually taken, so a configuration
    with a warning area narrower than the relay polygon still trips on the danger zone. All
    three flags travel with the rows so the caller can label each trip with
    :meth:`DistanceProtectionSeverity.innermost`.

    Parameters
    ----------
    switch_results : pd.DataFrame
        Switch result table already joined with relay characteristics.
    cascade_configuration : CascadeConfig
        Cascade settings with the alarm and warning factors.

    Returns
    -------
    pd.DataFrame
        Subset of switch_results inside at least one zone, carrying the ``danger_inside``,
        ``alarm_inside`` and ``warning_inside`` flags.
    """
    switch_results["r_ohm"], switch_results["x_ohm"] = get_complex_impedance(switch_results)
    switch_results["danger_inside"] = get_danger_area(switch_results)
    switch_results["alarm_inside"] = get_alarm_area(switch_results, cascade_configuration)
    switch_results["warning_inside"] = get_warning_area(switch_results, cascade_configuration)
    return switch_results[
        switch_results["warning_inside"] | switch_results["alarm_inside"] | switch_results["danger_inside"]
    ]
